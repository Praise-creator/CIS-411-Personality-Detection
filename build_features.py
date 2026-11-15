"""Build every cached dataset the team needs, from dataset.csv.

Stages (run in this order):
  text     transcript -> sentence embeddings            -> features/text_<model>.npz
  frames   videos -> 8 frames per clip (face + full)    -> /content/frames/{face,full}/<file_name>/0..7.jpg
  clip     frames -> frozen CLIP ViT-B/32 embeddings    -> features/clip_b32_<view>.npz
  combine  align by file_name, z-score on TRAIN only    -> features/ready_*.npz

Every stage is resumable: re-running skips finished work.


"""
import argparse
import os

import numpy as np
import pandas as pd

TRAITS = ["extraversion", "agreeableness", "conscientiousness", "neuroticism", "openness"]
N_FRAMES = 8
SIZE = 224


def save_npz(path, ids, feats):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, ids=np.array(list(ids), dtype="U"), feats=feats)
    print("saved", path, feats.shape)


# --------------------------------------------------------------------------- text
def stage_text(a):
    from sentence_transformers import SentenceTransformer

    df = pd.read_csv(a.dataset)
    model = SentenceTransformer(a.text_model)
    feats = model.encode(df.text.fillna("").tolist(), batch_size=64, show_progress_bar=True)
    name = a.text_model.split("/")[-1]
    save_npz(f"{a.out}/text_{name}.npz", df.file_name.values, feats.astype(np.float32))


# ------------------------------------------------------------------------- frames
def sample_frames(path, n=N_FRAMES):
    import cv2

    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    targets = set(np.linspace(0, max(total - 1, 0), n).astype(int).tolist())
    frames = []
    # read front to back once: grab() skips frames cheaply, retrieve() decodes only the ones we keep
    for idx in range(max(targets) + 1):
        if not cap.grab():
            break
        if idx in targets:
            ok, f = cap.retrieve()
            if ok:
                frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    while frames and len(frames) < n:  # pad if some frames failed to decode
        frames.append(frames[-1])
    return frames


def center_crop(img):
    h, w = img.shape[:2]
    s = min(h, w)
    return img[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]


def face_crop(img, box):
    x1, y1, x2, y2 = box
    m = 0.3 * max(x2 - x1, y2 - y1)
    h, w = img.shape[:2]
    return img[max(int(y1 - m), 0):min(int(y2 + m), h), max(int(x1 - m), 0):min(int(x2 + m), w)]


def stage_frames(a):
    import cv2
    import torch
    from facenet_pytorch import MTCNN
    from huggingface_hub import snapshot_download
    from PIL import Image
    from tqdm import tqdm

    if a.download:
        snapshot_download("yeray142/first-impressions-v2", repo_type="dataset",
                          allow_patterns=["train/*.mp4"], local_dir=a.video_root)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mtcnn = MTCNN(keep_all=False, device=dev)
    df = pd.read_csv(a.dataset)
    ids = df.file_name.tolist()[a.start:a.end]
    for d in ("face", "full"):
        os.makedirs(f"{a.frames}/{d}", exist_ok=True)

    # find every .mp4 under video_root, however the folders are nested (HF or Kaggle)
    paths = {}
    for root, _, files in os.walk(a.video_root):
        for f in files:
            if f.endswith(".mp4"):
                paths[f] = os.path.join(root, f)
    found = sum(fn in paths for fn in df.file_name)
    print(f"found {len(paths)} mp4 files under {a.video_root}; {found}/{len(df)} match dataset.csv")
    if found == 0:
        raise SystemExit("No videos match dataset.csv file_name. Check --video-root.")

    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

    todo = [fn for fn in ids
            if not all(os.path.exists(f"{a.frames}/{d}/{fn}/{N_FRAMES - 1}.jpg") for d in ("face", "full"))]
    print(f"{len(todo)} clips to process ({len(ids) - len(todo)} already done)")
    nw = max(2, os.cpu_count() or 2)
    ex = ThreadPoolExecutor(nw)  # decode several videos at once while the GPU does faces

    def stream():
        q, it = deque(), iter(todo)
        for fn in it:  # keep a small window of videos decoding ahead (bounded memory)
            q.append((fn, ex.submit(lambda p: sample_frames(p) if p else [], paths.get(fn))))
            if len(q) >= nw * 2:
                break
        while q:
            fn, fut = q.popleft()
            nxt = next(it, None)
            if nxt is not None:
                q.append((nxt, ex.submit(lambda p: sample_frames(p) if p else [], paths.get(nxt))))
            yield fn, fut.result()

    log = []
    for fn, frames in tqdm(stream(), total=len(todo)):
        decode_fail = len(frames) == 0
        if decode_fail:
            frames = [np.zeros((SIZE, SIZE, 3), np.uint8)] * N_FRAMES
        try:
            # detect on downscaled copies (much faster), then map boxes back to full size
            sc = min(1.0, 640 / max(frames[0].shape[:2]))
            small = [Image.fromarray(cv2.resize(f, None, fx=sc, fy=sc)) if sc < 1 else Image.fromarray(f)
                     for f in frames]
            boxes, _ = mtcnn.detect(small)
            boxes = [None if b is None else b / sc for b in boxes]
        except Exception:
            boxes = [None] * len(frames)
        fallbacks = 0
        for d in ("face", "full"):
            os.makedirs(f"{a.frames}/{d}/{fn}", exist_ok=True)
        for i, img in enumerate(frames):
            b = boxes[i] if boxes is not None else None
            if b is not None and len(b) > 0:
                face = face_crop(img, b[0])
            else:
                face = center_crop(img)
                fallbacks += 1
            for d, im in (("face", face), ("full", center_crop(img))):
                im = cv2.resize(im, (SIZE, SIZE))
                cv2.imwrite(f"{a.frames}/{d}/{fn}/{i}.jpg", cv2.cvtColor(im, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 90])
        log.append((fn, int(decode_fail), fallbacks))

    if log:
        lp = f"{a.out}/frames_log.csv"
        os.makedirs(a.out, exist_ok=True)
        pd.DataFrame(log, columns=["file_name", "decode_fail", "face_fallbacks"]).to_csv(
            lp, mode="a", header=not os.path.exists(lp), index=False)
        l = pd.DataFrame(log, columns=["file_name", "decode_fail", "face_fallbacks"])
        print(f"this run: {len(l)} clips, decode failures {l.decode_fail.sum()}, "
              f"face fallback rate {l.face_fallbacks.sum() / (len(l) * N_FRAMES):.1%}")


# --------------------------------------------------------------------------- clip
def stage_clip(a):
    import open_clip
    import torch
    from PIL import Image
    from tqdm import tqdm

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model = model.to(dev).eval()
    df = pd.read_csv(a.dataset)
    ids = df.file_name.tolist()
    if a.end is not None:  # allow a small test run
        ids = ids[a.start:a.end]
    shard_dir = f"{a.out}/shards_clip_{a.view}"
    os.makedirs(shard_dir, exist_ok=True)

    for s in range(0, len(ids), a.chunk):
        p = f"{shard_dir}/{s:05d}.npz"
        if os.path.exists(p):
            continue
        chunk, feats = ids[s:s + a.chunk], []
        for fn in tqdm(chunk, desc=f"clip {a.view} {s}"):
            d = f"{a.frames}/{a.view}/{fn}"
            x = torch.stack([pre(Image.open(f"{d}/{i}.jpg").convert("RGB")) for i in range(N_FRAMES)]).to(dev)
            with torch.no_grad():
                e = torch.nn.functional.normalize(model.encode_image(x), dim=-1)
            feats.append(e.cpu().numpy())
        np.savez(p, ids=np.array(chunk, dtype="U"), feats=np.stack(feats))

    shards = [np.load(f"{shard_dir}/{f}") for f in sorted(os.listdir(shard_dir))]
    all_ids = np.concatenate([z["ids"] for z in shards])
    all_feats = np.concatenate([z["feats"] for z in shards]).astype(np.float32)
    suffix = "_test" if a.end is not None else ""
    save_npz(f"{a.out}/clip_b32_{a.view}{suffix}.npz", all_ids, all_feats)


# ------------------------------------------------------------------------ combine
def stage_combine(a):
    df = pd.read_csv(a.dataset)
    name = a.text_model.split("/")[-1]

    def load(path):
        z = np.load(path, allow_pickle=True)  # older text file stored ids as an object array
        f = z["feats"]
        if f.ndim == 3:  # (N, 8, 512) -> mean over frames
            f = f.mean(axis=1)
        return pd.DataFrame(f, index=z["ids"]).loc[df.file_name].values.astype(np.float32)  # align by id

    T = load(f"{a.out}/text_{name}.npz")
    V = load(f"{a.out}/clip_b32_{a.view}.npz")
    train = (df.split == "train").values

    def z(X):  # z-score using TRAIN statistics only
        mu, sd = X[train].mean(0), X[train].std(0) + 1e-8
        return (X - mu) / sd

    Tz, Vz = z(T), z(V)
    Y = df[TRAITS].values.astype(np.float32)
    perm = np.random.RandomState(42).permutation(len(df))  # control: visual features of random other clips
    sets = {
        "text_only": Tz,
        "visual_only": Vz,
        "text_visual": np.concatenate([Tz, Vz], axis=1),
        "text_visual_shuffled": np.concatenate([Tz, Vz[perm]], axis=1),
    }
    for k, X in sets.items():
        path = f"{a.out}/ready_{k}.npz"
        np.savez(path, ids=np.array(df.file_name.tolist(), dtype="U"), X=X.astype(np.float32), Y=Y,
                 split=np.array(df.split.tolist(), dtype="U"), traits=np.array(TRAITS, dtype="U"))
        print("saved", path, "X", X.shape, "Y", Y.shape)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["text", "frames", "clip", "combine"])
    p.add_argument("--dataset", default="dataset.csv")
    p.add_argument("--out", default="features")
    p.add_argument("--text-model", default="all-mpnet-base-v2")
    p.add_argument("--video-root", default="/content/fi2")
    p.add_argument("--frames", default="/content/frames")
    p.add_argument("--view", default="face", choices=["face", "full"])
    p.add_argument("--download", action="store_true", help="download train/*.mp4 from Hugging Face first")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--chunk", type=int, default=500)
    a = p.parse_args()
    {"text": stage_text, "frames": stage_frames, "clip": stage_clip, "combine": stage_combine}[a.stage](a)