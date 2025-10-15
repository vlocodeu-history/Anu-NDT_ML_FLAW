# predict_any.py  — works with .npy shards (by index) or PNG/JPG images

import argparse, json, os
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image

# ---- must match your training flags ----
USE_CROP     = True
FORCE_WIDTH  = 512
FLAW_LEFT_ORIG, FLAW_RIGHT_ORIG, ORIG_WIDTH = 1100, 3100, 7168

DATA_DIR, MODEL_DIR = Path("extracted_data"), Path("models")

def scaled_band(w: int):
    L = int(round(FLAW_LEFT_ORIG*(w/ORIG_WIDTH))); L = max(0, min(L, w-1))
    R = int(round(FLAW_RIGHT_ORIG*(w/ORIG_WIDTH))); R = max(L+1, min(R, w))
    return L, R

def load_model_and_thr():
    mp = MODEL_DIR/"best.keras"
    if not mp.exists():
        mp = MODEL_DIR/"final.keras"
    model = tf.keras.models.load_model(mp)
    thr_path = MODEL_DIR/"threshold.txt"
    thr = float(thr_path.read_text().strip()) if thr_path.exists() else 0.5
    return model, thr

def preprocess_strip(x: np.ndarray, target_H: int):
    # x: (H, W, 1) float32 in [0,1] or (H, W) --> we’ll expand last dim if needed
    if x.ndim == 2:
        x = x[..., None]
    # if height differs, scale to dataset H while keeping aspect ratio
    if x.shape[0] != target_H:
        new_w = int(round(x.shape[1]*(target_H/float(x.shape[0]))))
        x = tf.image.resize(x, (target_H, new_w), method="area").numpy()
    # crop + optional width resize
    if USE_CROP:
        L, R = scaled_band(x.shape[1])
        x = x[:, L:R, :]
    if FORCE_WIDTH is not None:
        x = tf.image.resize(x, (x.shape[0], FORCE_WIDTH), method="area").numpy()
    return x

def predict_png_or_jpg(path: Path, model, thr: float, target_H: int):
    img = Image.open(path).convert("L")
    x = np.array(img, dtype=np.float32)/255.0
    x = preprocess_strip(x, target_H)
    prob = float(model.predict(x[None,...], verbose=0).ravel()[0])
    pred = "FLAW" if prob >= thr else "NO FLAW"
    return prob, pred, None  # no ground-truth label

def guess_label_path(images_path: Path) -> Path | None:
    # typical pair: batch_XXX_pYYY_images.f16.npy  <->  batch_XXX_pYYY_labels.u1.npy
    name = images_path.name
    if "_images" in name:
        label_name = name.replace("_images.f16.npy", "_labels.u1.npy")
        lp = images_path.with_name(label_name)
        return lp if lp.exists() else None
    return None

def predict_npy_row(npy_path: Path, idx: int, model, thr: float, target_H: int):
    X = np.load(npy_path, mmap_mode="r")  # shape (N,H,W,1) float16 in [0,1]
    if idx < 0 or idx >= X.shape[0]:
        raise ValueError(f"idx out of range 0..{X.shape[0]-1}")
    x = X[idx].astype(np.float32)  # (H,W,1)
    x = preprocess_strip(x, target_H)
    prob = float(model.predict(x[None,...], verbose=0).ravel()[0])
    pred = "FLAW" if prob >= thr else "NO FLAW"

    y_true = None
    lp = guess_label_path(npy_path)
    if lp is not None:
        y = np.load(lp, mmap_mode="r")
        y_true = int(y[idx])
    return prob, pred, y_true

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="PNG/JPG image OR dataset shard .npy")
    ap.add_argument("--idx", type=int, default=None, help="Row index for .npy shards")
    ap.add_argument("--thr", type=float, default=None, help="Override threshold")
    args = ap.parse_args()

    # read dataset H from manifest (used to scale external images)
    mf = json.loads((DATA_DIR/"manifest.json").read_text())
    target_H = int(mf["height"])

    model, saved_thr = load_model_and_thr()
    thr = args.thr if args.thr is not None else saved_thr

    p = Path(args.path)
    ext = p.suffix.lower()

    if ext == ".npy":
        if args.idx is None:
            raise SystemExit("For .npy shards you must pass --idx <row> (e.g., --idx 0)")
        prob, pred, y_true = predict_npy_row(p, args.idx, model, thr, target_H)
    else:
        prob, pred, y_true = predict_png_or_jpg(p, model, thr, target_H)

    msg = f"prob={prob:.4f}  pred={pred}  thr={thr:.3f}"
    if y_true is not None:
        msg += f"  (label={y_true})"
    print(msg)

if __name__ == "__main__":
    main()
