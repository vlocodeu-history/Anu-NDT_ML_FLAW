# predict_image.py
import sys, json, numpy as np, tensorflow as tf
from pathlib import Path
from PIL import Image

USE_CROP     = True
FORCE_WIDTH  = 512
FLAW_LEFT_ORIG, FLAW_RIGHT_ORIG, ORIG_WIDTH = 1100, 3100, 7168
DATA_DIR, MODEL_DIR = Path("extracted_data"), Path("models")

def _scaled_band(w):
    left  = int(round(FLAW_LEFT_ORIG  * (w/ORIG_WIDTH))); left  = max(0, min(left,  w-1))
    right = int(round(FLAW_RIGHT_ORIG * (w/ORIG_WIDTH))); right = max(left+1, min(right, w))
    return left, right

def predict_image(path: Path, thr=0.5):
    mf = json.loads((DATA_DIR/"manifest.json").read_text())
    H = mf["height"]

    img = Image.open(path).convert("L")
    x = np.array(img, dtype=np.float32)/255.0
    x = x[..., None]  # (h,w,1)

    if x.shape[0] != H:
        new_w = int(round(x.shape[1]*(H/float(x.shape[0]))))
        x = tf.image.resize(x, (H, new_w), method="area").numpy()

    L, R = _scaled_band(x.shape[1]); x = x[:, L:R, :]
    if FORCE_WIDTH is not None:
        x = tf.image.resize(x, (x.shape[0], FORCE_WIDTH), method="area").numpy()

    x = x[None, ...]
    mp = MODEL_DIR/"best.keras"
    if not mp.exists(): mp = MODEL_DIR/"final.keras"
    model = tf.keras.models.load_model(mp)
    prob = float(model.predict(x, verbose=0).ravel()[0])
    return prob, ("FLAW" if prob>=thr else "NO FLAW")

if __name__ == "__main__":
    img_path = Path(sys.argv[1])
    thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    prob, pred = predict_image(img_path, thr=thr)
    print(f"prob={prob:.4f}  pred={pred}  (thr={thr})")
