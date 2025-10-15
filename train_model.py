# train_model.py  — fast CPU-friendly baseline for demo use

import os, json, random
from pathlib import Path
from collections import defaultdict

# ---------- CPU/threading & reproducibility (set BEFORE TF imports) ----------
SEED = 42
os.environ["PYTHONHASHSEED"]         = str(SEED)
os.environ["TF_NUM_INTRAOP_THREADS"] = "4"   # i3-4005U → 2 cores / 4 threads (try "2" if system feels laggy)
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"]  = "1"   # keep oneDNN on (fast on CPU)

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_fscore_support

random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass
try:
    # XLA can help on CPU for conv-heavy graphs
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

# ---------- Paths & knobs ----------
DATA_DIR   = Path("extracted_data")
MODEL_DIR  = Path("models"); MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Speed-first settings for your spec
USE_CROP     = True      # crop to flaw band → fewer pixels
FORCE_WIDTH  = 512       # 512 is a good sweet spot; try 384 if RAM tight, 640 if quality needs a bit more
BATCH_SIZE   = 24        # bump to 32 if it fits; drop to 16 if OOM
EPOCHS       = 12        # demo-speed; EarlyStopping will stop sooner if no gains
SHUFFLE_BUF  = 512

# TFLite export (fast inference on CPU)
EXPORT_TFLITE = True

# Flaw band (original width 7168 → scaled to manifest width)
FLAW_LEFT_ORIG, FLAW_RIGHT_ORIG, ORIG_WIDTH = 1100, 3100, 7168

# ---------- Manifest / data utilities ----------
def load_manifest():
    mf_path = DATA_DIR / "manifest.json"
    with open(mf_path, "r") as f: mf = json.load(f)
    entries = mf.get("parts") or mf.get("shards")
    if not entries: raise ValueError("manifest.json must contain 'parts' or 'shards'.")
    items = [e for e in entries if "images" in e and "labels" in e]
    if not items: raise ValueError("No entries with 'images' and 'labels'.")
    return {
        "items": items,
        "total": int(mf.get("total", 0)),
        "height": int(mf.get("height", items[0]["height"])),
        "width": int(mf.get("width", items[0]["width"])),
        "channels": int(mf.get("channels", items[0]["channels"])),
    }

def list_part_files(manifest):
    items = sorted(manifest["items"], key=lambda x: (x.get("stem",""), x.get("part",0), x["images"]))
    img_files = [str(x["images"]) for x in items]
    lab_files = [str(x["labels"]) for x in items]
    counts    = [int(x["count"])  for x in items]
    return img_files, lab_files, counts

def build_global_index(lab_files, counts):
    ys = [np.asarray(np.load(lp, mmap_mode="r")) for lp in lab_files]
    y_all = np.concatenate(ys, axis=0)
    if y_all.shape[0] != sum(counts):
        raise ValueError("Label length mismatch with counts.")
    idx_all = np.arange(y_all.shape[0], dtype=np.int64)
    return idx_all, y_all

def make_split_indices(y_all, test_size=0.15, seed=SEED):
    idx_all = np.arange(y_all.shape[0], dtype=np.int64)
    tr, va = train_test_split(idx_all, test_size=test_size, random_state=seed, stratify=y_all)
    return tr, va

def global_to_local_mapper(counts):
    offsets = np.cumsum([0] + counts)
    def g2l(gidx):
        s = int(np.searchsorted(offsets, gidx, side="right") - 1)
        return s, int(gidx - offsets[s])
    return g2l

def scaled_flaw_band(manifest_width):
    scale = manifest_width / float(ORIG_WIDTH)
    left  = max(0, min(int(round(FLAW_LEFT_ORIG  * scale)), manifest_width-1))
    right = max(left+1, min(int(round(FLAW_RIGHT_ORIG * scale)), manifest_width))
    return left, right

# ---------- Dataset pipeline ----------
def dataset_from_indices(img_files, lab_files, counts, gindices,
                         force_width=None, crop_band=None, use_aug=False, repeat=True):
    g2l = global_to_local_mapper(counts)
    by_file = defaultdict(list)
    for g in gindices:
        s, li = g2l(int(g)); by_file[s].append(li)
    file_plan = [(s, sorted(li)) for s, li in sorted(by_file.items())]

    def gen():
        for s, local_idxs in file_plan:
            X = np.load(img_files[s], mmap_mode="r")  # (N,H,W,1) float16
            y = np.load(lab_files[s], mmap_mode="r")  # (N,)     uint8
            for li in local_idxs:
                yield X[li], y[li]

    output_signature = (
        tf.TensorSpec(shape=(None, None, 1), dtype=tf.float16),
        tf.TensorSpec(shape=(), dtype=tf.uint8),
    )
    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)

    crop_l = crop_r = None
    if crop_band is not None:
        crop_l, crop_r = crop_band

    # very light aug (off by default for fastest baseline)
    aug = keras.Sequential([
        layers.RandomContrast(0.08),
        layers.RandomBrightness(0.08),
    ]) if use_aug else None

    def prep(x, y):
        x = tf.cast(x, tf.float32)  # from float16
        if crop_l is not None:      # crop width
            x = x[:, crop_l:crop_r, :]
        if force_width is not None: # shrink width
            h = tf.shape(x)[0]
            x = tf.image.resize(x, (h, force_width), method="area")
        if aug is not None:
            x = aug(x, training=True)
        return x, tf.cast(y, tf.float32)

    if repeat:
        ds = ds.shuffle(SHUFFLE_BUF, seed=SEED, reshuffle_each_iteration=True)
        ds = ds.map(prep, num_parallel_calls=tf.data.AUTOTUNE).repeat().batch(BATCH_SIZE)
        ds = ds.prefetch(tf.data.AUTOTUNE)
    else:
        ds = ds.map(prep, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE)

    opts = tf.data.Options(); opts.experimental_deterministic = True
    return ds.with_options(opts)

# ---------- Model (depthwise separable: very fast on CPU) ----------
def ds_block(x, filters, stride=1, rate=1):
    x = layers.DepthwiseConv2D(3, strides=stride, padding="same",
                               depth_multiplier=1, dilation_rate=rate, use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    return x

def make_model(input_shape):
    inputs = keras.Input(shape=input_shape)

    # Early downsample to reduce compute
    x = layers.Conv2D(24, 3, padding="same", strides=2, use_bias=False)(inputs)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)

    x = ds_block(x, 32, stride=2)
    x = ds_block(x, 48, stride=2)
    x = ds_block(x, 72, stride=2)
    x = ds_block(x, 96, stride=2)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(96, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc"), "accuracy"],
    )
    return model

# ---------- Train / Evaluate ----------
def main():
    manifest = load_manifest()
    img_files, lab_files, counts = list_part_files(manifest)
    idx_all, y_all = build_global_index(lab_files, counts)

    pos, neg, mean = int(y_all.sum()), int((y_all==0).sum()), float(y_all.mean())
    print(f"Total: {y_all.size}  Positives: {pos}  Negatives: {neg}")
    print(f"Class balance (mean ~0.5): {mean:.6f}")

    idx_tr, idx_va = make_split_indices(y_all, test_size=0.15, seed=SEED)
    y_val = y_all[idx_va]

    H, W, C = manifest["height"], manifest["width"], manifest["channels"]

    crop_band = None
    if USE_CROP:
        crop_l, crop_r = scaled_flaw_band(W)
        crop_band = (crop_l, crop_r)
        W_eff = FORCE_WIDTH if FORCE_WIDTH is not None else (crop_r - crop_l)
        input_shape = (H, W_eff, C)
    else:
        input_shape = (H, (FORCE_WIDTH if FORCE_WIDTH is not None else W), C)

    print("Input shape:", input_shape)
    print("Flaw band (scaled):", crop_band if crop_band else "none")
    print("Total samples:", y_all.shape[0], "| Train:", idx_tr.shape[0], "Val:", idx_va.shape[0])

    train_ds = dataset_from_indices(
        img_files, lab_files, counts, idx_tr,
        force_width=FORCE_WIDTH if not USE_CROP else None,
        crop_band=crop_band if USE_CROP else None,
        use_aug=False, repeat=True
    )
    val_ds = dataset_from_indices(
        img_files, lab_files, counts, idx_va,
        force_width=FORCE_WIDTH if not USE_CROP else None,
        crop_band=crop_band if USE_CROP else None,
        use_aug=False, repeat=True
    )

    steps_per_epoch = max(idx_tr.shape[0] // BATCH_SIZE, 1)
    val_steps       = max(idx_va.shape[0] // BATCH_SIZE, 1)
    print("Steps/epoch:", steps_per_epoch, "| Val steps:", val_steps)

    model = make_model(input_shape)

    class StepLogger(keras.callbacks.Callback):
        def on_train_batch_end(self, batch, logs=None):
            if batch % 50 == 0 and logs:
                try:
                    print(f"  step {batch} - loss={logs.get('loss'):.4f} acc={logs.get('accuracy'):.4f} auc={logs.get('auc'):.4f}")
                except Exception:
                    pass

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_DIR / "best.keras"),
            monitor="val_auc", mode="max", save_best_only=True, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=3, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc", mode="max", factor=0.5, patience=1, min_lr=1e-6, verbose=1
        ),
        keras.callbacks.CSVLogger(str(MODEL_DIR / "training_log.csv")),
        StepLogger(),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=1,
    )

    # Save final model
    model.save(MODEL_DIR / "final.keras")
    print("Saved model:", (MODEL_DIR / "final.keras").resolve())

    # ---------- Post-training evaluation ----------
    val_pred_ds = dataset_from_indices(
        img_files, lab_files, counts, idx_va,
        force_width=FORCE_WIDTH if not USE_CROP else None,
        crop_band=crop_band if USE_CROP else None,
        use_aug=False, repeat=False
    )
    y_prob = model.predict(val_pred_ds, verbose=1).ravel()
    y_prob = y_prob[: y_val.shape[0]]

    auc = roc_auc_score(y_val, y_prob)
    print(f"VAL AUC: {auc:.6f}")

    thr = 0.5
    y_pred_05 = (y_prob >= thr).astype(int)
    cm_05 = confusion_matrix(y_val, y_pred_05)
    prec_05, rec_05, f1_05, _ = precision_recall_fscore_support(y_val, y_pred_05, average="binary", zero_division=0)
    print("Confusion matrix @0.5:\n", cm_05)
    print(f"@0.5 Precision: {prec_05:.3f}  Recall: {rec_05:.3f}  F1: {f1_05:.3f}")

    # Scan thresholds for best F1
    thresholds = np.linspace(0.05, 0.95, 19)
    best_f1, best_thr = 0.0, 0.5
    for t in thresholds:
        yp = (y_prob >= t).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(y_val, yp, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = float(f1), float(t)

    yp_best = (y_prob >= best_thr).astype(int)
    cm_best = confusion_matrix(y_val, yp_best)
    prec_b, rec_b, f1_b, _ = precision_recall_fscore_support(y_val, yp_best, average="binary", zero_division=0)
    print(f"Best-F1 threshold: {best_thr:.2f}  F1: {best_f1:.3f}")
    print("Confusion matrix @best-F1:\n", cm_best)
    print(f"@best Precision: {prec_b:.3f}  Recall: {rec_b:.3f}  F1: {f1_b:.3f}")

    # Write metrics
    metrics_path = (MODEL_DIR / "metrics.txt").resolve()
    with open(metrics_path, "w") as f:
        f.write(f"VAL_AUC={auc:.6f}\n")
        f.write(f"THR_05_PREC={prec_05:.6f}  THR_05_REC={rec_05:.6f}  THR_05_F1={f1_05:.6f}\n")
        f.write(f"BEST_THR={best_thr:.4f}  BEST_F1={best_f1:.6f}  BEST_PREC={prec_b:.6f}  BEST_REC={rec_b:.6f}\n")
    print("Wrote metrics to:", metrics_path)
    print("models/ directory contains:", [p.name for p in MODEL_DIR.iterdir()])

    # ---------- Optional: export TFLite for very fast CPU inference ----------
    if EXPORT_TFLITE:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]  # dynamic range quant (8-bit weights)
        tflite = converter.convert()
        tfl_path = (MODEL_DIR / "model_int8.tflite").resolve()
        tfl_path.write_bytes(tflite)
        print("Wrote TFLite model:", tfl_path)

if __name__ == "__main__":
    main()
