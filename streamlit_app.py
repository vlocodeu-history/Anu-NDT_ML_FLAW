import os
import json
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image  # needed for fallback resizing / uploads

# Optional imports guarded at runtime
TF_OK = True
try:
    import tensorflow as tf
    from tensorflow import keras
except Exception as e:
    TF_OK = False

# -----------------------------
# Paths & basic config
# -----------------------------
DATA_DIR   = Path("extracted_data")
MODEL_DIR  = Path("models")
UPLOAD_DIR = Path("uploads"); UPLOAD_DIR.mkdir(exist_ok=True, parents=True)

# Must match training/prediction scripts for consistent preprocessing
USE_CROP     = True
FORCE_WIDTH  = 512
FLAW_LEFT_ORIG, FLAW_RIGHT_ORIG, ORIG_WIDTH = 1100, 3100, 7168

# -----------------------------
# Small helpers (mirrors train_model.py/predict_any.py behavior)
# -----------------------------
@st.cache_data(show_spinner=False)
def load_manifest():
    mf_path = DATA_DIR / "manifest.json"
    if not mf_path.exists():
        return None
    mf = json.loads(mf_path.read_text())
    parts = mf.get("parts") or mf.get("shards") or []
    items = [e for e in parts if "images" in e and "labels" in e]
    return {
        "items": items,
        "total": int(mf.get("total", 0)),
        "height": int(mf.get("height", items[0]["height"] if items else 480)),
        "width": int(mf.get("width", items[0]["width"] if items else 7168)),
        "channels": int(mf.get("channels", items[0]["channels"] if items else 1)),
    }

def scaled_band(w: int) -> Tuple[int, int]:
    L = int(round(FLAW_LEFT_ORIG*(w/ORIG_WIDTH))); L = max(0, min(L, w-1))
    R = int(round(FLAW_RIGHT_ORIG*(w/ORIG_WIDTH))); R = max(L+1, min(R, w))
    return L, R

def preprocess_strip(x: np.ndarray, target_H: int) -> np.ndarray:
    # x: (H,W) or (H,W,1) float32 in [0,1]
    if x.ndim == 2:
        x = x[..., None]
    if x.shape[0] != target_H:
        new_w = int(round(x.shape[1]*(target_H/float(x.shape[0]))))
        if TF_OK:
            x = tf.image.resize(x, (target_H, new_w), method="area").numpy()
        else:
            # fallback: PIL (nearest) if TF not present
            x = np.array(Image.fromarray((x[...,0]*255).astype(np.uint8)).resize((new_w, target_H)))
            x = x.astype(np.float32)/255.0
            x = x[...,None]
        if not TF_OK:
            st.sidebar.warning(
            "TensorFlow isn’t available on this deployment. "
            "‘Predict (shard)’ and ‘Threshold Tuning’ need TF. "
            "Use local run or deploy on Render/HF Spaces for full features."
            )
    if USE_CROP:
        L, R = scaled_band(x.shape[1])
        x = x[:, L:R, :]
    if FORCE_WIDTH is not None and TF_OK:
        x = tf.image.resize(x, (x.shape[0], FORCE_WIDTH), method="area").numpy()
    elif FORCE_WIDTH is not None and not TF_OK:
        new_w = FORCE_WIDTH
        x = np.array(Image.fromarray((x[...,0]*255).astype(np.uint8)).resize((new_w, x.shape[0])))
        x = (x.astype(np.float32)/255.0)[...,None]
    return x

@st.cache_resource(show_spinner=False)
def load_model_and_threshold():
    if not TF_OK:
        return None, 0.5
    mp = MODEL_DIR/"best.keras"
    if not mp.exists():
        mp = MODEL_DIR/"final.keras"
    model = None
    if mp.exists():
        model = tf.keras.models.load_model(mp)
    thr = 0.5
    thr_path = MODEL_DIR/"threshold.txt"
    if thr_path.exists():
        try:
            thr = float(thr_path.read_text().strip())
        except Exception:
            pass
    return model, thr

def list_part_files(manifest):
    items = sorted(manifest["items"], key=lambda x: (x.get("stem",""), x.get("part",0), x["images"]))
    img_files = [str(x["images"]) for x in items]
    lab_files = [str(x["labels"]) for x in items]
    counts    = [int(x["count"])  for x in items]
    return img_files, lab_files, counts

def build_global_index(lab_files: List[str], counts: List[int]):
    ys = [np.asarray(np.load(lp, mmap_mode="r")) for lp in lab_files]
    y_all = np.concatenate(ys, axis=0)
    idx_all = np.arange(y_all.shape[0], dtype=np.int64)
    return idx_all, y_all

def global_to_local_mapper(counts: List[int]):
    offsets = np.cumsum([0] + counts)
    def g2l(gidx):
        s = int(np.searchsorted(offsets, gidx, side="right") - 1)
        return s, int(gidx - offsets[s])
    return g2l

def dataset_from_indices(img_files, lab_files, counts, gindices,
                         force_width=None, crop_band=None, batch=24):
    if not TF_OK:
        raise RuntimeError("TensorFlow not available. Install tensorflow to compute validation metrics.")
    from collections import defaultdict
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

    def prep(x, y):
        x = tf.cast(x, tf.float32)
        if crop_l is not None:
            x = x[:, crop_l:crop_r, :]
        if force_width is not None:
            h = tf.shape(x)[0]
            x = tf.image.resize(x, (h, force_width), method="area")
        return x, tf.cast(y, tf.float32)

    ds = ds.map(prep, num_parallel_calls=tf.data.AUTOTUNE).batch(batch)
    return ds

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="NDT Flaw Detection – Streamlit", layout="wide")

st.sidebar.title("NDT Flaw UI")
page = st.sidebar.radio("Go to", [
    "About Project",
    "Overview", "Predict", "Threshold Tuning", "Validate (Stats)", "Explore Dataset", "Training Logs"
])

st.sidebar.markdown("---")
st.sidebar.caption("Paths")
st.sidebar.code(f"DATA_DIR  = {DATA_DIR.resolve()}\nMODEL_DIR = {MODEL_DIR.resolve()}")

manifest = load_manifest()
model, saved_thr = load_model_and_threshold()

# -------- About Project --------
if page == "About Project":
    st.title("About Project")
    st.markdown("""
    ## 🧠 About the Project

    ### 🔍 Purpose
    This project focuses on **automated flaw detection in Non-Destructive Testing (NDT)** using deep learning.
    It helps identify **flaw (defect)** and **no-flaw (normal)** regions in industrial inspection images
    (ultrasonic, X-ray, etc.) to improve quality, safety, and throughput.

    ---

    ### 🧩 Dataset
    We use the **Koomas NDT_ML_Flaw** dataset. Data is extracted into `.npy` shards
    (`images.f16.npy` and `labels.u1.npy`) with a `manifest.json` that indexes parts and counts for efficient IO.

    ---

    ### 🧠 Model Overview
    A CNN is trained (via `train_model.py`) to classify each image strip as:
    - `1 → Flaw`
    - `0 → No Flaw`

    Preprocessing includes normalization and cropping a specific flaw band (≈ 1100–3100 px from a 7168-px width),
    mirroring training/inference for consistency. Models and artifacts are saved in `models/` (`best.keras`,
    optional `threshold.txt`, `training_log.csv`, `metrics.txt`).

    ---

    ### ⚡ Prediction Pipeline
    1) Load the trained model (`best.keras` / `final.keras`).
    2) Preprocess (crop, resize, normalize) each strip.
    3) Output a **probability** for *flaw*.
    4) Apply a configurable **decision threshold** to classify **Flaw** vs **No Flaw**.

    ---

    ### 📊 Streamlit Features
    - **Predict:** Upload PNG/JPG or pick a row from dataset shards for inference.
    - **Threshold Tuning:** Confusion matrix, Precision, Recall, F1, ROC-AUC vs threshold.
    - **Validate (Stats):** Class balance and shard size distribution.
    - **Explore Dataset:** Browse sample rows and labels.
    - **Training Logs:** View training CSV logs and metrics.

    ---

    ### 🏭 Industrial Application
    Integrates into QC pipelines to automate detection, reduce inspection time, and provide auditable statistics.

    """)

# -------- Overview --------
elif page == "Overview":
    st.title("NDT Flaw/No-Flaw – Streamlit Frontend")

    st.markdown("""
    This app lets you **predict flaws** on single images or dataset shards and view **statistical metrics**
    like confusion matrix, ROC-AUC, and threshold tuning. It is compatible with the training/prediction
    code you already have.
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("TensorFlow", "Available" if TF_OK and model is not None else "Not ready")
    with col2:
        st.metric("Models found", "Yes" if (MODEL_DIR/"best.keras").exists() or (MODEL_DIR/"final.keras").exists() else "No")
    with col3:
        st.metric("Manifest", "Loaded" if manifest else "Missing")

    st.subheader("Dataset Summary")
    if manifest:
        st.write({
            "Total samples": manifest["total"],
            "Image shape": (manifest["height"], manifest["width"], manifest["channels"]),
            "Shards": len(manifest["items"])
        })
    else:
        st.info("Place your extracted dataset under `extracted_data/` and ensure `manifest.json` exists.")

    st.subheader("Quick Start")
    st.markdown("""
    1. **Upload** a PNG/JPG or select a dataset shard in **Predict**.\n
    2. **Tune threshold** to match your precision/recall goals in **Threshold Tuning**.\n
    3. Compute **validation metrics** in **Validate (Stats)**.\n
    4. Inspect **training logs** in **Training Logs**.
    """)

# -------- Predict --------
elif page == "Predict":
    st.title("Predict – Single image or dataset shard")

    if model is None:
        st.error("Model not found. Train a model and place it under `models/best.keras` or `models/final.keras`.")
    else:
        st.caption(f"Loaded model. Default threshold = {saved_thr:.3f}")

    tab_file, tab_shard = st.tabs(["Upload Image", "Pick from Shard (.npy)"])

    with tab_file:
        up = st.file_uploader("Upload PNG/JPG image", type=["png","jpg","jpeg"], accept_multiple_files=False)
        thr = st.slider("Decision threshold (probability for 'FLAW')", 0.05, 0.95, float(saved_thr), 0.01)

        if up and model is not None:
            pth = UPLOAD_DIR / up.name
            pth.write_bytes(up.read())
            img = Image.open(pth).convert("L")
            x = np.array(img, dtype=np.float32)/255.0

            target_H = manifest["height"] if manifest else 480
            x = preprocess_strip(x, target_H)

            prob = float(model.predict(x[None,...], verbose=0).ravel()[0])
            pred = "FLAW" if prob >= thr else "NO FLAW"

            st.subheader("Result")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Prediction", pred)
            with col2:
                st.metric("Probability", f"{prob:.4f}")
            with col3:
                st.metric("Threshold", f"{thr:.3f}")

            st.subheader("Preprocessed View")
            fig = plt.figure(figsize=(8,3))
            plt.imshow(x[...,0], cmap="gray", aspect="auto")
            plt.title(f"Processed strip – shape {x.shape}")
            plt.axis("off")
            st.pyplot(fig, clear_figure=True)

    with tab_shard:
        if not manifest:
            st.info("No manifest found. Put dataset shards under `extracted_data/` and run your extraction first.")
        else:
            img_files, lab_files, counts = list_part_files(manifest)
            # build index once
            idx_all, y_all = build_global_index(lab_files, counts)

            st.write(f"Total rows across shards: **{idx_all.size:,}**")
            shard_idx = st.number_input("Global row index", min_value=0, max_value=int(idx_all.size-1), value=0, step=1)
            thr2 = st.slider("Decision threshold", 0.05, 0.95, float(saved_thr), 0.01, key="thr2")

            if model is not None and st.button("Predict This Row"):
                # map global index to local file + row
                offsets = np.cumsum([0] + counts)
                s = int(np.searchsorted(offsets, shard_idx, side="right") - 1)
                li = int(shard_idx - offsets[s])

                X = np.load(img_files[s], mmap_mode="r")  # (N,H,W,1)
                y = np.load(lab_files[s], mmap_mode="r")
                x = X[li].astype(np.float32)  # (H,W,1)

                # crop/resize to match training behavior
                H = manifest["height"]
                if USE_CROP:
                    L, R = scaled_band(x.shape[1])
                    x = x[:, L:R, :]
                if FORCE_WIDTH is not None and TF_OK:
                    x = tf.image.resize(x, (H, FORCE_WIDTH), method="area").numpy()

                prob = float(model.predict(x[None,...], verbose=0).ravel()[0])
                pred = "FLAW" if prob >= thr2 else "NO FLAW"
                y_true = int(y[li])

                st.subheader("Result")
                col1, col2, col3, col4 = st.columns(4)
                with col1: st.metric("Prediction", pred)
                with col2: st.metric("Probability", f"{prob:.4f}")
                with col3: st.metric("Threshold", f"{thr2:.3f}")
                with col4: st.metric("Ground truth", "FLAW" if y_true==1 else "NO FLAW")

                fig = plt.figure(figsize=(8,3))
                plt.imshow(x[...,0], cmap="gray", aspect="auto")
                plt.title(f"Shard view – file {Path(img_files[s]).name} row {li}")
                plt.axis("off")
                st.pyplot(fig, clear_figure=True)

# -------- Threshold Tuning --------
elif page == "Threshold Tuning":
    st.title("Threshold tuning (Validation set)")
    if not (manifest and TF_OK and model is not None):
        st.warning("Need manifest + TensorFlow + a trained model to compute curves.")
    else:
        img_files, lab_files, counts = list_part_files(manifest)
        idx_all, y_all = build_global_index(lab_files, counts)

        # simple deterministic split like train_model.py (15% validation)
        from sklearn.model_selection import train_test_split
        idx_tr, idx_va = train_test_split(np.arange(y_all.shape[0], dtype=np.int64),
                                          test_size=0.15, random_state=42, stratify=y_all)
        y_val = y_all[idx_va]
        crop_l, crop_r = (scaled_band(manifest["width"]) if USE_CROP else (None, None))
        crop_band = (crop_l, crop_r) if USE_CROP else None

        val_ds = dataset_from_indices(
            img_files, lab_files, counts, idx_va,
            force_width=None if USE_CROP else FORCE_WIDTH,
            crop_band=crop_band, batch=48
        )
        y_prob = model.predict(val_ds, verbose=1).ravel()
        y_prob = y_prob[: y_val.shape[0]]

        st.write(f"Validation samples: **{y_val.shape[0]:,}**")
        thr = st.slider("Decision threshold", 0.05, 0.95, float(saved_thr), 0.01)

        # Confusion matrix at chosen threshold
        y_pred = (y_prob >= thr).astype(int)
        from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_fscore_support, roc_curve
        cm = confusion_matrix(y_val, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_val, y_pred, average="binary", zero_division=0)
        auc = roc_auc_score(y_val, y_prob)

        c0, c1, c2, c3 = st.columns(4)
        c0.metric("AUC", f"{auc:.3f}")
        c1.metric("Precision", f"{prec:.3f}")
        c2.metric("Recall", f"{rec:.3f}")
        c3.metric("F1", f"{f1:.3f}")

        st.subheader("Confusion matrix")
        fig_cm = plt.figure(figsize=(4,4))
        plt.imshow(cm, interpolation='nearest')
        plt.xticks([0,1], ["NO FLAW","FLAW"])
        plt.yticks([0,1], ["NO FLAW","FLAW"])
        for (i,j), v in np.ndenumerate(cm):
            plt.text(j, i, f"{v}", ha="center", va="center")
        plt.xlabel("Predicted"); plt.ylabel("True")
        st.pyplot(fig_cm, clear_figure=True)

        # ROC curve
        fpr, tpr, _ = roc_curve(y_val, y_prob)
        fig_roc = plt.figure(figsize=(4,4))
        plt.plot(fpr, tpr)
        plt.plot([0,1],[0,1], linestyle="--")
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve")
        st.pyplot(fig_roc, clear_figure=True)

# -------- Validate (Stats) --------
elif page == "Validate (Stats)":
    st.title("Dataset class distribution & shard stats")
    if not manifest:
        st.info("Manifest missing.")
    else:
        img_files, lab_files, counts = list_part_files(manifest)
        idx_all, y_all = build_global_index(lab_files, counts)
        num_flaw = int(y_all.sum()); num_ok = int((y_all==0).sum())

        colA, colB = st.columns(2)
        with colA:
            st.metric("Total", f"{y_all.size:,}")
            st.metric("Flaws (1)", f"{num_flaw:,}")
            st.metric("No-Flaw (0)", f"{num_ok:,}")
            st.metric("Class mean", f"{y_all.mean():.4f}")
        with colB:
            fig = plt.figure(figsize=(4,4))
            plt.pie([num_ok, num_flaw], labels=["No-Flaw","Flaw"], autopct="%1.1f%%", startangle=90)
            plt.title("Class distribution")
            st.pyplot(fig, clear_figure=True)

        st.subheader("Shard sizes")
        fig2 = plt.figure(figsize=(8,3))
        plt.bar(np.arange(len(counts)), counts)
        plt.xlabel("Shard index"); plt.ylabel("Rows")
        st.pyplot(fig2, clear_figure=True)

# -------- Explore Dataset --------
elif page == "Explore Dataset":
    st.title("Browse a few samples")
    if not manifest:
        st.info("Manifest missing.")
    else:
        img_files, lab_files, counts = list_part_files(manifest)
        shard_sel = st.number_input("Select shard", 0, len(img_files)-1, 0, 1)
        X = np.load(img_files[shard_sel], mmap_mode="r")
        y = np.load(lab_files[shard_sel], mmap_mode="r")
        n = X.shape[0]

        start = st.slider("Row range start", 0, max(0, n-8), 0, 1)
        show_n = st.slider("How many to show", 1, min(8, n-start), 4, 1)

        cols = st.columns(show_n)
        for i in range(show_n):
            idx = start + i
            xi = X[idx][...,0]
            yi = int(y[idx])
            with cols[i]:
                st.caption(f"Row {idx} – {'FLAW' if yi==1 else 'NO FLAW'}")
                fig = plt.figure(figsize=(3,1.5))
                plt.imshow(xi, cmap="gray", aspect="auto")
                plt.axis("off")
                st.pyplot(fig, clear_figure=True)

# -------- Training Logs --------
elif page == "Training Logs":
    st.title("Training logs & metrics")
    log_path = MODEL_DIR / "training_log.csv"
    met_path = MODEL_DIR / "metrics.txt"

    cols = st.columns(2)
    with cols[0]:
        st.subheader("CSV log")
        if log_path.exists():
            import pandas as pd  # for nice table; add to your requirements
            df = pd.read_csv(log_path)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No training_log.csv found.")

    with cols[1]:
        st.subheader("Key metrics")
        if met_path.exists():
            st.code(met_path.read_text())
        else:
            st.info("No metrics.txt found.")

    st.caption("Tip: Run your training script; this page will read its outputs from the models/ folder.")
