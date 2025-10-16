import os
import json
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image  # for resizing / uploads

# Optional: ONNX Runtime for inference (no TensorFlow)
ORT_OK = True
try:
    import onnxruntime as ort
except Exception:
    ORT_OK = False

# -----------------------------
# Paths & basic config
# -----------------------------
DATA_DIR   = Path("extracted_data")
MODEL_DIR  = Path(os.getenv("MODEL_DIR", "models"))
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)
MODEL_DIR.mkdir(exist_ok=True, parents=True)

# Must match training/prediction scripts for consistent preprocessing
USE_CROP     = True
FORCE_WIDTH  = 512
FLAW_LEFT_ORIG, FLAW_RIGHT_ORIG, ORIG_WIDTH = 1100, 3100, 7168

# -----------------------------
# Small helpers (TF-free)
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

def _pil_resize_gray(x_01: np.ndarray, new_h: int = None, new_w: int = None) -> np.ndarray:
    """
    x_01: float32 [0,1], shape (H,W) or (H,W,1)
    Returns float32 [0,1], shape (new_h,new_w,1)
    """
    if x_01.ndim == 3 and x_01.shape[-1] == 1:
        x2d = x_01[..., 0]
    else:
        x2d = x_01
    x8 = (np.clip(x2d, 0, 1)*255.0).astype(np.uint8)
    img = Image.fromarray(x8, mode="L")
    if new_h is not None and new_w is not None:
        img = img.resize((new_w, new_h), resample=Image.BILINEAR)
    elif new_h is not None:
        h, w = x2d.shape[:2]
        new_w = int(round(w*(new_h/float(h))))
        img = img.resize((new_w, new_h), resample=Image.BILINEAR)
    elif new_w is not None:
        h, w = x2d.shape[:2]
        new_h = int(round(h*(new_w/float(w))))
        img = img.resize((new_w, new_h), resample=Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr[..., None]

def preprocess_strip(x: np.ndarray, target_H: int) -> np.ndarray:
    # x: (H,W) or (H,W,1) float32 in [0,1]
    if x.ndim == 2:
        x = x[..., None]
    # 1) resize by height
    if x.shape[0] != target_H:
        x = _pil_resize_gray(x, new_h=target_H)
    # 2) crop flaw band if enabled
    if USE_CROP:
        L, R = scaled_band(x.shape[1])
        x = x[:, L:R, :]
    # 3) force width
    if FORCE_WIDTH is not None:
        x = _pil_resize_gray(x, new_w=FORCE_WIDTH)
    return x.astype(np.float32)

@st.cache_resource(show_spinner=False)
def load_onnx_and_threshold():
    """
    Loads ONNX model and threshold if available.
    Returns (onnx_session_or_None, thr_float)
    """
    thr = 0.5
    thr_path = MODEL_DIR / "threshold.txt"
    if thr_path.exists():
        try:
            thr = float(thr_path.read_text().strip())
        except Exception:
            pass

    if not ORT_OK:
        return None, thr

    mp = None
    for name in ["best.onnx", "final.onnx", "model.onnx"]:
        p = MODEL_DIR / name
        if p.exists():
            mp = p
            break
    if mp is None:
        return None, thr

    try:
        sess = ort.InferenceSession(str(mp), providers=["CPUExecutionProvider"])
        return sess, thr
    except Exception as e:
        st.error(f"Failed to load ONNX model: {e}")
        return None, thr

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

def _discover_io(sess: "ort.InferenceSession"):
    """Return (inp_name, out_name, layout) where layout ∈ {'NHWC1','NCHW','NHW'}."""
    inps = sess.get_inputs()
    outs = sess.get_outputs()
    inp_name = inps[0].name
    out_name = outs[0].name
    shape = inps[0].shape
    if len(shape) == 4:
        layout = "NCHW" if str(shape[1]) in ("1", "1?") else "NHWC1"
    elif len(shape) == 3:
        layout = "NHW"
    else:
        layout = "NHWC1"
    return inp_name, out_name, layout

def _pack_batch(batch_x: List[np.ndarray], layout: str) -> np.ndarray:
    """batch_x: list of (H,W,1) float32 [0,1]"""
    if not batch_x:
        return np.zeros((0,1,1,1), dtype=np.float32)
    H, W = batch_x[0].shape[:2]
    nhwc1 = np.stack([b.reshape(H, W, 1) for b in batch_x], axis=0)  # (N,H,W,1)
    if layout == "NCHW":
        return np.transpose(nhwc1, (0,3,1,2)).astype(np.float32)      # (N,1,H,W)
    elif layout == "NHW":
        return nhwc1[:,:,:,0].astype(np.float32)                      # (N,H,W)
    else:
        return nhwc1.astype(np.float32)                                # (N,H,W,1)

def infer_proba(sess: Optional["ort.InferenceSession"], x: np.ndarray, layout_cache: dict) -> float:
    if sess is None:
        raise RuntimeError("ONNX model not loaded.")
    if "io" not in layout_cache:
        layout_cache["io"] = _discover_io(sess)
    inp_name, out_name, layout = layout_cache["io"]
    batch = _pack_batch([x], layout)
    out = sess.run([out_name], {inp_name: batch})[0]
    return float(np.clip(np.ravel(out)[0], 0.0, 1.0))

def infer_proba_batch(sess: Optional["ort.InferenceSession"], xs: List[np.ndarray], layout_cache: dict) -> np.ndarray:
    if sess is None or not xs:
        return np.zeros((0,), dtype=np.float32)
    if "io" not in layout_cache:
        layout_cache["io"] = _discover_io(sess)
    inp_name, out_name, layout = layout_cache["io"]
    batch = _pack_batch(xs, layout)
    out = sess.run([out_name], {inp_name: batch})[0]
    return np.clip(np.ravel(out), 0.0, 1.0)

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="NDT Flaw Detection – Streamlit (No TF)", layout="wide")

st.sidebar.title("NDT Flaw UI")
page = st.sidebar.radio("Go to", [
    "About Project",
    "Overview", "Predict", "Threshold Tuning", "Validate (Stats)", "Explore Dataset", "Training Logs"
])

st.sidebar.markdown("---")
st.sidebar.caption("Paths")
st.sidebar.code(f"DATA_DIR  = {DATA_DIR.resolve()}\nMODEL_DIR = {MODEL_DIR.resolve()}")

manifest = load_manifest()
session, saved_thr = load_onnx_and_threshold()

# -------- About Project --------
if page == "About Project":
    st.title("About Project")
    st.markdown("""
    This build removes **TensorFlow** completely. For inference, it supports **ONNX** via **onnxruntime**.
    Put your model at `models/best.onnx` (or `final.onnx`). Pages that need a model will show a note if none is found.
    """)

# -------- Overview --------
elif page == "Overview":
    st.title("NDT Flaw/No-Flaw – Streamlit Frontend (No TF)")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("ONNX Runtime", "Available" if (ORT_OK and session is not None) else ("Installed" if ORT_OK else "Not installed"))
    with col2:
        st.metric("Model loaded", "Yes" if session is not None else "No")
    with col3:
        st.metric("Manifest", "Loaded" if manifest else "Missing")
    if manifest:
        st.subheader("Dataset Summary")
        st.write({
            "Total samples": manifest["total"],
            "Image shape": (manifest["height"], manifest["width"], manifest["channels"]),
            "Shards": len(manifest["items"])
        })
    else:
        st.info("Place your extracted dataset under `extracted_data/` and ensure `manifest.json` exists.")

# -------- Predict --------
elif page == "Predict":
    st.title("Predict – Single image or dataset shard (ONNX)")
    if session is None:
        if not ORT_OK:
            st.error("onnxruntime is not installed. Add it to requirements.txt to enable inference.")
        else:
            st.warning("No ONNX model found in models/. Add best.onnx or final.onnx.")
    else:
        st.caption(f"Loaded ONNX model. Default threshold = {saved_thr:.3f}")

    tab_file, tab_shard = st.tabs(["Upload Image", "Pick from Shard (.npy)"])
    layout_cache = {}

    with tab_file:
        up = st.file_uploader("Upload PNG/JPG image", type=["png","jpg","jpeg"], accept_multiple_files=False)
        thr = st.slider("Decision threshold (prob for 'FLAW')", 0.05, 0.95, float(saved_thr), 0.01)
        if up:
            pth = UPLOAD_DIR / up.name
            pth.write_bytes(up.read())
            img = Image.open(pth).convert("L")
            x = (np.array(img, dtype=np.float32) / 255.0)
            target_H = manifest["height"] if manifest else 480
            x = preprocess_strip(x, target_H)
            if session is None:
                st.info("Model not loaded. Showing preprocessed view only.")
                prob, pred = np.nan, "—"
            else:
                prob = float(infer_proba(session, x, layout_cache))
                pred = "FLAW" if prob >= thr else "NO FLAW"

            st.subheader("Result")
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Prediction", pred)
            with c2: st.metric("Probability", "—" if np.isnan(prob) else f"{prob:.4f}")
            with c3: st.metric("Threshold", f"{thr:.3f}")

            st.subheader("Preprocessed View")
            fig = plt.figure(figsize=(8,3))
            plt.imshow(x[...,0], cmap="gray", aspect="auto")
            plt.title(f"Processed strip – shape {x.shape}")
            plt.axis("off")
            st.pyplot(fig, clear_figure=True)

    with tab_shard:
        if not manifest:
            st.info("No manifest found. Put dataset shards under extracted_data/.")
        else:
            img_files, lab_files, counts = list_part_files(manifest)
            idx_all, y_all = build_global_index(lab_files, counts)
            st.write(f"Total rows across shards: **{idx_all.size:,}**")
            shard_idx = st.number_input("Global row index", min_value=0, max_value=int(idx_all.size-1), value=0, step=1)
            thr2 = st.slider("Decision threshold", 0.05, 0.95, float(saved_thr), 0.01, key="thr2")

            if st.button("Predict This Row"):
                offsets = np.cumsum([0] + counts)
                s = int(np.searchsorted(offsets, shard_idx, side="right") - 1)
                li = int(shard_idx - offsets[s])
                X = np.load(img_files[s], mmap_mode="r")  # (N,H,W,1) or (N,H,W)
                y = np.load(lab_files[s], mmap_mode="r")
                x = X[li].astype(np.float32)
                if x.ndim == 2:
                    x = x[..., None]
                H = manifest["height"]
                if x.shape[0] != H:
                    x = _pil_resize_gray(x, new_h=H)
                if USE_CROP:
                    L, R = scaled_band(x.shape[1])
                    x = x[:, L:R, :]
                if FORCE_WIDTH is not None:
                    x = _pil_resize_gray(x, new_w=FORCE_WIDTH)

                if session is None:
                    st.info("Model not loaded. Showing shard image only.")
                    prob, pred = np.nan, "—"
                else:
                    prob = float(infer_proba(session, x, layout_cache))
                    pred = "FLAW" if prob >= thr2 else "NO FLAW"
                y_true = int(y[li])

                st.subheader("Result")
                c1, c2, c3, c4 = st.columns(4)
                with c1: st.metric("Prediction", pred)
                with c2: st.metric("Probability", "—" if np.isnan(prob) else f"{prob:.4f}")
                with c3: st.metric("Threshold", f"{thr2:.3f}")
                with c4: st.metric("Ground truth", "FLAW" if y_true==1 else "NO FLAW")

                fig = plt.figure(figsize=(8,3))
                plt.imshow(x[...,0], cmap="gray", aspect="auto")
                plt.title(f"Shard view – file {Path(img_files[s]).name} row {li}")
                plt.axis("off")
                st.pyplot(fig, clear_figure=True)

# -------- Threshold Tuning --------
elif page == "Threshold Tuning":
    st.title("Threshold tuning (Validation set, ONNX)")
    if not manifest:
        st.warning("Need manifest to compute curves.")
    elif session is None:
        if not ORT_OK:
            st.error("onnxruntime is not installed. Add it to requirements.txt.")
        else:
            st.warning("No ONNX model found in models/. Add best.onnx or final.onnx.")
    else:
        img_files, lab_files, counts = list_part_files(manifest)
        idx_all, y_all = build_global_index(lab_files, counts)

        from sklearn.model_selection import train_test_split
        idx_tr, idx_va = train_test_split(np.arange(y_all.shape[0], dtype=np.int64),
                                          test_size=0.15, random_state=42, stratify=y_all)
        y_val = y_all[idx_va]

        BATCH = 64
        layout_cache = {}
        g2l = global_to_local_mapper(counts)

        probs, ys_acc = [], []
        for g in idx_va:
            s, li = g2l(int(g))
            X = np.load(img_files[s], mmap_mode="r")
            y = np.load(lab_files[s], mmap_mode="r")
            xi = X[li].astype(np.float32)
            if xi.ndim == 2:
                xi = xi[..., None]
            H = manifest["height"]
            if xi.shape[0] != H:
                xi = _pil_resize_gray(xi, new_h=H)
            if USE_CROP:
                L, R = scaled_band(xi.shape[1])
                xi = xi[:, L:R, :]
            if FORCE_WIDTH is not None:
                xi = _pil_resize_gray(xi, new_w=FORCE_WIDTH)
            p = infer_proba(session, xi, layout_cache)
            probs.append(p)
            ys_acc.append(int(y[li]))
        y_prob = np.asarray(probs, dtype=np.float32)
        y_val = np.asarray(ys_acc, dtype=np.uint8)

        st.write(f"Validation samples: **{y_val.shape[0]:,}**")
        thr = st.slider("Decision threshold", 0.05, 0.95, float(saved_thr), 0.01)

        from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_fscore_support, roc_curve
        y_pred = (y_prob >= thr).astype(int)
        cm = confusion_matrix(y_val, y_pred, labels=[0,1])
        prec, rec, f1, _ = precision_recall_fscore_support(y_val, y_pred, average="binary", zero_division=0)
        try:
            auc = roc_auc_score(y_val, y_prob)
        except Exception:
            auc = float("nan")

        c0, c1, c2, c3 = st.columns(4)
        c0.metric("AUC", f"{auc:.3f}" if np.isfinite(auc) else "—")
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
            xi = X[idx]
            if xi.ndim == 3 and xi.shape[-1] == 1:
                xi = xi[...,0]
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
            import pandas as pd
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

    st.caption("Tip: Place your training outputs in the models/ folder; this page will read them.")
