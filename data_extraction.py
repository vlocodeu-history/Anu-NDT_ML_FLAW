# data_extraction.py
import io
import json
import lzma
import pickle
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

# ----------------------------
# Config
# ----------------------------
DATA_DIR = Path("data")

# CHANGE THIS to a drive/folder with plenty of space if needed, e.g. Path("E:/ndt_out")
OUT_DIR = Path("extracted_data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

COMPRESSED_EXTS = {".xz", ".lzma"}

# Magic headers
XZ_MAGIC   = b"\xFD\x37\x7A\x58\x5A\x00"
NPY_MAGIC  = b"\x93NUMPY"      # .npy
ZIP_MAGIC  = b"PK\x03\x04"     # .npz (zip)
PICKLE_HDR = b"\x80"           # pickle protocol marker

# Geometry
IMG_H = 480
IMG_W = 7168
PIX_PER_IMG = IMG_H * IMG_W  # 3,440,640

# OUTPUT GEOMETRY (downsample width to save disk/ram)
TARGET_WIDTH = 1024                # set None to keep 7168 (not recommended)
BLOCK_PART   = 20                  # images per part file (~20MB each at 1024 width, float16)

# ----------------------------
# Helpers
# ----------------------------
def is_real_xz(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(6) == XZ_MAGIC
    except Exception:
        return False

def sniff(buf: bytes, n=8) -> str:
    return " ".join(f"{b:02X}" for b in buf[:n])

def load_decompressed(buf: bytes):
    """Try pickle / npy / npz (kept for completeness; your files are raw int16)."""
    if buf[:1] == PICKLE_HDR:
        return pickle.loads(buf)
    if buf.startswith(NPY_MAGIC):
        return np.load(io.BytesIO(buf), allow_pickle=True)
    if buf.startswith(ZIP_MAGIC):
        with np.load(io.BytesIO(buf), allow_pickle=True) as z:
            arrs = [z[k] for k in sorted(z.files)]
        if not arrs:
            raise ValueError("empty .npz archive")
        if len(arrs) == 1:
            return arrs[0]
        try:
            return np.stack(arrs, axis=0)
        except Exception:
            return np.array(arrs, dtype=object)
    raise ValueError("Unknown decompressed format (not pickle/.npy/.npz).")

def try_raw_dtype_and_bpp() -> list[tuple[str,int]]:
    """We’ll try int16, then uint16, then uint8."""
    return [("<i2", 2), ("<u2", 2), ("|u1", 1)]

def read_txt_labels(txt_path: Optional[Path], n_images: int) -> Tuple[np.ndarray, list]:
    labels, md = [], []
    if not txt_path or not txt_path.exists():
        return np.zeros(n_images, dtype=np.uint8), [{"flaw": 0} for _ in range(n_images)]
    with open(txt_path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    m = min(len(lines), n_images)
    for i in range(m):
        parts = lines[i].split()
        flaw = 0
        try:
            if parts:
                flaw = int(float(parts[0]))
        except Exception:
            flaw = 0
        labels.append(1 if flaw == 1 else 0)
        meta = {"flaw": flaw}
        keys = ["augmentation", "flaw_depth", "flaw_location", "flaw_size", "index_line", "flaw_type"]
        vals = parts[1:]
        for k_idx, k in enumerate(keys):
            if k_idx < len(vals):
                v = vals[k_idx]
                if k == "flaw_type":
                    meta[k] = v
                else:
                    try:
                        meta[k] = float(v)
                    except Exception:
                        meta[k] = v
        md.append(meta)
    if m < n_images:
        pad = n_images - m
        labels.extend([0]*pad)
        md.extend([{"flaw": 0} for _ in range(pad)])
    return np.array(labels, dtype=np.uint8), md

def find_txt_for_blob(blob: Path) -> Optional[Path]:
    txt = blob.with_suffix(".txt")
    if txt.exists():
        return txt
    alt = DATA_DIR / (blob.stem + ".txt")
    return alt if alt.exists() else None

def downsample_width_block(arr2d: np.ndarray, target_w: int) -> np.ndarray:
    """Downsample width by integer factor via block-averaging (fast, no deps)."""
    if target_w is None or target_w == IMG_W:
        return arr2d
    factor = IMG_W // target_w
    if factor * target_w != IMG_W:
        raise ValueError(f"TARGET_WIDTH {target_w} must divide {IMG_W} exactly.")
    # reshape to (H, target_w, factor) and average over factor
    return arr2d.reshape(IMG_H, target_w, factor).mean(axis=2)

# ----------------------------
# RAW streaming (two-pass) but writing parts, not one huge file
# ----------------------------
def raw_stream_stats(blob: Path, dtype: str, bpp: int, n_images_hint: Optional[int]) -> tuple[int, float, float]:
    """Pass 1: compute N, vmin, vmax."""
    bytes_per_img = PIX_PER_IMG * bpp
    n = 0
    vmin, vmax = np.inf, -np.inf

    with lzma.open(blob, "rb") as f:
        if n_images_hint is None:
            while True:
                buf = f.read(bytes_per_img)
                if not buf or len(buf) < bytes_per_img:
                    break
                arr = np.frombuffer(buf, dtype=dtype, count=PIX_PER_IMG).reshape(IMG_H, IMG_W)
                a_min, a_max = float(arr.min()), float(arr.max())
                vmin = min(vmin, a_min); vmax = max(vmax, a_max)
                n += 1
        else:
            for _ in range(n_images_hint):
                buf = f.read(bytes_per_img)
                if not buf or len(buf) < bytes_per_img:
                    break
                arr = np.frombuffer(buf, dtype=dtype, count=PIX_PER_IMG).reshape(IMG_H, IMG_W)
                a_min, a_max = float(arr.min()), float(arr.max())
                vmin = min(vmin, a_min); vmax = max(vmax, a_max)
                n += 1

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = 0.0, 1.0
    return n, vmin, vmax

def write_parts_raw(blob: Path, dtype: str, bpp: int, n: int, vmin: float, vmax: float,
                    lbls: np.ndarray, stem: str, manifest: list):
    """Pass 2: stream again, normalize, downsample, and save in small parts."""
    bytes_per_img = PIX_PER_IMG * bpp
    rng = (vmax - vmin) if (vmax - vmin) > 1e-8 else 1.0

    target_w = IMG_W if TARGET_WIDTH is None else TARGET_WIDTH

    with lzma.open(blob, "rb") as f:
        part_idx = 0
        i = 0
        while i < n:
            # allocate a small part buffer
            part_size = min(BLOCK_PART, n - i)
            Xp = np.empty((part_size, IMG_H, target_w, 1), dtype=np.float16)
            yp = lbls[i:i+part_size].astype(np.uint8, copy=False)

            for j in range(part_size):
                buf = f.read(bytes_per_img)
                if not buf or len(buf) < bytes_per_img:
                    raise ValueError(f"Unexpected EOF at image {i+j+1}/{n} for {blob.name}")
                arr = np.frombuffer(buf, dtype=dtype, count=PIX_PER_IMG).reshape(IMG_H, IMG_W)
                # normalize to [0,1] then downsample width
                arr = (arr.astype(np.float32) - vmin) / rng
                arr = downsample_width_block(arr, target_w)
                Xp[j, :, :, 0] = arr.astype(np.float16)

            # save this part
            img_path = OUT_DIR / f"{stem}_p{part_idx:03d}_images.f16.npy"
            lab_path = OUT_DIR / f"{stem}_p{part_idx:03d}_labels.u1.npy"
            np.save(img_path, Xp)
            np.save(lab_path, yp)

            manifest.append({
                "stem": stem,
                "part": part_idx,
                "images": str(img_path),
                "labels": str(lab_path),
                "count": int(part_size),
                "height": IMG_H,
                "width": target_w,
                "channels": 1,
            })
            i += part_size
            part_idx += 1

# ----------------------------
# Per-blob processing
# ----------------------------
def process_blob(blob: Path, manifest_parts: list) -> Optional[int]:
    print(f"Processing {blob.name}...")

    if blob.suffix.lower() == ".xz" and not is_real_xz(blob):
        print(f"  ✗ Skipping {blob.name}: not a real XZ (bad download).")
        return None

    txt_path = find_txt_for_blob(blob)
    n_from_txt = None
    if txt_path and txt_path.exists():
        with open(txt_path, "r") as f:
            n_from_txt = sum(1 for _ in f if _.strip())

    # Peek header
    with lzma.open(blob, "rb") as f:
        head = f.read(16)
    print(f"  - head (magic): {sniff(head)}")

    # We will follow RAW path (your files are raw int16). Still, keep structured path in case.
    looks_structured = head[:1] == PICKLE_HDR or head.startswith(NPY_MAGIC) or head.startswith(ZIP_MAGIC)

    stem = blob.stem

    if looks_structured:
        # Structured (rare in your set) → still write parts to keep memory low
        with lzma.open(blob, "rb") as f:
            raw = f.read()
        data = load_decompressed(raw)
        imgs = np.array(data)
        if imgs.ndim == 3:
            imgs = imgs[..., None]
        if imgs.ndim != 4:
            print(f"  ✗ {blob.name}: unexpected array shape {imgs.shape}. Skipping.")
            return None

        vmin, vmax = float(imgs.min()), float(imgs.max())
        rng = (vmax - vmin) if (vmax - vmin) > 1e-8 else 1.0
        n = imgs.shape[0]
        # labels
        if txt_path and txt_path.exists():
            labels, _ = read_txt_labels(txt_path, n)
        else:
            labels = np.zeros(n, dtype=np.uint8)

        # save in parts
        target_w = IMG_W if TARGET_WIDTH is None else TARGET_WIDTH
        part_idx = 0
        i = 0
        while i < n:
            part_size = min(BLOCK_PART, n - i)
            Xp = imgs[i:i+part_size].astype(np.float32)
            Xp = (Xp - vmin) / rng
            # downsample per sample
            Xp_ds = np.empty((part_size, IMG_H, target_w, 1), dtype=np.float16)
            for j in range(part_size):
                Xp_ds[j, :, :, 0] = downsample_width_block(Xp[j, :, :, 0], target_w).astype(np.float16)
            yp = labels[i:i+part_size].astype(np.uint8, copy=False)

            img_path = OUT_DIR / f"{stem}_p{part_idx:03d}_images.f16.npy"
            lab_path = OUT_DIR / f"{stem}_p{part_idx:03d}_labels.u1.npy"
            np.save(img_path, Xp_ds)
            np.save(lab_path, yp)

            manifest_parts.append({
                "stem": stem, "part": part_idx,
                "images": str(img_path), "labels": str(lab_path),
                "count": int(part_size), "height": IMG_H, "width": target_w, "channels": 1
            })
            i += part_size
            part_idx += 1
        print(f"  ✓ saved {part_idx} parts for {stem}")
        return n

    # RAW path (int16/uint16/uint8)
    for dtype, bpp in try_raw_dtype_and_bpp():
        try:
            n, vmin, vmax = raw_stream_stats(blob, dtype, bpp, n_from_txt)
            if n <= 0:
                continue
            print(f"  · raw dtype {dtype}, images={n}, vmin={vmin:.3f}, vmax={vmax:.3f}")
            # labels (based on n)
            if txt_path and txt_path.exists():
                labels, _ = read_txt_labels(txt_path, n)
            else:
                labels = np.zeros(n, dtype=np.uint8)
            # write small parts
            write_parts_raw(blob, dtype, bpp, n, vmin, vmax, labels, stem, manifest_parts)
            print(f"  ✓ saved parts for {stem}")
            return n
        except OSError as e:
            print(f"  ! OS error while processing {stem} as {dtype}: {e}")
        except Exception as e:
            print(f"  ! failed as raw {dtype}: {e}")

    print(f"  ✗ {blob.name}: cannot parse as raw/structured. Skipping.")
    return None

# ----------------------------
# Main
# ----------------------------
def extract_all():
    blobs: List[Path] = [p for p in sorted(DATA_DIR.glob("*")) if p.suffix.lower() in COMPRESSED_EXTS]
    print(f"Found {len(blobs)} compressed files in '{DATA_DIR}/'")
    if not blobs:
        raise FileNotFoundError("No .xz/.lzma files found under ./data")

    manifest = {"parts": [], "total": 0, "height": IMG_H,
                "width": (IMG_W if TARGET_WIDTH is None else TARGET_WIDTH),
                "channels": 1}

    for blob in blobs:
        n = process_blob(blob, manifest["parts"])
        if n:
            manifest["total"] += int(n)

    if not manifest["parts"]:
        raise RuntimeError("No valid datasets were parsed. Check your files and paths.")

    with open(OUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n----- Summary -----")
    print("Total images:", manifest["total"])
    print("Image shape :", (manifest["height"], manifest["width"], manifest["channels"]))
    print(f"Parts saved : {len(manifest['parts'])}")
    print(f"Manifest    : {OUT_DIR/'manifest.json'}")

if __name__ == "__main__":
    extract_all()
