"""Image ingestion + quality assessment.

This stage is deliberately real even in Phase 1 — garbage input detection is
one of the most important defensive layers. Nothing here requires a neural
network: PIL + numpy is enough for blur, brightness, resolution, duplication.

Role tagging is currently heuristic (aspect ratio + brightness + hash bucket).
Phase 2 will replace `_guess_role` with a small CLIP-based classifier.
"""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from PIL import Image

from api.schemas import ImageQualityReport


MIN_RESOLUTION = (400, 400)
BLUR_THRESHOLD = 60.0          # Laplacian variance; lower = blurrier
DARK_THRESHOLD = 40.0           # mean luminance 0-255
BRIGHT_THRESHOLD = 220.0


@dataclass
class ImageRecord:
    filename: str
    raw_bytes: bytes
    pil: Image.Image
    array: np.ndarray            # HxWx3 uint8
    phash: str
    blur: float
    brightness: float
    report: ImageQualityReport = field(default=None)  # type: ignore


def _phash(pil: Image.Image) -> str:
    """Simple 16-byte perceptual hash via average-hash on 16x16 grayscale."""
    small = pil.convert("L").resize((16, 16), Image.BILINEAR)
    arr = np.asarray(small, dtype=np.int32)
    avg = arr.mean()
    bits = (arr > avg).flatten()
    packed = np.packbits(bits).tobytes()
    return hashlib.md5(packed).hexdigest()[:16]


def _laplacian_variance(gray: np.ndarray) -> float:
    """Cheap blur proxy: variance of a 3x3 Laplacian filter (hand-rolled)."""
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    h, w = gray.shape
    # pad and convolve with numpy (small kernel, fine for 400-2000px images).
    padded = np.pad(gray.astype(np.float32), 1, mode="edge")
    out = (
        k[0, 1] * padded[:-2, 1:-1]
        + k[1, 0] * padded[1:-1, :-2]
        + k[1, 1] * padded[1:-1, 1:-1]
        + k[1, 2] * padded[1:-1, 2:]
        + k[2, 1] * padded[2:, 1:-1]
    )
    return float(out.var())


def _guess_role(pil: Image.Image, array: np.ndarray, brightness: float) -> str:
    """Heuristic role router. Phase 2 swaps this for CLIP zero-shot.

    Rules-of-thumb used here:
      - Very tall portraits of shelves -> interior_shelf
      - Horizontal wide outdoor-ish (sky pixels) -> exterior
      - Low brightness + high color variance -> street_stall
      - Otherwise counter / unknown
    """
    h, w = array.shape[:2]
    aspect = w / max(h, 1)
    # "sky" = top 20% rows with high brightness + blue-ish
    top = array[: h // 5]
    top_bright = top.mean()
    top_blueish = top[..., 2].mean() - top[..., 0].mean()
    if top_bright > 150 and top_blueish > 10:
        return "exterior"
    if aspect < 0.85:       # portrait
        return "interior_shelf"
    if brightness < 70:
        return "street_stall"
    return "counter"


def _load(filename: str, raw: bytes) -> ImageRecord:
    pil = Image.open(io.BytesIO(raw)).convert("RGB")
    array = np.asarray(pil)
    gray = array.mean(axis=2)
    blur = _laplacian_variance(gray)
    brightness = float(gray.mean())
    return ImageRecord(
        filename=filename,
        raw_bytes=raw,
        pil=pil,
        array=array,
        phash=_phash(pil),
        blur=blur,
        brightness=brightness,
    )


def assess_images(files: Iterable[tuple[str, bytes]]) -> list[ImageRecord]:
    """Load each image, compute QA metrics, tag role, detect dupes.

    Returns a list of `ImageRecord`s with `.report` populated. The caller is
    responsible for deciding whether to proceed if too many fail.
    """
    records = [_load(name, raw) for name, raw in files]

    seen_hashes: dict[str, str] = {}
    for rec in records:
        notes: list[str] = []
        ok = True

        w, h = rec.pil.size
        resolution_ok = w >= MIN_RESOLUTION[0] and h >= MIN_RESOLUTION[1]
        if not resolution_ok:
            notes.append(f"low_resolution ({w}x{h})")
            ok = False

        if rec.blur < BLUR_THRESHOLD:
            notes.append(f"blurry (laplacian_var={rec.blur:.1f})")
            ok = False

        if rec.brightness < DARK_THRESHOLD:
            notes.append(f"too_dark (mean={rec.brightness:.1f})")
            # dark is a warning, not a hard fail — kirana shops are often dim
        if rec.brightness > BRIGHT_THRESHOLD:
            notes.append(f"overexposed (mean={rec.brightness:.1f})")
            ok = False

        duplicate_of = None
        if rec.phash in seen_hashes:
            duplicate_of = seen_hashes[rec.phash]
            notes.append(f"duplicate_of {duplicate_of}")
            ok = False
        else:
            seen_hashes[rec.phash] = rec.filename

        # Very cheap watermark heuristic: a strong horizontal mid-band with
        # text-like edge density. Real impl: OCR + known-stock-site DB.
        mid = rec.array[rec.array.shape[0] // 2 - 20 : rec.array.shape[0] // 2 + 20]
        watermark_suspect = bool(mid.std() > 70 and mid.mean() > 150)
        if watermark_suspect:
            notes.append("possible_watermark_or_overlay")

        # Phase 2b: use the enhanced classifier instead of the old heuristic.
        from pipelines.vision.classifiers import classify_role
        role, _role_evidence = classify_role(rec.pil, rec.array)

        rec.report = ImageQualityReport(
            filename=rec.filename,
            role=role,  # type: ignore[arg-type]
            blur_score=round(rec.blur, 2),
            brightness=round(rec.brightness, 2),
            resolution_ok=resolution_ok,
            duplicate_of=duplicate_of,
            watermark_suspect=watermark_suspect,
            quality_pass=ok,
            notes=notes,
        )

    return records
