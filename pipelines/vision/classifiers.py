"""Image role classification and shop-type routing via image analysis.

No ML dependencies — uses color histograms, edge profiles, spatial zone
analysis, and text/brand detection heuristics. This is deliberately
explainable: every classification decision can be traced to a measurable
image property, which matters when an NBFC auditor asks "why did you
classify this as exterior?"

Phase 3+ can add CLIP on top as a refinement layer, but the structural
features here will always remain as cross-checks.
"""
from __future__ import annotations

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Image role classification
# ---------------------------------------------------------------------------

def classify_role(pil: Image.Image, array: np.ndarray) -> tuple[str, dict]:
    """Classify an image's role in the submission.

    Returns (role, evidence_dict) where evidence_dict contains the raw
    measurements that drove the decision — useful for debugging and for
    the explainability surface.

    Roles: interior_shelf, counter, exterior, street, street_stall, unknown
    """
    h, w, _ = array.shape
    aspect = w / max(h, 1)
    gray = array.mean(axis=2)
    brightness = float(gray.mean())

    # --- Zone analysis: split image into top/middle/bottom thirds ---
    third = h // 3
    top = array[:third]
    mid = array[third : 2 * third]
    bot = array[2 * third :]

    top_bright = float(top.mean())
    top_blue_excess = float(top[..., 2].mean() - top[..., 0].mean())
    mid_bright = float(mid.mean())
    bot_bright = float(bot.mean())

    # --- Edge density (gradient magnitude) ---
    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    edge_density = float((gx > 20).mean() + (gy > 20).mean()) / 2

    # --- Color variance as packaging/diversity signal ---
    color_var = float(array.reshape(-1, 3).astype(np.float32).var(axis=0).mean())

    # --- Saturation ---
    mx = array.max(axis=2).astype(np.float32)
    mn = array.min(axis=2).astype(np.float32)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1), 0.0)
    saturation = float(sat.mean())

    # --- Horizontal line density (shelves produce strong horizontal edges) ---
    gy_padded = np.pad(gy, ((0, 1), (0, 0)))
    horiz_lines = float((gy_padded > 15).mean(axis=1).mean())

    # --- Vertical regularity (product columns on shelves) ---
    gx_padded = np.pad(gx, ((0, 0), (0, 1)))
    vert_lines = float((gx_padded > 15).mean(axis=0).mean())

    evidence = {
        "brightness": round(brightness, 1),
        "top_bright": round(top_bright, 1),
        "top_blue_excess": round(top_blue_excess, 1),
        "edge_density": round(edge_density, 4),
        "color_variance": round(color_var, 1),
        "saturation": round(saturation, 3),
        "horiz_lines": round(horiz_lines, 4),
        "vert_lines": round(vert_lines, 4),
        "aspect": round(aspect, 2),
    }

    # --- Decision rules (ordered by specificity) ---

    # EXTERIOR: bright sky in the top zone + blue excess + wider than tall.
    if top_bright > 140 and top_blue_excess > 8 and aspect > 0.9:
        return "exterior", evidence

    # STREET: wide horizontal, somewhat bright, lower edge density than interior.
    if top_bright > 120 and top_blue_excess > 3 and aspect > 1.2 and edge_density < 0.15:
        return "street", evidence

    # INTERIOR SHELF: high horizontal line density (shelves), high edge density,
    # colorful (packaged goods). Portrait or square.
    if horiz_lines > 0.04 and edge_density > 0.10 and color_var > 500:
        return "interior_shelf", evidence

    # COUNTER: medium edge density, moderate brightness, tends to be squarish
    # with clutter in the center/bottom zone.
    if edge_density > 0.08 and 0.7 < aspect < 1.5 and mid_bright > 80:
        return "counter", evidence

    # STREET STALL: dim, cluttered, wide angle.
    if brightness < 90 and edge_density > 0.05:
        return "street_stall", evidence

    # INTERIOR (fallback for anything with decent edges in low-to-medium light).
    if edge_density > 0.06:
        if brightness < 120:
            return "interior_shelf", evidence
        return "counter", evidence

    return "unknown", evidence


# ---------------------------------------------------------------------------
# Shop-type routing
# ---------------------------------------------------------------------------

def classify_shop_type(
    records: list,  # list of ImageRecord — imported weakly to avoid circular deps
) -> tuple[str, dict]:
    """Classify overall shop type from the set of submitted images.

    Returns (shop_type, evidence_dict).

    Types: packaged_kirana, loose_goods, street_stall, organized_retail,
           exterior_only, unknown
    """
    usable = [r for r in records if r.report.quality_pass]
    if not usable:
        return "unknown", {"reason": "no_usable_images"}

    roles = {r.report.role for r in usable}

    # Exterior-only: no interior or counter views.
    if "interior_shelf" not in roles and "counter" not in roles:
        return "exterior_only", {"roles_seen": sorted(roles)}

    # Collect aggregate stats from usable interior/counter images.
    arrays = [r.array for r in usable if r.report.role in ("interior_shelf", "counter", "street_stall")]
    if not arrays:
        arrays = [r.array for r in usable]

    avg_edge = 0.0
    avg_sat = 0.0
    avg_bright = 0.0
    avg_color_var = 0.0
    for arr in arrays:
        gray = arr.mean(axis=2)
        gx = np.abs(np.diff(gray, axis=1))
        avg_edge += float((gx > 20).mean())
        mx = arr.max(axis=2).astype(np.float32)
        mn = arr.min(axis=2).astype(np.float32)
        sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1), 0.0)
        avg_sat += float(sat.mean())
        avg_bright += float(gray.mean())
        avg_color_var += float(arr.reshape(-1, 3).astype(np.float32).var(axis=0).mean())

    n = len(arrays)
    avg_edge /= n
    avg_sat /= n
    avg_bright /= n
    avg_color_var /= n

    evidence = {
        "avg_edge": round(avg_edge, 4),
        "avg_saturation": round(avg_sat, 3),
        "avg_brightness": round(avg_bright, 1),
        "avg_color_var": round(avg_color_var, 1),
        "roles_seen": sorted(roles),
    }

    # ORGANIZED RETAIL: very bright, very high edge density (clean shelving),
    # very high saturation (fluorescent lighting + branded packaging).
    if avg_bright > 150 and avg_edge > 0.20 and avg_sat > 0.35:
        return "organized_retail", evidence

    # STREET STALL: dim, low edge density, low saturation.
    if avg_bright < 80 and avg_edge < 0.10:
        return "street_stall", evidence

    # LOOSE GOODS: moderate brightness, lower edge density (no hard box edges),
    # lower color variance (earthy tones of grains/spices).
    if avg_color_var < 600 and avg_edge < 0.12 and avg_sat < 0.30:
        return "loose_goods", evidence

    # PACKAGED KIRANA: default for everything else (colorful, moderate-to-high
    # edge density = packaged goods on shelves).
    return "packaged_kirana", evidence


# ---------------------------------------------------------------------------
# Brand / POS detection
# ---------------------------------------------------------------------------

def detect_pos_signals(array: np.ndarray) -> dict:
    """Detect digital payment / POS signals via color-based heuristics.

    Paytm blue (#00BAF2), PhonePe purple (#5F259F), GPay green/blue, and
    Coca-Cola red (#F40009) are distinctive enough to detect via color
    histogram peaks. This is a lightweight proxy for "does this shop accept
    digital payments?"

    Returns a dict with detected brand signals and a composite POS score.
    """
    h, w, _ = array.shape
    pixels = array.reshape(-1, 3).astype(np.float32)

    # Convert to HSV-like signals (cheap, no OpenCV needed).
    r, g, b = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    chroma = mx - mn

    detections: dict[str, bool] = {}
    pos_score = 0.0

    # PAYTM BLUE: R<100, G>150, B>200 (light blue region)
    paytm_mask = (r < 100) & (g > 150) & (b > 200) & (chroma > 50)
    paytm_ratio = float(paytm_mask.mean())
    detections["paytm_blue"] = paytm_ratio > 0.003
    if detections["paytm_blue"]:
        pos_score += 0.35

    # PHONEPE PURPLE: R~90-120, G<80, B>130
    phonepe_mask = (r > 60) & (r < 140) & (g < 80) & (b > 120) & (chroma > 40)
    phonepe_ratio = float(phonepe_mask.mean())
    detections["phonepe_purple"] = phonepe_ratio > 0.002
    if detections["phonepe_purple"]:
        pos_score += 0.30

    # GPAY: multicolor, hard to detect; skip for now.

    # COCA-COLA RED: very distinct bright red (R>200, G<80, B<80).
    coke_mask = (r > 200) & (g < 80) & (b < 80)
    coke_ratio = float(coke_mask.mean())
    detections["cocacola_red"] = coke_ratio > 0.005
    # Coca-Cola presence implies brand distributor relationship.
    if detections["cocacola_red"]:
        pos_score += 0.10

    # QR CODE heuristic: a region with high local B/W contrast in a small
    # rectangular patch. Approximate by looking for tight clusters of black
    # pixels next to white pixels in the image.
    gray = array.mean(axis=2)
    # Look at 50x50 blocks for high binary variance.
    block_sz = 50
    qr_found = False
    for y in range(0, h - block_sz, block_sz // 2):
        for x in range(0, w - block_sz, block_sz // 2):
            block = gray[y : y + block_sz, x : x + block_sz]
            if block.std() > 70 and block.mean() > 80 and block.mean() < 180:
                # High local contrast in a small patch — possible QR.
                dark = (block < 80).mean()
                light = (block > 180).mean()
                if dark > 0.2 and light > 0.2:
                    qr_found = True
                    break
        if qr_found:
            break

    detections["qr_code_region"] = qr_found
    if qr_found:
        pos_score += 0.25

    return {
        "detections": detections,
        "pos_score": round(min(pos_score, 1.0), 3),
    }
