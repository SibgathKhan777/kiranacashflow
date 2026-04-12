"""Tests for pipelines.vision.extractor."""
from __future__ import annotations

from pipelines.ingest import assess_images
from pipelines.vision import extract_vision_features


def test_clean_inputs_produce_reasonable_features(clean_metro_files):
    records = assess_images(clean_metro_files)
    vf = extract_vision_features(records)

    assert vf.shop_type in {"packaged_kirana", "street_stall", "exterior_only"}
    assert 0.0 <= vf.shelf_density_index <= 1.0
    assert 0.0 <= vf.sku_diversity <= 1.0
    assert vf.inventory_value_est > 0
    assert vf.store_size_sqft > 0
    assert 0.0 <= vf.coverage_score <= 1.0
    assert 0.0 < vf.extraction_confidence <= 1.0


def test_empty_inputs_produce_zero_confidence(make_image):
    # All duplicates -> only one survives QA -> still few usable, but the
    # extractor should still gracefully produce low-confidence features.
    dupe = make_image(42, style="dim")
    records = assess_images([("a.png", dupe), ("b.png", dupe), ("c.png", dupe)])
    vf = extract_vision_features(records)
    assert vf.extraction_confidence < 0.5


def test_no_usable_images_returns_unknown(make_image):
    # Force everything to be tiny (below min resolution).
    import io
    from PIL import Image
    import numpy as np

    tiny_bytes = io.BytesIO()
    Image.fromarray(np.zeros((50, 50, 3), dtype=np.uint8)).save(tiny_bytes, format="PNG")
    data = tiny_bytes.getvalue()
    records = assess_images([("tiny.png", data)])
    vf = extract_vision_features(records)
    assert vf.shop_type == "unknown"
    assert vf.extraction_confidence < 0.2
