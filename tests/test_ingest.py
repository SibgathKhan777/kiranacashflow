"""Tests for pipelines.ingest.quality."""
from __future__ import annotations

from pipelines.ingest import assess_images


def test_clean_inputs_all_pass(clean_metro_files):
    records = assess_images(clean_metro_files)
    assert len(records) == 3
    assert all(r.report.quality_pass for r in records), [r.report.notes for r in records]
    # Each record should have been assigned a role.
    assert all(r.report.role != "unknown" for r in records)


def test_duplicate_detection(poor_input_files):
    records = assess_images(poor_input_files)
    dupes = [r for r in records if r.report.duplicate_of]
    assert len(dupes) == 1
    assert dupes[0].report.quality_pass is False
    assert "duplicate_of" in " ".join(dupes[0].report.notes)


def test_blurry_image_fails(make_image):
    files = [("flat.png", make_image(0, style="blurry"))]
    records = assess_images(files)
    assert records[0].report.quality_pass is False
    assert any("blurry" in n for n in records[0].report.notes)


def test_bright_exterior_detected_as_exterior(make_image):
    files = [("out.png", make_image(3, style="bright_exterior"))]
    records = assess_images(files)
    assert records[0].report.role == "exterior"


def test_dark_image_is_warning_not_failure(make_image):
    """Dark is a note, not a hard failure — kirana interiors are often dim."""
    files = [("dark.png", make_image(10, style="dim"))]
    records = assess_images(files)
    # Synthetic dim images are noisy enough to be non-blurry; darkness alone
    # should not flip quality_pass off.
    rec = records[0]
    if any("too_dark" in n for n in rec.report.notes):
        # dark alone is a warning; other notes (blur/res) may fail it, but
        # darkness in isolation must not.
        other_failures = [n for n in rec.report.notes if "too_dark" not in n]
        assert rec.report.quality_pass == (len(other_failures) == 0)
