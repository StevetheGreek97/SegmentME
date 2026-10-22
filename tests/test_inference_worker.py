"""Inference worker helpers (core/inference_worker.py) that need no model."""
import os

import cv2
import numpy as np
import pytest
from PIL import Image

from core import inference_worker as iw


# --------------------------------------------------------------------------
# HiReS settings
# --------------------------------------------------------------------------
def test_hires_defaults():
    assert iw._hires_settings({}) == {"chunk_size": (1024, 1024), "overlap": 150, "imgsz": 1024,
                                      "edge_threshold": 0.01, "iou_thresh": 0.7}


@pytest.mark.parametrize("params", [
    {"chunk_size": (16, 16)},
    {"chunk_size": (512, 512), "overlap": 512},
    {"overlap": -1},
    {"edge_threshold": 0.5},
    {"iou_thresh": 1.5},
])
def test_hires_rejects_bad_settings(params):
    with pytest.raises(ValueError):
        iw._hires_settings({"hires": params})


# --------------------------------------------------------------------------
# Reading the scan for tiling: same on every OS, any path
# --------------------------------------------------------------------------
@pytest.fixture
def greek_dir(tmp_path):
    d = tmp_path / "Σταύρος" / "Έργο"
    d.mkdir(parents=True)
    return d


def _scan(h=300, w=400):
    rng = np.random.default_rng(1)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


def test_reads_paths_with_non_english_letters(greek_dir):
    rgb = _scan()
    path = greek_dir / "σάρωση.tif"
    Image.fromarray(rgb).save(path)
    img = iw._read_image_unchanged(str(path))
    assert np.array_equal(img, rgb[:, :, ::-1])          # BGR, like cv2.imread


def test_tiff_with_a_thumbnail_page_reads_the_first_page(greek_dir):
    rgb = _scan()
    path = greek_dir / "with_thumb.tif"
    Image.fromarray(rgb).save(path, save_all=True, append_images=[Image.fromarray(rgb[:50, :50])])
    img = iw._read_image_unchanged(str(path))
    assert img.shape == (300, 400, 3)


def test_grayscale_and_alpha_kept_as_stored(greek_dir):
    gray = _scan()[:, :, 0]
    Image.fromarray(gray).save(greek_dir / "g.png")
    assert iw._read_image_unchanged(str(greek_dir / "g.png")).shape == (300, 400)
    rgba = np.dstack([_scan(), np.full((300, 400), 128, np.uint8)])
    Image.fromarray(rgba).save(greek_dir / "a.png")
    assert iw._read_image_unchanged(str(greek_dir / "a.png")).shape == (300, 400, 4)


def test_unreadable_image_says_so(greek_dir):
    bad = greek_dir / "broken.tif"
    bad.write_bytes(b"II*\x00garbage")
    with pytest.raises(ValueError, match="could not read"):
        iw._read_image_unchanged(str(bad))


def test_tiles_match_hires_own_chunker(tmp_path):
    chunker = pytest.importorskip("hires.processing.chunker")
    path = tmp_path / "scan.png"
    cv2.imwrite(str(path), _scan(2500, 2300))
    a, b = tmp_path / "a", tmp_path / "b"
    chunker.ImageChunker(str(path)).slice(save_folder=str(a), chunk_size=(1024, 1024), overlap=150)
    chunker.ImageChunker(str(path))._chunk_and_save(
        iw._read_image_unchanged(str(path)), str(b), "scan", (1024, 1024), 150)
    names = sorted(os.listdir(a))
    assert names and names == sorted(os.listdir(b))
    for n in names:
        assert (a / n).read_bytes() == (b / n).read_bytes()


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def test_mask_to_polygon_keeps_the_largest_piece():
    seg = np.zeros((100, 100), np.uint8)
    seg[10:60, 10:60] = 1
    seg[80:83, 80:83] = 1
    poly = np.array(iw._mask_to_polygon(seg))
    assert poly[:, 0].min() == 10 and poly[:, 0].max() == 59
    assert iw._mask_to_polygon(np.zeros((10, 10), np.uint8)) is None
