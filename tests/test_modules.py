"""
Unit Test untuk Modul Pra-pemrosesan, Pasca-pemrosesan, Metrik, dan Pembanding.
"""

import pytest
import numpy as np
import cv2

from core.preprocess import (
    to_grayscale,
    apply_gaussian_blur,
    apply_median_blur,
    apply_clahe,
    process_preprocessing
)
from core.postprocess import (
    apply_morphology,
    remove_small_components,
    process_postprocessing
)
from core.metrics import (
    evaluate_segmentation,
    ensure_ground_truth_compatibility
)
from core.compare import compare_methods
from core.experiment import run_parameter_grid_experiment, generate_experiment_csv, generate_experiment_zip


def test_preprocess_to_grayscale():
    bgr = np.zeros((50, 60, 3), dtype=np.uint8)
    bgr[:, :] = [100, 150, 200]
    gray = to_grayscale(bgr)
    assert gray.ndim == 2
    assert gray.shape == (50, 60)
    assert gray.dtype == np.uint8

    # Input sudah grayscale
    assert to_grayscale(gray).shape == (50, 60)


def test_preprocess_filters():
    img = np.random.randint(0, 256, (80, 80), dtype=np.uint8)

    # Gaussian Blur
    g_blur = apply_gaussian_blur(img, ksize=5)
    assert g_blur.shape == (80, 80)
    with pytest.raises(ValueError):
        apply_gaussian_blur(img, ksize=4)  # Genap harus ditolak

    # Median Blur
    m_blur = apply_median_blur(img, ksize=3)
    assert m_blur.shape == (80, 80)
    with pytest.raises(ValueError):
        apply_median_blur(img, ksize=2)  # Genap harus ditolak

    # CLAHE
    c_img = apply_clahe(img, clip_limit=2.5, tile_grid_size=(4, 4))
    assert c_img.shape == (80, 80)


def test_postprocess_morphology():
    bin_img = np.zeros((100, 100), dtype=np.uint8)
    bin_img[40:60, 40:60] = 255

    opened = apply_morphology(bin_img, operation="opening", ksize=3)
    closed = apply_morphology(bin_img, operation="closing", ksize=3)
    eroded = apply_morphology(bin_img, operation="erosion", ksize=3)
    dilated = apply_morphology(bin_img, operation="dilation", ksize=3)

    assert np.count_nonzero(eroded) < np.count_nonzero(bin_img)
    assert np.count_nonzero(dilated) > np.count_nonzero(bin_img)


def test_postprocess_remove_small():
    bin_img = np.zeros((100, 100), dtype=np.uint8)
    # Objek besar
    bin_img[20:60, 20:60] = 255
    # Noise kecil 2x2 = 4 piksel
    bin_img[80:82, 80:82] = 255

    cleaned = remove_small_components(bin_img, min_area=10, target="white")
    assert np.all(cleaned[80:82, 80:82] == 0)  # Noise putih hilang
    assert np.all(cleaned[20:60, 20:60] == 255)  # Objek utama tetap ada


def test_metrics_identical_images():
    gt = np.zeros((50, 50), dtype=np.uint8)
    gt[10:30, 10:30] = 255
    pred = gt.copy()

    res = evaluate_segmentation(pred, gt, foreground_is_black=False)
    m = res["metrics"]
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1_score"] == 1.0
    assert m["accuracy"] == 1.0
    assert m["mse"] == 0.0


def test_metrics_size_mismatch_warning():
    pred = np.zeros((100, 100), dtype=np.uint8)
    gt = np.zeros((50, 50), dtype=np.uint8)

    res = evaluate_segmentation(pred, gt)
    assert res["was_resized"] is True
    assert "diselaraskan otomatis" in res["warning"]
    assert res["diff_map_rgb"].shape == (100, 100, 3)


def test_compare_methods():
    gray = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    res = compare_methods(gray, global_thresh=120, block_size=15, c=3)
    assert "global" in res
    assert "otsu" in res
    assert "adaptive_mean" in res
    assert "adaptive_gaussian" in res
    for k in ["global", "otsu", "adaptive_mean", "adaptive_gaussian"]:
        assert res[k]["binary"].shape == (100, 100)
        assert res[k]["time_ms"] >= 0


def test_experiment_grid():
    gray = np.random.randint(0, 256, (80, 80), dtype=np.uint8)
    exp = run_parameter_grid_experiment(
        gray,
        block_sizes=[11, 21],
        c_values=[2, 5],
        engine="opencv"
    )
    assert exp["total_combinations"] == 4
    assert len(exp["results"]) == 4

    # Test CSV dan ZIP
    csv_str = generate_experiment_csv(exp["results"])
    assert "block_size,c" in csv_str
    zip_bytes = generate_experiment_zip(exp["results"])
    assert len(zip_bytes) > 0
