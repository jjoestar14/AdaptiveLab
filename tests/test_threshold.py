"""
Unit Test untuk Modul Thresholding Adaptif Gaussian (core/threshold.py).
"""

import pytest
import numpy as np
import cv2

from core.threshold import (
    validate_parameters,
    build_gaussian_kernel_2d,
    adaptive_gaussian_opencv,
    adaptive_gaussian_manual,
    get_threshold_map
)


def test_validate_parameters_valid():
    """Memastikan parameter valid tidak melempar eksepsi."""
    validate_parameters(3, 0)
    validate_parameters(11, 5)
    validate_parameters(51, -10)
    validate_parameters(101, 25)


def test_validate_parameters_invalid_type():
    """Memastikan tipe data non-integer melempar TypeError."""
    with pytest.raises(TypeError):
        validate_parameters("11", 5)  # type: ignore

    with pytest.raises(TypeError):
        validate_parameters(11.0, 5)  # type: ignore

    with pytest.raises(TypeError):
        validate_parameters(11, "5")  # type: ignore

    with pytest.raises(TypeError):
        validate_parameters(11, 5.5)  # type: ignore


def test_validate_parameters_invalid_values():
    """Memastikan block size genap atau < 3 melempar ValueError."""
    with pytest.raises(ValueError, match="harus >= 3"):
        validate_parameters(1, 2)

    with pytest.raises(ValueError, match="harus >= 3"):
        validate_parameters(-5, 2)

    with pytest.raises(ValueError, match="harus ganjil"):
        validate_parameters(4, 2)

    with pytest.raises(ValueError, match="harus ganjil"):
        validate_parameters(20, 5)


def test_build_gaussian_kernel_2d():
    """Memastikan struktur kernel Gaussian 2D benar dan ternormalisasi."""
    for bs in [3, 7, 15, 31]:
        kernel = build_gaussian_kernel_2d(bs)
        assert kernel.shape == (bs, bs)
        # Jumlah bobot harus sangat mendekati 1.0
        assert np.isclose(np.sum(kernel), 1.0, atol=1e-6)
        # Kernel harus simetris horizontal dan vertikal
        assert np.allclose(kernel, kernel.T, atol=1e-6)
        assert np.allclose(kernel, np.fliplr(kernel), atol=1e-6)
        # Nilai tengah harus merupakan nilai tertinggi (puncak Gaussian)
        center = bs // 2
        assert kernel[center, center] == np.max(kernel)


@pytest.mark.parametrize("block_size, c", [
    (7, 2),
    (11, 4),
    (21, 5),
    (31, 10),
    (51, -3),
])
def test_manual_vs_opencv_difference_under_one_percent(block_size, c):
    """
    Syarat Utama 3.e: Membuktikan hasil manual dan OpenCV hampir identik
    (selisih piksel < 1 persen) pada beberapa ukuran citra dan kombinasi parameter.
    """
    np.random.seed(42)

    # 1. Citra berpola gradien bergradasi + noise (mensimulasikan dokumen berbayang)
    h, w = 180, 240
    y_grad = np.linspace(50, 220, h, dtype=np.float64)[:, None]
    x_grad = np.linspace(20, 180, w, dtype=np.float64)[None, :]
    base_grad = (y_grad + x_grad) / 2.0
    noise = np.random.normal(0, 15, (h, w))
    test_img = np.clip(base_grad + noise, 0, 255).astype(np.uint8)

    # Tambahkan beberapa goresan teks tiruan (garis gelap)
    test_img[30:35, 20:200] = 30
    test_img[60:65, 30:220] = 35
    test_img[90:95, 10:190] = 25
    test_img[120:125, 40:210] = 40

    res_cv = adaptive_gaussian_opencv(test_img, block_size, c)
    res_manual = adaptive_gaussian_manual(test_img, block_size, c)

    assert res_cv.shape == res_manual.shape
    assert res_cv.dtype == np.uint8
    assert res_manual.dtype == np.uint8

    mismatches = np.count_nonzero(res_cv != res_manual)
    total_pixels = test_img.size
    diff_percentage = (mismatches / total_pixels) * 100.0

    # Syarat: selisih < 1.0 persen
    print(f"BS={block_size}, C={c} -> Mismatch: {mismatches}/{total_pixels} ({diff_percentage:.4f}%)")
    assert diff_percentage < 1.0, f"Selisih piksel {diff_percentage:.4f}% melebihi toleransi 1.0%!"


def test_get_threshold_map():
    """Memastikan get_threshold_map menghasilkan citra uint8 valid berdimensi sama."""
    img = np.full((100, 120), 150, dtype=np.uint8)
    t_map = get_threshold_map(img, block_size=15, c=5)
    assert t_map.shape == (100, 120)
    assert t_map.dtype == np.uint8
    # Nilai ambang rata-rata harus 150 - 5 = 145
    assert np.all(t_map == 145)
