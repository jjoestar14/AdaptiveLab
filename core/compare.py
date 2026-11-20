"""
Modul Pembanding Metode Segmentasi Biner (Compare).

Menyediakan fungsi untuk menjalankan dan membandingkan empat metode thresholding:
1. Global Thresholding (dengan ambang T manual, default 127)
2. Otsu's Thresholding (ambang global optimal otomatis)
3. Adaptive Mean Thresholding (rata-rata aritmatika lokal)
4. Adaptive Gaussian Thresholding (rata-rata berbobot Gaussian lokal)
"""

import time
from typing import Dict, Any
import cv2
import numpy as np

from core.threshold import validate_parameters


def compare_methods(
    gray: np.ndarray,
    global_thresh: int = 127,
    block_size: int = 21,
    c: int = 5
) -> Dict[str, Dict[str, Any]]:
    """
    Menjalankan 4 metode thresholding pada citra grayscale yang sama
    dan menghitung statistik dasar serta waktu eksekusi.

    Parameter:
        gray: Citra grayscale 2D uint8.
        global_thresh: Nilai ambang untuk thresholding global manual (0-255).
        block_size: Ukuran blok untuk metode adaptif (ganjil >= 3).
        c: Konstanta C untuk metode adaptif.

    Return:
        Dict hasil tiap metode dengan key:
        'global', 'otsu', 'adaptive_mean', 'adaptive_gaussian'.
    """
    if not isinstance(gray, np.ndarray) or gray.ndim != 2:
        raise ValueError("Input citra harus berupa array numpy 2D grayscale.")
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    validate_parameters(block_size, c)
    global_t_val = int(np.clip(global_thresh, 0, 255))

    total_pixels = float(gray.size)
    results = {}

    # 1. Global Thresholding (Manual T)
    t0 = time.perf_counter()
    _, bin_global = cv2.threshold(gray, global_t_val, 255, cv2.THRESH_BINARY)
    t_global_ms = (time.perf_counter() - t0) * 1000.0
    white_g = float(np.count_nonzero(bin_global == 255))
    results["global"] = {
        "name": "Global Thresholding",
        "description": f"Ambang global konstan T = {global_t_val}",
        "binary": bin_global,
        "threshold_value": global_t_val,
        "time_ms": round(t_global_ms, 3),
        "white_percent": round((white_g / total_pixels) * 100.0, 2),
        "black_percent": round(((total_pixels - white_g) / total_pixels) * 100.0, 2),
    }

    # 2. Otsu's Thresholding
    t0 = time.perf_counter()
    computed_otsu, bin_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t_otsu_ms = (time.perf_counter() - t0) * 1000.0
    white_o = float(np.count_nonzero(bin_otsu == 255))
    results["otsu"] = {
        "name": "Otsu's Thresholding",
        "description": f"Optimal bimodal clustering T = {int(round(computed_otsu))}",
        "binary": bin_otsu,
        "threshold_value": int(round(computed_otsu)),
        "time_ms": round(t_otsu_ms, 3),
        "white_percent": round((white_o / total_pixels) * 100.0, 2),
        "black_percent": round(((total_pixels - white_o) / total_pixels) * 100.0, 2),
    }

    # 3. Adaptive Mean Thresholding
    t0 = time.perf_counter()
    bin_mean = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, c
    )
    t_mean_ms = (time.perf_counter() - t0) * 1000.0
    white_m = float(np.count_nonzero(bin_mean == 255))
    results["adaptive_mean"] = {
        "name": "Adaptive Mean Thresholding",
        "description": f"Rata-rata aritmatika blok {block_size}x{block_size}, C = {c}",
        "binary": bin_mean,
        "threshold_value": f"Lokal (Mean - {c})",
        "time_ms": round(t_mean_ms, 3),
        "white_percent": round((white_m / total_pixels) * 100.0, 2),
        "black_percent": round(((total_pixels - white_m) / total_pixels) * 100.0, 2),
    }

    # 4. Adaptive Gaussian Thresholding (Metode Utama)
    t0 = time.perf_counter()
    bin_gauss = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c
    )
    t_gauss_ms = (time.perf_counter() - t0) * 1000.0
    white_ga = float(np.count_nonzero(bin_gauss == 255))
    results["adaptive_gaussian"] = {
        "name": "Adaptive Gaussian Thresholding",
        "description": f"Rata-rata berbobot Gaussian blok {block_size}x{block_size}, C = {c}",
        "binary": bin_gauss,
        "threshold_value": f"Lokal (Gaussian - {c})",
        "time_ms": round(t_gauss_ms, 3),
        "white_percent": round((white_ga / total_pixels) * 100.0, 2),
        "black_percent": round(((total_pixels - white_ga) / total_pixels) * 100.0, 2),
    }

    return results
