"""
Modul Thresholding Adaptif Gaussian (Core).

Menyediakan implementasi Adaptive Gaussian Thresholding baik berbasis OpenCV
maupun implementasi manual (scratch kernel Gaussian), pembuatan peta ambang lokal,
serta validasi parameter input.
"""

from typing import Union
import cv2
import numpy as np


def validate_parameters(block_size: int, c: int) -> None:
    """
    Memvalidasi parameter block_size dan konstanta C.

    Parameter:
        block_size: Ukuran jendela/blok lingkungan lokal (harus ganjil dan >= 3).
        c: Nilai konstanta pengurang ambang lokal (harus bertipe integer).

    Raises:
        TypeError: Jika block_size atau c bukan bilangan bulat (int).
        ValueError: Jika block_size < 3 atau block_size bernilai genap.
    """
    if not isinstance(block_size, (int, np.integer)):
        raise TypeError(f"Block size harus bertipe integer, bukan {type(block_size).__name__}.")

    if not isinstance(c, (int, np.integer)):
        raise TypeError(f"Konstanta C harus bertipe integer, bukan {type(c).__name__}.")

    if block_size < 3:
        raise ValueError(f"Block size harus >= 3, diterima {block_size}.")

    if block_size % 2 == 0:
        raise ValueError(f"Block size harus ganjil, diterima {block_size} (genap).")


def build_gaussian_kernel_2d(block_size: int) -> np.ndarray:
    """
    Membangun kernel Gaussian 2D kustom sesuai spesifikasi standar OpenCV:
    sigma = 0.3 * ((block_size - 1) * 0.5 - 1) + 0.8
    dan ternormalisasi (jumlah bobot = 1.0).

    Parameter:
        block_size: Ukuran kernel sisi ganjil (N x N).

    Return:
        np.ndarray: Kernel 2D float64 berukuran (block_size, block_size).
    """
    sigma = 0.3 * ((block_size - 1) * 0.5 - 1.0) + 0.8
    half = (block_size - 1) / 2.0
    x = np.arange(-half, half + 1.0, dtype=np.float64)
    # Formula 1D Gaussian
    kernel_1d = np.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d /= np.sum(kernel_1d)

    # 2D Kernel outer product
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    kernel_2d /= np.sum(kernel_2d)
    return kernel_2d


def adaptive_gaussian_opencv(gray: np.ndarray, block_size: int, c: int) -> np.ndarray:
    """
    Segmentasi biner adaptif Gaussian menggunakan cv2.adaptiveThreshold bawaan OpenCV.

    Parameter:
        gray: Citra grayscale 2D bertipe np.uint8.
        block_size: Ukuran jendela lokal ganjil >= 3.
        c: Konstanta pengurang ambang lokal.

    Return:
        np.ndarray: Citra biner (0 atau 255) bertipe np.uint8.
    """
    if not isinstance(gray, np.ndarray) or gray.ndim != 2:
        raise ValueError("Input citra harus berupa array numpy 2D grayscale.")
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    validate_parameters(block_size, c)

    return cv2.adaptiveThreshold(
        gray,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=int(block_size),
        C=int(c)
    )


def adaptive_gaussian_manual(gray: np.ndarray, block_size: int, c: int) -> np.ndarray:
    """
    Implementasi manual Adaptive Gaussian Thresholding:
    1. Bangun kernel Gaussian 2D (sigma = 0.3*((N-1)*0.5 - 1) + 0.8).
    2. Hitung rata-rata berbobot Gaussian lokal menggunakan konvolusi (BORDER_REPLICATE).
    3. Hitung ambang batas lokal T(x, y) = mean_gaussian(x, y) - C.
    4. Terapkan aturan biner: jika gray(x, y) > T(x, y) maka 255, selain itu 0.

    Parameter:
        gray: Citra grayscale 2D bertipe np.uint8.
        block_size: Ukuran jendela lokal ganjil >= 3.
        c: Konstanta pengurang ambang lokal.

    Return:
        np.ndarray: Citra biner (0 atau 255) bertipe np.uint8.
    """
    if not isinstance(gray, np.ndarray) or gray.ndim != 2:
        raise ValueError("Input citra harus berupa array numpy 2D grayscale.")
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    validate_parameters(block_size, c)

    kernel_2d = build_gaussian_kernel_2d(block_size)

    # Konvolusi 2D dengan border replicate | isolated seperti implementasi C++ OpenCV
    mean_map = cv2.filter2D(
        gray,
        ddepth=cv2.CV_8U,
        kernel=kernel_2d,
        borderType=cv2.BORDER_REPLICATE | cv2.BORDER_ISOLATED
    )

    # Ambang batas lokal T(x,y) = mean(x,y) - C
    thresh_map = mean_map.astype(np.int32) - int(c)

    # Binerisasi: piksel > T bernilai 255, lainnya 0
    binary = np.where(gray.astype(np.int32) > thresh_map, 255, 0).astype(np.uint8)
    return binary


def get_threshold_map(gray: np.ndarray, block_size: int, c: int = 0) -> np.ndarray:
    """
    Mendapatkan representasi visual peta ambang lokal T(x, y) = mean_gaussian(x, y) - C.
    Nilai diklip ke rentang [0, 255] dan dikembalikan sebagai uint8 untuk visualisasi.

    Parameter:
        gray: Citra grayscale 2D bertipe np.uint8.
        block_size: Ukuran jendela lokal ganjil >= 3.
        c: Konstanta pengurang ambang lokal (default: 0).

    Return:
        np.ndarray: Peta ambang lokal bertipe np.uint8.
    """
    if not isinstance(gray, np.ndarray) or gray.ndim != 2:
        raise ValueError("Input citra harus berupa array numpy 2D grayscale.")
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    validate_parameters(block_size, c)

    kernel_2d = build_gaussian_kernel_2d(block_size)
    mean_map = cv2.filter2D(
        gray,
        ddepth=cv2.CV_8U,
        kernel=kernel_2d,
        borderType=cv2.BORDER_REPLICATE | cv2.BORDER_ISOLATED
    )
    thresh_map = mean_map.astype(np.int32) - int(c)
    clamped_map = np.clip(thresh_map, 0, 255).astype(np.uint8)
    return clamped_map
