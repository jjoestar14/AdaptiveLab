"""
Modul Pra-pemrosesan Citra (Preprocess).

Menyediakan fungsi konversi grayscale, filter Gaussian Blur, filter Median Blur,
dan Contrast Limited Adaptive Histogram Equalization (CLAHE).
"""

from typing import Dict, Any, Tuple
import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Mengonversi citra warna BGR atau BGRA ke grayscale uint8.

    Parameter:
        image: Citra input (2D atau 3D).

    Return:
        np.ndarray: Citra 2D grayscale uint8.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Citra harus berupa array numpy.")

    if image.ndim == 2:
        return image.astype(np.uint8)

    if image.ndim == 3:
        channels = image.shape[2]
        if channels == 1:
            return image[:, :, 0].astype(np.uint8)
        if channels == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if channels == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

    raise ValueError(f"Dimensi citra tidak didukung: shape {image.shape}")


def apply_gaussian_blur(gray: np.ndarray, ksize: int = 3, sigma: float = 0.0) -> np.ndarray:
    """
    Menerapkan Gaussian Blur untuk mereduksi noise Gaussian frekuensi tinggi.

    Parameter:
        gray: Citra grayscale 2D uint8.
        ksize: Ukuran kernel (harus ganjil dan >= 1). Jika 1, citra tidak diubah.
        sigma: Standar deviasi Gaussian. Jika 0, dihitung otomatis dari ksize.

    Return:
        np.ndarray: Citra hasil filtering.
    """
    if ksize < 1:
        raise ValueError(f"Ukuran kernel Gaussian blur harus >= 1, diterima {ksize}.")
    if ksize % 2 == 0:
        raise ValueError(f"Ukuran kernel Gaussian blur harus ganjil, diterima {ksize}.")
    if ksize == 1:
        return gray.copy()

    return cv2.GaussianBlur(gray, (ksize, ksize), sigmaX=float(sigma), sigmaY=float(sigma))


def apply_median_blur(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Menerapkan Median Filter untuk mereduksi noise impulsif (salt-and-pepper).

    Parameter:
        gray: Citra grayscale 2D uint8.
        ksize: Ukuran aperture ganjil (harus >= 3).

    Return:
        np.ndarray: Citra hasil filtering median.
    """
    if ksize < 3:
        raise ValueError(f"Ukuran kernel median blur harus ganjil dan >= 3, diterima {ksize}.")
    if ksize % 2 == 0:
        raise ValueError(f"Ukuran kernel median blur harus ganjil, diterima {ksize}.")

    return cv2.medianBlur(gray, ksize)


def apply_clahe(
    gray: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Menerapkan Contrast Limited Adaptive Histogram Equalization (CLAHE)
    untuk pemerataan kontras lokal adaptif.

    Parameter:
        gray: Citra grayscale 2D uint8.
        clip_limit: Batas pemotongan kontras (biasanya 1.0 - 5.0).
        tile_grid_size: Ukuran grid pembagian blok (misal (8, 8)).

    Return:
        np.ndarray: Citra hasil perataan kontras CLAHE.
    """
    if clip_limit <= 0:
        raise ValueError(f"clip_limit CLAHE harus > 0, diterima {clip_limit}.")
    gx, gy = tile_grid_size
    if gx < 2 or gy < 2:
        raise ValueError(f"tile_grid_size harus minimal (2, 2), diterima {tile_grid_size}.")

    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(gx), int(gy)))
    return clahe.apply(gray)


def process_preprocessing(
    image: np.ndarray,
    gaussian_blur: bool = False,
    gaussian_ksize: int = 3,
    gaussian_sigma: float = 0.0,
    median_blur: bool = False,
    median_ksize: int = 3,
    clahe: bool = False,
    clahe_clip_limit: float = 2.0,
    clahe_tile_size: int = 8
) -> Dict[str, np.ndarray]:
    """
    Menjalankan rantai pra-pemrosesan secara berurutan dan menyimpan tiap tahapan.
    Urutan: Grayscale -> Median Blur -> Gaussian Blur -> CLAHE.

    Return:
        Dict: {"grayscale": np.ndarray, "preprocessed": np.ndarray, "steps": dict}
    """
    gray = to_grayscale(image)
    current = gray.copy()
    steps = {"grayscale": gray}

    # 1. Median Blur (efektif menghilangkan noise salt-and-pepper sebelum penghalusan)
    if median_blur:
        current = apply_median_blur(current, ksize=median_ksize)
        steps["median_blur"] = current.copy()

    # 2. Gaussian Blur (menghaluskan variasi intensitas mikro)
    if gaussian_blur:
        current = apply_gaussian_blur(current, ksize=gaussian_ksize, sigma=gaussian_sigma)
        steps["gaussian_blur"] = current.copy()

    # 3. CLAHE (meningkatkan kontras lokal jika diperlukan)
    if clahe:
        current = apply_clahe(current, clip_limit=clahe_clip_limit, tile_grid_size=(clahe_tile_size, clahe_tile_size))
        steps["clahe"] = current.copy()

    return {
        "grayscale": gray,
        "preprocessed": current,
        "steps": steps
    }
