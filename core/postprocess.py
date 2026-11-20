"""
Modul Pasca-pemrosesan Citra Biner (Postprocess).

Menyediakan operasi morfologi matematika (opening, closing, erosion, dilation)
dan penyaringan komponen terhubung kecil (remove small connected components).
"""

from typing import Dict, Any, Literal
import cv2
import numpy as np


def get_structuring_element(shape: str = "rect", ksize: int = 3) -> np.ndarray:
    """
    Membuat elemen penstruktur (structuring element) untuk operasi morfologi.

    Parameter:
        shape: Bentuk kernel ('rect', 'ellipse', 'cross').
        ksize: Ukuran kernel (integer >= 1).

    Return:
        np.ndarray: Kernel structuring element.
    """
    if ksize < 1:
        raise ValueError(f"Ukuran kernel morfologi harus >= 1, diterima {ksize}.")

    shape_map = {
        "rect": cv2.MORPH_RECT,
        "ellipse": cv2.MORPH_ELLIPSE,
        "cross": cv2.MORPH_CROSS
    }
    cv_shape = shape_map.get(shape.lower(), cv2.MORPH_RECT)
    return cv2.getStructuringElement(cv_shape, (ksize, ksize))


def apply_morphology(
    binary: np.ndarray,
    operation: str = "none",
    ksize: int = 3,
    iterations: int = 1,
    shape: str = "rect"
) -> np.ndarray:
    """
    Menerapkan operasi morfologi pada citra biner.

    Parameter:
        binary: Citra biner 2D uint8 (0 atau 255).
        operation: Jenis operasi ('none', 'opening', 'closing', 'erosion', 'dilation').
        ksize: Ukuran kernel ganjil (>= 1).
        iterations: Jumlah iterasi pengulangan (>= 1).
        shape: Bentuk kernel ('rect', 'ellipse', 'cross').

    Return:
        np.ndarray: Citra biner hasil operasi morfologi.
    """
    op = operation.lower().strip()
    if op in ("none", "", "tidak ada"):
        return binary.copy()

    if iterations < 1:
        raise ValueError(f"Iterasi morfologi harus >= 1, diterima {iterations}.")

    kernel = get_structuring_element(shape=shape, ksize=ksize)

    if op == "opening":
        return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=iterations)
    elif op == "closing":
        return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    elif op in ("erosion", "erosi"):
        return cv2.erode(binary, kernel, iterations=iterations)
    elif op in ("dilation", "dilasi"):
        return cv2.dilate(binary, kernel, iterations=iterations)
    else:
        raise ValueError(f"Operasi morfologi tidak dikenal: '{operation}'. Pilih: opening, closing, erosion, dilation.")


def remove_small_components(
    binary: np.ndarray,
    min_area: int = 10,
    target: Literal["black", "white", "both"] = "both"
) -> np.ndarray:
    """
    Menghapus komponen terhubung kecil yang memiliki luas piksel < min_area.

    Parameter:
        binary: Citra biner 2D uint8 (0 atau 255).
        min_area: Ambang batas luas minimal (piksel).
        target: Objek yang dibersihkan ('black' untuk bintik hitam, 'white' untuk bintik putih, 'both' untuk keduanya).

    Return:
        np.ndarray: Citra biner bersih setelah komponen kecil dihapus.
    """
    if min_area <= 1:
        return binary.copy()

    output = binary.copy()

    # Bersihkan noise putih (foreground 255)
    if target in ("white", "both"):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(output, connectivity=8)
        for label_idx in range(1, num_labels):  # 0 adalah background
            area = stats[label_idx, cv2.CC_STAT_AREA]
            if area < min_area:
                output[labels == label_idx] = 0

    # Bersihkan noise hitam (foreground 0 dari inverse)
    if target in ("black", "both"):
        inverted = cv2.bitwise_not(output)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)
        for label_idx in range(1, num_labels):
            area = stats[label_idx, cv2.CC_STAT_AREA]
            if area < min_area:
                output[labels == label_idx] = 255

    return output


def process_postprocessing(
    binary: np.ndarray,
    morph_op: str = "none",
    morph_ksize: int = 3,
    morph_iter: int = 1,
    morph_shape: str = "rect",
    remove_small: bool = False,
    min_area: int = 15,
    remove_target: str = "both"
) -> Dict[str, np.ndarray]:
    """
    Menjalankan rantai pasca-pemrosesan:
    1. Morfologi matematika (jika dipilih)
    2. Penghapusan komponen terhubung kecil (jika aktif)

    Return:
        Dict: {"postprocessed": np.ndarray, "steps": dict}
    """
    current = binary.copy()
    steps = {}

    if morph_op and morph_op.lower() != "none":
        current = apply_morphology(
            current,
            operation=morph_op,
            ksize=morph_ksize,
            iterations=morph_iter,
            shape=morph_shape
        )
        steps["morphology"] = current.copy()

    if remove_small and min_area > 1:
        current = remove_small_components(
            current,
            min_area=min_area,
            target=remove_target  # type: ignore
        )
        steps["remove_small"] = current.copy()

    return {
        "postprocessed": current,
        "steps": steps
    }
