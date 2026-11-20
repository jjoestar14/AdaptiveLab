"""
Modul Utilitas Citra, Statistik, dan Histogram (Utils).

Menyediakan helper konversi format citra aman, encoding data-URL,
penghitungan histogram intensitas untuk Chart.js, serta metrik komponen terhubung.
"""

import base64
from typing import Dict, Any, Tuple, Optional
import cv2
import numpy as np


def decode_image_bytes(file_bytes: bytes, max_dimension: int = 2048) -> np.ndarray:
    """
    Memvalidasi dan mendecode bytes file citra menggunakan cv2.imdecode.
    Mencegah file exploit dan membatasi resolusi maksimum dengan proporsional resize.

    Parameter:
        file_bytes: Data mentah byte citra.
        max_dimension: Batas sisi terpanjang (default: 2048 px).

    Return:
        np.ndarray: Citra decoded (BGR atau grayscale).

    Raises:
        ValueError: Jika data bytes tidak dapat didecode sebagai citra valid.
    """
    if not file_bytes or len(file_bytes) == 0:
        raise ValueError("Data file kosong.")

    np_arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("File bukan citra yang valid atau format file korup.")

    h, w = img.shape[:2]
    max_side = max(h, w)
    if max_side > max_dimension:
        scale = max_dimension / float(max_side)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return img


def numpy_to_base64_data_url(image: np.ndarray, ext: str = "png") -> str:
    """
    Mengonversi array numpy citra ke string Data URL Base64 yang siap ditampilkan di tag <img>.

    Parameter:
        image: Citra numpy (grayscale atau BGR).
        ext: Format gambar ('png', 'jpg', 'webp').

    Return:
        str: String data URL base64, contoh 'data:image/png;base64,...'.
    """
    format_clean = ext.lower().replace(".", "")
    if format_clean not in ("png", "jpg", "jpeg", "webp", "bmp"):
        format_clean = "png"

    success, buffer = cv2.imencode(f".{format_clean}", image)
    if not success:
        raise RuntimeError("Gagal melakukan encoding citra ke buffer memori.")

    b64_str = base64.b64encode(buffer).decode("utf-8")
    mime = "image/jpeg" if format_clean in ("jpg", "jpeg") else f"image/{format_clean}"
    return f"data:{mime};base64,{b64_str}"


def compute_grayscale_histogram(gray: np.ndarray) -> Dict[str, Any]:
    """
    Menghitung histogram intensitas grayscale 256 bin beserta statistik intensitas dasar.

    Parameter:
        gray: Citra grayscale 2D uint8.

    Return:
        Dict: labels [0..255], counts list, mean, std, min, max intensitas.
    """
    if gray.ndim != 2:
        raise ValueError("Histogram memerlukan citra 2D grayscale.")

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    return {
        "labels": list(range(256)),
        "counts": [int(val) for val in hist],
        "stats": {
            "mean": round(float(np.mean(gray)), 2),
            "std": round(float(np.std(gray)), 2),
            "min": int(np.min(gray)),
            "max": int(np.max(gray)),
            "median": int(np.median(gray))
        }
    }


def compute_binary_statistics(binary_img: np.ndarray, execution_time_ms: float = 0.0) -> Dict[str, Any]:
    """
    Menghitung statistik kuantitatif citra biner:
    - Resolusi dan total piksel
    - Persentase piksel hitam (nilai 0) dan putih (nilai 255)
    - Jumlah komponen terhubung (connected components)
    - Waktu eksekusi proses (ms)

    Parameter:
        binary_img: Citra biner 2D uint8.
        execution_time_ms: Durasi komputasi dalam milidetik.

    Return:
        Dict berisi statistik citra biner.
    """
    h, w = binary_img.shape[:2]
    total_pixels = binary_img.size

    white_count = int(np.count_nonzero(binary_img == 255))
    black_count = total_pixels - white_count

    white_pct = round((white_count / total_pixels) * 100.0, 2)
    black_pct = round((black_count / total_pixels) * 100.0, 2)

    # Hitung komponen terhubung untuk objek putih dan hitam
    num_labels_white, _, _, _ = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
    connected_white = max(0, num_labels_white - 1)  # Kurangi label background

    inverted = cv2.bitwise_not(binary_img)
    num_labels_black, _, _, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)
    connected_black = max(0, num_labels_black - 1)

    return {
        "resolution": f"{w} x {h}",
        "width": w,
        "height": h,
        "total_pixels": total_pixels,
        "white_pixels": white_count,
        "black_pixels": black_count,
        "white_percentage": white_pct,
        "black_percentage": black_pct,
        "connected_components_objects": connected_white,
        "connected_components_text_black": connected_black,
        "execution_time_ms": round(execution_time_ms, 2)
    }
