"""
Modul Evaluasi Kuantitatif dan Metrik Segmentasi (Metrics).

Menyediakan perhitungan metrik evaluasi citra biner terhadap Ground Truth:
- Confusion Matrix (TP, FP, TN, FN)
- Precision, Recall, F1-Score, Accuracy
- Peak Signal-to-Noise Ratio (PSNR)
- Peta visualisasi perbedaan selisih (Confusion Color Map)
"""

import math
from typing import Dict, Any, Tuple
import cv2
import numpy as np


def ensure_ground_truth_compatibility(
    binary_pred: np.ndarray,
    binary_gt: np.ndarray
) -> Tuple[np.ndarray, bool, str]:
    """
    Memastikan citra ground truth kompatibel secara dimensi dan format biner uint8 (0 atau 255).
    Jika ukuran berbeda, ground truth di-resize dengan nearest-neighbor dan menghasilkan peringatan.

    Parameter:
        binary_pred: Citra hasil segmentasi (H, W) uint8.
        binary_gt: Citra ground truth (H_gt, W_gt) uint8.

    Return:
        Tuple: (binary_gt_disesuaikan, apakah_di_resize: bool, pesan_peringatan: str)
    """
    if binary_gt.ndim == 3:
        binary_gt = cv2.cvtColor(binary_gt, cv2.COLOR_BGR2GRAY)

    # Pastikan biner murni 0 atau 255 (jika ada nilai ambang antara)
    if not np.array_equal(np.unique(binary_gt), [0, 255]) and len(np.unique(binary_gt)) > 2:
        # Binarisasi otomatis jika GT belum murni biner
        _, binary_gt = cv2.threshold(binary_gt, 127, 255, cv2.THRESH_BINARY)
    else:
        # Jika nilai dalam rentang 0-1 (misal format biner boolean/0-1)
        if binary_gt.max() == 1:
            binary_gt = (binary_gt * 255).astype(np.uint8)

    pred_h, pred_w = binary_pred.shape[:2]
    gt_h, gt_w = binary_gt.shape[:2]

    is_resized = False
    warning_msg = ""

    if (pred_h, pred_w) != (gt_h, gt_w):
        binary_gt = cv2.resize(
            binary_gt,
            (pred_w, pred_h),
            interpolation=cv2.INTER_NEAREST
        )
        is_resized = True
        warning_msg = (
            f"Dimensi ground truth ({gt_w}x{gt_h}) berbeda dengan citra input ({pred_w}x{pred_h}). "
            f"Ground truth telah diselaraskan otomatis menggunakan interpolasi nearest-neighbor."
        )

    return binary_gt.astype(np.uint8), is_resized, warning_msg


def evaluate_segmentation(
    binary_pred: np.ndarray,
    binary_gt: np.ndarray,
    foreground_is_black: bool = True
) -> Dict[str, Any]:
    """
    Menghitung metrik kuantitatif: Precision, Recall, F1-Score, Accuracy, dan PSNR.

    Parameter:
        binary_pred: Citra biner hasil prediksi (0 dan 255).
        binary_gt: Citra biner ground truth (0 dan 255).
        foreground_is_black: True jika objek/teks yang diuji berwarna hitam (nilai 0),
                             False jika objek/teks berwarna putih (nilai 255).

    Return:
        Dict berisi metrik kuantitatif dan visualisasi.
    """
    gt_aligned, was_resized, warning = ensure_ground_truth_compatibility(binary_pred, binary_gt)

    if foreground_is_black:
        # Teks dokumen standar: piksel hitam (0) adalah Foreground
        pred_fg = (binary_pred == 0)
        gt_fg = (gt_aligned == 0)
    else:
        pred_fg = (binary_pred == 255)
        gt_fg = (gt_aligned == 255)

    tp = int(np.count_nonzero(pred_fg & gt_fg))
    fp = int(np.count_nonzero(pred_fg & (~gt_fg)))
    fn = int(np.count_nonzero((~pred_fg) & gt_fg))
    tn = int(np.count_nonzero((~pred_fg) & (~gt_fg)))

    total_pixels = binary_pred.size

    precision = float(tp) / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = float(tp) / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2.0 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    accuracy = float(tp + tn) / total_pixels if total_pixels > 0 else 0.0

    # Hitung MSE dan PSNR (Peak Signal to Noise Ratio) terhadap 255.0
    mse = float(np.mean((binary_pred.astype(np.float64) - gt_aligned.astype(np.float64)) ** 2))
    if mse <= 1e-9:
        psnr_val = 99.99  # Praktis identik / tak hingga
        psnr_display = "Infinity (Identik)"
    else:
        psnr_val = 10.0 * math.log10((255.0 ** 2) / mse)
        psnr_display = f"{psnr_val:.2f} dB"

    # Buat peta visualisasi selisih (Confusion RGB map)
    # TP (True Positive): Hijau [46, 204, 113] (Teks benar terdeteksi)
    # FP (False Positive): Merah [231, 76, 60] (Noise dianggap teks)
    # FN (False Negative): Biru [52, 152, 219] (Teks terlewat / hilang)
    # TN (True Negative): Latar abu-abu terang [245, 245, 245]
    h, w = binary_pred.shape[:2]
    diff_rgb = np.full((h, w, 3), 245, dtype=np.uint8)

    diff_rgb[pred_fg & gt_fg] = [46, 204, 113]       # Hijau: True Positive
    diff_rgb[pred_fg & (~gt_fg)] = [231, 76, 60]     # Merah: False Positive
    diff_rgb[(~pred_fg) & gt_fg] = [52, 152, 219]     # Biru: False Negative
    diff_rgb[(~pred_fg) & (~gt_fg)] = [245, 247, 250] # Latar netral

    return {
        "metrics": {
            "precision": round(precision, 4),
            "precision_percent": round(precision * 100.0, 2),
            "recall": round(recall, 4),
            "recall_percent": round(recall * 100.0, 2),
            "f1_score": round(f1, 4),
            "f1_score_percent": round(f1 * 100.0, 2),
            "accuracy": round(accuracy, 4),
            "accuracy_percent": round(accuracy * 100.0, 2),
            "psnr_db": round(psnr_val, 2),
            "psnr_display": psnr_display,
            "mse": round(mse, 3)
        },
        "counts": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "total_pixels": total_pixels
        },
        "warning": warning,
        "was_resized": was_resized,
        "diff_map_rgb": diff_rgb,
        "aligned_gt": gt_aligned
    }
