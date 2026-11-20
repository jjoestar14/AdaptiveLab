"""
Modul Eksperimen Parameter Kombinasi (Experiment).

Menjalankan eksplorasi grid kombinasi Block Size dan C, menghitung metrik,
serta menyediakan fungsi ekspor CSV dan ZIP untuk laporan penelitian.
"""

import io
import time
import zipfile
from typing import List, Dict, Any, Optional
import cv2
import numpy as np
import pandas as pd

from core.threshold import (
    validate_parameters,
    adaptive_gaussian_opencv,
    adaptive_gaussian_manual
)
from core.preprocess import process_preprocessing
from core.postprocess import process_postprocessing
from core.metrics import evaluate_segmentation
from core.utils import numpy_to_base64_data_url


def run_parameter_grid_experiment(
    image: np.ndarray,
    block_sizes: List[int],
    c_values: List[int],
    engine: str = "opencv",
    ground_truth: Optional[np.ndarray] = None,
    preprocess_cfg: Optional[Dict[str, Any]] = None,
    postprocess_cfg: Optional[Dict[str, Any]] = None,
    foreground_is_black: bool = True
) -> Dict[str, Any]:
    """
    Menjalankan grid search untuk semua kombinasi Block Size dan C.

    Parameter:
        image: Citra input (BGR atau grayscale).
        block_sizes: Daftar ukuran jendela lokal ganjil >= 3.
        c_values: Daftar nilai konstanta C.
        engine: 'opencv' atau 'manual'.
        ground_truth: Citra ground truth opsional untuk evaluasi metrik.
        preprocess_cfg: Konfigurasi pra-pemrosesan opsional.
        postprocess_cfg: Konfigurasi pasca-pemrosesan opsional.
        foreground_is_black: True jika teks berwarna hitam.

    Return:
        Dict: {"results": List[dict], "summary": dict}
    """
    # 1. Jalankan pra-pemrosesan jika ada
    if preprocess_cfg:
        prep_res = process_preprocessing(image, **preprocess_cfg)
        gray = prep_res["preprocessed"]
    else:
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

    total_pixels = float(gray.size)
    has_gt = ground_truth is not None

    results = []
    run_fn = adaptive_gaussian_manual if engine == "manual" else adaptive_gaussian_opencv

    # Validasi daftar block_size dan c
    clean_blocks = []
    for bs in block_sizes:
        if bs >= 3 and bs % 2 == 1:
            clean_blocks.append(int(bs))
    clean_blocks = sorted(list(set(clean_blocks)))

    clean_c = sorted(list(set([int(c) for c in c_values])))

    if not clean_blocks:
        raise ValueError("Daftar Block Size harus memuat setidaknya satu bilangan ganjil >= 3.")
    if not clean_c:
        raise ValueError("Daftar konstanta C tidak boleh kosong.")

    for bs in clean_blocks:
        for c in clean_c:
            t0 = time.perf_counter()
            bin_raw = run_fn(gray, bs, c)

            # Jalankan pasca-pemrosesan jika aktif
            if postprocess_cfg and (
                postprocess_cfg.get("morph_op", "none") != "none" or
                postprocess_cfg.get("remove_small", False)
            ):
                post_res = process_postprocessing(bin_raw, **postprocess_cfg)
                bin_final = post_res["postprocessed"]
            else:
                bin_final = bin_raw

            duration_ms = (time.perf_counter() - t0) * 1000.0

            white_count = int(np.count_nonzero(bin_final == 255))
            black_count = int(total_pixels - white_count)
            white_pct = round((white_count / total_pixels) * 100.0, 2)
            black_pct = round((black_count / total_pixels) * 100.0, 2)

            # Buat thumbnail base64 untuk grid tampilan cepat
            h, w = bin_final.shape[:2]
            thumb_scale = 160.0 / max(h, w)
            thumb_w = max(1, int(round(w * thumb_scale)))
            thumb_h = max(1, int(round(h * thumb_scale)))
            thumb_img = cv2.resize(bin_final, (thumb_w, thumb_h), interpolation=cv2.INTER_NEAREST)
            thumb_b64 = numpy_to_base64_data_url(thumb_img, ext="png")

            row_data: Dict[str, Any] = {
                "block_size": bs,
                "c": c,
                "engine": engine,
                "time_ms": round(duration_ms, 2),
                "black_percent": black_pct,
                "white_percent": white_pct,
                "thumbnail_url": thumb_b64,
                # Simpan full image binary di memori internal untuk ekspor
                "_binary_img": bin_final
            }

            if has_gt:
                eval_data = evaluate_segmentation(bin_final, ground_truth, foreground_is_black=foreground_is_black)
                m = eval_data["metrics"]
                row_data.update({
                    "precision": m["precision"],
                    "precision_percent": m["precision_percent"],
                    "recall": m["recall"],
                    "recall_percent": m["recall_percent"],
                    "f1_score": m["f1_score"],
                    "f1_score_percent": m["f1_score_percent"],
                    "accuracy": m["accuracy"],
                    "accuracy_percent": m["accuracy_percent"],
                    "psnr_db": m["psnr_db"]
                })

            results.append(row_data)

    return {
        "results": results,
        "total_combinations": len(results),
        "has_ground_truth": has_gt
    }


def generate_experiment_csv(results: List[Dict[str, Any]]) -> str:
    """
    Mengonversi list hasil eksperimen ke format string CSV menggunakan Pandas.
    """
    clean_rows = []
    for r in results:
        copy_r = {k: v for k, v in r.items() if not k.startswith("_") and k != "thumbnail_url"}
        clean_rows.append(copy_r)

    df = pd.DataFrame(clean_rows)
    return df.to_csv(index=False)


def generate_experiment_zip(results: List[Dict[str, Any]]) -> bytes:
    """
    Membuat paket ZIP di memori yang berisi tabel CSV eksperimen dan seluruh
    gambar biner hasil kombinasi resolusi penuh (PNG).
    """
    csv_str = generate_experiment_csv(results)

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Tulis file CSV
        zf.writestr("hasil_eksperimen_parameter.csv", csv_str)

        # Tulis gambar masing-masing kombinasi
        for row in results:
            bs = row["block_size"]
            c = row["c"]
            bin_img = row.get("_binary_img")
            if bin_img is not None:
                _, png_buf = cv2.imencode(".png", bin_img)
                filename = f"biner_bs{bs}_c{c}.png"
                zf.writestr(f"citra_hasil/{filename}", png_buf.tobytes())

    zip_buffer.seek(0)
    return zip_buffer.getvalue()
