"""
AdaptiveLab - Aplikasi Web Adaptive Gaussian Thresholding untuk Computer Vision.
Dikembangkan menggunakan Flask dan OpenCV.
"""

import os
import uuid
import re
import time
import io
import logging
from typing import Dict, Any, Optional
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_file,
    Response,
    abort
)
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import pandas as pd

from core.threshold import (
    validate_parameters,
    adaptive_gaussian_opencv,
    adaptive_gaussian_manual,
    get_threshold_map
)
from core.preprocess import process_preprocessing, to_grayscale
from core.postprocess import process_postprocessing
from core.compare import compare_methods
from core.metrics import evaluate_segmentation
from core.experiment import (
    run_parameter_grid_experiment,
    generate_experiment_csv,
    generate_experiment_zip
)
from core.utils import (
    decode_image_bytes,
    numpy_to_base64_data_url,
    compute_grayscale_histogram,
    compute_binary_statistics
)

# Konfigurasi Logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s in %(module)s: %(message)s"
)
logger = logging.getLogger("AdaptiveLab")

# Inisialisasi Flask App
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024  # Maksimum 12 MB
app.config["SECRET_KEY"] = os.urandom(24).hex()

# Direktori Penyimpanan
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_images")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
UPLOAD_DIR = os.path.join(OUTPUT_DIR, "uploads")
EXPERIMENT_DIR = os.path.join(OUTPUT_DIR, "experiments")

for d in [SAMPLE_DIR, OUTPUT_DIR, UPLOAD_DIR, EXPERIMENT_DIR]:
    os.makedirs(d, exist_ok=True)

# Cache Citra Aktif di Memori Server (image_id -> data)
# format: { "image_id": {"image": np.ndarray, "gt": Optional[np.ndarray], "name": str, "source": str} }
IMAGE_STORE: Dict[str, Dict[str, Any]] = {}

# Cache Hasil Eksperimen Aktif (exp_id -> Dict)
EXPERIMENT_STORE: Dict[str, Dict[str, Any]] = {}

# Cache Laporan Hasil Proses Terakhir (process_id -> Dict)
REPORT_STORE: Dict[str, Dict[str, Any]] = {}


def load_default_samples() -> None:
    """
    Mendaftarkan citra sampel ke dalam IMAGE_STORE:
    1. Memuat sampel terdaftar dengan judul & deskripsi terperinci.
    2. Secara otomatis memindai folder sample_images/ untuk mendeteksi dataset baru
       (termasuk mencocokkan file Ground Truth <nama>_gt.<ext> jika tersedia).
    """
    registered_samples = [
        {
            "id": "sample1_uneven_light",
            "name": "Dokumen Pencahayaan Tidak Merata",
            "file": "sample1_uneven_light.png",
            "gt_file": "sample1_uneven_light_gt.png",
            "description": "Gradien bayangan kuat dan lipatan buku. Menguji keunggulan ambang lokal terhadap metode global."
        },
        {
            "id": "sample2_salt_pepper",
            "name": "Dokumen Noise Salt-and-Pepper",
            "file": "sample2_salt_pepper.png",
            "gt_file": "sample2_salt_pepper_gt.png",
            "description": "Gangguan noise bintik putih dan hitam. Menguji kombinasi pra-proses Median Blur + Adaptive Gaussian."
        },
        {
            "id": "sample3_low_contrast",
            "name": "Dokumen Kontras Rendah (Pudar)",
            "file": "sample3_low_contrast.png",
            "gt_file": "sample3_low_contrast_gt.png",
            "description": "Intensitas latar dan teks sangat berdekatan. Menguji sensitivitas konstanta C dan filter CLAHE."
        }
    ]

    loaded_files = set()

    # 1. Muat sampel terdaftar
    for s in registered_samples:
        img_path = os.path.join(SAMPLE_DIR, s["file"])
        gt_path = os.path.join(SAMPLE_DIR, s["gt_file"]) if s.get("gt_file") else ""

        if os.path.exists(img_path):
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(gt_path) else None
            IMAGE_STORE[s["id"]] = {
                "id": s["id"],
                "image": img,
                "gt": gt,
                "name": s["name"],
                "description": s["description"],
                "source": "sample"
            }
            loaded_files.add(s["file"])
            if s.get("gt_file"):
                loaded_files.add(s["gt_file"])
            logger.info(f"Loaded registered sample: {s['id']}")

    # 2. Deteksi otomatis file baru di folder sample_images/
    valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    for fname in os.listdir(SAMPLE_DIR):
        if fname in loaded_files:
            continue
        lower_fname = fname.lower()
        if not lower_fname.endswith(valid_exts):
            continue
        # Lewati file ground truth terpisah agar tidak terdaftar sebagai citra sampel utama
        if "_gt." in lower_fname or lower_fname.endswith("_gt"):
            continue

        base_name, ext = os.path.splitext(fname)
        clean_id = re.sub(r"[^a-zA-Z0-9_-]", "_", base_name)

        # Cari pasangan ground truth jika ada (<base_name>_gt.<ext>)
        gt_img = None
        for test_ext in valid_exts:
            potential_gt = os.path.join(SAMPLE_DIR, f"{base_name}_gt{test_ext}")
            if os.path.exists(potential_gt):
                gt_img = cv2.imread(potential_gt, cv2.IMREAD_GRAYSCALE)
                loaded_files.add(f"{base_name}_gt{test_ext}")
                break

        full_img_path = os.path.join(SAMPLE_DIR, fname)
        img = cv2.imread(full_img_path, cv2.IMREAD_COLOR)
        if img is not None:
            display_title = base_name.replace("_", " ").replace("-", " ").title()
            IMAGE_STORE[clean_id] = {
                "id": clean_id,
                "image": img,
                "gt": gt_img,
                "name": display_title,
                "description": f"Dataset sampel dari berkas {fname}",
                "source": "sample"
            }
            loaded_files.add(fname)
            logger.info(f"Auto-discovered and loaded dataset: {clean_id} (GT: {gt_img is not None})")


load_default_samples()


def sanitize_id(identifier: str) -> str:
    """Memvalidasi identifier alfanumerik aman (menolak path traversal)."""
    if not identifier or not re.match(r"^[a-zA-Z0-9_-]+$", identifier):
        raise ValueError("Identifier tidak valid.")
    return identifier


# ==========================================
# ROUTES & CONTROLLERS
# ==========================================

@app.route("/")
def index():
    """Halaman Utama Aplikasi Single Page Application (SPA)."""
    return render_template("index.html")


@app.route("/api/samples", methods=["GET"])
def api_get_samples():
    """Mengembalikan daftar sampel bawaan yang tersedia."""
    result = []
    for img_id, item in IMAGE_STORE.items():
        if item.get("source") == "sample":
            img = item["image"]
            h, w = img.shape[:2]
            thumb_scale = 120.0 / max(h, w)
            thumb_w = max(1, int(round(w * thumb_scale)))
            thumb_h = max(1, int(round(h * thumb_scale)))
            thumb = cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)

            result.append({
                "id": img_id,
                "name": item["name"],
                "description": item.get("description", ""),
                "resolution": f"{w} x {h}",
                "has_ground_truth": item["gt"] is not None,
                "thumbnail_url": numpy_to_base64_data_url(thumb, ext="png")
            })
    return jsonify({"success": True, "samples": result})


@app.route("/api/upload", methods=["POST"])
def api_upload_image():
    """Menerima unggahan citra dari pengguna (maks 12MB, validasi via imdecode)."""
    try:
        if "file" not in request.files:
            return jsonify({"success": False, "error": "Tidak ada file yang diunggah."}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"success": False, "error": "Nama file kosong."}), 400

        raw_bytes = file.read()
        if len(raw_bytes) > 12 * 1024 * 1024:
            return jsonify({"success": False, "error": "Ukuran file melebihi batas 12 MB."}), 413

        img = decode_image_bytes(raw_bytes, max_dimension=2048)
        img_id = f"up_{uuid.uuid4().hex[:12]}"
        orig_name = secure_filename(file.filename) or "uploaded_image.png"

        IMAGE_STORE[img_id] = {
            "id": img_id,
            "image": img,
            "gt": None,
            "name": orig_name,
            "source": "upload"
        }

        gray = to_grayscale(img)
        h, w = img.shape[:2]
        hist_data = compute_grayscale_histogram(gray)

        return jsonify({
            "success": True,
            "image_id": img_id,
            "filename": orig_name,
            "resolution": f"{w} x {h}",
            "width": w,
            "height": h,
            "has_ground_truth": False,
            "histogram": hist_data,
            "image_url": numpy_to_base64_data_url(img, ext="png")
        })

    except Exception as exc:
        logger.error(f"Upload error: {str(exc)}", exc_info=True)
        return jsonify({"success": False, "error": str(exc)}), 400


@app.route("/api/upload-gt", methods=["POST"])
def api_upload_ground_truth():
    """Menerima unggahan citra Ground Truth untuk image_id tertentu."""
    try:
        image_id = sanitize_id(request.form.get("image_id", ""))
        if image_id not in IMAGE_STORE:
            return jsonify({"success": False, "error": "Image ID citra utama tidak ditemukan."}), 404

        if "file" not in request.files:
            return jsonify({"success": False, "error": "Tidak ada file ground truth yang dikirim."}), 400

        file = request.files["file"]
        raw_bytes = file.read()
        gt_img = decode_image_bytes(raw_bytes, max_dimension=2048)
        gt_gray = to_grayscale(gt_img)

        # Simpan GT ke store
        IMAGE_STORE[image_id]["gt"] = gt_gray

        h, w = gt_gray.shape[:2]
        return jsonify({
            "success": True,
            "message": "Ground truth berhasil diperbarui.",
            "image_id": image_id,
            "gt_resolution": f"{w} x {h}"
        })

    except Exception as exc:
        logger.error(f"Upload GT error: {str(exc)}", exc_info=True)
        return jsonify({"success": False, "error": str(exc)}), 400


@app.route("/api/process", methods=["POST"])
def api_process_image():
    """
    Memproses citra dengan parameter tunggal (Adaptive Gaussian + Pre/Post Pipeline).
    Mendukung pembaruan otomatis realtime (debounce) via JSON.
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"success": False, "error": "Payload JSON kosong."}), 400

        image_id = sanitize_id(data.get("image_id", ""))
        if image_id not in IMAGE_STORE:
            return jsonify({"success": False, "error": f"Citra dengan id '{image_id}' tidak ditemukan."}), 404

        img_record = IMAGE_STORE[image_id]
        img = img_record["image"]

        # Parameter Utama
        block_size = int(data.get("block_size", 21))
        c = int(data.get("c", 5))
        engine = str(data.get("engine", "opencv")).lower()
        foreground_is_black = bool(data.get("foreground_is_black", True))

        # Parameter Pra-pemrosesan
        prep_cfg = data.get("preprocess", {})
        gaussian_blur = bool(prep_cfg.get("gaussian_blur", False))
        gaussian_ksize = int(prep_cfg.get("gaussian_ksize", 3))
        gaussian_sigma = float(prep_cfg.get("gaussian_sigma", 0.0))

        median_blur = bool(prep_cfg.get("median_blur", False))
        median_ksize = int(prep_cfg.get("median_ksize", 3))

        clahe = bool(prep_cfg.get("clahe", False))
        clahe_clip = float(prep_cfg.get("clahe_clip_limit", 2.0))
        clahe_tile = int(prep_cfg.get("clahe_tile_size", 8))

        # Parameter Pasca-pemrosesan
        post_cfg = data.get("postprocess", {})
        morph_op = str(post_cfg.get("morph_op", "none")).lower()
        morph_ksize = int(post_cfg.get("morph_ksize", 3))
        morph_iter = int(post_cfg.get("morph_iter", 1))
        morph_shape = str(post_cfg.get("morph_shape", "rect")).lower()

        remove_small = bool(post_cfg.get("remove_small", False))
        min_area = int(post_cfg.get("min_area", 15))
        remove_target = str(post_cfg.get("remove_target", "both"))

        # 1. Pipeline Pra-pemrosesan
        t_start = time.perf_counter()
        prep_res = process_preprocessing(
            img,
            gaussian_blur=gaussian_blur,
            gaussian_ksize=gaussian_ksize,
            gaussian_sigma=gaussian_sigma,
            median_blur=median_blur,
            median_ksize=median_ksize,
            clahe=clahe,
            clahe_clip_limit=clahe_clip,
            clahe_tile_size=clahe_tile
        )
        gray = prep_res["grayscale"]
        preprocessed = prep_res["preprocessed"]

        # 2. Inti Metode: Adaptive Gaussian Thresholding
        t_thresh_0 = time.perf_counter()
        if engine == "manual":
            bin_raw = adaptive_gaussian_manual(preprocessed, block_size, c)
        else:
            bin_raw = adaptive_gaussian_opencv(preprocessed, block_size, c)
        t_thresh_ms = (time.perf_counter() - t_thresh_0) * 1000.0

        # Peta Ambang Lokal untuk visualisasi
        thresh_map = get_threshold_map(preprocessed, block_size, c)

        # 3. Pipeline Pasca-pemrosesan
        post_res = process_postprocessing(
            bin_raw,
            morph_op=morph_op,
            morph_ksize=morph_ksize,
            morph_iter=morph_iter,
            morph_shape=morph_shape,
            remove_small=remove_small,
            min_area=min_area,
            remove_target=remove_target
        )
        final_binary = post_res["postprocessed"]
        total_time_ms = (time.perf_counter() - t_start) * 1000.0

        # Hitung statistik
        stats = compute_binary_statistics(final_binary, execution_time_ms=total_time_ms)
        stats["threshold_time_ms"] = round(t_thresh_ms, 2)
        histogram = compute_grayscale_histogram(preprocessed)

        # 4. Evaluasi Ground Truth jika ada
        eval_result = None
        diff_map_url = None
        has_gt = img_record["gt"] is not None
        if has_gt:
            eval_data = evaluate_segmentation(
                final_binary,
                img_record["gt"],
                foreground_is_black=foreground_is_black
            )
            eval_result = {
                "metrics": eval_data["metrics"],
                "counts": eval_data["counts"],
                "warning": eval_data["warning"],
                "was_resized": eval_data["was_resized"]
            }
            diff_map_url = numpy_to_base64_data_url(eval_data["diff_map_rgb"], ext="png")

        # Simpan ke cache laporan untuk diunduh (JSON/CSV)
        report_id = f"rep_{uuid.uuid4().hex[:10]}"
        REPORT_STORE[report_id] = {
            "report_id": report_id,
            "image_id": image_id,
            "filename": img_record["name"],
            "block_size": block_size,
            "c": c,
            "engine": engine,
            "statistics": stats,
            "evaluation": eval_result,
            "preprocess_config": prep_cfg,
            "postprocess_config": post_cfg,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        return jsonify({
            "success": True,
            "report_id": report_id,
            "statistics": stats,
            "histogram": histogram,
            "has_ground_truth": has_gt,
            "evaluation": eval_result,
            "images": {
                "original": numpy_to_base64_data_url(img, ext="png"),
                "grayscale": numpy_to_base64_data_url(gray, ext="png"),
                "preprocessed": numpy_to_base64_data_url(preprocessed, ext="png"),
                "threshold_map": numpy_to_base64_data_url(thresh_map, ext="png"),
                "binary_raw": numpy_to_base64_data_url(bin_raw, ext="png"),
                "binary_final": numpy_to_base64_data_url(final_binary, ext="png"),
                "diff_map": diff_map_url
            },
            "pipeline_stages": {
                "grayscale": True,
                "gaussian_blur": gaussian_blur,
                "median_blur": median_blur,
                "clahe": clahe,
                "thresholding": True,
                "morphology": morph_op != "none",
                "remove_small": remove_small
            }
        })

    except Exception as exc:
        logger.error(f"Process error: {str(exc)}", exc_info=True)
        return jsonify({"success": False, "error": str(exc)}), 400


@app.route("/api/compare", methods=["POST"])
def api_compare():
    """Membandingkan 4 metode thresholding pada citra yang sama secara berdampingan."""
    try:
        data = request.get_json(force=True) or {}
        image_id = sanitize_id(data.get("image_id", ""))
        if image_id not in IMAGE_STORE:
            return jsonify({"success": False, "error": "Citra tidak ditemukan."}), 404

        img = IMAGE_STORE[image_id]["image"]
        gray = to_grayscale(img)

        global_thresh = int(data.get("global_thresh", 127))
        block_size = int(data.get("block_size", 21))
        c = int(data.get("c", 5))

        res_dict = compare_methods(
            gray,
            global_thresh=global_thresh,
            block_size=block_size,
            c=c
        )

        formatted_results = {}
        for key, item in res_dict.items():
            formatted_results[key] = {
                "name": item["name"],
                "description": item["description"],
                "threshold_value": str(item["threshold_value"]),
                "time_ms": item["time_ms"],
                "white_percent": item["white_percent"],
                "black_percent": item["black_percent"],
                "image_url": numpy_to_base64_data_url(item["binary"], ext="png")
            }

        return jsonify({
            "success": True,
            "methods": formatted_results
        })

    except Exception as exc:
        logger.error(f"Compare error: {str(exc)}", exc_info=True)
        return jsonify({"success": False, "error": str(exc)}), 400


@app.route("/api/experiment", methods=["POST"])
def api_experiment():
    """Menjalankan eksperimen grid search untuk kombinasi Block Size dan C."""
    try:
        data = request.get_json(force=True) or {}
        image_id = sanitize_id(data.get("image_id", ""))
        if image_id not in IMAGE_STORE:
            return jsonify({"success": False, "error": "Citra tidak ditemukan."}), 404

        img_record = IMAGE_STORE[image_id]
        img = img_record["image"]
        gt = img_record["gt"]

        raw_blocks = data.get("block_sizes", [11, 15, 21, 31])
        raw_c = data.get("c_values", [2, 5, 10])

        if isinstance(raw_blocks, str):
            raw_blocks = [int(x.strip()) for x in raw_blocks.split(",") if x.strip().isdigit()]
        if isinstance(raw_c, str):
            raw_c = [int(x.strip()) for x in raw_c.split(",") if x.strip().lstrip("-").isdigit()]

        engine = str(data.get("engine", "opencv")).lower()
        foreground_is_black = bool(data.get("foreground_is_black", True))

        prep_cfg = data.get("preprocess", None)
        post_cfg = data.get("postprocess", None)

        exp_data = run_parameter_grid_experiment(
            image=img,
            block_sizes=raw_blocks,
            c_values=raw_c,
            engine=engine,
            ground_truth=gt,
            preprocess_cfg=prep_cfg,
            postprocess_cfg=post_cfg,
            foreground_is_black=foreground_is_black
        )

        exp_id = f"exp_{uuid.uuid4().hex[:12]}"
        EXPERIMENT_STORE[exp_id] = {
            "exp_id": exp_id,
            "image_id": image_id,
            "filename": img_record["name"],
            "results": exp_data["results"],
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        # Format rows untuk JSON response (tanpa internal numpy arrays)
        clean_rows = []
        for r in exp_data["results"]:
            row_copy = {k: v for k, v in r.items() if not k.startswith("_")}
            clean_rows.append(row_copy)

        return jsonify({
            "success": True,
            "experiment_id": exp_id,
            "total_combinations": exp_data["total_combinations"],
            "has_ground_truth": exp_data["has_ground_truth"],
            "results": clean_rows
        })

    except Exception as exc:
        logger.error(f"Experiment error: {str(exc)}", exc_info=True)
        return jsonify({"success": False, "error": str(exc)}), 400


@app.route("/api/export/<exp_id>", methods=["GET"])
def api_export_experiment(exp_id: str):
    """Mengunduh hasil eksperimen dalam bentuk CSV atau berkas ZIP."""
    try:
        clean_id = sanitize_id(exp_id)
        export_type = request.args.get("type", "csv").lower()

        # Kasus 1: Ekspor hasil eksperimen
        if clean_id in EXPERIMENT_STORE:
            exp_record = EXPERIMENT_STORE[clean_id]
            results = exp_record["results"]

            if export_type == "csv":
                csv_data = generate_experiment_csv(results)
                return Response(
                    csv_data,
                    mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename=eksperimen_{clean_id}.csv"}
                )
            elif export_type == "zip":
                zip_bytes = generate_experiment_zip(results)
                return Response(
                    zip_bytes,
                    mimetype="application/zip",
                    headers={"Content-Disposition": f"attachment; filename=eksperimen_{clean_id}.zip"}
                )
            else:
                return jsonify({"success": False, "error": "Tipe ekspor tidak didukung (gunakan csv atau zip)."}), 400

        # Kasus 2: Ekspor laporan ringkas single process (JSON atau CSV)
        elif clean_id in REPORT_STORE:
            rep = REPORT_STORE[clean_id]
            if export_type == "json":
                return jsonify(rep)
            else:
                # Convert summary to single row CSV
                flat = {
                    "report_id": rep["report_id"],
                    "filename": rep["filename"],
                    "block_size": rep["block_size"],
                    "c": rep["c"],
                    "engine": rep["engine"],
                    **rep["statistics"]
                }
                if rep.get("evaluation") and rep["evaluation"].get("metrics"):
                    flat.update(rep["evaluation"]["metrics"])
                df = pd.DataFrame([flat])
                return Response(
                    df.to_csv(index=False),
                    mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename=laporan_{clean_id}.csv"}
                )

        else:
            return jsonify({"success": False, "error": "ID ekspor tidak ditemukan."}), 404

    except Exception as exc:
        logger.error(f"Export error: {str(exc)}", exc_info=True)
        return jsonify({"success": False, "error": str(exc)}), 400


# Error Handlers
@app.errorhandler(404)
def not_found(e):
    return jsonify({"success": False, "error": "Endpoint tidak ditemukan (404)."}), 404


@app.errorhandler(413)
def entity_too_large(e):
    return jsonify({"success": False, "error": "File terlalu besar. Batas maksimum adalah 12 MB."}), 413


@app.errorhandler(500)
def server_error(e):
    return jsonify({"success": False, "error": "Terjadi kesalahan internal pada server (500)."}), 500


if __name__ == "__main__":
    # Menjalankan server lokal pada port 5050 dengan debug=False
    print("=" * 60)
    print("  AdaptiveLab - Web Server Berjalan")
    print("  Buka di peramban: http://127.0.0.1:5050")
    print("  Port: 5050 | Debug: False")
    print("=" * 60)
    app.run(host="127.0.0.1", port=5050, debug=False)
