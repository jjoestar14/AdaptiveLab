"""
Unit Test untuk Endpoint API Flask (tests/test_api.py).
"""

import json
import pytest
from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_api_samples(client):
    """Memastikan endpoint /api/samples mengembalikan daftar sampel default."""
    res = client.get("/api/samples")
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert len(data["samples"]) >= 3
    sample_ids = [s["id"] for s in data["samples"]]
    assert "sample1_uneven_light" in sample_ids


def test_api_process_sample(client):
    """Memastikan endpoint /api/process berjalan pada sampel bawaan."""
    payload = {
        "image_id": "sample1_uneven_light",
        "block_size": 25,
        "c": 5,
        "engine": "opencv",
        "preprocess": {
            "gaussian_blur": True,
            "gaussian_ksize": 3,
            "clahe": False
        },
        "postprocess": {
            "morph_op": "opening",
            "morph_ksize": 3,
            "remove_small": True,
            "min_area": 5
        }
    }
    res = client.post("/api/process", data=json.dumps(payload), content_type="application/json")
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert "statistics" in data
    assert "images" in data
    assert data["images"]["binary_final"].startswith("data:image/png;base64,")
    assert data["has_ground_truth"] is True
    assert "evaluation" in data
    assert data["evaluation"]["metrics"]["f1_score"] > 0.0


def test_api_compare(client):
    """Memastikan endpoint /api/compare mengembalikan 4 metode."""
    payload = {
        "image_id": "sample1_uneven_light",
        "global_thresh": 127,
        "block_size": 21,
        "c": 5
    }
    res = client.post("/api/compare", data=json.dumps(payload), content_type="application/json")
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert len(data["methods"]) == 4
    for key in ["global", "otsu", "adaptive_mean", "adaptive_gaussian"]:
        assert key in data["methods"]


def test_api_experiment_and_export(client):
    """Memastikan grid search eksperimen dan ekspor CSV/ZIP bekerja."""
    payload = {
        "image_id": "sample1_uneven_light",
        "block_sizes": [11, 21],
        "c_values": [3, 7],
        "engine": "opencv"
    }
    res = client.post("/api/experiment", data=json.dumps(payload), content_type="application/json")
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert data["total_combinations"] == 4
    exp_id = data["experiment_id"]

    # Test export CSV
    res_csv = client.get(f"/api/export/{exp_id}?type=csv")
    assert res_csv.status_code == 200
    assert b"block_size,c" in res_csv.data

    # Test export ZIP
    res_zip = client.get(f"/api/export/{exp_id}?type=zip")
    assert res_zip.status_code == 200
    assert res_zip.headers["Content-Type"] == "application/zip"
