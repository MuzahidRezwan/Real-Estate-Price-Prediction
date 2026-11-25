"""Automated API, security, validation, and model integrity tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app import app

CLEAN_CSV = ROOT_DIR / "data" / "cleaned_dataset.csv"
METADATA_JSON = ROOT_DIR / "models" / "model_metadata.json"

client = TestClient(app)


def test_home_page_renders_with_security_meta_and_disclaimer():
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.text
    assert "Real Estate Price Prediction System" in html
    assert 'http-equiv="X-Content-Type-Options" content="nosniff"' in html
    assert 'http-equiv="X-XSS-Protection" content="1; mode=block"' in html
    assert 'name="referrer" content="strict-origin-when-cross-origin"' in html
    assert "not</strong> constitute a certified legal appraisal" in html
    assert "/static/charts/price_distribution.png" in html


def test_enterprise_security_headers_and_hidden_server_signature():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    headers = {k.lower(): v for k, v in resp.headers.items()}
    assert headers.get("x-frame-options") == "DENY"
    assert headers.get("x-content-type-options") == "nosniff"
    assert headers.get("x-xss-protection") == "1; mode=block"
    assert headers.get("referrer-policy") == "strict-origin-when-cross-origin"
    assert "max-age=31536000" in headers.get("strict-transport-security", "")
    assert "camera=(), microphone=(), geolocation=()" in headers.get(
        "permissions-policy", ""
    )
    assert "x-powered-by" not in headers
    assert "server" not in headers


def test_locations_endpoint_returns_dynamic_options():
    resp = client.get("/api/locations")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["locations"]) >= 10
    assert "Other" in data["locations"]
    assert "Dhanmondi" in data["locations"]
    assert "Dhaka" in data["cities"]
    assert data["location_city_map"]["Dhanmondi"] == "Dhaka"
    assert "Floor_area_sft" in data["numeric_bounds"]


def test_metrics_endpoint_has_all_four_models_and_target_r2():
    resp = client.get("/api/metrics")
    assert resp.status_code == 200
    data = resp.json()
    names = {m["name"] for m in data["models_comparison"]}
    assert names == {
        "LinearRegression",
        "DecisionTreeRegressor",
        "RandomForestRegressor",
        "GradientBoostingRegressor",
    }
    best_r2 = data["best_model_metrics"]["r2"]
    assert best_r2 >= 0.80
    assert data["dataset_stats"]["total_clean_rows"] >= 1000


def test_predict_valid_apartment_under_latency_budget():
    payload = {
        "location": "Dhanmondi",
        "city": "Dhaka",
        "floor_area_sft": 1600,
        "bedrooms": 3,
        "bathrooms": 3,
        "parking_count": 1,
    }
    resp = client.post("/api/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["predicted_price_bdt"] > 1_000_000
    assert "৳" in body["formatted_bdt"]
    assert "BDT" in body["lakh_crore"]
    assert body["estimated_price_per_sft"] > 1000
    assert body["out_of_bounds"] is False
    assert body["warnings"] == []
    assert body["latency_ms"] < 500.0


def test_predict_out_of_bounds_triggers_fr5_warning():
    payload = {
        "location": "Gulshan 2",
        "city": "Dhaka",
        "floor_area_sft": 9500,
        "bedrooms": 12,
        "bathrooms": 10,
        "parking_count": 1,
    }
    resp = client.post("/api/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["predicted_price_bdt"] > 0
    assert body["out_of_bounds"] is True
    assert len(body["warnings"]) >= 2
    assert body["latency_ms"] < 500.0


def test_predict_unknown_location_gracefully_handled_with_notice():
    payload = {
        "location": "Unseen Future Township",
        "city": "Dhaka",
        "floor_area_sft": 1300,
        "bedrooms": 3,
        "bathrooms": 2,
        "parking_count": 1,
    }
    resp = client.post("/api/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["predicted_price_bdt"] > 0
    assert body["out_of_bounds"] is True
    assert any("Unseen Future Township" in w for w in body["warnings"])


@pytest.mark.parametrize(
    "bad_payload",
    [
        {
            "location": "Mirpur",
            "city": "Dhaka",
            "floor_area_sft": -1200,
            "bedrooms": 3,
            "bathrooms": 2,
            "parking_count": 1,
        },
        {
            "location": "Mirpur",
            "city": "Dhaka",
            "floor_area_sft": 0,
            "bedrooms": 3,
            "bathrooms": 2,
            "parking_count": 1,
        },
        {
            "location": "Mirpur",
            "city": "Dhaka",
            "floor_area_sft": 1200,
            "bedrooms": 0,
            "bathrooms": 2,
            "parking_count": 1,
        },
        {
            "location": "Mirpur",
            "city": "Dhaka",
            "floor_area_sft": 1200,
            "bedrooms": 3,
            "bathrooms": -1,
            "parking_count": 1,
        },
        {
            "location": "Mirpur",
            "city": "Dhaka",
            "floor_area_sft": 1200,
            "bedrooms": 3,
            "bathrooms": 2,
            "parking_count": -1,
        },
        {
            "location": "   ",
            "city": "Dhaka",
            "floor_area_sft": 1200,
            "bedrooms": 3,
            "bathrooms": 2,
            "parking_count": 1,
        },
        {
            "location": "Mirpur",
            "floor_area_sft": "invalid-string",
            "bedrooms": 3,
            "bathrooms": 2,
        },
    ],
)
def test_predict_rejects_invalid_inputs_with_http_400(bad_payload):
    resp = client.post("/api/predict", json=bad_payload)
    assert resp.status_code == 400


def test_payload_size_limit_rejects_over_100kb():
    oversized = {
        "location": "Mirpur",
        "city": "Dhaka",
        "floor_area_sft": 1200,
        "bedrooms": 3,
        "bathrooms": 2,
        "parking_count": 1,
        "padding": "A" * (105 * 1024),
    }
    resp = client.post("/api/predict", json=oversized)
    assert resp.status_code == 413


def test_anti_leakage_and_anonymization_constraints():
    df_clean = pd.read_csv(CLEAN_CSV)
    forbidden = {
        "Price_Per_Sft_Taka",
        "price_per_sft",
        "Raw_Sale_Price",
        "Title",
        "Property_ID",
        "Source_URL",
    }
    assert forbidden.isdisjoint(set(df_clean.columns))

    meta = json.loads(METADATA_JSON.read_text(encoding="utf-8"))
    schema_cols = set(meta["feature_schema"]["numeric_features"]) | set(
        meta["feature_schema"]["categorical_features"]
    )
    assert forbidden.isdisjoint(schema_cols)
    assert meta["feature_schema"]["target"] not in schema_cols
