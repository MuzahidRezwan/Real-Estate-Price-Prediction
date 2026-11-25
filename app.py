"""FastAPI backend for the Real Estate Price Prediction System (DIU SE331 Capstone)."""

from __future__ import annotations

import json
import math
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.base import BaseHTTPMiddleware

ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "models" / "best_model.pkl"
METADATA_PATH = ROOT_DIR / "models" / "model_metadata.json"
TEMPLATES_DIR = ROOT_DIR / "templates"
STATIC_DIR = ROOT_DIR / "static"

MAX_PAYLOAD_BYTES = 100 * 1024  # 100 KB payload limit (Universal Security Standard)


class SecurityAndPayloadMiddleware(BaseHTTPMiddleware):
    """Enforce payload limits, hide server headers, and set security headers."""

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > MAX_PAYLOAD_BYTES:
                    return JSONResponse(
                        status_code=413,
                        content={"detail": "Payload too large. Maximum allowed size is 100 KB."},
                    )
            except ValueError:
                return JSONResponse(
                    status_code=400,
                    content={"detail": "Invalid Content-Length header."},
                )

        if request.method in {"POST", "PUT", "PATCH"}:
            body = await request.body()
            if len(body) > MAX_PAYLOAD_BYTES:
                return JSONResponse(
                    status_code=413,
                    content={"detail": "Payload too large. Maximum allowed size is 100 KB."},
                )

        response = await call_next(request)

        # Hide server signature headers
        for sig_header in ("server", "x-powered-by"):
            if sig_header in response.headers:
                del response.headers[sig_header]

        # Enterprise security headers
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains; preload"
        )
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=()"
        )
        return response


@lru_cache(maxsize=1)
def get_model():
    """Load and cache the serialized scikit-learn inference Pipeline."""
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Trained model not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def get_metadata() -> dict[str, Any]:
    """Load and cache model metadata JSON."""
    if not METADATA_PATH.exists():
        raise RuntimeError(f"Model metadata not found at {METADATA_PATH}")
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def format_bdt_comma(amount: float) -> str:
    """Format an integer BDT amount using Bangladeshi lakh/crore comma grouping."""
    rounded = int(round(amount))
    sign = "-" if rounded < 0 else ""
    s = str(abs(rounded))
    if len(s) <= 3:
        return f"{sign}৳ {s}"
    last3 = s[-3:]
    rest = s[:-3]
    groups = []
    while len(rest) > 2:
        groups.insert(0, rest[-2:])
        rest = rest[:-2]
    if rest:
        groups.insert(0, rest)
    return f"{sign}৳ {','.join(groups)},{last3}"


def format_lakh_crore(amount: float) -> str:
    """Convert raw BDT amount to human-readable Crore or Lakh notation."""
    if amount >= 1e7:
        return f"{amount / 1e7:.2f} Crore BDT"
    return f"{amount / 1e5:.2f} Lakh BDT"


class PredictionRequest(BaseModel):
    """Input schema for property valuation requests."""

    model_config = ConfigDict(extra="ignore")

    location: str = Field(..., min_length=1, max_length=100)
    city: str | None = Field(default=None, max_length=100)
    floor_area_sft: float
    bedrooms: float
    bathrooms: float
    parking_count: float = 0.0


def build_boundary_warnings(
    location: str,
    city: str,
    floor_area_sft: float,
    bedrooms: float,
    bathrooms: float,
    parking_count: float,
    metadata: dict[str, Any],
) -> list[str]:
    """Check inputs against training bounds and domain sanity rules (FR-5)."""
    warnings: list[str] = []
    bounds = metadata.get("numeric_bounds", {})
    valid_locs = set(metadata.get("locations", []))

    if location not in valid_locs:
        warnings.append(
            f"Location '{location}' is outside the 35 modeled neighborhood tiers; fallback category encoding was applied."
        )

    area_b = bounds.get("Floor_area_sft", {"min": 450, "max": 3748})
    if floor_area_sft < area_b["min"] or floor_area_sft > area_b["max"]:
        warnings.append(
            f"Floor area ({floor_area_sft:,.0f} sqft) is outside training bounds "
            f"[{area_b['min']:,.0f} – {area_b['max']:,.0f} sqft]; prediction extrapolates beyond observed data."
        )

    bed_b = bounds.get("Bedrooms", {"min": 1, "max": 6})
    if bedrooms < bed_b["min"] or bedrooms > bed_b["max"]:
        warnings.append(
            f"Bedrooms ({bedrooms:g}) falls outside training bounds "
            f"[{bed_b['min']:g} – {bed_b['max']:g}]."
        )

    bath_b = bounds.get("Bathrooms", {"min": 1, "max": 6})
    if bathrooms < bath_b["min"] or bathrooms > bath_b["max"]:
        warnings.append(
            f"Bathrooms ({bathrooms:g}) falls outside training bounds "
            f"[{bath_b['min']:g} – {bath_b['max']:g}]."
        )

    park_b = bounds.get("Parking_Count", {"min": 0, "max": 4})
    if parking_count < park_b["min"] or parking_count > park_b["max"]:
        warnings.append(
            f"Parking spaces ({parking_count:g}) falls outside training bounds "
            f"[{park_b['min']:g} – {park_b['max']:g}]."
        )

    total_rooms = bedrooms + bathrooms
    if total_rooms > 0 and (floor_area_sft / total_rooms) < 110:
        warnings.append(
            "Unusually compact area-per-room ratio (< 110 sqft/room); verify layout inputs."
        )

    if bedrooms >= 1 and floor_area_sft / bedrooms > 1500:
        warnings.append(
            "Unusually spacious area-per-bedroom (> 1,500 sqft/bedroom); estimate may carry higher variance."
        )

    loc_city_map = metadata.get("location_city_map", {})
    expected_city = loc_city_map.get(location)
    if location != "Other" and expected_city and city.lower() != expected_city.lower():
        warnings.append(
            f"Selected city '{city}' differs from the dataset's primary city '{expected_city}' for '{location}'."
        )

    return warnings


def create_app() -> FastAPI:
    """Application factory creating configured FastAPI app."""
    application = FastAPI(
        title="Real Estate Price Prediction System",
        description="DIU SE331 Software Engineering Design Capstone — ML Property Valuation API",
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url=None,
    )

    allowed_origins_env = os.getenv(
        "ALLOWED_ORIGINS",
        "http://127.0.0.1:8000,http://localhost:8000",
    )
    allowed_origins = [
        o.strip() for o in allowed_origins_env.split(",") if o.strip()
    ]

    application.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Accept"],
    )
    application.add_middleware(SecurityAndPayloadMiddleware)

    application.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    @application.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Convert Pydantic schema validation errors (422) into clean HTTP 400 responses."""
        errors = []
        for err in exc.errors():
            loc = ".".join(str(x) for x in err.get("loc", []) if x != "body")
            errors.append(f"{loc}: {err.get('msg', 'invalid input')}")
        return JSONResponse(
            status_code=400,
            content={"detail": "; ".join(errors) or "Invalid request payload."},
        )

    @application.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        metadata = get_metadata()
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "metadata": metadata,
            },
        )

    @application.get("/api/health")
    async def health():
        metadata = get_metadata()
        _ = get_model()
        return {
            "status": "ok",
            "best_model": metadata.get("best_model_name"),
            "clean_rows": metadata.get("dataset_stats", {}).get("total_clean_rows"),
        }

    @application.get("/api/locations")
    async def list_locations():
        metadata = get_metadata()
        return {
            "locations": metadata.get("locations", []),
            "cities": metadata.get("cities", []),
            "location_city_map": metadata.get("location_city_map", {}),
            "location_stats": metadata.get("location_stats", {}),
            "numeric_bounds": metadata.get("numeric_bounds", {}),
        }

    @application.get("/api/metrics")
    async def get_metrics():
        metadata = get_metadata()
        return {
            "best_model_name": metadata.get("best_model_name"),
            "best_model_metrics": metadata.get("best_model_metrics"),
            "models_comparison": metadata.get("models_comparison", []),
            "feature_schema": metadata.get("feature_schema", {}),
            "dataset_stats": metadata.get("dataset_stats", {}),
        }

    @application.post("/api/predict")
    async def predict_price(payload: PredictionRequest):
        t_start = time.perf_counter()

        location = payload.location.strip()
        if not location:
            raise HTTPException(
                status_code=400, detail="Location must not be empty."
            )

        for field_name, val in (
            ("floor_area_sft", payload.floor_area_sft),
            ("bedrooms", payload.bedrooms),
            ("bathrooms", payload.bathrooms),
            ("parking_count", payload.parking_count),
        ):
            if not math.isfinite(val):
                raise HTTPException(
                    status_code=400,
                    detail=f"{field_name} must be a finite number.",
                )

        if payload.floor_area_sft <= 0:
            raise HTTPException(
                status_code=400,
                detail="floor_area_sft must be strictly greater than zero.",
            )
        if payload.bedrooms <= 0:
            raise HTTPException(
                status_code=400,
                detail="bedrooms must be strictly greater than zero.",
            )
        if payload.bathrooms <= 0:
            raise HTTPException(
                status_code=400,
                detail="bathrooms must be strictly greater than zero.",
            )
        if payload.parking_count < 0:
            raise HTTPException(
                status_code=400,
                detail="parking_count cannot be negative.",
            )

        metadata = get_metadata()
        model = get_model()

        loc_city_map = metadata.get("location_city_map", {})
        city = (
            payload.city.strip()
            if payload.city and payload.city.strip()
            else loc_city_map.get(location, "Dhaka")
        )

        # Match exact DataFrame schema used during model fitting
        input_df = pd.DataFrame(
            [
                {
                    "Floor_area_sft": float(payload.floor_area_sft),
                    "Bedrooms": float(payload.bedrooms),
                    "Bathrooms": float(payload.bathrooms),
                    "Parking_Count": float(payload.parking_count),
                    "Location": location,
                    "City": city,
                }
            ]
        )

        raw_pred = float(model.predict(input_df)[0])
        predicted_bdt = max(round(raw_pred), 100_000)
        est_pps = round(predicted_bdt / float(payload.floor_area_sft), 2)

        warnings = build_boundary_warnings(
            location=location,
            city=city,
            floor_area_sft=float(payload.floor_area_sft),
            bedrooms=float(payload.bedrooms),
            bathrooms=float(payload.bathrooms),
            parking_count=float(payload.parking_count),
            metadata=metadata,
        )

        latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)

        return {
            "predicted_price_bdt": predicted_bdt,
            "formatted_bdt": format_bdt_comma(predicted_bdt),
            "lakh_crore": format_lakh_crore(predicted_bdt),
            "estimated_price_per_sft": est_pps,
            "formatted_price_per_sft": f"{format_bdt_comma(est_pps)} / sqft",
            "latency_ms": latency_ms,
            "model_used": metadata.get("best_model_name"),
            "out_of_bounds": len(warnings) > 0,
            "warnings": warnings,
            "normalized_input": {
                "location": location,
                "city": city,
                "floor_area_sft": float(payload.floor_area_sft),
                "bedrooms": float(payload.bedrooms),
                "bathrooms": float(payload.bathrooms),
                "parking_count": float(payload.parking_count),
            },
        }

    return application


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, server_header=False)
