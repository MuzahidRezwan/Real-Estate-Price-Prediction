"""Model training, hyperparameter tuning, evaluation, and serialization.

Reads data/cleaned_dataset.csv, enforces strict anti-leakage checks, trains
and tunes 4 scikit-learn regression pipelines on an 80/20 train/test split
(random_state=42), prints a model comparison table, and persists:
  - models/best_model.pkl
  - models/model_metadata.json
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore", category=UserWarning)

ROOT_DIR = Path(__file__).resolve().parent.parent
CLEAN_PATH = ROOT_DIR / "data" / "cleaned_dataset.csv"
SUMMARY_PATH = ROOT_DIR / "data" / "cleaning_summary.json"
MODELS_DIR = ROOT_DIR / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

NUMERIC_FEATURES = ["Floor_area_sft", "Bedrooms", "Bathrooms", "Parking_Count"]
CATEGORICAL_FEATURES = ["Location", "City"]
TARGET_COL = "Estimated_Total_Price_Taka"
FORBIDDEN_LEAKAGE_COLS = {
    "Price_Per_Sft_Taka",
    "price_per_sft",
    "Raw_Sale_Price",
    "Title",
    "Property_ID",
    "Source_URL",
    TARGET_COL,
}


def build_preprocessor() -> ColumnTransformer:
    """Create strict anti-leakage scikit-learn ColumnTransformer."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def evaluate_predictions(
    y_true: pd.Series, y_pred: np.ndarray, mean_price: float
) -> dict[str, float]:
    """Compute R2, MAE, MAE% of mean price, and RMSE."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    mae_pct = (mae / mean_price) * 100.0 if mean_price > 0 else 0.0
    return {
        "r2": round(r2, 4),
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mae_pct_of_mean": round(mae_pct, 2),
    }


def train_and_evaluate(
    clean_path: Path = CLEAN_PATH,
    summary_path: Path = SUMMARY_PATH,
    models_dir: Path = MODELS_DIR,
) -> dict[str, Any]:
    """Train 4 regressors, select best pipeline, and save model + metadata."""
    df = pd.read_csv(clean_path)

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    # Strict anti-leakage assertion
    overlap = FORBIDDEN_LEAKAGE_COLS.intersection(set(feature_cols))
    assert not overlap, f"Data leakage detected in X columns: {overlap}"
    assert set(feature_cols).issubset(df.columns), "Missing expected feature columns"

    X = df[feature_cols].copy()
    y = df[TARGET_COL].copy()

    # Double-check anti-leakage on constructed matrix
    for bad_col in FORBIDDEN_LEAKAGE_COLS:
        assert bad_col not in X.columns, f"Forbidden column {bad_col} present in X"

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    mean_price = float(y.mean())

    # Model specifications and hyperparameter tuning grids
    model_specs: list[tuple[str, Any, str]] = [
        (
            "LinearRegression",
            LinearRegression(),
            "Baseline OLS linear regression",
        ),
        (
            "DecisionTreeRegressor",
            GridSearchCV(
                estimator=DecisionTreeRegressor(random_state=42),
                param_grid={
                    "reg__max_depth": [6, 8, 10, 12],
                    "reg__min_samples_split": [2, 5, 10],
                    "reg__min_samples_leaf": [3, 5, 8],
                },
                cv=cv,
                scoring="r2",
                n_jobs=1,
            ),
            "Decision Tree tuned via GridSearchCV",
        ),
        (
            "RandomForestRegressor",
            RandomizedSearchCV(
                estimator=RandomForestRegressor(random_state=42, n_jobs=1),
                param_distributions={
                    "reg__n_estimators": [200, 300, 400],
                    "reg__max_depth": [10, 12, 16, 20, None],
                    "reg__min_samples_split": [2, 4, 6],
                    "reg__min_samples_leaf": [1, 2, 3, 5],
                    "reg__max_features": [0.7, 1.0],
                },
                n_iter=15,
                cv=cv,
                scoring="r2",
                random_state=42,
                n_jobs=1,
            ),
            "Random Forest tuned via RandomizedSearchCV",
        ),
        (
            "GradientBoostingRegressor",
            GridSearchCV(
                estimator=GradientBoostingRegressor(random_state=42),
                param_grid={
                    "reg__n_estimators": [250, 300],
                    "reg__learning_rate": [0.03, 0.04, 0.05],
                    "reg__max_depth": [3, 4],
                    "reg__subsample": [0.85],
                    "reg__min_samples_leaf": [3, 5],
                },
                cv=cv,
                scoring="r2",
                n_jobs=1,
            ),
            "Gradient Boosting tuned via GridSearchCV",
        ),
    ]

    comparison: list[dict[str, Any]] = []
    fitted_pipelines: dict[str, Pipeline] = {}

    print("\nTraining & tuning 4 regression pipelines (80/20 split, random_state=42)...")
    for name, model_obj, desc in model_specs:
        t0 = time.perf_counter()
        if isinstance(model_obj, (GridSearchCV, RandomizedSearchCV)):
            base_pipe = Pipeline(
                steps=[("preprocessor", build_preprocessor()), ("reg", model_obj.estimator)]
            )
            model_obj.estimator = base_pipe
            model_obj.fit(X_train, y_train)
            best_pipe: Pipeline = model_obj.best_estimator_
            best_params = {
                k.replace("reg__", ""): v for k, v in model_obj.best_params_.items()
            }
        else:
            best_pipe = Pipeline(
                steps=[("preprocessor", build_preprocessor()), ("reg", model_obj)]
            )
            best_pipe.fit(X_train, y_train)
            best_params = {}

        elapsed = time.perf_counter() - t0
        y_pred = best_pipe.predict(X_test)
        metrics = evaluate_predictions(y_test, y_pred, mean_price)
        cv_r2 = float(
            cross_val_score(best_pipe, X, y, cv=cv, scoring="r2", n_jobs=1).mean()
        )

        record = {
            "name": name,
            "description": desc,
            **metrics,
            "cv_r2": round(cv_r2, 4),
            "train_seconds": round(elapsed, 2),
            "best_params": best_params,
        }
        comparison.append(record)
        fitted_pipelines[name] = best_pipe

    # Print comparison table sorted by R2 descending
    comparison_sorted = sorted(
        comparison, key=lambda r: (r["r2"], -r["mae"]), reverse=True
    )
    print("\n" + "=" * 96)
    print("MODEL COMPARISON TABLE (Test Set Evaluation, 20% Holdout)")
    print("=" * 96)
    header = (
        f"{'Model':<26} | {'R2':>7} | {'5-CV R2':>7} | "
        f"{'MAE (BDT)':>13} | {'MAE %':>6} | {'RMSE (BDT)':>13}"
    )
    print(header)
    print("-" * 96)
    for row in comparison_sorted:
        print(
            f"{row['name']:<26} | {row['r2']:>7.4f} | {row['cv_r2']:>7.4f} | "
            f"{row['mae']:>13,.0f} | {row['mae_pct_of_mean']:>5.2f}% | {row['rmse']:>13,.0f}"
        )
    print("=" * 96)

    best_row = comparison_sorted[0]
    best_name = best_row["name"]
    best_pipeline = fitted_pipelines[best_name]

    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, BEST_MODEL_PATH)

    # Extract categorical choices, modal Location->City mapping, and numeric bounds
    unique_locs = sorted([loc for loc in df["Location"].unique() if loc != "Other"])
    if "Other" in df["Location"].values:
        unique_locs.append("Other")
    unique_cities = sorted(df["City"].unique().tolist())

    loc_city_map = (
        df.groupby("Location")["City"].agg(lambda s: s.mode().iloc[0]).to_dict()
    )

    loc_stats = (
        df.groupby("Location")
        .agg(
            count=("Estimated_Total_Price_Taka", "count"),
            median_price_bdt=("Estimated_Total_Price_Taka", "median"),
            mean_price_bdt=("Estimated_Total_Price_Taka", "mean"),
            median_area_sft=("Floor_area_sft", "median"),
        )
        .round(2)
        .to_dict(orient="index")
    )

    numeric_bounds: dict[str, dict[str, float]] = {}
    for col in NUMERIC_FEATURES + [TARGET_COL]:
        s = df[col]
        numeric_bounds[col] = {
            "min": round(float(s.min()), 2),
            "max": round(float(s.max()), 2),
            "q01": round(float(s.quantile(0.01)), 2),
            "q05": round(float(s.quantile(0.05)), 2),
            "median": round(float(s.median()), 2),
            "mean": round(float(s.mean()), 2),
            "q95": round(float(s.quantile(0.95)), 2),
            "q99": round(float(s.quantile(0.99)), 2),
        }

    cleaning_summary: dict[str, Any] = {}
    if summary_path.exists():
        cleaning_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    implied_pps = df[TARGET_COL] / df["Floor_area_sft"]
    dataset_stats = {
        **cleaning_summary,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "total_clean_rows": int(len(df)),
        "mean_price_bdt": round(mean_price, 2),
        "median_price_bdt": round(float(y.median()), 2),
        "mean_price_per_sft_bdt": round(float(implied_pps.mean()), 2),
        "median_price_per_sft_bdt": round(float(implied_pps.median()), 2),
    }

    metadata = {
        "best_model_name": best_name,
        "best_model_metrics": best_row,
        "models_comparison": comparison_sorted,
        "feature_schema": {
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "target": TARGET_COL,
        },
        "locations": unique_locs,
        "cities": unique_cities,
        "location_city_map": loc_city_map,
        "location_stats": loc_stats,
        "numeric_bounds": numeric_bounds,
        "dataset_stats": dataset_stats,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\nBest model selected: {best_name} (Test R2={best_row['r2']:.4f})")
    print(f"Saved pipeline to   : {BEST_MODEL_PATH}")
    print(f"Saved metadata to   : {METADATA_PATH}")
    return metadata


if __name__ == "__main__":
    train_and_evaluate()
