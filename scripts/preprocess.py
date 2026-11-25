"""Dataset inspection, cleaning, outlier removal, and dynamic categorization.

Reads data/pbazaar_apartments_raw.xlsx (sheet 'Raw_Data'), parses string
currency/unit fields, removes duplicates and invalid rows, applies domain
sanity bounds + IQR (k=1.5) outlier filtering, groups sparse locations (<10)
into 'Other', strips identifiers/URLs/titles/leakage columns, and saves
data/cleaned_dataset.csv plus data/cleaning_summary.json.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT_DIR / "data" / "pbazaar_apartments_raw.xlsx"
CLEAN_PATH = ROOT_DIR / "data" / "cleaned_dataset.csv"
SUMMARY_PATH = ROOT_DIR / "data" / "cleaning_summary.json"

MIN_LOCATION_COUNT = 10
IQR_K = 1.5


def parse_numeric_currency(value: object) -> float:
    """Extract numeric float from messy currency/unit text or return NaN."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)

    text = str(value).strip()
    if not text or text.lower() in {"negotiable", "n/a", "na", "none", "null", "-"}:
        return np.nan

    cleaned = re.sub(
        r"[৳,]|BDT|Tk\.?|sqft|sft|sq\.?\s*ft|/sft",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    return float(match.group(0)) if match else np.nan


def parse_parking_count(value: object) -> float:
    """Convert strings like '1 Parking(s)' or 'No Parking' into integer counts."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)

    text = str(value).strip().lower()
    if not text or text in {"nan", "none", "n/a", "na", "-"}:
        return np.nan
    if "no parking" in text or text in {"no", "none", "0"}:
        return 0.0

    match = re.search(r"\d+", text)
    return float(match.group(0)) if match else np.nan


def compute_iqr_mask(series: pd.Series, k: float = IQR_K, use_log: bool = False) -> pd.Series:
    """Return boolean mask for rows within [Q1 - k*IQR, Q3 + k*IQR]."""
    values = np.log1p(series) if use_log else series
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    if iqr <= 0:
        return pd.Series(True, index=series.index)
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return (values >= lower) & (values <= upper)


def preprocess_dataset(
    raw_path: Path = RAW_PATH,
    clean_path: Path = CLEAN_PATH,
    summary_path: Path = SUMMARY_PATH,
) -> pd.DataFrame:
    """Run full cleaning & preprocessing pipeline and persist outputs."""
    df_raw = pd.read_excel(raw_path, sheet_name="Raw_Data")
    raw_rows = int(len(df_raw))
    raw_cols = list(df_raw.columns)
    raw_nulls = {col: int(df_raw[col].isna().sum()) for col in raw_cols}
    raw_unique_locations = int(df_raw["Location"].dropna().astype(str).str.strip().nunique())

    df = df_raw.copy()

    # 1. Parse string-formatted currency, area, and room columns
    numeric_source_cols = [
        "Floor_area_sft",
        "Bedrooms",
        "Bathrooms",
        "Price_Per_Sft_Taka",
        "Estimated_Total_Price_Taka",
    ]
    for col in numeric_source_cols:
        df[col] = df[col].apply(parse_numeric_currency)
    df["Parking_Count"] = df["Parking"].apply(parse_parking_count)

    # Normalize categorical text columns
    for col in ["Location", "City"]:
        df[col] = df[col].astype(str).str.strip()
        df.loc[df[col].isin(["nan", "None", "NaN", "null", ""]), col] = np.nan

    negotiable_or_missing_price = int(df["Estimated_Total_Price_Taka"].isna().sum())
    missing_location = int(df["Location"].isna().sum())

    # 2. Remove rows with missing target price, missing location, or missing area
    df = df.dropna(subset=["Estimated_Total_Price_Taka", "Location", "Floor_area_sft"]).copy()
    after_dropna_rows = int(len(df))

    # Fill missing City from most frequent City for each Location
    loc_city_lookup = (
        df.dropna(subset=["City"])
        .groupby("Location")["City"]
        .agg(lambda s: s.mode().iloc[0])
        .to_dict()
    )
    df["City"] = df["City"].fillna(df["Location"].map(loc_city_lookup)).fillna("Unknown")

    # 3. Remove duplicate records (ignoring Property_ID and Source_URL scrape identifiers)
    dedup_subset = [
        "Location",
        "City",
        "Floor_area_sft",
        "Bedrooms",
        "Bathrooms",
        "Parking_Count",
        "Estimated_Total_Price_Taka",
    ]
    df = df.drop_duplicates(subset=dedup_subset, keep="first").copy()
    after_dedup_rows = int(len(df))

    # 4. Domain sanity bounds for residential apartments in Bangladesh
    df = df[
        (df["Floor_area_sft"] >= 450.0)
        & (df["Floor_area_sft"] <= 5500.0)
        & (df["Estimated_Total_Price_Taka"] > 0)
    ].copy()

    # Median imputation for minor missing / invalid room or parking counts
    imputed_counts: dict[str, int] = {}
    for col in ["Bedrooms", "Bathrooms", "Parking_Count"]:
        if col in {"Bedrooms", "Bathrooms"}:
            df.loc[df[col] <= 0, col] = np.nan
        missing_before = int(df[col].isna().sum())
        median_val = float(df[col].median())
        df[col] = df[col].fillna(median_val)
        imputed_counts[col] = missing_before

    df = df[
        (df["Bedrooms"] >= 1)
        & (df["Bedrooms"] <= 8)
        & (df["Bathrooms"] >= 1)
        & (df["Bathrooms"] <= 8)
        & (df["Parking_Count"] >= 0)
        & (df["Parking_Count"] <= 4)
    ].copy()

    # Derived unit rate used strictly for statistical anomaly detection (never exported to ML features)
    calc_pps = df["Estimated_Total_Price_Taka"] / df["Floor_area_sft"]
    df = df[(calc_pps >= 2500.0) & (calc_pps <= 28000.0)].copy()
    df["_calc_pps"] = df["Estimated_Total_Price_Taka"] / df["Floor_area_sft"]
    after_domain_rows = int(len(df))

    # 5. Statistical IQR (k=1.5) outlier removal on area & price columns
    # Global log-IQR on Floor_area_sft eliminates extreme mansion/tiny anomalies without truncating luxury hubs.
    df = df[compute_iqr_mask(df["Floor_area_sft"], k=IQR_K, use_log=True)].copy()

    # Per-location IQR (k=1.5) on implied price/sft & total price preserves genuine neighborhood tiers
    # (e.g., Gulshan 2 vs Mirpur), while removing anomalous listings within each frequent neighborhood.
    kept_indices: list[int] = []
    for _, grp in df.groupby("Location"):
        if len(grp) >= MIN_LOCATION_COUNT:
            mask = compute_iqr_mask(grp["_calc_pps"], k=IQR_K, use_log=False) & compute_iqr_mask(
                grp["Estimated_Total_Price_Taka"], k=IQR_K, use_log=False
            )
            kept_indices.extend(grp[mask].index.tolist())
        else:
            kept_indices.extend(grp.index.tolist())

    df = df.loc[kept_indices].copy()

    # Re-check sparse locations (<10 after per-location filtering) against dataset-wide log-IQR
    # bounds on unit price and total price so extreme non-hub anomalies do not corrupt "Other".
    post_loc_counts = df["Location"].value_counts()
    sparse_mask = df["Location"].map(post_loc_counts) < MIN_LOCATION_COUNT
    global_log_pps_mask = compute_iqr_mask(df["_calc_pps"], k=IQR_K, use_log=True)
    global_log_price_mask = compute_iqr_mask(
        df["Estimated_Total_Price_Taka"], k=IQR_K, use_log=True
    )
    df = df[(~sparse_mask) | (global_log_pps_mask & global_log_price_mask)].copy()
    after_iqr_rows = int(len(df))

    # 6. Dynamic Location Categorization: group <10 listings into "Other"
    final_loc_counts = df["Location"].value_counts()
    df["Location"] = df["Location"].where(
        df["Location"].map(final_loc_counts) >= MIN_LOCATION_COUNT,
        "Other",
    )

    # Round discrete integer structural columns cleanly
    for col in ["Bedrooms", "Bathrooms", "Parking_Count"]:
        df[col] = df[col].round().astype(int)

    # 7. Anonymize & prevent data leakage: drop IDs, URLs, titles, Raw_Sale_Price, and derived Price_Per_Sft
    output_columns = [
        "Location",
        "City",
        "Floor_area_sft",
        "Bedrooms",
        "Bathrooms",
        "Parking_Count",
        "Estimated_Total_Price_Taka",
    ]
    df_clean = df[output_columns].reset_index(drop=True)

    clean_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(clean_path, index=False)

    clean_unique_locations = int(df_clean["Location"].nunique())
    other_count = int((df_clean["Location"] == "Other").sum())
    summary = {
        "raw_rows": raw_rows,
        "raw_columns": raw_cols,
        "raw_null_counts": raw_nulls,
        "negotiable_or_missing_price_rows": negotiable_or_missing_price,
        "missing_location_rows": missing_location,
        "after_dropna_rows": after_dropna_rows,
        "after_dedup_rows": after_dedup_rows,
        "duplicates_removed": after_dropna_rows - after_dedup_rows,
        "after_domain_bounds_rows": after_domain_rows,
        "after_iqr_outlier_removal_rows": after_iqr_rows,
        "iqr_k": IQR_K,
        "min_location_count": MIN_LOCATION_COUNT,
        "imputed_missing_counts": imputed_counts,
        "cleaned_rows": int(len(df_clean)),
        "cleaned_columns": output_columns,
        "cleaned_null_counts": {col: int(df_clean[col].isna().sum()) for col in output_columns},
        "raw_unique_locations": raw_unique_locations,
        "cleaned_unique_locations": clean_unique_locations,
        "other_grouped_listings": other_count,
        "cleaned_unique_cities": int(df_clean["City"].nunique()),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 68)
    print("DATASET PREPROCESSING SUMMARY (PBazaar Real Estate Dataset)")
    print("=" * 68)
    print(f"Raw dataset rows              : {raw_rows:,}")
    print(f"Raw unique locations          : {raw_unique_locations}")
    print(f"Missing/Negotiable prices     : {negotiable_or_missing_price:,}")
    print(f"Missing locations             : {missing_location:,}")
    print(f"After dropna (price/loc/area) : {after_dropna_rows:,}")
    print(f"After deduplication           : {after_dedup_rows:,} (-{summary['duplicates_removed']})")
    print(f"After domain sanity bounds    : {after_domain_rows:,}")
    print(f"After IQR (k={IQR_K}) outliers  : {after_iqr_rows:,}")
    print(f"Final cleaned rows            : {len(df_clean):,}")
    print(f"Cleaned unique locations      : {clean_unique_locations} ('Other' count: {other_count})")
    print(f"Cleaned null counts           : {summary['cleaned_null_counts']}")
    print(f"Saved cleaned dataset to      : {clean_path}")
    print(f"Saved summary report to       : {summary_path}")
    print("=" * 68)

    return df_clean


if __name__ == "__main__":
    preprocess_dataset()
