# Real Estate Price Prediction System

> **DIU SE331 Software Engineering Design Capstone Project**  
> Machine Learning-Driven Residential Property Valuation Web Application for Bangladesh

---

## Project Overview

The **Real Estate Price Prediction System** addresses pricing opacity in the Bangladesh residential property market by training and serving a zero-leakage scikit-learn regression pipeline on anonymized historical apartment listings (`pbazaar_apartments_raw.xlsx`).

Users can search 35 neighborhood tiers (`Dynamic Location Categorization` grouping sparse locations `<10` into `"Other"`), validate structural inputs, receive instant BDT (`৳` & Lakh/Crore) price estimates with `<500ms` API latency, inspect boundary extrapolation warnings (FR-5), and review model benchmarks and Exploratory Data Analysis (EDA) visualizations.

---

## Project Structure

```text
Real-Estate/
├── app.py                        # FastAPI backend API & security middleware
├── requirements.txt              # Pinned Python dependencies
├── README.md                     # Documentation & reproduction guide
├── .gitignore                    # Secret, environment & cache exclusions
├── Real-Estate-proposal.pdf      # DIU SE331 Capstone Design Proposal
├── data/
│   ├── pbazaar_apartments_raw.xlsx  # Raw PBazaar apartment dataset (2,393 rows)
│   ├── cleaned_dataset.csv          # Cleaned, anonymized ML dataset (1,843 rows)
│   └── cleaning_summary.json        # Structured before/after cleaning metrics
├── scripts/
│   ├── preprocess.py             # Data cleaning, IQR (k=1.5), & location grouping
│   ├── eda.py                    # Matplotlib/Seaborn EDA chart generator
│   └── train_model.py            # 4-model tuning, evaluation & serialization
├── models/
│   ├── best_model.pkl            # Serialized winning scikit-learn Pipeline
│   └── model_metadata.json       # Metrics, schema, locations & training bounds
├── templates/
│   └── index.html                # Responsive dashboard UI (with security meta tags)
├── static/
│   ├── css/style.css             # Modern architectural dashboard styling
│   ├── js/app.js                 # Searchable combobox, validation & fetch client
│   └── charts/                   # Generated EDA visualizations (PNG)
└── tests/
    └── test_api.py               # Automated API, security, validation & leakage tests
```

---

## Data Preprocessing & Anti-Leakage Design (`scripts/preprocess.py`)

| Stage | Row Count | Notes |
| --- | --- | --- |
| Raw Scraped Dataset (`Raw_Data`) | 2,393 | 94 unique locations; 202 `Negotiable`/missing prices; 30 missing locations |
| After Missing Target/Location/Area Removal | 2,171 | Modal `Location -> City` fill applied for minor missing cities |
| After Deduplication | 2,107 | Removed 64 duplicate records ignoring `Property_ID` and `Source_URL` |
| After Domain Sanity Bounds & Imputation | 2,030 | Parsed currency/`Parking_Count`, median-imputed minor room/parking gaps |
| After Statistical IQR (`k=1.5`) Filtering | **1,843** | Global log-IQR on area + per-location IQR on frequent areas + dataset log-IQR on sparse areas |

- **Dynamic Location Categorization:** Locations with `<10` cleaned listings are grouped into `"Other"`, reducing 94 raw locations to **35** balanced categories (`177` listings in `"Other"`).
- **Anonymization & Strict Anti-Leakage:** `Property_ID`, `Source_URL`, `Title`, `Raw_Sale_Price`, and `Price_Per_Sft_Taka` are strictly excluded from the ML feature matrix `X`.

---

## Model Benchmark Results (`scripts/train_model.py`)

Evaluated using an 80/20 holdout split (`train_test_split(random_state=42)`) and 5-fold shuffled cross-validation across identical `ColumnTransformer` preprocessing (`SimpleImputer(median)` + `StandardScaler()` for numeric features; `OneHotEncoder(handle_unknown="ignore")` for categoricals):

| Model | Test R² | 5-Fold CV R² | MAE (BDT) | MAE (% of Mean) | RMSE (BDT) | Status |
| --- | --- | --- | --- | --- | --- | --- |
| **GradientBoostingRegressor** | **0.8542** | **0.8186** | **৳ 2,166,029** | **20.07%** | **৳ 3,369,264** | **Selected Best** |
| RandomForestRegressor | 0.8378 | 0.7991 | ৳ 2,336,824 | 21.65% | ৳ 3,553,999 | Benchmarked |
| DecisionTreeRegressor | 0.8224 | 0.7460 | ৳ 2,415,518 | 22.38% | ৳ 3,718,700 | Benchmarked |
| LinearRegression | 0.7799 | 0.7812 | ৳ 2,555,398 | 23.67% | ৳ 4,140,223 | Baseline |

> **Transparency Note on MAE Dispersion:** Because raw public listings omit unrecorded building micro-attributes (road width, building age, floor number, developer tier, sub-block), identical `(Location, City, Floor_area_sft, Bedrooms, Bathrooms, Parking_Count)` listings naturally vary in asking price, bounding achievable leak-free MAE on this dataset.

---

## API Endpoints & Security Hardening (`app.py`)

- `GET /`: Serves the interactive valuation dashboard.
- `GET /api/locations`: Returns dynamically modeled locations, cities, location-to-city mapping, neighborhood stats, and training numeric bounds.
- `GET /api/metrics`: Returns 4-regressor benchmark metrics and dataset preprocessing stats.
- `POST /api/predict`: Validates inputs (returns **HTTP 400** on zero/negative area or rooms, empty location, or malformed JSON), flags out-of-bounds extrapolation warnings (FR-5), formats output in BDT (`৳`), Lakh/Crore, and implied `৳/sqft`, and measures inference latency (`<500ms`).
- `GET /api/health`: Lightweight health check.

**Implemented Security Controls:**
- Strips `Server` / `X-Powered-By` headers (`server_header=False`).
- Sets `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`, `X-XSS-Protection: 1; mode=block`, `Referrer-Policy`, `Strict-Transport-Security`, and `Permissions-Policy`.
- Enforces a strict **100 KB** request payload cap (`HTTP 413`) and restricted CORS (`GET`, `POST`).

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run preprocessing, EDA chart generation, and model training
python scripts/preprocess.py
python scripts/eda.py
python scripts/train_model.py

# 3. Run automated test suite
pytest -v

# 4. Launch the web application
uvicorn app:app --host 127.0.0.1 --port 8000 --no-server-header
```

Open `http://127.0.0.1:8000` in your browser.

---

## Disclaimer (Proposal Section 5.2)

All predictions generated by this application are statistical estimates based on anonymized historical listing data and are intended solely for academic and informational support. They do **not** constitute a certified legal appraisal, licensed property survey, or binding financial valuation.
