# Assignment 02 — Application 2: House-Price Prediction

**Intelligent System Development · From Data Representation to a Deployable Intelligent System**
Lecturer: Dinh Que Tran, Ph.D., Assoc. Prof. · Semester I.2026

This is the **house-price** application — Application 2 of the Assignment 02 submission.
It follows the official spec (`slide_assign/intel_sys_dev_assignment_02_final.pdf`):
Appendix A repository layout, Appendix B 23-section notebook, Appendix C `POST /predict`.

Parent folder `Assignment_2/` is the Assignment 02 repo and also holds `diabetes/` and
`customer_behavior/` (Applications 1 and 3), plus `report/` — see `../README.md`.

## Pipeline

```
Data → Understand → Clean → Represent → Learn → Evaluate → Persist → Deploy (Web + Mobile)
```

This is a **regression** task: `X` = property attributes, `y` = listing price
(million VND). It differs from the diabetes classification app in target type
(continuous, not class), loss (error-based), and metrics (MAE / MSE / RMSE / R²,
no confusion matrix).

## Structure (Appendix A)

```
Assignment_2/house_price/
├── data/
│   ├── README.md                     dataset source, columns, known quality issues
│   └── VN-real-estate-Apr-Sept-2025.csv   (gitignored — 210 MB; see data/README.md)
├── notebook/
│   └── house_price.ipynb             the 23-section experiment (Appendix B)
├── model/
│   ├── input_schema.json             committed input contract
│   ├── model_pipeline.joblib         produced by notebook §22 (gitignored)
│   └── feature_names.joblib          produced by notebook §22 (gitignored)
├── api/                              FastAPI service — POST /predict (Appendix C)
├── web/                              Vite + React + TS client (Appendix D)
├── mobile/                           Flutter (Android + iOS) native client (Appendix E)
├── requirements.txt
└── README.md                        (this file)
```

## Dataset

| Field | Value |
|---|---|
| Name | VN Real Estate Listings (April–September 2025) |
| File | `data/VN-real-estate-Apr-Sept-2025.csv` — 236,226 rows × 28 cols, ~210 MB |
| Encoding | UTF-8 **with BOM** (`encoding="utf-8-sig"`), separator `,` |
| Target | `Price`, in **million VND** ("2.7 tỷ" ↔ `Price` = 2700) |

Full column roles and the data-quality issues (corrupted `Price`/`Area`, 72–84% missing
`Bedrooms`/`Bathrooms`/`Floors`, constant `VIP Account`, `Title` leaks the target) are in
[`data/README.md`](data/README.md).

An earlier version used the ~30k-row Kaggle `vietnam_housing_dataset.csv`; it and its
trained artifacts were removed when the dataset was switched to this larger scrape.

## Reproduce

### 1. Environment

```bash
python -m venv .venv && source .venv/bin/activate     # Python 3.14 (see notebook §0)
pip install -r requirements.txt
```

`RANDOM_SEED = 42` throughout. Exact library versions are printed in notebook section 0.

### 2. Data

```bash
cp ../../VN-real-estate-Apr-Sept-2025.csv data/       # if not already present
```

### 3. Notebook

Open `notebook/house_price.ipynb` and **Run All**. It runs the full pipeline
(load → clean → represent → 5 regression models → evaluate on the held-out test set →
persist) and writes `model/model_pipeline.joblib`, `model/feature_names.joblib` and
`model/input_schema.json`. Section 23 reloads the artifact from disk and turns one raw
input dict into a JSON prediction — the contract the API depends on.

### 4. API

```bash
# from Assignment_2/
uvicorn house_price.api.main:app --reload --port 8002
# Swagger UI: http://localhost:8002/docs
```

```bash
curl -s -X POST http://localhost:8002/predict -H 'content-type: application/json' \
  -d '{"Area": 78.7, "Bedrooms": 3, "Bathrooms": 2, "Floors": 2, "Property Type": "Nhà riêng", "Province": "an-giang", "district": "Rạch Giá"}'
# -> { "predicted_price": 2381.21, "price_per_m2": 30.26, "formatted_price_billion": "2.38 tỷ VNĐ", "model_name": "RandomForestRegressor" }
```

See [`api/README.md`](api/README.md) for the full endpoint list and request shape.
For **Postman**, import `api/house_price_api.postman_collection.json` (set `baseUrl` to `http://localhost:8002`).

### 5. Web Client

```bash
cd web
npm install
npm run dev          # http://localhost:5173
```

Built fresh with Vite + React 18 + TypeScript, connecting directly to FastAPI `POST /predict` and `GET /healthz`.

### 6. Mobile Client (Flutter)

```bash
cd mobile
flutter pub get
flutter run          # Runs on Android emulator, iOS simulator, or connected device
```

The mobile client is a native **Flutter** cross-platform application (Material 3). Training happens only in the notebook; the mobile app is a pure REST client (**Training ≠ Inference**). Tap the server status badge in the top-right corner to point the app to `http://10.0.2.2:8002` (Android emulator) or your host machine's Wi-Fi LAN IP.
