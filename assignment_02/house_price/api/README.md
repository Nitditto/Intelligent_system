# House-Price Prediction API

FastAPI service (Appendix C: `POST /predict`) that wraps the persisted sklearn
pipeline from `notebook/house_price.ipynb`.

```
JSON input → validation (schema.py) → saved preprocessing (model_pipeline.joblib) → regressor → JSON output
```

The service **loads** the preprocessing pipeline learned from the training data — it
never fits a new scaler / imputer / encoder on request data (Appendix §15, "Important:
Data Leakage").

## Prerequisites

1. `pip install -r ../requirements.txt`
2. `notebook/house_price.ipynb` has been run through **section 22**, producing:
   - `model/model_pipeline.joblib` — the full `Pipeline` (`ColumnTransformer` + regressor)
   - `model/feature_names.joblib` — ordered raw feature columns
   - `model/input_schema.json` — the input contract (committed)

Until then, `/healthz` returns `model_loaded: false` and `/predict` returns `503`.

## Run

```bash
# from Assignment_2/house_price/
python -m uvicorn api.main:app --reload --port 8002
```

Swagger UI: http://localhost:8002/docs

## Testing with Postman

Import `api/house_price_api.postman_collection.json` into Postman. It has a `{{baseUrl}}`
collection variable (default `http://localhost:8002`) and five requests: health check,
full-input predict, minimal predict (`Area` only), an apartment example, and a
missing-`Area` request that returns `422` (use it as the input-validation evidence for
the report). Start the API first, then Send each request.

| Method / path | Purpose |
|---|---|
| `POST /predict` | property attributes → `{ "predicted_price": <million VND>, "price_per_m2": ..., "model_name": ... }` |
| `GET /healthz` | `{ status, model_loaded, model_name }` |

## Example

```bash
curl -s -X POST http://localhost:8002/predict \
  -H 'content-type: application/json' \
  -d '{
        "Area": 78.7, "Width": 4.0, "Bedrooms": 3, "Bathrooms": 2, "Floors": 2,
        "Property Type": "Nhà riêng", "Position": "Đường chính", "Direction": "Nam",
        "Road Type": "Đường nhựa", "Province": "an-giang",
        "Agent Role": "Chính chủ", "ward": "Phường An Hòa", "district": "Rạch Giá"
      }'
# -> {"predicted_price": 2731.4, "price_per_m2": 34.71, "currency": "million VND", "model_name": "XGBoost"}
```

Only `Area` is required. Every other field may be omitted or `null`; the saved
pipeline imputes it the same way it did during training.

## Files

| File | Role |
|---|---|
| `main.py` | app, routes (`/predict`, `/healthz`), CORS, startup model warm-up |
| `schema.py` | `ListingInput` (request) / `PredictionOut` (response) Pydantic models |
| `inference.py` | loads `model_pipeline.joblib`, builds the 1-row frame, `expm1` back-transform |
| `config.py` | artifact paths |

## TODO (after the notebook rewrite)

- Sync `schema.ListingInput` field names + categorical `Literal` sets with the final
  feature list from notebook section 13 and `model/input_schema.json`.
- Confirm `inference._TARGET_IS_LOG1P` matches the notebook's target transform (§17).
- The `web/` and `mobile/` clients still call the **old Node API shape**
  (`/api/predict`, camelCase, `/api/meta`, `/api/result/:id`, SHAP). They need
  rewiring to this endpoint (or add compatibility routes here). Tracked in `web/README.md`.
