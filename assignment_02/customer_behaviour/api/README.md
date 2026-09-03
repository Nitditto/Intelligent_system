# Customer-behaviour API — order-satisfaction prediction

FastAPI service that loads the persisted pipeline from
`customer_behaviour/model/model_pipeline.joblib` and predicts whether a completed
Olist order will receive a **positive review** (`review_score >= 4`).

```
raw order JSON ──▶ validation (pydantic) ──▶ build_features()  ──▶ fitted pipeline ──▶ JSON
                                            (stateless, no fit)   (ColumnTransformer:
                                                                   impute+log1p+scale,
                                                                   one-hot, TF-IDF)
                                                                   + LogisticRegression
```

The API **never fits** a scaler / encoder / vectoriser — it only calls `transform` /
`predict_proba` on the object trained in notebook section 22.

## Run — local

```bash
cd customer_behaviour
pip install -r requirements.txt
uvicorn api.main:app --port 8000
# docs / try-it:  http://localhost:8000/docs
```

## Run — Docker

`customer_behaviour/docker-compose.yml` builds this API **and** the nginx-served web
client:

```bash
cd customer_behaviour
docker compose up --build
#   web  -> http://localhost:5174   (its /api/* is proxied to the api container)
#   api  -> http://localhost:8000/docs
```

Just the API:

```bash
cd customer_behaviour
docker build -f api/Dockerfile -t cb-api .        # build context must be customer_behaviour/
docker run --rm -p 8000:8000 cb-api
```

The image installs `api/requirements.txt` (serving deps only — `scikit-learn==1.9.0`
pinned to the version that pickled `model_pipeline.joblib`) and copies `api/` + `model/`.
For a physical phone to reach the API, expose it with a tunnel — uncomment the `ngrok`
service in the compose file (`docker compose --profile tunnel up --build`).

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/predict` | score one order |
| POST | `/predict/batch` | score `{ "orders": [ ... ] }` |
| GET | `/questions` | form metadata — the web + mobile clients render their form from this |
| GET | `/samples` | `{ "defaults": {…typical values…}, "examples": [ {…order…, "_label", "_true_score"} ] }` — used by the web client's Quick mode and its "Load a real order" picker |
| GET | `/model-info` | chosen model, representation, feature order, seed |
| GET | `/healthz` | liveness + model name |

`api/samples.json` is a serving convenience (not graded). Regenerate it with
`python api/make_samples.py` (joins the CSVs in `../data/`, picks ~40 diverse real
orders). It is small (~30 KB) and committed so the API works out of the box.

## Appendix D — Web report template

| | |
|---|---|
| **Framework** | FastAPI (Uvicorn) |
| **Endpoint** | `POST /predict` |
| **Input** | order value & freight (BRL), payment method & instalments, item/seller counts, product category / weight / photos / description length, customer state, purchase date, promised delivery date, actual delivery date, review comment (PT, optional) |
| **Validation** | pydantic `OrderInput` — 5 required fields (`price_total`, `freight_total`, the 3 dates) all `>= 0`; `max_installments` in `1..24`; `customer_state` 2 chars; everything else optional |
| **Preprocessing** | loaded from training: `SimpleImputer(median)` → `log1p` (money) → `StandardScaler`; `OneHotEncoder(handle_unknown="ignore", min_frequency=30)`; `TfidfVectorizer(1–2-gram, min_df=5, max 20000)` on the cleaned comment |
| **Loaded model** | `LogisticRegression(class_weight="balanced")`, representation `tab+text`, seed 42, scikit-learn 1.9.0 |
| **Output** | `{"prediction","confidence","p_satisfied","threshold","signals","contributions","model","representation"}` |

### Example request

```bash
curl -s http://localhost:8000/predict -H "content-type: application/json" -d '{
  "price_total": 129.90, "freight_total": 18.30, "main_payment_type": "credit_card",
  "max_installments": 3, "customer_state": "SP", "category": "bed_bath_table",
  "product_weight_g": 1200, "product_desc_len": 850, "product_photos_qty": 3,
  "order_purchase_timestamp": "2018-05-01 10:00:00",
  "order_estimated_delivery_date": "2018-05-20 00:00:00",
  "order_delivered_customer_date": "2018-05-31 14:00:00",
  "review_comment_message": "Produto chegou muito atrasado e a embalagem estava danificada."
}'
```

### Example response

```json
{
  "prediction": "dissatisfied",
  "confidence": 0.9252,
  "p_satisfied": 0.0748,
  "threshold": 0.5,
  "signals": {"days_vs_promise": 11.6, "late": true, "has_comment": true,
              "delivery_days": 30.2, "category_group": "bed_bath_table",
              "customer_region": "Southeast"},
  "contributions": {
    "base_p": 0.6309, "final_p": 0.0748, "dataset_base_rate": 0.79,
    "items": [
      {"label": "Delivered 12 days late", "kind": "tabular", "effect": -0.47},
      {"label": "comment: \"atrasado\"",   "kind": "text",    "effect": -0.09},
      {"label": "Category: bed_bath_table","kind": "tabular", "effect": -0.17}
    ],
    "other_effect": -0.23,
    "note": "bars are log-odds contributions; reference = the model's average order"
  },
  "model": "LogisticRegression", "representation": "tab+text"
}
```

### `contributions` — the explanation

An **exact linear-SHAP decomposition** (no `shap` library). For the deployed
`LogisticRegression`, with `model/feature_means.joblib` giving the mean of every
transformed feature over the fit set:

```
phi_j   = coef_j * (x_j - mean_j)              # per-feature log-odds pull
z_base  = intercept + coef . mean              # sigmoid(z_base) = base_p
sigmoid(z_base + sum_j phi_j) == final_p       # holds exactly
```

`base_p` (~0.63) is the model's neutral point — lower than the 0.79 dataset positive
rate because training used `class_weight="balanced"`. `items` are the top ~12 factors
by |effect| with human labels derived from the raw order (`delivery_delay_days > 0`
→ `"Delivered 12 days late"`, one-hot → `"Category: …"`, `txt__<tok>` →
`comment: "<tok>"`); the rest are folded into `other_effect`. `contributions` is
`null` if the deployed model is not linear.
