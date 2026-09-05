# Report guide — Application 2: House Price Prediction

What to write for the **house price** part of the Assignment 02 report (Part A §9,
plus the shared Reproducibility, Web-app and Mobile-app templates). Each subsection
lists the points to cover and where the material already exists. Screenshot
placeholders are marked `📸` with a caption describing exactly what the image must
show — replace each with the actual image when you take it.

Source material:
- `notebook/house_price.ipynb` — the 23-section ML experiment (executed, with outputs).
- `data/README.md` — dataset provenance, column roles, known data-quality issues.
- `api/README.md`, `mobile/README.md` — how each part runs.
- Key executed numbers are quoted below so the report can cite them directly.

---

## 9.1 Problem description

- Template sentence: *"The objective of this application is to predict the listing
  price of a property (in million VND) based on **its physical characteristics
  (area, dimensions, room counts), its location (province/district), and listing
  metadata**. The prediction target is **`Price`**. The prediction can potentially
  support **buyers/sellers sanity-checking an asking price, or an agent triaging
  new listings**."*
- Define `X` = property characteristics (area, width, length, bedrooms, bathrooms,
  floors, alley width, property type, position, direction, road type, province,
  district, agent role/listing count); `y` = `Price` (million VND).
- State this is a **regression** problem, and explain why it differs from the
  diabetes classification task: the target is continuous and unbounded rather than
  a class label, so the loss function (squared/absolute error, not cross-entropy)
  and the evaluation metrics (MAE/RMSE/R², not accuracy/F1) are different, and there
  is no notion of a confusion matrix.
- Source: notebook §1.

## 9.2 Dataset

- Name: **VN Real Estate Listings (April–September 2025)**, file
  `VN-real-estate-Apr-Sept-2025.csv` (scraped Vietnamese property-listing portal;
  research use only — cite the source portal in the report).
- Observations: **236,226** raw rows → **201,654** after de-duplication and
  domain-validity filtering (93.1% kept; see §9.3).
- Features: **28** raw columns → **18** retained as model input (7 numeric + 3
  missing-indicator flags + 7 categorical, after dropping leakage/free-text/constant
  columns).
- Target: `Price` (million VND). Title text such as *"2.7 tỷ"* → `Price = 2700`
  (1 tỷ = 1,000 million VND).
- Numerical features: `Area`, `Width`, `Length`, `Bedrooms`, `Bathrooms`, `Floors`,
  `Alley Width`, `Agent Listing Count`.
- Categorical features: `Property Type` (7 levels), `Position`, `Direction`,
  `Road Type`, `Province` (63 levels), `Agent Role`, plus `district` (2,082 levels,
  engineered from `Location`).
- Dropped columns and why: `Title` (**leaks `Price`** — literally contains the price
  text), `Listing ID` (identifier, used only for de-dup), `Description` (free text,
  unused), `VIP Account` (constant), `Agent Name` (28k near-unique), `Avatar`,
  `Property Type Slug` (duplicate of `Property Type`), timestamp columns.

## 9.3 Data Understanding and Cleaning

Report and interpret (notebook §4–§9):
- Raw shape `(236226, 28)`.
- Missing values (top offenders): `Bathrooms` 83.8%, `Floors` 79.4%, `Bedrooms`
  72.1%, `Alley Width` 69.9%, `Direction` 69.6%, `Latitude`/`Longitude` 67.9%,
  `Road Type` 67.1%, `Length` 58.2%, `Position` 40.5%, `Width` 26.3%, `Area` 0.1%.
  → strategy: numeric columns get a **missing-indicator flag + median impute inside
  the pipeline** (fitted on train only); `Latitude`/`Longitude` are dropped (too
  sparse to be useful); categorical missing → `"Unknown"` category.
- Duplicates: 0 exact duplicate rows, but **19,512 duplicate `Listing ID` values**
  (re-scrapes of the same listing) → kept the first occurrence, `N`: 236,226 →
  216,714.
- Invalid values: `Price <= 0` (608 rows), `Price > 200,000` million VND / ~200 tỷ
  (10,486 rows, implausible outliers/typos), `Area <= 0` (4 rows), `Area > 10,000 m²`
  (4,285 rows), `Floors < 0` (9 rows) → **dropped**. `N`: 216,714 → **201,654**
  (93.1% kept).
- Skew: raw `Price` skew **3.36** → `log1p(Price)` skew **−0.07** (the model is
  trained on `log1p(Price)`, predictions are inverted with `expm1`). `Area` skew
  **5.11** → `log1p(Area)` skew **1.32**; `Width`/`Length` even more skewed
  (**380.5** / **295.3** raw) — engineered features use the log-transformed values.
- Outliers: IQR-rule flags a large fraction of rows on `Price` (24,952), `Area`
  (30,345) and the room-count columns — **not removed**, because in this domain
  (land + houses + apartments + warehouses mixed together) a "high area, high price"
  row is usually a genuine large property, not noise; explain this in the report as
  a deliberate keep-not-drop decision, contrasted with the diabetes app's outlier
  handling.
- 📸 **Screenshot: notebook §4** — `df.info()` / `df.describe()` showing the raw
  236,226 × 28 shape and dtypes.
- 📸 **Screenshot: notebook §5** — the data-quality issue → count → planned-action
  table.

## 9.4 Representation

Show the transformation chain with concrete numbers (notebook §12–§15):
```
Raw CSV → DataFrame → Clean Data (201,654 rows) → Encoded/Scaled Features → X
```
- One raw listing (selected columns) and the resulting feature vector — quote the
  worked example from notebook §12: `{'Title': 'Bán nhà 192m² 2 tỷ ...', 'Price':
  2000.0, 'Area': 192.0, 'Property Type': 'Nhà riêng', 'Location': 'Phường Long
  Xuyên,An Giang(Mới)'}` → `x = [-0.223, -0.005, -0.002, ..., 1., 0., 0., ...]`
  (first 15 of `d` dims shown).
- DataFrame shape after cleaning: `(201654, 31)`. Final **raw** feature count before
  encoding: **18** (11 numeric incl. missing-indicators, 7 categorical incl.
  `district`).
- Train/val/test split (70/15/15, seed 42): train `(141157, 18)`, val
  `(30248, 18)`, test `(30249, 18)`.
- **After** the preprocessing pipeline (`ColumnTransformer`: median impute +
  `StandardScaler` on 11 numeric cols; `OneHotEncoder(handle_unknown="infrequent_if_exist",
  min_frequency=...)` on 7 categorical cols, rare levels folded into an
  `<infrequent>` bucket): preprocessed train shape **`(141157, 404)`** — i.e.
  **`d = 404`** (11 numeric + 393 one-hot indicator columns, dominated by the
  `Province`/`district` cardinality).
- Report `X ∈ ℝ^{N×404}`, `N = 201,654` overall (141,157 train), `y ∈ ℝ^{N}`
  (`log1p(Price)`, million VND). One API request = `ℝ^{1×404}` after the same
  pipeline transform.
- 📸 **Screenshot: notebook §12** — the raw-record → feature-vector printout and the
  four shape/dtype lines.

## 9.5 Exploratory Data Analysis

At least three meaningful plots (notebook §10, 9 sub-plots total); for each give
Observation / Interpretation / ML implication (already written in the notebook):
- **Price distribution** — raw `Price` is heavily right-skewed (skew 3.36); the
  `log1p` transform is close to symmetric (skew −0.07) → justifies training on
  `log1p(Price)`.
- **Area vs price relationship** — pooled Pearson `r(log Area, log Price) = 0.017`
  (near zero!) — but this is **confounded by `Property Type`**: per-type slopes
  range from `0.99` (Văn phòng/office, n=384) down to `0.14` (Đất/land, n=102,636)
  and even negative for hotels — i.e. area only predicts price *within* a property
  type, and land listings (the largest group) barely show the relationship at all.
  This is one of the most important findings to discuss.
- **Property-type distribution** — `Đất` (land) dominates at 102,636 rows (43%),
  followed by `Nhà riêng` (house, 67,790) and `Căn hộ chung cư` (apartment, 24,860).
- **Correlation with `log Price`** — numeric features are all weak: `Bathrooms`
  0.215 is the strongest, everything else (`Area`, `Bedrooms`, `Floors`, `Width`,
  `Length`) is below 0.05. Categorical association (correlation ratio η²):
  `Province` 0.237, `Property Type` 0.120 — **location and property type matter
  more than the numeric physical attributes**, which is the central EDA finding
  and explains why the final model leans on the 404-dim one-hot representation.
- **Missingness heatmap** — confirms the `Bathrooms`/`Floors`/`Bedrooms` columns are
  missing on 70–84% of rows (mostly land listings, which have no rooms).
- 📸 **Screenshot: notebook §10.1** — Price/log-Price distribution + property-type
  bar chart.
- 📸 **Screenshot: notebook §10.2** — log-Area vs log-Price scatter (pooled and
  split by property type) + the numeric/categorical correlation bar charts.

## 9.6 Regression Models

Five models compared (notebook §17), trained on a **50,000-row training subsample**
(the full 141k-row training pipeline with 404-dim one-hot features is too slow to
fit five models repeatedly; the winner should ideally be refit on the full training
set the same way the diabetes app refits its winner — confirm this was done before
reporting the final numbers):
- Linear Regression
- Ridge Regression
- Decision Tree Regressor
- Random Forest Regressor
- **XGBoost Regressor** (the assignment's suggested 5th model is Gradient Boosting;
  XGBoost is the gradient-boosting implementation used here — name it explicitly as
  the "Gradient Boosting Regressor" required by the assignment and justify the
  substitution: same family, better default performance/speed).

Training time (50,000-row subsample): Linear 1.06s, Ridge 0.85s, DecisionTree 1.91s,
RandomForest 28.98s, XGBoost 1.96s.

## 9.7 Evaluation

Validation-set comparison (notebook §18):

| Model | MAE | MSE | RMSE | R² | MAPE% | Train (s) |
|---|---|---|---|---|---|---|
| RandomForest | 10,810.59 | 7.48e8 | 27,348.89 | **0.183** | 311.75 | 28.98 |
| XGBoost | 10,793.79 | 7.55e8 | 27,478.05 | 0.176 | 322.22 | 1.96 |
| DecisionTree | 11,522.09 | 7.84e8 | 27,991.80 | 0.144 | 349.13 | 1.91 |
| LinearRegression | 12,149.13 | 8.90e8 | 29,830.37 | 0.028 | 375.27 | 1.06 |
| Ridge | 12,152.93 | 8.92e8 | 29,860.10 | 0.026 | 375.59 | 0.85 |
| *baseline (predict median)* | *14,397.21* | *1.04e9* | *32,184.83* | *−0.131* | *398.00* | – |

Chosen model: **Random Forest** — held-out **test** set result (notebook §19):
**MAE 10,626.40**, **MSE 7.22e8**, **RMSE 26,872.78**, **R² 0.186**, **MAPE 349.97%**
(all monetary units in million VND).

- **Report and be honest about the low R² (≈0.19).** Explain what each metric means
  here: MAE ≈ 10.6 billion VND typical absolute error is large relative to the
  median price (~3–5 tỷ), and MAPE is inflated by very cheap listings. Tie this back
  to §9.5: `Province`/`district`/`Property Type` explain more variance than the
  numeric attributes, and 70–84% missingness on room counts removes most of the
  physical-size signal the model could otherwise use — the ceiling on accuracy is a
  **data-quality limitation of the scraped source**, not primarily a modelling
  choice. Contrast this candidly against Diabetes (ROC-AUC 0.81) and Customer
  behaviour (ROC-AUC 0.86) in §11/§17 of the final report — the assignment rewards
  correctly *diagnosing* why a model underperforms, not only reporting good numbers.
- Error analysis (notebook §20): the 10 worst absolute errors are all
  extremely-high-priced listings (~195,000–199,000 million VND, i.e. ~195–199 tỷ)
  that the model under-predicts by ~98% — the random forest regresses toward the
  bulk of the (much cheaper) training distribution for these rare luxury listings.
- 📸 **Screenshot: notebook §18** — the full model-comparison table.
- 📸 **Screenshot: notebook §19–§20** — the test-set metrics, the predicted-vs-actual
  scatter/residual plot, and the worst-10-errors table.

## 9.8 Model Selection

Justify Random Forest for deployment (notebook §21):
- **Predictive performance** — best R²/MAE/RMSE among the five, narrowly ahead of
  XGBoost.
- **Interpretability** — feature importances available directly; comparable
  transparency to XGBoost, clearly ahead of an opaque ensemble alternative.
- **Computational cost** — the slowest to train (29s vs ~2s for XGBoost) but
  inference-time cost is what matters for deployment, and both are sub-millisecond
  per row.
- **Robustness** — bagging tolerates the heavy missingness/outliers documented in
  §9.3 without needing extra outlier removal.
- **Deployment constraints** — persisted pipeline is **~46 MB** (`model_pipeline.joblib`,
  mostly the 404-dim one-hot `ColumnTransformer` + forest); no GPU or extra runtime
  dependency beyond scikit-learn.
- Note the near-tie with XGBoost (R² 0.183 vs 0.176 on validation) — mention this as
  a legitimate alternative if training time or model size becomes a constraint.

## 9.9 Deployment (Part XI/XII + Appendix D/E)

- Architecture (same as the other two apps):
  ```
  User input → API request → validation → same preprocessing → saved model → prediction → result
  ```
- The service loads `model/model_pipeline.joblib` (`ColumnTransformer` + Random
  Forest regressor) and never re-fits it; predictions are trained on
  `log1p(Price)` and inverted with `expm1` before being returned.
- Inference flow: raw listing dict (`Area`, `Width`, `Length`, `Bedrooms`,
  `Bathrooms`, `Floors`, `Alley Width`, `Property Type`, `Position`, `Direction`,
  `Road Type`, `Province`, `Agent Role`, `Agent Listing Count`, `Location`) →
  `schema.py` validation → district parsed from `Location` → pipeline → `expm1` →
  `{ "predicted_price": ..., "price_per_m2": ..., "currency": "million VND",
  "model": "RandomForest" }`.
- Example (notebook §23 inference test, reloaded from disk):
  ```json
  { "predicted_price": 2540.82, "price_per_m2": 32284.83,
    "currency": "million VND", "model": "RandomForest" }
  ```
- Web framework: **FastAPI**, `POST /predict` (`house_price/api/main.py`), also
  `GET /healthz` (reports `model_loaded: false` / HTTP 503 until the pipeline files
  exist — good practice to mention as a deployment safeguard). Client: **React +
  TypeScript + Vite**. Mobile: **Flutter** (Provider state management, Material 3).
- ⚠️ **Port mismatch to fix before writing the report:** `api/README.md` runs the
  service on **port 8001**; `mobile/README.md`'s default API host targets **port
  8002**. Confirm which port the web app's `VITE_API_BASE`/`.env` actually points at
  and align all three (API run command, web env var, mobile `api_config.dart`)
  before taking screenshots, otherwise the demo will show a connection error.
- `web/README.md` is still the default Vite template (never customized) — write a
  short real one (framework, `npm run dev` port, env var) before submission, mirroring
  `diabetes/web/README.md` / `customer_behaviour/web/README.md`.

### 9.9.1 Web application (Appendix D template)

```
Web Application — House Price Prediction
Framework:  React + TypeScript + Vite  +  FastAPI (POST /predict)
Endpoint:   POST http://<host>:8001/predict
Input:      Area, Width, Length, Bedrooms, Bathrooms, Floors, Alley Width,
            Property Type, Position, Direction, Road Type, Province, Location
Output:     { predicted_price, price_per_m2, currency, model }
```

- 📸 **Screenshot W1 — Input form filled in.** The multi-field `PredictionForm`
  with a realistic listing entered (area, rooms, property type, province).
  Caption: *the input interface for the property characteristics `X`.*
- 📸 **Screenshot W2 — Prediction result.** The `ResultCard` showing the predicted
  price and price-per-m². Caption: *the predicted price returned by the same
  `RandomForest` pipeline trained in the notebook.*
- 📸 **Screenshot W3 (optional) — Error/validation state.** `ErrorBanner` triggered
  by an invalid input (e.g. missing `Area`), demonstrating server-side validation.
- 📸 **Screenshot W4 (optional) — FastAPI `/docs`.** Swagger UI or the Postman
  collection (`api/house_price_api.postman_collection.json`) showing a successful
  `POST /predict` call — this collection already has 5 ready-made requests including
  a deliberate `422` validation-failure case, useful evidence for the "input
  validation" requirement.

### 9.9.2 Mobile application (Appendix E template)

```
Mobile Application — House Price Prediction
Framework:  Flutter (Provider state management, Material 3)
Platform:   Android / Web (flutter run -d chrome for a quick demo without an emulator)
API:        POST http://<host>:<port>/predict
```

- 📸 **Screenshot M1 — Home / input screen.** `home_screen.dart` with the preset
  bar and property fields filled in. Caption: *mobile input screen; same fields as
  the web app.*
- 📸 **Screenshot M2 — Result.** `result_card.dart` showing the predicted price
  (Tỷ/Triệu VNĐ formatting) and the server-status badge. Caption: *the prediction
  displayed on device, formatted in Vietnamese currency units.*
- 📸 **Screenshot M3 (optional) — Settings modal.** `settings_modal.dart` showing
  the configurable API host — useful evidence that the mobile app is a thin REST
  client pointed at a separate server, not running the model on-device.
- Also capture evidence the app calls the API: the API terminal log line for the
  `POST /predict` request triggered from the phone/emulator/Chrome.

---

## Reproducibility (shared section — house price entries)

| Item | Value |
|---|---|
| Python | 3.14.6 (Anaconda), Windows 11 |
| Key libraries | numpy 2.4.6, pandas 3.0.3, scikit-learn 1.9.0, matplotlib 3.11.0, xgboost 3.3.0 (notebook); FastAPI (API); React 18 + TypeScript + Vite (web); Flutter 3.47 (mobile) |
| Random seed | `RANDOM_SEED = 42` (numpy + `random`; every split / subsample / estimator) |
| Dataset source | scraped Vietnamese property-listing portal, `VN-real-estate-Apr-Sept-2025.csv` (~210 MB, **not committed** — see `data/README.md` for how to obtain it; place at `house_price/data/VN-real-estate-Apr-Sept-2025.csv`) |
| Preprocessing | `ColumnTransformer`: missing-indicator + median `SimpleImputer` + `StandardScaler` on 11 numeric cols; `OneHotEncoder(handle_unknown="infrequent_if_exist")` on 7 categorical cols (incl. engineered `district`); fitted on train only |
| Feature representation | 18 raw features → `X ∈ ℝ^{N×404}` after encoding; target `log1p(Price)`, inverted with `expm1` at inference |
| Train/val/test split | 70/15/15 (141,157 / 30,248 / 30,249), seed 42, duplicate `Listing ID`s and invalid rows dropped before the split |
| Model hyperparameters | RandomForestRegressor (full list: notebook §17); trained on a 50,000-row subsample, **confirm/refit on the full training set before final deployment**, matching the diabetes app's protocol |
| Evaluation metrics | test: MAE 10,626.40 · RMSE 26,872.78 · R² 0.186 · MAPE 349.97% (million VND) |
| Saved pipeline | `model/model_pipeline.joblib` (preprocessing + RandomForestRegressor) |
| Saved model config | `model/feature_names.joblib`, `model/input_schema.json` |
| API code | `house_price/api/` (FastAPI) |
| Web app code | `house_price/web/` (React + TypeScript + Vite) |
| Mobile app code | `house_price/mobile/` (Flutter) |
| Reproduce | copy the CSV into `data/`, run `notebook/house_price.ipynb` end to end (writes `model/*.joblib` + `input_schema.json`) → `uvicorn api.main:app --port 8001` → `npm --prefix web install && npm --prefix web run dev` → `flutter run` in `mobile/` |

---

## Cross-application comparison — house price row

| Aspect | House price |
|---|---|
| Problem type | Regression |
| One observation | one property listing |
| Target | `Price` (continuous, million VND; trained on `log1p(Price)`) |
| Input representation | `X ∈ ℝ^{N×404}` feature matrix (mostly one-hot: `Province` × `district`) |
| Data-quality issues | 8.3% duplicate re-scrapes; 5.9% invalid `Price`/`Area`; 70–84% missing on room counts; heavy right-skew on `Price`/`Area`/`Width`/`Length` |
| Best model | Random Forest Regressor (R² 0.186, narrowly ahead of XGBoost) |
| Main metric | RMSE / R² — MAE is easiest to communicate in currency units, but R² is what shows the model still leaves most price variance unexplained |
| Web deployment | Yes (React + TS + Vite client → FastAPI) |
| Mobile deployment | Yes (Flutter) |
| Main limitation | weak signal from the available numeric attributes (heavy missingness); price driven mostly by location/property-type, which the model captures only through high-cardinality one-hot encoding rather than richer geospatial features |

---

## Screenshot checklist

| ID | Where | Shows |
|---|---|---|
| N1 | notebook §4 | `df.info()` + `describe()` — raw 236,226 × 28 shape and dtypes |
| N2 | notebook §5 | data-quality issue → count → action table |
| N3 | notebook §10.1 | Price/log-Price distribution + property-type bar chart |
| N4 | notebook §10.2 | log-Area vs log-Price scatter (pooled + by property type) + correlation bars |
| N5 | notebook §12 | raw listing → feature vector printout + shapes |
| N6 | notebook §18 | 5-model comparison table |
| N7 | notebook §19–§20 | test metrics, predicted-vs-actual/residual plot, worst-10-errors table |
| W1–W2 | web app | input form filled in / prediction result |
| W3 | web app | validation error state (optional) |
| W4 | FastAPI `/docs` or Postman | the shared REST API (optional) |
| M1–M2 | mobile app | input screen / result screen |
| M3 | mobile app | settings modal showing the API host (optional) |
| X1 | API terminal | a `POST /predict` log line while the web/mobile app is used |
