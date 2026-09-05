# Assignment 02 — From Data Representation to Deployable Intelligent Systems

Three intelligent applications built on the **same pipeline**, so their differences can
be compared side by side:

```
Raw data → Understand → Clean → Represent → Learn → Evaluate → Persist → Deploy
```

| # | Application | Task | Raw form | Representation | Status |
|---|---|---|---|---|---|
| 1 | **Diabetes** prediction | binary classification | CSV (BRFSS survey) | feature matrix `X ∈ ℝ^{N×23}` | ✅ notebook · API · web · mobile |
| 2 | **House price** prediction | regression | CSV | encoded + scaled feature matrix | ⏳ not started |
| 3 | **Customer behaviour** (Olist e-commerce) | binary classification | 9 CSV tables + review comments | tabular `ℝ^{43}` **‖** TF-IDF text `ℝ^{~13000}` | ✅ notebook · API · web · mobile |

Each application is self-contained under its own folder with an identical layout
(Appendix A):

```
<app>/
  data/        raw dataset (or download reference)
  notebook/    <app>.ipynb  — the 23-section ML experiment, executed with outputs
  model/       model_pipeline.joblib  +  feature_names.joblib  +  input_schema.json
  api/         FastAPI service exposing POST /predict
  web/         React + Vite single-page client (no model in the browser)
  mobile/      Flutter client (REST client of the API)
  requirements.txt
report/        Assignment_02.pdf  (final ~10-page report)
```

The **report** is written from each app's `Report_Deliverable.md` guide
(`diabetes/Report_Deliverable.md`, `customer_behaviour/Report_Deliverable.md`) plus the
executed notebooks.

---

## Environment

| | |
|---|---|
| Python | 3.13 (3.14 also tested for diabetes) |
| OS | Windows 11 |
| Random seed | `RANDOM_SEED = 42` everywhere (numpy, `random`, every split / subsample / estimator) |
| Node (web) | 18+ |
| Flutter (mobile) | 3.19+ |

Each app pins its Python deps in `<app>/requirements.txt`. Install per app:

```bash
cd assignment_02/<app>
pip install -r requirements.txt
```

---

## Data-representation summary (mandatory table)

| Application | Raw form | Numerical representation | Model input |
|---|---|---|---|
| Diabetes | CSV / table | 21 raw + 2 engineered → feature vector, `StandardScaler` on 8 numeric cols, no one-hot | `X ∈ ℝ^{N×23}` dense |
| House price | CSV / table | encoded + scaled feature matrix | `X ∈ ℝ^{N×d}` *(pending)* |
| Customer behaviour | 9 CSV tables + PT review comments | 20 tabular cols → `ℝ^{43}` (impute → log1p money → scale; one-hot payment/region/category) **‖** `TfidfVectorizer(1–2-gram)` on the comment → `ℝ^{~13000}` sparse | `X ∈ ℝ^{N×~13000}` sparse, `N = 95,824` |

Every dimension is explained in the corresponding notebook §12 and in the report.

---

## Application 1 — Diabetes

**Dataset:** Kaggle `alexteboul/diabetes-health-indicators-dataset`
(`diabetes_012_health_indicators_BRFSS2015.csv`), already in `diabetes/data/`.
253,680 rows → 229,781 after de-duplication. Target `Diabetes_binary`.
**Deployed model:** Random Forest, test ROC-AUC ≈ 0.81, recall ≈ 0.74.

```bash
cd assignment_02/diabetes

# 1. reproduce the experiment (writes model/*.joblib)
jupyter nbconvert --to notebook --execute notebook/diabetes.ipynb --output diabetes.ipynb
python api/build_artifacts.py            # one-off serving artifacts (neighbour index, SHAP bg)

# 2. run the API  ->  http://localhost:8000/docs
uvicorn api.main:app --port 8000

# 3. run the web client  ->  http://localhost:5173
npm --prefix web install && npm --prefix web run dev

# 4. run the mobile client (Android emulator: 10.0.2.2 is the host)
cd mobile && flutter create . && flutter pub get
flutter run --dart-define=API_URL=http://10.0.2.2:8000
```

`POST /predict` example:

```bash
curl -s http://localhost:8000/predict -H 'content-type: application/json' -d '{
  "Age": 9, "Sex": 1, "HighBP": 1, "HighChol": 1, "BMI": 34, "GenHlth": 4,
  "DiffWalk": 1, "PhysActivity": 0, "Smoker": 1
}'
# -> { "prediction": "diabetic", "confidence": 0.85, ... }
```

More detail: `diabetes/api/README.md`, `diabetes/web/README.md`, `diabetes/mobile/README.md`.

---

## Application 2 — House price

⏳ **Not started.** Folder skeleton only. Planned: pick a Kaggle house-price dataset,
23-section notebook, 5 regression models (Linear, Ridge/Lasso, Decision Tree, Random
Forest, Gradient Boosting), `POST /predict` → `{ "predicted_price": ... }`, web + mobile.

---

## Application 3 — Customer behaviour (Olist e-commerce)

**Dataset:** Kaggle `olistbr/brazilian-ecommerce` (9 CSVs), in `customer_behaviour/data/`.
98,673 reviewed orders → 95,824 after keeping delivered orders. One row = one order,
aggregated from the item / payment / product / customer tables. Target
`satisfied = review_score ≥ 4` (~79% positive).
**Deployed model:** Logistic Regression on the **tabular + comment-text** representation,
test ROC-AUC 0.859, dissatisfied-class recall 0.672. Adding the comment text lifts
mean ROC-AUC by ~0.08 over tabular features alone.

```bash
cd assignment_02/customer_behaviour
pip install -r requirements.txt

# 1. reproduce the experiment (writes model/*.joblib + input_schema.json)
python -m nbclient notebook/customer_behaviour.ipynb        # or: jupyter nbconvert --execute

# 2. run the API  ->  http://localhost:8000/docs
uvicorn api.main:app --port 8000

# 3. run the web client  ->  http://localhost:5174
npm --prefix web install && npm --prefix web run dev

# 4. run the mobile client
cd mobile && flutter create . && flutter pub get
flutter run --dart-define=API_URL=http://10.0.2.2:8000
```

Or run the API + web together in Docker:

```bash
cd assignment_02/customer_behaviour
docker compose up --build
#   web -> http://localhost:5174   ·   api -> http://localhost:8000/docs
```

`POST /predict` example:

```bash
curl -s http://localhost:8000/predict -H 'content-type: application/json' -d '{
  "price_total": 129.90, "freight_total": 18.30, "main_payment_type": "credit_card",
  "max_installments": 3, "customer_state": "SP", "category": "bed_bath_table",
  "order_purchase_timestamp": "2018-05-01 10:00:00",
  "order_estimated_delivery_date": "2018-05-20 00:00:00",
  "order_delivered_customer_date": "2018-05-31 14:00:00",
  "review_comment_message": "Produto chegou muito atrasado e a embalagem estava danificada."
}'
# -> { "prediction": "dissatisfied", "confidence": 0.9252, "p_satisfied": 0.0748, ... }
```

More detail: `customer_behaviour/api/README.md`, `customer_behaviour/web/README.md`,
`customer_behaviour/mobile/README.md`, `customer_behaviour/Report_Deliverable.md`.

---

## Deployment architecture (shared by all apps)

```
User input → API request → validation → SAME preprocessing (loaded from training) → saved model → prediction → JSON → web / mobile UI
```

**Data-leakage rule:** the deployed service loads the preprocessing pipeline that was
fitted on the *training* split and only calls `.transform()` / `.predict_proba()`. It
never fits a new scaler, encoder, imputer or vectoriser on user input or test data.
The notebooks verify this in §23 by reloading the artifact from disk and asserting the
prediction matches the in-memory pipeline.
