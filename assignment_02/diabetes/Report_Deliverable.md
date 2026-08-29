# Report guide — Application 1: Diabetes Prediction

What to write for the **diabetes** part of the Assignment 02 report (Part A §4, plus
the shared Reproducibility, Web-app and Mobile-app templates). Each subsection lists
the points to cover and where the material already exists. Screenshot placeholders
are marked `📸` with a caption describing exactly what the image must show — replace
each with the actual image when you take it.

Source material:
- `notebook/diabetes.ipynb` — the 23-section ML experiment (executed, with outputs).
- `APP_DESIGN.md` — deployment design (personas, endpoints, the extra features).
- `api/README.md`, `web/README.md`, `mobile/README.md` — how each part runs.
- Key executed numbers are quoted below so the report can cite them directly.

---

## 4.1 Problem description

- Use the template sentence:
  *"The objective of this application is to predict whether a person is likely to
  have diabetes or pre-diabetes based on **21 self-reported items from a short health
  questionnaire (no blood test)**. The prediction target is **`Diabetes_binary`
  (0 = no diabetes, 1 = pre-diabetes or diabetes)**. The prediction can potentially
  support **population screening — deciding who to refer for a confirmatory blood
  test or a lifestyle-intervention programme**."*
- Define `X` = the 21 questionnaire features (list the groups: clinical-history flags,
  lifestyle flags, access-to-care flags, self-rated health, demographics); `y` =
  `Diabetes_binary`.
- State it is a **binary classification** problem. Explain the 3→2 class collapse:
  the screening action is identical for pre-diabetes and diabetes, and the raw
  pre-diabetes level is only ~1.8% of rows.
- Source: notebook §1.

## 4.2 Kaggle dataset

- Name: **Diabetes Health Indicators Dataset** (`diabetes_012_health_indicators_BRFSS2015.csv`).
- Kaggle URL: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset
  Primary reference: UCI ML Repository #891.
- Observations: **253,680** raw → **229,781** after removing exact duplicates.
- Features: **21** input + 1 label; +2 engineered → **23** model features.
- Target: `Diabetes_012` (0/1/2) → `Diabetes_binary`.
- Include the **feature dictionary table** from notebook §1.3 (feature, type, coding,
  meaning) — this satisfies the "feature descriptions" requirement.
- Origin: CDC BRFSS 2015 telephone survey; self-reported; CDC bands several fields
  (age → 13 bands, income → 8, general health → 5).

## 4.3 Data understanding

Report **and interpret** (do not just paste):
- Shape `(253680, 22)`; all columns `float64` though most are integer codes.
- Missing values: **none** (`isna().sum()` = 0 everywhere).
- Duplicates: **23,899 exact duplicate rows (~9.4%)**; no respondent id, so only
  whole-row duplicates can be detected.
- Invalid values: only `BMI` has out-of-domain values (range 12–98; a few dozen rows
  < 14 or > 80). All coded fields are in range.
- Class imbalance: raw positive rate ~15.8%; **~17.3% after de-duplication** (dropping
  duplicates removes mostly "healthy" profiles).
- 📸 **Screenshot: `df.info()` + `df.describe()` output** from notebook §4 — shows the
  253,680 × 22 shape, the float64 dtypes, and the BMI 12–98 / MentHlth 0–30 ranges.
- 📸 **Screenshot: the data-quality table** from notebook §5 (issue → count → planned
  action) — the compact summary of every problem found.

## 4.4 Data cleaning

For each operation state **what** and **why**:
- **Duplicate removal** — drop 23,899 exact duplicates *before* the split, so no
  identical answer-vector appears in both train and test (leakage). `N`: 253,680 →
  229,781.
- **Missing values** — none to handle; a `SimpleImputer(median)` is kept in the
  pipeline anyway so the deployed service tolerates a missing field at inference.
- **Invalid `BMI`** — not clipped or dropped in training. Justification: the extreme
  rows are rare and the `60–80` band shows an *above-average* diabetes rate (genuine
  severe obesity, not noise), and the deployed random forest is rank-invariant to
  outliers. Notebook **Appendix A** shows keeping / clipping / dropping changes
  ROC-AUC and F1 by ≤ 0.003.
- **Outliers in `MentHlth` / `PhysHlth`** — kept; the large values are valid survey
  answers ("bad-health days"), and `StandardScaler` puts them on a comparable scale.
- **Categorical processing** — none needed; the 4 ordinal codes stay numeric because
  each is monotonic with risk (notebook §10).
- 📸 **Screenshot: notebook §9 output** — the BMI-band table (n rows, % of data,
  diabetes rate per band) + the boxplots for BMI / MentHlth / PhysHlth.

## 4.5 Data representation

Show the transformation chain with concrete numbers (notebook §12):
```
CSV  ->  DataFrame  ->  clean feature matrix  ->  scaled matrix  ->  model input
```
- **One raw CSV record** (print `df.iloc[0]` of the 21 features).
- **The feature vector it becomes**: 8 scaled columns + 15 pass-through, `x ∈ ℝ²³`.
- Original DataFrame shape `(229781, 23)`; feature matrix `X ∈ ℝ^{N×23}`, `N = 229781`.
- `y ∈ {0,1}^N`. Dtype of `X`: dense `float64` ndarray. One API request = `ℝ^{1×23}`.
- Encoding: **none** (all numeric/ordinal). Scaling: `StandardScaler` on the 8-column
  numeric block (needed for logistic regression / SVM / KNN).
- 📸 **Screenshot: notebook §12 printed output** — the raw record, the resulting
  feature vector, and the four shape/dtype lines.

## 4.6 Exploratory data analysis

Include ≥ 3 of the 5 plots from notebook §10; for **each** give Observation /
Interpretation / ML implication (already written under §10):
- Plot 1 (bar) — target class balance (~190,055 vs ~39,726; ~17.3% positive).
- Plot 2 (violin) — BMI by class (median ~31 vs ~27, heavy overlap).
- Plot 3 (line) — diabetes rate vs age band (2% → ~25% plateau).
- Plot 4 (bar) — diabetes rate vs income band (28% → 11% gradient).
- Plot 5 (heatmap) — correlation matrix (top: GenHlth 0.28, HighBP 0.26, BMI 0.21;
  no feature pair |r| > 0.6).
- 📸 **Screenshot: the 2×2 plot grid** from notebook §10 (plots 1–4).
- 📸 **Screenshot: the correlation-matrix heatmap** from notebook §10 (plot 5).

## 4.7 Model development

- Five models compared (notebook §17): Logistic Regression, Decision Tree, Random
  Forest, SVM (RBF), KNN.
- State the fair-comparison protocol: **all five trained on the same stratified
  25,000-row subsample** with the same preprocessing, because an RBF-SVM does not
  train in acceptable time on the full 160k rows; the winner is then **refitted on the
  full training set** for the held-out evaluation and for deployment.
- Give the hyperparameter table from notebook §17.
- 📸 **Screenshot: notebook §18 comparison table** — accuracy / precision / recall /
  F1 / ROC-AUC per model on validation, plus the stability table (winner @ 25k vs @
  full train).

Reference numbers (validation, 25k fit):
| model | ROC-AUC | F1 | recall |
|---|---|---|---|
| RandomForest | 0.810 | 0.487 | 0.748 |
| LogisticRegression | 0.804 | 0.481 | 0.760 |
| SVM (RBF) | 0.800 | 0.478 | 0.776 |
| KNN | 0.790 | 0.174 | 0.104 |
| DecisionTree | 0.787 | 0.459 | 0.760 |
| *baseline B (3-feature LogReg)* | *0.77* | | |
| *baseline A (majority class)* | *0.50* | *0* | *0* |

## 4.8 Evaluation

- Chosen model: **Random Forest**, refit on full training data, evaluated once on the
  held-out **test** set (notebook §19).
- Report: Accuracy **0.72**, Precision **0.36**, Recall **0.74**, F1 **0.48**,
  ROC-AUC **0.81**.
- Confusion matrix `[[20612, 7897], [1579, 4380]]` (rows = actual, cols = predicted).
  **Interpret it**: FN (1,579) = diabetics told they are healthy — the dangerous
  error; FP (7,897) = healthy people referred — cost is one extra blood test.
- **Which metric matters most: recall on the positive class**, because for a
  population screen a false negative (missed, untreated case) is far worse than a
  false positive. Accuracy is explicitly *not* the headline — the majority-class
  baseline scored higher on accuracy and was useless.
- 📸 **Screenshot: notebook §19** — the classification report + the confusion-matrix
  heatmap + ROC curve.

## 4.9 Model selection

Justify Random Forest for deployment across the five criteria (notebook §21):
- **Predictive performance** — best ROC-AUC and F1 in the comparison.
- **Interpretability** — feature importances + per-tree paths; SHAP added at serving.
- **Computational cost** — trains in seconds, < 10 ms per prediction.
- **Robustness** — bagging absorbs noise/outliers, models interactions, rank-invariant
  to the BMI tail.
- **Deployment constraints** — ~55 MB pickled, CPU-only, no dependency beyond
  scikit-learn. Logistic Regression kept as a documented lightweight fallback
  (~0.01 ROC-AUC behind, ~30 KB).

## 4.10 Deployment

- Architecture (same as the other two apps):
  ```
  User input → API request → validation → same preprocessing → saved model → prediction → result
  ```
- The service loads `model/model_pipeline.joblib` (imputer + scaler + random forest)
  and never re-fits it. Explain the leakage rule: the deployed preprocessing is the
  one fitted on training data.
- Inference flow for one request: raw 21-field dict → `validate.prepare()` (BMI from
  height+weight, clamp to [12, 98], completeness) → `engineer()` (adds
  `TotalUnhealthyDays`, `CardioRisk`) → pipeline → probability → band.
- Example response:
  ```json
  { "prediction": "diabetic", "confidence": 0.85 }
  ```
  (the API's richer form: `probability`, `band`, `band_label`, `uncertainty_band`,
  `completeness`, `warnings`, and optional `explain` / `similar` / `whatif` blocks.)
- Web framework: **FastAPI** (`POST /predict`). Client: **React (Vite)**, a
  single-page app that flows questionnaire → result on one screen, with a separate
  Operator view. Mobile: **Flutter** (REST client, no on-device model).

### 4.10.1 Interesting features (the "beyond `model.predict()`" part)

Frame each against a persona and an ML-system concern:
- **SHAP explanation** — clinicians distrust an opaque score, so every prediction
  returns the signed contribution of each feature ("High blood pressure +0.11",
  "Eats vegetables daily −0.03"). Endpoint `POST /explain`.
- **Similar cases (random-forest proximity)** — case-based trust: the 5 training
  respondents whose answers land in the same forest leaves most often, with their
  real outcomes ("4 of your 5 closest matches had diabetes"). Endpoint `POST /similar`.
  Uses the model's *own* similarity, not a separate k-NN.
- **What-if / counterfactual** — actionability: the risk change if each *modifiable*
  factor (BMI, smoking, heavy drinking, activity, fruit, vegetables) were at its
  healthier value, a BMI sweep, and the smallest set of changes that lowers the band.
  Every response carries the "model association, not medical advice" caveat.
  Endpoints `POST /whatif`, `POST /counterfactual`.
- **Missing-data handling** — every questionnaire item is optional (yes/no answers
  rest at "No"; dropdowns can be skipped; BMI is computed from height + weight or
  filled from an age×sex table). The API reports a completeness score and an
  uncertainty band (re-scoring with each unknown binary forced 0/1). Out-of-range
  BMI is clamped to the dataset range [12, 98] with a visible warning.
- **Operator threshold dashboard** — the Low/Moderate/High cut-points are a clinic
  policy, not part of the model. The dashboard shows the precision/recall/flag-rate
  trade-off at any cut-point on held-out data (`GET /threshold-curve`), a
  recommendation helper, and a guarded `POST /config/thresholds` to persist the
  choice. Live monitoring (`GET /metrics`) shows the actual flag rate over time.

### 4.10.2 Web application (Appendix D template)

```
Web Application — Diabetes Prediction
Framework:  React + Vite (single-page client)  +  FastAPI (POST /predict)
Endpoint:   POST http://<host>:8000/predict?include=explain,similar,whatif,counterfactual
Input:      21 questionnaire fields (Age, Sex required; rest optional/nullable) + height/weight
Output:     { probability, band, band_label, uncertainty_band, completeness, warnings,
              explain{...}, similar{...}, whatif{...}, counterfactual{...} }
```

- 📸 **Screenshot W1 — Questionnaire with answers filled in.** Show several sections
  answered (yes/no toggles, dropdowns) and height/weight entered. Caption: *the input
  interface: one scrolling form; unanswered yes/no items rest at "No" and blanks are
  filled with typical values.*
- 📸 **Screenshot W2 — Result, top of the page.** Immediately below the form after
  submitting: the tinted risk headline (risk % + Low/Moderate/High pill + one-line
  verdict + uncertainty range), any warnings, and the **interactive SHAP force
  plot** ("Why this estimate") with a tooltip visible on one segment. Caption: *the
  prediction and the force plot showing which answers pushed the estimate up (red)
  or down (blue) from the population average.*
- 📸 **Screenshot W3 — Result, scrolled down.** The "People like you in the survey"
  card ("N of 5 had diabetes" + de-identified profiles) and the "What could change
  the estimate" card (per-habit risk deltas, the live BMI slider with its sparkline,
  and the counterfactual sentence with the caveat). Caption: *case-based evidence
  and the modifiable-factor analysis, all re-scored through the same API.*
- 📸 **Screenshot W4 — Operator view.** The single referral cut-off slider, the
  plain-language "out of every 1,000 people screened…" readout, and the
  recall-vs-flag-rate trade-off curve with the current operating point marked.
  Caption: *the operator view: one decision (the referral cut-off) with its effect
  on held-out validation data stated in plain numbers.*
- 📸 **Screenshot W5 (optional) — FastAPI `/docs`.** The Swagger UI listing the
  endpoints, or a successful `POST /predict` "Try it out" response. Caption: *the
  REST API the web and mobile clients share.*

**Figure explanation to write for each:** describe the input shown, the prediction
returned, and how to read the band / explanation.

### 4.10.3 Mobile application (Appendix E template)

```
Mobile Application — Diabetes Prediction
Framework:  Flutter
Platform:   Android (emulator / device)
API:        POST http://<host>:8000/predict   (same endpoint as the web app)
```

- 📸 **Screenshot M1 — Questionnaire screen.** A partly filled form: the Yes / No /
  Not sure segmented buttons, a dropdown, the height/weight fields. Caption: *mobile
  input screen; identical questions to the web app, rendered from `GET /questions`.*
- 📸 **Screenshot M2 — Result screen.** The coloured risk band card, the completeness
  note, the SHAP factor bars. Caption: *the prediction and its explanation on the
  device.*
- 📸 **Screenshot M3 — Result screen, scrolled.** The similar-cases list and the
  counterfactual sentence. Caption: *similar survey respondents and the smallest
  change that would lower the risk band.*
- 📸 **Screenshot M4 — "What could change" screen.** The modifiable-factor delta list
  and the BMI slider with the live risk bar. Caption: *interactive what-if; moving the
  BMI slider re-scores against the API in real time.*
- 📸 **Screenshot M5 (optional) — History screen.** The list of this device's past
  screenings. Caption: *screening history, retrieved from `GET /history`.*
- Also capture **evidence the app calls the API**: a screenshot of the API terminal
  showing the incoming `POST /predict` log line, or the phone's network inspector.
  Caption: *the mobile app is a REST client — inference runs on the server, not the
  device.*

**Figure explanation to write:** how the mobile UI collects input, sends it to
`POST /predict`, and displays the returned band + confidence + explanation.

---

## Reproducibility (shared section — diabetes entries)

| Item | Value |
|---|---|
| Python | 3.14 (Anaconda) |
| OS | Windows 10 |
| Key libraries | numpy 2.4, pandas 3.0, scikit-learn 1.9.0, shap 0.52, fastapi 0.141 (API); React 18 + Vite 5, Node 24 (web); Flutter 3.19+ (mobile) |
| Random seed | `RANDOM_SEED = 42` (numpy + `random`; also every split and estimator) |
| Dataset source | Kaggle `alexteboul/diabetes-health-indicators-dataset`, file `diabetes_012_health_indicators_BRFSS2015.csv`, downloaded 2026-08-29 |
| Preprocessing | `ColumnTransformer`: median `SimpleImputer` + `StandardScaler` on 8 numeric cols, pass-through on 15 binary cols; fitted on train only |
| Feature representation | 21 raw + 2 engineered = 23 features; `X ∈ ℝ^{N×23}` dense float64 |
| Train/val/test split | 70 / 15 / 15, stratified, seed 42; duplicates dropped before the split |
| Model hyperparameters | RandomForest: `n_estimators=300, max_depth=12, min_samples_leaf=20, class_weight="balanced_subsample"` (full list: notebook §17) |
| Evaluation metrics | test: Acc 0.72 · Prec 0.36 · Rec 0.74 · F1 0.48 · ROC-AUC 0.81 |
| Saved pipeline | `model/model_pipeline.joblib` (preprocessing + model) |
| Saved model config | `model/feature_names.joblib`, `model/input_schema.json` |
| Serving artifacts | `api/build_artifacts.py` → `model/holdout_scores.joblib`, `api/artifacts/{neighbor_index, shap_background, bmi_by_age_sex}.joblib`, `api/runtime/thresholds.json` |
| API code | `diabetes/api/` (FastAPI) |
| Web app code | `diabetes/web/` (React + Vite; former Streamlit client kept in `web/legacy_streamlit/`) |
| Mobile app code | `diabetes/mobile/` (Flutter) |
| Reproduce | `nbconvert --execute notebook/diabetes.ipynb` → `python api/build_artifacts.py` → `uvicorn api.main:app` → `npm --prefix web install && npm --prefix web run dev`; `flutter run` in `mobile/` |

---

## Cross-application comparison — diabetes row

| Aspect | Diabetes |
|---|---|
| Problem type | Binary classification |
| One observation | one survey respondent |
| Target | `Diabetes_binary` (0 / 1) |
| Input representation | `X ∈ ℝ^{N×23}` feature matrix, no one-hot |
| Data-quality issues | 9.4% exact duplicates; out-of-range BMI; 17% class imbalance; no missing values |
| Best model | Random Forest |
| Main metric | Recall (sensitivity) on the positive class — 0.74 at ROC-AUC 0.81 |
| Web deployment | Yes (React + Vite client → FastAPI) |
| Mobile deployment | Yes (Flutter) |
| Main limitation | associational model on a 2015 US survey; misses cases driven by factors not in the questionnaire (family history, fasting glucose) |

---

## Screenshot checklist

| ID | Where | Shows |
|---|---|---|
| N1 | notebook §4 | `df.info()` + `describe()` — dataset shape and dtypes |
| N2 | notebook §5 | data-quality issue → count → action table |
| N3 | notebook §9 | BMI-band table + boxplots |
| N4 | notebook §12 | raw record → feature vector + shapes |
| N5 | notebook §10 | 2×2 EDA plot grid |
| N6 | notebook §10 | correlation heatmap |
| N7 | notebook §18 | model comparison + stability tables |
| N8 | notebook §19 | classification report + confusion matrix + ROC |
| W1–W4 | web app | questionnaire / result-top (force plot) / result-scrolled (similar + what-if) / operator view |
| W5 | FastAPI `/docs` | the shared REST API (optional) |
| M1–M4 | mobile app | questionnaire / result / result-scrolled / what-if |
| M5 | mobile app | history (optional) |
| X1 | API terminal | a `POST /predict` log line while the mobile app is used |
