# Report_mock.md — worked example of the Assignment 02 report

> This is a **filled-in mock**, not the real report. Every number in `«guillemets»` is an
> example so you can see the *shape* of a good answer. Replace them with your real values.
> Delete this note before submitting.
>
> Target length ≈ 10 pages (excluding code + appendices). Same framework for all 3 apps
> so they can be compared side by side.

---

## Cover Page

```
INTELLIGENT SYSTEM DEVELOPMENT
ASSIGNMENT 02
From Data Representation to Deployable Intelligent Systems

Student name: «Nguyen Van A»
Student ID:   «B21DCCN001»
Class:        «D21CQCN01-B»
Team:         «—»
Lecturer:     Dinh Que Tran, Ph.D., Assoc. Prof.
Semester:     I.2026
```

---

## 1. Executive Summary

«This report builds three deployable intelligent systems from raw Kaggle data: a diabetes
screening classifier, a house-price regressor, and an e-commerce customer-interest
classifier that also uses review text. Each system is taken through the full pipeline
— Data → Understand → Clean → Represent → Learn → Evaluate → Persist → Deploy — and
exposed as a REST API consumed by both a web page and a mobile app.»

| Application | Prediction / Task | Main Representation |
|---|---|---|
| Diabetes | Binary classification: diabetic vs not | Scaled feature matrix `X ∈ ℝ^{N×d}` |
| House Price | Regression: sale price in currency | One-hot + scaled feature matrix `X ∈ ℝ^{N×d}` |
| Customer Behavior | Binary classification: will recommend product | Tabular matrix `X ∈ ℝ^{N×d}` **+** TF-IDF text vectors |

Per application:

| Field | Diabetes | House Price | Customer Behavior |
|---|---|---|---|
| Dataset | «Pima Indians Diabetes Database» | «Housing Prices Dataset (yasserh)» | **Sephora Products and Skincare Reviews** (`nadyinky`) |
| Kaggle URL | «kaggle.com/datasets/uciml/pima-indians-diabetes-database» | «kaggle.com/datasets/yasserh/housing-prices-dataset» | kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews |
| Problem | Flag patients at risk so they get a confirmatory blood test | Estimate a fair listing price from house attributes | Predict whether a skincare reviewer recommends the product (`is_recommended`), from their skin profile + the product + the review text |
| Selected model | «Random Forest — best recall at acceptable precision» | «Gradient Boosting Regressor — lowest RMSE, robust to skew» | Logistic Regression on tabular + TF-IDF text — best macro-F1 / minority recall, one small CPU artifact |
| Headline metric | «Recall = 0.78, F1 = 0.74, ROC-AUC = 0.83» | «RMSE = «1,150,000 VND-units», R² = 0.68» | test ROC-AUC 0.964, macro-F1 0.858, recall on "won't recommend" 0.876, accuracy 0.918 |
| Deployment | «FastAPI + Flutter» | «FastAPI + Flutter» | FastAPI + React/Vite web + Flutter mobile |

---

## 2. Connection to Lecture 02 — Data Representation

```
Real-world object → Raw data → Numerical representation → Tensor → Model
   (a patient)       (CSV row)     (feature vector x)      (X matrix)  (RF)
```

- A **tabular sample** is a vector `x = [x₁, x₂, …, x_d]ᵀ ∈ ℝ^d`. One patient / house / customer = one such vector.
- A **dataset** is a matrix `X ∈ ℝ^{N×d}` — `N` rows stacked, each row a sample, each column a feature.
- **Text** is turned into numbers by: `Text → Tokens → Token IDs → Embeddings`. A batch of
  embedded comments is a 3-D tensor `E ∈ ℝ^{B×T×d}` (B comments, T tokens each, d numbers per token).
- **The representation is part of the solution.** Marks are given for explaining *how raw
  data becomes computational data*, not only for accuracy.

---

## 3. Data Representation Overview

| Application | Raw Data | ML Representation |
|---|---|---|
| Diabetes | CSV, 1 row = 1 patient | scaled feature matrix, d = «8» |
| House Price | CSV, 1 row = 1 house | one-hot + scaled matrix, d = ««13 → 20 after encoding»» |
| Customer Behaviour | CSV + review text, 1 row = 1 review | tabular one-hot+scaled matrix (d_tab = 138) ⊕ TF-IDF vector (d_text = 28 595) → d = 28 733 (sparse) |

Answer for **each** application (example = diabetes):

| Question | Answer (diabetes example) |
|---|---|
| What does one row represent? | One patient examined in the study |
| What does one column represent? | One clinical measurement (or the diagnosis) |
| Input feature columns | Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age |
| Target column | `Outcome` (0 = not diabetic, 1 = diabetic) |
| Numerical features | all 8 |
| Categorical features | none (house price / e-commerce *do* have some) |
| Encoding of categoricals | n/a here; house price uses one-hot `{furnished, semi-furnished, unfurnished} → furnished = [1,0,0]` |
| Final feature dimension `d` | «8» (no encoding changes it) |
| Shape of model input | `X_train ∈ ℝ^{«537×8»}`, one request = `ℝ^{1×8}` |

General notation used throughout:

- Model input: `X ∈ ℝ^{N×d}` — `N` = rows in the batch, `d` = features after encoding.
- Target: `y ∈ ℝ^N` (regression) or `y ∈ {0,1,…,K−1}^N` (classification).
- Text (App 3): `E ∈ ℝ^{B×T×d}` demo — `B` = 1 review, `T` = 40 tokens (padded), `d` = 16 embedding dims → `ℝ^{1×40×16}`. The deployed model uses TF-IDF instead: a 2-D sparse matrix `ℝ^{B×V}`, `V` = 28 595 (training vocabulary).

---

## 4. Application 1 — Diabetes Prediction

### 4.1 Problem description

«The objective of this application is to predict whether a patient is likely to have
diabetes based on **8 routine clinical and demographic measurements**. The prediction
target is **`Outcome`, a binary label (1 = diabetes diagnosed within 5 years, 0 = not)**.
The prediction can potentially support **primary-care triage: patients flagged as
high-risk are prioritised for a confirmatory oral glucose tolerance test.**»

- `X` = patient features: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age.
- `y` = diabetes class ∈ {0, 1}.
- This is primarily a **classification problem**.

### 4.2 Kaggle Dataset

| Field | Value |
|---|---|
| Name | «Pima Indians Diabetes Database» |
| URL | «https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database» |
| Observations (N) | «768» |
| Features (d) | «8 + 1 target» |
| Target | «`Outcome` (binary)» |

| Feature | Type | Meaning |
|---|---|---|
| Pregnancies | Numerical (int) | Number of times pregnant |
| Glucose | Numerical | Plasma glucose concentration, 2-h oral test |
| BloodPressure | Numerical | Diastolic blood pressure (mm Hg) |
| SkinThickness | Numerical | Triceps skin-fold thickness (mm) |
| Insulin | Numerical | 2-h serum insulin (mu U/ml) |
| BMI | Numerical | Body mass index (kg/m²) |
| DiabetesPedigreeFunction | Numerical | Family-history diabetes score |
| Age | Numerical | Age in years |

### 4.3 Data Understanding

Report **and interpret** (do not just paste):

- **Dataset shape:** `df.shape = («768», «9»)` → 768 patients, 9 columns (8 features + target). *This tells us N is small — models must be regularised and we should use cross-validation.*
- **Data types:** «7 int64, 2 float64» → all numeric, no parsing needed.
- **Missing values:** `df.isna().sum()` → «0 explicit NaN**, but** Glucose, BloodPressure, SkinThickness, Insulin, BMI contain zeros that are physiologically impossible → these are *hidden* missing values («Insulin = 0 in 374 rows»).»
- **Duplicated records:** `df.duplicated().sum() = «0»`.
- **Class distribution:** «500 non-diabetic (65 %) / 268 diabetic (35 %) → moderate imbalance; accuracy alone will be misleading.»
- Include 1–2 tables/screenshots from the notebook and say what they mean.

### 4.4 Data Cleaning

| Operation | What | Why |
|---|---|---|
| Hidden missing values | Replace `0` with `NaN` in Glucose, BloodPressure, SkinThickness, Insulin, BMI | A glucose of 0 is impossible; treating it as a real value would bias the model toward 0 |
| Imputation | Median impute those columns **inside the pipeline, fitted on train only** | Median is robust to the skew seen in Insulin; fitting on train only prevents leakage |
| Duplicates | none found → no action | — |
| Outliers | Cap Insulin and SkinThickness at the 99th percentile | A few values are 10× the median and look like recording errors, not real physiology |
| Categorical processing | none (no categorical columns) | — |

### 4.5 Data Representation

```
CSV → DataFrame → clean feature matrix → median-imputed → StandardScaler → model input
```

- **One original CSV record (raw row):**
  `6, 148, 72, 35, 0, 33.6, 0.627, 50, 1`
- **Corresponding feature vector after cleaning + scaling** (target removed):
  `x = [ 0.64, 0.86, -0.03, 0.83, «NaN→0.00 after impute+scale», 0.17, 0.47, 1.42 ]`  → `x ∈ ℝ^8`
- **Original DataFrame shape:** `(768, 9)`.
- **Final feature-matrix shape:** `X ∈ ℝ^{768×8}`; after 70/15/15 split `X_train ∈ ℝ^{537×8}`.
- **Data type of X:** `float64` (numpy `ndarray` produced by the `ColumnTransformer`).
- **Feature encoding:** none needed (all numeric).
- **Scaling:** `StandardScaler` (zero mean, unit variance) — SVM and KNN are distance-based and need comparable scales; also speeds up Logistic Regression convergence.
- **Final model-input shape / tensor supplied to the model:** `ℝ^{1×8}` per request, `ℝ^{B×8}` per batch.

### 4.6 Exploratory Data Analysis  (≥ 3 plots — each with Observation / Interpretation / ML implication)

1. **Target bar chart.** *Observation:* 65/35 split. *Interpretation:* class imbalance. *ML implication:* use F1 / ROC-AUC, consider `class_weight='balanced'`.
2. **Glucose distribution split by Outcome.** *Observation:* diabetic patients centre ~140, non-diabetic ~110. *Interpretation:* Glucose is strongly discriminative. *ML implication:* expect it to dominate feature importance; keep it.
3. **Correlation heatmap.** *Observation:* Age–Pregnancies r ≈ 0.54; no pair > 0.9. *Interpretation:* mild collinearity, no redundant column. *ML implication:* linear models are usable without dropping features; tree models unaffected.

### 4.7 Model Development

Train and compare **five models** on the *same* train split and *same* fitted preprocessing:
Logistic Regression, Decision Tree, Random Forest, SVM (RBF), KNN.
State hyperparameters, e.g. `RandomForest(n_estimators=300, max_depth=8, class_weight='balanced', random_state=42)`.

### 4.8 Evaluation

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | «0.77» | «0.71» | «0.62» | «0.66» | «0.83» |
| Decision Tree | «0.71» | «0.60» | «0.60» | «0.60» | «0.70» |
| **Random Forest** | «0.78» | «0.70» | «**0.78**» | «**0.74**» | «0.83» |
| SVM (RBF) | «0.76» | «0.70» | «0.60» | «0.65» | «0.82» |
| KNN | «0.74» | «0.65» | «0.58» | «0.61» | «0.79» |

**Confusion matrix (Random Forest, test set) — interpreted:**

```
             pred 0   pred 1
actual 0       «68»     «12»     ← 12 healthy patients wrongly flagged (false positives) → an extra blood test, low harm
actual 1       «9»      «31»     ← 9 diabetic patients missed (false negatives) → dangerous, this is the number to minimise
```

**Most important metric = Recall (sensitivity).** A false negative sends a diabetic patient
home untreated; a false positive only costs one confirmatory test. We therefore select the
model with the best recall at precision ≥ «0.65».

### 4.9 Model Selection

«Random Forest is deployed: best recall (0.78) and F1 (0.74), ROC-AUC tied with Logistic
Regression, gives feature importances for clinician trust, trains in «0.4 s», model file is
«3 MB» — acceptable for a container. Logistic Regression was the runner-up and is kept as a
fallback.»

### 4.10 Deployment

- Selected model deployed as **Web Service + Mobile App**.
- Inference flow: `Input patient info → same preprocessing (loaded) → RF model → prediction`.
- Example response:
  ```json
  { "prediction": "diabetic", "confidence": 0.91 }
  ```
- «Web screenshot here» — *Figure explanation: the form takes the 8 measurements, the API returns class + probability, shown as a coloured risk badge.*
- «Mobile screenshot here» — *Figure explanation: Flutter form POSTs JSON to `/predict`, displays "Diabetic (91%)".*

---

## 5. Application 2 — House Price Prediction

### 5.1 Problem description

- Predict the **sale price** of a house from its physical and location attributes.
- `X` = house characteristics (area, bedrooms, bathrooms, stories, parking, mainroad, furnishingstatus, …), `y` = price (continuous, currency units).
- This is a **regression** problem. It differs from diabetes because: the target is a
  continuous real number, the loss is squared/absolute error (not log-loss), and it is
  scored with MAE / RMSE / R² (not accuracy / recall). There is no "class", so no confusion matrix.

### 5.2 Dataset

| Field | Value |
|---|---|
| Name / URL | «Housing Prices Dataset» / «kaggle.com/datasets/yasserh/housing-prices-dataset» |
| Observations | «545» |
| Features | «12 + 1 target» |
| Target | «`price`» |
| Numerical features | «area, bedrooms, bathrooms, stories, parking» |
| Categorical features | «mainroad, guestroom, basement, hotwaterheating, airconditioning, prefarea, furnishingstatus» |

### 5.3 Data Understanding and Cleaning

- Missing values: «none». Duplicates: «none». Invalid: check `price > 0`, `area > 0` → «all valid».
- **Skew:** `price` is right-skewed (skew ≈ «1.6») → apply `log1p` transform to the target, invert with `expm1` at prediction time. *Explain: makes residuals more symmetric so RMSE-based models fit better.*
- Outliers: «3 houses > 13 M» — kept, they are genuine luxury properties, but flagged.

### 5.4 Representation

```
Raw CSV → DataFrame → Clean Data → one-hot(categoricals) + StandardScaler(numericals) → X
```

- Numerical vs categorical are combined into **one feature vector**:
  `x = [x_area, x_bedrooms, x_bathrooms, x_stories, x_parking, x_mainroad_yes, x_basement_yes, …, x_furnish_furnished, x_furnish_semi, x_furnish_unfurnished]ᵀ`
- Categorical encoding: **one-hot**, e.g. `furnishingstatus {furnished, semi-furnished, unfurnished} → furnished → [1, 0, 0]`. Binary yes/no columns → single 0/1 column.
- Result: `X ∈ ℝ^{«545 × 20»}`, `y ∈ ℝ^{545}` (or `log1p(price)`), with `d = 20` after encoding (was 12 raw columns).

### 5.5 Exploratory Data Analysis

Price histogram (before/after log); price vs area scatter; price by airconditioning boxplot; correlation heatmap. Observation / Interpretation / ML implication for each.

### 5.6 Regression Models

Five models, same split + preprocessing: Linear Regression, Ridge (or Lasso), Decision Tree Regressor, Random Forest Regressor, Gradient Boosting Regressor.

### 5.7 Evaluation

$$MAE = \frac{1}{N}\sum_{i=1}^{N}|y_i-\hat y_i| \qquad RMSE=\sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat y_i)^2} \qquad R^2 = 1-\frac{\sum(y_i-\hat y_i)^2}{\sum(y_i-\bar y)^2}$$

| Model | MAE | RMSE | R² | Train time |
|---|---|---|---|---|
| Linear Regression | «970 000» | «1 320 000» | «0.58» | «0.01 s» |
| Ridge | «965 000» | «1 305 000» | «0.59» | «0.01 s» |
| Decision Tree | «1 050 000» | «1 480 000» | «0.47» | «0.02 s» |
| Random Forest | «820 000» | «1 190 000» | «0.66» | «0.3 s» |
| **Gradient Boosting** | «790 000» | «**1 150 000**» | «**0.68**» | «0.5 s» |

- **MAE** = on average the prediction is off by «790 000 currency units» — directly interpretable to a seller.
- **RMSE** ≥ MAE because it punishes big misses (luxury houses) harder.
- **R²** = «0.68» → the model explains 68 % of the variance in price; 32 % is unexplained (location detail, condition, market timing not in the data).

### 5.8 Model Selection

«Gradient Boosting: lowest RMSE and highest R², train time 0.5 s, model «1.2 MB». Chosen over Random Forest for the «40 000»-unit RMSE improvement.»

### 5.9 Deployment

Web + mobile. User enters house characteristics → predicted price:
```json
{ "predicted_price": 4275000 }
```
Screenshots + figure explanations.

---

## 6. Application 3 — E-Commerce Customer Behaviour and Interest

> **Full written section: `customer_behaviour/Report_App3.md`** (real numbers,
> screenshot placeholders, and the App-3 cells for the shared tables). Summary below.

### 6.1 Problem description

- Objective: analyse **customer behaviour and interest** on a beauty marketplace and
  predict, from the reviewer's profile **and their review text**, whether they recommend
  the product.
- **Target:** `recommended = is_recommended ∈ {0,1}` — a field *separate* from the 1–5 star
  rating (which is excluded as leakage, §6.6a). Binary classification; one observation =
  one product review.
- `X` = skin profile (`skin_type/skin_tone/eye_color/hair_color`) + product
  (`price_usd`, `secondary_category`, `brand_name`, `loves_count`, review count, flags) +
  review timing/engagement **+** TF-IDF of `review_title + review_text`. `y = recommended`.

### 6.2 Dataset

| Field | Value |
|---|---|
| Name / URL | **Sephora Products and Skincare Reviews** (`nadyinky`, CC0) / kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews |
| Files | `reviews_500-750.csv` joined to `product_info.csv` on `product_id` |
| Rows | **116 262** raw → **104 313** after cleaning (§6.4); 1 row = 1 review; positive rate **0.8465** |
| Scope | 249 products · ~79 brands · 12 `secondary_category` values, all in the Skincare category |

### 6.3 Customer representation

- **Tabular:** 21 columns (9 numeric log/scale, 4 numeric scale, 6 binary, 6 one-hot
  categorical) → `x_tab ∈ ℝ^{138}` after one-hot.
- **Text — required `Comment → Tokens → Token IDs → Embedding` demo** on a real review:
  40 tokens → 40 IDs (index into a 41-row vocab, 0 = PAD) → `E ∈ ℝ^{T×d} = ℝ^{40×16}`,
  batched `E_batch ∈ ℝ^{B×T×d} = ℝ^{1×40×16}`. `B` = reviews per batch, `T` = padded
  token length, `d` = embedding width. The **deployed** model uses the sparse **TF-IDF**
  form (`ngram_range=(1,2)`, `min_df=10`, ≤ 40 000 features) → `x_txt ∈ ℝ^{28 595}` on
  the training vocabulary.
- **Combined:** `X = [x_tab ‖ x_txt] ∈ ℝ^{104 313 × 28 733}` sparse.

### 6.4 Data cleaning

Drop 11 803 blank-target + 125 empty-body + a few duplicate reviews (116 262 → 104 313).
Skin-profile blanks → explicit `__na__` one-hot level (missingness is a behaviour).
Incidental numeric gaps → `SimpleImputer(median)` fitted **on train only**. Lower-case all
categoricals; merge `eye_color "grey" → "gray"`. Outliers kept (genuine — `log1p` handles
the skew). The review's own `rating` and `rating_product` are quarantined to the leakage
demo (§6.6a). State *why* for each.

### 6.5 Interest discovery / EDA

Six figures (each Observation / Interpretation / ML implication): class balance 84.7%
recommend; recommend rate by skin type (spread ~0.03) and category (~0.16) and brand
(~0.40); recommend rate vs the excluded star rating (0.01 → 1.00 — the leak); topic
buckets (texture ~42%, skin outcome ~39%, irritation ~23%, scent ~26%, price ~13%) +
frequent terms; mutual information (excluded `rating` 0.35, `log_loves` 0.037, category
0.002). Plus customer segmentation: reviewer `xᵢ = [Rᵢ, Fᵢ, Mᵢ, avg_priceᵢ, recommend-rateᵢ,
avg-ratingᵢ, Cᵢ₁…C_ik]`, K-Means `k = 5` (73 819 reviewers, `F ≥ 2` for 22%).

### 6.6 Model development

**Eight pipelines** — the assignment's six (Logistic Regression, Decision Tree, Random
Forest, Linear SVM, a text-based linear classifier `SGD(log_loss)`, and `HistGradientBoosting`
as the extra justified model) plus Complement Naive Bayes and the deployed Logistic
Regression on tab+text. Validation:

| Representation | ROC-AUC | macro-F1 | recall(0) |
|---|---|---|---|
| Tabular only (best: HistGradientBoosting) | 0.811 | 0.664 | 0.678 |
| Tabular only (linear / Decision Tree) | ~0.77 | ~0.62 | ~0.68 |
| **Text only** (SGD log-loss, TF-IDF) | 0.965 | 0.856 | 0.885 |
| **Tabular + text** (deployed) | **0.966** | **0.861** | **0.885** |

→ Both representations carry real skill; the text is well ahead; adding it to the tabular
block gives a small, consistent lift (macro-F1 0.856 → 0.861). Ladder: skin profile 0.536
→ + product 0.656 → full tabular 0.768 → + text 0.966.

### 6.6a Data leakage (assignment requirement)

(1) The review text is **co-authored** with the recommend tick — the text model partly
*reads* the verdict; the ~0.965 is a retrospective upper bound. (2) The review's own star
`rating` **is** the label in another column — adding it → ROC-AUC ~0.985; excluded.
(3) Engagement vote counts are post-publication — kept for this retrospective model; a
prospective variant that drops them falls to ROC-AUC ~0.66. No preprocessing object is
fitted on validation/test/user input.

### 6.7 Evaluation

Deployed Logistic Regression (tab + text), held-out test (`N = 20 864`): **accuracy
0.918, macro-F1 0.858, ROC-AUC 0.964**, **recall on "does not recommend" = 0.876**.
Confusion `[[2804, 398], [1315, 16347]]` — FN (1 315) = a customer who won't recommend,
predicted as recommending (the costly miss); FP (398) = a happy customer flagged (cheap).
Most important metric: **recall(0)**. Error analysis: 22% of the false negatives carry a
4–5★ rating — a positive-reading review with a hidden veto — the irreducible ceiling.

### 6.8 Business interpretation

Whether a customer recommends is **mostly in what they write** (text ~0.965); the
structured fit (skin-type match, category, brand, price, popularity) is a real but weaker,
redundant signal (~0.8). Uses: review-consistency QA (flag 5★ + "would not repurchase"),
cold-start ranking from the **tabular-only** model (no text needed, ~0.8), merchandising
on low-recommend categories. It is retrospective — it needs the review.

### 6.9 Deployment

`POST /predict` (FastAPI) → `{ prediction, confidence, p_recommend, threshold,
review_terms, signals, contributions, model, representation }`. The service loads
`model/model_pipeline.joblib` (`ColumnTransformer` + `TfidfVectorizer` + `LogisticRegression`)
and **never re-fits** it. `contributions` = an exact linear-SHAP decomposition
(`φⱼ = coefⱼ·(xⱼ − x̄ⱼ)`) rendered as a diverging bar chart + waterfall. **Web** = React +
Vite 3-step wizard (skin profile → product → review) then a result screen; **mobile** =
Flutter, two screens, same design tokens (light + dark). Screenshots W1–W5, M1–M4 — see
`Report_App3.md`.

---

## 7. Comparison of the Three Intelligent Systems

| Aspect | Diabetes | House Price | Customer Behavior |
|---|---|---|---|
| Problem type | Classification | Regression | Classification (chosen target) |
| Raw data | CSV | CSV | CSV (+ review text) |
| Observation (one row) | Patient | House | Customer review |
| Input shape | `ℝ^{N×8}` | `ℝ^{N×20}` | `ℝ^{N×28 733}` (tabular ⊕ TF-IDF text, sparse) |
| Target | `{0,1}` | price ∈ ℝ⁺ | `{0,1}` |
| Representation | scaled numeric matrix | one-hot + scaled matrix | matrix + TF-IDF text vectors |
| Best model | «Random Forest» | «Gradient Boosting» | Logistic Regression (tabular + text) |
| Main metric | «Recall / F1» | «RMSE / R²» | recall on "does not recommend" (0.876) at ROC-AUC 0.964 |
| Web deployment | Yes | Yes | Yes (React + Vite 3-step wizard) |
| Mobile deployment | Yes | Yes | Yes (Flutter, 2 screens) |
| Main limitation | «small N, single population» | «no location detail, 545 rows» | review text is co-authored with the recommend tick (~0.96 is partly leakage; leak-safe tabular ~0.8); retrospective only; one product category; ~78% single-review reviewers (RFM Frequency near-degenerate) |

Then discuss (1–8): dataset differences, representation differences, common vs
app-specific preprocessing, why targets differ, why metrics differ, easiest to deploy
(«diabetes — 8 numeric inputs»), most compute-heavy (e-commerce — ~28 700-dim sparse tabular+text, 8 models).

---

## 8. Deployment Architecture (shared by all three)

```
User Input → API Request → Validation → SAME Preprocessing (loaded) → Saved ML Model → Prediction → JSON Result
```

1. Trained model — loaded from `model/model_pipeline.joblib`, **never retrained** at request time.
2. Preprocessing pipeline — the exact `ColumnTransformer`/`Pipeline` fitted on training data, saved with the model.
3. API endpoint — `POST /predict`.
4. Input validation — Pydantic schema (types, ranges, e.g. `0 < Age < 120`).
5. Prediction function — `pipeline.predict(raw_df)` / `predict_proba`.
6. Result — JSON `{prediction, confidence}`.
7. Web UI — form → fetch → result badge.
8. Mobile UI — native form → REST call → result screen.

**Data-leakage warning.** The service loads the preprocessing pipeline *fitted on training
data*. It must **never** call `.fit()` on user input or test data — a scaler re-fitted on
one request would standardise that request against itself and produce nonsense. Training
and inference must apply *identical* transformations.

### Web application template (fill per app)

```
Web Application — Diabetes Prediction
Framework: FastAPI 0.111
Endpoint:  POST /predict
Input:     {Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age}
Output:    {"prediction": "diabetic", "confidence": 0.91}
[ screenshot of working web application ]
Figure explanation: describe the input entered, the prediction, and how to read the confidence.
```

### Mobile application template (fill per app)

```
Mobile Application — Diabetes Prediction
Framework: Flutter 3.x
Platform:  Android
API:       POST http://<host>:8000/predict
[ screenshot of mobile input + prediction screens ]
Figure explanation: how the app builds the JSON body, calls the API, and shows the result + confidence.
```

Mobile must let the user: (1) enter/select input, (2) submit, (3) see the prediction, (4) see a confidence / short explanation.

---

## 9. Reproducibility

For **each** app: Python version «3.11», OS «Windows 10 / Ubuntu 22.04», library versions
(`scikit-learn «1.4.2»`, `pandas «2.2»`, `fastapi «0.111»`, …), `RANDOM_SEED = 42`, Kaggle
dataset URL + download date «2026-08-20», preprocessing steps, feature list + final `d`,
train/val/test split «70/15/15 stratified», model hyperparameters, metrics, saved
`model/model_pipeline.joblib`, API code, web code, mobile code.

```python
import numpy as np, random
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
```

Ship `requirements.txt` per app + a `README.md` that lets another student reproduce the
experiment **and** run each deployed service.

---

## 10. Final Comparative Discussion

Answer the 16 questions across `Data → Representation → Model → Evaluation → Deployment`
(datasets differ, one observation = patient/house/customer-review, targets =
binary/continuous/binary, representations = numeric matrix / one-hot matrix / matrix+text,
diabetes & e-commerce = classification, house = regression, categoricals = one-hot,
numericals = StandardScaler, data-quality issues per set, common preprocessing = impute +
scale + split, best models, most appropriate metrics, whether best model also deployed
best, easiest vs hardest to deploy, leakage forms considered, remaining limitations,
improvements with more time). Include the Cross-Application Comparison table.

---

## 11. Conclusion

```
Raw Data → Clean → Represent → Learn → Evaluate → Persist → Deploy
```

A deployable intelligent system = **data + representation + learning + evaluation +
software + deployment + user interaction**, not just a trained model. Cover: (1) main
lesson, (2) biggest technical challenge, (3) most important data-representation issue,
(4) most important ML lesson, (5) most important deployment lesson, (6) one future improvement.

---

## Mandatory Data-Representation Summary (must appear in the report)

| Application | Raw form | Numerical representation | Model input |
|---|---|---|---|
| Diabetes | CSV / table | Scaled feature matrix | `B × d` = «B × 8» |
| House price | CSV / table | One-hot + scaled feature matrix | `B × d` = «B × 20» |
| E-commerce | CSV + review comments | tabular one-hot+scaled + TF-IDF text vectors | `B × d` = `N × 28 733` sparse (N = 104 313); embedding demo `B × T × d` = `1 × 40 × 16` |

**Every dimension above must be explained** (what B, d, T mean and where the numbers come from).
