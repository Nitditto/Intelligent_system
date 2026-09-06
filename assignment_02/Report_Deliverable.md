# Assignment 02 — From Data Representation to Deployable Intelligent Systems

**Report deliverable — all three applications.**
Framework applied identically to each app: `Data → Understand → Clean → Represent →
Learn → Evaluate → Persist → Deploy`. Every number below is the executed value from the
corresponding `notebook/*.ipynb` (Python 3.13, Windows 11, scikit-learn 1.9.0,
`RANDOM_SEED = 42`). The 📸 markers below point at captured images in
`report/screenshots/` — see that folder's `README.md` for the file-to-section index.

---

## Cover Page

```
INTELLIGENT SYSTEM DEVELOPMENT
ASSIGNMENT 02
From Data Representation to Deployable Intelligent Systems

Student name: ____________________
Student ID:   ____________________
Class:        ____________________
Team:         ____________________
Lecturer:     Dinh Que Tran, Ph.D., Assoc. Prof.
Semester:     I.2026
```

---

## 1. Executive Summary

This report builds three deployable intelligent systems from raw Kaggle data: a
**diabetes screening classifier**, a **house-price regressor**, and an **e-commerce
product-recommendation classifier that also uses review text**. Each is taken through the
full pipeline and exposed as a FastAPI `POST /predict` service consumed by a React/Vite
web page and a Flutter mobile app.

| Application | Task | Main representation |
|---|---|---|
| Diabetes | Binary classification — diabetic vs not | scaled + passthrough feature matrix `X ∈ ℝ^{N×23}` |
| House price | Regression — sale price (log1p, million VND) | imputed + scaled + one-hot matrix `X ∈ ℝ^{N×92}` |
| Customer behaviour | Binary classification — will the reviewer recommend the product | tabular one-hot+scaled matrix `X ∈ ℝ^{N×138}` **+** TF-IDF text vectors `ℝ^{~28 600}` |

| Field | Diabetes | House Price | Customer Behaviour |
|---|---|---|---|
| Dataset | Diabetes Health Indicators (BRFSS 2015) | VN Real Estate Listings (Apr–Sept 2025) | Sephora Products and Skincare Reviews (`nadyinky`) |
| Kaggle | kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset | kaggle.com/datasets/qmanhbeo/vietnamese-real-estate-listings-may-2024 (CC BY-NC 4.0; Guland.vn scrape) | kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews |
| Rows (raw → clean) | 253 680 → 229 781 | 236 226 → 201 654 | 116 262 → 104 313 |
| Target | `Diabetes_binary ∈ {0,1}` (17.3% positive) | `log1p(Price)` in million VND | `is_recommended ∈ {0,1}` (84.7% positive) |
| Selected model | Random Forest | Random Forest | Logistic Regression on tabular + TF-IDF text |
| Headline test metric | ROC-AUC **0.809**, recall(1) **0.735**, F1 0.48, acc 0.725 | RMSE **26 873** (million VND), R² **0.186**, MAE 10 626 | ROC-AUC **0.964**, macro-F1 **0.858**, recall(0) **0.876**, acc 0.918 |
| Deployment | FastAPI + React/Vite web + Flutter | FastAPI + React/Vite web + Flutter | FastAPI + React/Vite web + Flutter |

---

## 2. Connection to Lecture 02 — Data Representation

```
Real-world object → Raw data → Numerical representation → Tensor → Model
  (patient / house / review)    (CSV row)   (feature vector x)   (X matrix / + E)   (RF / RF / LogReg)
```

- A **tabular sample** is a vector `x = [x₁,…,x_d]ᵀ ∈ ℝ^d`; a dataset is a matrix
  `X ∈ ℝ^{N×d}` (N rows = samples, d columns = features after encoding).
- **Categorical → numeric** by one-hot encoding, e.g. `Property Type ∈ {Nhà riêng, Căn
  hộ, …} → Nhà riêng = [1,0,0,…]`. Binary survey answers are already `{0,1}`; ordinal
  survey codes (`GenHlth 1–5`, `Age` bracket 1–13) are kept as monotonic numbers.
- **Numeric → scaled** with `StandardScaler` on the continuous columns; skewed money
  columns get `log1p` first.
- **Text → numbers** by `Text → Tokens → Token IDs → Embeddings`. For App 3 a batch of
  embedded reviews is a 3-D tensor `E ∈ ℝ^{B×T×d}` (B reviews, T tokens each, d numbers
  per token); the deployed model instead uses the 2-D sparse **TF-IDF** form.
- **The representation is part of the solution** — marks are for explaining *how raw data
  becomes computational data*, not only for accuracy.

## 3. Data-Representation Overview

| Application | Raw data | ML representation | Model input |
|---|---|---|---|
| Diabetes | CSV, 1 row = 1 survey respondent | 8 scaled + 15 passthrough columns | `X ∈ ℝ^{229781 × 23}`, one request `ℝ^{1×23}` |
| House price | CSV, 1 row = 1 listing | median-impute → `log1p` (skewed) → `StandardScaler` (numeric); `OneHotEncoder(min_frequency=50)` (categorical) | `X ∈ ℝ^{201654 × 92}`, `y ∈ ℝ^N` (log1p million VND) |
| Customer behaviour | CSV + review text, 1 row = 1 review | tabular one-hot+scaled (`d_tab = 138`) ⊕ TF-IDF 1–2-gram (`d_txt = 28 595`, training vocabulary) | `X = [x_tab ‖ x_txt] ∈ ℝ^{104313 × 28 733}` sparse; embedding demo `E ∈ ℝ^{1×40×16}` |

**Per-application row/column meaning:**

| Question | Diabetes | House price | Customer behaviour |
|---|---|---|---|
| One row represents | one BRFSS phone respondent | one property listing | one product review |
| Input feature columns | 21 raw survey items + 2 engineered | 11 numeric + 7 categorical (post-engineering) | 15 numeric/binary + 6 categorical + 1 text |
| Target column | `Diabetes_binary` | `Price` (modelled as `log1p`) | `is_recommended` |
| Categoricals & encoding | none (binary/ordinal already numeric) | `Property Type / Position / Direction / Road Type / Province / Agent Role / district` → one-hot | `skin_type/skin_tone/eye_color/hair_color/secondary_category/brand_name` → one-hot |
| Final `d` | 23 | 92 | 28 733 (sparse) |

---

## 4. Application 1 — Diabetes Prediction

### 4.1 Problem description

The objective is to predict whether a survey respondent **has diabetes** from routine
health-behaviour and demographic items, so a primary-care service can prioritise
high-risk people for a confirmatory blood test. `X` = 21 BRFSS survey items
(`HighBP, HighChol, BMI, GenHlth, Age, …`); `y = Diabetes_binary ∈ {0,1}` (derived from
the 3-class `Diabetes_012` as `≥ 1`). **Binary classification.** One observation = one
respondent.

### 4.2 Dataset

| Field | Value |
|---|---|
| Name / source | **Diabetes Health Indicators** (CDC BRFSS 2015, via Kaggle `alexteboul`) |
| Rows | **253 680** → **229 781** after cleaning |
| Features | 21 raw (all numeric: 14 binary, 4 ordinal codes, 3 continuous) + target |
| Target | `Diabetes_binary` — positive rate 15.76% raw → **17.29%** after de-duplication |

### 4.3 Data understanding

`df.shape = (253680, 22)`; all columns numeric (`float64`); **0 explicit NaN**;
**23 899 exact duplicate rows**; `Diabetes_binary` **imbalanced** (17.3% positive) → use
stratified split, `class_weight="balanced"`, judge on recall / F1 / ROC-AUC not accuracy.
`BMI` has a long tail (12–98); `MentHlth` / `PhysHlth` are mostly 0 (31% / 37% non-zero
"unhealthy days") — genuine survey answers, kept.

📸 **N1-D — notebook §4** `df.info()` + `describe()`; 📸 **N2-D — §5** data-quality table.

### 4.4 Data cleaning

| Operation | What | Why |
|---|---|---|
| Drop exact duplicates | 23 899 rows removed **before** the split | otherwise an identical answer vector is in both train and test → optimistic leakage |
| Keep `BMI` outliers | no clip / drop | §9 sensitivity check (clip [12,80] / [14,60] / drop) moves ROC-AUC by ≤ 0.003 → not worth discarding rows |
| Median imputer in the pipeline | fitted on train only | robustness to a missing field at inference, one code path |
| Ordinal codes kept as numbers | `GenHlth/Age/Education/Income` | monotonic with risk; one-hot would add 30 sparse columns for no gain |

### 4.5 Data representation

One raw record → `engineer()` adds `TotalUnhealthyDays = clip(MentHlth+PhysHlth, 0, 60)`
and `CardioRisk = (Stroke ∨ HeartDiseaseorAttack)` → 23 model features (8 continuous/count
`→ StandardScaler`, 15 binary/ordinal passthrough). `X ∈ ℝ^{229781 × 23}`, dense float64;
`y ∈ {0,1}^N`. One API request = `ℝ^{1×23}`.

📸 **N4-D — §12**: the raw record + its 23-dim feature vector + the shape lines.

### 4.6 Exploratory data analysis (≥ 3 plots, each Observation / Interpretation / ML)

- **Target balance** — 17.3% positive → a majority classifier already scores 0.83
  accuracy; use recall / ROC-AUC.
- **`GenHlth` vs diabetes rate** — monotonic: "excellent" ~4% → "poor" ~35% → the single
  strongest predictor.
- **`BMI` distribution by class** — diabetic respondents shifted ~4 BMI points higher;
  overlap is large → BMI helps but is not decisive alone.
- **`Age` bracket vs rate** — rises steadily with age → keep `Age` as an ordinal number.
- **Correlation heatmap** — `HighBP, HighChol, GenHlth, DiffWalk, Age, BMI` most
  correlated with the target; lifestyle items (`Fruits, Veggies`) near zero.

📸 **N5-D — §10** the plot grid; 📸 **N6-D — §10** the correlation heatmap.

### 4.7 Model development

**Five models** (notebook §17), same 25 000-row stratified training subsample and same
preprocessing; the winner is refitted on the full training split.

| model | hyperparameters |
|---|---|
| Logistic Regression | `class_weight="balanced", max_iter=1000` |
| Decision Tree | `max_depth=6, min_samples_leaf=50, class_weight="balanced"` |
| Random Forest | `n_estimators=300, max_depth=12, min_samples_leaf=20, class_weight="balanced"` |
| SVM (RBF) | `class_weight="balanced"` |
| KNN | `n_neighbors` default |

**Split:** stratified 70 / 15 / 15 (`train_test_split` twice, seed 42) → train 160 846 ·
val 34 467 · test 34 468, each at positive rate 0.1729. Every fitted object is fitted on
`X_train` only.

### 4.8 Evaluation & 4.9 comparison

Validation (25 000-row fit):

| model | accuracy | precision | recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Random Forest** | 0.727 | 0.361 | 0.748 | 0.487 | **0.810** |
| Logistic Regression | 0.716 | 0.352 | 0.760 | 0.481 | 0.804 |
| SVM (RBF) | 0.707 | 0.345 | 0.776 | 0.478 | 0.800 |
| KNN | 0.830 | 0.550 | 0.104 | 0.174 | 0.790 |
| Decision Tree | 0.690 | 0.328 | 0.760 | 0.459 | 0.787 |
| *baseline B — 3-feature LogReg* | 0.699 | 0.328 | 0.708 | 0.448 | 0.77 |
| *baseline A — majority class* | 0.827 | 0.000 | 0.000 | 0.000 | 0.50 |

**Held-out test — Random Forest, refit on full train:**

| | precision | recall | F1 | support |
|---|---|---|---|---|
| 0 — not diabetic | 0.929 | 0.723 | 0.813 | 28 509 |
| 1 — diabetic | 0.357 | **0.735** | 0.480 | 5 959 |
| accuracy | | | 0.725 | 34 468 |
| macro avg | 0.643 | 0.729 | 0.647 | |

**ROC-AUC 0.809.** Confusion `[[20612, 7897], [1579, 4380]]` — FN (1 579) = diabetic
people told they are fine (the dangerous error); FP (7 897) = healthy people sent for a
blood test (a cost, not a harm). **Most important metric: recall on class 1 (0.735)** — a
missed diabetic is far worse than a wasted test; the class-weighted RF trades precision
(0.357) for that recall deliberately. Accuracy (0.725) is *below* the 0.83 majority
baseline because the model chooses to flag many at-risk people.

📸 **N7-D — §18** comparison table; 📸 **N8-D — §19** report + confusion heatmap + ROC.

### 4.10 Model selection & deployment

**Random Forest** — highest ROC-AUC, best recall/F1 balance, clears both baselines,
`feature_importances_` for interpretation, trains in ~2 s, one small artifact. Logistic
Regression is the documented lightweight fallback (coefficient-level interpretability, near
the same ROC-AUC). Persisted as `diabetes/model/model_pipeline.joblib` (`Pipeline([prep,
RandomForestClassifier])`) + `feature_names.joblib` + `input_schema.json`. Inference test
(§23): reload-from-disk `predict_proba` == in-memory to 10 decimals; example raw input →
`{"prediction": "diabetic", "confidence": 0.8496}`.

---

## 5. Application 2 — House-Price Prediction

### 5.1 Problem description

Predict a **fair listing price** from property attributes so a marketplace can flag
mispriced listings and help sellers set a number. `X` = property + location + agent
attributes; `y = Price` in million VND, modelled as `log1p(Price)` (raw skew 3.36 →
−0.07 after `log1p`) and inverted with `expm1`. **Regression.** One observation = one
listing. It differs from Application 1: the target is a continuous positive quantity, so
the metrics are MAE / MSE / RMSE / R² (error magnitude and variance explained), not
precision/recall.

### 5.2 Dataset

| Field | Value |
|---|---|
| Name / URL | **Vietnamese Real Estate Listings 2025** (Kaggle, author `qmanhbeo`) — kaggle.com/datasets/qmanhbeo/vietnamese-real-estate-listings-may-2024 |
| Licence / provenance | **CC BY-NC 4.0** (non-commercial, educational); scraped from public sale adverts on **Guland.vn**, `Scraped At` = 12–14 Sept 2025; file `VN-real-estate-Apr-Sept-2025.csv` (~210 MB, not committed) |
| Rows | **236 226** → **201 654** after cleaning |
| Numeric features | `Area, Width, Length, Bedrooms, Bathrooms, Floors, Alley Width, Agent Listing Count` + 3 `*_missing` indicators |
| Categorical features | `Property Type, Position, Direction, Road Type, Province, Agent Role, district` |
| Target | `Price` (million VND) → `log1p` |

### 5.3 Data understanding & cleaning

Heavy, structural missingness: `Bedrooms` 72%, `Bathrooms` 84%, `Floors` 79%, `Length`
58% missing, `Latitude/Longitude` 68% missing. `Price` has 611 non-positive and ~10 600
implausible (> 200 000 million VND) values → **rows dropped**; `Area` similarly. Cleaning:

| Operation | What | Why |
|---|---|---|
| Drop impossible rows | `Price ≤ 0` or `> 200 000`, `Area ≤ 0` or `> 10 000` | not real prices/areas; would dominate the loss |
| `*_missing` indicator features | `Bedrooms_missing`, `Bathrooms_missing`, `Floors_missing` | the *fact* that a field is blank is itself informative (§10.3.2) |
| Median-impute the rest, in the pipeline | fitted on train only | one training/inference code path, no leakage |
| `district` from `Location` string | parsed, one-hot with `min_frequency=50` | 63-province `Province` + district is the main price driver (η² ≈ 0.24) |
| `log1p(Price)` target | | raw skew 3.36; the QQ-plot is straight after `log1p` (§10.3.3) |
| Drop `Listing ID`, `VIP Account` (constant), `Title` (embeds the price) | | ids leak nothing useful; a constant column is dead; the title literally contains the price string |

### 5.4 Representation

Cleaned frame `(201 654, 31)` → `ColumnTransformer`: `SimpleImputer(median) → log1p`
(skewed numeric) `→ StandardScaler`; `OneHotEncoder(handle_unknown="ignore",
min_frequency=50)` on the 7 categoricals → **`X ∈ ℝ^{201654 × 92}`** (sparse);
`y ∈ ℝ^N` = `log1p(Price)`. One API request → `ℝ^{1×92}`; the response inverts with
`expm1`.

📸 **N4-H — §12**: one raw listing + its 92-dim feature row + the shape lines.

### 5.5 Exploratory data analysis

- **10.1 Distributions** — `Price` raw skew 3.36 → `log1p` −0.07; `Area` 5.11 → 1.32.
- **10.2 Relationships & correlation** — surprisingly weak numeric signal:
  `|r(log Area, log Price)| = 0.017`, `Bathrooms` 0.215 is the strongest numeric;
  categorical η² with `log Price`: `Province` 0.237, `Property Type` 0.120. →
  **location + property type carry most of the (limited) signal**; raw area is unreliable
  (mixed units / free-text origin).
- **10.3 Supporting analyses** — missing-by-property-type justifies the `*_missing`
  indicators; the skew/QQ view justifies the `log1p` target.

📸 **N5-H — §10.1** distributions; 📸 **N6-H — §10.2** correlation + η² bars.

### 5.6 Regression models & 5.7 evaluation

**Five models** (notebook §17), 50 000-row training subsample, same preprocessing:

| model | hyperparameters | val MAE | val RMSE | val R² |
|---|---|---|---|---|
| **Random Forest** | `n_estimators` default, bagged | 10 811 | 27 349 | **0.183** |
| XGBoost | gradient boosting | 10 794 | 27 478 | 0.176 |
| Decision Tree | `max_depth=12, min_samples_leaf=25` | 11 522 | 27 992 | 0.144 |
| Linear Regression | — | 12 149 | 29 830 | 0.028 |
| Ridge | `alpha` default | 12 153 | 29 860 | 0.026 |
| *baseline — predict the median* | | 14 397 | 32 185 | −0.131 |

**Held-out test — Random Forest, refit on full train:** MAE **10 626**, MSE
722 149 435, RMSE **26 873**, **R² 0.186**, MAPE 350%. *(All money units = million VND.)*

**Interpretation.** RMSE 26 873 means the typical error is ~27 000 million VND; **R² 0.19**
means the model explains only ~19% of price variance. This is a **property of the
dataset**, not the algorithm — the EDA shows almost no numeric feature correlates with
price (`r(Area, Price) = 0.017`), 70–84% of the size fields are missing, and geolocation
is 68% missing. The linear models (R² ≈ 0.03) confirm there is little linear signal; the
tree ensembles recover what non-linear structure exists (location × property type). The
huge MAPE is dominated by the many low-priced listings (most sit at 2–5 tỷ) where a modest
absolute miss is a large percentage. **The honest headline is "a weak-signal regression
problem"**, reported as a limitation.

📸 **N7-H — §18** model table; 📸 **N8-H — §19** predicted-vs-actual + residual plots.

### 5.8 Model selection & 5.9 deployment

**Random Forest** — lowest test RMSE, highest R² of the five, `feature_importances_`,
~15 s to train, ms per prediction, bagging absorbs the price outliers. Persisted as
`house_price/model/model_pipeline.joblib` (`Pipeline([prep, RandomForestRegressor])`) +
`feature_names.joblib` + `input_schema.json` (records `price_unit = million VND` and the
`log1p`/`expm1` convention). §23 inference test: disk == in-memory.

---

## 6. Application 3 — E-Commerce Customer Behaviour and Interest

### 6.1 Problem description

Predict whether a Sephora skincare reviewer **recommends the product**, from **who they
are** and **what they wrote**. Target `recommended = is_recommended ∈ {0,1}` — a field
*separate* from the 1–5 star rating (excluded as leakage, §6.6a). **Binary
classification.** One observation = one product review. Supports review-consistency QA,
cold-start ranking, and merchandising.

### 6.2 Dataset

| Field | Value |
|---|---|
| Name / URL | **Sephora Products and Skincare Reviews** (`nadyinky`, CC0) — kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews |
| Files | `reviews_500-750.csv` joined to `product_info.csv` on `product_id` |
| Rows | **116 262** → **104 313** after cleaning; positive rate **0.8465** |
| Scope | 249 products · ~79 brands · 12 `secondary_category` values, all Skincare |

**Feature types:** 5 numeric log1p+scale (`price_usd, loves_count, reviews,
total_feedback_count, total_neg_feedback_count`), 4 numeric scale (`n_ingredients,
n_highlights, review_age_days, pos_feedback_ratio`), 6 binary passthrough
(`price_missing, has_title, limited_edition, new, online_only, sephora_exclusive`), 6
one-hot categorical (`skin_type, skin_tone, eye_color, hair_color, secondary_category,
brand_name`), 1 text (`review_all = review_title + " . " + review_text`).
**Excluded (leakage):** the review's own `rating` and the product average `rating_product`.

### 6.3 Customer representation

**Tabular:** one raw review → 21 columns → `SimpleImputer(median) → log1p` (money/
popularity) `→ StandardScaler`; binary passthrough; `OneHotEncoder(handle_unknown=
"ignore", min_frequency=25)` → **`x_tab ∈ ℝ^{138}`**.

**Text — required `Comment → Tokens → Token IDs → Embedding` demo** (notebook §12) on a
real review:

```
Comment  : "I will be the first to say that the price on these is a lot, but for what they do? …"
Tokens   : ['say','price','lot','unbeatable','deep','painful','nodules', …]      (T = 40)
Token IDs: [29, 26, 17, 39, 6, 22, 21, …]                                        (index into a 41-row vocab, 0 = PAD)
Embedding: each ID → a row of a (|V|+1)×d table  →  E ∈ ℝ^{T×d} = ℝ^{40×16}
Batch    : E_batch ∈ ℝ^{B×T×d} = ℝ^{1×40×16}      (B = reviews per batch, T = padded token length, d = embedding width)
```

The **deployed** model uses the sparse **TF-IDF** bag-of-words form
(`ngram_range=(1,2)`, `min_df=10`, `max_features=40000`, `sublinear_tf`, English
stop-words) → `x_txt ∈ ℝ^{28 595}` on the training vocabulary. **Combined:**
`X = [x_tab ‖ x_txt] ∈ ℝ^{104313 × 28 733}` sparse.

📸 **N4-C — §12**: the raw review + tabular row + shapes, and the `B = 1, T = 40, d = 16`
block.

### 6.4 Data cleaning

Drop 11 803 blank-target + 125 empty-body + a few duplicate reviews (116 262 → 104 313).
Skin-profile blanks → explicit `__na__` one-hot level (missingness is a real behaviour).
Incidental numeric gaps → `SimpleImputer(median)` fitted **on train only**. Lower-case all
categoricals; merge `eye_color "grey" → "gray"`. Outliers kept (genuine; `log1p` handles
the skew). TF-IDF vocabulary built **on train only**. The review's own `rating` /
`rating_product` are quarantined to the leakage demo (§6.6a). State *why* for each.

### 6.5 Interest discovery / EDA

Six figures (each Observation / Interpretation / ML implication): class balance 84.7%
recommend; recommend rate spread by skin type ~0.03, category ~0.16, brand ~0.40, price
decile ~0.14; recommend rate vs the *excluded* star rating 0.01 → 1.00 (the leak); topic
buckets — texture/feel ~42% (recommend 0.89), skin outcome/breakouts ~39% (0.90),
irritation ~23%, scent ~26%, price ~13%; frequent terms `skin, product, love, like, use,
face, dry, feel`; distinctive "won't recommend" terms `waste money, disappointed,
returning, meh, just okay`. Mutual information with the target: excluded `rating` 0.35,
`log_loves` 0.037, `log_price` / `brand` 0.018, `secondary_category` 0.002, skin fields
~0. **Customer segmentation (K-Means):** reviewer `xᵢ = [Rᵢ, Fᵢ, Mᵢ, avg_priceᵢ,
recommend-rateᵢ, avg-ratingᵢ, Cᵢ₁…C_ik]`; 73 819 reviewers, `F ≥ 2` for 22%, `k = 5`
(descriptive only, not fed to the classifier).

📸 **N5-C — §10** plot grid; 📸 **N6-C — §10** topic-bucket + mutual-information tables;
📸 **N-B-C — Appendix B** elbow/silhouette + segment profile + PCA scatter.

### 6.6 Model development

**Eight pipelines** — the assignment's six for Application 3 (Logistic Regression,
Decision Tree `max_depth=12`, Random Forest `n_estimators=350`, Linear SVM `LinearSVC
C=0.5`, `SGD(loss="log_loss")` as the text-based linear classifier, `HistGradientBoosting`
as the extra justified model) plus Complement Naive Bayes and the deployed Logistic
Regression on tab+text. All class-weighted; hyperparameters fixed; fitted on the training
split only. **Split:** `StratifiedGroupKFold(n_splits=5)` grouped on `author_id` (no
reviewer spans splits) → train 62 587 · val 20 862 · test 20 864, each at recommend rate
0.8465.

**Representation ladder** (one Logistic Regression, growing features, validation):
skin profile 0.536 → + product 0.656 → **full tabular 0.768** → **text only 0.965** →
**tabular + text 0.966** (ROC-AUC).

**Eight models** (validation):

| model | representation | ROC-AUC | macro-F1 | recall(0) | accuracy |
|---|---|---|---|---|---|
| **Logistic Regression (tab + text)** | combo | **0.966** | **0.861** | **0.885** | 0.919 |
| HistGradientBoosting | tab | 0.811 | 0.664 | 0.678 | 0.771 |
| Random Forest | tab | 0.810 | 0.686 | 0.611 | 0.806 |
| Logistic Regression | tab | 0.768 | 0.622 | 0.679 | 0.722 |
| Decision Tree | tab | 0.767 | 0.637 | 0.649 | 0.746 |
| Linear SVM | tab | 0.767 | 0.622 | 0.676 | 0.723 |
| SGD (log-loss, text) | text | 0.965 | 0.856 | 0.885 | 0.916 |
| Complement Naive Bayes (text) | text | 0.955 | 0.840 | 0.846 | 0.907 |
| *baseline — majority class* | | 0.500 | 0.458 | 0.000 | 0.847 |

**Tabular vs tabular + text — the assignment's key question.** Both representations carry
real skill (tabular ~0.77 linear / ~0.81 trees; a single Decision Tree ~0.77). The review
text reaches ~0.965. Adding text to the tabular block gives a small, consistent lift
(macro-F1 0.856 → 0.861). They are **redundant more than complementary** — a customer who
will withhold a recommendation has usually already said so in the review.

### 6.6a Data leakage (assignment requirement — notebook §14a)

1. **The review text is co-authored with the label** — `is_recommended` is a checkbox on
   the same form as the free-text box; a text model partly *reads* the verdict, so the
   ~0.965 is a retrospective upper bound (not pure leakage — the text also carries
   product experience, so tab+text still survives a temporal split at ~0.96).
2. **The review's own star `rating` IS the label in another column** — adding it →
   ROC-AUC **~0.985** (recommend rate 0.01 → 1.00 across stars; MI 0.35). Excluded. The
   product *average* rating is milder (0.768 → 0.789) but also excluded as look-ahead.
3. **Engagement counts are post-publication** — kept for this retrospective model; a
   prospective variant that drops them falls to ROC-AUC **~0.66**.

No preprocessing object is fitted on validation, test or user input — the deployed API
loads the exact pipeline fitted on `X_train` (§8).

### 6.7 Evaluation

Deployed Logistic Regression (tab + text), held-out test (`N = 20 864`):

| | precision | recall | F1 | support |
|---|---|---|---|---|
| 0 — does not recommend | 0.681 | **0.876** | 0.766 | 3 202 |
| 1 — recommends | 0.976 | 0.926 | 0.950 | 17 662 |
| accuracy | | | **0.918** | 20 864 |
| macro avg | 0.828 | 0.901 | 0.858 | |

**ROC-AUC 0.964.** Same test set, single representation: tabular-only 0.801, text-only
0.963. Confusion `[[2804, 398], [1315, 16347]]` — FN (1 315) = a customer who won't
recommend, predicted as recommending (a misleading product page); FP (398) = a happy
customer flagged (cheap). **Most important metric: recall on class 0 (0.876).** Accuracy
is not the headline — the majority baseline scores 0.847 while catching zero unhappy
reviewers. Error analysis: 22% of the false negatives carry a 4–5★ rating — a
positive-reading review with a hidden veto — the irreducible ceiling.

📸 **N7-C — §18** ladder + 8-model table; 📸 **N8-C — §19** report + confusion + ROC;
📸 **N9-C — §20** FN/FP feature table + sample missed reviews.

### 6.8 Business interpretation

Whether a customer recommends is **mostly in what they write** (text ~0.965); the
structured customer–product fit (skin-type match, category, brand, price, popularity) is a
real but weaker, redundant signal (~0.8). Uses: **review-consistency QA** (flag 5★ +
"would not repurchase"), **cold-start ranking** from the tabular-only model (no text
needed, ~0.8), **merchandising** on low-recommend categories. It is retrospective — it
needs the review; its accuracy edge over a plain text classifier is small, so its value is
the *structured* view.

### 6.9 Deployment

Persisted as `customer_behaviour/model/model_pipeline.joblib` (`ColumnTransformer` +
`TfidfVectorizer` + `LogisticRegression`, ~2 MB) + `feature_names.joblib` +
`feature_means.joblib` (linear-SHAP reference: 37 240 transformed-feature means +
`coef`/`intercept`) + `input_schema.json`. `POST /predict` returns
`{prediction, confidence, p_recommend, threshold, review_terms, signals, contributions,
model, representation}`; `contributions` is an exact linear-SHAP decomposition
(`φⱼ = coefⱼ·(xⱼ − x̄ⱼ)`) rendered as a diverging bar chart + waterfall. §23 inference
test: reload == in-memory; positive review → `p_recommend = 0.9376`, negative → `0.0026`.

---

## 7. Comparison of the Three Intelligent Systems

| Aspect | Diabetes | House Price | Customer Behaviour |
|---|---|---|---|
| Problem type | Classification | Regression | Classification |
| One observation | one survey respondent | one property listing | one product review |
| Target | `Diabetes_binary ∈ {0,1}` | `Price` ∈ ℝ⁺ (modelled as `log1p`) | `is_recommended ∈ {0,1}` |
| Input representation | `ℝ^{N×23}` scaled + passthrough | `ℝ^{N×92}` impute + scale + one-hot (sparse) | `ℝ^{N×28 733}` tabular ⊕ TF-IDF text (sparse) |
| Data-quality issues | 23 899 duplicate rows; 17% imbalance; BMI tail; hidden zeros | 70–84% missing size fields; ~11 000 impossible prices; near-zero numeric correlation; title leaks price | ~10% blank target; 85% imbalance; missing skin-profile fields; text co-authored with the label |
| Best model | Random Forest | Random Forest | Logistic Regression (tabular + text) |
| Main metric | recall on diabetic class — **0.735** at ROC-AUC 0.809 | RMSE **26 873** / R² **0.186** (million VND) | recall on "does not recommend" — **0.876** at ROC-AUC 0.964 |
| Web deployment | Yes (React + Vite) | Yes (React + Vite) | Yes (React + Vite 3-step wizard) |
| Mobile deployment | Yes (Flutter) | Yes (Flutter) | Yes (Flutter, 2 screens) |
| Main limitation | recall bought with low precision (0.36); survey self-report; one country/year | weak price signal — R² ≈ 0.19; `corr(Area, Price) ≈ 0.02` pooled (the land-vs-built mix cancels the area effect — notebook §10.2); Bedrooms/Bathrooms/Floors ~72–84% missing; total price needs finer location than `Province` | review text is co-authored with the recommend tick (~0.96 partly leakage; leak-safe tabular ~0.8); retrospective only; one product category |

**Discussion (1–8).** *(1) Datasets differ* — a balanced-ish health survey, a skewed
real-estate log, and a review corpus with free text. *(2) Representations differ* — a
plain scaled matrix, an imputed one-hot matrix, and a matrix concatenated with sparse
text vectors. *(3) Common preprocessing* — median imputation fitted on train, scaling,
stratified/grouped splitting, one saved `Pipeline`. *(4) App-specific* — `log1p` target +
`*_missing` indicators (house price); duplicate drop + engineered risk flags (diabetes);
`OneHotEncoder(min_frequency)` + `TfidfVectorizer` + author-grouped split + a leakage
section (customer behaviour). *(5) Targets differ* because the questions differ —
"is this true?" (0/1) vs "how much?" (ℝ⁺) vs "will they endorse?" (0/1). *(6) Metrics
differ* — a classifier is judged on the confusion matrix / ROC-AUC (and, when imbalanced,
recall on the rare class), a regressor on error magnitude (MAE/RMSE) and variance
explained (R²). *(7) Easiest to deploy* — diabetes: 23 numeric inputs, no text, one small
form. *(8) Most compute-heavy* — customer behaviour: a ~28 700-dim sparse tabular+text
matrix and eight models.

---

## 8. Deployment Architecture (shared by all three)

```
User Input → API Request → Validation → SAME Preprocessing (loaded) → Saved ML Model → Prediction → JSON Result
```

1. **Trained model** — loaded from `model/model_pipeline.joblib`, **never retrained** at
   request time.
2. **Preprocessing pipeline** — the exact `ColumnTransformer` / `Pipeline` fitted on the
   training split, saved inside the same object.
3. **API endpoint** — FastAPI `POST /predict` (`GET /healthz`, `GET /model-info`,
   `GET /questions` for the clients to render their forms).
4. **Input validation** — a Pydantic schema (types, ranges; missing optional fields are
   median-imputed by the pipeline).
5. **Prediction function** — `pipeline.predict_proba` (classification) /
   `expm1(pipeline.predict)` (house-price regression).
6. **Result** — JSON `{prediction, confidence}` / `{predicted_price}` /
   `{prediction, p_recommend, contributions, …}`.
7. **Web UI** — React + Vite form → `fetch` → result screen with a plain-language verdict.
8. **Mobile UI** — Flutter form → REST call → result screen.

**Data-leakage rule.** The service loads the preprocessing pipeline *fitted on training
data* and **never calls `.fit()`** on a request. A scaler re-fitted on one request would
standardise that request against itself. Training and inference apply *identical*
transforms — this is verified in each notebook's §23 inference test (disk == in-memory).

### Web application — per app

```
Web Application — Diabetes Prediction
Framework: React + Vite  +  FastAPI (POST /predict)
Input:     21 BRFSS survey items (HighBP, HighChol, BMI, GenHlth, Age, …)
Output:    { "prediction": "diabetic", "confidence": 0.85 }

Web Application — House-Price Prediction
Framework: React + Vite  +  FastAPI (POST /predict)
Input:     Area, Property Type, Province/district, Bedrooms, Bathrooms, Floors, Direction, Road Type, …
Output:    { "predicted_price": 2150 }        # million VND

Web Application — Customer Behaviour (product recommendation)
Framework: React + Vite (3-step wizard)  +  FastAPI (POST /predict)
Input:     skin_type/skin_tone/eye_color/hair_color, secondary_category, brand_name,
           price_usd, loves_count, reviews, review_title, review_text        (rating NOT sent)
Output:    { "prediction": "not recommend", "p_recommend": 0.14, "review_terms": {…}, "contributions": {…} }
```

📸 **W1–W3 per app** — input screen, an example of valid input, and the result screen with
its confidence/probability and one-line interpretation.

### Mobile application — per app

```
Mobile Application — <app>
Framework: Flutter        Platform: Android (emulator / device)
API:       POST http://<host>:8000/predict     (same endpoint as the web app)
Request:   JSON body of the fields above       Response: the JSON above
```

The mobile app performs **no training** — `Training ≠ Inference`. It lets the user enter
input, submit, and see the prediction + confidence + a short explanation.

📸 **M1–M2 per app** — the input screen and the result screen. 📸 **M-evidence** — a
`POST /predict 200` line in the Uvicorn log while the app is used.

---

## 9. Reproducibility

| Item | Diabetes | House Price | Customer Behaviour |
|---|---|---|---|
| Python / OS | 3.13 / Windows 11 | 3.13 / Windows 11 | 3.13 / Windows 11 |
| Key libraries | scikit-learn 1.9.0, numpy, pandas, scipy, matplotlib; fastapi + uvicorn + pydantic 2 | + xgboost | + `TfidfVectorizer` (scikit-learn); React 18 + Vite 5; Flutter 3.19+ |
| Random seed | `RANDOM_SEED = 42` (numpy + `random`, every split and estimator) | same | same |
| Dataset | Kaggle `alexteboul/diabetes-health-indicators-dataset`, `diabetes_012_health_indicators_BRFSS2015.csv` — committed | Kaggle `qmanhbeo/vietnamese-real-estate-listings-may-2024` (CC BY-NC 4.0), `VN-real-estate-Apr-Sept-2025.csv` — **not committed** (~210 MB) | Kaggle `nadyinky/sephora-products-and-skincare-reviews` — **not committed** (~505 MB), re-download `reviews_500-750.csv` + `product_info.csv` into `data/sephora/` |
| Rows | 253 680 → 229 781 | 236 226 → 201 654 | 116 262 → 104 313 |
| Split | stratified 70/15/15 (train 160 846 / val 34 467 / test 34 468) | 70/15/15 (train 141 157 / val 30 248 / test 30 249) | `StratifiedGroupKFold(5)` on `author_id` (train 62 587 / val 20 862 / test 20 864) |
| Preprocessing | `SimpleImputer(median)` → `StandardScaler` (8 cols) + passthrough (15) | `SimpleImputer(median)` → `log1p` → `StandardScaler`; `OneHotEncoder(min_frequency=50)` | `SimpleImputer(median)` → `log1p` → `StandardScaler`; `OneHotEncoder(min_frequency=25)`; `TfidfVectorizer(ngram_range=(1,2), min_df=10, max_features=40000, sublinear_tf, stop_words="english")` — all fitted on train only |
| Feature dim `d` | 23 | 92 | 28 733 (sparse) |
| Model | `RandomForestClassifier(n_estimators=300→150, max_depth=12, min_samples_leaf=20, class_weight="balanced", random_state=42)` | `RandomForestRegressor(random_state=42)` on `log1p(Price)` | `LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=42)` on tabular + TF-IDF |
| Test metrics | Acc 0.725 · recall(1) 0.735 · F1 0.480 · ROC-AUC 0.809 · confusion `[[20612,7897],[1579,4380]]` | MAE 10 626 · RMSE 26 873 · R² 0.186 (million VND) | Acc 0.918 · macro-F1 0.858 · ROC-AUC 0.964 · recall(0) 0.876 · confusion `[[2804,398],[1315,16347]]` |
| Saved artifacts | `model/model_pipeline.joblib`, `feature_names.joblib`, `input_schema.json` | same | + `feature_means.joblib` (linear-SHAP) |
| Code | `diabetes/{api,web,mobile}/`, `requirements.txt` | `house_price/{api,web,mobile}/`, `requirements.txt` | `customer_behaviour/{api,web,mobile}/`, `requirements.txt` |
| Reproduce | `pip install -r requirements.txt` → run the notebook (writes `model/`) → `python -m uvicorn api.main:app --port 8000` → `npm --prefix web run dev` → `flutter run` in `mobile/` |

```python
import numpy as np, random
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED); random.seed(RANDOM_SEED)
```

---

## 10. Final Comparative Discussion

Answering the assignment's questions across `Data → Representation → Model → Evaluation →
Deployment`:

1. **Datasets differ** — a 250k-row health survey (all numeric), a 236k-row real-estate
   log (skewed money target, heavy missingness, free text), and a 116k-row review corpus
   (structured fields + English free text).
2. **One observation** = a patient / a house / a customer review.
3. **Targets** — binary / continuous positive / binary.
4. **Computational representation** — scaled matrix `ℝ^{N×23}` / one-hot+scaled matrix
   `ℝ^{N×92}` / matrix ⊕ TF-IDF `ℝ^{N×28 733}` (+ an `ℝ^{B×T×d}` embedding demo).
5–6. Diabetes and customer behaviour are **classification**; house price is
   **regression**.
7. **Categoricals** — one-hot (`OneHotEncoder(handle_unknown="ignore")`, with
   `min_frequency` on the high-cardinality ones); diabetes has none (binary/ordinal
   already numeric).
8. **Numericals** — `StandardScaler`, with `log1p` first on skewed money columns.
9. **Data-quality problems** — duplicates (diabetes); impossible values + massive
   missingness + a target-leaking title column (house price); a blank target + severe
   imbalance + text/label co-authorship (customer behaviour).
10. **Common preprocessing** — median imputation fitted on train, scaling, a stratified
    or grouped train/val/test split, one persisted `Pipeline`.
11. **Best models** — Random Forest, Random Forest, Logistic Regression (tab + text).
12. **Most appropriate metrics** — recall on the rare class + ROC-AUC (both classifiers);
    MAE / RMSE / R² (regressor).
13. **Did the best model deploy best?** Yes for all three — each is a single scikit-learn
    `Pipeline`, CPU-only, ms per prediction.
14–15. **Easiest to deploy** — diabetes (23 numeric inputs, no text). **Hardest** —
    customer behaviour (a 28 700-dim sparse tabular+text vector, a TF-IDF vocabulary to
    ship, a review-text input box).
16. **Leakage forms considered** — a scaler/encoder/vectoriser re-fitted at inference; a
    duplicate row shared by train and test; a title column that embeds the price; the
    review star rating that *is* the recommend label; engagement counts that only exist
    after publication.
17. **Remaining limitations** — diabetes recall is bought with 0.36 precision; house
    price explains only ~19% of variance (weak-signal data); the customer-behaviour text
    score is partly leakage and the model is retrospective.
18. **With more time** — calibrated thresholds and cost-weighted decisions (diabetes);
    join external location/price-per-m² data (house price); an aspect-level text model and
    a multi-category pull to restore RFM Frequency variance (customer behaviour).

---

## 11. Conclusion

```
Raw Data → Clean → Represent → Learn → Evaluate → Persist → Deploy
```

A deployable intelligent system is **data + representation + learning + evaluation +
software + deployment + user interaction**, not just a trained model.

1. **Main lesson** — most of the work and most of the risk is in turning raw data into a
   correct, reproducible representation; the model is a small final step.
2. **Biggest technical challenge** — building **one** preprocessing object that is fitted
   once on training data and applied *identically* in the notebook, the API and the tests.
3. **Most important data-representation issue** — deciding what each raw column *is*
   (a real feature, a leak, a post-hoc signal, an id) before encoding it — most visible in
   Application 3's §14a leakage analysis.
4. **Most important ML lesson** — accuracy is the wrong headline on imbalanced or
   weak-signal data; the confusion matrix / recall on the class that matters, and R² vs a
   naive baseline, tell the real story.
5. **Most important deployment lesson** — `Training ≠ Inference`: the service loads the
   fitted pipeline and never re-fits; identical transforms are what make a saved model
   usable.
6. **One future improvement** — a shared internal library for the three
   `build_features()` / schema / inference code paths so a change is made once.

---

## Mandatory Data-Representation Summary

| Application | Raw form | Numerical representation | Model input |
|---|---|---|---|
| Diabetes | CSV / table | scaled + passthrough feature matrix | `B × d` = `N × 23` (N = 229 781); one request `1 × 23` |
| House price | CSV / table | median-impute → `log1p` → scale (numeric) + one-hot (categorical) | `B × d` = `N × 92` sparse (N = 201 654); `y ∈ ℝ^N` = `log1p(Price)` |
| E-commerce | CSV + review comments | tabular one-hot + scaled ⊕ TF-IDF 1–2-gram text vectors | `B × d` = `N × 28 733` sparse (N = 104 313); embedding demo `B × T × d` = `1 × 40 × 16` |

**Every dimension explained.** `N` = cleaned rows (after §4/§5/§6 filtering). `d` for
diabetes = 8 scaled + 15 passthrough. `d` for house price = 11 numeric + 81 one-hot
columns after `min_frequency=50`. `d` for e-commerce = 138 tabular columns after one-hot
+ 28 595 TF-IDF features on the training vocabulary. In the embedding demo `B = 1` review,
`T = 40` tokens after padding/truncation, `d = 16` embedding width.

---

## Screenshot checklist

Captured screenshots live in **`report/screenshots/`** (see its `README.md` for the full
index and the ports each stack ran on). All three stacks were run locally on 2026-09-07:
FastAPI + Uvicorn backend, React/Vite web, and the Flutter client built for web at a
390 × 844 device viewport.

| ID | App | File(s) in `report/screenshots/` | Shows |
|---|---|---|---|
| W1 (–D/–H/–C) | each | `W1-D_input.png` · `W1-H_input.png` · `W1-C_input.png` | web input screen with a valid example entered |
| W2 (–D/–H/–C) | each | `W2-D_result.png` · `W2-H_result.png` · `W2-C_result.png` | web result screen — probability / predicted value + plain-language interpretation + SHAP |
| W-docs (–D/–H/–C) | each | `W3-D_docs.png` · `W3-H_docs.png` · `W3-C_docs.png` | FastAPI `/docs`, `POST /predict` expanded (request + response schema) |
| M1 (–D/–H/–C) | each | `M1-D_input.png` · `M1-H_input.png` · `M1-C_input.png` | Flutter mobile input screen |
| M2 (–D/–H/–C) | each | `M2-D_result.png` · `M2-H_result.png` · `M2-C_result.png` | Flutter mobile result screen (verdict + confidence/probability); each run logged `POST /predict 200` — the evidence the client calls the service |
| N (–D/–H/–C) | each | `NB_D_*` · `NB_H_*` · `NB_C_*` | notebook figures pulled from the executed `*.ipynb`: §9 outliers, §10 EDA + correlation, §19 confusion/ROC (classifiers) or predicted-vs-actual + residuals (regressor) |

Still to add before the final PDF: the cover-page details (name / ID / class / team) and,
optionally, a notebook `df.info()`/`describe()` grab and the App 3 Appendix-B
elbow/silhouette figure.
