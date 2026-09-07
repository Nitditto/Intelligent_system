**POSTS AND TELECOMMUNICATIONS INSTITUTE OF TECHNOLOGY**

**FACULTY OF INFORMATION TECHNOLOGY**

---

**INTELLIGENT SYSTEM DEVELOPMENT**

**ASSIGNMENT 02**

From Data Representation to Deployable Intelligent Systems

> Student name : Nguyễn Văn Trường
>
> Student ID   : B23DCCE095
>
> Class        : E23CNPM02
>
> Lecturer     : Dinh Que Tran, Ph.D., Assoc. Prof.
>
> Semester     : I.2026

---

*The body (Sections 1-11) is the ~10-page report. The web and mobile screenshots and the
per-app technical templates are in Appendices D and E, which are excluded from the page
count as the assignment allows. Every number is the executed value from the corresponding
`notebook/*.ipynb` (Python 3.13, Windows 11, scikit-learn 1.9.0, `RANDOM_SEED = 42`).*

---

## 1. Introduction and Objectives (Executive Summary)

This report builds three deployable intelligent systems from raw Kaggle data - a **diabetes
screening classifier**, a **house-price regressor**, and an **e-commerce
product-recommendation classifier that also uses review text** - and takes each through the
full pipeline, ending in a FastAPI `POST /predict` service consumed by a React/Vite web
page and a Flutter mobile app. The purpose is to demonstrate how three different kinds of
real-world data become numerical representations that a machine-learning model can process.

| Application | Task | Selected model | Headline test result |
|---|---|---|---|
| Diabetes | Binary classification (diabetic vs not) | Random Forest | ROC-AUC 0.809, recall on diabetic class 0.735, accuracy 0.725 |
| House price | Regression (sale price, million VND) | Random Forest on `log1p(Price)` | MAE 10 626, RMSE 26 873, R^2 0.186 |
| Customer behaviour | Binary classification (will the reviewer recommend) | Logistic Regression on tabular + TF-IDF text | ROC-AUC 0.964, macro-F1 0.858, recall on "does not recommend" 0.876 |

| Application | Dataset (Kaggle) | Rows (raw -> clean) | Representation | Deployment |
|---|---|---|---|---|
| Diabetes | `alexteboul/diabetes-health-indicators-dataset` (BRFSS 2015) | 253 680 -> 229 781 | `X in R^{N x 23}` scaled + passthrough | FastAPI + React/Vite + Flutter |
| House price | `qmanhbeo/vietnamese-real-estate-listings-may-2024` (CC BY-NC 4.0) | 236 226 -> 201 654 | `X in R^{N x 92}` impute + scale + one-hot (sparse) | FastAPI + React/Vite + Flutter |
| Customer behaviour | `nadyinky/sephora-products-and-skincare-reviews` (CC0) | 116 262 -> 104 313 | `X in R^{N x 28 733}` tabular + TF-IDF text (sparse) | FastAPI + React/Vite + Flutter |

---

## 2. Connection to Lecture 02 - Data Representation

```
Real-world object  ->  Raw data  ->  Numerical representation  ->  Tensor  ->  Model
 (patient/house/review)  (CSV row)     (feature vector x)        (X matrix / +E)  (RF/RF/LogReg)
```

- A **tabular sample** is a vector `x = [x_1, ..., x_d]^T in R^d`; a dataset is a matrix
  `X in R^{N x d}` (N rows = samples, d columns = features after encoding).
- **Categorical -> numeric** by one-hot encoding, e.g. `Property Type in {house, apartment,
  ...} -> house = [1, 0, 0, ...]`. Binary survey answers are already `{0, 1}`; ordinal
  survey codes (`GenHlth` 1-5, `Age` bracket 1-13) are kept as monotonic numbers.
- **Numeric -> scaled** with `StandardScaler` on the continuous columns; skewed money
  columns get `log1p` first.
- **Text -> numbers** by `Text -> Tokens -> Token IDs -> Embeddings`. For Application 3 a
  batch of embedded reviews is a 3-D tensor `E in R^{B x T x d}` (B reviews, T tokens each,
  d numbers per token); the deployed model instead uses the 2-D sparse **TF-IDF** form.

The representation is part of the solution: the marks are for explaining *how raw data
becomes computational data*, not only for accuracy.

---

## 3. Data Sources and Application Definitions

For each application, one row of the raw data and the target are defined below; the full
dataset tables (name, URL, observations, attributes, feature names, data types) are in
Sections 4.2, 5.2 and 6.2.

| Question | Diabetes | House price | Customer behaviour |
|---|---|---|---|
| What does one row represent? | one BRFSS phone respondent | one property listing | one product review |
| Raw data representation | CSV, 22 numeric columns | CSV, ~30 mixed columns + free text | 2 joined CSVs, mixed columns + `review_title`/`review_text` |
| Input feature columns | 21 raw survey items + 2 engineered | 11 numeric + 7 categorical (post-engineering) | 15 numeric/binary + 6 categorical + 1 text |
| Target column | `Diabetes_binary in {0,1}` | `Price` (modelled as `log1p`) | `is_recommended in {0,1}` |
| Which features need encoding? | none (binary/ordinal already numeric) | `Property Type / Position / Direction / Road Type / Province / Agent Role / district` -> one-hot | `skin_type / skin_tone / eye_color / hair_color / secondary_category / brand_name` -> one-hot; `review_text` -> TF-IDF |
| Which features need normalization? | 8 continuous/count columns -> `StandardScaler` | numeric block -> `log1p` (skewed) then `StandardScaler` | 9 numeric columns -> `log1p` (money/popularity) then `StandardScaler` |
| Final feature dimension `d` | 23 | 92 | 28 733 (sparse) |
| Model input shape | `X in R^{229 781 x 23}`; one request `R^{1 x 23}` | `X in R^{201 654 x 92}`; one request `R^{1 x 92}` | `X in R^{104 313 x 28 733}`; one request `R^{1 x 28 733}` |

### 3.1 Mandatory Data-Representation Summary

| Application | Raw form | Numerical representation | Model input (`B x d`, or `B x T x d`) |
|---|---|---|---|
| Diabetes | CSV / table | scaled + passthrough feature matrix | `N x 23` (N = 229 781); one request `1 x 23` |
| House price | CSV / table | median-impute -> `log1p` -> scale (numeric) + one-hot (categorical) | `N x 92` sparse (N = 201 654); target `y in R^N` = `log1p(Price)` |
| E-commerce | CSV + review comments | tabular one-hot + scaled, concatenated with a TF-IDF (1-2 gram) of the review text | `N x 28 733` sparse (N = 104 313); tokenisation demo `B x T x d` = `1 x 40 x 16` |

**Every dimension.** `N` = cleaned rows (after the Section 4/5/6 filtering). `d` for diabetes
= 8 scaled + 15 passthrough. `d` for house price = 11 numeric + 81 one-hot columns after
`min_frequency = 50`. `d` for e-commerce = 138 tabular columns after one-hot, plus 28 595
TF-IDF features on the training vocabulary. In the embedding demo `B = 1` review, `T = 40`
tokens after padding/truncation, `d = 16` embedding width.

---

## 4. Application 1 - Diabetes Prediction

### 4.1 Problem description

Predict whether a survey respondent **has diabetes** from routine health-behaviour and
demographic items, so a primary-care service can prioritise high-risk people for a
confirmatory blood test. `X` = 21 BRFSS survey items (`HighBP, HighChol, BMI, GenHlth, Age,
...`); `y = Diabetes_binary in {0, 1}`, derived from the 3-class `Diabetes_012` as `>= 1`.
**Binary classification;** one observation = one respondent.

### 4.2 Dataset

| Field | Value |
|---|---|
| Name / source | **Diabetes Health Indicators** (CDC BRFSS 2015, Kaggle `alexteboul/diabetes-health-indicators-dataset`) |
| Observations | **253 680** raw -> **229 781** after de-duplication |
| Attributes | 21 features + target; all numeric `float64` (14 binary, 4 ordinal codes, 3 continuous) |
| Target | `Diabetes_binary` - positive rate 15.76 % raw -> **17.29 %** after de-duplication |
| One observation | one respondent's answers, e.g. `HighBP=1, HighChol=1, BMI=40, GenHlth=5, Age=9, DiffWalk=1, ...` -> `Diabetes_binary=1` |

### 4.3 Data understanding and quality (notebook Sections 4-9)

`df.shape = (253 680, 22)`; `df.info()` - all columns numeric (`float64`); `df.isna().sum()`
- **0 missing values**; `df.duplicated().sum()` - **23 899 exact duplicate rows**;
`df.describe()` - `BMI` ranges 12-98, `MentHlth`/`PhysHlth` are mostly 0 (genuine survey
answers). There are **no categorical or text columns** and no invalid values. The target is
**imbalanced** (17.3 % positive). The executed inspection/quality cells are in
`diabetes/notebook/diabetes.ipynb` (Sections 4-9).

### 4.4 Data cleaning (notebook Section 13)

| Operation | What | Why |
|---|---|---|
| Drop exact duplicates | 23 899 rows removed **before** the split | an identical answer vector in both train and test is optimistic leakage |
| Keep `BMI` outliers | no clip / drop | the Section 9 sensitivity check (clip to [12,80] / [14,60] / drop) moves ROC-AUC by <= 0.003 |
| Median imputer inside the pipeline | fitted on train only | robustness to a field missing at inference, one code path |
| Ordinal codes kept as numbers | `GenHlth / Age / Education / Income` | monotonic with risk; one-hot would add ~30 sparse columns for no gain |

### 4.5 Data representation (notebook Section 12)

One raw record -> `engineer()` adds `TotalUnhealthyDays = clip(MentHlth + PhysHlth, 0, 60)`
and `CardioRisk = Stroke OR HeartDiseaseorAttack` -> **23 model features** (8
continuous/count -> `StandardScaler`, 15 binary/ordinal passed through). Example:

```
raw    : HighBP=1, HighChol=1, BMI=40.0, GenHlth=5, Age=9, DiffWalk=1, ...
vector : x = [x_1, ..., x_23]^T in R^23   (8 scaled values, 15 passthrough 0/1/ordinal)
matrix : X in R^{229 781 x 23}, dense float64;  y in {0,1}^{229 781}
request: one API call = R^{1 x 23}
```

![Figure 4.1](report/screenshots/NB_D_N4_sec12_representation.png){width=80%}

*Figure 4.1 - Diabetes: one raw record -> its 23-dim feature vector -> X shape / dtype
(notebook Section 12).*

### 4.6 Exploratory data analysis (notebook Section 10)

![Figure 4.2](report/screenshots/NB_D_02_sec10.png){width=80%}

*Figure 4.2 - Diabetes: EDA plot grid - target balance, `GenHlth` vs diabetes rate, `BMI`
distribution by class, `Age` bracket vs rate.*

![Figure 4.3](report/screenshots/NB_D_03_sec10.png){width=72%}

*Figure 4.3 - Diabetes: feature / target correlation heatmap.*

- **Target balance** - 17.3 % positive. *Observation:* a majority classifier already scores
  0.83 accuracy. *ML implication:* judge on recall / F1 / ROC-AUC, not accuracy.
- **`GenHlth` vs diabetes rate** - monotonic, ~4 % at "excellent" to ~35 % at "poor".
  *Interpretation:* self-rated health is the single strongest predictor. *ML implication:*
  keep `GenHlth` as an ordinal number.
- **`BMI` distribution by class** - diabetic respondents ~4 BMI points higher, large
  overlap. *Interpretation:* BMI helps but does not separate the classes alone.
- **`Age` bracket vs rate** - rises steadily with age. *ML implication:* keep `Age`
  ordinal.
- **Correlation heatmap (Figure 4.3)** - `HighBP, HighChol, GenHlth, DiffWalk, Age, BMI`
  are most correlated with the target; lifestyle items (`Fruits, Veggies`) near zero.
  *ML implication:* all columns kept (a tree ensemble is robust to weak features); no single
  feature dominates.

### 4.7 Model development (notebook Sections 14-17)

Five models on the same 25 000-row stratified training subsample and the same preprocessing;
the winner is refitted on the full training split.

| Model | Hyperparameters |
|---|---|
| Logistic Regression | `class_weight="balanced", max_iter=1000` |
| Decision Tree | `max_depth=6, min_samples_leaf=50, class_weight="balanced"` |
| Random Forest | `n_estimators=300, max_depth=12, min_samples_leaf=20, class_weight="balanced"` |
| SVM (RBF) | `class_weight="balanced"` |
| KNN | default `n_neighbors` |

**Split:** stratified 70/15/15 (`train_test_split` twice, seed 42) -> train 160 846 /
val 34 467 / test 34 468, each at positive rate 0.1729. Every fitted object is fitted on
`X_train` only - the test set never influences training or preprocessing.

### 4.8 Evaluation and model comparison (notebook Sections 18-19)

![Figure 4.4](report/screenshots/NB_D_N7_sec18_comparison.png){width=80%}

*Figure 4.4 - Diabetes: five-model comparison table and the stability check (notebook
Section 18).*

![Figure 4.5](report/screenshots/NB_D_04_sec19.png){width=80%}

*Figure 4.5 - Diabetes: classification report, confusion-matrix heatmap and ROC curve for
the held-out test (notebook Section 19).*

Validation (25 000-row fit):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Random Forest** | 0.727 | 0.361 | 0.748 | 0.487 | **0.810** |
| Logistic Regression | 0.716 | 0.352 | 0.760 | 0.481 | 0.804 |
| SVM (RBF) | 0.707 | 0.345 | 0.776 | 0.478 | 0.800 |
| KNN | 0.830 | 0.550 | 0.104 | 0.174 | 0.790 |
| Decision Tree | 0.690 | 0.328 | 0.760 | 0.459 | 0.787 |
| baseline B (3-feature LogReg) | 0.699 | 0.328 | 0.708 | 0.448 | 0.77 |
| baseline A (majority class) | 0.827 | 0.000 | 0.000 | 0.000 | 0.50 |

**Held-out test - Random Forest, refit on full train:** accuracy 0.725, ROC-AUC **0.809**,
recall on the diabetic class **0.735** (precision 0.357, F1 0.480). **Confusion matrix
`[[20 612, 7 897], [1 579, 4 380]]`:** the 1 579 false negatives are diabetic people told
they are fine - the dangerous error; the 7 897 false positives are healthy people sent for
a blood test - a cost, not a harm. **Most important metric: recall on class 1.** A missed
diabetic is far worse than a wasted test, so the class-weighted model trades precision for
recall on purpose, and overall accuracy (0.725) sits below the 0.83 majority baseline.

### 4.9 Model selection

**Random Forest** - highest ROC-AUC, best recall/F1 balance, clears both baselines,
`feature_importances_` for interpretation, ~2 s to train, one small artifact. Logistic
Regression is the documented lightweight fallback (near-identical ROC-AUC, coefficient-level
interpretability).

### 4.10 Persistence and deployment (notebook Sections 22-23)

Persisted as `diabetes/model/model_pipeline.joblib` = `Pipeline([prep,
RandomForestClassifier])`, plus `feature_names.joblib` and `input_schema.json`. The Section
23 inference test reloads the file and confirms `predict_proba` matches the in-memory model
to 10 decimals; example raw input -> `{"prediction": "diabetic", "confidence": 0.8496}`.
Deployment: Section 8, Appendix D.1 (web), Appendix E (mobile).

---

## 5. Application 2 - House-Price Prediction

### 5.1 Problem description

Predict a **fair listing price** from property attributes so a marketplace can flag
mispriced listings. `X` = property + location + agent attributes; `y = Price` in million
VND, modelled as `log1p(Price)` (raw skew 3.36 -> -0.07) and inverted with `expm1`.
**Regression;** one observation = one listing. It differs from Application 1 because the
target is a continuous positive quantity, so the metrics are MAE / MSE / RMSE / R^2, not
precision/recall.

### 5.2 Dataset

| Field | Value |
|---|---|
| Name / URL | **Vietnamese Real Estate Listings 2025** (Kaggle `qmanhbeo/vietnamese-real-estate-listings-may-2024`) |
| Licence / provenance | **CC BY-NC 4.0**; scraped from public sale adverts on Guland.vn, `Scraped At` = 12-14 Sept 2025; file ~210 MB, not committed |
| Observations | **236 226** raw -> **201 654** after cleaning |
| Numeric features | `Area, Width, Length, Bedrooms, Bathrooms, Floors, Alley Width, Agent Listing Count` + 3 `*_missing` indicators |
| Categorical features | `Property Type, Position, Direction, Road Type, Province, Agent Role, district` |
| Target | `Price` (million VND) -> `log1p` |
| One observation | one advert, e.g. `Area=192, Property Type="house", Location="Long Xuyen, An Giang", ...` -> `Price=2000` (million VND) |

### 5.3 Data understanding and cleaning (notebook Sections 4-9, 13)

`df.shape = (236 226, ~30)`; `df.info()` separates numeric columns from categorical/text
(the executed cells are in `house_price/notebook/house_price.ipynb`). `df.isna().sum()`
shows **structural missingness** - `Bedrooms` 72 %, `Bathrooms` 84 %, `Floors` 79 %,
`Length` 58 %, `Latitude`/`Longitude` 68 %. Invalid values: 611 non-positive and ~10 600
implausible (> 200 000 million VND) prices, plus out-of-range areas.

| Operation | What | Why |
|---|---|---|
| Drop impossible rows | `Price <= 0` or `> 200 000`; `Area <= 0` or `> 10 000` | not real values; would dominate the loss |
| `*_missing` indicator features | `Bedrooms_missing, Bathrooms_missing, Floors_missing` | the fact that a field is blank is itself informative |
| Median-impute the rest, inside the pipeline | fitted on train only | one training/inference path, no leakage |
| Parse `district` from `Location` | one-hot with `min_frequency = 50` | province + district is the main price driver (eta^2 ~ 0.24) |
| `log1p(Price)` target | | raw skew 3.36; the QQ-plot is straight after `log1p` |
| Drop `Listing ID`, `VIP Account` (constant), `Title` | | ids leak nothing; a constant column is dead; **the title literally contains the price string** |

### 5.4 Representation (notebook Section 12)

Cleaned frame `(201 654, 31)` -> `ColumnTransformer`: `SimpleImputer(median) -> log1p ->
StandardScaler` (numeric); `OneHotEncoder(handle_unknown="ignore", min_frequency=50)` (7
categoricals) -> **`X in R^{201 654 x 92}`** (sparse); `y in R^N` = `log1p(Price)`. One API
request -> `R^{1 x 92}`; the response inverts with `expm1`.

![Figure 5.1](report/screenshots/NB_H_N4_sec12_representation.png){width=80%}

*Figure 5.1 - House price: one raw listing -> its 92-dim feature row -> X / y shape
(notebook Section 12).*

### 5.5 Exploratory data analysis (notebook Section 10)

![Figure 5.2](report/screenshots/NB_H_02_sec10.1.png){width=72%}

*Figure 5.2 - House price: target distribution - `Price` raw vs `log1p(Price)`.*

![Figure 5.3](report/screenshots/NB_H_07_sec10.2.png){width=72%}

*Figure 5.3 - House price: numeric correlation heatmap.*

- **Distributions (10.1)** - `Price` raw skew 3.36 -> `log1p` -0.07; `Area` skew 5.11 ->
  1.32. *ML implication:* train on `log1p(Price)`, report metrics back on the million-VND
  scale.
- **Relationships and correlation (10.2, Figure 5.3)** - **surprisingly weak numeric
  signal:** `|r(log Area, log Price)| ~ 0.02`, `Bathrooms` 0.215 is the strongest numeric;
  categorical correlation ratios `Province` eta^2 ~ 0.24, `Property Type` ~ 0.12.
  *Interpretation:* location and property type carry most of the (limited) signal; raw area
  is unreliable because listings mix land and built-up property. *ML implication:* this is a
  weak-signal regression problem - a tree ensemble captures the location x type structure,
  and the low R^2 is reported honestly as a data limitation.
- **Supporting analyses (10.3)** - missingness by property type justifies the `*_missing`
  indicators; the skew/QQ view justifies the `log1p` target.

### 5.6 Regression models and evaluation (notebook Sections 16-19)

![Figure 5.4](report/screenshots/NB_H_N7_sec18_comparison.png){width=80%}

*Figure 5.4 - House price: five-model regression comparison table (notebook Section 18).*

![Figure 5.5](report/screenshots/NB_H_18_sec19.png){width=80%}

*Figure 5.5 - House price: predicted-vs-actual scatter and residuals-vs-actual for the
Random Forest (notebook Section 19).*

Five models, 50 000-row training subsample, same preprocessing:

| Model | val MAE | val RMSE | val R^2 |
|---|---|---|---|
| **Random Forest** | 10 811 | 27 349 | **0.183** |
| XGBoost | 10 794 | 27 478 | 0.176 |
| Decision Tree (`max_depth=12, min_samples_leaf=25`) | 11 522 | 27 992 | 0.144 |
| Linear Regression | 12 149 | 29 830 | 0.028 |
| Ridge | 12 153 | 29 860 | 0.026 |
| baseline (predict the median) | 14 397 | 32 185 | -0.131 |

**Held-out test - Random Forest, refit on full train:** MAE **10 626**, MSE 722 149 435,
RMSE **26 873**, **R^2 0.186**, MAPE 350 % (all money in million VND). *What each metric
means here:* MAE - the average listing is off by about 10.6 billion VND, large because the
price range spans three orders of magnitude; RMSE is ~2.5x the MAE, so a minority of
listings are missed by far more than the typical one; R^2 0.186 - the model explains ~19 %
of the variance in `log1p(Price)` beyond a constant-mean predictor (which scores R^2 =
-0.13), so it adds real signal but most price variance is **not recoverable from these
fields**; MAPE is inflated by the many low-priced listings and is reported, not used to
rank models. There is no confusion matrix (regression). **The honest headline is a
weak-signal regression problem**, driven by the near-zero numeric correlation, the 70-84 %
missing size fields and 68 % missing geolocation - a property of the dataset, not the
algorithm.

### 5.7 Model selection, persistence and deployment (notebook Sections 21-23)

**Random Forest** - lowest test RMSE, highest R^2, `feature_importances_`, ~15 s to train,
bagging absorbs the price outliers. Persisted as `house_price/model/model_pipeline.joblib`
(`Pipeline([prep, RandomForestRegressor])`) + `feature_names.joblib` + `input_schema.json`
(records `price_unit = million VND` and the `log1p`/`expm1` convention). Section 23
inference test: disk == in-memory. Deployment: Section 8, Appendix D.2, Appendix E.

---

## 6. Application 3 - E-Commerce Customer Behaviour and Interest

### 6.1 Problem description

Predict whether a Sephora skincare reviewer **recommends the product**, from **who they
are** and **what they wrote**. Target `is_recommended in {0, 1}` - a field *separate* from
the 1-5 star rating (the rating is excluded as leakage, Section 6.6). **Binary
classification;** one observation = one product review. Supports review-consistency QA,
cold-start ranking, and merchandising.

### 6.2 Dataset

| Field | Value |
|---|---|
| Name / URL | **Sephora Products and Skincare Reviews** (Kaggle `nadyinky/sephora-products-and-skincare-reviews`, CC0) |
| Files | `reviews_500-750.csv` joined to `product_info.csv` on `product_id` |
| Observations | **116 262** raw -> **104 313** after cleaning; positive rate **0.8465** |
| Scope | 249 products / ~79 brands / 12 `secondary_category` values, all Skincare |
| Feature types | 5 numeric log1p+scale (`price_usd, loves_count, reviews, total_feedback_count, total_neg_feedback_count`), 4 numeric scale (`n_ingredients, n_highlights, review_age_days, pos_feedback_ratio`), 6 binary passthrough, 6 one-hot categorical (`skin_type, skin_tone, eye_color, hair_color, secondary_category, brand_name`), 1 text (`review_title + " . " + review_text`) |
| Excluded (leakage) | the review's own `rating` and the product average `rating_product` |
| One observation | one review: a skin profile + product fields + a title and body of free text -> `is_recommended in {0,1}` |

`df.shape = (116 262, ~30)`; `df.isna().sum()` - 11 803 rows with a blank target, many
blank skin-profile fields; `df.duplicated().sum()` - a few exact duplicates; 125 empty
bodies; class balance **84.7 % positive**. The executed cells are in
`customer_behaviour/notebook/customer_behaviour.ipynb` (Sections 4-9).

### 6.3 Customer representation (notebook Section 12)

**Tabular:** one raw review -> 21 columns -> `SimpleImputer(median) -> log1p` (money /
popularity) `-> StandardScaler`; binary passthrough; `OneHotEncoder(handle_unknown=
"ignore", min_frequency=25)` -> **`x_tab in R^{138}`**.

**Text - the required `Comment -> Tokens -> Token IDs -> Embedding` demo** on a real review:

```
Comment  : "I will be the first to say that the price on these is a lot, but for what they do? ..."
Tokens   : ['say', 'price', 'lot', 'unbeatable', 'deep', 'painful', 'nodules', ...]   (T = 40)
Token IDs: [29, 26, 17, 39, 6, 22, 21, ...]   (index into a 41-row vocab, 0 = PAD)
Embedding: each ID -> a row of a (|V|+1) x d table  ->  E in R^{T x d} = R^{40 x 16}
Batch    : E_batch in R^{B x T x d} = R^{1 x 40 x 16}
```

The **deployed** model uses the sparse **TF-IDF** form (`ngram_range=(1,2)`, `min_df=10`,
`max_features=40000`, `sublinear_tf`, English stop-words) -> `x_txt in R^{28 595}` on the
training vocabulary. **Combined:** `X = [x_tab || x_txt] in R^{104 313 x 28 733}` sparse.

![Figure 6.1](report/screenshots/NB_C_N4_sec12_representation.png){width=80%}

*Figure 6.1 - Customer behaviour: one raw review -> its tabular row and TF-IDF block, and
the `B = 1, T = 40, d = 16` embedding demo (notebook Section 12).*

### 6.4 Data cleaning (notebook Section 13)

Drop the 11 803 blank-target rows, 125 empty bodies and a few duplicates
(116 262 -> 104 313). Skin-profile blanks -> an explicit `__na__` one-hot level (missingness
is a real behaviour). Incidental numeric gaps -> `SimpleImputer(median)` fitted **on train
only**. Lower-case all categoricals; merge `eye_color "grey" -> "gray"`. Outliers kept
(`log1p` handles the skew). The TF-IDF vocabulary is built **on train only**. The review's
own `rating` / `rating_product` are quarantined to the leakage demo.

### 6.5 Interest discovery / EDA (notebook Section 10)

![Figure 6.2](report/screenshots/NB_C_02_sec10.png){width=80%}

*Figure 6.2 - Customer behaviour: EDA plot grid - class balance, recommend rate by skin
type / category / brand / price decile, recommend rate vs the excluded star rating.*

![Figure 6.3](report/screenshots/NB_C_N6_sec10_topics_mi.png){width=80%}

*Figure 6.3 - Customer behaviour: topic-bucket recommend rates and mutual information with
the target.*

![Figure 6.4](report/screenshots/NB_C_NB_appendixB_kmeans_a.png){width=60%}

*Figure 6.4 - Customer segmentation (K-Means): elbow and silhouette (notebook Appendix B).*

![Figure 6.5](report/screenshots/NB_C_NB_appendixB_kmeans_b.png){width=60%}

*Figure 6.5 - Customer segmentation (K-Means): 5-segment profile and PCA scatter.*

- **Class balance** 84.7 % recommend. Recommend-rate spread by skin type ~0.03, by category
  ~0.16, by brand ~0.40, by price decile ~0.14; against the *excluded* star rating it runs
  0.01 -> 1.00. *Interpretation:* the structured customer-product fit is a real but weak
  signal; the star rating is essentially the label. *ML implication:* `class_weight=
  "balanced"`, group the split on `author_id`, quarantine `rating`, expect the text to
  carry most of the skill.
- **Topic buckets (Figure 6.3)** - texture/feel ~42 % of reviews (recommend 0.89), skin
  outcome/breakouts ~39 % (0.90), irritation ~23 %, scent ~26 %, price ~13 %. **Mutual
  information** with the target: excluded `rating` 0.35, `log_loves` 0.037,
  `log_price`/`brand` 0.018, `secondary_category` 0.002, skin fields ~0. *Interpretation:*
  what the reviewer writes about predicts the recommendation; demographics do not.
- **Customer segmentation (K-Means, Figures 6.4-6.5)** - reviewer `x_i = [R_i, F_i, M_i,
  avg_price_i, recommend_rate_i, avg_rating_i, C_i1..C_ik]`; 73 819 reviewers, `F >= 2` for
  22 %, `k = 5` from the elbow/silhouette. Descriptive only; not fed to the classifier.

### 6.6 Data leakage (assignment requirement; notebook Section 14a)

1. **The review text is co-authored with the label** - `is_recommended` is a checkbox on
   the same form as the free-text box, so a text model partly *reads* the verdict; the
   ~0.965 is a retrospective upper bound (not pure leakage - the text also carries product
   experience, and tab+text still survives a temporal split at ~0.96).
2. **The review's own star `rating` IS the label in another column** - adding it -> ROC-AUC
   **~0.985** (MI 0.35). Excluded. The product *average* rating is milder (0.768 -> 0.789)
   but also excluded as look-ahead.
3. **Engagement counts are post-publication** - kept for this retrospective model; a
   prospective variant that drops them falls to ROC-AUC ~0.66.

No preprocessing object is fitted on validation, test or user input - the deployed API
loads the exact pipeline fitted on `X_train`.

### 6.7 Model development (notebook Sections 16-18)

**Eight pipelines** - the assignment's six for Application 3 (Logistic Regression, Decision
Tree `max_depth=12`, Random Forest `n_estimators=350`, Linear SVM `LinearSVC C=0.5`,
`SGD(loss="log_loss")` as the text-based linear classifier, `HistGradientBoosting` as the
extra justified model) plus Complement Naive Bayes and the deployed Logistic Regression on
tabular + text. All class-weighted; hyperparameters fixed; fitted on the training split
only. **Split:** `StratifiedGroupKFold(5)` grouped on `author_id` (no reviewer spans
splits) -> train 62 587 / val 20 862 / test 20 864, each at recommend rate 0.8465.

**Representation ladder** (one Logistic Regression, growing features, validation ROC-AUC):
skin profile 0.536 -> + product 0.656 -> full tabular 0.768 -> text only 0.965 ->
**tabular + text 0.966**.

| Model | Representation | ROC-AUC | macro-F1 | recall(0) | Accuracy |
|---|---|---|---|---|---|
| **Logistic Regression (tab + text)** | combo | **0.966** | **0.861** | **0.885** | 0.919 |
| HistGradientBoosting | tabular | 0.811 | 0.664 | 0.678 | 0.771 |
| Random Forest | tabular | 0.810 | 0.686 | 0.611 | 0.806 |
| Logistic Regression | tabular | 0.768 | 0.622 | 0.679 | 0.722 |
| Decision Tree | tabular | 0.767 | 0.637 | 0.649 | 0.746 |
| Linear SVM | tabular | 0.767 | 0.622 | 0.676 | 0.723 |
| SGD (log-loss, text) | text | 0.965 | 0.856 | 0.885 | 0.916 |
| Complement Naive Bayes | text | 0.955 | 0.840 | 0.846 | 0.907 |
| baseline (majority class) | | 0.500 | 0.458 | 0.000 | 0.847 |

**Tabular vs tabular + text - the assignment's key question.** Both representations carry
real skill (tabular ~0.77 linear / ~0.81 trees). The review text alone reaches ~0.965;
adding it to the tabular block gives only a small, consistent lift (macro-F1 0.856 ->
0.861). They are **redundant more than complementary** - a customer who will withhold a
recommendation has usually already said so in the review.

### 6.8 Evaluation and error analysis (notebook Sections 19-20)

![Figure 6.6](report/screenshots/NB_C_N7_sec18_comparison.png){width=80%}

*Figure 6.6 - Customer behaviour: the representation ladder and the 8-model comparison
table (notebook Section 18).*

![Figure 6.7](report/screenshots/NB_C_03_sec19.png){width=80%}

*Figure 6.7 - Customer behaviour: classification report, confusion-matrix heatmap and ROC
curve for the deployed Logistic Regression on tabular + TF-IDF text (notebook Section 19).*

![Figure 6.8](report/screenshots/NB_C_N9_sec20_error_analysis.png){width=80%}

*Figure 6.8 - Customer behaviour: false-negative / false-positive feature comparison with
sample missed reviews (notebook Section 20).*

Deployed Logistic Regression (tab + text), held-out test (`N = 20 864`): accuracy **0.918**,
ROC-AUC **0.964**, recall on "does not recommend" **0.876** (precision 0.681, F1 0.766).
Single-representation on the same test set: tabular-only 0.801, text-only 0.963. **Confusion
matrix `[[2 804, 398], [1 315, 16 347]]`:** the 1 315 false negatives are customers who
will not recommend but are predicted as recommending - a misleading product page; the 398
false positives are happy customers flagged for QA - cheap. **Most important metric: recall
on class 0.** Accuracy is not the headline - the majority baseline already scores 0.847
while catching zero unhappy reviewers. **Error analysis:** 22 % of the false negatives carry
a 4-5 star rating - a positive-reading review with a hidden veto, which is the irreducible
ceiling for a text model.

### 6.9 Business interpretation

Whether a customer recommends is **mostly in what they write** (text ~0.965); the
structured customer-product fit is a real but weaker, redundant signal (~0.8). Uses:
**review-consistency QA** (flag "5 stars" + "would not repurchase"), **cold-start ranking**
from the tabular-only model (no text needed, ~0.8), **merchandising** on low-recommend
categories. The model is retrospective (it needs the review), so its value is the
*structured* view, not the small accuracy edge over a plain text classifier.

### 6.10 Persistence and deployment (notebook Sections 22-23)

Persisted as `customer_behaviour/model/model_pipeline.joblib` (`ColumnTransformer` +
`TfidfVectorizer` + `LogisticRegression`, ~2 MB) + `feature_names.joblib` +
`feature_means.joblib` (linear-SHAP reference) + `input_schema.json`. `POST /predict`
returns `{prediction, confidence, p_recommend, threshold, review_terms, signals,
contributions, model, representation}`, where `contributions` is an exact linear-SHAP
decomposition. Section 23 inference test: reload == in-memory; positive review ->
`p_recommend = 0.9376`, negative -> `0.0026`. Deployment: Section 8, Appendix D.3,
Appendix E.

---

## 7. Comparison of the Three Intelligent Systems

| Aspect | Diabetes | House Price | Customer Behaviour |
|---|---|---|---|
| Problem type | Classification | Regression | Classification |
| One observation | one survey respondent | one property listing | one product review |
| Target | `Diabetes_binary in {0,1}` | `Price in R+` (as `log1p`) | `is_recommended in {0,1}` |
| Input representation | `R^{N x 23}` scaled + passthrough | `R^{N x 92}` impute + scale + one-hot (sparse) | `R^{N x 28 733}` tabular + TF-IDF text (sparse) |
| Data-quality issues | 23 899 duplicate rows; 17 % imbalance; BMI tail | 70-84 % missing size fields; ~11 000 impossible prices; near-zero numeric correlation; title leaks price | ~10 % blank target; 85 % imbalance; missing skin-profile fields; text co-authored with the label |
| Best model | Random Forest | Random Forest | Logistic Regression (tab + text) |
| Main metric | recall on diabetic class - **0.735** at ROC-AUC 0.809 | RMSE **26 873** / R^2 **0.186** | recall on "does not recommend" - **0.876** at ROC-AUC 0.964 |
| Web deployment | Yes | Yes | Yes (3-step wizard) |
| Mobile deployment | Yes | Yes | Yes |
| Main limitation | recall bought with 0.36 precision; survey self-report | weak price signal (R^2 ~ 0.19); needs finer location than province | text co-authored with the tick (~0.96 partly leakage; leak-safe tabular ~0.8); retrospective |

**Discussion.** The three datasets need three different representations: a plain scaled
matrix, an imputed one-hot matrix, and a matrix concatenated with sparse text vectors. The
common preprocessing is median imputation fitted on train, scaling, a stratified or
author-grouped split, and one persisted `Pipeline`; the application-specific parts are the
`log1p` target and `*_missing` indicators (house price), the duplicate drop and engineered
risk flags (diabetes), and the `OneHotEncoder(min_frequency)` + `TfidfVectorizer` +
author-grouped split + leakage section (customer behaviour). The targets differ because the
questions differ ("is this true?" / "how much?" / "will they endorse?"), which is why a
classifier is judged on the confusion matrix and ROC-AUC while a regressor is judged on
error magnitude and variance explained. Diabetes is the easiest to deploy (23 numeric
inputs, no text); customer behaviour is the most compute-heavy (a ~28 700-dim sparse
tabular+text matrix, a TF-IDF vocabulary to ship, and eight models).

---

## 8. Deployment Architecture

All three applications follow the same inference path:

```
User input  ->  API request  ->  validation  ->  SAME preprocessing (loaded)  ->  saved model  ->  prediction  ->  JSON
```

1. **Trained model** - loaded from `model/model_pipeline.joblib`, **never retrained** at
   request time.
2. **Preprocessing pipeline** - the exact `ColumnTransformer` / `Pipeline` fitted on the
   training split, saved inside the same object.
3. **API endpoint** - FastAPI `POST /predict` (plus `GET /healthz`, `GET /model-info`,
   `GET /questions` for the clients to render their forms).
4. **Input validation** - a Pydantic schema (types, ranges); missing optional fields are
   median-imputed by the pipeline.
5. **Prediction** - `pipeline.predict_proba` (classification) or `expm1(pipeline.predict)`
   (house-price regression).
6. **Result** - JSON: `{prediction, confidence}` / `{predicted_price}` /
   `{prediction, p_recommend, contributions, ...}`.
7. **Web UI** - React + Vite form -> `fetch` -> result screen with a plain-language verdict.
8. **Mobile UI** - Flutter form -> REST call -> result screen. `Training != Inference`: the
   mobile app trains nothing, it calls the same endpoint.

**Data-leakage rule.** The service loads the preprocessing pipeline *fitted on training
data* and **never calls `.fit()`** on a request; a scaler re-fitted on one request would
standardise it against itself. Training and inference apply identical transforms - verified
by each notebook's Section 23 inference test (disk == in-memory).

Per-app request/response summary:

| App | Endpoint | Example input | Example output |
|---|---|---|---|
| Diabetes | `POST :8000/predict` | `{Age, Sex, HighBP, HighChol, BMI, GenHlth, DiffWalk, ...}` | `{"prediction": "at risk", "confidence": 0.82, "band": "High"}` |
| House price | `POST :8002/predict` | `{Area, "Property Type", Province, district, Bedrooms, ...}` | `{"predicted_price": 2540.82, "price_per_m2": 32.28, "currency": "million VND"}` |
| Customer behaviour | `POST :8000/predict` | `{skin_type, secondary_category, brand_name, price_usd, review_title, review_text}` (no `rating`) | `{"prediction": "not recommend", "p_recommend": 0.0026, "review_terms": {...}, "contributions": [...]}` |

![Figure 8.1](report/screenshots/W2-H_result.png){width=70%}

*Figure 8.1 - Representative end-to-end screen (house-price web client). The user enters
property attributes; the page `POST`s them to `/predict`; the API runs the loaded pipeline
and Random Forest, inverts `log1p` with `expm1`, and returns the price; the result screen
shows the value (about 9.90 billion VND), a low-high range, a SHAP contribution waterfall
and a plain-language summary.*

The complete web and mobile screenshots (input screen, result screen and, for the web, the
FastAPI `/docs` route) for every application, each with a Figure-explanation note as the
assignment requires, are in **Appendix D** (web) and **Appendix E** (mobile). Every mobile
run logged a `POST /predict 200` on the server - the required communication evidence.

---

## 9. Reproducibility

| Item | Diabetes | House Price | Customer Behaviour |
|---|---|---|---|
| Python / OS | 3.13 / Windows 11 | 3.13 / Windows 11 | 3.13 / Windows 11 |
| Key libraries | scikit-learn 1.9.0, numpy, pandas, scipy, matplotlib; fastapi + uvicorn + pydantic 2 | + xgboost | + React 18 + Vite 5; Flutter 3.19+ |
| Random seed | `RANDOM_SEED = 42` (numpy + `random`, every split and estimator) | same | same |
| Dataset | Kaggle `alexteboul/...`, `diabetes_012_health_indicators_BRFSS2015.csv` - committed | Kaggle `qmanhbeo/...` (CC BY-NC 4.0) - not committed (~210 MB) | Kaggle `nadyinky/...` (CC0) - not committed (~505 MB); `reviews_500-750.csv` + `product_info.csv` |
| Rows (raw -> clean) | 253 680 -> 229 781 | 236 226 -> 201 654 | 116 262 -> 104 313 |
| Split | stratified 70/15/15 (160 846 / 34 467 / 34 468) | 70/15/15 (141 157 / 30 248 / 30 249) | `StratifiedGroupKFold(5)` on `author_id` (62 587 / 20 862 / 20 864) |
| Preprocessing | `SimpleImputer(median)` -> `StandardScaler` (8 cols) + passthrough (15) | `SimpleImputer(median)` -> `log1p` -> `StandardScaler`; `OneHotEncoder(min_frequency=50)` | + `OneHotEncoder(min_frequency=25)`; `TfidfVectorizer(ngram_range=(1,2), min_df=10, max_features=40000, sublinear_tf, stop_words="english")` - all fitted on train only |
| Feature dim `d` | 23 | 92 | 28 733 (sparse) |
| Model | `RandomForestClassifier(max_depth=12, min_samples_leaf=20, class_weight="balanced", random_state=42)` | `RandomForestRegressor(random_state=42)` on `log1p(Price)` | `LogisticRegression(C=1.0, class_weight="balanced", random_state=42)` on tabular + TF-IDF |
| Test metrics | Acc 0.725 / recall(1) 0.735 / ROC-AUC 0.809 / conf `[[20612,7897],[1579,4380]]` | MAE 10 626 / RMSE 26 873 / R^2 0.186 | Acc 0.918 / macro-F1 0.858 / ROC-AUC 0.964 / recall(0) 0.876 / conf `[[2804,398],[1315,16347]]` |
| Saved artifacts | `model_pipeline.joblib`, `feature_names.joblib`, `input_schema.json` | same | + `feature_means.joblib` |
| Reproduce | `pip install -r requirements.txt` -> run the notebook (writes `model/`) -> `python -m uvicorn api.main:app` -> `npm --prefix web run dev` -> `flutter run` in `mobile/` |

```python
import numpy as np, random
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED); random.seed(RANDOM_SEED)
```

Repository layout: Appendix A. Notebook structure: Appendix B. API specification: Appendix
C. Submission checklist: Appendix F.

---

## 10. Final Comparative Discussion

The comparison table and the "how do the datasets / representations / metrics differ"
answers are in Section 7. This section covers the parts that go beyond that table.

**Which model performed best, and did it deploy best?** Random Forest (diabetes, house
price) and Logistic Regression on tabular + text (customer behaviour). In all three the best
model was also the best to deploy: each is a single scikit-learn `Pipeline`, CPU-only,
milliseconds per prediction, one small artifact.

**Forms of data leakage considered.** (a) A scaler, encoder or vectoriser re-fitted at
inference - avoided by loading the fitted pipeline and never calling `.fit()`. (b) A
duplicate row shared by train and test - the 23 899 diabetes duplicates are dropped *before*
the split. (c) A column that embeds the target - the house-price `Title` contains the price
string and is dropped. (d) The review star `rating`, which *is* the recommend label -
quarantined; adding it back pushes ROC-AUC to ~0.985. (e) Engagement counts that only exist
after publication - kept for the retrospective customer-behaviour model but flagged, and a
prospective variant that drops them falls to ~0.66.

**What information is lost and preserved during representation.** One-hot encoding loses the
ordering of nominal categories but preserves identity; `log1p` on money columns loses the
raw scale but preserves rank and makes the target near-symmetric; TF-IDF loses word order
and syntax but preserves which terms are distinctive; `min_frequency` merges rare
categories, losing tail identity to control dimensionality. What is preserved in every case
is the information the chosen model can actually use.

**Remaining limitations.** Diabetes: recall is bought with 0.36 precision, and the data is
one country and one year of self-reported survey answers. House price: the model explains
only ~19 % of variance because the dataset has almost no numeric signal and coarse
location; a better result needs street-level location and price-per-m^2 references.
Customer behaviour: the headline text score is partly leakage and the model is retrospective
(it needs the finished review).

**What we would improve with more time.** Calibrated thresholds and cost-weighted decisions
for diabetes; joining external location / price-per-m^2 data for house price; an
aspect-level text model and a multi-category pull (to restore RFM Frequency variance) for
customer behaviour; and a shared internal library for the three `build_features()` / schema
/ inference code paths so a change is made once.

---

## 11. Conclusion

```
Raw Data  ->  Clean  ->  Represent  ->  Learn  ->  Evaluate  ->  Persist  ->  Deploy
```

A deployable intelligent system is **data + representation + learning + evaluation +
software + deployment + user interaction**, not just a trained model.

1. **Main lesson** - most of the work and most of the risk is in turning raw data into a
   correct, reproducible representation; the model is a small final step.
2. **Biggest technical challenge** - building **one** preprocessing object that is fitted
   once on training data and applied identically in the notebook, the API and the tests.
3. **Most important data-representation issue** - deciding what each raw column *is* (a real
   feature, a leak, a post-hoc signal, an id) before encoding it - most visible in
   Application 3's leakage analysis.
4. **Most important ML lesson** - accuracy is the wrong headline on imbalanced or
   weak-signal data; the confusion matrix, recall on the class that matters, and R^2 versus
   a naive baseline tell the real story.
5. **Most important deployment lesson** - `Training != Inference`: the service loads the
   fitted pipeline and never re-fits.
6. **One future improvement** - a shared internal library for the three feature-build /
   schema / inference code paths.

---

## Appendix A - Repository Structure

```
assignment_02/
  Report_Deliverable.md / report/Report_Deliverable.docx     this report
  report/screenshots/                                        all figures + README index
  diabetes/  house_price/  customer_behaviour/               one folder per application, each:
    data/            dataset (or a note on how to download it) + data/README.md
    notebook/        the 23-section Jupyter notebook (0 error cells)
    model/           model_pipeline.joblib (preprocessing + model in one object),
                     feature_names.joblib, input_schema.json  [+ feature_means.joblib for App 3]
    api/             FastAPI service - main.py, schema.py, inference.py, config.py, requirements.txt
    web/             React + Vite client - src/, package.json, README.md
    mobile/          Flutter client - lib/, pubspec.yaml, README.md
    requirements.txt
  README.md                                                  how to reproduce every system
```

---

## Appendix B - Notebook Structure Checklist

Each application notebook contains all 23 required sections and runs top-to-bottom with zero
error cells.

| # | Section | Diabetes | House price | Customer behaviour |
|---|---|---|---|---|
| 1 | Problem definition | Yes | Yes | Yes |
| 2 | Dataset source | Yes | Yes | Yes |
| 3 | Dataset loading | Yes | Yes | Yes |
| 4 | Dataset inspection | Yes | Yes | Yes |
| 5 | Data-quality analysis | Yes | Yes | Yes |
| 6 | Missing-value analysis | Yes | Yes | Yes |
| 7 | Duplicate analysis | Yes | Yes | Yes |
| 8 | Invalid-value analysis | Yes | Yes | Yes |
| 9 | Outlier analysis | Yes | Yes | Yes |
| 10 | Exploratory data analysis | Yes | Yes | Yes |
| 11 | Feature types | Yes | Yes | Yes |
| 12 | Data representation | Yes | Yes | Yes |
| 13 | Feature engineering | Yes | Yes | Yes |
| 14 | Train / validation / test split | Yes | Yes | Yes (+ 14a leakage) |
| 15 | Preprocessing pipeline | Yes | Yes | Yes |
| 16 | Baseline model | Yes | Yes | Yes |
| 17 | Model training | Yes | Yes | Yes |
| 18 | Model comparison | Yes (5 models) | Yes (5 models) | Yes (8 models) |
| 19 | Evaluation | Yes | Yes | Yes |
| 20 | Error analysis | Yes | Yes | Yes |
| 21 | Model selection | Yes | Yes | Yes |
| 22 | Model persistence | Yes | Yes | Yes |
| 23 | Inference test | Yes | Yes | Yes |

---

## Appendix C - API Endpoint Specification

Workflow for every application: `JSON input -> Pydantic validation -> loaded preprocessing
-> model -> JSON output`. The service never calls `.fit()`.

| | Diabetes | House price | Customer behaviour |
|---|---|---|---|
| Framework | FastAPI + Uvicorn | FastAPI + Uvicorn | FastAPI + Uvicorn |
| Port | 8000 | 8002 | 8000 |
| Prediction endpoint | `POST /predict` (optional `?include=explain,similar,whatif`) | `POST /predict` | `POST /predict`, `POST /predict/batch` |
| Support endpoints | `GET /healthz`, `/model-info`, `/questions`, `/metrics`, `/threshold-curve` | `GET /healthz` | `GET /healthz`, `/model-info`, `/questions`, `/samples` |
| Prediction function | `pipeline.predict_proba` | `expm1(pipeline.predict)` | `pipeline.predict_proba` + linear-SHAP |
| Output keys | `prediction, confidence, band, contributions` | `predicted_price, price_per_m2, currency, model, contributions` | `prediction, confidence, p_recommend, threshold, review_terms, signals, contributions, model, representation` |

The interactive Swagger UI is served at each service's `/docs` route, with `POST /predict`
expanded, and documents the request and response schema.

---

## Appendix D - Web Application Report (assignment Appendix D)

The three web clients share one design - React 18 + Vite, `axios`, **no model in the
browser**: each renders its form from `GET /questions`, `POST`s the entered fields to
`/predict`, and shows the returned prediction with a plain-language interpretation.

### D.1 Web Application - Diabetes

| Item | Value |
|---|---|
| Framework | React 18 + Vite; the dev server proxies `/api` to the FastAPI service |
| API endpoint | `POST http://localhost:8000/predict` |
| Input variables | 21 BRFSS items (`Age, Sex, HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, GenHlth, MentHlth, PhysHlth, DiffWalk, Education, Income`) plus optional `height_cm` / `weight_kg` |
| Validation rules | Pydantic schema on the API; any omitted field is median-imputed by the loaded pipeline and the result is marked less certain |
| Preprocessing used | the saved `Pipeline` fitted on training data (`engineer()` then `SimpleImputer(median) -> StandardScaler` on 8 columns, 15 passthrough); never re-fitted |
| Loaded model | `RandomForestClassifier(class_weight="balanced")` inside the same pipeline object |
| Example request | `{ "Age": 9, "Sex": 1, "HighBP": 1, "HighChol": 1, "BMI": 34, "GenHlth": 4, "DiffWalk": 1, "Smoker": 1 }` |
| Example response | `{ "prediction": "at risk", "confidence": 0.82, "band": "High", "contributions": [ ... ] }` |

![Figure D1](report/screenshots/W1-D_input.png){width=60%}

*Figure D1 - Diabetes web: the input questionnaire with a valid example entered.*

![Figure D2](report/screenshots/W2-D_result.png){width=60%}

*Figure D2 - Diabetes web: the result screen - 82 % High risk, SHAP force plot, the five
most similar survey respondents, and a what-if BMI slider.*

**Figure explanation (D1-D2).** The user answers the questionnaire (D1); blank fields are
median-imputed. The service returns `p(diabetes) = 0.82` and the page (D2) shows the band
"High risk - refer for a confirmatory blood test", a SHAP force plot of which answers moved
the estimate, the five most similar survey respondents, and a what-if BMI slider - a
high-but-not-certain probability is deliberately surfaced as "refer for a test", not a
diagnosis. The `/docs` route documents the same request and response schema.

### D.2 Web Application - House price

| Item | Value |
|---|---|
| Framework | React 18 + TypeScript + Vite; `axios` |
| API endpoint | `POST http://localhost:8002/predict` |
| Input variables | `Area` (required, m^2); optional `Width, Length, Bedrooms, Bathrooms, Floors, "Alley Width", "Agent Listing Count", "Property Type", Position, Direction, "Road Type", Province, "Agent Role", district` |
| Validation rules | Pydantic schema - `Area > 0` required; missing optional numerics median-imputed; unknown categoricals map to an all-zero one-hot row |
| Preprocessing used | saved `ColumnTransformer` - `SimpleImputer(median) -> log1p -> StandardScaler` (numeric), `OneHotEncoder(handle_unknown="ignore", min_frequency=50)` (categorical); never re-fitted |
| Loaded model | `RandomForestRegressor` trained on `log1p(Price)`; the response inverts with `expm1` |
| Example request | `{ "Area": 78.7, "Width": 4.0, "Bedrooms": 3, "Bathrooms": 2, "Floors": 2, "Property Type": "house", "Province": "an-giang", "district": "Rach Gia" }` |
| Example response | `{ "predicted_price": 2540.82, "price_per_m2": 32.28, "currency": "million VND", "model": "RandomForest", "contributions": [ ... ] }` |

![Figure D3](report/screenshots/W1-H_input.png){width=60%}

*Figure D3 - House-price web: the valuation wizard step 1 with the "District 7 Apartment"
preset filled.*

![Figure D4](report/screenshots/W2-H_result.png){width=60%}

*Figure D4 - House-price web: the result screen - predicted about 9.90 billion VND, a
low-high range, a SHAP contribution waterfall and a plain-language summary.*

**Figure explanation (D3-D4).** The user enters property attributes (D3; only `Area > 0` is
required). The service returns `predicted_price` and the page (D4) shows about 9.90 billion
VND (about 151 million VND/m^2) with a low-high range and a SHAP waterfall (usable area
+3 381 M, location +2 093 M, ...); the model predicts `log1p(Price)` and the response
inverts it with `expm1`.

### D.3 Web Application - Customer behaviour

| Item | Value |
|---|---|
| Framework | React 18 + Vite, 3-step wizard; the dev server proxies `/api` to FastAPI |
| API endpoint | `POST http://localhost:8000/predict` (also `/predict/batch`, `/questions`, `/samples`) |
| Input variables | `skin_type, skin_tone, eye_color, hair_color, secondary_category, brand_name, price_usd, loves_count, reviews, review_title, review_text` - the star `rating` is never sent |
| Validation rules | Pydantic schema; blank skin-profile fields map to the `__na__` one-hot level; numeric gaps median-imputed |
| Preprocessing used | saved `ColumnTransformer` (`SimpleImputer(median) -> log1p -> StandardScaler`, binary passthrough, `OneHotEncoder(min_frequency=25)`) plus `TfidfVectorizer(ngram_range=(1,2), min_df=10, max_features=40000, sublinear_tf, stop_words="english")`; all fitted on the training split only |
| Loaded model | `LogisticRegression(C=1.0, class_weight="balanced")` on the concatenated tabular + TF-IDF matrix |
| Example request | `{ "skin_type": "oily", "secondary_category": "Moisturizers", "brand_name": "Murad", "price_usd": 49, "review_title": "New formula is awful", "review_text": "Broke me out within a week and left me greasy all day ..." }` |
| Example response | `{ "prediction": "not recommend", "p_recommend": 0.0026, "threshold": 0.5, "review_terms": {"against": [...], "toward": [...]}, "signals": {...}, "contributions": [...], "model": "LogisticRegression", "representation": "tabular + tfidf" }` |

![Figure D5](report/screenshots/W1-C_input.png){width=60%}

*Figure D5 - Customer-behaviour web: step 1 (skin profile) of the 3-step wizard, "API
connected".*

![Figure D6](report/screenshots/W2-C_result.png){width=60%}

*Figure D6 - Customer-behaviour web: the result screen - "WON'T RECOMMEND", P(recommend) =
0 %, probability meter, structured signals, review terms, linear-SHAP chart.*

**Figure explanation (D5-D6).** A 3-step wizard (D5): skin profile, then product, then the
review title and text; the star rating is never sent. The service returns
`prediction = "not recommend"`, `p_recommend = 0.00`, and the page (D6) shows "WON'T
RECOMMEND", a probability meter against the 50 % cut-off, the structured signals, the review
words that moved the call, and an exact linear-SHAP chart - a negative-reading review is
flagged for review-consistency QA.

---

## Appendix E - Mobile Application Report (assignment Appendix E)

All three mobile clients are Flutter apps that act as REST clients of the deployed service
(`Training != Inference`): they collect the same fields as the web form, `POST` them to
`/predict`, and render the returned JSON. Every screenshot run below logged a
`POST /predict 200` on the server - the required communication evidence. (Screenshots were
captured from the Flutter app built for web and driven at a 390 x 844 phone viewport.)

| Item | Diabetes | House price | Customer behaviour |
|---|---|---|---|
| Mobile framework | Flutter 3.x / Dart, Material 3 | Flutter 3.x / Dart, Provider state | Flutter 3.x / Dart |
| Platform | Android (emulator / device) | same | same |
| Input screen | About-you / Body / Conditions form, "Not sure" for unknowns | single scrolling form: area, rooms, province/district, type | 3-step wizard: skin profile / product / review |
| API endpoint | `POST http://10.0.2.2:8000/predict` (emulator) / `localhost:8000` | `.../:8002/predict` | `.../:8000/predict` |
| Request format | JSON body of the fields above (same keys as the web form) | JSON body (`Area` required) | JSON body (no `rating`) |
| Response format | `{ prediction, confidence, band, contributions }` | `{ predicted_price, price_per_m2, currency, model }` | `{ prediction, p_recommend, threshold, review_terms, signals, contributions }` |
| Prediction display | probability + risk band + "Why this score" bars + similar respondents | predicted price (billion VND) + unit price + one-line summary | verdict + probability meter + signals + review terms |

![Figure E1](report/screenshots/M1-D_input.png){width=42%} ![Figure E2](report/screenshots/M2-D_result.png){width=42%}

*Figures E1-E2 - Diabetes mobile: input screen (About you / Body / Conditions) and result
screen (27 % Low, range 15-46 %, "Why this score" bars). The run logged `POST /predict 200`.*

![Figure E3](report/screenshots/M1-H_input.png){width=42%} ![Figure E4](report/screenshots/M2-H_result.png){width=42%}

*Figures E3-E4 - House-price mobile: input screen (Rach Gia preset, "API Ready") and result
screen (about 3.01 billion VND, unit-price card, one-line summary). The run logged
`POST /predict 200`.*

![Figure E5](report/screenshots/M1-C_input.png){width=42%} ![Figure E6](report/screenshots/M2-C_result.png){width=42%}

*Figures E5-E6 - Customer-behaviour mobile: step 1 (skin profile, "API connected") and
result screen ("WON'T RECOMMEND", 0 % meter, signals, review terms). The run logged
`POST /predict 200`.*

**Figure explanation (E1-E6).** Each input screen collects the same fields as the matching
web form; on submit the app builds a JSON body and calls `POST .../predict` over HTTP. Each
result screen renders the returned JSON - the same payload the web page receives - as a
probability / predicted value plus a short interpretation. The `POST /predict 200` line
logged on the server during each run is the evidence that the mobile client communicates
with the deployed service.

---

## Appendix F - Final Submission Checklist and Additional Evidence

### F.1 Submission checklist

| Requirement | Done |
|---|---|
| Three Kaggle datasets selected | Yes |
| Diabetes / House-price / Customer-behaviour applications completed | Yes / Yes / Yes |
| Problem definition for all three | Yes |
| Dataset structure, data quality, missing values, duplicates, invalid values, outliers analysed | Yes |
| Data representation explained; numerical and categorical features identified; feature engineering explained | Yes |
| EDA completed | Yes |
| Train / validation / test split performed correctly; data leakage considered | Yes |
| Preprocessing pipeline implemented (fitted on train only, saved) | Yes |
| At least four ML models compared per application (5 / 5 / 8) | Yes |
| Appropriate metrics reported; confusion matrix reported where appropriate | Yes |
| Best model selected and justified; error analysis completed | Yes |
| Model saved; preprocessing saved; inference tested | Yes |
| REST API implemented | Yes |
| Web application implemented; web screenshots included (all three) | Yes (Appendix D) |
| Mobile application implemented; mobile screenshots included (all three) | Yes (Appendix E) |
| Reproducibility information included | Yes (Section 9) |
| Source code submitted; README included | Yes |
| Final comparative discussion included | Yes (Sections 7, 10) |
| Final PDF submitted | (produced from this document) |

### F.2 Additional notebook evidence

The `df.shape` / `df.info()` / `df.describe()` / `df.isna().sum()` / `df.duplicated().sum()`
outputs are tabulated with their executed values in Sections 4.3, 5.3 and 6.2; the raw
notebook cells are in the three `*/notebook/*.ipynb` files, which run top-to-bottom with
zero error cells. The house-price Section 10.1-10.3 supporting plots (area / unit-price
distributions, correlation-ratio bars, missingness and QQ views) and the Section 9 outlier
plots are in the notebooks and are not reproduced here.

### F.3 Reproduction and file index

All screenshots, and the ports each stack ran on, are indexed in
`report/screenshots/README.md`. Stacks were run locally on 2026-09-07: FastAPI + Uvicorn
backend, React/Vite web dev server, and the Flutter client built for web
(`flutter build web`) driven at a 390 x 844 device viewport. Backend ports: diabetes 8000,
house price 8002, customer behaviour 8000. The three notebooks run top-to-bottom with
`RANDOM_SEED = 42` and write `model/` before the API is started.
