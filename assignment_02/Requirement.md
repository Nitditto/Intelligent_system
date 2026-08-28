# Assignment 02 — Deliverable Structure & Content Guide

**Source of truth:** `materials/intel_sys_dev_assignment_02_final.pdf`
**Central objective:** Show *how raw real-world data becomes a numerical representation, then a model, then a persisted artifact, then a usable web + mobile service.* Training three models is **not** enough — every stage of the pipeline must be demonstrated and explained.

```
Data → Understand → Clean → Represent → Learn → Evaluate → Persist → Deploy
```

## Three required applications

| # | Application | Task type | Raw form | Numerical representation | Model input |
|---|---|---|---|---|---|
| 1 | Diabetes prediction | Classification | CSV / tabular | Feature vector / matrix | $X \in \mathbb{R}^{N \times d}$ |
| 2 | House-price prediction | Regression | CSV / tabular | Encoded + scaled feature matrix | $X \in \mathbb{R}^{N \times d}$ |
| 3 | E-commerce customer behavior / interest | Classification (or task chosen from dataset) | CSV + customer comments / reviews | Tabular features **+** text token / embedding representation | $X \in \mathbb{R}^{N \times d}$ and/or $E \in \mathbb{R}^{B \times T \times d}$ |

## Final submission checklist (Appendix F)

- [ ] Three Kaggle datasets selected
- [ ] Three Jupyter notebooks (diabetes, house price, customer behavior)
- [ ] Three trained + persisted ML pipelines (`preprocessor.joblib` + `model.joblib`, or a single `model_pipeline.joblib`)
- [ ] Three web prediction services (`POST /predict`)
- [ ] Three mobile demonstrations (screenshots + evidence of API calls)
- [ ] One final report (~10 pages, this document's Part A)
- [ ] Source code repository (Appendix A layout)
- [ ] Dataset references
- [ ] `README.md` explaining how to reproduce every system
- [ ] `requirements.txt` (or equivalent) per application

---

# PART A — REPORT STRUCTURE

**Length:** ~10 pages, excluding source code and appendices.
The three applications must be presented with the **same framework** so their differences can be compared side by side.

### Suggested page budget

| Report section | Pages |
|---|---|
| Cover + Executive Summary + Connection to Lecture 02 | 1 |
| Data Representation Overview (all 3 apps) | 1 |
| Application 1 — Diabetes | 1.5 |
| Application 2 — House Price | 1.5 |
| Application 3 — E-commerce | 1.5 |
| Comparison of the three systems | 1 |
| Deployment architecture (web + mobile, shared) | 1 |
| Reproducibility + Final comparative discussion + Conclusion | 1 |

---

## Cover Page

```
INTELLIGENT SYSTEM DEVELOPMENT
ASSIGNMENT 02
From Data Representation to Deployable Intelligent Systems

Student name: ................................................
Student ID:   ................................................
Class:        ................................................
Team:         ................................................
Lecturer:     Dinh Que Tran, Ph.D., Assoc. Prof.
Semester:     I.2026
```

---

## 1. Executive Summary

One short paragraph introducing the three intelligent applications, followed by the summary table and per-application bullets.

| Application | Prediction / Task | Main Representation |
|---|---|---|
| Diabetes | Diabetes classification | Feature vector / matrix |
| House Price | Price regression | Feature vector / matrix |
| Customer Behavior | Behavior / interest prediction | Feature vector / matrix (+ text) |

For **each** application, state briefly:
- **Dataset:** name + Kaggle URL
- **Problem:** what is predicted and why it matters
- **Representation:** how raw data becomes $X$ (and $E$ if text is used)
- **Selected model:** the deployed model and one-line justification
- **Main eval result:** headline metric (e.g. F1 = 0.82 / RMSE = 24,500 / Accuracy = 0.79)
- **Deployment method:** web framework + mobile framework used

---

## 2. Connection to Lecture 02 — Data Representation

Explain the principle the assignment is built on:

```
Real-world object → Raw data → Numerical representation → Tensor → Model
```

- Restate that a tabular sample is a vector $x = [x_1, x_2, \dots, x_d]^T \in \mathbb{R}^d$ and a dataset is a matrix $X \in \mathbb{R}^{N \times d}$.
- For text: `Text → Tokens → Token IDs → Embeddings`, and a batch of embeddings is $E \in \mathbb{R}^{B \times T \times d}$.
- State explicitly: **the representation is part of the ML solution** — marks are given for explaining how raw data becomes computational data, not only for accuracy.

---

## 3. Data Representation Overview

| Application | Raw Data | ML Representation |
|---|---|---|
| Diabetes | CSV | feature matrix |
| House Price | CSV | feature matrix |
| Customer Behavior | CSV / transactions / comments | feature matrix (+ token IDs / embeddings) |

Answer these questions **for each of the three applications**:
- What does one row represent?
- What does one column represent?
- Which columns are input features?
- Which column is the target?
- Which features are numerical?
- Which features are categorical?
- How are categorical values encoded? (e.g. `Urban → [1, 0, 0]` one-hot; or ordinal)
- What is the final feature dimension $d$ after encoding?
- What is the shape of the model input?

Close with the general notation:
- Model input: $X \in \mathbb{R}^{N \times d}$, where $N$ = number of rows, $d$ = number of features after encoding.
- Target: $y \in \mathbb{R}^{N}$ (regression) or $y \in \{0,1,\dots,K-1\}^{N}$ (classification).
- For text (App 3, if used): $E \in \mathbb{R}^{B \times T \times d}$ with $B$ = batch size, $T$ = sequence length, $d$ = embedding dimension.

---

## 4. Application 1 — Diabetes Prediction

### 4.1 Problem description
- Describe the healthcare problem in plain language.
- Fill in the template:
```
The objective of this application is to predict whether a patient is likely to have diabetes
based on ..................................... The prediction target is ..................................... The
prediction can potentially support ......................................
```
- Define clearly:
  - $X$ = patient features (age, BMI, blood pressure, glucose, insulin, pregnancy-related, other clinical features)
  - $y$ = diabetes class (binary)
- State that this is primarily a **classification problem**.

### 4.2 Kaggle Dataset
Provide: dataset name, Kaggle URL, number of observations, number of features, target variable, feature descriptions.

| Feature | Type | Meaning |
|---|---|---|
| Age | Numerical | ... |
| BMI | Numerical | ... |
| Glucose | Numerical | ... |
| ... | ... | ... |

### 4.3 Data Understanding
Report and interpret (do not just paste): dataset shape, data types, missing values, duplicated records, invalid values (e.g. glucose = 0), outliers, class distribution / imbalance. Include one or two tables or screenshots from the notebook and **explain what they mean**.

### 4.4 Data Cleaning
For every operation, state **what** was done and **why**:
- missing-value treatment (drop / impute — which strategy, why)
- duplicate removal
- invalid-value treatment (e.g. physiologically impossible zeros)
- outlier analysis and handling
- categorical-value processing (if any)

### 4.5 Data Representation
Show the transformation chain with concrete numbers:
```
CSV → DataFrame → clean feature matrix → scaled / encoded matrix → model input
```
Report:
- **one original CSV record** (raw row)
- the corresponding **feature vector** after encoding
- **original DataFrame shape** and **final feature-matrix shape** ($N$, $d$)
- data type of $X$
- feature encoding used
- normalization / scaling method (e.g. StandardScaler) and why
- final model-input shape / tensor supplied to the model

### 4.6 Exploratory Data Analysis
At least **three meaningful plots**: target distribution, key feature distributions, feature–target relationships, correlation analysis. For **each** plot give:
- **Observation:** what the figure shows
- **Interpretation:** what it means
- **ML implication:** why it matters for modeling

### 4.7 Model Development
Train and compare **five models**: Logistic Regression, Decision Tree, Random Forest, SVM, KNN. State hyperparameters and that all use the same train split and preprocessing.

### 4.8 Evaluation
Report Accuracy, Precision, Recall, F1-score, ROC-AUC (where appropriate), and the **confusion matrix — interpreted, not just displayed**. Explain **which metric matters most** for diabetes screening (recall / false negatives) and why.

### 4.9 Model Selection
Justify the deployed model considering predictive performance, interpretability, computational cost, robustness, and deployment constraints.

### 4.10 Deployment
- Deploy the selected model as **Web Service + Mobile Application**.
- Inference flow: `Input patient information → Preprocessing → ML Model → Prediction`.
- Example response:
```json
{ "prediction": "diabetic", "confidence": 0.91 }
```
- Include web and mobile screenshots + a short figure explanation each.

---

## 5. Application 2 — House Price Prediction

### 5.1 Problem description
- Describe the house-price problem.
- Define $X$ = house characteristics (living area, bedrooms, bathrooms, location, floors, year built, parking/garage, other attributes), $y$ = house price.
- State this is a **regression problem** and explain **why it differs from diabetes classification** (continuous target, different loss, different metrics).

### 5.2 Dataset
Provide: Kaggle name + URL, number of observations, number of features, target variable, numerical features, categorical features.

### 5.3 Data Understanding and Cleaning
Analyze and explain: missing values, duplicates, invalid values, outliers, skewed numerical variables (and any log transform), categorical variables. Explain the reasoning behind each cleaning step.

### 5.4 Representation
Show:
```
Raw CSV → DataFrame → Clean Data → Encoded / Scaled Features → X
```
- Identify numerical vs categorical variables and show how they combine into **one feature vector**, e.g.
  $x = [x_\text{area}, x_\text{bedroom}, x_\text{bathroom}, x_\text{location}, \dots]^T$
- Explain categorical encoding, e.g. one-hot: `{Urban, Suburban, Rural}` → `Urban → [1, 0, 0]`.
- Report $X \in \mathbb{R}^{N \times d}$, $y \in \mathbb{R}^{N}$ with actual $N$, $d$.

### 5.5 Exploratory Data Analysis
Include: house-price distribution, relationship between important features and price, correlation analysis, relevant scatter plots, analysis of possible outliers. Give Observation / Interpretation / ML implication for each.

### 5.6 Regression Models
Train and compare **five models**: Linear Regression, Ridge / Lasso, Decision Tree Regressor, Random Forest Regressor, Gradient Boosting Regressor.

### 5.7 Evaluation
Report MAE, MSE, RMSE, $R^2$. Include the formulas and **explain what each metric means in this application** (e.g. MAE in currency units, $R^2$ as variance explained).

$$MAE = \frac{1}{N}\sum_{i=1}^{N}|y_i - \hat{y}_i| \qquad RMSE = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2}$$

### 5.8 Model Selection
Justify the selected final model (accuracy vs interpretability vs cost vs robustness).

### 5.9 Deployment
Deploy on web + mobile. User enters house characteristics, receives a predicted price:
```json
{ "predicted_price": 285000 }
```
Include screenshots + figure explanations.

---

## 6. Application 3 — E-Commerce Customer Behavior and Interest

### 6.1 Problem description
- Objective: analyze customer behavior and **discover customer interests** from e-commerce data, using customer information and — where available — **customer comments, reviews, feedback, or text descriptions**.
- Select **one clearly defined supervised target**, e.g. predict whether a customer will purchase / will return / customer segment / product-category interest / high-value customer behavior.
- Write the problem as: $X$ = customer behavioral features (+ text representation), $y$ = behavior / interest / preference / category. State the precise target from the chosen dataset.

### 6.2 Dataset
Provide: Kaggle name + URL, number of customers / transactions, feature descriptions, target definition.

### 6.3 Customer Representation
Explain how a customer's behavior becomes numerical features, e.g.
$x_i = [\text{age}, \text{frequency}, \text{monetary value}, \text{sessions}, \text{cart actions}, \text{category counts}, \dots]^T$
or an RFM-style vector $x_i = [R_i, F_i, M_i, C_{i1}, C_{i2}, \dots, C_{ik}]$ where $R$=recency, $F$=frequency, $M$=monetary value, $C_{ij}$=activity in category $j$.
If transaction-level data are used, explain aggregation: `Transactions → Customer Profile → Feature Vector`.

**Text representation requirement (if comments / reviews exist):** demonstrate
```
Comment → Tokens → Token IDs → Vector / Embedding
```
Worked example: `"I like wireless headphones"` → `[t1, t2, t3, t4]` (tokens) → `[id1, id2, id3, id4]` (IDs) → embedding $E \in \mathbb{R}^{T \times d}$. Report $B$, $T$, $d$ and the final representation $E \in \mathbb{R}^{B \times T \times d}$.

### 6.4 Data Cleaning
Investigate and explain: missing customer information, duplicate transactions, invalid transactions, impossible prices / quantities, inconsistent categories, extreme purchasing behavior. For text: whitespace removal, missing comments, text normalization, punctuation handling, tokenization — with reasons.

### 6.5 Interest Discovery
Identify behavioral patterns: frequently purchased categories, purchase frequency, average order value, recency, browsing behavior, cart behavior, customer activity. Add e-commerce-specific EDA: purchase behavior, product categories, frequency of interests, comment / review characteristics, frequent terms / keywords (where appropriate).

### 6.6 Model Development
Train and compare **six models**: Logistic Regression, Decision Tree, Random Forest, SVM, a text-based linear classifier, and one more justified model (e.g. KNN or Gradient Boosting).
Also **compare a tabular-only representation with a representation that adds customer-comment text** (when the dataset supports it), and report whether text improves prediction.

### 6.7 Evaluation
Report Accuracy, Precision, Recall, F1, ROC-AUC (where appropriate), confusion matrix (interpreted). If a non-classification task is chosen, use the matching metrics and justify.

### 6.8 Business Interpretation
Answer: *what customer behavior or interest has been discovered, and how could an e-commerce company use the prediction?* Cover personalized recommendations, targeted promotions, customer retention, product recommendations, marketing campaigns.

### 6.9 Deployment
Deploy as Web Application + Mobile Application. System accepts customer information and returns a predicted behavior / interest:
```json
{ "interest": "electronics", "confidence": 0.87 }
```
Include screenshots + figure explanations.

---

## 7. Comparison of the Three Intelligent Systems

Provide **one integrated comparison table**:

| Aspect | Diabetes | House Price | Customer Behavior |
|---|---|---|---|
| Problem type | Classification | Regression | Classification / selected task |
| Raw data | CSV | CSV | CSV (+ comments) |
| Observation (one row) | Patient | House | Customer / transaction |
| Input shape | $\mathbb{R}^{N \times d}$ | $\mathbb{R}^{N \times d}$ | $\mathbb{R}^{N \times d}$ (+ $E$) |
| Target | Diabetes class | Price (continuous) | Behavior / interest |
| Representation | Feature vector | Feature vector | Behavior vector (+ text) |
| Best model | ... | ... | ... |
| Main metric | ... | ... | ... |
| Web deployment | Yes | Yes | Yes |
| Mobile deployment | Yes | Yes | Yes |
| Main limitation | ... | ... | ... |

Then discuss:
1. How are the three datasets different?
2. How are their representations different?
3. Which preprocessing operations are common?
4. Which preprocessing operations are application-specific?
5. Why is the target representation different?
6. Why are different evaluation metrics required?
7. Which system is easiest to deploy?
8. Which system is most computationally demanding?

---

## 8. Deployment Architecture (shared by all three applications)

Show the complete inference pipeline:
```
User Input → API Request → Validation → Same Preprocessing → Saved ML Model → Prediction → Result
```
Describe, for the common architecture used across all three apps:
1. trained model (loaded, not retrained)
2. preprocessing pipeline (loaded from training)
3. API endpoint (`POST /predict`)
4. input validation
5. prediction function
6. result returned to client (JSON)
7. web user interface
8. mobile user interface

**Data leakage warning:** the deployed service must load the preprocessing pipeline fitted on training data — it must **never** fit a new scaler / encoder / imputer on user input or test data. Explain why the same preprocessing must be used at training and inference.

### Web application (per app)
Report: web framework (FastAPI / Flask / Streamlit / other), endpoint, input variables, validation rules, preprocessing used, loaded model, example request, example response, screenshot of input interface, screenshot of prediction result. Use the template:

```
Web Application — [Application Name]
Framework: ______________________
Endpoint:  ______________________
Input:     ______________________
Output:    ______________________
[ screenshot of working web application ]
Figure explanation: describe the input, prediction, and interpretation shown.
```

### Mobile application (per app)
The mobile app is a **client** of the deployed service (it does not retrain). Architecture:
```
Mobile UI → REST API → Preprocessing → Saved ML Model → Prediction → REST Response → Mobile UI
```
Report: mobile framework (Flutter / Android Studio / React Native / other), platform, input screen, API endpoint, request format, response format, prediction display, screenshots. Use the template:

```
Mobile Application — [Application Name]
Framework: ______________________
Platform:  ______________________
API:       ______________________
[ screenshot of mobile input + prediction screens ]
Figure explanation: how the mobile app sends input to the API and displays the prediction.
```

The mobile interface must let the user: (1) enter / select input, (2) submit a prediction request, (3) display the prediction, (4) display a confidence value or short explanation.

---

## 9. Reproducibility

For **each** of the three applications provide: Python version, operating system, library versions, random seed, Kaggle dataset source, dataset version / download info, preprocessing procedure, feature representation, train/test split, model hyperparameters, evaluation metrics, saved preprocessing pipeline, saved model, API code, web app code, mobile app code.

```python
import numpy as np, random
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
```

Provide `requirements.txt` (or equivalent) and a `README.md` that lets another student reproduce the main experiment **and** run the deployed application.

---

## 10. Final Comparative Discussion

Compare the applications across the full pipeline `Data → Representation → Model → Evaluation → Deployment`. Answer:
1. How are the three datasets different?
2. What does one observation represent in each dataset?
3. What is the target variable for each application?
4. What is the computational representation of each dataset?
5. Which application uses classification? Which uses regression?
6. How are categorical variables represented?
7. How are numerical variables represented and scaled?
8. Which data-quality problems occurred in each dataset?
9. Which preprocessing operations were required?
10. Which model performed best for each application?
11. Which evaluation metrics were most appropriate?
12. Did the best model also have the best deployment characteristics?
13. Which application was easiest to deploy? Which was most difficult?
14. What forms of data leakage were considered?
15. What limitations remain in each system?
16. What would you improve with more time?

Include the **Cross-Application Comparison** table (Aspect × {Diabetes, House Price, Customer Behavior}) covering: problem type, observation, target, input representation, data-quality issues, best model, main metric, web deployment, mobile deployment, main limitation.

---

## 11. Conclusion

Return to the central principle:
```
Raw Data → Clean → Represent → Learn → Evaluate → Persist → Deploy
```
State that a deployable intelligent system combines **data, representation, learning, evaluation, software, deployment, and user interaction** — not just a trained model. Include:
1. the main lesson learned
2. the most important technical challenge
3. the most important data-representation issue
4. the most important ML lesson
5. the most important deployment lesson
6. one proposed improvement for future work

---

## Mandatory Data-Representation Summary (must appear in the report)

| Application | Raw form | Numerical representation | Model input |
|---|---|---|---|
| Diabetes | CSV / table | Feature vector / matrix | $B \times d$ |
| House price | CSV / table | Encoded feature matrix | $B \times d$ |
| E-commerce | CSV + comments | Tabular features + text vectors / embeddings | $B \times d$ and/or $B \times T \times d$ |

**Every dimension in the reported shapes must be explained.**

---

## Assessment Rubric (for self-check)

| Component | Weight | Key evidence |
|---|---|---|
| Problem understanding | 10% | Clear application and target |
| Data quality | 10% | Missing, duplicate, invalid data handled |
| Data representation | 15% | Vector / matrix / tensor explanation |
| EDA | 10% | Meaningful visualization + interpretation |
| Preprocessing | 10% | Reproducible pipeline |
| ML models | 15% | Multiple models + comparison |
| Evaluation | 10% | Correct metrics + interpretation |
| Web deployment | 7.5% | Working prediction API |
| Mobile deployment | 7.5% | Working mobile interface |
| Discussion & reproducibility | 5% | Technical explanation + README |

---

# PART B — JUPYTER NOTEBOOK STRUCTURE

**One notebook per application** (`diabetes`, `house_price`, `customer_behavior`), each with the **same 23 sections** (Appendix B). Each section = one or more markdown cells (explanation) + code cells (execution). Every output must be followed by a short markdown interpretation — never leave a raw table or plot unexplained.

> Naming: `diabetes/notebook/diabetes.ipynb`, `house_price/notebook/house_price.ipynb`, `customer_behavior/notebook/customer_behavior.ipynb`.

### 0. Header & setup (before section 1)
- Title, student name/ID, application name, date.
- Imports; set and print `RANDOM_SEED = 42` (`numpy`, `random`, and framework seeds).
- Print Python version and key library versions (for the Reproducibility report section).

### 1. Problem definition
Markdown only. State the real-world problem, the supervised task (classification / regression), $X$ (input features), $y$ (target). For house price, explain why it is regression not classification. For e-commerce, state the single chosen target and why.

### 2. Dataset source
Markdown: Kaggle dataset name, full URL, licence/version, download date, brief description of how the data was collected. Note where the raw file lives (`data/`).

### 3. Dataset loading
Code: load the CSV(s) into a DataFrame. For e-commerce with transaction-level data, load all relevant tables. Show `df.head()`. State the file path and separator used.

### 4. Dataset inspection
Code: `df.shape`, `df.head()`, `df.info()`, `df.describe()`, `df.isna().sum()`, `df.duplicated().sum()`. Markdown: state number of observations, number of attributes, column names, dtypes, target column, and a one-sentence description of one observation (row).

### 5. Data-quality analysis
Markdown + code summarizing all quality issues found: numerical columns, categorical columns, text columns, target column, missing values, duplicated records, invalid values, outliers, class imbalance (if applicable). Produce a compact "issue → count → planned action" table. **Explain the results.**

### 6. Missing-value analysis
Code: per-column missing counts and percentages; visualize (bar / heatmap) if useful. Markdown: decide and justify the strategy per column (drop rows, drop column, mean/median/mode impute, model-based impute). No fitting on test data — imputers are fitted later inside the pipeline on train only.

### 7. Duplicate analysis
Code: detect exact and key-based duplicates (e.g. duplicate transactions by order id). Markdown: how many, why they occur, whether to drop, and the effect on $N$.

### 8. Invalid-value analysis
Code: domain checks — e.g. diabetes `Glucose == 0`, `BMI == 0`; house `price <= 0`, `area <= 0`, `year_built` in the future; e-commerce negative price / quantity, impossible values. Markdown: how each invalid value is treated (set to NaN then impute, clip, or drop) and why.

### 9. Outlier analysis
Code: IQR / z-score / boxplots for key numerical features. Markdown: which points are outliers, whether they are genuine (e.g. luxury houses, high-value customers) or errors, and the handling decision (keep, cap, transform, remove).

### 10. Exploratory data analysis
Code: at least **three meaningful plots** — target distribution; important feature distributions; feature–target relationships; correlation heatmap. E-commerce also: purchase behavior, product categories, interest frequency, comment/review characteristics, frequent terms/keywords. For **every** plot add a markdown cell with **Observation / Interpretation / ML implication**.

### 11. Feature types
Markdown + code: final classification of every column into numerical / categorical / text / identifier / target. List which columns will be dropped (IDs, leakage-prone, constant) and why.

### 12. Data representation
The core Lecture-02 link. Show:
```
CSV → DataFrame → clean feature matrix → scaled / encoded matrix → model input
```
- Print **one raw record** and the **feature vector it becomes**.
- State original DataFrame shape and target definition.
- Define $X \in \mathbb{R}^{N \times d}$, $y \in \mathbb{R}^{N}$ (or class labels) with the actual $N$, $d$.
- For e-commerce text: demonstrate `Comment → Tokens → Token IDs → Vector / Embedding` on one real comment; report $B$, $T$, $d$ and $E \in \mathbb{R}^{B \times T \times d}$.

### 13. Feature engineering
Code: new features (ratios, aggregates, RFM for customers, date parts, log transforms of skewed targets/features), categorical encoding choice (one-hot / ordinal / target encoding) with justification, text vectorization choice (bag-of-words / TF-IDF / embeddings). Show the feature list and final $d$ after encoding. Example: `{Urban, Suburban, Rural} → Urban → [1, 0, 0]`.

### 14. Train/test split
Code: split into train / validation / test, sets kept independent:
$$D = D_\text{train} \cup D_\text{validation} \cup D_\text{test}$$
Use 70/15/15 (or 80/20 with justification). Use stratification for classification. Fix `random_state`. Markdown: explain **why test data must not influence training or preprocessing fitting** (data leakage).

### 15. Preprocessing pipeline
Code: build a single `sklearn` `Pipeline` / `ColumnTransformer` containing imputation + encoding + scaling + (text vectorizer). **Fit on training data only.** This exact object is what gets persisted and reused at inference. Markdown: list every step and its purpose; state that deployment will load this same fitted pipeline.

### 16. Baseline model
Code: one simple model (e.g. `DummyClassifier` / `DummyRegressor`, or Logistic / Linear Regression) to establish a reference score. Markdown: baseline metric — every later model must beat this.

### 17. Model training
Code: train the required models on the **same** preprocessed training data:
- Diabetes: Logistic Regression, Decision Tree, Random Forest, SVM, KNN (**5 models**)
- House price: Linear, Ridge/Lasso, Decision Tree Regressor, Random Forest Regressor, Gradient Boosting Regressor (**5 models**)
- E-commerce: Logistic Regression, Decision Tree, Random Forest, SVM, text-based linear classifier, + one more justified (**6 models**); also train a tabular-only vs tabular+text variant.
Record hyperparameters and training time per model.

### 18. Model comparison
Code: assemble a comparison table across all models on the validation set.
- Classification: Model | Accuracy | Precision | Recall | F1 (| ROC-AUC)
- Regression: Model | MAE | MSE | RMSE | $R^2$ | Training time
Markdown: which model leads and on which metric.

### 19. Evaluation
Code: evaluate the chosen model on the **held-out test set**.
- Classification: Accuracy, Precision, Recall, F1, ROC-AUC (where appropriate), confusion matrix.
- Regression: MAE, MSE, RMSE, $R^2$; predicted-vs-actual and residual plots.
Markdown: interpret the confusion matrix (not just show it); explain **which metric matters most** for this application and why.

### 20. Error analysis
Code: inspect the worst predictions / misclassified samples; look for patterns (a feature range, a category, long comments). Markdown: what the model struggles with, likely causes (representation gaps, imbalance, noise), and what could fix it.

### 21. Model selection
Markdown: justify the final deployed model on predictive performance, interpretability, computational cost, robustness, and deployment constraints. Note if a slightly weaker but simpler/faster model is chosen for deployment.

### 22. Model persistence
Code: save the full inference artifact so it can be reused without retraining:
```python
import joblib
joblib.dump(pipeline, "model/model_pipeline.joblib")
# or separately: preprocessor.joblib + model.joblib
```
Save: preprocessing, feature transformation, trained model, and any model configuration needed. Markdown: state exactly which files are produced and where (`model/`).

### 23. Inference test
Code: reload the saved artifact **from disk** (fresh objects), pass **one raw, unprocessed input example** (a dict / raw row like the API will receive), run the full `raw input → validation → same preprocessing → model → prediction` path, and print the JSON-style output:
```json
{ "prediction": "...", "confidence": 0.91 }
```
Markdown: confirm the reloaded pipeline reproduces the notebook's prediction, and that no new preprocessing was fitted here — this is the contract the web/mobile services rely on.

---

# PART C — REPOSITORY STRUCTURE (Appendix A)

```
Assignment_02/
│
├── diabetes/
│   ├── data/                     # raw Kaggle CSV (or download script + reference)
│   ├── notebook/                 # diabetes.ipynb (23 sections)
│   ├── model/
│   │   ├── preprocessor.joblib
│   │   └── model.joblib          # (or a single model_pipeline.joblib)
│   ├── api/                      # FastAPI/Flask service exposing POST /predict
│   ├── web/                      # web UI (form → API → result)
│   ├── mobile/                   # mobile client (Flutter/Android/React Native)
│   └── requirements.txt
│
├── house_price/                  # same layout as diabetes/
│   └── ...
│
├── customer_behavior/            # same layout as diabetes/
│   └── ...
│
├── report/
│   └── Assignment_02.pdf
│
└── README.md                     # how to reproduce notebooks + run each service
```

## Required API structure (Appendix C)

Each application exposes:
```
POST /predict
```
Workflow: `JSON Input → Validation → Preprocessing (loaded from training) → Model → JSON Output`.

Example responses:
```json
{ "prediction": "diabetic", "confidence": 0.91 }
{ "predicted_price": 285000 }
{ "interest": "electronics", "confidence": 0.87 }
```

The API must: load the saved preprocessor + model, validate input, transform with the **same** preprocessing used in training, return prediction (+ confidence/probability where appropriate). Demonstrate it via browser / Swagger UI / Postman.

## README.md must contain

- Project overview and the `Data → … → Deploy` pipeline.
- Per application: how to get the dataset, run the notebook, retrain if needed.
- How to start each web service and call `POST /predict` (example `curl` / Swagger).
- How to run each mobile client and point it at the API.
- Environment: Python version, OS, `requirements.txt`, `RANDOM_SEED`.
- Location of saved models and expected input schema for each app.
