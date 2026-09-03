# Report guide — Application 3: E-Commerce Customer Behaviour & Interest

What to write for the **customer-behaviour** part of the Assignment 02 report
(Part A §6, plus the shared Reproducibility, Web-app and Mobile-app templates).
Each subsection lists the points to cover and where the material already exists.
Screenshot placeholders are marked `📸` with a caption describing exactly what the
image must show — replace each with the real image when you take it.

Source material:
- `notebook/customer_behaviour.ipynb` — the 23-section ML experiment (executed, with outputs).
- `api/README.md`, `web/README.md`, `mobile/README.md` — how each part runs.
- Key executed numbers are quoted below so the report can cite them directly.

---

## 6.1 Problem description

- Objective sentence:
  *"The objective of this application is to analyse customer purchasing behaviour and
  discover, at the moment an order is completed, whether the customer will be
  **satisfied** with it. The prediction target is **`satisfied` (1 if the post-delivery
  `review_score` is 4 or 5, else 0)**. The prediction can support **proactive customer
  retention** — triggering a support contact, a goodwill gesture, or a shipment
  priority before a bad experience becomes a public 1-star review."*
- `X` = an **order-level behavioural / transactional vector** (order value, freight,
  payment plan, basket size, delivery speed and lateness vs. the promised date,
  product category, customer region, order timing) **+** the customer's **review
  comment text** (Portuguese, present for ~41% of orders).
- `y` = `satisfied` (binary). It is a **binary classification** problem.
- Explain the target choice: the business action is binary (intervene / not), the
  middle score 3 is rare (~8%) and behaves like the negative class in every
  exploratory cut, and a single supervised target is required by the assignment.
- One observation = **one order**, assembled by aggregating several transaction tables
  (`Transactions → Order profile → Feature vector`). Source: notebook §1, §3.

## 6.2 Dataset

- Name: **Brazilian E-Commerce Public Dataset by Olist**.
- Kaggle URL: <https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce> (CC BY-NC-SA 4.0),
  downloaded 2026-09-03 into `data/`.
- Structure: **9 CSV files** linked by id columns (orders, order_reviews, order_items,
  order_payments, products, product_category_name_translation, customers, sellers,
  geolocation). Sellers / geolocation are not used.
- Observations: **98,673** orders that received a review (inner join orders ↔ reviews) →
  **95,824** after keeping only `delivered` orders with a valid delivery timestamp
  (2,849 dropped, 2.89%).
- Raw `review_score` distribution: 1 → 11,424 · 2 → 3,151 · 3 → 8,179 · 4 → 19,142 ·
  5 → 57,328. Target positive rate: **77.1%** raw → **79.0%** after cleaning.
- Comment text present for **40,792 orders (41.3%)**; empty for the rest.
- Feature descriptions: include the section-11 role table (each column → numeric
  (log/scale) / binary / one-hot / text / target).
- Numerical features: `price_total`, `freight_total`, `payment_value_total`,
  `freight_ratio`, `n_items`, `n_sellers`, `max_installments`, `n_payment_types`,
  `product_weight_g`, `product_desc_len`, `product_photos_qty`, `delivery_days`,
  `estimated_days`, `delivery_delay_days`, `comment_len` (15).
- Categorical features: `main_payment_type` (4), `customer_region` (5 macro-regions,
  grouped from 27 states), `category_grp` (top-15 product categories + `__other__`).
  Binary: `is_late`, `has_comment`. Text: `comment_clean`.

## 6.3 Customer representation

Show the transformation chain with concrete numbers (notebook §3, §12):
```
9 CSV tables ─aggregate─▶ one order row ─clean─▶ feature frame ─┬─ ColumnTransformer ─▶ x_tab ∈ ℝ^{d_tab}
                                                               └─ TF-IDF(comment)   ─▶ x_txt ∈ ℝ^{d_txt}
                                       model input  X = [x_tab ‖ x_txt] ∈ ℝ^{N × d}
```
- **Aggregation:** `order_items` → `n_items`, `n_sellers`, `price_total`,
  `freight_total`, main product; `order_payments` → `payment_value_total`,
  `max_installments`, `n_payment_types`, main payment type. This is the
  `Transactions → Customer/Order profile → Feature vector` step the assignment asks for.
  (Customer-level RFM was rejected: ~97% of Olist customers place only one order, so
  Frequency ≈ 1 for almost everyone — the order is the usable unit.)
- **One raw order** (print `raw.iloc[0]`) → **the tabular feature row** it becomes
  (20 columns) → after `impute → log1p(money) → scale` and `OneHotEncoder` →
  `x_tab ∈ ℝ^{43}`.
- Original assembled frame `(98673, 25)`; cleaned modelling frame `(95824, 22)`;
  `X_tab ∈ ℝ^{N×43}` dense float64, `X_txt ∈ ℝ^{N×~13000}` sparse CSR,
  combined `d ≈ 13,000`; `y ∈ {0,1}^N`, `N = 95,824`.
- **Categorical encoding:** one-hot (`handle_unknown="ignore"`, `min_frequency=30`) —
  e.g. `{North, Northeast, Centre-West, Southeast, South}` → `Southeast → [0,0,0,1,0]`.
- **Scaling:** `StandardScaler` on the 15-column numeric block; `price_total`,
  `freight_total`, `payment_value_total` get `log1p` first (heavy right skew, §9).

### Text representation requirement (assignment §6.3)

Demonstrate the chain on **one real comment** (notebook §12, second cell):
```
raw comment ─clean─▶ tokens ─▶ token IDs ─▶ embedding lookup ─▶ E
"aguardando retorno da loja"
  → ['aguardando','retorno','da','loja']          (T = 4 tokens)
  → [1, 4, 2, 3]                                    (token IDs, 0 = PAD)
  → E  ∈ ℝ^{T×d}   with d = 8   (row i = embedding table[id_i])
  → E_batch ∈ ℝ^{B×T×d}   with B = 1, T = 4, d = 8
```
- Explain **each dimension**: `B` = comments in the batch, `T` = tokens per comment
  (padded/truncated), `d` = embedding width.
- State the deployment choice: the served model uses the **TF-IDF bag-of-words** form
  (a fixed `d_txt`-vector per comment) — the assignment's required "text-based linear
  classifier" — not the `B×T×d` sequence tensor, which is shown only to satisfy the
  representation requirement and is what an embedding/RNN model would consume.
- 📸 **Screenshot N4 — notebook §12 output**: the raw order + tabular vector + the
  shape/dtype lines, and the `Comment → Tokens → IDs → E` block with `B, T, d`.

## 6.4 Data cleaning

For each operation state **what** and **why** (notebook §5–§9):
- **Non-delivered orders removed** — `order_status != 'delivered'` (~2.9%) plus 8
  delivered orders with no delivery timestamp. *Why:* delivery-speed features are the
  strongest signals and are undefined when the order never arrived. `N`: 98,673 → 95,824.
  Stated as a **scope limitation** in the report.
- **Missing comment text kept as a feature** — 58.7% of orders have no comment. *Why:*
  commenting is itself predictive (satisfied rate 0.675 with a comment vs 0.868
  without), so fill `""` + add a `has_comment` flag rather than drop the rows.
- **Incidental numeric missingness** (`price_total` ~0.8%, `category` ~2.2%) — filled
  **inside the pipeline** with `SimpleImputer(median)` fitted on train only; missing
  `category` → `__other__` level. *Why:* keep one training/inference code path, no
  leakage.
- **Invalid values** — `price_total == 0` (voucher orders) and `max_installments == 0`
  kept as valid; the `freight_ratio` division is guarded against zero.
- **Outliers not removed** (notebook §9) — long deliveries and expensive orders are
  genuine business events; the band table shows satisfied rate falling 0.86 → 0.25 as
  delivery time rises, which is signal to learn. Handled by `StandardScaler` +
  `log1p` on money columns; tree/boosting models are rank-invariant anyway.
- **Text cleaning** — lower-case, strip punctuation, collapse whitespace
  (`clean_text()`); tokenisation + 1–2-gram vocabulary is done by `TfidfVectorizer`
  fitted on train only.
- **Encoding note** — the reviews CSV is **Latin-1**, read with `encoding="latin-1"`
  so accented Portuguese ("atrasado", "não recebi", "péssimo") is preserved; reading
  it as UTF-8 corrupts exactly the words the text branch needs.
- 📸 **Screenshot N2 — notebook §5** data-quality table (issue → amount → planned action).
- 📸 **Screenshot N3 — notebook §9** IQR outlier counts + the delivery-days band table
  + the three boxplots.

## 6.5 Interest discovery / EDA

Include ≥ 3 of the 5 plots from notebook §10; for **each** give Observation /
Interpretation / ML implication (already written under §10):
- **Plot 1 (bar)** — target class balance: ~79% satisfied / ~21% not.
- **Plot 2 (bar)** — satisfied rate vs delivery-delay bucket: ~0.85 when 10+ days early,
  collapsing toward ~0.35 once > 15 days late.
- **Plot 3 (barh)** — satisfied rate by product category (top-12 by volume): spreads
  ~15–20 points; office/furniture/home categories low, watches/gifts high.
- **Plot 4 (stacked bar)** — comment presence vs satisfaction: orders with a comment are
  far more likely to be negative.
- **Plot 5 (heatmap)** — numeric correlation with `satisfied`: `is_late` −0.32,
  `delivery_days` −0.29, `has_comment` −0.23, `delivery_delay_days` −0.23; price /
  freight / payment near zero.
- E-commerce-specific cuts to mention: satisfied | has-comment 0.675 vs no-comment
  0.868; satisfied | late 0.347 vs on-time 0.828; the by-category satisfied rates;
  the delivery-days band table from §9.
- 📸 **Screenshot N5 — the 2×2 plot grid** (plots 1–4).
- 📸 **Screenshot N6 — the correlation heatmap** (plot 5).

## 6.6 Model development

- **Six models** compared (notebook §17), each under **two representations**
  (tabular-only vs tabular + comment text):
  Logistic Regression · LinearSVC · SGDClassifier (log-loss) — *the text-based linear
  classifier* · Decision Tree · Random Forest · HistGradientBoosting.
- Fair-comparison protocol: all trained on the same **stratified 45,000-row
  subsample** with the same preprocessing; the winner is refitted on the full
  training split. Linear models take the raw sparse TF-IDF; tree/boosting models take
  a 100-component `TruncatedSVD` (LSA) compression of it concatenated to the tabular
  block.
- Give the hyperparameter table from notebook §17.
- **Tabular vs tabular+text** (the assignment's key question): adding the comment
  branch improves every model — **mean F1 +0.042, mean ROC-AUC +0.08** — with the
  largest lift on the models that read the full bag-of-words (LogReg +0.089 ROC-AUC,
  SGD +0.092) and the smallest on the tree family (+0.06–0.08). **Text improves
  prediction over tabular features alone.**
- 📸 **Screenshot N7 — notebook §18** the 12-row comparison table + the
  `*_gain_from_text` table + the stability table (winner @ 45k vs @ full train).

Reference numbers (validation, 45k fit):

| model | representation | ROC-AUC | F1 | recall |
|---|---|---|---|---|
| **LogisticRegression** | **tab+text** | **0.866** | **0.913** | **0.914** |
| LinearSVC | tab+text | 0.864 | 0.928 | 0.970 |
| HistGradientBoosting | tab+text | 0.860 | 0.928 | 0.968 |
| RandomForest | tab+text | 0.852 | 0.915 | 0.924 |
| SGD (log-loss, text) | tab+text | 0.840 | 0.872 | 0.833 |
| DecisionTree | tab+text | 0.827 | 0.891 | 0.876 |
| *(all models)* | *tab-only* | *0.75–0.78* | | |
| *baseline B — LogReg tab-only* | | *0.778* | *0.855* | *0.824* |
| *baseline A — majority class* | | *0.500* | *0.882* | *1.000* |

## 6.7 Evaluation

- Chosen model: **Logistic Regression, tab+text representation**, refit on the full
  training data, evaluated once on the held-out **test** set (notebook §19).
- Report (test):

  | | precision | recall | F1 | support |
  |---|---|---|---|---|
  | class 0 — dissatisfied (`score ≤ 3`) | 0.671 | 0.672 | 0.672 | 3,025 |
  | class 1 — satisfied (`score ≥ 4`) | 0.913 | 0.912 | 0.912 | 11,349 |
  | accuracy | | | **0.862** | 14,374 |
  | macro avg | 0.792 | 0.792 | 0.792 | |

  **ROC-AUC 0.859.**
- Confusion matrix `[[2034, 991], [997, 10352]]` (rows = actual, cols = predicted).
  **Interpret it:** FN (997) = *dissatisfied customers predicted happy* → no
  intervention, the bad review lands — the costly error; FP (991) = satisfied
  customers flagged at-risk → one wasted support contact; TN (2034) = unhappy
  customers correctly caught.
- **Which metric matters most: recall on class 0 (dissatisfied)** — a missed unhappy
  customer is far more expensive than a wasted support touch. ROC-AUC (0.859) is the
  threshold-independent summary used for selection. Accuracy (0.862) is *not* the
  headline: the majority baseline scored 0.79 accuracy while catching zero unhappy
  customers.
- 📸 **Screenshot N8 — notebook §19** classification report + confusion-matrix heatmap
  + ROC curve.

## 6.8 Business interpretation

From notebook **Appendix A** — answer *"what behaviour/interest was discovered and how
could an e-commerce company use it?"*:
- **Discovered:** post-purchase satisfaction is driven first by **delivery performance
  vs. the promised date**, then by **product category**, and the customer's own
  **decision to write a comment** (and its wording) is an early sentiment signal;
  price / freight / payment plan barely matter alone.
- **Uses:** proactive retention (contact / voucher before the 1-star review);
  operations prioritisation (expedite orders predicted to tip a customer negative);
  seller / category management (where returns and expectation gaps concentrate);
  marketing suppression (don't send "buy again" to a predicted-unhappy customer —
  route to service); review-response triage (work highest-risk orders first).

## 6.9 Deployment

- Architecture (shared with the other two apps):
  ```
  User input → API request → validation → same preprocessing → saved model → prediction → result
  ```
- The service loads `model/model_pipeline.joblib` (tabular `ColumnTransformer` +
  fitted `TfidfVectorizer` + `LogisticRegression`) and **never re-fits** it. Leakage
  rule: the deployed preprocessing is the object fitted on the training split only.
- Inference flow for one request: raw order dict → pydantic `OrderInput` validation →
  stateless `build_features()` (delivery_days / delay / is_late / freight_ratio /
  has_comment / comment_clean / region / category group) → pipeline → `P(satisfied)` →
  satisfied / dissatisfied at threshold 0.5.
- Example response (the API's richer form):
  ```json
  { "prediction": "dissatisfied", "confidence": 0.9252, "p_satisfied": 0.0748,
    "threshold": 0.5,
    "signals": { "days_vs_promise": 11.6, "late": true, "has_comment": true,
                 "delivery_days": 30.2, "category_group": "bed_bath_table",
                 "customer_region": "Southeast" },
    "contributions": { "base_p": 0.63, "final_p": 0.075, "dataset_base_rate": 0.79,
      "items": [ {"label": "Delivered 12 days late", "kind": "tabular", "effect": -0.47},
                 {"label": "comment: \"atrasado\"", "kind": "text", "effect": -0.09} ],
      "other_effect": -0.23 },
    "model": "LogisticRegression", "representation": "tab+text" }
  ```
- Web framework: **FastAPI** (`POST /predict`). Client: **React (Vite)**, a
  **full-viewport 4-step wizard** (Product → Payment → Delivery & review → Review &
  predict) then a full-screen result. Mobile: **Flutter** (REST client), two screens.
- **`contributions` = an exact linear-SHAP decomposition** — no `shap` library. From
  `model/feature_means.joblib` (mean of every transformed feature over the fit set, plus
  `coef`/`intercept`): $\phi_j = coef_j\,(x_j - \bar x_j)$, and
  $\sigma(z_{base} + \sum_j \phi_j) = P(\text{satisfied})$ exactly, with
  $z_{base} = intercept + coef\cdot\bar x$ so `base_p` ≈ the model's average prediction.
  `base_p` ≈ 0.63 (below the 0.79 dataset positive rate because training used
  `class_weight="balanced"`). The web renders it as a **diverging bar chart** (red =
  pull toward a bad review, green = toward a good one) with a waterfall line
  `63% → −50 pts → 8%`. Feature labels are built from the raw value
  (`delivery_delay_days > 0` → "Delivered 12 days late", one-hot → "Category: …",
  `txt__<tok>` → `comment: "<tok>"`).
- 📸 **Screenshot N-inf — notebook §23**: the reload-from-disk inference test — one raw
  order → `{ "prediction": "dissatisfied", "confidence": 0.9252 }`, and the
  `disk == in-memory` assertion passing.

### Web application (Appendix D template)

```
Web Application — Customer Behaviour (order satisfaction)
Framework:  React + Vite (single-page client)  +  FastAPI (POST /predict)
Endpoint:   POST http://<host>:8000/predict
Input:      order value & freight (BRL), payment method & instalments, item/seller counts,
            product category / weight / photos / description length, customer state,
            purchase date, promised delivery date, actual delivery date, review comment (PT, optional)
Output:     { prediction, confidence, p_satisfied, threshold, signals, contributions, model, representation }
```

- 📸 **Screenshot W1 — wizard step 1 (Product)**: the category emoji tile + the order
  fields, progress dots "Step 1 of 4". Caption: *the multi-step input, one section per
  screen; `Next` is disabled until required fields are valid.*
- 📸 **Screenshot W2 — wizard step 3 (Delivery & review)**: the live delivery-timeline
  SVG showing a late delivery + the review-comment box with example chips.
- 📸 **Screenshot W3 — the result screen**: *Negative review likely*, the `P(satisfied)`
  gauge with the cut-off tick, and the **"Why this prediction" diverging-bar chart**
  with the waterfall line `63% → −50 pts → 8%`. Caption: *the linear-SHAP breakdown —
  each factor's pull in log-odds; "Delivered N days late" dominates.*
- 📸 **Screenshot W4 — flip it**: use *Load a real order* to pick a 5★ example (or edit
  the delivery date earlier) → *Positive review likely*, chart mostly green.
- 📸 **Screenshot W5 — FastAPI `/docs`** `POST /predict` "Try it out".

**Figure explanation to write for each:** the input shown, the prediction returned,
how to read the gauge, and how the contribution bars sum (with the base rate) to the
final probability.

### Mobile application (Appendix E template)

```
Mobile Application — Customer Behaviour (order satisfaction)
Framework:  Flutter
Platform:   Android (emulator / device)
API:        POST http://<host>:8000/predict   (same endpoint as the web app)
```

- 📸 **Screenshot M1 — order form screen** (prefilled sample; required fields marked,
  numeric validation). Caption: *mobile input screen, fields from `GET /questions`.*
- 📸 **Screenshot M2 — result screen**: verdict, confidence, `P(satisfied)` bar,
  signals list, comment-term chips, interpretation. Caption: *the prediction on the
  device.*
- 📸 **Screenshot M3 — result after editing** the delivery date earlier + clearing the
  comment → flips to *Likely satisfied*. Caption: *re-submitting a changed order.*
- 📸 **Screenshot M4 — evidence the app calls the API**: the Uvicorn access log line
  `POST /predict 200` (or the run console) while the app is used. Caption: *inference
  runs on the server; the app is a REST client (`Training ≠ Inference`).*

**Figure explanation to write:** how the mobile UI collects the order, POSTs it to
`/predict`, and displays the returned verdict + confidence + signals.

---

## Reproducibility (shared section — customer-behaviour entries)

| Item | Value |
|---|---|
| Python | 3.13 |
| OS | Windows 11 |
| Key libraries | numpy 2.5, pandas 3.0, scikit-learn 1.9.0, scipy 1.18, matplotlib 3.11 (notebook + API); fastapi + uvicorn + pydantic 2 (API); React 18 + Vite 5, Node 22 (web); Flutter 3.19+ (mobile) |
| Random seed | `RANDOM_SEED = 42` (numpy + `random`; every split, subsample and estimator) |
| Dataset source | Kaggle `olistbr/brazilian-ecommerce` (9 CSVs), downloaded 2026-09-03; reviews file read as Latin-1 |
| Preprocessing | `ColumnTransformer`: median `SimpleImputer` → `log1p` (3 money cols) → `StandardScaler` (15 numeric); pass-through (2 binary); `OneHotEncoder(handle_unknown="ignore", min_frequency=30)` (3 categorical); `TfidfVectorizer(ngram_range=(1,2), min_df=5, max_features=20000, sublinear_tf=True)` on the cleaned comment. Fitted on train only. |
| Feature representation | 20 tabular columns → `x_tab ∈ ℝ^{43}` after one-hot; `x_txt ∈ ℝ^{~13000}` sparse; combined `d ≈ 13,000`; `X` sparse CSR, `y ∈ {0,1}^N`, `N = 95,824` |
| Train/val/test split | 70 / 15 / 15, stratified on `satisfied`, seed 42 (train 67,076 · val 14,374 · test 14,374); duplicate reviews collapsed before the split |
| Model hyperparameters | `LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)` on the tab+text representation (full 6-model table: notebook §17) |
| Evaluation metrics | test: Acc 0.862 · macro-F1 0.792 · ROC-AUC 0.859 · class-0 recall 0.672 · confusion `[[2034, 991], [997, 10352]]` |
| Saved pipeline | `model/model_pipeline.joblib` (tabular transformer + TF-IDF + LogisticRegression) |
| Saved model config | `model/feature_names.joblib`, `model/feature_means.joblib` (SHAP reference: transformed-feature means + `coef`/`intercept`), `model/input_schema.json` |
| API code | `customer_behaviour/api/` (FastAPI: `config.py`, `features.py`, `inference.py`, `schema.py`, `main.py`) |
| Web app code | `customer_behaviour/web/` (React + Vite) |
| Mobile app code | `customer_behaviour/mobile/` (Flutter) |
| Reproduce | execute `notebook/customer_behaviour.ipynb` (writes `model/`) → `pip install -r requirements.txt` → `uvicorn api.main:app --port 8000` → `npm --prefix web install && npm --prefix web run dev` → `flutter run` in `mobile/` |

---

## Cross-application comparison — customer-behaviour row

| Aspect | Customer Behaviour |
|---|---|
| Problem type | Binary classification |
| One observation | one delivered order (aggregated from several transaction tables) |
| Target | `satisfied` — `review_score ≥ 4` (0 / 1) |
| Input representation | `x_tab ∈ ℝ^{43}` (one-hot + scaled) **‖** TF-IDF `x_txt ∈ ℝ^{~13000}`; combined `d ≈ 13,000`, sparse |
| Data-quality issues | 59% missing comments; ~3% non-delivered orders dropped; 79% class imbalance; ~0.8–2.2% unmatched item/product rows; Latin-1 encoding |
| Best model | Logistic Regression (tab + text) |
| Main metric | Recall on the dissatisfied class — 0.672 at ROC-AUC 0.859 |
| Web deployment | Yes (React + Vite client → FastAPI) |
| Mobile deployment | Yes (Flutter) |
| Main limitation | order-level (not customer-level — ~97% single-order customers); misses dissatisfaction from causes not in the data (faulty product, wrong item) — those false negatives are ~100% on-time, ~85% comment-free; Portuguese-only text |

---

## Screenshot checklist

| ID | Where | Shows |
|---|---|---|
| N1 | notebook §4 | `df.info()` + `describe()` — assembled shape (98,673 × 25), dtypes, missingness |
| N2 | notebook §5 | data-quality issue → amount → action table |
| N3 | notebook §9 | IQR outlier counts + delivery-days band table + boxplots |
| N4 | notebook §12 | raw order → tabular vector + shapes; `Comment → Tokens → IDs → E` with B, T, d |
| N5 | notebook §10 | 2×2 EDA plot grid (plots 1–4) |
| N6 | notebook §10 | correlation heatmap (plot 5) |
| N7 | notebook §18 | 12-row model comparison + text-gain table + stability table |
| N8 | notebook §19 | classification report + confusion matrix + ROC |
| N-inf | notebook §23 | reload-from-disk inference test + `disk == in-memory` assertion |
| W1 | web app | wizard step 1 (Product) — category tile + fields + progress dots |
| W2 | web app | wizard step 3 — live delivery timeline + review-comment box |
| W3 | web app | result screen — verdict + gauge + SHAP diverging-bar chart + waterfall line |
| W4 | web app | a flipped result (5★ example) — chart mostly green |
| W5 | FastAPI `/docs` | the shared REST API |
| M1–M3 | mobile app | order form / result / result after editing |
| M4 | API terminal | a `POST /predict 200` log line while the mobile app is used |
