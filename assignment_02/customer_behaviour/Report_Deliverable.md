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

> **One-paragraph summary.** The task is a binary classification: from a Sephora skincare
> reviewer's **skin profile + the product + the review text**, predict whether they
> **recommend the product** (`is_recommended`). Two representations are compared: a
> **tabular** block (ROC-AUC ~0.77 linear / ~0.81 trees) and the **review text** as
> TF-IDF (ROC-AUC ~0.965). The deployed model is Logistic Regression on **both**
> (test ROC-AUC **0.964**, macro-F1 **0.858**, recall on the minority "won't recommend"
> class **0.876**). Section 14a is the required **data-leakage discussion**: the review
> and the recommend tick are written together, so much of the text's lead is
> co-authorship, not forecasting skill.

---

## 6.1 Problem description

- Objective sentence:
  *"The objective of this application is to analyse customer behaviour on a beauty
  marketplace and predict, from who the customer is and what they wrote, whether they
  **recommend** the product. The target is **`recommended` (= `is_recommended`, 1 if the
  reviewer ticked "recommends this product", else 0)**. It supports **review-consistency
  QA** (flag a 5-star review whose text says "would not repurchase"), **cold-start
  ranking** ("show products this skin type tends to recommend"), and **merchandising**
  (categories / price tiers with low recommend rates)."*
- `X` = a **review-level vector**: the reviewer's **skin profile** (`skin_type`,
  `skin_tone`, `eye_color`, `hair_color`), the **product** (`price_usd`,
  `secondary_category`, `brand_name`, `loves_count`, product review count, edition
  flags, listing richness), review **timing / engagement**, **and** the **review text**
  (`review_title` + `review_text`, English).
- `y` = `recommended` (binary). It is a **binary classification** problem.
- Explain the target choice: `is_recommended` is a **separate field** from the 1–5 star
  `rating` (which is excluded as leakage — §1.3 / §6.6); a single supervised target is
  required by the assignment; the business action ("surface / flag this review") is
  binary.
- One observation = **one product review**, from `reviews_500-750.csv` joined to
  `product_info.csv` on `product_id`. Source: notebook §1, §3.

## 6.2 Dataset

- Name: **Sephora Products and Skincare Reviews** (Kaggle, author `nadyinky`, scraped
  March 2023).
- Kaggle URL: <https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews>
  (CC0), placed in `data/sephora/`.
- Structure: 5 `reviews_*.csv` shards (split by product-id band) + `product_info.csv`.
  This application uses **one shard — `reviews_500-750.csv`** — joined to
  `product_info.csv`.
- Observations: **116,262** raw review rows across **249 skincare products** (each product
  has hundreds of reviews) → **104,313** after dropping **11,949** rows (11,803 with a
  blank `is_recommended`, 125 with empty body text, a few exact duplicates). Target
  positive rate: **0.8465** (84.7% recommend).
- Products span **~79 brands** and **12 `secondary_category` values** (*Cleansers, Eye
  Care, Masks, Moisturizers, Sunscreen, Treatments, Self Tanners, Value & Gift Sets,
  Lip Balms & Treatments, Mini Size, High Tech Tools, Wellness*). All products are in the
  **Skincare** primary category — a stated scope limitation.
- Review text present for **116,137 / 116,262 (99.9%)**; `review_title` present for
  **82,281 (70.8%)**. Skin-profile coverage: `skin_type` 92%, `skin_tone` 87%,
  `eye_color` 86%, `hair_color` 83%.
- Feature descriptions: include the notebook §11 role table (each column → numeric
  (log/scale) / binary / one-hot / text / target).
- **Tabular numeric** (9): `price_usd`, `loves_count`, `reviews` (product review count),
  `total_feedback_count`, `total_neg_feedback_count` (log1p → scale); `n_ingredients`,
  `n_highlights`, `review_age_days`, `pos_feedback_ratio` (scale).
- **Tabular binary** (6): `price_missing`, `has_title`, `limited_edition`, `new`,
  `online_only`, `sephora_exclusive`.
- **Tabular categorical** (6, one-hot): `skin_type` (4), `skin_tone` (~13), `eye_color`
  (5), `hair_color` (7), `secondary_category` (12), `brand_name` (~79).
- **Text**: `review_all` = `review_title + " . " + review_text` → TF-IDF.
- **Excluded** (§1.3): the review's own `rating` (1–5, co-authored with the label),
  `rating_product` (the product's displayed average — look-ahead / near-circular).

## 6.3 Customer representation

Show the transformation chain with concrete numbers (notebook §3, §12):
```
reviews_500-750.csv  +  product_info.csv ─join(product_id)─▶ one review row
        │
        ├─ build_features() ── ColumnTransformer ─▶ x_tab ∈ ℝ^{d_tab}   (skin profile + product + timing; one-hot)
        └─ TfidfVectorizer(review_all)          ─▶ x_txt ∈ ℝ^{d_txt}    (word 1–2 grams)
                                model input  X = [x_tab ‖ x_txt]
```
- **Assembly:** each review already carries the reviewer profile and part of the product
  fields; the join adds `secondary_category`, `loves_count`, product review count and the
  edition flags. This is the `Transactions → Customer/Product profile → Feature vector`
  step the assignment asks for. (Customer-level RFM is done separately in **Appendix B**;
  it is not fed to the classifier because ~78% of reviewers in this shard appear once.)
- **One raw review** (`skin_type=normal`, `price_usd=20`, `brand=dr. jart+`,
  `category=treatments`, `is_recommended=1`) → **the tabular feature row** it becomes
  (21 columns, e.g. `loves_count=54955`, `review_age_days=8`, `has_title=1`,
  `sephora_exclusive=1`) → after `impute → log1p(money/popularity) → scale` and
  `OneHotEncoder` → `x_tab ∈ ℝ^{138}`.
- Assembled frame `(116262, 30)`; cleaned modelling frame `(104313, 22)` (21 tabular
  columns + `review_all`); train `x_tab ∈ ℝ^{138}` dense, `x_txt ∈ ℝ^{~28,600}` sparse
  CSR, combined `d ≈ 28,700`; `y ∈ {0,1}^N`, `N = 104,313`, positive rate 0.8465.
- **Categorical encoding:** one-hot (`handle_unknown="ignore"`, `min_frequency=25`) — e.g.
  `skin_type ∈ {dry, combination, normal, oily}` → `oily → [0,0,0,1]`; rare brands fold
  into an "infrequent" bucket.
- **Scaling:** `StandardScaler` on all numerics; `price_usd`, `loves_count`, `reviews`,
  the two vote counts get `log1p` first (heavy right skew, §9).

### Text representation requirement (assignment §6.3)

Demonstrate the chain on **one real review** (notebook §12, second cell):
```
raw review ─clean─▶ tokens ─▶ token IDs ─▶ embedding lookup ─▶ E
"I will be the first to say that the price on these is a lot, but for what they do? …"
  → ['say','price','lot','unbeatable','deep','painful','nodules', …]   (T = 40 tokens)
  → [29, 26, 17, 39, 6, 22, 21, …]                                     (token IDs, 0 = PAD)
  → E  ∈ ℝ^{T×d}   with d = 16   (row i = embedding table[id_i])
  → E_batch ∈ ℝ^{B×T×d}   with B = 1, T = 40, d = 16
```
- Explain **each dimension**: `B` = reviews in the batch, `T` = tokens per review
  (padded/truncated), `d` = embedding width.
- State the deployment choice: the served model uses the **TF-IDF bag-of-words** form
  (a fixed `d_txt`-vector per review, `TfidfVectorizer(ngram_range=(1,2), min_df=10,
  max_features=40000, sublinear_tf=True, English stop-words)`) — the assignment's
  required *text-based linear classifier* — not the `B×T×d` sequence tensor, which is
  shown only to satisfy the representation requirement and is what an embedding/RNN model
  would consume.
- 📸 **Screenshot N4 — notebook §12 output**: the raw review + tabular vector + the
  shape/dtype lines, and the `Comment → Tokens → IDs → E` block with `B, T, d`.

## 6.4 Data cleaning

For each operation state **what** and **why** (notebook §5–§9):
- **Blank-target rows removed** — 11,803 reviews have no `is_recommended` (~10%). *Why:*
  cannot supervise them. Plus 125 empty-body reviews and a few exact duplicates.
  `N`: 116,262 → 104,313.
- **Skin-profile blanks kept as a category** — 2–17% of reviews leave a profile field
  empty. *Why:* the missingness is itself a behaviour; encoded as an explicit `__na__`
  one-hot level, not imputed.
- **Incidental numeric missingness** (`price_usd` 0%, `ingredients`/`highlights` a few %,
  `submission_time` parseable) — the ingredient / highlight counts and `review_age_days`
  are filled **inside the pipeline** with `SimpleImputer(median)` fitted on train only.
  *Why:* one training/inference code path, no leakage.
- **Case / spelling normalisation** — all categorical strings lower-cased;
  `eye_color "grey" → "gray"` merged.
- **Invalid values** — none: `rating` always 1–5, prices and counts non-negative.
- **Outliers not removed** (notebook §9) — `price_usd` (US$4–US$500+), `loves_count` and
  review length have long right tails but are genuine; the price-band table shows the
  recommend rate is nearly flat (~0.83–0.87), i.e. price alone is weak. Handled by
  `log1p` + `StandardScaler`; tree models are rank-invariant.
- **Text cleaning** — the TF-IDF vectoriser lower-cases, strips accents, drops English
  stop-words and builds the 1–2-gram vocabulary, fitted on **train only**.
- **Leakage quarantine** — the review's own star `rating` and `rating_product` are read
  only in §10 (EDA) and §14a (leakage demo); they are never model inputs (§6.6).
- 📸 **Screenshot N2 — notebook §5** data-quality table (issue → amount → planned action).
- 📸 **Screenshot N3 — notebook §9** IQR outlier counts + the price-band recommend-rate
  table + the three boxplots.

## 6.5 Interest discovery / EDA

Include ≥ 3 of the 6 plots from notebook §10; for **each** give Observation /
Interpretation / ML implication (already written under §10):
- **Plot 1 (bar)** — target class balance: **84.7% recommend / 15.3% not**.
- **Plot 2 (barh)** — recommend rate by `skin_type`: shifts a few points (0.82 → 0.85),
  and the gap is larger *within a product* (a serum built for dry skin under-delivers for
  oily reviewers) — the **customer-product fit** signal.
- **Plot 3 (barh)** — recommend rate by `secondary_category`: spread **~0.16**
  (peels/treatments, self-tanners low; moisturisers, gift sets high).
- **Plot 4 (bar)** — recommend rate vs the review's **own star rating** (an *excluded*
  feature): ~0.01 at 1★ → ~1.00 at 5★ — an almost perfect step; shown to justify the
  exclusion.
- **Plot 5 (topic buckets + frequent terms)** — the reviews are about the **product
  experience**: texture/feel ~42% (recommend rate 0.89), skin outcome/breakouts ~39%
  (0.90), scent ~26% (0.84), irritation ~23% (0.84), price/value ~13% (0.80). Frequent
  terms: `skin, product, love, like, use, face, dry, feel, moisturizer, serum`.
  Distinctive "won't recommend" terms: `waste money, disappointed, returning, meh, just
  okay, won repurchasing`.
- **Plot 6 (spread + mutual information)** — recommend-rate spread ~0.03–0.16 across skin
  type / category / price / loves deciles, ~0.40 across brands, vs ~0.99 across the
  excluded star rating. **Mutual information** with `is_recommended`: the excluded
  `rating` **0.35**; `log_loves` 0.037, `log_price` 0.018, `brand` 0.018;
  `secondary_category` 0.002; `skin_type` / `skin_tone` ~0.
- E-commerce cuts to mention: recommend | brand ranges 0.56 → 0.97; recommend | category
  0.79 → 0.95; recommend | 5★-text-context vs 1★; the topic-bucket rates.
- 📸 **Screenshot N5 — the 2×2 plot grid** (plots 1–4).
- 📸 **Screenshot N6 — §10 topic-bucket table + the spread / mutual-information table**
  (plots 5–6).

## 6.6 Model development

- **Eight pipelines** compared (notebook §17) — the assignment's six models for
  Application 3 (rows 1–6) plus two extras — across the **two representations** it asks
  about:

  | # | model | representation | assignment slot |
  |---|---|---|---|
  | 1 | Logistic Regression | tabular | *Logistic Regression* |
  | 2 | Decision Tree | tabular | *Decision Tree* |
  | 3 | Random Forest | tabular | *Random Forest* |
  | 4 | Linear SVM (`LinearSVC`) | tabular | *SVM* |
  | 5 | SGD (`loss="log_loss"`) | text (TF-IDF) | *text-based linear classifier* |
  | 6 | HistGradientBoosting | tabular | *another justified model* |
  | 7 | Complement Naive Bayes | text (TF-IDF) | extra |
  | 8 | Logistic Regression | **tabular + text** | **deployed** (the representation comparison) |

- Protocol: all trained on the same training split with the same preprocessing (fitted
  on train only); hyperparameters fixed and recorded (notebook §17 table); the deployed
  model is refitted on train + validation for persistence (§22).
- **Representation ladder** (notebook §18a, single LogReg, growing feature set):

  | rung | representation | val ROC-AUC | macro-F1 | recall(0) |
  |---|---|---|---|---|
  | A | skin profile only | 0.536 | 0.438 | 0.569 |
  | B | + product (category, brand, price, popularity) | 0.656 | 0.522 | 0.634 |
  | C | + review timing / engagement (**full tabular**) | **0.768** | 0.622 | 0.679 |
  | D | **text only** (TF-IDF) | **0.965** | 0.859 | 0.882 |
  | E | **tabular + text** (deployed) | **0.966** | 0.861 | 0.885 |

- **Eight models** (notebook §18b, validation — quote the executed table):

  | model | representation | ROC-AUC | macro-F1 | recall(0) | accuracy |
  |---|---|---|---|---|---|
  | **LogReg (tab + text)** | **combo** | **0.966** | **0.861** | **0.885** | 0.919 |
  | HistGradientBoosting | tab | 0.811 | 0.664 | 0.678 | 0.771 |
  | RandomForest | tab | 0.810 | 0.686 | 0.611 | 0.806 |
  | LogisticRegression | tab | 0.768 | 0.622 | 0.679 | 0.722 |
  | LinearSVC | tab | 0.767 | 0.622 | 0.676 | 0.723 |
  | **Decision Tree** | tab | 0.767 | 0.637 | 0.649 | 0.746 |
  | SGD (log-loss, text) | text | 0.965 | 0.856 | 0.885 | 0.916 |
  | ComplementNB (text) | text | 0.955 | 0.840 | 0.846 | 0.907 |
  | *baseline B — LogReg skin-profile only* | | *0.536* | | | *0.495* |
  | *baseline A — majority class* | | *0.500* | *0.458* | *0.000* | *0.847* |

  A single depth-12 Decision Tree lands with the linear models (~0.77) — it captures the
  main splits but not the skin×category×brand interactions the ensembles (~0.81) exploit.

- **Tabular vs tabular + text — the assignment's key question.** *Both representations
  have real predictive skill* — tabular reaches **~0.77 (linear) / ~0.81 (trees)** from
  the skin-profile fit + category + brand + price + popularity + engagement; the review
  text reaches **~0.965**. Adding the text to the tabular block gives a **small but
  consistent lift** (macro-F1 0.856 → 0.861, recall(0) up). They are **redundant more
  than complementary** — a customer who will withhold a recommendation has usually
  already *said so* in the review — but the comparison is genuine, unlike a corpus where
  one side is empty.
- 📸 **Screenshot N7 — notebook §18**: the 5-rung ladder table + the 8-model comparison
  table.

### 6.6a Data leakage (assignment requirement — notebook §14a)

Write this as its own subsection. Three effects, each measured in §14a:
1. **The review text is co-authored with the label.** `is_recommended` is a checkbox on
   the *same form* as the free-text box; reviews literally contain "highly recommend" /
   "would not repurchase". A text model partly **reads the verdict** — the ~0.965 is an
   upper bound for a *retrospective* setting, not a forecast.
2. **The review's own star `rating` IS the label in another column.** Adding it takes
   the tabular model to **ROC-AUC ~0.985** (Plot 4: recommend rate 0.01 → 1.00 across
   stars; MI 0.35). Excluded. (The product *average* rating is milder — 0.768 → 0.789 —
   but also excluded as look-ahead.)
3. **Engagement counts are post-publication.** `total_feedback_count` etc. are votes cast
   by other users *after* the review is posted. This model is **retrospective** (it scores
   reviews that already exist) so they are valid inputs; a **prospective** variant that
   drops them (`PROSPECTIVE_TAB`) falls to ROC-AUC **~0.66**.
- **Temporal split** (train older, score newer): text holds at ~0.96, tabular drops from
  ~0.81 to ~0.74 — what customers *write* travels across time better than the structured
  recommend pattern.
- **Conclusion for the report:** the deployed model uses **both** representations; its
  headline ROC-AUC ~0.964 is honest for a retrospective review-scoring task, and every
  results table also prints the tabular-only (~0.80) and text-only (~0.963) numbers so
  the leakage-inflated part of the text's lead stays visible.
- 📸 **Screenshot N7b — notebook §14a**: the representation-comparison print (tab /
  prospective / text / tab+text / +rating / +product-avg) + the temporal-split print.

## 6.7 Evaluation

- Chosen model: **Logistic Regression, tabular + text representation**, refit on the full
  training data, evaluated once on the held-out **test** set (notebook §19).
- Report (test, `N = 20,864`):

  | | precision | recall | F1 | support |
  |---|---|---|---|---|
  | class 0 — does not recommend | 0.681 | 0.876 | 0.766 | 3,202 |
  | class 1 — recommends | 0.976 | 0.926 | 0.950 | 17,662 |
  | accuracy | | | **0.918** | 20,864 |
  | macro avg | 0.828 | 0.901 | 0.858 | |

  **ROC-AUC 0.964.** Same test set, single representation: tabular-only
  (HistGradientBoosting) ROC-AUC **0.801**; text-only (SGD log-loss) **0.963**.
- Confusion matrix `[[2804, 398], [1315, 16347]]` (rows = actual, cols = predicted).
  **Interpret it:** FN (1,315) = *reviewers who won't recommend, predicted as
  recommending* → a misleading product page — the costly error; FP (398) = happy
  reviewers flagged → one wasted QA check; TN (2,804) = unhappy reviewers correctly
  caught.
- **Which metric matters most: recall on class 0 (does not recommend) — 0.876.** ROC-AUC
  (0.964) is the threshold-independent selection metric. Accuracy (0.918) is *not* the
  headline: the majority baseline scored 0.847 accuracy while catching zero unhappy
  reviewers.
- 📸 **Screenshot N8 — notebook §19** classification report + confusion-matrix heatmap +
  ROC curve.

### 6.7a Error analysis (notebook §20)

- **398 false negatives, 1,315 false positives.** The missed "won't recommend" reviews
  look ordinary on price / popularity / length, and **22% of them carry a 4–5★ rating**
  — the text reads positively but the customer ticked "no" over one dealbreaker (a scent,
  a price, a mild reaction) buried in otherwise happy prose. This is the **irreducible**
  ceiling of the task; fixes = aspect-level text features or a lower decision threshold.
- 📸 **Screenshot N9 — notebook §20** the FN/FP mean-feature table + the sample missed
  reviews.

## 6.8 Business interpretation

From notebook **Appendix A** — answer *"what behaviour/interest was discovered and how
could an e-commerce company use it?"*:
- **Discovered:** whether a customer recommends a skincare product is **largely
  recoverable from what they write** (text ROC-AUC ~0.965). The structured
  customer-product fit — skin-type match, category risk, brand, price tier, popularity —
  is a **real but weaker, redundant** signal (~0.77–0.81) that reaches most of the way on
  its own but adds little on top of the text. The text lead is partly co-authorship
  (§6.6a).
- **Uses:** *review-consistency QA* — flag reviews whose text disagrees with the ticked
  recommendation (5★ + "would not repurchase"); *cold-start ranking* — the **tabular-only**
  model (no text needed, ROC-AUC ~0.80) gives, per skin type / tone, which categories and
  brands that segment tends to recommend, usable before any review exists;
  *merchandising* — categories / price tiers with low recommend rates get better on-page
  guidance and samples.
- **What it cannot do:** it is retrospective (needs the review); its accuracy edge over a
  plain text classifier is small — its added value is the *structured* view.

## 6.9 Deployment

- Architecture (shared with the other two apps):
  ```
  User input → API request → validation → same preprocessing → saved model → prediction → result
  ```
- The service loads `model/model_pipeline.joblib` (tabular `ColumnTransformer` + fitted
  `TfidfVectorizer` + `LogisticRegression`) and **never re-fits** it. Leakage rule: the
  deployed preprocessing is the object fitted on the training split only.
- Inference flow for one request: raw review dict → pydantic `ReviewInput` validation →
  stateless `build_features()` (review_age_days / price_missing / has_title /
  n_ingredients / n_highlights / pos_feedback_ratio / lower-cased categoricals /
  `review_all`) → pipeline → `P(recommend)` → recommend / not-recommend at threshold 0.5.
- Example response (`api/README.md` has the full one):
  ```json
  { "prediction": "not recommend", "confidence": 0.996, "p_recommend": 0.004,
    "threshold": 0.5,
    "review_terms": { "toward": [{"term":"love","effect":0.21}],
                      "against": [{"term":"wanted love","effect":-1.80},
                                  {"term":"broke","effect":-1.44},
                                  {"term":"returned","effect":-1.07}] },
    "signals": { "skin_type":"Oily", "category":"Moisturizers", "brand":"Skinfix",
                 "price_usd":32.0, "price_tier":"mid", "review_tokens":27, "has_title":true },
    "contributions": { "base_p": 0.8723, "final_p": 0.004, "dataset_base_rate": 0.846,
      "items": [ {"label":"review term: wanted love","kind":"text","effect":-1.80},
                 {"label":"Brand: Skinfix","kind":"tabular","effect":0.29} ],
      "other_effect": -0.31 },
    "model": "LogisticRegression(class_weight=balanced)",
    "representation": "tabular + TF-IDF(text)" }
  ```
- Web framework: **FastAPI** (`POST /predict`). Client: **React (Vite)**, a
  **full-viewport 3-step wizard** (Your skin profile → The product → The review) then a
  full-screen result. Mobile: **Flutter** (REST client), two screens.
- **`contributions` = an exact linear-SHAP decomposition** — no `shap` library. From
  `model/feature_means.joblib` (mean of every transformed feature over the fit set, plus
  `coef`/`intercept`, 37,240 features): $\phi_j = coef_j\,(x_j - \bar x_j)$, and
  $\sigma(z_{base} + \sum_j \phi_j) = P(\text{recommend})$ exactly, with
  $z_{base} = intercept + coef\cdot\bar x$. `base_p` is the model's neutral point for an
  average review (class-balanced, so not the 84.7% dataset rate). The web renders it as a
  **diverging bar chart** (red = pull toward "won't recommend", green = toward
  "recommends") with a waterfall line. Feature labels are built from the raw value
  (`brand_name_skinfix` → "Brand: Skinfix", `txt__<tok>` → "review term: `<tok>`").
  `review_terms` re-expresses the text branch as the words present in *this* review, split
  by sign.
- 📸 **Screenshot N-inf — notebook §23**: the reload-from-disk inference test — a positive
  review → `{"prediction":"recommend","p_recommend":0.9376}`, a negative one →
  `{"prediction":"not recommend","p_recommend":0.0026}`, and the `disk == in-memory`
  assertion passing.

### Web application (Appendix D template)

```
Web Application — Customer Behaviour (product recommendation)
Framework:  React + Vite (single-page client)  +  FastAPI (POST /predict)
Endpoint:   POST http://<host>:8000/predict
Input:      skin_type / skin_tone / eye_color / hair_color, secondary_category, brand_name,
            price_usd, loves_count, reviews, review_title, review_text  (rating is NOT sent)
Output:     { prediction, confidence, p_recommend, threshold, review_terms, signals,
              contributions, model, representation }
```

- 📸 **Screenshot W1 — wizard step 1 (Your skin profile)**: the skin type / tone / eye /
  hair dropdowns, the stepper "Step 1 of 3", API-connected pill. Caption: *the multi-step
  input, one section per screen; `Next` is disabled until required fields are valid.*
- 📸 **Screenshot W2 — wizard step 3 (The review)**: the title + body boxes with an
  example loaded and the real-review chips.
- 📸 **Screenshot W3 — the result screen (verdict + meter)**: *This customer probably
  would not recommend*, the `P(recommend)` meter with the 50% cut-off tick and the
  "read it as…" line, and the "what the model saw" key–value list.
- 📸 **Screenshot W4 — the result screen (SHAP)**: the **"Why this prediction"
  diverging-bar chart** with the waterfall line, and the review-term chips (toward /
  against). Caption: *the linear-SHAP breakdown — each factor's pull in log-odds; review
  terms dominate (§6.6a).*
- 📸 **Screenshot W5 — FastAPI `/docs`** `POST /predict` "Try it out".

**Figure explanation to write for each:** the input shown, the prediction returned, how
to read the meter, and how the contribution bars sum (with the base rate) to the final
probability.

### Mobile application (Appendix E template)

```
Mobile Application — Customer Behaviour (product recommendation)
Framework:  Flutter
Platform:   Android (emulator / device)
API:        POST http://<host>:8000/predict   (same endpoint as the web app)
```

- 📸 **Screenshot M1 — review form screen** (prefilled sample; fields grouped by section
  from `GET /questions`; required fields marked). Caption: *mobile input screen.*
- 📸 **Screenshot M2 — result screen**: verdict, `P(recommend)` bar with the cut-off,
  the "what the model saw" list, the review-term chips, interpretation. Caption: *the
  prediction on the device.*
- 📸 **Screenshot M3 — result after swapping in a positive example** → flips to *Would
  probably recommend*. Caption: *re-submitting a changed review.*
- 📸 **Screenshot M4 — evidence the app calls the API**: the Uvicorn access log line
  `POST /predict 200` while the app is used. Caption: *inference runs on the server; the
  app is a REST client (`Training ≠ Inference`).*

**Figure explanation to write:** how the mobile UI collects the review, POSTs it to
`/predict`, and displays the returned verdict + probability + signals.

---

## Reproducibility (shared section — customer-behaviour entries)

| Item | Value |
|---|---|
| Python | 3.13 |
| OS | Windows 11 |
| Key libraries | numpy, pandas, scikit-learn **1.9.0**, scipy, matplotlib (notebook + API); fastapi + uvicorn + pydantic 2 (API); React 18 + Vite 5, Node (web); Flutter 3.19+ (mobile) |
| Random seed | `RANDOM_SEED = 42` (numpy + `random`; every split and estimator) |
| Dataset source | Kaggle `nadyinky/sephora-products-and-skincare-reviews` (CC0); `data/sephora/reviews_500-750.csv` joined to `product_info.csv` on `product_id` |
| Rows | 116,262 raw → **104,313** after dropping blank-target / empty-text / duplicate rows; recommend rate **0.8465** |
| Excluded features | the review's own `rating` (1–5) and `rating_product` (product average) — co-authored / near-circular with the label (§6.6a) |
| Preprocessing | `ColumnTransformer`: median `SimpleImputer` → `log1p` (5 money/popularity/vote cols) → `StandardScaler`; scale (4 numeric); pass-through (6 binary); `OneHotEncoder(handle_unknown="ignore", min_frequency=25)` (6 categorical); `TfidfVectorizer(ngram_range=(1,2), min_df=10, max_features=40000, sublinear_tf=True, stop_words="english")` on `review_all`. Fitted on train only. |
| Feature representation | 21 tabular columns → `x_tab ∈ ℝ^{138}` after one-hot; `x_txt ∈ ℝ^{~28,600}` sparse; combined `d ≈ 28,700`, sparse CSR; `y ∈ {0,1}^N`, `N = 104,313` |
| Train/val/test split | `StratifiedGroupKFold(n_splits=5)` **grouped on `author_id`** (no reviewer spans splits; ~22% review > once) → train 62,587 · val 20,862 · test 20,864, each at recommend rate 0.8465 |
| Model hyperparameters | `LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=42)` on the tabular + text representation (full 8-model table: notebook §17) |
| Evaluation metrics | test: Acc 0.918 · macro-F1 0.858 · ROC-AUC 0.964 · class-0 recall 0.876 · confusion `[[2804, 398], [1315, 16347]]`; single-representation on the same test set: tab-only 0.801, text-only 0.963 |
| Saved pipeline | `model/model_pipeline.joblib` (~2.0 MB — tabular transformer + TF-IDF + LogisticRegression) |
| Saved model config | `model/feature_names.joblib`, `model/feature_means.joblib` (linear-SHAP reference: 37,240 transformed-feature means + `coef`/`intercept`), `model/input_schema.json` (raw fields, engineered fields, framing / leakage note) |
| API code | `customer_behaviour/api/` (FastAPI: `config.py`, `features.py`, `inference.py`, `schema.py`, `main.py`, `make_samples.py`) |
| Web app code | `customer_behaviour/web/` (React + Vite, 3-step wizard) |
| Mobile app code | `customer_behaviour/mobile/` (Flutter, 2 screens) |
| Reproduce | download `reviews_500-750.csv` + `product_info.csv` from Kaggle into `data/sephora/` → `python -m venv .venv && . .venv/Scripts/activate` → `pip install -r requirements.txt` → execute `notebook/customer_behaviour.ipynb` (writes `model/`) → `python api/make_samples.py` → `python -m uvicorn api.main:app --port 8000` (from `customer_behaviour/`) → `npm --prefix web install && npm --prefix web run dev` → `flutter run` in `mobile/` |

---

## Cross-application comparison — customer-behaviour row

| Aspect | Customer Behaviour |
|---|---|
| Problem type | Binary classification |
| One observation | one product review (reviewer profile + product join) |
| Target | `recommended` — `is_recommended` (0 / 1), separate from the 1–5★ rating |
| Input representation | `x_tab ∈ ℝ^{138}` (one-hot + scaled) **‖** TF-IDF `x_txt ∈ ℝ^{~28,600}`; combined `d ≈ 28,700`, sparse |
| Data-quality issues | ~10% blank target dropped; 84.7% class imbalance; 2–17% missing skin-profile fields (kept as `__na__`); single category (Skincare); English-only text |
| Best model | Logistic Regression (tabular + text) |
| Main metric | Recall on the "does not recommend" class — **0.876** at ROC-AUC **0.964** |
| Web deployment | Yes (React + Vite 3-step wizard → FastAPI) |
| Mobile deployment | Yes (Flutter) |
| Main limitation | the review text is **co-authored** with the recommend tick, so ~0.96 is partly leakage — the leak-safe tabular signal is ~0.80 (§6.6a); retrospective only (needs the review); ~78% single-review reviewers so RFM Frequency is near-degenerate (Appendix B); one product category |

---

## Appendix B — customer segmentation (RFM + K-Means, notebook Appendix B)

- Each reviewer → `x_i = [R_i, F_i, M_i, \overline{price}_i, \text{recommend-rate}_i,
  \overline{rating}_i, C_{i1}, \dots, C_{ik}]` (recency days, review count, total spend,
  mean price, recommend rate, mean rating, per-category share vector).
- **73,819 reviewers**; `F ≥ 2` for **22.1%**, `F ≥ 3` for **7.7%** (median F = 1) — so
  Frequency has little variance and the segmentation is driven by **recency**, **spend /
  average price**, **recommend-rate**, **mean rating** and the **category-mix vector**.
- After `log1p` on the skewed columns + `StandardScaler`, **K-Means** with `k = 5`
  (the inertia curve is smooth and the silhouette rises slowly with `k`; `k = 5` is the
  coarsest interpretable split). Report the profile table (median R/F/M, mean recommend
  rate, dominant categories per segment) + the PCA(2) scatter.
- 📸 **Screenshot N-B — notebook Appendix B**: the elbow/silhouette plots + the
  5-segment profile table + the PCA scatter.

---

## Screenshot checklist

| ID | Where | Shows |
|---|---|---|
| N1 | notebook §4 | `df.info()` — assembled shape (116,262 × 30), dtypes, `is_recommended` non-null 104,459, skin-profile coverage |
| N2 | notebook §5 | data-quality issue → amount → action table |
| N3 | notebook §9 | IQR outlier counts + price-band recommend-rate table + boxplots |
| N4 | notebook §12 | raw review → tabular vector + shapes; `Comment → Tokens → IDs → E` with B, T, d |
| N5 | notebook §10 | 2×2 EDA plot grid (plots 1–4) |
| N6 | notebook §10 | topic-bucket table + spread / mutual-information table (plots 5–6) |
| N7 | notebook §18 | 5-rung representation ladder + 7-model comparison table |
| N7b | notebook §14a | representation-comparison + temporal-split prints (the leakage measurements) |
| N8 | notebook §19 | classification report + confusion matrix + ROC curve |
| N9 | notebook §20 | FN/FP mean-feature table + sample missed reviews |
| N-inf | notebook §23 | reload-from-disk inference test + `disk == in-memory` assertion |
| N-B | notebook Appendix B | elbow/silhouette + 5-segment profile table + PCA scatter |
| W1 | web app | wizard step 1 (skin profile) — dropdowns + stepper |
| W2 | web app | wizard step 3 (the review) — title/body + example chips |
| W3 | web app | result screen — verdict + `P(recommend)` meter + "what the model saw" |
| W4 | web app | result screen — SHAP diverging-bar chart + waterfall + review-term chips |
| W5 | FastAPI `/docs` | the shared REST API `POST /predict` try-it |
| M1–M3 | mobile app | review form / result / result after swapping the example |
| M4 | API terminal | a `POST /predict 200` log line while the mobile app is used |
