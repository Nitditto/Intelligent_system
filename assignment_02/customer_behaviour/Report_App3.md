# Section 6 — Application 3: E-Commerce Customer Behaviour & Interest

*Finished report section. Paste into the combined Assignment-02 report as Section 6;
the App-3 cells for the Executive Summary, the Data-Representation Summary and the
Cross-Application Comparison are at the end. Replace each 📸 with the real screenshot.*
*All numbers are the executed values from `notebook/customer_behaviour.ipynb`.*

---

## 6.1 Problem description

The objective is to analyse **customer behaviour and interest** on a beauty marketplace
and predict, from **who the customer is** and **what they wrote**, whether they
**recommend the product**. The supervised target is the reviewer's own recommend tick:

```
recommended = is_recommended  ∈ {0, 1}     (1 = "recommends this product")
```

`X` = a **review-level vector**: the reviewer's skin profile (`skin_type`, `skin_tone`,
`eye_color`, `hair_color`), the product (`price_usd`, `secondary_category`, `brand_name`,
`loves_count`, review count, edition flags, listing richness), review timing / community
engagement, **and** the review text (`review_title` + `review_text`, English). `y` =
`recommended`. It is a **binary classification** problem; one observation is **one product
review**.

**Why this target.** `is_recommended` is a field *separate* from the 1–5 star `rating`
(which is excluded as leakage — §6.6a); a single supervised target is required; and the
business action — surface or flag a review — is binary. The prediction supports
**review-consistency QA** (a 5-star review whose text says "would not repurchase"),
**cold-start ranking** ("show products this skin type recommends") and **merchandising**.

## 6.2 Dataset

| Field | Value |
|---|---|
| Name / URL | **Sephora Products and Skincare Reviews** (author `nadyinky`, scraped March 2023) — `kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews` (CC0) |
| Files used | `reviews_500-750.csv` (one review-id shard) joined to `product_info.csv` on `product_id` |
| Rows | **116 262** raw reviews → **104 313** after cleaning (§6.4); 1 row = 1 review |
| Products / brands / categories | 249 products · ~79 brands · 12 `secondary_category` values (all in the **Skincare** primary category) |
| Time span | reviews up to 2023-03-21 |
| Target | `recommended` = `is_recommended` (0/1); positive rate **0.8465** |

**Feature types** (notebook §11):

- **Numeric, log1p + scale** (5): `price_usd`, `loves_count`, `reviews`,
  `total_feedback_count`, `total_neg_feedback_count`.
- **Numeric, scale** (4): `n_ingredients`, `n_highlights`, `review_age_days`,
  `pos_feedback_ratio`.
- **Binary passthrough** (6): `price_missing`, `has_title`, `limited_edition`, `new`,
  `online_only`, `sephora_exclusive`.
- **Categorical, one-hot** (6): `skin_type` (4), `skin_tone` (~13), `eye_color` (5),
  `hair_color` (7), `secondary_category` (12), `brand_name` (~79).
- **Text**: `review_all` = `review_title + " . " + review_text`.
- **Excluded (leakage, §6.6a)**: the review's own `rating` (1–5, co-authored with the
  label) and the product's average rating `rating_product`.

*(This shard is already review-level, so no transaction→customer aggregation is needed
for the classifier. Customer-level RFM `xᵢ = [Rᵢ, Fᵢ, Mᵢ, Cᵢ₁…C_ik]` is built separately
for the K-Means segmentation — see the note in §6.5.)*

## 6.3 Customer representation

```
reviews_500-750.csv + product_info.csv ─join(product_id)─▶ one review row
        │
        ├─ build_features()  ── ColumnTransformer ─▶ x_tab ∈ ℝ^{d_tab}   (skin + product + timing; one-hot)
        └─ TfidfVectorizer(review_all)            ─▶ x_txt ∈ ℝ^{d_txt}    (word 1–2 grams)
                        X = [x_tab ‖ x_txt]      y ∈ {0,1}^N
```

**Tabular part.** One raw review (e.g. `skin_type=normal`, `price_usd=20`,
`brand=dr. jart+`, `category=treatments`, `is_recommended=1`) → a 21-column feature row
(`loves_count=54955`, `review_age_days=8`, `has_title=1`, `sephora_exclusive=1`, …) →
`SimpleImputer(median) → log1p` on the money/popularity columns `→ StandardScaler`; binary
flags pass through; `OneHotEncoder(handle_unknown="ignore", min_frequency=25)` on the six
categoricals → **`x_tab ∈ ℝ^{138}`** (dense).

**Text part — required `Comment → Tokens → Token IDs → Embedding` demonstration** (notebook
§12) on a real review:

```
Comment  : "I will be the first to say that the price on these is a lot, but for what they do? …"
Tokens   : ['say','price','lot','unbeatable','deep','painful','nodules','thanks', …]      (T = 40)
Token IDs: [29, 26, 17, 39, 6, 22, 21, 36, …]                                            (index into a 41-row vocab, 0 = PAD)
Embedding: each ID indexes a row of a (|V|+1) × d table  →  E ∈ ℝ^{T×d} = ℝ^{40×16}
Batch    : E_batch ∈ ℝ^{B×T×d} = ℝ^{1×40×16}
```

- **`B`** = reviews in the batch (here 1); **`T`** = tokens per review after
  padding/truncation (40); **`d`** = embedding width (16).
- **Deployment choice.** The served model uses the **TF-IDF bag-of-words** form
  (`TfidfVectorizer`, `ngram_range=(1,2)`, `min_df=10`, `max_features=40000`,
  `sublinear_tf=True`, English stop-words) — the assignment's *text-based linear
  classifier*. On the training split this gives `x_txt ∈ ℝ^{28 595}` (sparse). The
  `B×T×d` sequence tensor above is shown only to satisfy the representation requirement;
  it is what an embedding/RNN model would consume.

**Combined model input.** Horizontally stack → `X = [x_tab ‖ x_txt] ∈ ℝ^{N × 28 733}`,
sparse CSR; `y ∈ {0,1}^N`, `N = 104 313`, positive rate 0.8465. Assembled frame
`(116262, 30)` → cleaned modelling frame `(104313, 22)` (21 tabular columns + `review_all`).

📸 **N4 — notebook §12**: the raw review + tabular feature row + the shape/dtype lines, and
the `Comment → Tokens → IDs → E` block showing `B = 1, T = 40, d = 16`.

## 6.4 Data cleaning

For each operation, **what** and **why** (notebook §5–§9):

- **Blank-target rows removed** — 11 803 reviews (~10%) have no `is_recommended`; *why:*
  cannot supervise them. Plus 125 empty-body reviews and a few exact duplicates.
  `N`: 116 262 → **104 313**.
- **Skin-profile blanks kept as a category** — 2–17% of reviews leave a profile field
  empty; *why:* the missingness is a real behaviour, so it becomes an explicit `__na__`
  one-hot level, not an imputed value.
- **Incidental numeric missingness** — the ingredient / highlight counts and
  `review_age_days` are filled **inside the pipeline** with `SimpleImputer(median)`
  fitted on train only; *why:* one code path for training and serving, no leakage.
- **Case / spelling normalisation** — all categorical strings lower-cased;
  `eye_color "grey"` merged into `"gray"`; *why:* so `"Grey"` and `"grey"` are one level.
- **Text cleaning** — the TF-IDF vectoriser lower-cases, strips accents, drops English
  stop-words and builds the 1–2-gram vocabulary, **fitted on train only**; *why:* fitting
  the vocabulary on the full data would leak test-set words.
- **Outliers not removed** (notebook §9) — `price_usd` (US$4–US$500+), `loves_count` and
  review length have long right tails but are genuine; the price-band table shows the
  recommend rate is nearly flat (~0.83–0.87), i.e. price alone is weak. Handled by
  `log1p` + `StandardScaler`; tree models are rank-invariant.
- **Leakage quarantine** — the review's own star `rating` and `rating_product` are read
  only in the EDA and the §6.6a leakage demo; never model inputs.

📸 **N2 — notebook §5**: data-quality table (issue → amount → action).
📸 **N3 — notebook §9**: IQR outlier counts + price-band recommend-rate table + boxplots.

## 6.5 Interest discovery / EDA

Six figures (notebook §10); each with Observation / Interpretation / ML implication.

- **Class balance (bar).** 84.7% recommend / 15.3% not → predicting "recommend" for
  everyone already scores 0.847 accuracy but flags nobody; stratify, class-weight, judge
  on macro-F1 / recall(0) / ROC-AUC.
- **Recommend rate by `skin_type` (barh).** Shifts a few points (0.82 → 0.85) overall,
  and the gap is larger *within a product* — a serum built for dry skin under-delivers
  for oily reviewers. This **customer–product fit** is the main structured signal;
  `skin_type` / `skin_tone` one-hot are core features.
- **Recommend rate by `secondary_category` (barh).** Spread **~0.16** — peels/treatments
  and self-tanners rate lower than moisturisers and gift sets.
- **Recommend rate vs the review's own star rating (bar).** ~0.01 at 1★ → ~1.00 at 5★, an
  almost perfect step → `rating` *is* the label in another column; **excluded** (§6.6a).
- **What the reviews talk about (topic buckets + frequent terms).** Reviews are about the
  **product experience**: texture/feel ~42% (recommend rate 0.89), skin outcome/breakouts
  ~39% (0.90), scent ~26% (0.84), irritation ~23% (0.84), price/value ~13% (0.80).
  Frequent terms: `skin, product, love, like, use, face, dry, feel, moisturizer, serum`.
  Distinctive "won't recommend" terms: `waste money, disappointed, returning, meh, just
  okay, won repurchasing`. → a TF-IDF model has real content to learn from.
- **Spread + mutual information.** Recommend-rate spread ~0.03–0.16 across skin type /
  category / price / loves deciles, **~0.40 across brands**, vs ~0.99 across the excluded
  star rating. Mutual information with `is_recommended`: the excluded `rating` **0.35**;
  `log_loves` 0.037, `log_price` 0.018, `brand` 0.018; `secondary_category` 0.002;
  `skin_type` / `skin_tone` ~0. → no single structured attribute is decisive, but several
  weak ones combine to the ~0.77–0.81 tabular model (§6.6).

**Customer segmentation (K-Means, notebook Appendix B).** Each reviewer →
`xᵢ = [Rᵢ, Fᵢ, Mᵢ, avg_priceᵢ, recommend-rateᵢ, avg-ratingᵢ, Cᵢ₁…C_ik]` (recency days,
review count, total spend, mean price, recommend rate, mean rating, per-category share
vector). **73 819 reviewers**; `F ≥ 2` for **22.1%**, `F ≥ 3` for **7.7%** (median
`F = 1`), so Frequency has little variance and K-Means (`k = 5`) is driven by recency,
spend, recommend rate, mean rating and the category-mix vector. Descriptive only, not fed
to the classifier.

📸 **N5 — notebook §10**: the 2×2 plot grid.
📸 **N6 — notebook §10**: the topic-bucket table + the spread / mutual-information table.
📸 **N-B — notebook Appendix B**: elbow/silhouette + 5-segment profile table + PCA scatter.

## 6.6 Model development

**Eight pipelines** (notebook §17) — the assignment's six models for Application 3
(rows 1–6) plus two extras — across the two representations it asks about:

| # | model | representation | assignment slot |
|---|---|---|---|
| 1 | Logistic Regression | tabular | *Logistic Regression* |
| 2 | Decision Tree (`max_depth=12`) | tabular | *Decision Tree* |
| 3 | Random Forest (`n_estimators=350`) | tabular | *Random Forest* |
| 4 | Linear SVM (`LinearSVC`, `C=0.5`) | tabular | *SVM* |
| 5 | SGD (`loss="log_loss"`) | text (TF-IDF) | *text-based linear classifier* |
| 6 | HistGradientBoosting | tabular | *another justified model* |
| 7 | Complement Naive Bayes | text (TF-IDF) | extra — NB for imbalanced text |
| 8 | Logistic Regression | **tabular + text** | **deployed** (the representation comparison) |

All are class-weighted, hyperparameters fixed (no tuning search — the comparison is about
*representations*), fitted on the training split only; the deployed model is refitted on
train + validation for persistence (§6.9).

**Representation ladder** (notebook §18a, one Logistic Regression, growing feature set,
validation):

| rung | representation | ROC-AUC | macro-F1 | recall(0) |
|---|---|---|---|---|
| A | skin profile only | 0.536 | 0.438 | 0.569 |
| B | + product (category, brand, price, popularity) | 0.656 | 0.522 | 0.634 |
| C | + review timing / engagement (**full tabular**) | **0.768** | 0.622 | 0.679 |
| D | **text only** (TF-IDF) | **0.965** | 0.859 | 0.882 |
| E | **tabular + text** (deployed) | **0.966** | 0.861 | 0.885 |

**Eight models** (notebook §18b, validation):

| model | representation | ROC-AUC | macro-F1 | recall(0) | accuracy |
|---|---|---|---|---|---|
| **Logistic Regression (tab + text)** | **combo** | **0.966** | **0.861** | **0.885** | 0.919 |
| HistGradientBoosting | tab | 0.811 | 0.664 | 0.678 | 0.771 |
| Random Forest | tab | 0.810 | 0.686 | 0.611 | 0.806 |
| Logistic Regression | tab | 0.768 | 0.622 | 0.679 | 0.722 |
| Decision Tree | tab | 0.767 | 0.637 | 0.649 | 0.746 |
| Linear SVM (`LinearSVC`) | tab | 0.767 | 0.622 | 0.676 | 0.723 |
| SGD (log-loss, text) | text | 0.965 | 0.856 | 0.885 | 0.916 |
| Complement Naive Bayes (text) | text | 0.955 | 0.840 | 0.846 | 0.907 |
| *baseline B — LogReg skin-profile only* | | *0.536* | *0.438* | *0.569* | *0.495* |
| *baseline A — majority class* | | *0.500* | *0.458* | *0.000* | *0.847* |

**Tabular vs tabular + text — the assignment's key question.** *Both representations
carry real predictive skill.* The tabular block reaches **ROC-AUC ~0.77 (linear) / ~0.81
(trees)** from the skin-profile fit + category + brand + price + popularity + engagement;
a single Decision Tree lands with the linear models (~0.77). The **review text** reaches
**~0.965**. Adding the text to the tabular block gives a **small but consistent lift**
(macro-F1 0.856 → 0.861, recall(0) up). The two are **redundant more than complementary**
— a customer who will withhold a recommendation has usually already *said so* in the
review — but the comparison is genuine: neither side is empty.

📸 **N7 — notebook §18**: the 5-rung ladder table + the 8-model comparison table.

### 6.6a Data leakage (assignment requirement — notebook §14a)

Three effects, each measured:

1. **The review text is co-authored with the label.** `is_recommended` is a checkbox on
   the *same form* as the free-text box; reviews literally contain "highly recommend" /
   "would not repurchase". A text model partly **reads the verdict** — the ~0.965 is an
   upper bound for a **retrospective** setting, not a forecast. It is not *pure* leakage:
   the text also carries genuine product experience, so tab+text still generalises across
   a temporal split.
2. **The review's own star `rating` IS the label in another column.** Adding it takes the
   tabular model to ROC-AUC **~0.985** (recommend rate 0.01 → 1.00 across stars; mutual
   information 0.35). **Excluded.** The *product average* rating is milder (0.768 → 0.789)
   but also excluded as look-ahead.
3. **Engagement counts are post-publication.** `total_feedback_count` etc. are votes cast
   by other users *after* the review is posted. This model is **retrospective** (it scores
   reviews that already exist) so they are valid inputs; a **prospective** variant that
   drops them falls to ROC-AUC **~0.66**.

**Temporal split** (train older, score newer): text holds at ~0.96, tabular drops from
~0.81 to ~0.74 — what customers *write* travels across time better than the structured
recommend pattern.

**Also**: no scaler/encoder/vectoriser is fitted on validation or test data — every
fitted object is fitted on `X_train` only, and the deployed API loads that exact pipeline
(§6.9). `user_avg_rating` / `rating_product` snapshot columns are not used.

📸 **N7b — notebook §14a**: the representation-comparison print + the temporal-split print.

## 6.7 Evaluation

Chosen model: **Logistic Regression, tabular + text**, refit on the full training data,
evaluated once on the held-out **test** set (`N = 20 864`; notebook §19):

| | precision | recall | F1 | support |
|---|---|---|---|---|
| class 0 — does not recommend | 0.681 | **0.876** | 0.766 | 3 202 |
| class 1 — recommends | 0.976 | 0.926 | 0.950 | 17 662 |
| accuracy | | | **0.918** | 20 864 |
| macro avg | 0.828 | 0.901 | 0.858 | |

**ROC-AUC 0.964.** Same test set, single representation: tabular-only
(HistGradientBoosting) ROC-AUC **0.801**; text-only (SGD log-loss) **0.963**.

**Confusion matrix `[[2804, 398], [1315, 16347]]`** (rows = actual, cols = predicted).
FN (1 315) = reviewers who won't recommend, predicted as recommending → a misleading
product page — the **costly error**. FP (398) = happy reviewers flagged → one wasted QA
check. TN (2 804) = unhappy reviewers correctly caught.

**Which metric matters most: recall on the "does not recommend" class — 0.876.** ROC-AUC
(0.964) is the threshold-independent selection metric. Accuracy (0.918) is *not* the
headline: the majority baseline scored 0.847 accuracy while catching **zero** unhappy
reviewers.

### 6.7a Error analysis (notebook §20)

398 false negatives, 1 315 false positives. The missed "won't recommend" reviews look
ordinary on price / popularity / length, and **22% of them carry a 4–5★ rating** — the
text reads positively but the customer ticked "no" over one dealbreaker (a scent, a
price, a mild reaction) buried in otherwise happy prose. This is the **irreducible**
ceiling of the task; fixes = aspect-level text features or a lower decision threshold.

📸 **N8 — notebook §19**: classification report + confusion-matrix heatmap + ROC curve.
📸 **N9 — notebook §20**: the FN/FP mean-feature table + the sample missed reviews.

## 6.8 Business interpretation

**Discovered.** Whether a customer recommends a skincare product is **largely recoverable
from what they write** (text ROC-AUC ~0.965). The structured customer–product fit —
skin-type match, category risk, brand, price tier, popularity — is a **real but weaker,
redundant** signal (~0.77–0.81) that reaches most of the way alone but adds little on top
of the text. Part of the text's lead is co-authorship (§6.6a).

**Uses for an e-commerce company:**

- *Review-consistency QA* — flag reviews whose text disagrees with the ticked
  recommendation (5★ + "would not repurchase"); the main production use of the tab+text
  model.
- *Cold-start ranking* — the **tabular-only** model (no text needed, ROC-AUC ~0.80) gives,
  per skin type / tone, which categories and brands that segment recommends — usable
  before any review exists.
- *Merchandising* — categories / price tiers with low recommend rates (peels, high-price
  treatments) get better on-page guidance and samples.

**What it cannot do.** It is retrospective (needs the review); its accuracy edge over a
plain text classifier is small — its added value is the *structured* view.

## 6.9 Deployment

```
Raw review JSON → Pydantic validation → stateless build_features() → SAME fitted pipeline → P(recommend) → JSON
```

- **Persistence (notebook §22).** `model/model_pipeline.joblib` (~2.0 MB — tabular
  `ColumnTransformer` + fitted `TfidfVectorizer` + `LogisticRegression`),
  `feature_names.joblib`, `feature_means.joblib` (linear-SHAP reference: 37 240
  transformed-feature means + `coef`/`intercept`), `input_schema.json`. The API **loads**
  this pipeline and **never re-fits** it — training and inference apply identical
  transforms.
- **Inference test (notebook §23).** Reload from disk == in-memory (max |Δ| = 0). A
  positive review → `{"prediction":"recommend","p_recommend":0.9376}`; a negative one →
  `{"prediction":"not recommend","p_recommend":0.0026}`.
- **`contributions`** = an exact **linear-SHAP** decomposition (no `shap` library):
  `φⱼ = coefⱼ·(xⱼ − x̄ⱼ)`, and `σ(z_base + Σⱼ φⱼ) = P(recommend)` exactly with
  `z_base = intercept + coef·x̄`. `base_p` is the model's neutral point for an average
  review (class-balanced, so *not* the 84.7% dataset rate). The clients render it as a
  diverging bar chart (red = toward "won't recommend", green = toward "recommends") with
  a waterfall line; labels come from the raw value (`brand_name_skinfix` → "Brand:
  Skinfix", `txt__<tok>` → "review term: `<tok>`").

**Web application** (React + Vite → FastAPI):

```
Framework:  React + Vite (single-page client)  +  FastAPI (POST /predict)
Endpoint:   POST http://<host>:8000/predict
Input:      skin_type / skin_tone / eye_color / hair_color, secondary_category, brand_name,
            price_usd, loves_count, reviews, review_title, review_text     (rating is NOT sent)
Output:     { prediction, confidence, p_recommend, threshold, review_terms, signals,
              contributions, model, representation }
Validation: Pydantic ReviewInput — review_text (≥ 1 char) and price_usd (≥ 0) effectively
            required; everything else optional; missing numerics median-imputed in the pipeline.
```

The client is a **3-step wizard** (Your skin profile → The product → The review) with a
stepper, then a full-screen result: a verdict block, a `P(recommend)` meter with the 50%
cut-off, a "what the model saw" key–value card, review-term chips, and the diverging-bar
contribution chart with a waterfall line.

📸 **W1 — wizard step 1** (skin profile) with the stepper.
📸 **W2 — wizard step 3** (the review) with an example loaded.
📸 **W3 — result screen**: verdict block + `P(recommend)` meter + "what the model saw".
📸 **W4 — result screen**: the SHAP diverging-bar chart + waterfall + review-term chips.
📸 **W5 — FastAPI `/docs`** `POST /predict` "Try it out".

**Figure explanation to write for each:** the input entered, the prediction returned, how
to read the meter, and how the contribution bars sum (with the base rate) to the final
probability.

**Mobile application** (Flutter):

```
Framework:  Flutter
Platform:   Android (emulator / device)
API:        POST http://<host>:8000/predict     (same endpoint as the web app)
Request:    JSON body of the fields above       Response: the JSON above
```

Two screens built from the same design tokens as the web client (light + dark): a 3-step
review form (fields from `GET /questions`, "load a real review" sheet, sticky footer) and
a result screen (verdict block, `P(recommend)` meter, "what the model saw", review-term
chips, contribution chart, "how this works"). `Training ≠ Inference` — the model is
trained once and saved; the app is a REST client.

📸 **M1 — review form, step 1** (skin profile) with the stepper.
📸 **M2 — review form, step 3** (the review) with an example loaded.
📸 **M3 — result screen**: verdict block + `P(recommend)` meter + "what the model saw".
📸 **M4 — result screen** scrolled to the contribution chart + review-term chips.
📸 **M4b — evidence** the app calls the API: a `POST /predict 200` line in the Uvicorn log
while the app is used.

---

## App-3 discussion questions (assignment §14, e-commerce extras)

1. **What information is in customer comments?** The product *experience*: texture/feel
   (~42% of reviews), skin outcome / breakouts (~39%), irritation (~23%), scent (~26%),
   price/value (~13%), and an explicit recommend / repurchase verdict (~21%).
2. **How are comments transformed into numerical data?**
   `review_title + " . " + review_text` → lower-case, strip accents, drop English
   stop-words → word 1–2-grams → **TF-IDF** row (`sublinear_tf`, `min_df=10`,
   ≤ 40 000 features). The embedding view: tokens → integer IDs → rows of a `(|V|+1)×d`
   table → `E ∈ ℝ^{B×T×d}`.
3. **What do token IDs represent?** The position of a token in a fixed vocabulary built
   from the training reviews (0 reserved for PAD). They carry no meaning themselves —
   they are keys into the TF-IDF weights or the embedding table.
4. **What do embedding vectors represent?** A learned `d`-dimensional dense encoding of a
   token where semantically similar tokens sit close together; a review becomes the
   `T × d` stack of its tokens' rows. (The deployed model uses the sparse TF-IDF form
   instead; the embedding form is shown for the representation requirement.)
5. **Which customer interests can be discovered?** Per skin type / tone: which categories
   and brands that segment tends to recommend; which product aspects (texture, scent,
   irritation) drive or sink a recommendation; K-Means segments by recency / spend /
   recommend-rate / category mix (Appendix B).
6. **Does text improve prediction over tabular alone?** **Yes, dramatically on this
   dataset** — tabular ROC-AUC ~0.77–0.81 vs text ~0.965 — but §6.6a shows much of that
   gap is the review being *co-authored* with the recommend tick. Adding text to the
   tabular block gives a small, honest lift (macro-F1 0.856 → 0.861); the leak-free
   deployable signal is the ~0.8 tabular model.

---

## Cells for the shared tables

### Executive Summary — Customer Behaviour row

| Field | Value |
|---|---|
| Dataset | Sephora Products and Skincare Reviews (`nadyinky`), `reviews_500-750.csv` + `product_info.csv` |
| Problem | Predict whether a skincare reviewer recommends the product (`is_recommended`), from their skin profile + the product + the review text |
| Representation | tabular `x_tab ∈ ℝ^{138}` (one-hot + scaled) **‖** TF-IDF `x_txt ∈ ℝ^{~28 600}`; combined `d ≈ 28 700`, sparse |
| Selected model | Logistic Regression on tabular + TF-IDF text — best macro-F1 / minority recall, one small CPU artifact |
| Headline metric | test ROC-AUC **0.964**, macro-F1 **0.858**, recall on "won't recommend" **0.876**, accuracy 0.918 |
| Deployment | FastAPI (`POST /predict`) + React/Vite web + Flutter mobile |

### Mandatory Data-Representation Summary — E-commerce row

| Application | Raw form | Numerical representation | Model input |
|---|---|---|---|
| E-commerce (customer behaviour) | CSV + review comments | tabular one-hot + scaled matrix **+** TF-IDF text vectors | `B × d` = `N × 28 733` sparse (N = 104 313), and/or `B × T × d` = `1 × 40 × 16` for the embedding demo |

**Explain the dimensions:** `N = 104 313` cleaned reviews; `d = 28 733` = 138 tabular
columns after one-hot + `28 595` TF-IDF 1–2-gram features on the training vocabulary;
for the embedding demo `B = 1` review, `T = 40` tokens (padded), `d = 16` embedding width.

### Cross-Application Comparison — Customer Behaviour column

| Aspect | Customer Behaviour |
|---|---|
| Problem type | Binary classification |
| Observation (one row) | one product review (reviewer profile + product join) |
| Target | `recommended` = `is_recommended` (0/1), separate from the 1–5★ rating |
| Input representation | `x_tab ∈ ℝ^{138}` (one-hot + scaled) **‖** TF-IDF `x_txt ∈ ℝ^{~28 600}`; combined `d ≈ 28 700`, sparse |
| Data-quality issues | ~10% blank target dropped; 84.7% class imbalance; 2–17% missing skin-profile fields (kept as `__na__`); single category (Skincare); English-only text |
| Best model | Logistic Regression (tabular + text) |
| Main metric | recall on the "does not recommend" class — **0.876** at ROC-AUC **0.964** |
| Web deployment | Yes (React + Vite 3-step wizard → FastAPI) |
| Mobile deployment | Yes (Flutter, 2 screens) |
| Main limitation | the review text is **co-authored** with the recommend tick, so ~0.96 is partly leakage — the leak-safe tabular signal is ~0.8 (§6.6a); retrospective only; ~78% single-review reviewers so RFM Frequency is near-degenerate; one product category |

### Reproducibility — Customer Behaviour entries

| Item | Value |
|---|---|
| Python / OS | 3.13 / Windows 11 |
| Key libraries | numpy, pandas, **scikit-learn 1.9.0**, scipy, matplotlib (notebook + API); fastapi + uvicorn + pydantic 2 (API); React 18 + Vite 5 (web); Flutter 3.19+ (mobile) |
| Random seed | `RANDOM_SEED = 42` (numpy + `random`; every split and estimator) |
| Dataset source | Kaggle `nadyinky/sephora-products-and-skincare-reviews` (CC0); `data/sephora/reviews_500-750.csv` joined to `product_info.csv` on `product_id` — **not committed** (~505 MB), re-download from Kaggle |
| Rows | 116 262 raw → **104 313** after dropping blank-target / empty-text / duplicate rows; recommend rate 0.8465 |
| Preprocessing | `ColumnTransformer`: median `SimpleImputer` → `log1p` (5 cols) → `StandardScaler`; scale (4 numeric); passthrough (6 binary); `OneHotEncoder(handle_unknown="ignore", min_frequency=25)` (6 categorical); `TfidfVectorizer(ngram_range=(1,2), min_df=10, max_features=40000, sublinear_tf=True, stop_words="english")` on `review_all`. Fitted on train only. |
| Feature representation | 21 tabular columns → `x_tab ∈ ℝ^{138}` after one-hot; `x_txt ∈ ℝ^{~28 600}` sparse; combined `d ≈ 28 700`; `y ∈ {0,1}^N`, `N = 104 313` |
| Train/val/test split | `StratifiedGroupKFold(n_splits=5)` **grouped on `author_id`** (no reviewer spans splits) → train 62 587 · val 20 862 · test 20 864, each at recommend rate 0.8465 |
| Model hyperparameters | `LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=42)` on the tabular + text representation (full 8-model table: notebook §17) |
| Evaluation metrics | test: Acc 0.918 · macro-F1 0.858 · ROC-AUC 0.964 · class-0 recall 0.876 · confusion `[[2804, 398], [1315, 16347]]`; single-representation on the same test set: tab-only 0.801, text-only 0.963 |
| Saved artifacts | `model/model_pipeline.joblib`, `model/feature_names.joblib`, `model/feature_means.joblib`, `model/input_schema.json` |
| Code | `customer_behaviour/api/` (FastAPI), `customer_behaviour/web/` (React + Vite), `customer_behaviour/mobile/` (Flutter), `customer_behaviour/requirements.txt` |
| Reproduce | download the two CSVs into `data/sephora/` → `python -m venv .venv && . .venv/Scripts/activate` → `pip install -r requirements.txt` → execute `notebook/customer_behaviour.ipynb` (writes `model/`) → `python api/make_samples.py` → `python -m uvicorn api.main:app --port 8000` (from `customer_behaviour/`) → `npm --prefix web install && npm --prefix web run dev` → `flutter run` in `mobile/` |
