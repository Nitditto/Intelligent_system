# customer_behaviour — Application Plan

Self-contained plan for `customer_behaviour/notebook/customer_behaviour.ipynb`:
dataset, representation, model choices, DL architecture, and the literal cell-by-cell
layout. See the root `../PLAN.md` for how this app's results feed into the
assignment_03 report.

## Conventions (same across all 3 apps in this assignment)
- `RANDOM_SEED = 42` everywhere: splits, shuffles, estimators.
- Train/test split: 80/20, computed once, reused for both the classical models and the
  DL model, so the 4-way comparison is apples-to-apples.
- Standardization/vectorization is **fit on the training split only**, then applied to
  test — no leakage. Same rule assignment_02 enforced.
- DL weight init: `np.random.randn(fan_in, fan_out) * sqrt(2/fan_in)` (He-style),
  biases `np.zeros((1, fan_out))` — per the slides.
- DL training loop records loss every epoch into a plain list, for the required
  loss-curve plot.
- **"Write each part into different cells" (verbal requirement)**: one function, one
  plot, one model fit, one explanation, per cell. The cell plan below is literal, not
  a guideline.
- Ends with persisted artifacts in `model/`: `model_pipeline.joblib`,
  `dl_weights.npz`, `feature_names.joblib`, `input_schema.json` — matching
  assignment_02's contract even though nothing loads them this round (no API layer).

## Problem
Binary classification: is this review a "recommend" (derived label) or not? Same task
shape as assignment_02's `is_recommended`, and the lecture's Application-3 framing
(`text → learned representation → sentiment/intent`).

## Dataset
`flipkart_reviews.csv` from `niraliivaghani/flipkart-dataset` (already in `data/`).

- Rows: **364,401**, vs. assignment_02's 116,264 (Sephora reviews) — **3.1x bigger**,
  56MB, well under the gigabyte ceiling.
- Deliberately a **different e-commerce platform** (Flipkart, general merchandise —
  75+ product categories) rather than pulling more Sephora review chunks. Reasoning:
  `REQUIREMENT.md` doesn't say whether reusing the same source dataset counts toward
  "bigger" — using a genuinely different platform removes the ambiguity entirely
  rather than relying on an interpretation. It doesn't need to be skincare-specific;
  the prior app's skincare focus was incidental to Sephora being the source, not a
  requirement of the app itself.
- Columns: `Product_name, Price, Rate, Review, Summary`.
- No DL tutorial or dataset mandate exists for customer_behaviour anywhere in the
  materials — this is the materials-silent default: same recommend/not text-plus-
  tabular problem framing as assignment_02, just bigger and on a different platform.

## Target
Derive `is_recommended = (Rate >= 4)` from the 1-5 `Rate` column. Mirrors
assignment_02's binary recommend/not framing and matches `REQUIREMENT.md`'s explicit
allowance ("binary classification, e.g. recommend/not, or sentiment").

## Representation — what and why
- `Review` text → TF-IDF, 1-2 gram. Two vocabulary sizes, documented explicitly:
  - Classical models: full/larger vocabulary (sklearn's sparse linear models handle
    this cheaply even at 364K rows).
  - DL model: capped at `max_features=5000` (a dense-matrix NumPy forward/backward
    pass at full TF-IDF width would be far too slow/large without a framework's sparse
    autodiff — capping is a deliberate, stated engineering trade-off, not an
    oversight).
  - Optionally concatenate `Summary` into the same TF-IDF input (short text, likely
    carries similar signal to `Review` but more concise) — try both, keep whichever
    helps in EDA/validation, document the choice.
- `Price`: raw values are corrupted by an encoding issue (confirmed while inspecting
  the download — the ₹ symbol got mis-decoded into stray characters, e.g. values that
  render as `"?10,499"`). Clean with a regex that strips everything except digits,
  commas, and decimal points, then parse to a numeric rupee amount. `StandardScaler`
  fit on train only.
- `Product_name`: free text, not a clean category — dropped for this pass (not
  required by `REQUIREMENT.md`, which asks for TF-IDF over *review* text specifically;
  noted as a possible extension, not pursued to keep scope matched to the
  requirement).
- Resulting `d` = `max_features` (5,000 for the DL model) + 1 tabular column
  (`Price`).

## 3 classical ML models — which, and why
| Model | Why |
|---|---|
| **Logistic Regression** | the model assignment_02 deployed for this app — continuity, required by `REQUIREMENT.md` |
| Linear SVM (`LinearSVC` or `SGDClassifier` with hinge loss) | the classic strong performer on high-dimensional sparse TF-IDF; scales to 364K rows cheaply |
| Multinomial Naive Bayes | classic cheap text baseline, trains near-instantly even at this row count and vocabulary size — a useful lower bound to compare the other two against |

Trees (Random Forest/Gradient Boosting) are deliberately **not** among the 3 here:
they're a poor fit for very high-dimensional sparse TF-IDF input (each split only
looks at one of thousands of near-orthogonal columns), so the 3 chosen models cover 3
genuinely different inductive biases suited to this representation, rather than 3
variations that would all struggle the same way.

## From-scratch DL model
Architecture: `5000 → 128 → 32 → 1`, ReLU hidden layers, sigmoid output, binary
cross-entropy loss.

- Why the widest first hidden layer of the three apps: TF-IDF input is sparse and
  high-dimensional (5,000 columns, mostly zero per row) — more first-layer capacity is
  needed to compress that down usefully, mirroring the lecture's own narrowing example
  ("10,000 → 512 → 128 → 32 → ...").
- Practical note to carry into the notebook: `X_train` from `TfidfVectorizer` is a
  scipy sparse matrix; the from-scratch `forward`/`backward` functions need
  `X_train.toarray()` (or equivalent) before matrix multiplication, since plain NumPy
  doesn't operate on sparse matrices — state this explicitly in a markdown cell so
  it's a documented decision, not a silent conversion.

## Notebook cell-by-cell plan

**Setup & data**
1. Markdown: title, problem statement, link to this plan's reasoning.
2. Imports — explicit comment: no `tensorflow`/`torch`/`keras`.
3. `RANDOM_SEED = 42`.
4. Load `flipkart_reviews.csv`.
5. Markdown: "Data understanding."
6. `.shape`, `.dtypes`, `.head()`, `.describe()`.
7. Missing-value audit.
8. `Rate` distribution plot (1-5) + derived `is_recommended` class balance + markdown
   Observation/Interpretation.
9. EDA plot 2 (e.g. review length vs. `Rate`) + markdown Observation/Interpretation.
10. EDA plot 3 (e.g. `Price` distribution, before/after cleaning) + markdown
    Observation/Interpretation.
11. EDA plot 4 (e.g. top product categories by review count, from `Product_name`) +
    markdown Observation/Interpretation.

**Cleaning & representation**
12. Markdown: "Data cleaning."
13. Clean `Price` (regex strip + numeric parse) — one cell, this operation only.
14. Derive `is_recommended = (Rate >= 4)` — one cell, this operation only.
15. Markdown: "Representation" + why (restates reasoning above).
16. Train/test split (80/20, `RANDOM_SEED`, stratified on `is_recommended`).
17. `StandardScaler` fit-on-train / transform-both for `Price`.
18. TF-IDF vectorizer, full vocabulary, fit on train `Review` only — for the classical
    models.
19. TF-IDF vectorizer, `max_features=5000`, fit on train `Review` only — for the DL
    model (separate cell from 18, since it's a separate representation with a stated,
    different reason).
20. Shape sanity-check cell (both TF-IDF variants + tabular).

**3 classical models**
21. Markdown: "Classical machine-learning models."
22. Logistic Regression: instantiate + `.fit()` (full-vocab TF-IDF ‖ Price).
23. Linear SVM: instantiate + `.fit()`.
24. Multinomial Naive Bayes: instantiate + `.fit()` (text-only input — Naive Bayes
    assumes non-negative counts, so fit on the TF-IDF text block alone, noted in
    markdown as a deliberate scope difference from the other two models).
25. Predictions cell: all 3 models on the test set.
26. Metrics cell: Accuracy/Precision/Recall/F1 + confusion matrix per model, one
    results table.

**From-scratch DL model**
27. Markdown: "Deep learning from scratch (NumPy only)" + architecture statement + why
    (from this plan) + the sparse-to-dense conversion note.
28. `relu`, `relu_derivative`.
29. `sigmoid`, `sigmoid_derivative`.
30. `binary_cross_entropy` loss function.
31. Weight init: `W1,b1,W2,b2,W3,b3`, He-init, seeded, sized `5000→128→32→1`.
32. `forward(X)` function.
33. `backward(y, cache)` function.
34. Training loop: forward → loss → backward → update, `loss_history` per epoch
    (using the 5,000-feature TF-IDF ‖ Price input).
35. Predict on test set, threshold at 0.5.
36. DL metrics cell (same structure as cell 26).

**Comparison & persistence**
37. Markdown: "4-model comparison."
38. Combined results table (3 classical + DL) as one `DataFrame`.
39. Bar chart: Accuracy across all 4 models.
40. Bar chart: Precision/Recall/F1 across all 4 models.
41. Loss curve: epoch vs. training loss (DL model).
42. Confusion-matrix heatmap (DL model).
43. Markdown: written comparison — which model wins and why (capacity vs. overfitting
    vs. dataset size vs. vocabulary-size handicap on the DL model specifically), in
    the lecture's framing.
44. Save classical model + feature names + `input_schema.json`.
45. Save DL weights (`np.savez`).
46. Reload-and-verify cell: reload both artifacts, confirm predictions match the
    in-memory versions.

## Open questions
- Concatenate `Summary` into the TF-IDF input or keep `Review` only — deferred to
  cells 18-19, decided from validation results.
- Exact hyperparameters (regularization strength, DL epoch count/learning rate) are
  left to the notebook stage — this plan fixes *which* models/architecture and *why*,
  not tuning, since tuning is data-dependent.
