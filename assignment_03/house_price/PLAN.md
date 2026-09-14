# house_price — Application Plan

Self-contained plan for `house_price/notebook/house_price.ipynb`: dataset,
representation, model choices, DL architecture, and the literal cell-by-cell layout.
See the root `../PLAN.md` for how this app's results feed into the assignment_03
report.

## Conventions (same across all 3 apps in this assignment)
- `RANDOM_SEED = 42` everywhere: splits, shuffles, estimators.
- Train/test split: 80/20, computed once, reused for both the classical models and the
  DL model, so the 4-way comparison is apples-to-apples.
- Standardization/encoding is **fit on the training split only**, then applied to
  test — no leakage. Same rule assignment_02 enforced.
- DL weight init: `np.random.randn(fan_in, fan_out) * sqrt(2/fan_in)` (He-style),
  biases `np.zeros((1, fan_out))` — per the slides.
- DL training loop records loss **and** MAE every epoch into plain lists, for a
  paired loss/MAE training-curve plot (not loss alone) — visualizations are not
  confined to the EDA section in this assignment; each of the classical-model,
  DL-model, and comparison sections below carries its own plots, not just tables.
- **"Write each part into different cells" (verbal requirement)**: one function, one
  plot, one model fit, one explanation, per cell. The cell plan below is literal, not
  a guideline.
- Ends with persisted artifacts in `model/`: `model_pipeline.joblib`,
  `dl_weights.npz`, `feature_names.joblib`, `input_schema.json` — matching
  assignment_02's contract even though nothing loads them this round (no API layer).

## Problem
Regression: predict `price` (USD) from listing attributes. Same task shape as
assignment_02, and the lecture's Application-2 framing (`house attributes → learned
representation → price`).

## Dataset
`realtor-data-sample.csv` from `ahmedshahriarsakib/usa-real-estate-dataset`, a
600,000-row seeded (`RANDOM_SEED = 42`) sample of the source's 2,226,382 rows (already
in `data/`).

- Rows: **600,000**, vs. assignment_02's 238,924 (VN real estate) — **2.5x bigger**,
  47MB, well under the gigabyte ceiling.
- Why sampled rather than the full 2.2M: keeps the growth step moderate (2-4x per the
  agreed pace across this assignment series) and leaves the remaining ~1.6M rows of
  the source available for a later assignment to grow into, without a new dataset
  search.
- Columns: `brokered_by, status, price, bed, bath, acre_lot, street, city, state,
  zip_code, house_size, prev_sold_date`.
- No DL tutorial or dataset mandate exists for house_price anywhere in the materials —
  this is the materials-silent default: same regression problem framing as
  assignment_02, just a bigger, genuinely US-market dataset.

## Target
`price` (USD). Train on `log1p(price)` (assignment_02 did the same for VND prices —
sale prices are heavy-tailed; log-space training is standard practice and keeps this
notebook consistent with the prior one). Invert with `expm1` when reporting metrics in
original-dollar terms.

## Representation — what and why
- Drop `brokered_by` and `street`: both are already anonymized/categorically-encoded
  IDs in the source (the dataset's own documentation flags this as a privacy measure) —
  high cardinality, no generalizable signal.
- `status` (`for_sale` / `ready_to_build`): one-hot, 2 categories.
- `state`: one-hot (≤50 categories, manageable at this row count).
- `city`, `zip_code`: too high-cardinality for one-hot at 600K rows — either drop, or
  frequency-encode (replace each value with how often it appears in the training set).
  Decide and document in the representation cell; frequency-encoding is preferred if
  city-level signal proves useful in EDA, since dropping both `city` and `zip_code`
  entirely would throw away all location granularity below state level.
- `bed`, `bath`, `acre_lot`, `house_size`: numeric; median-impute missing values, then
  `StandardScaler` (fit on train only).
- `prev_sold_date`: engineer "days since previous sale" (numeric) if EDA shows it
  correlates with price, else drop. Document the decision either way — a genuine
  feature-engineering call, not a given.
- Resulting `d ≈ 15-20` depending on the city/zip_code decision.

## 3 classical ML models — which, and why
| Model | Why |
|---|---|
| Linear Regression | linear baseline on the encoded/scaled feature matrix |
| **Random Forest** | the model assignment_02 deployed for this app — continuity, required by `REQUIREMENT.md` |
| Gradient Boosting Regressor | boosted-tree ensemble — tests whether extra capacity over bagging helps here, especially relevant since assignment_02's Random Forest had a weak R² (0.186) on VN data; worth seeing if a different ensemble or the larger US dataset changes that |

## From-scratch DL model
Architecture: `d → 64 → 32 → 1` (d ≈ 15-20), ReLU hidden layers, **linear** output (no
activation — regression, not classification), MSE loss.

- Why wider hidden layers than diabetes's: house_price's `d` is smaller (~15-20 vs
  ~33) but the target (log-price) is continuous and the relationship to bed/bath/size/
  location is plausibly more nonlinear than diabetes's more linear survey-style
  predictors — more width gives the network more room to compose nonlinear
  interactions (e.g. `size × location` effects) before collapsing to one output.
- Explicitly note in the notebook (markdown cell) that the output layer has **no**
  sigmoid — the one place the diabetes tutorial's math must be deliberately changed,
  not copied: `ŷ = H2·W3 + b3` directly, matching the lecture's Application-2 slide
  ("Deep learning: ŷ = w^T h + b" for regression, no squashing function).

## Notebook cell-by-cell plan

**Setup & data**
1. Markdown: title, problem statement, link to this plan's reasoning.
2. Imports — explicit comment: no `tensorflow`/`torch`/`keras`.
3. `RANDOM_SEED = 42`.
4. Load `realtor-data-sample.csv`.
5. Markdown: "Data understanding."
6. `.shape`, `.dtypes`, `.head()`, `.describe()`.
7. Missing-value audit.
8. Target (`price`) distribution histogram, raw and log1p, side by side + markdown
   Observation/Interpretation (this is what justifies training in log-space).
9. EDA plot 2 (e.g. `house_size` vs. `price`) + markdown Observation/Interpretation.
10. EDA plot 3 (e.g. `state` vs. median `price`) + markdown Observation/Interpretation.
11. EDA plot 4 (e.g. `bed`/`bath` vs. `price`) + markdown Observation/Interpretation.
12. EDA plot 5: correlation heatmap of numeric features (`bed`, `bath`, `acre_lot`,
    `house_size`, `price`) + markdown Observation/Interpretation.

**Cleaning & representation**
13. Markdown: "Data cleaning."
14. Drop `brokered_by`, `street` (one cell, this operation only).
15. Median-impute missing numeric values (one cell, this operation only).
16. Engineer or drop `prev_sold_date` per the EDA finding (one cell, this operation
    only, with a one-line comment stating which way it went and why).
17. Markdown: "Representation" + why (restates reasoning above).
18. Train/test split (80/20, `RANDOM_SEED`).
19. One-hot encode `status`, `state` (fit categories on train only).
20. City/zip_code frequency-encoding or drop, per the representation decision.
21. `StandardScaler` fit-on-train / transform-both for numeric columns.
22. `log1p` the target for both splits.
23. Shape sanity-check cell.

**3 classical models**
24. Markdown: "Classical machine-learning models."
25. Linear Regression: instantiate + `.fit()` (on log-target).
26. Random Forest: instantiate + `.fit()`.
27. Gradient Boosting Regressor: instantiate + `.fit()`.
28. Predictions cell: all 3 models on the test set, `expm1`-inverted back to dollars.
29. Metrics cell: RMSE/MAE/R² per model (in dollar terms), one results table.
30. **Plot**: predicted-vs-actual scatter grid, 1x3 subplots (3 classical models),
    dollar axes, y=x reference line.
31. **Plot**: residual plot grid, 1x3 subplots (residual = actual - predicted, vs.
    predicted; 3 classical models) — checks for heteroscedasticity/bias by model.
32. **Plot**: feature-importance bar chart — Random Forest / Gradient Boosting
    `feature_importances_` (top 15) alongside Linear Regression's top-15
    coefficients, as two panels in one figure.

**From-scratch DL model**
33. Markdown: "Deep learning from scratch (NumPy only)" + architecture statement + why
    (from this plan) + the no-sigmoid-output note.
34. `relu`, `relu_derivative`.
35. `mse_loss` function.
36. Weight init: `W1,b1,W2,b2,W3,b3`, He-init, seeded, sized `d→64→32→1`.
37. `forward(X)` function (linear output layer, no activation).
38. `backward(y, cache)` function (MSE gradient, not BCE's `ŷ-y` shortcut — derive and
    state the difference explicitly in a markdown cell, since it's not the same
    formula as the classification case).
39. Training loop: forward → loss → backward → update, `loss_history` **and**
    `mae_history` per epoch.
40. Predict on test set, `expm1`-invert to dollars.
41. DL metrics cell (same structure as cell 29).
42. **Plot**: loss & MAE training curves, two subplots sharing the epoch axis — the
    required loss-curve visualization, paired with MAE rather than left alone.
43. **Plot**: predicted-vs-actual scatter (DL model), same dollar-axis/y=x convention
    as cell 30, for direct visual comparison.
44. **Plot**: histogram of learned first-layer weights (`W1.flatten()`) — sanity check
    on the weight distribution after training (dead/exploded units).
45. **Plot**: feature-signal bar chart from the DL model (mean `|W1|` per input
    feature, top 15) — the DL model's analogue to cell 32, so the report can compare
    *which features each model leaned on*.

**Comparison & persistence**
46. Markdown: "4-model comparison."
47. Combined results table (3 classical + DL) as one `DataFrame`.
48. Bar chart: R² across all 4 models.
49. Bar chart: RMSE/MAE across all 4 models.
50. **Plot**: predicted-vs-actual small multiples, 2x2 grid, all 4 models on a shared
    dollar-axis scale — the single figure that ties the whole comparison together
    (extends cells 30/43).
51. Markdown: written comparison — which model wins and why (capacity vs. overfitting
    vs. dataset size), in the lecture's framing. No confusion matrix here (regression).
52. Save classical model + feature names + `input_schema.json`.
53. Save DL weights (`np.savez`).
54. Reload-and-verify cell: reload both artifacts, confirm predictions match the
    in-memory versions.

## Open questions
- `prev_sold_date` → recency feature, or drop — deferred to cell 16, decided from EDA.
- `city`/`zip_code` → frequency-encode or drop — deferred to cell 20, decided from EDA.
- Exact hyperparameters (tree depth/count, DL epoch count/learning rate) are left to
  the notebook stage — this plan fixes *which* models/architecture and *why*, not
  tuning, since tuning is data-dependent.
