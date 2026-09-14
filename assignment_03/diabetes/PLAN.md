# diabetes — Application Plan

Self-contained plan for `diabetes/notebook/diabetes.ipynb`: dataset, representation,
model choices, DL architecture, and the literal cell-by-cell layout. See the root
`../PLAN.md` for how this app's results feed into the assignment_03 report.

## Conventions (same across all 3 apps in this assignment)
- `RANDOM_SEED = 42` everywhere: splits, shuffles, estimators.
- Train/test split: 80/20, computed once, reused for both the classical models and the
  DL model, so the 4-way comparison is apples-to-apples.
- Standardization is **fit on the training split only**, then applied to test — no
  leakage. Same rule assignment_02 enforced, and exactly what
  `materials/int_sys_dev_slide_03_basicML_deepLearning_04.09.pdf` does
  (`mean = X_train.mean(axis=0)`, reused for `X_test`).
- DL weight init: `np.random.randn(fan_in, fan_out) * sqrt(2/fan_in)` (He-style),
  biases `np.zeros((1, fan_out))` — per the slides.
- DL training loop records loss **and** accuracy every epoch into plain lists, for a
  paired loss/accuracy training-curve plot (not loss alone) — visualizations are not
  confined to the EDA section in this assignment; each of the classical-model,
  DL-model, and comparison sections below carries its own plots, not just tables.
- **"Write each part into different cells" (verbal requirement)**: one function, one
  plot, one model fit, one explanation, per cell. No cell mixes "define a function"
  with "call it and print a result." The cell plan below is literal, not a guideline.
- Ends with persisted artifacts in `model/`: `model_pipeline.joblib`,
  `dl_weights.npz`, `feature_names.joblib`, `input_schema.json` — matching
  assignment_02's contract even though nothing loads them this round (no API layer).

## Problem
Binary classification: does this BRFSS respondent have diabetes? Same task as
assignment_02, and the lecture's Application-1 framing (`patient data → learned
representation → P(diabetes)`).

## Dataset
`DATASET_2019.csv` + `DATASET_2020.csv` from
`spandanjit2005/brfss-diabetes-indicator-dataset` (already in `data/`). Concatenate the
two years (identical 33-column schema; a `year` column already present distinguishes
provenance).

- Rows: 416,661 (2019) + 399,050 (2020) = **815,711**, vs. assignment_02's 253,680
  (2015 only) — **3.2x bigger**, ~210MB combined, well under the gigabyte ceiling.
- Why 2 years and not more/fewer: keeps the growth step moderate (2-4x per the agreed
  pace across this assignment series) rather than jumping straight to the full 20-year
  (~8M row) file, leaving headroom for a later assignment to grow into without a new
  dataset search.
- Materials never mandate a specific diabetes dataset for assignment_03 beyond the
  tutorial PDF's toy 768-row/8-feature Pima example, which is a generic illustration,
  not a requirement — `REQUIREMENT.md` explicitly allows keeping "the assignment_02
  BRFSS feature set," so this dataset is the materials-silent default: same
  problem/representation family as assignment_02, just bigger.

## Target
The source's diagnosis field is 3-class (non-diabetic / pre-diabetic / diabetic).
Collapse to binary — `0` = non-diabetic, `1` = pre-diabetic or diabetic — to match
assignment_02's `Diabetes_binary` and the lecture's binary framing (`ŷ ≥ 0.5 ⇒
diabetes`). Continuation of the same problem type, not a new one.

## Representation — what and why
- Prefer the `_00`/`_01`-suffixed ordinally-encoded columns over their raw categorical
  counterparts (the source repo documents these as pre-encoded convenience columns) —
  avoids a manual re-encoding pass and keeps every input already numeric.
- Continuous columns (BMI, weight, height, days-of-poor-health counts):
  `StandardScaler` fit on train only.
- Binary/ordinal survey fields (high blood pressure, high cholesterol, smoker,
  general-health 1-5, etc.): pass through as-is — already small-integer codes, not
  unordered categories, so one-hot would be wrong (same rule assignment_02 used for
  BRFSS: "no one-hot" on this kind of column).
- `year` (2019 vs. 2020): keep as a candidate feature, decide empirically in the EDA
  cell whether it carries signal or just adds noise; document the decision either way
  in a markdown cell — don't silently drop it without saying so.
- Resulting `d ≈ 33` (minus target and any dropped raw-categorical duplicates).

## 3 classical ML models — which, and why
| Model | Why |
|---|---|
| Logistic Regression | linear baseline, cheap, interpretable coefficients |
| **Random Forest** | the model assignment_02 deployed for this app — continuity, required by `REQUIREMENT.md` |
| Gradient Boosting Classifier | boosted-tree ensemble — tests whether extra capacity over bagging helps on this feature set |

## From-scratch DL model
Architecture: `33 → 32 → 16 → 1`, ReLU hidden layers, sigmoid output, binary
cross-entropy loss.

- Why these widths: the tutorial's `8 → 16 → 8 → 1` roughly doubles-then-halves the
  8-feature input; scaling that pattern to `d≈33` gives `33 → 32 → 16 → 1` (first
  hidden layer ≈ input width, second layer half that) — same shape philosophy, sized
  for the wider BRFSS input.
- This is the one app where the from-scratch implementation can be checked directly
  against the tutorial PDF's derivation line-by-line (same sigmoid+BCE math, same
  He-init formula, same forward/backward structure) — only `d` and `N` differ. Worth
  calling out in the report as the strongest evidence the implementation matches the
  lecture's math, not just "a" NumPy network.

## Notebook cell-by-cell plan
One logical step per cell, per the "different cells" rule.

**Setup & data**
1. Markdown: title, problem statement, link to this plan's reasoning.
2. Imports (`numpy`, `pandas`, `sklearn`, `matplotlib`/`seaborn`) — explicit comment:
   no `tensorflow`/`torch`/`keras`.
3. `RANDOM_SEED = 42` (numpy + `random`).
4. Load `DATASET_2019.csv` and `DATASET_2020.csv`, concatenate with a `year` column.
5. Markdown: "Data understanding."
6. `.shape`, `.dtypes`, `.head()`, `.describe()`.
7. Missing-value audit (`.isna().sum()`).
8. Target class-balance plot + markdown Observation/Interpretation.
9. EDA plot 2 (e.g. BMI vs. target) + markdown Observation/Interpretation.
10. EDA plot 3 (e.g. general-health score vs. target) + markdown
    Observation/Interpretation.
11. EDA plot 4 (e.g. `year` vs. target rate, to inform the year-as-feature decision)
    + markdown Observation/Interpretation.
12. EDA plot 5: correlation heatmap of the numeric/ordinal feature block + markdown
    Observation/Interpretation.

**Cleaning & representation**
13. Markdown: "Data cleaning."
14. Collapse 3-class target to binary (one cell, this operation only).
15. Drop raw-categorical duplicate columns, keep `_00`/`_01` ordinal versions (one
    cell, this operation only).
16. Markdown: "Representation" + why (restates the reasoning above for the reader).
17. Train/test split (80/20, `RANDOM_SEED`).
18. `StandardScaler` fit-on-train / transform-both for continuous columns.
19. Shape sanity-check cell (`X_train.shape`, `X_test.shape`).

**3 classical models**
20. Markdown: "Classical machine-learning models."
21. Logistic Regression: instantiate + `.fit()`.
22. Random Forest: instantiate + `.fit()`.
23. Gradient Boosting: instantiate + `.fit()`.
24. Predictions cell: all 3 models on the test set.
25. Metrics cell: Accuracy/Precision/Recall/F1 + confusion matrix per model, one
    results table.
26. **Plot**: confusion-matrix heatmaps for the 3 classical models, 1x3 subplot grid.
27. **Plot**: ROC curve overlay, all 3 classical models on one axes (AUC in the
    legend).
28. **Plot**: feature-importance bar chart from Random Forest (`feature_importances_`,
    top 15) — which BRFSS fields the tree ensemble actually leaned on.

**From-scratch DL model**
29. Markdown: "Deep learning from scratch (NumPy only)" + architecture statement + why
    (from this plan).
30. `relu`, `relu_derivative`.
31. `sigmoid`, `sigmoid_derivative`.
32. `binary_cross_entropy` loss function.
33. Weight init: `W1,b1,W2,b2,W3,b3`, He-init, seeded, sized `33→32→16→1`.
34. `forward(X)` function.
35. `backward(y, cache)` function.
36. Training loop: forward → loss → backward → update, `loss_history` **and**
    `accuracy_history` per epoch.
37. Predict on test set, threshold at 0.5.
38. DL metrics cell (same structure as cell 25).
39. **Plot**: loss & accuracy training curves, two subplots sharing the epoch axis —
    the required loss-curve visualization, paired with accuracy rather than left
    alone.
40. **Plot**: confusion-matrix heatmap (DL model) — the required DL confusion-matrix
    visualization.
41. **Plot**: histogram of learned first-layer weights (`W1.flatten()`) — sanity check
    on the weight distribution after training (dead/exploded units).
42. **Plot**: feature-signal bar chart from the DL model (mean `|W1|` per input
    feature, top 15) — the DL model's analogue to cell 28's Random Forest importance
    chart, so the report can compare *which features each model leaned on*.

**Comparison & persistence**
43. Markdown: "4-model comparison."
44. Combined results table (3 classical + DL) as one `DataFrame`.
45. Bar chart: Accuracy across all 4 models.
46. Bar chart: Precision/Recall/F1 across all 4 models.
47. **Plot**: ROC curve overlay, all 4 models on one axes (extends cell 27 with the DL
    model's curve).
48. Markdown: written comparison — which model wins and why (capacity vs. overfitting
    vs. dataset size), in the lecture's "Function Composition + Representation
    Learning + Optimization" framing.
49. Save classical model + feature names + `input_schema.json`.
50. Save DL weights (`np.savez`).
51. Reload-and-verify cell: reload both artifacts, confirm predictions match the
    in-memory versions (same discipline as assignment_02 §23, cheap insurance).

## Open questions
- Keep `year` as a feature or drop it — deferred to cell 11, decided empirically,
  documented either way.
- Exact hyperparameters (tree depth/count, DL epoch count/learning rate) are left to
  the notebook stage — this plan fixes *which* models/architecture and *why*, not
  tuning, since tuning is data-dependent.
