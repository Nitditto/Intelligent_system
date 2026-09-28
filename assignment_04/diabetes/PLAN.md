# diabetes — Application Plan

Self-contained plan for `diabetes/notebook/diabetes.ipynb`: dataset, representation,
architecture, and the literal cell-by-cell layout. See the root `../PLAN.md` for how
this app's results feed into the assignment_04 report, and `../REQUIREMENT.md` for the
assignment-wide rules this plan follows, including the "Notebook writing style" rules
that govern every markdown cell the notebook itself contains (natural, reasoning-first
prose, no em dashes, short sentences) — not repeated here, but binding on every
markdown cell listed in this plan's cell-by-cell section below.

## Conventions (same across all 3 apps in this assignment)
- `RANDOM_SEED = 42` everywhere: split, shuffle, weight init.
- Train/test split: 80/20, computed **once**, reused identically by all 3
  implementations (scratch / Keras / PyTorch) — required for the fair comparison rule
  (same dataset + same split + same architecture + comparable hyperparameters).
- Standardization fit on the training split only, applied to test — no leakage.
- Weight init: He-style (`np.random.randn(fan_in, fan_out) * sqrt(2/fan_in)`, zero
  biases) for the scratch model; Keras/PyTorch use their own default init (also
  He/Kaiming-family for ReLU networks) — document the default each framework actually
  uses rather than assuming they match exactly, since that's itself part of what the
  report's comparison should surface.
- One function, one plot, one model step, one explanation, per cell — no cell mixes
  "define" with "call and print."
- Ends with persisted artifacts in `model/`: `scratch_weights.npz`,
  `keras_model.keras`, `pytorch_model.pt`, `feature_names.joblib`,
  `input_schema.json`.

## Problem
Binary classification: does this BRFSS respondent have diabetes? Tabular data, no
spatial structure between features — per the lecture's own reasoning (materials slide
20: "There is no natural convolution operation" for this kind of data), the shared
architecture across all 3 implementations is an **MLP**, not a CNN.

## Dataset
`spandanjit2005/brfss-diabetes-indicator-dataset` (Kaggle), one CSV per year, a
`year` column distinguishing provenance. Concatenate the per-year files listed below.

- Years: `2016, 2017, 2018, 2019, 2020` — five per-year CSVs, already downloaded to
  `diabetes/data/`. Confirmed real row counts (via `wc -l`, minus header): 2016
  483,230; 2017 447,952; 2018 434,232; 2019 416,661; 2020 399,050. Combined:
  **2,181,125 rows**, 2.67x assignment_03's 815,711 (2019+2020 only), inside the
  1.6M-2.4M target band with room either side. A 6th year (2015, 439,881 rows) was
  downloaded too but deliberately dropped: 6 years landed at 2,621,006 rows (3.2x,
  over the target band) and ~700MB combined; 5 years keeps the row count centered in
  the target band and cuts total CSV size to ~557MB. These are the confirmed raw
  counts before any cleaning (duplicate/null drops) — the notebook's own `.shape`
  cell after concatenation should still be treated as the source of truth, but no
  further adjustment to the year list is expected to be necessary.
- No dataset-size comparison language anywhere in this notebook's markdown cells —
  keep any such reasoning out of what ends up in the report.
- **Do not assume the 5 years share an identical schema.** A prior data-prep pass on
  the 2019+2020 pair from this same source found that `high_bp_00` and
  `high_cholesterol_00` exist only in the 2019 file, not 2020, and were silently
  dropped on concat. With 3 more years added, more such year-specific columns are
  plausible. Check `set(df.columns)` per year explicitly in its own cell before
  concatenating, print any columns that aren't common to all 5, and either drop them
  (documented, with a one-line reason: not available across the full date range) or
  keep them only if there's a good reason to accept the resulting missingness for the
  years that lack them. Concatenate on the **intersection** of columns, not on
  whatever `pd.concat` happens to produce by default.
- **Expect BRFSS sentinel/placeholder values in raw numeric fields** (e.g. `999` for
  "don't know" on a weight field, implausible outliers like a BMI over 100). These
  are a known BRFSS survey-coding convention, not data-quality noise to leave as-is.
  Audit the actual value distribution of every continuous column before scaling it
  (a `.describe()` and a histogram are enough to catch this), and clean sentinel
  values to `NaN` (then impute, e.g. median) rather than letting them sit in a
  `StandardScaler` fit and distort the scale for every real observation.

## Target
Source diagnosis field is 3-class (non-diabetic / pre-diabetic / diabetic). Collapse
to binary: `0` = non-diabetic, `1` = pre-diabetic or diabetic.

## Representation — what and why
- Prefer the `_00`/`_01`-suffixed ordinally-encoded columns over their raw categorical
  counterparts (pre-encoded convenience columns documented by the source repo) — keeps
  every input numeric without a manual re-encoding pass.
- Continuous columns (BMI, weight, height, days-of-poor-health counts):
  `StandardScaler` (or an equivalent computed manually) fit on train only.
- Binary/ordinal survey fields (high blood pressure, high cholesterol, smoker,
  general-health 1-5, etc.): pass through as small-integer codes, not one-hot — they
  are ordered, not unordered categories.
- `year` (now 5 distinct values instead of 2): decide empirically in the EDA cell
  whether it carries signal (e.g. a real prevalence trend across 2016-2020) or is
  noise for a 5-way spread; document the decision either way. **If kept, it must not
  enter the feature matrix as a raw calendar year** (`2016`-`2020`) — a prior
  from-scratch MLP on this same data family collapsed to majority-class prediction
  because a raw `2019`/`2020` year value, two-to-three orders of magnitude larger
  than the surrounding standardized/small-integer features, dominated the first
  layer's weighted sum and overflowed the sigmoid on the very first epoch. Either
  re-encode `year` to a small `0-4` offset (`year - 2016`) and fold it into the same
  `StandardScaler` group as the other continuous columns, or one-hot it (5 categories
  is a reasonable one-hot width, unlike the ordinal survey fields this plan otherwise
  avoids one-hotting). Whichever is chosen, verify it directly: print the actual
  min/max of every column in the final feature matrix right before training and
  confirm nothing is left on a wildly different scale than the rest.
- Resulting `d ≈ 33` (minus target and any dropped raw-categorical duplicates) —
  confirm the exact value once the representation cell runs, and use that confirmed
  `d` (not an assumed one) for the architecture below.
- **Class balance**: BRFSS diabetes prevalence is known to be imbalanced (roughly
  85/15 non-diabetic/diabetic in this same data family previously). Check the actual
  split in the class-balance EDA cell, and if it's meaningfully skewed, apply
  balanced class weighting consistently across all 3 implementations from the start
  rather than discovering a collapsed-to-majority-class model after training: pass
  `class_weight="balanced"`-equivalent sample weights into the scratch loss (a
  `weights` argument multiplying the per-row BCE term before the mean, computed via
  the standard `n_total / (n_classes * n_in_class)` formula), into Keras's `fit(...,
  class_weight={0: w0, 1: w1})`, and into PyTorch's loss via `pos_weight` on
  `BCEWithLogitsLoss` or per-sample weighting. This is a fairness-rule requirement,
  not an optional extra: an imbalance fix applied to only one implementation would
  make its accuracy/F1 numbers incomparable to the other two.

## Shared architecture (built 3 times: scratch, Keras, PyTorch)
`d → 32 → 16 → 1`, ReLU hidden layers, sigmoid output, binary cross-entropy loss.

- Why these widths: first hidden layer ≈ input width, second hidden layer roughly
  half that — the same shape philosophy as a standard MLP baseline for a ~33-feature
  tabular input, kept identical across all 3 implementations so differences in
  results trace back to *implementation*, not architecture.
- Training: mini-batch gradient descent, same batch size and epoch count across all
  3 implementations (fix concrete values once the confirmed row count is known —
  large `N` means relatively few epochs are needed to see many gradient updates).

### Scratch (NumPy only)

This is the assignment's central deliverable — plan the forward/backward math
explicitly here, not just as named functions, since "manual chain-rule gradients"
without the actual derivation is exactly the kind of code the from-scratch
requirement exists to prevent (copying a shape-correct backward pass without being
able to explain why each line is there).

**Forward pass** (batch of `m` rows, `d→32→16→1`):
```
Z1 = X·W1 + b1        A1 = relu(Z1)          # (m,d)·(d,32) → (m,32)
Z2 = A1·W2 + b2        A2 = relu(Z2)          # (m,32)·(32,16) → (m,16)
Z3 = A2·W3 + b3        ŷ  = sigmoid(Z3)       # (m,16)·(16,1) → (m,1)
```
Cache `X, Z1, A1, Z2, A2, Z3, ŷ` — the backward pass needs all of these, not just the
final output.

**Loss**: `L = -mean( y·log(ŷ) + (1-y)·log(1-ŷ) )` (binary cross-entropy).

**Backward pass** — derive each gradient in a markdown cell, in words, before the
code, same standard assignment_03's from-scratch DL section used:
```
dZ3 = ŷ - y                          # combined sigmoid+BCE gradient — this exact
                                      # simplification is *why* sigmoid+BCE are paired:
                                      # d(BCE)/d(sigmoid output) · d(sigmoid)/dZ3
                                      # collapses algebraically to (ŷ - y)
dW3 = A2ᵀ·dZ3 / m        db3 = mean(dZ3, axis=0)

dA2 = dZ3·W3ᵀ                        # how much layer 2's output contributed to
                                      # the output error, distributed back through W3
dZ2 = dA2 * relu_derivative(Z2)      # zero out the contribution of any unit that
                                      # was inactive (Z2 ≤ 0) during the forward pass
dW2 = A1ᵀ·dZ2 / m        db2 = mean(dZ2, axis=0)

dA1 = dZ2·W2ᵀ
dZ1 = dA1 * relu_derivative(Z1)
dW1 = Xᵀ·dZ1 / m         db1 = mean(dZ1, axis=0)
```
Update: `W -= lr * dW`, `b -= lr * db` for every layer (plain SGD; a momentum/Adam-
style update is a reasonable upgrade but state explicitly which was used, since Keras/
PyTorch will default to Adam and an unstated optimizer mismatch would silently break
the "comparable hyperparameters" fairness rule).

**Gradient check (do this before trusting the training loop):** on a small random
batch (e.g. 5 rows) with the network's *actual* initialized weights, compute
`dW1[0,0]` (or any single entry) two ways — (a) the analytic formula above, (b)
numerically via central difference, `(L(W+ε) - L(W-ε)) / (2ε)` with `ε≈1e-5` — and
confirm they agree to several significant figures. This is cheap (one perturbed
forward pass per checked entry) and catches transposition/broadcasting bugs in the
backward derivation that would otherwise silently train "successfully" on a wrong
gradient. Run it once, in its own cell, before the full training loop.

**Functions to implement**: `linear(x,W,b)`, `relu(x)`/`relu_derivative(z)`,
`sigmoid(z)`/(`sigmoid` alone is enough — its derivative isn't needed separately once
the `dZ3 = ŷ-y` shortcut is used), `binary_cross_entropy(y_true,y_pred)`,
`forward(X,params)` returning `(ŷ, cache)`, `backward(y,cache,params)` returning a
gradients dict, `numerical_gradient_check(X,y,params,epsilon)` (the check above, kept
as a reusable function rather than one-off scratch code).

Mini-batch loop: shuffle each epoch (seeded), iterate batches, forward → loss →
backward → update; record loss (and accuracy) per epoch. No `tensorflow`/`keras`/
`torch` import anywhere in this section.

### Keras
- `tf.keras.Sequential` with two `Dense(..., activation="relu")` layers and a final
  `Dense(1, activation="sigmoid")`.
- `model.compile(optimizer=..., loss="binary_crossentropy", metrics=["accuracy"])`,
  `model.fit(X_train, y_train, epochs=..., batch_size=..., validation_split=...)` —
  same epoch/batch-size values chosen for the scratch model.

### PyTorch
- `nn.Module` subclass: `self.net = nn.Sequential(nn.Linear(d,32), nn.ReLU(),
  nn.Linear(32,16), nn.ReLU(), nn.Linear(16,1))`, sigmoid applied either in `forward`
  or folded into `nn.BCEWithLogitsLoss` (prefer the latter — numerically stabler,
  matches the lecture's guidance on not manually applying softmax/sigmoid before a
  framework's fused loss, slide 17's equivalent rule for classification losses).
- Explicit training loop: `optimizer.zero_grad() → outputs = model(x) → loss =
  criterion(outputs, y) → loss.backward() → optimizer.step()`, same batch size/epochs.

## Component-by-implementation table
Filled in during the notebook (this plan states the expected content, not the final
wording — write the actual cell text once the code exists):

| Component | Scratch | Keras | PyTorch |
|---|---|---|---|
| Data loading | `pandas.read_csv` × 5, `pd.concat` | same | same |
| Preprocessing | manual mean/std standardization | `StandardScaler` or manual | same as scratch |
| Model definition | explicit functions + a params dict | `Sequential([...])` | `nn.Module` subclass |
| Activation | `relu()`, `sigmoid()` | `activation="relu"/"sigmoid"` | `nn.ReLU()`, fused in `BCEWithLogitsLoss` |
| Dense/Linear layer | `linear()` (manual matmul) | `Dense(...)` | `nn.Linear(...)` |
| Loss | manual `binary_cross_entropy` | `"binary_crossentropy"` | `nn.BCEWithLogitsLoss()` |
| Gradient | manual backprop | `GradientTape`/autodiff (implicit in `fit`) | `loss.backward()` (autograd) |
| Optimizer | manual `W -= lr * dW` | `optimizer="adam"` | `torch.optim.Adam(...)` |
| Training loop | explicit `for epoch / for batch` | `model.fit(...)` | explicit `for epoch / for batch` |
| Evaluation | manual metric functions | `model.evaluate(...)` or manual | manual (`model.eval()`, no-grad) |
| Prediction | `forward(X_test, params)` | `model.predict(...)` | `model(x_test)` under `torch.no_grad()` |

(No Convolution/Pooling/Flatten rows — not applicable to this app, per
`REQUIREMENT.md`'s note that diabetes keeps the MLP architecture.)

## Notebook cell-by-cell plan

**Setup & data**
1. Markdown: title, problem statement, and what this notebook builds (MLP,
   3 implementations, why an MLP and not a CNN for this data) — written self-
   contained, in the notebook's own words; do not link to or assume the reader has
   this plan file open.
2. Imports (`numpy`, `pandas`, `matplotlib`/`seaborn`, `sklearn.metrics`,
   `tensorflow`/`keras`, `torch`) — one cell, clearly separating "framework" imports
   from the numpy-only imports the scratch section will restrict itself to.
3. `RANDOM_SEED = 42` (numpy, `random`, `tf.random`, `torch.manual_seed`).
4. Load the 5 per-year CSVs individually (not yet concatenated) and print
   `set(df.columns)` per year; identify any columns not common to all 5.
4a. Concatenate on the column **intersection** only, add/verify `year`; markdown
   cell documenting which columns (if any) were dropped for not being common to all
   5 years, and why.
5. Markdown: "Data understanding."
6. `.shape`, `.dtypes`, `.head()`, `.describe()` — confirm actual row count here
   against the confirmed 2,181,125-row target from this plan's §Dataset.
6a. Sentinel-value audit on the continuous columns (BMI, weight, height, etc.):
   check for known BRFSS placeholder codes (e.g. `999`, implausible outliers) via
   value counts / histograms, before anything gets scaled.
7. Missing-value audit (`.isna().sum()`).
8. Target class-balance plot + markdown Observation/Interpretation.
9. EDA plot 2 (e.g. BMI vs. target) + markdown Observation/Interpretation.
10. EDA plot 3 (e.g. general-health score vs. target) + markdown
    Observation/Interpretation.
11. EDA plot 4 (`year` vs. target rate across all 5 years, to inform the
    year-as-feature decision) + markdown Observation/Interpretation.

**Cleaning & representation**
12. Markdown: "Data cleaning."
13. Collapse 3-class target to binary (one cell, this operation only).
14. Drop raw-categorical duplicate columns, keep `_00`/`_01` ordinal versions.
14a. Clean sentinel values identified in cell 6a to `NaN`, then impute (e.g. median)
    — one cell, this operation only, with a one-line markdown note of which columns
    and which sentinel values were cleaned.
15. Markdown: "Representation" + confirmed `d`, the `year`-feature decision with its
    reasoning, and how `year` is encoded if kept (small ordinal + scaled, or one-hot
    — per this plan's §Representation note on why a raw calendar year is unsafe).
16. Train/test split (80/20, `RANDOM_SEED`) — this exact split object is reused by
    all 3 implementations below, not re-split per implementation.
17. Standardization: fit on train only, transform both, for continuous columns
    (including `year` if it was folded into this group rather than one-hot).
18. Shape sanity-check cell (`X_train.shape`, `X_test.shape`, class balance in each
    split) — **also print the min/max of every column in `X_train`** to directly
    verify nothing (especially `year`) is left on a wildly different scale than the
    rest, per this plan's §Representation warning.
18a. Class-balance check + weight computation: confirm the actual positive-class
    proportion, and if meaningfully skewed, compute balanced sample weights (the
    `n_total / (n_classes * n_in_class)` formula) once here, reused identically by
    all 3 implementations below — markdown cell stating the actual imbalance ratio
    found and the resulting per-class weight values.

**Shared architecture definition**
19. Markdown: "Shared architecture" — state `d → 32 → 16 → 1`, ReLU/sigmoid, BCE,
    and the fixed batch size / epoch count used by all 3 implementations below.
20. Markdown: shape-trace table (`(batch, d) → (batch,32) → (batch,16) → (batch,1)`).

**Implementation 1 — scratch (NumPy)**
21. Markdown: "Implementation 1: from scratch (NumPy only)."
22. `relu`, `relu_derivative`.
23. `sigmoid`.
24. `binary_cross_entropy` loss.
25. Weight init: `W1,b1,W2,b2,W3,b3`, He-init, seeded, sized `d→32→16→1`.
26. `forward(X, params)` function, returns `(ŷ, cache)`.
27. `backward(y, cache, params)` function — implements the `dZ3=ŷ-y` derivation from
    this plan's §Scratch section exactly; markdown cell above it restating the
    derivation in words before the code.
28. **Gradient-check cell**: on a small random batch, compare `backward()`'s analytic
    gradient against a numerical central-difference gradient for a few weight entries
    (one per layer); assert they agree to ~4-5 significant figures before proceeding.
    Do not skip this — it is the cheapest available evidence that the hand-derived
    backward pass is actually correct, not just shape-compatible.
29. Mini-batch training loop: shuffle → batch → forward → loss → backward → update,
    passing the sample weights from cell 18a into both the loss and `backward()`
    calls if class weighting is in use; `loss_history`/`acc_history` per epoch.
30. Predict on test set (threshold 0.5), compute Accuracy/Precision/Recall/F1 +
    confusion matrix.
31. Record parameter count and training time (wall-clock) for this implementation.

**Implementation 2 — Keras**
32. Markdown: "Implementation 2: TensorFlow/Keras."
33. Build `Sequential` model exactly matching the shared architecture.
34. `model.summary()` — confirms parameter count independently of cell 31's manual
    count (should match, since the architecture is the same).
35. `model.compile(...)`.
36. `model.fit(..., class_weight={0: w0, 1: w1})` if class weighting is in use (same
    weight values as cell 18a) — same batch size/epochs as cell 29; capture
    `history` and wall-clock training time.
37. Predict on test set, compute the same metric set as cell 30.

**Implementation 3 — PyTorch**
38. Markdown: "Implementation 3: PyTorch."
39. `nn.Module` class definition matching the shared architecture.
40. Instantiate model, loss (`BCEWithLogitsLoss(pos_weight=...)` if class weighting
    is in use, matching cell 18a's weight ratio), optimizer (`Adam`, same LR as
    Keras where possible — note explicitly if not exactly comparable).
41. Wrap `X_train`/`y_train` in a `DataLoader` with the same batch size as cells 29/36.
42. Explicit training loop: `zero_grad → forward → loss → backward → step`, same
    epoch count; capture per-epoch loss and wall-clock training time.
43. Predict on test set (`model.eval()`, `torch.no_grad()`), compute the same metric
    set as cells 30/37.

**Comparison & visualization**
44. Markdown: "Comparing the three implementations."
45. Component-by-implementation table (this plan's table above, as a markdown or
    `DataFrame` cell).
46. Quantitative comparison table: params, training time, epochs, final training
    loss, test accuracy, precision/recall/F1 — one row per implementation.
47. Combined training-curve plot: loss vs. epoch, 3 lines (scratch/Keras/PyTorch).
48. Confusion-matrix heatmaps, one panel per implementation (or 3 separate cells).
49. Bar chart: test accuracy across the 3 implementations.
50. Bar chart: parameter count and training time across the 3 implementations.
51. Markdown: written comparison — which implementation hid the most detail, which
    gave the most control, whether accuracy actually converged to comparable numbers
    (expected, since the architecture is identical) or diverged and why.

**Persistence**
52. Save scratch weights (`np.savez`).
53. Save Keras model (`model.save(".../keras_model.keras")`).
54. Save PyTorch model (`torch.save(model.state_dict(), ".../pytorch_model.pt")`).
55. Save `feature_names.joblib` and `input_schema.json`.
56. Reload-and-verify cell: reload all 3 artifacts, confirm predictions match the
    in-memory versions.

## Open questions
- Final year list/row count for the diabetes dataset — deferred to cell 6, adjusted
  there if needed, documented in that cell's markdown.
- Keep or drop `year` as a feature — deferred to cell 15, decided empirically.
- Exact batch size / epoch count / learning rate — fixed once in cell 19-20 after
  seeing the confirmed dataset size, then held constant across all 3 implementations;
  not tuned per implementation, since untuned-but-fixed hyperparameters are what the
  fairness rule requires.
- Whether plain SGD or an Adam-style update is used for the scratch model's parameter
  updates — deferred to cell 27's implementation, but whichever is chosen must be
  stated explicitly in that cell's markdown so the Keras/PyTorch sections' optimizer
  choice can be called "comparable" or explicitly flagged as not.
