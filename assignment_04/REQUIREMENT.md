# Assignment 04 — Requirements

Source material: `materials/intel_sys_dev_slide_04_compare_models_CNN.pdf` (Lecture 04 —
Comparing CNN Implementations) and `materials/Deep_Learning_CNN_Function_Composition_Tutorial.pdf`
(Deep Learning as Function Composition — From Neurons and Functions to CNN, ResNet,
Attention, and Vision Transformers). Project layout follows `assignment_03/` where the
slides don't say otherwise.

## Goal

Lecture 04's whole point is **not** to learn three CNN libraries — it's to show that the
same learning problem, expressed as the same architecture, looks different only in
*abstraction level* across three implementations:

```
CNN from Scratch (NumPy)  →  TensorFlow/Keras (high-level API)  →  PyTorch (flexible framework)
```

and that "different names do not imply different machine-learning concepts" (slide 18).
The lecture's own worked example does this with two datasets precisely because they
expose different structure:

- **Diabetes (tabular)** — `X ∈ ℝ^{n×d}`, no spatial neighborhood between features, so
  the natural model is an **MLP**: `X → Linear → ReLU → Linear → ŷ` (slide 20). This is
  explicit in the material — diabetes is never given a convolution in the lecture.
- **MNIST (image)** — `X ∈ ℝ^{1×28×28}`, has 2-D spatial structure a CNN can exploit:
  `28×28 → Conv → ReLU → Pool → Conv → ReLU → Pool → Flatten → Linear → 10` (slide 21).

This assignment keeps that same Part A / Part B split from slide 27 ("Part A —
Diabetes: implement an MLP from scratch / Keras / PyTorch"; "Part B — MNIST: implement
a simple CNN from scratch / Keras / PyTorch") but widens Part B to **two** image
datasets instead of one, and grows Part A's diabetes dataset, per the professor's
separate verbal requirement that each assignment's dataset be larger than the previous
one's.

```
Raw data → Represent → Build (same architecture, 3 implementations) → Train → Evaluate → Compare → Visualize
```

**A note on "3 CNN models" in the verbal brief**: read literally against the material,
only the two image apps get an actual convolutional architecture — the diabetes app
keeps the MLP architecture the lecture itself assigns to tabular data (slide 20 states
plainly "There is no natural convolution operation" for Diabetes). What's constant
across all three apps, and what "3 CNN models" actually refers to in spirit, is the
**3-implementation comparison pattern** (scratch / Keras / PyTorch) applied to
whichever architecture fits the data. Building a convolution over an 18-ish-column
tabular feature vector would contradict the lecture's own stated rationale for why
CNNs exist (local spatial connectivity, weight sharing — materials §9.3-9.4), so this
plan does not do that.

## The 3 applications

| # | App | Data type | Task | Architecture (from slides / tutorial) | Implementations |
|---|---|---|---|---|---|
| 1 | `diabetes` | tabular | binary classification | `d → Linear → ReLU → Linear → ŷ` (MLP, slide 20) | scratch (NumPy) + Keras + PyTorch |
| 2 | `fashion_mnist` | image, 28×28 grayscale | 10-class classification | `28×28 → Conv→ReLU→Pool → Conv→ReLU→Pool → Flatten→Linear→10` (slide 21) | scratch (NumPy) + Keras + PyTorch |
| 3 | `cifar10` | image, 32×32 RGB | 10-class classification | `3×32×32 → Conv(32)→ReLU→Pool → Conv(64)→ReLU→Pool → Flatten→Linear(128)→ReLU→Linear(10)` (tutorial §16, "A Complete Small CNN") | scratch (NumPy) + Keras + PyTorch |

### Why these two image datasets specifically

Materials-silent on *which* image dataset beyond the lecture's own MNIST demo, so the
two choices below are picked because each is a **direct, unmodified instantiation of an
architecture the materials already worked out numerically** — nothing is invented:

- **Fashion-MNIST** (`60,000` train + `10,000` test, `28×28`, grayscale, 10 classes) —
  same tensor shape (`1×28×28`) and same architecture slide 21 gives for MNIST, just a
  harder classification problem (clothing categories, not digits) so results aren't a
  trivial restatement of the lecture's own demo dataset.
- **CIFAR-10** (`50,000` train + `10,000` test, `32×32×3`, RGB, 10 classes) — the
  tutorial's own worked convolution-output-size example uses `H=W=32, K=3, P=1, S=1`
  (tutorial §10.1) and its "A Complete Small CNN" section (§16) builds `SmallCNN`
  against a `3×32×32` input with the exact `Conv(3→32)→Pool→Conv(32→64)→Pool→Flatten
  (64·8·8=4096)→Linear(128)→Linear(num_classes)` architecture this app uses verbatim.

Both are loaded via each framework's/library's built-in dataset loader
(`torchvision.datasets`, `tf.keras.datasets` / `keras.datasets`) — no manual download,
no Kaggle credentials, no large files to gitignore, unlike the tabular app.

### Diabetes dataset — bigger than assignment_03's

`assignment_03/diabetes` used 2 years (`DATASET_2019.csv` + `DATASET_2020.csv`) from
`spandanjit2005/brfss-diabetes-indicator-dataset` for **815,711** rows. That app's own
plan explicitly reserved room to grow ("leaving headroom for a later assignment to grow
into without a new dataset search" — `assignment_03/diabetes/PLAN.md` §Dataset), so
this assignment draws from the **same source repo**, adding more of its per-year files
(the repo covers 2005-2024, one CSV per year, identical 33-column schema) rather than
switching datasets:

- Years: **2016, 2017, 2018, 2019, 2020** (5 years). Downloaded and confirmed by
  direct row count: 483,230 + 447,952 + 434,232 + 416,661 + 399,050 = **2,181,125
  rows**, 2.67x assignment_03's 815,711 — inside the 2-3x target band. A 6th year
  (2015) was downloaded and checked too but dropped: adding it pushed the total to
  2,621,006 rows (3.2x, over the target band) and added ~120MB for no benefit to the
  fit, so 5 years was kept as the better-centered, smaller choice.
- Concatenate exactly as assignment_03 did: identical schema across years, a `year`
  column already present distinguishes provenance, same target-collapse rule
  (3-class diagnosis → binary: non-diabetic vs. pre-diabetic-or-diabetic).

**Do not carry the "bigger than the previous assignment" framing into the final
report.** `assignment_03`'s README/report both stated the growth multiplier explicitly
(e.g. "3.2x bigger") as a deliberate, professor-stated requirement worth documenting.
For this assignment, treat the dataset's size as simply *the dataset chosen* — no
"vs. assignment_03" comparison table, no multiplier callout, no sentence framing the
choice as satisfying a sizing rule. This paragraph and the "Years" bullet above exist
so the choice can be reasoned about and verified while planning; none of that reasoning
belongs in `report/`.

## Model requirements per app

Every app builds **one architecture, three times** — same weights-equivalent structure,
different level of abstraction, per slide 6 ("The conceptual architecture is
unchanged. What changes is: Level of abstraction"):

1. **From scratch (NumPy only)** — every component listed on slide 7/18 written as an
   explicit Python function and manually composed:
   - `linear(x, W, b)`, `relu(x)`, and for the two CNN apps also `conv2d(x, kernel,
     stride)` and `max_pool2d(x)` (materials §9, tutorial §9-13) plus `flatten(x)`.
   - Manual forward pass as explicit function composition, matching tutorial §7's
     `F(x) = f_3 ∘ f_2 ∘ f_1(x)` framing and slide 8's
     `f(x) = f_linear(f_flatten(f_pool(f_relu(f_conv(x)))))`.
   - Manual loss: binary cross-entropy (diabetes) or softmax cross-entropy over 10
     classes (fashion_mnist, cifar10), per tutorial §17.
   - Manual backpropagation via the chain rule (no autograd) and manual gradient
     descent / Adam-style parameter updates — same "no TensorFlow/PyTorch/Keras
     imports in the DL implementation" rule assignment_03 used for its from-scratch
     model.
   - Weight init: He-style (`np.random.randn(...) * sqrt(2/fan_in)`, zero biases),
     consistent with assignment_03.
   - Explicit training loop tracking loss (and accuracy) per epoch, matching slide 11's
     `Forward → Loss → Gradient → Update` cycle.
2. **TensorFlow/Keras** — same architecture expressed with `tf.keras.layers`
   (`Conv2D`, `MaxPooling2D`, `Flatten`, `Dense` for the CNN apps; `Dense` only for
   diabetes), compiled with `model.compile(...)` and trained with `model.fit(...)`
   (slide 12-13). No custom training loop — that abstraction *is* the point of
   comparison.
3. **PyTorch** — same architecture expressed as an `nn.Module` with `nn.Conv2d`,
   `nn.ReLU`, `nn.MaxPool2d`, `nn.Flatten`, `nn.Linear` for the CNN apps (`nn.Linear`
   only for diabetes) inside `self.features` / `self.classifier` (slide 14), with an
   explicit training loop (`zero_grad → forward → loss → backward → step`, slide 16) —
   PyTorch sits deliberately between scratch's full manual loop and Keras's `fit()`.

Same train/test split and no-leakage preprocessing discipline as assignment_02/03
(`RANDOM_SEED = 42`, scaler/normalization statistics computed on train only, reused on
test) — applies to diabetes feature standardization and to image pixel normalization
(e.g. `/255.0` or per-channel mean/std) alike.

**Fair comparison constraint (slide 29, explicit in the material):**
> Same Dataset + Same Split + Same Architecture + Comparable Hyperparameters

All three implementations of a given app must use the same train/test split, the same
layer widths/depths, and the same (or documented-equivalent) learning rate, batch size,
and epoch count — differences in the results should come from the implementation, not
from silently different setups. Two concrete mechanisms follow from this:

- **Fix hyperparameters once per app, before building any of the three
  implementations** — not per implementation. Once the confirmed dataset size/shape is
  known (e.g. after the EDA cells), pick batch size, epoch count, and learning rate in
  one place and reuse those exact values in the scratch, Keras, and PyTorch sections
  that follow. Do not retune per implementation.
- **Weight initialization is not assumed to match across frameworks.** The scratch
  implementation uses explicit He-style init (`np.random.randn(...) *
  sqrt(2/fan_in)`, zero biases). Keras and PyTorch each apply their own default
  initializer (both are also ReLU-appropriate, He/Kaiming-family, but not guaranteed
  bit-identical to the scratch formula or to each other) — state whichever default
  each framework actually used rather than assuming parity. A difference here is a
  legitimate, reportable source of divergence between implementations, not something
  to paper over.

## Required comparisons & visualizations (per app)

This is the assignment's central deliverable — slide 27-29 are explicit that the
comparison must go beyond a single accuracy number:

**1. Component-by-implementation table** (slide 28's exact table, filled in per app):

| Component | Scratch | Keras | PyTorch |
|---|---|---|---|
| Data loading | | | |
| Preprocessing | | | |
| Model definition | | | |
| Convolution *(CNN apps only)* | | | |
| Activation | | | |
| Pooling *(CNN apps only)* | | | |
| Flatten *(CNN apps only)* | | | |
| Dense/Linear layer | | | |
| Loss | | | |
| Gradient | | | |
| Optimizer | | | |
| Training loop | | | |
| Evaluation | | | |
| Prediction | | | |

**2. Quantitative comparison** (slide 29's list, all recorded per implementation):
number of parameters, training time, number of epochs, final training loss,
validation/test accuracy, precision/recall/F1, confusion matrix.

**3. Visualizations, at each step of the pipeline** (the verbal "add visualizations in
each step" requirement), per app:
- **Data understanding**: for images, a grid of sample images per class + class-balance
  bar chart; for diabetes, feature distributions / class-balance bar chart (same EDA
  discipline as assignment_02/03).
- **Architecture / shape progression**: a diagram or printed shape trace of the tensor
  size after every layer (slide 22's `B×1×28×28 → B×32×28×28 → B×32×14×14 → ...`
  pattern) — produced once per app and shown identically for all 3 implementations,
  since the architecture is identical by construction.
- **Training curves**: loss (and accuracy) vs. epoch, one line per implementation on
  a shared plot so the 3 curves are visually comparable.
- **Confusion matrix heatmap** per implementation (or one shared figure with 3 panels).
- **Cross-implementation bar charts**: accuracy comparison, parameter-count comparison,
  training-time comparison — 3 bars each (scratch / Keras / PyTorch).
- A short written comparison per app answering slide 28's own question: *"Which
  framework hides the most implementation details? Which gives the programmer the
  most control?"* — grounded in what was actually observed building it, not restated
  from the slide.

A top-level summary (in the final report) comparing all 3 apps × 3 implementations on
the primary metric is expected, mirroring assignment_03's cross-application summary
table.

### Canonical notebook structure (per app)

Every `<app>.ipynb` follows the same section order, so the three per-app notebooks
stay structurally comparable and each `<app>/PLAN.md` only has to state what differs
(dataset specifics, architecture widths) rather than re-deriving the shape of the
notebook itself:

1. **Setup & data** — imports, `RANDOM_SEED`, load raw data, EDA + visualizations.
2. **Cleaning & representation** — target/feature preparation, train/test split
   (computed once, reused by all 3 implementations below), normalization/
   standardization fit on train only.
3. **Shared architecture definition** — state the exact layer-by-layer architecture
   and its tensor shape trace once, before any implementation, since it is identical
   across all three by construction; fix batch size/epoch count/learning rate here
   too (see the fairness-mechanism bullets above) so every implementation section
   below reuses the same values without re-deciding them.
4. **Implementation 1 — scratch (NumPy)**, **Implementation 2 — Keras**,
   **Implementation 3 — PyTorch** — one section each, same internal shape (build →
   train → predict → per-implementation metrics), in that order.
5. **Comparison & visualization** — component-by-implementation table, quantitative
   comparison table, combined training-curve plot, confusion matrices, cross-
   implementation bar charts, written comparison.
6. **Persistence** — save all 3 trained artifacts + `feature_names.joblib`
   (diabetes only) + `input_schema.json`, then a reload-and-verify cell.

### Notebook writing style (applies to every markdown cell, in all 3 notebooks)

This governs the prose inside the notebooks themselves, not this plan or
`REQUIREMENT.md`. Per-app plans reference this section rather than restating it.

- Write like someone explaining their own reasoning to a colleague while building
  the thing, not like a report generator summarizing a finished result. A markdown
  cell should show the thinking, not just announce the outcome. For example, not
  "Continuous columns were standardized," but something closer to: "Gradient descent
  converges faster when inputs are on a similar scale. Without scaling, BMI (values
  around 20 to 50) would dominate a 0/1 flag column in the first layer's weighted
  sum, just because of its magnitude, not because it's more informative. So the
  continuous columns get standardized before training, fit on the training split
  only."
- No em dashes. Use a period, a comma, or a plain connector ("so," "because,"
  "which") instead of breaking a sentence with —.
- Prefer short, direct sentences. If a sentence has more than one comma-separated
  clause, look for a place to split it into two sentences instead.
- Avoid stock academic phrasing: "This section explores...", "It is worth noting
  that...", "In conclusion...". Say the thing plainly.
- State claims directly rather than as a negation followed by a correction ("This
  isn't a sign of X. It's actually Y."). Prefer "Y, because..." or "Y: ..." — say
  what's true and why in one move, instead of first denying something then
  explaining the real answer in a separate sentence.
- Don't forward-reference something the reader hasn't seen yet ("this notebook
  comes back to this later," "as explained below") unless the very next cell
  actually delivers on it. If a forward reference is more than one or two cells
  away, either state the fact directly now or cut the reference.
- Every markdown cell that sets up an observation (a plot, a printed value) needs
  to actually land the observation, not trail off into a section header for
  something else. If a paragraph ends with "it's worth checking whether X," the
  next cell must say what was actually found, before any new section starts.
- Prefer prose over bullet lists for reasoning and interpretation. Reserve bullet
  lists for things that are genuinely enumerable (a set of classes, a list of
  hyperparameters), not for explaining why a choice was made.
- First person plural ("we standardized," "we observed") is fine and matches how a
  student narrates their own build process.
- Code comments stay terse, unlike markdown cells. A one-line comment noting a
  non-obvious choice is enough; the reasoning belongs in the markdown cell above the
  code, not embedded in the code itself.

## Project structure (per app, following `assignment_03/<app>/`)

```
<app>/
  data/                 raw dataset (diabetes only — image apps load via
                         torchvision/keras built-in loaders, no files committed)
  notebook/              <app>.ipynb — EDA + shared architecture definition +
                         3 implementations (scratch/Keras/PyTorch) + evaluation/
                         comparison + visualizations, executed with outputs
  model/                 scratch_weights.npz + keras_model.keras + pytorch_model.pt
                         + feature_names.joblib (diabetes only) + input_schema.json
  requirements.txt       numpy, pandas, matplotlib/seaborn, scikit-learn (metrics
                         only), tensorflow/keras, torch, torchvision
report/                  Assignment_04.pdf / .tex — final report across all 3 apps
slides/                  presentation deck — see SLIDES_REQUIREMENT.md
```

No `api/`, `web/`, or `mobile/` subfolders — same reasoning as assignment_03: neither
source deck goes past the modeling/comparison pipeline, so deployment stays out of
scope unless explicitly requested later.

## Constraints checklist

- [ ] Every app implements the **same architecture** three times: from-scratch NumPy,
      Keras, PyTorch (3 apps × 3 implementations = 9 trained models total).
- [ ] Diabetes keeps the MLP architecture (no convolution) per slide 20; both image
      apps use a CNN per slide 21 / tutorial §16.
- [ ] No `tensorflow`/`keras`/`torch` imports anywhere inside the from-scratch
      implementation's forward/backward/update code.
- [ ] Same train/test split, same architecture, comparable hyperparameters across the
      3 implementations within an app (slide 29's fairness rule).
- [ ] `RANDOM_SEED = 42`, no leakage (fit scalers/normalization on train only).
- [ ] Component-by-implementation table (slide 28) filled in for every app.
- [ ] Quantitative comparison recorded for every implementation: params, training
      time, epochs, training loss, test accuracy, precision/recall/F1, confusion
      matrix.
- [ ] Visualizations present at each pipeline step (data, architecture/shapes,
      training curves, confusion matrices, cross-implementation bars) for every app.
- [ ] Diabetes dataset drawn from more years of the same assignment_03 source,
      targeting ~2-3x assignment_03's row count — verified empirically in-notebook,
      and **not** framed as a sizing comparison anywhere in `report/`.
- [ ] Final report compares all 3 implementations per app and summarizes across all
      3 apps.
- [ ] A presentation deck exists per `SLIDES_REQUIREMENT.md`, covering definitions,
      model development process, and a demonstration.
