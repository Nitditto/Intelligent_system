# Assignment 04 — Report Plan

This file plans the **final report**: what sections it has, what goes in each section,
and — for every per-app section — exactly what to pull out of that app's notebook to
write it. It does not plan the notebooks themselves; that would live in each app's own
`<app>/PLAN.md` (`diabetes/PLAN.md`, `fashion_mnist/PLAN.md`, `cifar10/PLAN.md`), not
yet written — this file references where that detail will live rather than repeating it.

## Why this structure is an assumption, and what it's based on

Neither `REQUIREMENT.md` nor either PDF in `materials/` says anything about a report:
no required sections, no page count, no cover-page format, no submission format —
same gap assignment_03 had. `REQUIREMENT.md`'s own header states *"Project layout
follows assignment_03/ where the slides don't say otherwise"* — the report is exactly
a case where the slides say nothing, so this plan reuses **assignment_03's actual
report shape** (`assignment_03/PLAN.md`, recovered from git history) as the template,
trimmed of what's now out of scope and extended with what this round's material
actually asks for.

**Treat this as a draft to confirm, not a settled requirement** — check with the
professor if there's a chance to ask before finalizing.

What changed relative to assignment_03's report:
- **Removed**: the "classical ML vs. from-scratch DL" comparison — this assignment
  isn't about classical ML at all. There is no 4-model comparison this round.
- **Removed**: the from-scratch-only framing — every app now has **three**
  implementations of the *same* architecture (scratch / Keras / PyTorch), not one
  from-scratch model compared against classical baselines.
- **Added**: a framework-level section explaining Lecture 04's central idea —
  "same architecture, different abstraction level" — parallel to how assignment_03's
  report had a framework-level section on function composition/representation
  learning (`assignment_03/PLAN.md` §3), because both source decks build toward one
  idea that each per-app section then instantiates rather than re-derives.
- **Added**: the component-by-implementation table (slide 28) and the
  quantitative implementation-comparison table (slide 29: params, training time,
  epochs, loss, accuracy, precision/recall/F1, confusion matrix) for every app —
  these are explicit, named deliverables in the material this round, distinct from
  assignment_03's per-model metrics tables.
- **Added**: an explicit "which framework hid the most, which gave the most control"
  discussion per app, echoing slide 28's own question.
- **Kept**: cover page fields, the mandatory per-app EDA/visualization discipline, a
  cross-application comparison table, the reproducibility section.
- **Changed emphasis**: assignment_03's report leaned on dataset-size growth as a
  headline fact (explicit multiplier callouts). This report does **not** do that for
  the diabetes dataset — see `REQUIREMENT.md`'s explicit instruction not to carry that
  framing into `report/`. The diabetes dataset is introduced like any other dataset:
  source, size, target, representation — no "vs. assignment_03" language anywhere.

---

## Section-by-section plan

### 1. Cover page
Same fields as assignment_02/03: student name, student ID, class, team, lecturer
(Dinh Que Tran), semester. Nothing app-specific — write once.

### 2. Executive summary
One paragraph stating what was built: 3 applications, each with the *same* learning
problem/architecture implemented three separate times (NumPy from scratch, TensorFlow/
Keras, PyTorch), to compare abstraction level rather than to compare models against
each other. One summary table:

| Application | Data type | Task | Dataset | Architecture | Winning implementation (by accuracy) |
|---|---|---|---|---|---|
| Diabetes | tabular | binary classification | BRFSS (multi-year) | MLP (`50 → 32 → 16 → 1`) | PyTorch (0.782 vs. Keras 0.753, scratch 0.731) |
| Fashion-MNIST | image, 28×28 grayscale | 10-class classification | Fashion-MNIST | CNN (32→64 channels, 421,642 params) | Scratch (0.897 vs. PyTorch 0.866, Keras 0.860) |
| CIFAR-10 | image, 32×32 RGB | 10-class classification | CIFAR-10 | CNN (32→64 channels, 545,098 params) | Keras (0.610 vs. scratch 0.603, PyTorch 0.570) |

**Diabetes notebook: executed, 0 errors, 96 cells, all 3 models trained and
persisted (`diabetes/model/`).** Confirmed input width `d = 50` (13 scaled numeric
columns + 8 binary yes/no columns + 29 one-hot columns from race/marital
status/employment status/personal-doctor). Confirmed positive rate 15.56%
(imbalanced, as expected), addressed with balanced class weighting
(`weight_neg=0.5922`, `weight_pos=3.2126`) applied identically across all 3
implementations. Real per-implementation results:

| Implementation | Params | Train time | Test accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Scratch | 2,177 | 75.1s | 0.7310 | 0.3383 | 0.7616 | 0.4685 |
| Keras | 2,177 | 29.1s | 0.7533 | 0.3554 | 0.7194 | 0.4758 |
| PyTorch | 2,177 | 345.7s | 0.7819 | 0.3814 | 0.6458 | 0.4796 |

Parameter counts match exactly across all 3 (confirms identical architecture).
Accuracy/F1 converge within ~5 points of each other, as expected for the same
architecture trained 3 ways. Two real findings worth carrying into the report's
Discussion section:
- **Final training loss is not comparable across implementations** (scratch/Keras
  ≈0.51, PyTorch ≈0.87), not because PyTorch trained worse, but because
  `BCEWithLogitsLoss(pos_weight=...)` only reweights the positive class's loss
  term, while scratch's manual weighting and Keras's `class_weight` both reweight
  *every* row (both classes), keeping the loss on a different numeric scale. Same
  balanced-weighting intent, different framework-level parameterization — exactly
  the kind of "same name, different mechanics" gap this assignment is built to
  surface.
- **PyTorch was the slowest by a wide margin** (345.7s vs. Keras's 29.1s and
  scratch's 75.1s) despite the network being tiny (2,177 params). This traces to
  PyTorch's per-batch autograd graph construction/teardown overhead, which is a
  fairly fixed cost per step that a network this small can't amortize away, unlike
  Keras's compiled-graph `fit()` loop. Worth stating explicitly that this is a
  small-model artifact, not a general "PyTorch is slower" claim.

Balanced weighting's expected trade-off shows up clearly: recall rose to
0.65-0.76 across all 3 (vs. what an unweighted model would have gotten), at a
real cost to precision (0.34-0.38) — the models were pushed to catch more of the
rare positive class, and did, at the cost of more false positives.

**Fashion-MNIST notebook: executed, 0 errors, 57 cells, all 3 models trained and
persisted.** A real bug was caught and fixed here via the gradient-check cell
before any full training run: `linear_backward` was dividing by batch size a
second time on top of `softmax_cross_entropy_backward`'s existing `/N`, silently
shrinking every dense-layer gradient (conv-layer gradients were unaffected, since
`conv2d_backward` never divided). Fixed by removing the redundant division
(`db` now uses `sum` instead of `mean`); confirmed via the gradient check
matching to ~1e-9 relative error after the fix, then executed clean. Real
per-implementation results:

| Implementation | Params | Train time | Test accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|---|---|
| Scratch | 421,642 | 1,136.1s | 0.8973 | 0.8976 | 0.8973 | 0.8970 |
| Keras | 421,642 | 76.2s | 0.8603 | 0.8695 | 0.8603 | 0.8565 |
| PyTorch | 421,642 | 192.5s | 0.8660 | 0.8676 | 0.8660 | 0.8641 |

Two real findings worth carrying into the report's Discussion section:
- **Scratch actually won on accuracy** (0.897 vs. PyTorch 0.866, Keras 0.860) —
  likely attributable to weight-initialization differences (scratch uses explicit
  seeded He-init; Keras/PyTorch use their own unseeded-relative-to-scratch
  defaults) mattering more over only 10 epochs, not a claim that hand-written
  code trains better in general. All three sit within ~4 points of each other,
  consistent with the same architecture trained three ways.
- **Training time gap is dramatic and the opposite pattern from diabetes**:
  scratch took 1,136s, roughly 15x Keras's 76s and 6x PyTorch's 193s. This is a
  CNN-specific effect: Keras/PyTorch's convolution operations are compiled,
  heavily optimized routines, while scratch's `im2col`-based conv (already ~40x
  faster than the naive position-loop it replaced) is still plain NumPy issued
  from a Python loop, with none of a framework's operator-level optimization for
  convolution specifically. Worth contrasting directly with diabetes's MLP result
  in the report, where scratch was actually *faster* than PyTorch — the
  scratch-vs-framework speed gap is architecture-dependent, not universal.

**CIFAR-10 notebook: executed, 0 errors, 50 cells, all 3 models trained and
persisted.** Same `linear_backward` fix applied before the first execution (no
wasted run). Gradient check passed including the 3-input-channel `K1` entries
specifically (rel. error ~3e-8 to 4e-11 across 5 checked entries). One cosmetic
issue found and fixed post-execution: loading CIFAR-10 triggered a one-time
~170MB download whose progress bar was captured as ~20,000 separate tiny output
chunks, bloating the executed notebook; trimmed to a single summary line via a
direct nbformat patch (no re-run needed, cosmetic only). Real per-implementation
results:

| Implementation | Params | Train time | Test accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|---|---|
| Scratch | 545,098 | 1,981.2s | 0.6033 | 0.6335 | 0.6033 | 0.5794 |
| Keras | 545,098 | 98.6s | 0.6102 | 0.6477 | 0.6102 | 0.6078 |
| PyTorch | 545,098 | 243.9s | 0.5696 | 0.6163 | 0.5696 | 0.5655 |

Findings worth carrying into the report's Discussion section:
- All 3 within ~4 points of accuracy (0.57-0.61), same spread pattern as the
  other two apps. Absolute accuracy is much lower than Fashion-MNIST's (~0.86-0.90)
  as expected: 32×32 RGB photos with genuinely ambiguous category pairs
  (cat/dog, automobile/truck) are a harder task than grayscale clothing icons,
  and this architecture uses no augmentation/regularization — a low-60s accuracy
  reflects the task and the deliberately modest setup, not a bug.
- **Training time gap is the widest of all 3 apps**: scratch (1,981.2s) is ~20x
  Keras (98.6s) and ~8x PyTorch (243.9s) — a bigger multiple than Fashion-MNIST's
  ~15x/~6x, despite CIFAR-10's architecture having only modestly more parameters
  (545,098 vs. 421,642). This points at input size/channel-count (32×32×3 vs.
  28×28×1), not parameter count, as the real driver of scratch's `im2col`
  convolution cost — a good three-way contrast for the report: MLP (scratch
  comparable to frameworks) → grayscale CNN (scratch ~15x slower) → RGB CNN
  (scratch ~20x slower), the gap widening precisely as the convolution's actual
  workload grows.

No dataset-size column here (unlike assignment_03's equivalent table) — deliberately,
per the no-sizing-framing rule above. "Winning implementation" is expected to be
**close** across the three, since they share the same architecture; a large gap is
itself a finding worth explaining (bug, hyperparameter mismatch, or a genuine
framework-level effect like different default weight init or optimizer internals),
not something to gloss over.

### 3. Connection to Lecture 04 — Same Architecture, Different Abstraction
Written once, here, so the three per-app sections don't each re-derive it:
- Restate the central question posed on slide 2: *"What is the same, and what is
  different?"* across scratch/Keras/PyTorch.
- Restate slide 6's framing: *"The conceptual architecture is unchanged. What
  changes is: level of abstraction."*
- Walk through the concept↔API correspondence table from slide 7/18 once, in prose:
  convolution, activation, pooling, flatten, dense, loss, optimizer — same
  computational role, different names, per implementation.
- State the three implementations' position on the visibility spectrum (slide 17/23):
  scratch has maximum visibility, Keras has maximum convenience, PyTorch sits in
  between (explicit training loop, but modular layers) — this is the framework-level
  claim each per-app section's "which hid the most" discussion will test against real
  numbers.
- Restate the fairness rule from slide 29 that governs every per-app comparison:
  *"Same Dataset + Same Split + Same Architecture + Comparable Hyperparameters."*

### 4. Per-application sections (×3): Diabetes, Fashion-MNIST, CIFAR-10

Each app gets an identical subsection skeleton, pulled from the app's (not-yet-written)
`<app>/PLAN.md` for the "what/why" reasoning and from the executed notebook for actual
numbers, plots, and trained-model behavior. Do not write the numeric parts from memory —
pull them from the executed notebook's output cells once they exist.

**4.x.1 Problem description** — one paragraph: task, why this architecture fits this
data (tabular → MLP per slide 20, image → CNN per slide 21), pulled from `<app>/PLAN.md`.

**4.x.2 Dataset** — source, size, target variable, class distribution. For the two
image apps, this is short (standard benchmark datasets, no cleaning decisions). For
diabetes, state the years used and the row count actually loaded — as a plain fact,
not as a comparison to any prior assignment.

**4.x.3 Data understanding & preparation** — EDA visualizations (sample-image grid +
class balance for image apps; feature distributions + class balance for diabetes) with
Observation/Interpretation pulled from the notebook's own markdown cells, plus
whatever cleaning/normalization was applied (pixel scaling, feature standardization).

**4.x.4 Shared architecture** — state the exact layer-by-layer architecture and the
tensor shape after every layer (slide 22's shape-trace pattern), once per app, since
it's identical across all 3 implementations by construction. Include the shape-trace
diagram/table here.

**4.x.5 Three implementations** — one short subsection each:
- *From scratch*: which functions were written (`conv2d`, `relu`, `max_pool2d`,
  `flatten`, `linear`, loss, backward, update), and one paragraph on what the manual
  backward pass computes in words (same "explain the gradient, don't just show code"
  standard assignment_03's DL section used).
- *Keras*: the `Sequential`/layer definition, `compile()`/`fit()` call, and what
  disappeared from the student's own code relative to scratch (slide 12's own framing:
  "the convolution and pooling loops have disappeared... because they are
  encapsulated by layers").
- *PyTorch*: the `nn.Module` definition, explicit training loop, and where it sits
  between scratch's full manual loop and Keras's `fit()`.

**4.x.6 Component-by-implementation table** — slide 28's table, filled in for this
app (Data loading / Preprocessing / Model definition / Convolution / Activation /
Pooling / Flatten / Dense / Loss / Gradient / Optimizer / Training loop / Evaluation /
Prediction × Scratch / Keras / PyTorch).

**4.x.7 Training curves & evaluation** — one shared plot with all 3 implementations'
loss (and accuracy) curves; confusion matrix per implementation; the quantitative
table (params, training time, epochs, final loss, test accuracy, precision/recall/F1).
Pull every number from the notebook's own results cells, not re-typed from memory.

**4.x.8 Comparison discussion** — bar charts (accuracy, parameter count, training
time) across the 3 implementations, then the written comparison answering slide 28's
question for *this* app specifically: which implementation hid the most detail, which
gave the most control, and whether accuracy differed meaningfully or (expected case)
converged to comparable numbers because the architecture is identical. Pull this from
the notebook's own written-comparison markdown cell rather than re-deriving it
independently in the report.

### 5. Cross-application comparison
One table, all 3 apps × 3 implementations, primary metric (accuracy for all three apps
here, since none of the three tasks is regression this round) — a transposition of the
three per-app tables from section 4, not new analysis. A second small table aggregating
parameter counts and training times across apps/implementations, since that comparison
is a named deliverable of this assignment (slide 29) in a way it wasn't for
assignment_03.

### 6. Discussion — what the comparison shows
Synthesis across all 3 apps: did the "scratch = most visible, Keras = most convenient,
PyTorch = in between" claim from section 3 actually hold once implemented? Where did
implementations diverge in accuracy or training time, and why (e.g. differing default
optimizer behavior, batch-norm/regularization present in one implementation but not
replicated by hand in scratch, floating-point/vectorization differences). Also address
what changed about the *data* this round relative to what the course has covered so
far (tabular → also image; single dataset → multiple image benchmarks) without
invoking any sizing-multiplier language for diabetes specifically.

### 7. Conclusion
Short — 1 paragraph. What was built, and what was learned about the relationship
between mathematical architecture and its framework-level expression — not a
restatement of section 6.

### 8. Reproducibility
Environment (Python version, key package versions — `numpy`, `tensorflow`/`keras`,
`torch`/`torchvision`, `scikit-learn`, `matplotlib`, OS), `RANDOM_SEED = 42` statement,
and per-app "how to re-run the notebook end-to-end" instructions (`jupyter nbconvert
--execute ...`). Note that the two image datasets download automatically via their
library loaders on first run (no manual Kaggle step), unlike the diabetes CSVs.

---

## What this file deliberately does not cover

- Notebook cell layout, exact architecture width choices and their reasoning, exact
  hyperparameters — that belongs in `diabetes/PLAN.md`, `fashion_mnist/PLAN.md`,
  `cifar10/PLAN.md`, none of which exist yet. Write those before writing notebooks,
  following the cell-by-cell style `assignment_03/<app>/PLAN.md` used (one function,
  one plot, one model, one explanation, per cell).
- Numbers that don't exist yet (metrics, winning implementation, actual row counts,
  actual training times) — filled in from executed notebooks, not estimated here.
- Slide content and deck design — see `SLIDES_REQUIREMENT.md`.

## Repo structure (for reference — constructed alongside this plan)

```
assignment_04/
  PLAN.md                          <- this file (report plan)
  REQUIREMENT.md
  SLIDES_REQUIREMENT.md
  materials/
    intel_sys_dev_slide_04_compare_models_CNN.pdf
    Deep_Learning_CNN_Function_Composition_Tutorial.pdf
  diabetes/
    PLAN.md          <- not yet written (app-specific implementation plan)
    data/            .gitkeep (holds BRFSS per-year CSVs once downloaded)
    notebook/        .gitkeep (holds diabetes.ipynb once written)
    model/           .gitkeep (holds scratch_weights.npz, keras_model.keras,
                                pytorch_model.pt, feature_names.joblib,
                                input_schema.json)
  fashion_mnist/
    PLAN.md          <- not yet written
    data/            .gitkeep (built-in loader downloads here or is left empty)
    notebook/        .gitkeep
    model/           .gitkeep
  cifar10/
    PLAN.md          <- not yet written
    data/            .gitkeep
    notebook/        .gitkeep
    model/           .gitkeep
  report/
    .gitkeep         (holds Assignment_04.md/.tex/.pdf once written)
  slides/
    .gitkeep         (holds the presentation deck once built, per
                       SLIDES_REQUIREMENT.md)
```

No `api/`, `web/`, `mobile/` folders — out of scope per `REQUIREMENT.md`.
