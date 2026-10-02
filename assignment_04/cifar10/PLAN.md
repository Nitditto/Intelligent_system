# cifar10 — Application Plan

Self-contained plan for `cifar10/notebook/cifar10.ipynb`. Follows the canonical
notebook structure, fairness/init conventions, and notebook writing style defined
once in `../REQUIREMENT.md` (§"Canonical notebook structure", §"Model requirements
per app", §"Notebook writing style"). Mirrors `../fashion_mnist/PLAN.md`'s structure
closely (both are CNN image apps built the same way) — this file states what's
specific to CIFAR-10: RGB input, larger spatial size, and the tutorial's own worked
architecture. Every markdown cell listed below must follow the writing-style rules
(natural, reasoning-first prose, no em dashes, short sentences), written out fresh
for this notebook per this plan's cell 1 note above, even though this plan's own
prose doesn't need to.

## Problem
10-class image classification on natural-scene color photos. Image data with 2-D
spatial structure *and* 3 color channels → CNN, same as `fashion_mnist`, with one
extra dimension (channels) the architecture must account for.

## Dataset
**CIFAR-10** — 50,000 training images + 10,000 test images, `32×32`, RGB (3
channels), 10 balanced classes: airplane, automobile, bird, cat, deer, dog, frog,
horse, ship, truck.

- Load via each framework's built-in loader: `tf.keras.datasets.cifar10.load_data()`
  for the Keras section, `torchvision.datasets.CIFAR10(...)` for the PyTorch section,
  and the same underlying arrays (pulled once via the Keras loader, reused as plain
  NumPy arrays) for the scratch section — same "load once, reuse the identical array
  pair across all 3 implementations" rule as `fashion_mnist/PLAN.md`.
- No manual download, no file committed to `data/`.
- Fixed canonical 50k/10k split — used as-is.

## Representation
- Pixel values rescaled from `[0, 255]` to `[0.0, 1.0]` (`float32`) — same no-leakage
  reasoning as `fashion_mnist`: a fixed `/255` rescale uses no dataset statistics, so
  there's no train-only-fit step required here, unlike diabetes's feature
  standardization.
- Shape: scratch and PyTorch use channel-first `(N, 3, 32, 32)`; Keras defaults to
  channel-last `(N, 32, 32, 3)` — same cross-framework shape difference as
  `fashion_mnist`, now with 3 channels instead of 1.
- Labels: integer class indices `0-9`, sparse/integer cross-entropy loss (same choice
  as `fashion_mnist`, same rationale).

## Shared architecture (built 3 times: scratch, Keras, PyTorch)
This is a direct, unmodified instantiation of the tutorial's own worked example
(`materials/Deep_Learning_CNN_Function_Composition_Tutorial.pdf` §16, "A Complete
Small CNN" / `SmallCNN`), which itself uses a `3×32×32` input — nothing about the
architecture below is invented for this assignment:

```
3×32×32 → Conv2D(3→32, k=3, pad=1) → ReLU → MaxPool(2)
        → Conv2D(32→64, k=3, pad=1) → ReLU → MaxPool(2)
        → Flatten (64·8·8 = 4096) → Linear(4096→128) → ReLU → Linear(128→10)
```
Shape trace (matches the tutorial's own §16 shape progression exactly):
`(B,3,32,32) → (B,32,32,32) → (B,32,16,16) → (B,64,16,16) → (B,64,8,8) → (B,4096) →
(B,128) → (B,10)`.

- Same channel widths (32, 64) and dense width (128) as `fashion_mnist` — the two
  image apps' architectures differ only in input channels (3 vs. 1) and spatial size
  (32 vs. 28, which changes the flattened size: 4096 vs. 3136), keeping their
  comparison charts a matched pair.
- Batch size, epoch count, learning rate: fixed once in this notebook's "Shared
  architecture definition" section, reused unchanged by all 3 implementations.
  50,000 RGB images means somewhat more compute per image than `fashion_mnist`'s
  grayscale 28×28 (3x the input channels, more output positions per conv layer at
  32×32 vs. 28×28) — confirm actual scratch-implementation wall-clock time in cell 25
  below before committing to a specific epoch count in the report; if it's
  meaningfully slower than `fashion_mnist`'s, that gap between the two image apps'
  training times is itself a legitimate, reportable data point (naive from-scratch
  convolution cost scales with input size and channel count) rather than a problem to
  hide.

### Scratch (NumPy only) — same math as `fashion_mnist`, re-verified at 3 channels

**Implementation note**: same `im2col`/`col2im` implementation decision as
`fashion_mnist/PLAN.md` (the naive per-position loop benchmarked ~40x slower on real
layer sizes and was replaced with a strided-view-plus-matmul formulation, verified
against a numerical gradient check) — say so explicitly in this notebook's own
markdown too, not just by implication from sharing the fashion_mnist implementation.

The full forward/backward derivation — `conv2d`'s patch-dot-product forward,
`conv2d_backward`'s "reuse forward's loop, scatter instead of reduce" backward
(`dK[f] += dY_ij ⊗ patch`, `dXp[patch region] += tensordot(K, dY_ij, axes=([0],[0]))`),
`max_pool2d`'s argmax-routing backward, the `softmax(logits) - one_hot(y)` combined
loss gradient, and the full reverse-order backward composition — is exactly the same
math as `fashion_mnist/PLAN.md`'s §Scratch section derives in full. Read that
derivation before implementing this app's scratch model; it is not repeated here.

**What is genuinely different here, and must not be assumed to port over unchanged:**
- `conv2d`'s **first** layer now has `C_in = 3` (not 1): `K1` has shape
  `(32, 3, 3, 3)` instead of `fashion_mnist`'s `(32, 1, 3, 3)`, and the `patch`
  extracted in both forward and backward is `Xp[:, i:i+k, j:j+k]` with a leading
  dimension of 3, not 1. The forward dot product (`sum(K[f] * patch)`) and the
  backward outer-product accumulation (`dK += dY_ij[...,None,None,None] *
  patch[None,...]`) are shape-generic in `C_in` *if written generically in the first
  place* — but this must be verified by actually running the `fashion_mnist`-derived
  functions against a 3-channel input, not assumed. The second conv layer
  (`C_in=32 → C_out=64`) is unchanged in kind from `fashion_mnist`'s second layer.
- Flattened size differs: `64·8·8 = 4096` here vs. `64·7·7 = 3136` in `fashion_mnist`
  — only affects the first dense layer's input width (`Linear(4096→128)`), not the
  backward math itself.
- **Gradient check**: run the same mandatory numerical-gradient-check procedure
  (this plan's cell 24-equivalent, below) on a synthetic 3-channel input this time
  (e.g. 2 images, 6×6, 3 channels) — do not assume the check that passed for
  `fashion_mnist`'s 1-channel `conv2d` implies correctness here; the first
  convolution's channel-summation axis is exactly the part that changed, so it is
  exactly the part most worth re-checking numerically.

### Keras
`tf.keras.Sequential([Conv2D(32,3,padding="same",activation="relu",
input_shape=(32,32,3)), MaxPooling2D(2), Conv2D(64,3,padding="same",
activation="relu"), MaxPooling2D(2), Flatten(), Dense(128,activation="relu"),
Dense(10)])`, `SparseCategoricalCrossentropy(from_logits=True)`, `model.fit(...)` at
the fixed batch size/epoch count.

### PyTorch
`nn.Module` with `self.features = nn.Sequential(nn.Conv2d(3,32,3,padding=1),
nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(32,64,3,padding=1), nn.ReLU(),
nn.MaxPool2d(2))` and `self.classifier = nn.Sequential(nn.Flatten(),
nn.Linear(64*8*8,128), nn.ReLU(), nn.Linear(128,10))` — this is, almost verbatim, the
tutorial's own §16 PyTorch `SmallCNN` class. `nn.CrossEntropyLoss()`. Explicit
training loop, same batch size/epochs.

## Component-by-implementation table
| Component | Scratch | Keras | PyTorch |
|---|---|---|---|
| Data loading | Keras loader → shared NumPy arrays | `cifar10.load_data()` | `torchvision.datasets.CIFAR10` (or shared arrays wrapped in a `Dataset`) |
| Preprocessing | manual `/255`, channel-first reshape | `/255`, channel-last (default) | `/255`, channel-first |
| Model definition | explicit functions + a params dict | `Sequential([...])` | `nn.Module` subclass (mirrors tutorial §16's `SmallCNN`) |
| Convolution | `conv2d()` (manual, 3-channel-aware) | `Conv2D(...)` | `nn.Conv2d(...)` |
| Activation | `relu()` | `activation="relu"` | `nn.ReLU()` |
| Pooling | `max_pool2d()` | `MaxPooling2D(2)` | `nn.MaxPool2d(2)` |
| Flatten | `flatten()` (reshape) | `Flatten()` | `nn.Flatten()` |
| Dense/Linear layer | `linear()` | `Dense(...)` | `nn.Linear(...)` |
| Loss | manual softmax cross-entropy | `SparseCategoricalCrossentropy(from_logits=True)` | `nn.CrossEntropyLoss()` |
| Gradient | manual backprop through conv/pool/dense | autodiff (implicit in `fit`) | `loss.backward()` (autograd) |
| Optimizer | manual `W -= lr * dW` | `optimizer="adam"` | `torch.optim.Adam(...)` |
| Training loop | explicit `for epoch / for batch` | `model.fit(...)` | explicit `for epoch / for batch` |
| Evaluation | manual metric functions | `model.evaluate(...)` or manual | manual (`model.eval()`, no-grad) |
| Prediction | `forward(X_test, params)` | `model.predict(...)` | `model(x_test)` under `torch.no_grad()` |

## Notebook cell-by-cell plan
Follows `REQUIREMENT.md`'s canonical 6-section structure — cell numbering intentionally
mirrors `fashion_mnist/PLAN.md`'s so the two notebooks read as a matched pair.

**1. Setup & data**
1. Markdown: title, problem statement, and what this notebook builds (CNN,
   3 implementations, why a CNN and not an MLP for this data) — written self-
   contained, in the notebook's own words; do not link to or assume the reader has
   this plan file, or the `diabetes`/`fashion_mnist` notebooks, open. In particular,
   this notebook's own conv2d/backward derivation must be written out in its own
   markdown cells (cell 16-17 below) rather than telling the reader to "see
   fashion_mnist" — the two notebooks happen to share the same underlying math
   because the architecture family is the same, but each notebook proves and
   explains it independently.
2. Imports (`numpy`, `matplotlib`, `tensorflow`/`keras`, `torch`, `torchvision`,
   `sklearn.metrics`).
3. `RANDOM_SEED = 42`.
4. Load CIFAR-10 once via the Keras loader into `(X_train, y_train), (X_test,
   y_test)` NumPy arrays — the shared array pair every implementation reuses.
5. Markdown: "Data understanding."
6. `.shape`, `.dtype`, pixel value range before normalization.
7. Class-balance bar chart across the 10 classes (train and test) + markdown
   Observation/Interpretation.
8. Sample-image grid: a few RGB examples per class + markdown
   Observation/Interpretation.
9. Per-channel pixel-intensity histogram (R/G/B separately, before normalization) +
   markdown Observation/Interpretation — the RGB-specific counterpart to
   `fashion_mnist`'s single-channel histogram.

**2. Cleaning & representation**
10. Markdown: "Representation."
11. Normalize to `[0,1]`, cast to `float32`.
12. Reshape into channel-first `(N,3,32,32)` (scratch/PyTorch), note the
    channel-last variant Keras will use.
13. Shape sanity-check cell (`X_train.shape`, `X_test.shape`, confirm 50,000/10,000).

**3. Shared architecture definition**
14. Markdown: architecture diagram + shape trace (matches tutorial §16 exactly) +
    fixed batch size/epoch count/learning rate for this notebook.

**4. Implementation 1 — scratch (NumPy)** — same granularity as `fashion_mnist`'s
notebook (one function per cell), with every markdown explanation written out fresh
for this notebook rather than pointing elsewhere.
15. Markdown: "Implementation 1: from scratch (NumPy only)" + state the
    `f_conv → f_relu → f_pool → ... → f_linear` composition this section implements,
    for this app's own architecture (`3×32×32 → ... → 10`).
16. `conv2d(x, kernel, bias, stride, padding)` — the forward function: at each output
    position, extract the receptive-field patch (now shape `(C_in,k,k)` with
    `C_in=3` on the first layer, `C_in=32` on the second) and take the vectorized
    elementwise-multiply-and-sum against the kernel, plus bias.
17. `conv2d_backward(dY, cache)` — markdown cell first, spelling out in this
    notebook's own words what `dK` (a sum of `dY_ij ⊗ patch` outer products) and `dX`
    (a sum of `tensordot(K, dY_ij)` contributions scattered back into every
    overlapping receptive field) each represent, then the code. Explicitly confirm
    in this cell's markdown that the implementation is correct for `C_in=3` on the
    first call — this is the one place this notebook's math genuinely differs from a
    single-channel CNN.
18. `max_pool2d(x, size, stride)` and `max_pool2d_backward(dY, cache)` — markdown
    cell explaining the argmax-routing rule (forward caches which position in each
    window was the max; backward sends the incoming gradient only to that position)
    before the code.
19. `relu`/`relu_derivative`, `flatten`/`flatten_backward`, `linear`/
    `linear_backward`.
20. `softmax_cross_entropy(logits, y_true)` + its combined gradient shortcut —
    markdown deriving `softmax(logits) - one_hot(y)` before using it.
21. Weight init: `K1` (shape `32×3×3×3`), `K2` (shape `64×32×3×3`), and the two dense
    weight matrices (`W1: 4096×128`, `W2: 128×10`), He-style, seeded.
22. `forward(X, params)`: composes `conv2d → relu → max_pool2d → conv2d → relu →
    max_pool2d → flatten → linear → relu → linear`, returns `(logits, caches)`.
23. `backward(y, caches, params)`: composes every backward function above in exact
    reverse order; returns a gradients dict.
24. **Gradient-check cell**: on a tiny synthetic 3-channel input (e.g. 2 images,
    6×6×3 — not the real dataset), compare `backward()`'s analytic gradient against
    a numerical central-difference gradient for at least one entry each of `K1`,
    `K2`, `W1`, `W2`. Do not skip this cell — it is the one place this notebook can
    directly verify the 3-channel `conv2d_backward` is correct rather than merely
    shape-compatible.
25. Mini-batch training loop; `loss_history`/`acc_history` per epoch.
26. Predict on test set, compute Accuracy/Precision/Recall/F1 (macro-averaged) +
    confusion matrix.
27. Record parameter count and training time.

**5. Implementation 2 — Keras**
28. Markdown: "Implementation 2: TensorFlow/Keras."
29. Build the `Sequential` model matching the shared architecture.
30. `model.summary()` — cross-check parameter count against cell 27.
31. `model.compile(...)`.
32. `model.fit(...)` — same batch size/epochs as cell 25; capture `history` and
    training time.
33. Predict on test set, compute the same metric set as cell 26.

**6. Implementation 3 — PyTorch**
34. Markdown: "Implementation 3: PyTorch."
35. `nn.Module` class definition matching the shared architecture.
36. Instantiate model, `CrossEntropyLoss`, `Adam` optimizer (same LR as Keras where
    possible).
37. Wrap the shared train arrays in a `DataLoader`, same batch size.
38. Explicit training loop; capture per-epoch loss/accuracy and training time.
39. Predict on test set (`model.eval()`, `torch.no_grad()`), compute the same metric
    set as cells 26/33.

**Comparison & visualization**
40. Markdown: "Comparing the three implementations."
41. Component-by-implementation table (this plan's table above).
42. Quantitative comparison table: params, training time, epochs, final training
    loss, test accuracy, precision/recall/F1.
43. Combined training-curve plot: loss (and accuracy) vs. epoch, 3 lines.
44. Confusion-matrix heatmaps, one panel per implementation.
45. Bar chart: test accuracy across the 3 implementations.
46. Bar chart: parameter count and training time across the 3 implementations.
47. Markdown: written comparison — which implementation hid the most detail, which
    gave the most control, whether accuracy converged to comparable numbers.

**Persistence**
48. Save scratch weights (`np.savez`).
49. Save Keras model (`.keras` format).
50. Save PyTorch model (`state_dict`).
51. Save `input_schema.json` (input shape, class names, preprocessing spec).
52. Reload-and-verify cell: reload all 3 artifacts, confirm predictions match the
    in-memory versions.

## Open questions
- Exact batch size / epoch count / learning rate — fixed once in cell 14, held
  constant across all 3 implementations; this app's values are chosen independently
  from `fashion_mnist`'s (CIFAR-10's larger RGB input may need different settings to
  train in reasonable time) — record the reasoning in that cell.
- Whether scratch training time (higher per-image cost than a smaller grayscale
  input, per §Shared architecture above) is acceptable at the fixed epoch count — if
  not, reduce epochs uniformly across all 3 implementations, not the dataset size.
- Whether plain SGD or an Adam-style update is used for the scratch model's
  parameter updates (cell 25) — must be stated explicitly in that cell's markdown.
