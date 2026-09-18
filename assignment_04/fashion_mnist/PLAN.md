# fashion_mnist — Application Plan

Self-contained plan for `fashion_mnist/notebook/fashion_mnist.ipynb`. Follows the
canonical notebook structure, fairness/init conventions, and notebook writing style
defined once in `../REQUIREMENT.md` (§"Canonical notebook structure", §"Model
requirements per app", §"Notebook writing style") — this file states what's specific
to this app (dataset, architecture, scratch-CNN mechanics) rather than repeating
those cross-app rules. Every markdown cell listed below must follow the writing-style
rules (natural, reasoning-first prose, no em dashes, short sentences), even though
this plan's own prose doesn't need to.

## Problem
10-class image classification: identify the clothing category shown in a 28×28
grayscale image. Image data with 2-D spatial structure → the shared architecture is a
**CNN** (unlike `diabetes`, which stays an MLP).

## Dataset
**Fashion-MNIST** — 60,000 training images + 10,000 test images, `28×28`, grayscale
(1 channel), 10 balanced classes: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal,
Shirt, Sneaker, Bag, Ankle boot.

- Load via each framework's built-in loader: `tf.keras.datasets.fashion_mnist.
  load_data()` for the Keras section, `torchvision.datasets.FashionMNIST(...)` for
  the PyTorch section, and the same underlying arrays (e.g. pulled once via the Keras
  loader, then reused as plain NumPy arrays) for the scratch section — **the actual
  images and the actual train/test split must be identical across all 3
  implementations**, not just "the same dataset name." Load once at the top of
  section 1 into NumPy arrays, and have all three implementation sections consume
  that same array pair rather than each calling their own loader independently.
- No manual download, no file committed to `data/` — `data/.gitkeep` stays empty (or
  holds a cached copy if a loader writes one locally, gitignored like other `data/`
  contents).
- This is already a fixed, canonical 60k/10k split (not something this project
  chooses) — use it as-is rather than re-splitting 80/20 like the tabular app.

## Representation
- Pixel values rescaled from `[0, 255]` integers to `[0.0, 1.0]` floats
  (`X.astype("float32") / 255.0`) — the only "no leakage" concern that would normally
  require train-only statistics (e.g. per-channel mean/std) doesn't apply here, since
  a fixed `/255` rescale uses no dataset statistics at all; note this explicitly in
  the notebook rather than silently skipping a step assignment_02/03 usually required.
- Shape: scratch and PyTorch use channel-first `(N, 1, 28, 28)`; Keras's `Conv2D`
  defaults to channel-last `(N, 28, 28, 1)` — state this explicitly where the array is
  reshaped for each implementation, since it's a real (if superficial) difference
  slide 28's component table should capture under "Data loading"/"Preprocessing."
- Labels: integer class indices `0-9`, used directly with a sparse/integer
  cross-entropy loss (`SparseCategoricalCrossentropy` in Keras, `CrossEntropyLoss` in
  PyTorch, manual sparse softmax-CE in scratch) — no one-hot encoding needed, matching
  the lecture's own Keras example (slide 13 uses `SparseCategoricalCrossentropy`).

## Shared architecture (built 3 times: scratch, Keras, PyTorch)
```
28×28×1 → Conv2D(1→32, k=3, pad=1) → ReLU → MaxPool(2)
        → Conv2D(32→64, k=3, pad=1) → ReLU → MaxPool(2)
        → Flatten (64·7·7 = 3136) → Linear(3136→128) → ReLU → Linear(128→10)
```
Shape trace: `(B,1,28,28) → (B,32,28,28) → (B,32,14,14) → (B,64,14,14) →
(B,64,7,7) → (B,3136) → (B,128) → (B,10)`.

- Same channel widths (32, 64) and dense width (128) as `cifar10`'s architecture —
  deliberately kept parallel so the two image apps' component tables and comparison
  charts read as a matched pair, differing only in input channels (1 vs. 3) and
  spatial size (28 vs. 32).
- Batch size, epoch count, learning rate: fixed once in this notebook's "Shared
  architecture definition" section, then reused unchanged by all 3 implementations.
  60,000 training images is small enough that a modest epoch count (single digits) is
  enough to see meaningful training curves without excessive wall-clock time even for
  the scratch implementation.

### Scratch (NumPy only) — the hard part, planned in full

This is the assignment's central deliverable and the first CNN (not MLP) scratch
implementation in this project — every gradient below must be derivable by the
person writing it, not copied shape-correct from memory. Plan the actual math here,
the same standard `diabetes/PLAN.md` set for the MLP backward pass.

**Implementation note, decided after benchmarking during the build**: the naive
`for i, for j` loop below (materials' own teaching version) times each output
position with a small `einsum`. Benchmarked directly on this app's real layer sizes
at batch size 256, the second conv layer's forward pass alone took ~2.1s/batch, which
would put a full epoch at 60,000 real training images in the tens of minutes. An
`im2col` reformulation (extract every receptive-field patch as one strided view via
`np.lib.stride_tricks.as_strided`, flatten to a `(N*Ho*Wo, C_in*k*k)` matrix, and
replace the per-position loop with one matrix multiply against the reshaped kernel)
computes the exact same forward values in ~0.05s/batch on the same layer, roughly a
40x speedup, verified correct to 7 significant figures against a numerical gradient
check on `dK`, `dX`, and `db`. This is not a different algorithm, just the same
patch-dot-product math computed for every output position at once instead of one
position at a time in a Python loop — say so explicitly in the notebook's markdown
rather than silently swapping implementations, since a reader comparing this
notebook's code against the lecture's own `conv2d` slide should understand why the
code looks structurally different despite computing the same thing. The backward
pass needs a matching `col2im`-style scatter-accumulate step (looping only over the
`k*k` kernel offsets, not over every output position, since that loop is cheap and a
strided *view* can't be used for the accumulation itself: overlapping writes into a
view would overwrite rather than sum). Use `im2col`/`col2im` in the actual notebook
code; the loop-based formulas below remain the reference for *what* the operation
computes and why, which is unchanged.

**Forward pass, one conv layer** (single image for clarity; batch is an outer loop or
a vectorized leading dimension in the real implementation). Input `Xp` is the
*padded* input, shape `(C_in, H+2P, W+2P)`; kernel `K` has shape
`(C_out, C_in, k, k)`; bias `b` has shape `(C_out,)`. For every output channel `f`
and every output position `(i,j)`:
```
patch = Xp[:, i:i+k, j:j+k]                       # shape (C_in, k, k)
Y[f, i, j] = sum(K[f] * patch) + b[f]              # elementwise multiply + sum
```
This *is* the lecture's own `conv2d` code (materials slide 9 / tutorial §9.2) — the
outer `(i,j)` loop is plain Python, the `K[f] * patch` reduction is one vectorized
NumPy op. **Cache every `patch` extracted during the forward pass** (or recompute it
from the cached `Xp` during backward) — the backward pass below needs it.

**Backward pass, same conv layer** — given `dY` (shape `(C_out, Ho, Wo)`, the
upstream gradient `∂L/∂Y`), the key insight to state in a markdown cell before the
code: **backward reuses the exact same `(i,j)` loop as forward.** At each `(i,j)`,
forward turned one `patch` into one output value per filter; backward takes that
output value's gradient and does the reverse — spreads it back out over the patch's
shape, once into the kernel gradient and once into the input gradient:
```
for each output position (i, j):
    patch  = Xp[:, i:i+k, j:j+k]                   # same patch forward cached
    dY_ij  = dY[:, i, j]                            # shape (C_out,)

    db                    += dY_ij                              # (C_out,)
    dK                    += dY_ij[:, None, None, None] * patch[None, :, :, :]
                                                                  # (C_out,C_in,k,k)
    dXp[:, i:i+k, j:j+k]  += tensordot(K, dY_ij, axes=([0], [0]))
                                                                  # contract over f:
                                                                  # (C_in,k,k), scattered
                                                                  # (accumulated, not
                                                                  # overwritten — patches
                                                                  # from neighboring
                                                                  # output positions
                                                                  # overlap) into dXp
dX = dXp[:, P:-P, P:-P]   # crop the padding back off, if P > 0
```
In words, for the report/slides: `dK` asks "how much would increasing this kernel
weight have increased the loss, given every patch it was multiplied against and how
much each of those outputs mattered" (a sum of outer products); `dX` asks "how much
did each input pixel's value affect the loss, given every output position whose
receptive field included it" (a sum of contributions, since each interior pixel
participates in multiple output positions — this overlap-accumulation is *exactly*
what "weight sharing" (tutorial §9.5) means from the gradient's point of view).

**Max pooling forward/backward:**
```
forward:  for each pooling window, Y[c,i,j] = max(window); cache argmax_index[c,i,j]
backward: dX initialized to zero; for each (c,i,j),
          dX[c, i*stride + argmax_row, j*stride + argmax_col] += dY[c,i,j]
          (every other position in that window gets zero gradient — the max
          function's derivative is 1 at the maximizing input and 0 everywhere else)
```
State this explicitly in its own markdown cell before the code — an off-by-one in
the argmax→input-index mapping is the most common silent bug in a from-scratch
pooling layer (it won't crash; it will just train a network with wrong gradients).

**Flatten**: a pure reshape forward (`(C,H,W) → (C·H·W,)`), so its backward is also
just a reshape of `dY` back to `(C,H,W)` — no computation, but implement it as an
explicit `flatten_backward` function anyway (not an inline reshape) so the backward
composition stays uniform (every forward function has a matching named backward
function).

**Dense/linear and loss**: `linear`/`linear_backward` — same as `diabetes/PLAN.md`'s
MLP case. `softmax_cross_entropy(logits, y_true)`: forward is `softmax(logits)` then
cross-entropy against the integer label; backward uses the standard combined-gradient
shortcut `softmax(logits) - one_hot(y)` — derive this once in a markdown cell (same
kind of algebraic collapse as the MLP's `dZ3 = ŷ - y`, generalized from binary
sigmoid+BCE to multi-class softmax+CE) rather than computing the softmax Jacobian and
the cross-entropy gradient as two separate steps.

**Full backward composition** (the order matters — this is the concrete instance of
`F(x) = f_3∘f_2∘f_1(x)`'s chain rule, tutorial §7, applied to this exact
architecture):
```
dlogits              = softmax_cross_entropy_backward(logits, y)
dA_dense2, dW2, db2  = linear_backward(dlogits, cache_dense2)
dA_relu2             = dA_dense2 * relu_derivative(cache_relu2)       # dense1's ReLU
dA_dense1, dW1, db1  = linear_backward(dA_relu2, cache_dense1)
dflat                = flatten_backward(dA_dense1, pre_flatten_shape)
dpool2               = max_pool2d_backward(dflat, cache_pool2)
drelu_conv2          = dpool2 * relu_derivative(cache_conv2_pre_relu)
dconv2, dK2, db2c    = conv2d_backward(drelu_conv2, cache_conv2)
dpool1               = max_pool2d_backward(dconv2, cache_pool1)
drelu_conv1          = dpool1 * relu_derivative(cache_conv1_pre_relu)
_, dK1, db1c         = conv2d_backward(drelu_conv1, cache_conv1)      # input grad
                                                                        # not needed —
                                                                        # nothing is
                                                                        # upstream of
                                                                        # the input
```
Write this composition as one `backward(y, caches, params)` function that calls each
named backward function in this exact reverse order — mirroring `forward`'s call
order exactly reversed, which is worth stating explicitly as the general principle
this whole derivation is an instance of.

**Gradient check (mandatory, run before trusting any training result):** on a tiny
synthetic input (e.g. 2 images, small enough for `conv2d`'s Python loop to run near-
instantly — a 4×4 or 6×6 input, not the real 28×28 dataset), compare
`backward()`'s analytic gradient for a handful of entries — at least one weight from
`K1`, `K2`, `W1` (dense), and `W2` (dense) — against the numerical central-difference
gradient `(L(θ+ε) - L(θ-ε)) / (2ε)`. Do this in one dedicated cell, before the mini-
batch training loop. This is the single most important correctness check for a
from-scratch CNN: shape-compatible code that trains and produces a plausible-looking
loss curve can still have a wrong `conv2d_backward` (e.g. swapped axes in the
`tensordot`, or scattering into the wrong `dXp` slice) — a gradient check catches
that; a loss curve alone does not.

**Functions to implement**: `conv2d`, `conv2d_backward`, `max_pool2d`,
`max_pool2d_backward`, `relu`/`relu_derivative`, `flatten`/`flatten_backward`,
`linear`/`linear_backward`, `softmax_cross_entropy`, `forward(X, params)` (returns
logits + all caches), `backward(y, caches, params)` (returns a gradients dict, per
the composition above), `numerical_gradient_check(...)`.

Mini-batch training loop, same structure as `diabetes/PLAN.md`'s, with loss/accuracy
recorded per epoch.

**If wall-clock time is a real problem despite the vectorized-inner-loop approach**,
the fallback is to reduce the epoch count uniformly for all 3 implementations (never
just for scratch) — dataset size stays fixed, since shrinking only the scratch
implementation's data would break the fairness rule.

### Keras
`tf.keras.Sequential([Conv2D(32,3,padding="same",activation="relu"), MaxPooling2D(2),
Conv2D(64,3,padding="same",activation="relu"), MaxPooling2D(2), Flatten(),
Dense(128,activation="relu"), Dense(10)])`, compiled with
`SparseCategoricalCrossentropy(from_logits=True)` (matching lecture slide 13's own
example almost verbatim), trained with `model.fit(...)` at the fixed batch
size/epoch count.

### PyTorch
`nn.Module` with `self.features = nn.Sequential(nn.Conv2d(1,32,3,padding=1),
nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(32,64,3,padding=1), nn.ReLU(),
nn.MaxPool2d(2))` and `self.classifier = nn.Sequential(nn.Flatten(),
nn.Linear(64*7*7,128), nn.ReLU(), nn.Linear(128,10))` — same structural split as
lecture slide 14's `SmallCNN`-style example. `nn.CrossEntropyLoss()` (expects raw
logits, no manual softmax). Explicit training loop, same batch size/epochs.

## Component-by-implementation table
| Component | Scratch | Keras | PyTorch |
|---|---|---|---|
| Data loading | Keras loader → shared NumPy arrays | `fashion_mnist.load_data()` | `torchvision.datasets.FashionMNIST` (or shared arrays wrapped in a `Dataset`) |
| Preprocessing | manual `/255`, channel-first reshape | `/255`, channel-last (default) | `/255`, channel-first |
| Model definition | explicit functions + a params dict | `Sequential([...])` | `nn.Module` subclass |
| Convolution | `conv2d()` (manual, vectorized inner loop) | `Conv2D(...)` | `nn.Conv2d(...)` |
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
Follows `REQUIREMENT.md`'s canonical 6-section structure.

**1. Setup & data**
1. Markdown: title, problem statement, and what this notebook builds (CNN,
   3 implementations, why a CNN and not an MLP for this data) — written self-
   contained, in the notebook's own words; do not link to or assume the reader has
   this plan file, or the `diabetes`/`cifar10` notebooks, open.
2. Imports (`numpy`, `matplotlib`, `tensorflow`/`keras`, `torch`, `torchvision`,
   `sklearn.metrics`) — separate framework imports from the numpy-only imports the
   scratch section restricts itself to.
3. `RANDOM_SEED = 42` (numpy, `tf.random`, `torch.manual_seed`).
4. Load Fashion-MNIST once via the Keras loader into `(X_train, y_train), (X_test,
   y_test)` NumPy arrays — this is the shared array pair every implementation below
   reuses.
5. Markdown: "Data understanding."
6. `.shape`, `.dtype`, pixel value range (`min`/`max`) before normalization.
7. Class-balance bar chart across the 10 classes (train and test) + markdown
   Observation/Interpretation.
8. Sample-image grid: a few examples per class (e.g. 10×5 grid) + markdown
   Observation/Interpretation — this is the "understand the data" visualization slide
   21's own MNIST framing implies but never shows.
9. Pixel-intensity histogram (aggregate, before normalization) + markdown
   Observation/Interpretation.

**2. Cleaning & representation**
10. Markdown: "Representation."
11. Normalize to `[0,1]`, cast to `float32`.
12. Reshape into channel-first `(N,1,28,28)` (scratch/PyTorch) and note the
    channel-last variant Keras will use.
13. Shape sanity-check cell (`X_train.shape`, `X_test.shape`, confirm 60,000/10,000).

**3. Shared architecture definition**
14. Markdown: architecture diagram + shape trace (as in this plan's §Shared
    architecture) + fixed batch size/epoch count/learning rate for this notebook.

**4. Implementation 1 — scratch (NumPy)** — the most granular section in this
notebook; each function below gets its own cell so the backward composition can be
checked function-by-function, not as one opaque block.
15. Markdown: "Implementation 1: from scratch (NumPy only)" + restate the
    `f_conv → f_relu → f_pool → ... → f_linear` composition this section implements
    (tutorial §7/8's framing, made concrete for this architecture).
16. `conv2d(x, kernel, bias, stride, padding)` — the forward function from this
    plan's §Scratch derivation.
17. `conv2d_backward(dY, cache)` — markdown cell first, restating in words what `dK`
    and `dX` each represent (this plan's §Scratch "in words" paragraph), then the
    code.
18. `max_pool2d(x, size, stride)` and `max_pool2d_backward(dY, cache)` — markdown
    cell explaining the argmax-routing rule before the code (a common silent-bug
    source, per this plan's §Scratch section).
19. `relu`/`relu_derivative`, `flatten`/`flatten_backward`, `linear`/
    `linear_backward` — the components shared with the MLP case.
20. `softmax_cross_entropy(logits, y_true)` + its combined gradient shortcut —
    markdown deriving the `softmax(logits) - one_hot(y)` result before using it.
21. Weight init: `K1, b1, K2, b2` (conv kernels/biases) and `W1, b1_dense, W2,
    b2_dense` (dense weights/biases), He-style, seeded.
22. `forward(X, params)`: composes `conv2d → relu → max_pool2d → conv2d → relu →
    max_pool2d → flatten → linear → relu → linear`, returns `(logits, caches)`.
23. `backward(y, caches, params)`: composes every backward function above in the
    exact reverse order set out in this plan's §Scratch "Full backward composition"
    block; returns a gradients dict.
24. **Gradient-check cell**: on a tiny synthetic input (per this plan's §Scratch
    section — not the real dataset), compare `backward()`'s analytic gradient
    against a numerical central-difference gradient for at least one entry each of
    `K1`, `K2`, `W1`, `W2`; assert close agreement before proceeding. Do not skip
    this cell.
25. Mini-batch training loop; `loss_history`/`acc_history` per epoch.
26. Predict on test set, compute Accuracy/Precision/Recall/F1 (macro-averaged, 10
    classes) + confusion matrix.
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
  constant across all 3 implementations.
- Whether scratch training time is acceptable at the fixed epoch count — if not,
  reduce epochs uniformly for all 3 implementations (see §Scratch fallback above),
  not the dataset size.
- Whether plain SGD or an Adam-style update is used for the scratch model's
  parameter updates (cell 25) — must be stated explicitly, same open question as
  `diabetes/PLAN.md`'s.
