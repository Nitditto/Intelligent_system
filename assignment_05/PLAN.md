# Assignment 05 — Plan

**Authors:**  Lưu Anh Dũng (B23DCDK036)
**Class:** E23CNPM02 | **Course:** Intelligent System Development | **Lecturer:** Assoc. Prof. Dinh Que Tran, Ph.D.

## 1. Brief

- **Data:** 3 datasets from Kaggle — 2 image datasets + 1 diabetes dataset, **~100k
  samples each**, all different from datasets used in assignments 01–04.
- **Part 1 — CNN fundamentals**
  - **TensorFlow/Keras:** (a) a 3-layer and a 5-layer CNN on each of the three datasets;
    (b) comparison, evaluation, visualization.
  - **PyTorch:** (a) a 3-layer and a 5-layer CNN on each of the three datasets;
    (b) comparison, evaluation, visualization.
- **File structure:** `dataset/`, `notebook/`, `report/`.
- **Report:** follows the same structure, and explains everything with **one fixed set of
  term definitions and naming conventions** (§6).

Total: 3 datasets × 2 depths × 2 frameworks = **12 trained models**.
"3 layers" / "5 layers" = number of **convolutional layers**.

## 2. Datasets

### Already used (excluded)

| Assignment | Dataset |
|---|---|
| 01 | Pima Indians Diabetes (768 rows), Vietnam housing |
| earlier | MNIST, CIFAR-10 (per user) |
| 02 | BRFSS2015 `diabetes_012_health_indicators`, VN real estate |
| 03 | BRFSS 2019+2020 (`spandanjit2005/brfss-diabetes-indicator-dataset`), USA real estate, Flipkart reviews |
| 04 | BRFSS 2015–2020 (same source), Fashion-MNIST, CIFAR-10 |

### Chosen for assignment 05

Selection rules: **famous CNN benchmark**, on Kaggle, **download < 300 MB**, ~100k samples
(~52k accepted for GTSRB — no famous unused image set closer to 100k fits the limit), not used before (MNIST, Fashion-MNIST, CIFAR-10 already used; CIFAR-100 excluded as too close to CIFAR-10). Sizes/popularity checked against the Kaggle API (2026-09-26).

| Code | Dataset | Kaggle ref | Download | Kaggle votes / downloads | Input | Samples used | Classes |
|---|---|---|---|---|---|---|---|
| **SV** | **SVHN** — Street View House Numbers, cropped digits (Netzer et al., 2011) | [`sahityasetu/street-view-house-numbers-images`](https://www.kaggle.com/datasets/sahityasetu/street-view-house-numbers-images) | 246 MB (`train_32x32.mat`, `test_32x32.mat`) | 39 / 140 (dataset itself is a standard benchmark) | 32×32 RGB photo | **99,289** (73,257 + 26,032) | 10 (0–9) |
| **GT** | **GTSRB** — German Traffic Sign Recognition Benchmark (Stallkamp et al., IJCNN 2011) | [`harbhajansingh21/german-traffic-sign-dataset`](https://www.kaggle.com/datasets/harbhajansingh21/german-traffic-sign-dataset) | 124 MB (pickles `train.p`, `valid.p`, `test.p` + `signname.csv`) | 43 / 3.9k (dataset itself is a standard benchmark) | 32×32 RGB photo (pre-resized) | **51,839** (34,799 + 4,410 + 12,630) | 43 |
| **DB** | **Diabetes Prediction Challenge** — Kaggle Playground Series S5E12 (Dec 2025) | [`competitions/playground-series-s5e12`](https://www.kaggle.com/competitions/playground-series-s5e12) (join the competition once to download) | ~100 MB (`train.csv`, 700k × 26; exact size to confirm on download, well under 300 MB) | official Kaggle competition (not a user upload) | 24 tabular features: 15 numeric (age, alcohol, physical activity, diet_score, sleep, screen time, bmi, waist_to_hip_ratio, systolic/diastolic bp, heart_rate, cholesterol_total, hdl, ldl, triglycerides), 6 categorical (gender, ethnicity, education, income, smoking, employment), 3 binary history flags | **300,000** stratified sample of the 700,000 labelled rows | 2 (`diagnosed_diabetes`) |

Why this mix: both image sets are **real-world 32×32 RGB photos**, so the same CNN-3/CNN-5
code runs on both, only the output layer changes. They differ in task: SV = 10 digit
classes, balanced-ish, clutter from neighbouring digits; GT = 43 sign classes, strongly
unbalanced (~180 to ~2,000 per class), lighting/blur/occlusion. DB adds large-scale tabular data.

Other famous CNN datasets considered:

| Dataset | Kaggle size | Why not |
|---|---|---|
| MNIST, CIFAR-10, Fashion-MNIST | — | used in earlier assignments |
| CIFAR-100 (`fedesoriano/cifar100`) | 169 MB, 60k | too close to CIFAR-10 (same source images) |
| EMNIST (`crawford/emnist`) | 1.3 GB | over size limit |
| Kuzushiji-MNIST (`anokas/kuzushiji`) | 599 MB | over size limit |
| GTSRB full-res copy (`meowmeowmeowmeowmeow/gtsrb-german-traffic-sign`) | 642 MB | over size limit → use the 124 MB 32×32 pickle copy |
| Tiny ImageNet (`akash2sharma/tiny-imagenet`) | 498 MB | over size limit; 200 classes |
| STL-10 (`jessicali9530/stl10`) | 2 GB | over size limit |
| Cats vs Dogs (`tongpython/cat-and-dog`) | 228 MB, 10k | too small |
| FER-2013 (`msambare/fer2013`) | 63 MB, 35k | fits, famous; small — **first fallback** |
| Rice Image Dataset (`muratkokludataset/rice-image-dataset`) | 230 MB, 75k | fits, popular; too easy (~99%) — second fallback |

Diabetes datasets considered:

| Dataset | Rows | Why not |
|---|---|---|
| Pima (A01), BRFSS 2015–2020 (A02–A04) | — | used before |
| `iammustafatz/diabetes-prediction-dataset` | 100k | too small |
| `mohankrishnathalla/diabetes-health-indicators-dataset` | 100k | too small (and synthetic) |
| `brandao/diabetes` (130 US hospitals) | 101k | too small; target is readmission, not diabetes |
| `kamilpytlak/personal-key-indicators-of-heart-disease` (BRFSS 2022) | 445k | real data and popular (965 votes), but a heart-disease dataset from the BRFSS survey already used — **fallback** |
| QuickDraw (`drbeane/quickdraw-np`) | 39 GB | over size limit |

### Data rules

- Split every dataset **70 / 15 / 15** (train / val / test), stratified, `seed=42`.
  Save indices to `dataset/<code>/splits.npz` so TF and PyTorch use identical samples.
- Images: rescale to `[0, 1]`, no augmentation (keeps the framework comparison fair).
- GT: load `train.p`/`valid.p`/`test.p` (`features` `(N,32,32,3)`, `labels`), pool and re-split 70/15/15; class names from `signname.csv`; macro F1 because classes are unbalanced. SV: `.mat` loaded with `scipy.io.loadmat`, label `10` → `0`.
- DB: only `train.csv` has labels (the competition `test.csv` does not), so all splits come
  from `train.csv`. Drop `id`; one-hot the 6 categorical columns; standardize numeric
  columns with train-only statistics; check class balance in EDA and use class weights if
  needed (same weights in both frameworks). **Note in the report:** Playground data is
  synthetic (generated from a real diabetes dataset), so results describe this benchmark,
  not clinical accuracy.
- Raw files are **gitignored** and downloaded manually; `dataset/README.md` gives links, target paths and CLI commands.
  Each notebook creates `splits.npz` on its first run (committed).

## 3. Models

The dense head is the same for both depths, so conv depth is the only variable.

### Image CNNs (SV and GT: 3×32×32; output 10 for SV, 43 for GT)

| | CNN-3 | CNN-5 |
|---|---|---|
| Conv layers (filters) | 32 → 64 → 128 | 32 → 32 → 64 → 64 → 128 |
| Conv block | Conv 3×3, padding *same* → BatchNorm → ReLU | same |
| MaxPool 2×2 | after conv 1, 2, 3 | after conv 2, 4, 5 |
| Head | GlobalAvgPool → Dropout 0.3 → Dense 256 → ReLU → Dense `n_classes` (softmax) | same |

### Diabetes CNN (tabular → 1-D)

Features are treated as a 1-D sequence `(N, 1, d)` and use **Conv1D** (kernel 3). Tabular
columns have no natural order, so the report states this limitation and adds one MLP
baseline row for reference.

| | CNN-3 | CNN-5 |
|---|---|---|
| Conv1D layers (filters) | 32 → 64 → 64 | 32 → 32 → 64 → 64 → 128 |
| Head | Flatten → Dropout 0.3 → Dense 64 → ReLU → Dense 1 (sigmoid) | same |

No pooling (the sequence is only `d` ≈ 40 features). **Flatten, not global average pooling:** in a table the position *is* the feature (position 0 is always `age`), so averaging over positions would discard which feature a value came from.

### Fairness rules (TF vs PT)

Same architecture and parameter count (printed and asserted), Adam (lr 1e-3), batch 256,
max 20 epochs, early stopping (patience 3 on val loss), same loss, seed and split.
Weights: Glorot uniform + zero bias in both (PyTorch re-initialized to match Keras); BatchNorm momentum 0.9 (Keras) ≡ 0.1 (PyTorch), eps 1e-5; Adam eps 1e-7. Time per epoch = median of epochs 2+ (epoch 1 includes one-time setup).

**Portability:** runs on Windows, Linux and macOS, Python 3.10–3.12. PyTorch device `cuda` → `mps` → `cpu`; TensorFlow uses a GPU when available (Apple `tensorflow-metal`, which pins TF 2.18; NVIDIA on Linux/WSL2), otherwise CPU.

## 4. Evaluation & visualization

Per model:
- Train/val loss and accuracy curves
- Test accuracy, macro precision / recall / F1; DB also ROC-AUC, PR curve, F1 on the positive class
- Confusion matrix
- Parameter count, time per epoch, total training time, inference time per 1k samples, epochs until early stop
- Images: misclassified samples; first-layer filters and feature maps (CNN-3 vs CNN-5)

Per dataset (EDA): class distribution, sample grid (SV, GT); feature histograms,
correlation heatmap, target balance (DB).

Comparison (per-dataset in each notebook; across datasets in the report):
- 12-row results table (dataset × framework × depth)
- Grouped bar charts: accuracy / F1 and training time — CNN-3 vs CNN-5, TF vs PT
- Discussion: does depth help on each dataset? Do the frameworks agree under identical
  settings? Cost vs gain.

Each dataset notebook writes `notebook/results/<run_id>.json` (metrics + history) so
`report/make_summary.py` reads files instead of retraining.

## 5. File structure

```
assignment_05/
├── PLAN.md
├── README.md                     # how to run, environment, results summary
├── requirements.txt              # tensorflow, tensorflow-metal, torch, torchvision, kaggle, …
├── dataset/
│   ├── README.md                 # sources, licences, row counts, splits
│   ├── sv/                       # raw (gitignored) + splits.npz
│   ├── gt/
│   └── db/
├── notebook/
│   ├── sv_svhn.ipynb             # SV only: EDA → TF CNN-3/5 → PT CNN-3/5 → comparison
│   ├── gt_gtsrb.ipynb            # GT only: same sections
│   ├── db_diabetes.ipynb         # DB only: same sections
│   ├── results/                  # <run_id>.json (gitignored; metrics are also in each notebook's §6 table)
│   └── models/                   # saved weights (gitignored)
└── report/
    ├── Assignment_05.tex         # template from assignment_04
    ├── Assignment_05.pdf
    ├── PLAN.md                   # report plan: sections, tables, figure list
    ├── extract_figures.py        # saves every figure embedded in the executed notebooks
    ├── make_summary.py           # 12-run table + cross-dataset charts, read from the notebooks
    └── images/                   # figures from the notebooks, named <run_id>_<plot>.png (committed)
```

**One notebook per dataset — never combined.** Each notebook is self-contained (helper functions are in a cell, no shared module) and runs on its own
(top to bottom) and has the same section order, so the three can be read side by side:

1. **Setup**
   - Imports (TensorFlow, PyTorch, NumPy, pandas, matplotlib, scikit-learn).
   - Fixed seed (42) for Python, NumPy, TF and PT.
   - Device check: `tensorflow-metal` GPU for TF, `mps` for PT.
   - Run IDs for this notebook, e.g. `SV-TF-CNN-3`, `SV-TF-CNN-5`, `SV-PT-CNN-3`, `SV-PT-CNN-5`.
2. **Data**
   - Load the raw file from `dataset/<code>/`.
   - Apply the saved train / validation / test split from `splits.npz` (70/15/15), so TF
     and PT use exactly the same samples.
   - Preprocessing. Images: rescale to `[0, 1]`; channels-last `(N, H, W, C)` for TF,
     channels-first `(N, C, H, W)` for PT. DB: one-hot categoricals, standardize with
     train-only statistics, reshape to `(N, 1, d)` for Conv1D.
3. **EDA**
   - Images: class distribution chart, sample grid per class, image statistics
     (per-channel mean/std).
   - DB: target balance, feature histograms, correlation heatmap, category counts.
   - What each plot implies for modelling (e.g. unbalanced classes → macro F1, class weights).
4. **TensorFlow** — for each of `CNN-3` and `CNN-5`:
   - Build the model and show `model.summary()` (layers, parameter count).
   - Train with early stopping; record time per epoch and total time.
   - Evaluate on test: accuracy, macro precision / recall / F1 (+ ROC-AUC, PR curve for DB).
   - Plots: loss/accuracy curves, confusion matrix, misclassified samples; for images,
     first-layer filters and feature maps.
   - Save metrics + history to `results/<run_id>.json`, figures to `report/images/`.
5. **PyTorch** — the same steps as section 4:
   - Model as `nn.Module`, `DataLoader`, hand-written training loop and early stopping.
   - Assert the parameter count equals the TF model's (fair comparison).
   - Same metrics, plots and `results/<run_id>.json` output.
6. **Comparison (this dataset only)**
   - 4-row table: the four run IDs × accuracy, macro F1, parameters, training time,
     inference time per 1k samples, epochs.
   - CNN-3 vs CNN-5: does extra depth help here, and at what cost?
   - TF vs PT: same result under identical settings? Speed and code differences.
   - Training curves of all four models on one plot.
7. **Conclusion** — 3–5 short takeaways for this dataset.

Every section starts with a method explanation cell (§5.1).

The cross-dataset comparison (all 12 runs) is not a notebook; `report/make_summary.py`
builds it from `results/*.json`.

### 5.1 Explanation cells (method-focused)

Each code cell is preceded by a short markdown cell that explains the method, not the output. Rules:

- **Concise and concrete:** formula + a tiny worked example, e.g. `ReLU = max(0, x): −1.3 → 0, 0.8 → 0.8`;
  `MaxPool 2×2: [[0.1, 0.9], [0.4, 0.2]] → 0.9`; softmax/cross-entropy with actual numbers.
- **Layer by layer:** tables of how each layer changes the tensor shape (e.g. `32×32×3 → 16×16×32 → …`).
- **Small demo cells:** compute one convolution / ReLU / pooling / BatchNorm by hand and check it against the framework.
- **Specific to each dataset:** own examples per notebook (SV: vertical stroke of a digit; GT: diagonal edge of a
  triangular sign, BatchNorm on dark vs bright images, accuracy vs macro F1 with unbalanced classes; DB: one real
  patient row before/after preprocessing, Conv1D across neighbouring features, sigmoid + binary cross-entropy).
- No fixed "What / How / Why" template; no copy-pasted text between notebooks.
- Results are discussed only in §6 Comparison and §7 Conclusion.
- **Figures:** one or two plots per saved image, so each can go straight into the report.

## 6. Terminology & naming convention (used in notebooks and report)

### 6.1 Naming

| Item | Convention | Example |
|---|---|---|
| Dataset code | `SV`, `GT`, `DB` | `SV` |
| Framework code | `TF` (TensorFlow/Keras), `PT` (PyTorch) | `PT` |
| Model name | `CNN-3`, `CNN-5` (number = conv layers) | `CNN-5` |
| Run ID | `<dataset>-<framework>-<model>` | `SV-PT-CNN-5` |
| Result file | `results/<run_id>.json` | `results/SV-PT-CNN-5.json` |
| Figure file | `images/<run_id>_<plot>.png` or `images/<dataset>_<plot>.png` | `images/SV-PT-CNN-5_confusion.png` |
| Splits | *train* / *validation* / *test* (never "dev" or "eval") | |

The same run ID appears in the notebook heading, JSON file, figure file, and report
table/caption, so any number in the report can be traced back to a cell.

### 6.2 Notation (math in the report)

| Symbol | Meaning |
|---|---|
| `N` | number of samples |
| `C, H, W` | channels, height, width of an input (`C×H×W`) |
| `K` | kernel size; `S` stride; `P` padding |
| `F` | number of filters (output channels) of a conv layer |
| `X`, `y`, `ŷ` | input, true label, predicted label |
| `L` | loss function |
| Output size | `H_out = ⌊(H + 2P − K) / S⌋ + 1` |
| Conv parameters | `(K·K·C_in + 1)·F` |

### 6.3 Term definitions (glossary section in the report)

Each term is defined **once** in §3 of the report ("CNN Fundamentals") and then used
with exactly that meaning:

- **Convolutional layer** — learns `F` kernels of size `K×K` sliding over the input; counted in "CNN-3/CNN-5".
- **Kernel / filter** — the learned weight tensor of one output channel.
- **Feature map** — the output of a conv layer for one filter.
- **Stride, padding** — step of the kernel; zeros added at the border (*same* keeps `H, W`).
- **Pooling** — down-sampling (max pooling 2×2 halves `H, W`); has no parameters.
- **Batch normalization** — normalizes activations per batch; speeds and stabilizes training.
- **Activation (ReLU, softmax, sigmoid)** — non-linearity; softmax/sigmoid at the output.
- **Receptive field** — area of the input that affects one output unit; grows with depth.
- **Global average pooling** — averages each feature map to one value before the dense head.
- **Dropout** — randomly zeros units during training to reduce overfitting.
- **Epoch, batch, learning rate, optimizer (Adam), early stopping.**
- **Overfitting / underfitting** — train vs validation gap.
- **Metrics** — accuracy, precision, recall, F1 (macro), ROC-AUC, confusion matrix — each with its formula.
- **Conv1D** — the 1-D version used for DB.

Rules: one term per concept (e.g. always "filter", not alternately "kernel"/"channel" for
the same thing; "validation", not "val" in prose); first use of an abbreviation is
spelled out; every table and figure caption uses the run ID.

## 7. Report outline (`report/Assignment_05.tex`)

Full section-by-section plan with the figure list: [`report/PLAN.md`](report/PLAN.md).
Figures are taken from the executed notebooks (`python report/extract_figures.py`), not re-generated.

1. Cover page (same as assignment_04)
2. Executive summary — 12-run table, 3–4 takeaways
3. CNN fundamentals — terminology & notation (§6), convolution, pooling, BatchNorm,
   receptive field, parameter counting, why depth helps, Conv1D on tabular data
4. Project structure — `dataset/`, `notebook/`, `report/`, naming convention, how to reproduce
5. Datasets — SV, GT, DB: source, size, preprocessing, EDA figures
6. TensorFlow implementation — code structure, CNN-3/CNN-5 summaries, results per dataset
7. PyTorch implementation — same layout as §6
8. Comparison & evaluation — CNN-3 vs CNN-5; TF vs PT (accuracy, time, code verbosity/control); per-dataset discussion
9. Conclusion & limitations
10. Appendix — full glossary, environment, seeds, commands

## 8. Execution order

| Step | Task | Output |
|---|---|---|
| 0 | Kaggle API key (`~/.kaggle/kaggle.json`); branch `assignment-5` | setup |
| 1 | Scaffold folders, `requirements.txt`, `.gitignore` entries | structure |
| 2 | Download data manually (`dataset/README.md`); splits created by the notebooks | `splits.npz` × 3 |
| 3 | Smoke-test all notebooks on real data (`A5_SMOKE=1`) | ✅ done |
| 4 | Full run of `db_diabetes`, `sv_svhn`, `gt_gtsrb` (executed notebooks pushed, commit `9563ae2`) | ✅ done |
| 5 | `report/extract_figures.py`: figures from the notebooks → `report/images/` | ✅ done (67 figures) |
| 6 | Fill each notebook's §6 discussion and §7 conclusion from its results | todo |
| 7 | `report/make_summary.py`: 12-run table + cross-dataset charts | ✅ done |
| 8 | Report LaTeX + PDF following `report/PLAN.md` | ✅ done (15 pages) |
| 9 | README, final check, commit | todo |

**Compute estimate:** 12 runs × ≤20 epochs on ~70k training samples each. On an Apple
Silicon GPU this takes about 20–40 s per epoch for images and a few seconds for DB,
so about 1–2 hours in total.
