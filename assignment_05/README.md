# Assignment 05: CNN-3 vs CNN-5 in TensorFlow and PyTorch

Three datasets, two depths (3 and 5 convolutional layers), two frameworks: 12 trained models.
Plan and conventions: [PLAN.md](PLAN.md).

| Notebook | Dataset | Run IDs |
|---|---|---|
| [notebook/sv_svhn.ipynb](notebook/sv_svhn.ipynb) | SVHN (99k images, 10 classes) | `SV-TF-CNN-3`, `SV-TF-CNN-5`, `SV-PT-CNN-3`, `SV-PT-CNN-5` |
| [notebook/gt_gtsrb.ipynb](notebook/gt_gtsrb.ipynb) | GTSRB (52k images, 43 classes) | `GT-TF-CNN-3`, `GT-TF-CNN-5`, `GT-PT-CNN-3`, `GT-PT-CNN-5` |
| [notebook/db_diabetes.ipynb](notebook/db_diabetes.ipynb) | Playground S5E12 diabetes (300k rows) | `DB-TF-CNN-3`, `DB-TF-CNN-5`, `DB-PT-CNN-3`, `DB-PT-CNN-5` (+ `DB-TF-MLP` baseline) |

Each notebook is self-contained and runs top to bottom.

## Setup (Windows, macOS, Linux)

Python **3.10–3.12** (TensorFlow does not support newer versions yet).

```
python -m venv .venv
.venv\Scripts\activate            # Windows
source .venv/bin/activate         # macOS / Linux
pip install -r requirements.txt
```

GPU support:
- **Apple Silicon:** TensorFlow uses the GPU through `tensorflow-metal` (installed automatically); PyTorch uses `mps`.
- **NVIDIA:** PyTorch uses `cuda` if a CUDA build of torch is installed. TensorFlow uses the GPU on Linux/WSL2 only
  (`pip install tensorflow[and-cuda]`); on native Windows it runs on the CPU.
- Otherwise both run on the CPU (slower, same results).

## Data

Download the three datasets yourself and place them as described in [dataset/README.md](dataset/README.md).

## Run

Open a notebook in Jupyter or VS Code and run all cells, or from `notebook/`:

```
jupyter nbconvert --to notebook --execute --inplace sv_svhn.ipynb
```

Quick test first (small subsample, 2 epochs, outputs go to `*/smoke/` folders):

```
set A5_SMOKE=1                    # Windows (cmd)
$env:A5_SMOKE="1"                 # Windows (PowerShell)
export A5_SMOKE=1                 # macOS / Linux
```

## Outputs

- `notebook/results/<run_id>.json`: metrics, training history, timing for each run; `<code>_comparison.csv` per dataset.
- `report/images/<run_id>_<plot>.png` and `report/images/<code>_<plot>.png`: every figure, named by run ID.
- `notebook/models/`: saved models (`.keras`, `.pt`), not committed.
