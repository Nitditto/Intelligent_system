# Assignment 06: RNN from scratch, Keras and PyTorch

Concepts of RNNs, then a vanilla RNN on two time-related datasets in three implementations (NumPy scratch, Keras, PyTorch), plus LSTM/GRU in Keras and PyTorch.

- Requirements: [REQUIREMENT.md](REQUIREMENT.md)
- Plan and conventions: [PLAN.md](PLAN.md)
- Data: [dataset/README.md](dataset/README.md)

| Notebook | Content | Status |
|---|---|---|
| `notebook/00_rnn_concepts.ipynb` | functions, operators, BPTT, gates | done |
| `notebook/ec_retail.ipynb` | e-commerce order flow (Online Retail II), per-product daily | done |
| `notebook/st_sp500.ipynb` | S&P 500 daily returns | done |

Setup: `pip install -r requirements.txt` in a fresh venv.

## Results (test period; details in each notebook §7 and in the report)

| Dataset | Best vs baselines | Finding |
|---|---|---|
| EC (Online Retail II, 432k windows) | RMSE 62.0–63.0 units vs 67.9 (train mean), 81.1 (seasonal naive), 82.7 (persistence) | RNNs learn; scratch / Keras / PyTorch agree within 0.8 %; LSTM/GRU 0.5–1.5 % better |
| ST (S&P 500, 430k windows) | RMSE 0.0158–0.0159 vs 0.0157 (zero return) | no model beats zero return; direction accuracy 48–51 % < 53 % always-up; implementations agree |

## Run

Notebooks run top to bottom from `notebook/` (about 30 min each on a CPU). Quick test: `A6_SMOKE=1` (few series, 2 epochs, outputs to `smoke/` folders).
TensorFlow 2.21 and PyTorch 2.14 were used on Python 3.13 / Windows (CPU).
Report: `report/Assignment_06.tex` (build with XeLaTeX, twice) and `report/Assignment_06.pdf`.
