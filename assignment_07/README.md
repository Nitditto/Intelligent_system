# Assignment 07: RNN vs LSTM (5 layers) in PyTorch and Keras

Structure of a 5-layer RNN and a 5-layer LSTM as compositions of elementary functions, then both models in PyTorch and Keras on two datasets: e-commerce review comments (`EC`) and stock prices (`ST`).
Author: Nguyễn Văn Trường (B23DCCE095), individual project.

| Notebook | Content | Status |
|---|---|---|
| `notebook/00_rnn_lstm_concepts.ipynb` | 5-layer RNN and LSTM as function compositions, checked against PyTorch and Keras; gradient flow; exploding gradients and clipping | done (executed) |
| `notebook/ec_reviews.ipynb` | Amazon unlocked-phone reviews: **three-way** sentiment (negative / neutral / positive), RNN vs LSTM, PyTorch and Keras | done (executed) |
| `notebook/st_stocks.ipynb` | S&P 500: **one-week** volatility forecast (mean $\lvert r\rvert$ of the next 5 days), RNN vs LSTM, PyTorch and Keras | done (executed) |

Data: [dataset/README.md](dataset/README.md). Report: `report/Assignment_07.tex` / `.pdf` (20 pages).

## Results (test sets, one run per configuration)

| Run | main metric | second | s/epoch |
|---|---|---|---|
| EC-TF-RNN / EC-PT-RNN | macro-F1 0.614 / 0.605 | acc. 0.829 / 0.830 | 70 / 143 |
| EC-TF-LSTM / EC-PT-LSTM | macro-F1 0.682 / 0.695 | acc. 0.834 / 0.829 | 278 / 188 |
| EC baseline TF-IDF + logistic regression | macro-F1 0.688 | acc. 0.850 | — |
| ST-TF-RNN / ST-PT-RNN | RMSE 0.685 / 0.696 % | R² 0.320 / 0.299 | 20 / 31 |
| ST-TF-LSTM / ST-PT-LSTM | RMSE 0.679 / 0.683 % | R² 0.332 / 0.324 | 86 / 66 |
| ST baseline EWMA 0.94 | RMSE 0.697 % | R² 0.296 | — |

The LSTM wins on the neutral class of EC (F1 0.31–0.35 against 0.10–0.13) but not on accuracy; on ST its edge over the RNN (0.9–1.8 % RMSE) is about as large as the Keras–PyTorch spread. Details and caveats are in the report.

## Run

Set up the environment (Python 3.10–3.13 with TensorFlow support; the results were produced with Python 3.13, TensorFlow 2.21, PyTorch 2.14, CPU only):

```
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download the data (see [dataset/README.md](dataset/README.md)), then from `notebook/` run the notebooks in VS Code / Jupyter (*Run All*) or:

```
jupyter nbconvert --to notebook --execute --inplace 00_rnn_lstm_concepts.ipynb      # seconds
jupyter nbconvert --to notebook --execute --inplace st_stocks.ipynb                  # about 15 min on a CPU
jupyter nbconvert --to notebook --execute --inplace ec_reviews.ipynb                 # about 55 min on a CPU
```

Each run writes `notebook/results/<run_id>.json`, a comparison CSV per dataset and figures to `report/images/`. Then, from `report/`:

```
python make_tables.py                      # table_*.tex and the comparison charts
xelatex Assignment_07 && xelatex Assignment_07
```
