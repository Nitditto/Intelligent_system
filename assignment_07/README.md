# Assignment 07: RNN vs LSTM (5 layers) in PyTorch and Keras

Structure of a 5-layer RNN and a 5-layer LSTM as compositions of elementary functions, then both models in PyTorch and Keras on two datasets: e-commerce review comments (`EC`) and stock prices (`ST`).

| Notebook | Content | Status |
|---|---|---|
| `notebook/00_rnn_lstm_concepts.ipynb` | 5-layer RNN and LSTM as function compositions, checked against PyTorch and Keras | done (executed) |
| `notebook/ec_reviews.ipynb` | Amazon unlocked-phone reviews: sentiment, RNN vs LSTM, PyTorch and Keras | ready to run |
| `notebook/st_stocks.ipynb` | S&P 500 daily prices: next-day volatility forecast, RNN vs LSTM, PyTorch and Keras | ready to run (volatility forecast) |

Data: [dataset/README.md](dataset/README.md). Report: `report/Assignment_07.tex` / `.pdf`.

## Run

Set up the environment (Python 3.12):

```
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

From `notebook/` with the course venv (Python 3.12, TensorFlow 2.18 + Metal, PyTorch MPS), or open the notebooks in VS Code / Jupyter and *Run All*:

```
jupyter nbconvert --to notebook --execute --inplace ec_reviews.ipynb
jupyter nbconvert --to notebook --execute --inplace st_stocks.ipynb
```
Each run writes `notebook/results/<run_id>.json`, a comparison CSV per dataset, and figures to `report/images/`.
