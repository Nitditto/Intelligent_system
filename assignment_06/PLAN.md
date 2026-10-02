# Assignment 06 — Plan

**Authors:** Nguyễn Văn Trường (B23DCCE095)
**Class:** E23CNPM02 | **Course:** Intelligent System Development | **Lecturer:** Assoc. Prof. Dinh Que Tran, Ph.D.
Requirements and acceptance checklist: [REQUIREMENT.md](REQUIREMENT.md).

## 1. Scope

The plan delivers requirements R1–R7 of [REQUIREMENT.md](REQUIREMENT.md) in four parts:

- **Part 0 — Concepts** notebook: functions and operators of an RNN, checked numerically.
- **Part 1 — RNN from scratch** (NumPy) on both datasets.
- **Part 2 — RNN in Keras** on both datasets.
- **Part 3 — RNN in PyTorch** on both datasets.

Datasets: **EC** (service process in e-commerce) and **ST** (stock).

Core runs: 2 datasets × 3 implementations (`SC` scratch, `TF` Keras, `PT` PyTorch) = **6**, all with the
same single-layer vanilla RNN. Extras (Keras + PyTorch only): LSTM and GRU on both datasets = **8 more**.
Scratch LSTM/GRU is *not* built; the concepts notebook only walks one LSTM step by hand and checks it against PyTorch.

## 2. Datasets

Selection criteria: time-related; about **430k windows** per dataset (above the 300k samples of the previous assignment, still moderate so later assignments can grow);
raw download under ~100 MB; not used in any other assignment. Sizes and file names were confirmed after download (see `dataset/README.md`).

Rejected: Olist (only ~100k orders, ~15k hourly steps), single-ticker stock data (~1.2k rows), and anything with millions of rows
(Rees46 / e-commerce clickstream: GBs; Favorita: 125M rows).

### 2.1 EC — e-commerce service process

**Online Retail II (UCI)** — [`mashlyn/online-retail-ii-uci`](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci), ~95 MB CSV, 1,067,371 invoice lines (checked),
UK online retailer, 2009-12 → 2011-12 (`Invoice, StockCode, Quantity, InvoiceDate, Price, Customer ID, Country`).
Service-process view: every invoice is an *order*; invoices starting with `C` are *cancellations / returns*. Per product and day we
build the order-handling flow, not just sales.

Pooled per-product **daily** series for the **top 750 products by order count** over the **604 days on which the shop had sales** (it is closed on Saturdays and over the Christmas break, so closed days are *dropped*, not zero-filled; `gap` = calendar days since the previous open day is a feature). After removing 34k exact duplicate lines and excluding cancellations from the order counts: 750 × (604 − 29) ≈ **431k windows**.

| Feature (per product, per day) | Meaning |
|---|---|
| `units` | units ordered, `log1p` (**target at t+1**) |
| `orders` | distinct invoices containing the product |
| `customers` | distinct non-null customer IDs (22% of lines have no ID) |
| `cancel_units` | units in cancellation invoices (returns), `log1p` |
| `price_mean` | mean unit price (log) |
| `gap` | calendar days since the previous open day |
| `dow_sin/cos`, `doy_sin/cos` | calendar encodings (weekday, day of year) |

Task: window of **28 days** → next-day `units` (many-to-one regression). Baselines: persistence (`y_t = y_{t-1}`) and seasonal naive (`y_t = y_{t-6}`, one open-day week).
Lagged features only; the cancellation of a day's invoices is known the same day, so no leakage into t+1.

### 2.2 ST — stock

**S&P 500 stock data** — [`camnugent/sandp500`](https://www.kaggle.com/datasets/camnugent/sandp500), `all_stocks_5yr.csv`, ~30 MB, 619k daily rows,
505 tickers, 2013-02 → 2018-02 (`date, open, high, low, close, volume, Name`).

Pooled over the **350 tickers among the 470 that have the full 1,259-day history, ranked by mean dollar volume** (the other 35 have 44–1,258 days; missing `open/high/low` values, 11 rows, are dropped): 350 × (1,259 − 31) ≈ **430k windows**.
One model for all tickers.

| Feature (per day, per ticker) | |
|---|---|
| `ret` | log return `ln(close_t / close_{t-1})` (**target at t+1**) |
| `hl_range` | `(high − low) / close` |
| `oc_ret` | `ln(close / open)` |
| `vol_z` | volume z-scored with the ticker's train-period statistics |

Task: window of **30 days** → next-day `ret` (many-to-one regression); directional accuracy derived from the sign.
Baselines: predict 0 and previous-day return. **Expected outcome:** daily returns are close to a random walk, so a
test R² near 0 and directional accuracy ~50–53% are the expected, correct outcome; the point is a *fair implementation comparison*,
not a trading strategy. A price-level forecast for one ticker (e.g. AAPL) is a visual demo only.

### 2.3 Data rules

- **Chronological split 70 / 15 / 15.** EC: by calendar day. ST: by calendar date (same cut dates for all tickers). Indices saved to `dataset/<code>/splits.npz`.
- Scalers (z-score) fitted on the **train period only**; targets never cross a split boundary (a validation window may *look back* into train).
- Seed 42. No shuffling across time; shuffling windows *within train* during training is allowed.
- Raw files gitignored; `dataset/README.md` has links, paths and CLI commands; each notebook writes `splits.npz` on first run (committed).
- Windows are built once by one function; the identical arrays `(N, T, F)` go to scratch, Keras and PyTorch.

## 3. Models

### 3.1 Core model (all three implementations)

`x ∈ ℝ^{T×F}` → single-layer vanilla RNN (hidden `H = 32`, `tanh`) → last hidden state `h_T` → Dense(1) → ŷ.

| | EC | ST |
|---|---|---|
| T (window) / F (features) | 28 / 10 | 30 / 4 |
| H | 32 | 32 |
| Parameters `F·H + H·H + H + H + 1` | 1,409 | 1,217 |
| Loss | MSE on z-scored target | same |

Recurrence: `h_t = tanh(x_t W_xh + h_{t-1} W_hh + b)`, `h_0 = 0`; output `ŷ = h_T W_hy + b_y`.

### 3.2 Scratch implementation (NumPy)

- Hand-written forward pass storing `h_t` for every step; **BPTT** written out explicitly (`dh_t = dL/dh_t + dh_{t+1}·W_hhᵀ ⊙ (1−h²)`).
- **Gradient clipping** by global norm (1.0), also used in Keras (`clipnorm`) and PyTorch (`clip_grad_norm_`).
- Hand-written **Adam**, mini-batches (vectorised over batch, loop over time only).
- Verified by (a) a numerical gradient check on a tiny batch, (b) forward equality with Keras and PyTorch on identical weights.
- Demo cell: gradient norm `‖∂L/∂h_t‖` vs `t` for T = 10, 50, 100 → vanishing gradient, motivates LSTM/GRU.

### 3.3 Keras / PyTorch

| | Keras | PyTorch |
|---|---|---|
| Layer | `SimpleRNN(32)` (+ `LSTM`, `GRU` extras) | `nn.RNN(F, 32, batch_first=True)` (+ `nn.LSTM`, `nn.GRU`) |
| Training | `model.fit` + `EarlyStopping` | hand-written loop + early stopping |
| Init | weights copied from the scratch initialisation (Glorot input, orthogonal recurrent, zero bias) | same |

Bias note: `nn.RNN` has two biases (`b_ih`, `b_hh`); Keras and scratch have one. In PyTorch `b_hh` is set to 0 and frozen so the **parameter counts match and are asserted**.

### 3.4 Fairness rules

Same arrays, same initial weights, Adam (lr 1e-3, eps 1e-7), batch 128, max 30 epochs, early stopping patience 5 on validation loss, clip-norm 1.0, seed 42.
Time per epoch = median of epochs 2+. Scratch is expected to be slowest; reported as measured.
Portability: PyTorch device `cuda → mps → cpu`; TF GPU only on Linux/WSL2/Apple.

### 3.5 Extras

Per dataset in Keras and PyTorch: LSTM(32) and GRU(32) with identical settings; parameter table (RNN : GRU : LSTM = 1 : 3 : 4 recurrent blocks).
Optional, outside the delivered runs: window-length ablation T ∈ {6, 12, 28} on EC, to show where vanilla RNN memory runs out.

## 4. Evaluation & visualization

Per run:
- Train/validation loss curves.
- RMSE, MAE (original units), R²; ST also directional accuracy and the information coefficient (corr of ŷ and y).
- Forecast vs actual on a test slice (EC: one product, 8 weeks; ST: one ticker, 120 days) and residual histogram.
- Parameters, time per epoch, total time, inference time per 1k windows, epochs until stop.
- Scratch only: gradient-check table, per-step gradient-norm plot.

EDA per dataset: series plot, weekly/yearly seasonality and cancellation share (EC); return distribution, volatility clustering, autocorrelation of returns vs squared returns (ST); split boundaries drawn on the series.

Comparison (notebook §7): runs × metrics table against baselines; scratch vs Keras vs PyTorch curves on one plot; RNN vs LSTM vs GRU.
The report tables are generated by `report/make_tables.py` from `notebook/results/*.json`.

## 5. File structure

```
assignment_06/
├── REQUIREMENT.md
├── PLAN.md
├── README.md
├── requirements.txt
├── dataset/
│   ├── README.md                    # sources, paths, CLI commands
│   ├── ec/                          # Online Retail II raw (gitignored) + splits.npz
│   └── st/                          # S&P 500 raw (gitignored) + splits.npz
├── notebook/
│   ├── 00_rnn_concepts.ipynb        # Part 0: functions/operators, recurrence, BPTT, gates, tiny demos
│   ├── ec_retail.ipynb               # EC: EDA → scratch → Keras → PyTorch → comparison
│   ├── st_sp500.ipynb               # ST: same sections
│   ├── results/                     # <run_id>.json (gitignored)
│   └── models/                      # saved weights (gitignored)
└── report/
    ├── Assignment_06.tex / .pdf     # own LaTeX report
    ├── make_tables.py               # tables from results/*.json
    └── images/                      # <run_id>_<plot>.png
```

### 5.1 `00_rnn_concepts.ipynb` (requirement R1)

1. **Sequences as data:** `(N, T, F)` tensors, sliding windows on a toy series, many-to-one vs many-to-many.
2. **Operators:** matrix product `xW`, element-wise product `⊙`, `tanh`, `sigmoid`, softmax, each with a small worked example checked with NumPy; derivatives `tanh' = 1−tanh²`, `σ' = σ(1−σ)`.
3. **One RNN step by hand**, then unrolling for T = 3; weight sharing (parameter count independent of T).
4. **BPTT** derivation on T = 3 and a numerical check.
5. **Vanishing / exploding gradients:** product of Jacobians, spectral radius of `W_hh` vs gradient norm plot; clipping.
6. **LSTM and GRU:** gate equations, one step by hand, cell state as an additive path; check against `torch.nn.LSTMCell` / `GRUCell`.
7. **Same layer, three names:** table mapping scratch ↔ Keras ↔ PyTorch (shapes, bias, gate order).

### 5.2 Dataset notebooks (same order)

1. **Setup:** imports, seed, devices, run IDs.
2. **Data:** load raw, build features, chronological split, windows, scalers.
3. **EDA:** see §4.
4. **Scratch:** model, gradient check, training, evaluation, plots (`EC-SC-RNN`).
5. **Keras:** `SimpleRNN` core, then LSTM/GRU extras (`EC-TF-RNN`, `EC-TF-LSTM`, `EC-TF-GRU`).
6. **PyTorch:** same (`EC-PT-RNN`, …).
7. **Comparison:** table, baselines, curves, forecast overlay, discussion.
8. **Conclusion:** 3–5 takeaways.

Each code cell is preceded by a method-explanation cell with a worked numeric example (concise, dataset-specific, no copy-pasted text between notebooks). Results are discussed only in §7–8.

## 6. Terminology & naming

| Item | Convention | Example |
|---|---|---|
| Dataset code | `EC`, `ST` | `EC` |
| Implementation code | `SC` (scratch), `TF` (Keras), `PT` (PyTorch) | `PT` |
| Model name | `RNN`, `LSTM`, `GRU` | `LSTM` |
| Run ID | `<dataset>-<impl>-<model>` | `ST-TF-GRU` |
| Result / figure | `results/<run_id>.json`, `images/<run_id>_<plot>.png` | `images/EC-SC-RNN_forecast.png` |
| Splits | *train* / *validation* / *test*, chronological | |

Notation: `N` samples, `T` window length, `F` features, `H` hidden size, `x_t ∈ ℝ^F`, `h_t ∈ ℝ^H`, `W_xh ∈ ℝ^{F×H}`, `W_hh ∈ ℝ^{H×H}`, `ŷ` prediction, `L` loss.
Vanilla RNN parameters: `F·H + H·H + H` (+ `H + 1` head); GRU ×3 and LSTM ×4 on those three terms.

## 7. Work order

1. Plan and requirements.
2. `00_rnn_concepts.ipynb`.
3. Scratch RNN and window builder, written once and pasted into both dataset notebooks (no shared module).
4. `st_sp500.ipynb`, then `ec_retail.ipynb`, each first in smoke mode (`A6_SMOKE=1`), then in full.
5. Executed notebooks → `report/make_tables.py` → LaTeX report.

## 8. Risks and decisions

- Window counts: if they land far from ~430k, adjust the number of products / tickers (750 / 350), not the method.
- Environment: TensorFlow, Keras and PyTorch run on CPU on Windows; GPU timings are not covered.
- Stock returns are close to a random walk, so the stock runs are expected to match the trivial baselines; this is a result to measure, not a failure to fix.
