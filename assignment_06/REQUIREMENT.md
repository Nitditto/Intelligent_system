# Assignment 06 — Requirements

## Brief (verbatim from the lecturer)

> **Assignment_06 RNN**
> - Concepts, functions, operators for understanding RNN
> - Two data sets related to time: service process in e-commerce, stock
> - RNN scratch for two datasets
> - RNN Keras for two datasets
> - RNN PyTorch for two datasets

## Deliverables

| # | Requirement | Where it is met |
|---|---|---|
| R1 | **Concepts, functions, operators** needed to understand an RNN: sequence data, hidden state, recurrence `h_t = tanh(W_xh x_t + W_hh h_{t-1} + b)`, weight sharing across time, unrolling, BPTT, vanishing/exploding gradients, gradient clipping, LSTM and GRU gates, `tanh` / `sigmoid` / element-wise product / matrix product / softmax, sliding-window supervised framing, many-to-one vs many-to-many | `notebook/00_rnn_concepts.ipynb` and report chapter 2 |
| R2 | Two **time-related** datasets: **(EC)** a service process in e-commerce, **(ST)** stock prices | `dataset/`, PLAN §2 |
| R3 | **RNN from scratch** (NumPy only, hand-written forward pass and BPTT) on both datasets | `ec_*.ipynb` §4, `st_*.ipynb` §4 |
| R4 | **RNN in Keras** on both datasets | `ec_*.ipynb` §5, `st_*.ipynb` §5 |
| R5 | **RNN in PyTorch** on both datasets | `ec_*.ipynb` §6, `st_*.ipynb` §6 |
| R6 | Comparison, evaluation and visualization of the three implementations | each notebook §7, report chapter 5 |

Total: 2 datasets × 3 implementations = **6 core runs** (+ LSTM/GRU extras in Keras and PyTorch, PLAN §3).

## Conventions (assignment 6 is self-contained)

- Folder layout `dataset/`, `notebook/`, `report/`; **one notebook per dataset**, self-contained, same section order.
- The three implementations share the **same data, split, architecture, loss, optimizer, seed** — only the abstraction level changes ("different names do not imply different machine-learning concepts").
- Datasets differ from those of earlier assignments and are **larger than the previous assignment's (300k samples) but not much larger** (~400–450k windows), so later assignments can keep growing (lecturer's requirement).
- Fixed run-ID / figure / result naming (PLAN §6) so any number in the report traces back to a cell.
- Report is LaTeX; explanations are method-focused with worked numeric examples.
- Raw data is gitignored; `dataset/README.md` lists download links and target paths.
- Commit locally only; ask before any `git push`.

## Time-series-specific requirements

- **Chronological split** (train → validation → test in time order). Random splitting leaks the future and is forbidden.
- Scalers fitted on **train only**.
- Every model is compared with at least a **naive baseline** (last value / persistence) — a recurrent net that cannot beat persistence has learned nothing.
- Report error in original units (RMSE, MAE) and relative error (MAPE or sMAPE); for stocks also directional accuracy.
- Forecasts are plotted against the true series on the test period.

## Acceptance checklist

- [ ] Concepts notebook runs and every formula has a numeric example checked against NumPy.
- [ ] Scratch RNN passes a numerical gradient check (relative error < 1e-5).
- [ ] Scratch, Keras and PyTorch forward passes agree on identical weights (max abs diff < 1e-5).
- [ ] Both dataset notebooks run top to bottom without error, results saved to `notebook/results/*.json`.
- [ ] Comparison table + plots in each notebook; cross-dataset table in the report.
- [ ] Report PDF builds; README explains setup and run.
