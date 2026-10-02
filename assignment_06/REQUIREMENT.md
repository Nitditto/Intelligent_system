# Assignment 06 — Requirements

This document states **what must be delivered and when it counts as done**. How the work is organised is in [PLAN.md](PLAN.md); what was found is in the report.

## Brief (verbatim from the lecturer)

> **Assignment_06 RNN**
> - Concepts, functions, operators for understanding RNN
> - Two data sets related to time: service process in e-commerce, stock
> - RNN scratch for two datasets
> - RNN Keras for two datasets
> - RNN PyTorch for two datasets

## Deliverables

| # | Requirement |
|---|---|
| R1 | An explanation of the **concepts, functions and operators** needed to understand an RNN: sequence data, hidden state, recurrence, weight sharing, unrolling, back-propagation through time, vanishing/exploding gradients, gradient clipping, LSTM and GRU gates, and the mathematical operators involved (matrix product, element-wise product, `tanh`, `sigmoid`, softmax) |
| R2 | Two **time-related datasets**: one describing a **service process in e-commerce**, one of **stock** prices |
| R3 | An **RNN implemented from scratch** (NumPy only, hand-written forward pass and back-propagation) applied to both datasets |
| R4 | An **RNN implemented in Keras** applied to both datasets |
| R5 | An **RNN implemented in PyTorch** applied to both datasets |
| R6 | A **comparison, evaluation and visualization** of the three implementations |
| R7 | A written **report** (PDF with LaTeX source) and the runnable code (notebooks) |

## Constraints

- **Dataset size.** Each assignment's dataset is larger than the previous assignment's, but not much larger, so that later assignments can keep growing. The previous assignment used up to 300,000 samples; this assignment must be above that and stay moderate (a few hundred thousand more at most). Raw files should be small enough to download and train on a single CPU.
- **Independence.** The assignment is self-contained: its own data, code, environment file and report; datasets that were used in other assignments are not reused.
- **Same problem, three implementations.** Data, split, architecture, starting weights, loss, optimizer and stopping rule are identical across scratch, Keras and PyTorch; only the abstraction level differs.
- **Layout.** `dataset/`, `notebook/`, `report/`; one notebook per dataset, each runnable top to bottom, plus one notebook for the concepts.
- **Time-series validity.** The split is chronological (train, then validation, then test); no random split across time. Scalers are fitted on the training period only.
- **Baselines.** Every model is compared with at least one naive baseline (persistence or equivalent).
- **Metrics.** Error in original units (RMSE, MAE) and a scale-free score; for stocks also directional accuracy. Forecasts are plotted against the true series on the test period.
- **Report.** LaTeX; every term defined once and used consistently; explanations backed by worked numeric examples; results stated as measured, including negative results.

## Acceptance criteria

- [ ] Every formula in the concepts material has a numeric example checked against NumPy or PyTorch.
- [ ] The scratch RNN's analytic gradients match numerical gradients (relative error < 1e-5).
- [ ] With identical weights, the scratch, Keras and PyTorch forward passes agree (max absolute difference < 1e-5) and the parameter counts are equal.
- [ ] Both dataset notebooks run top to bottom without error and save a result file for every run.
- [ ] Each dataset has a results table with baselines, loss curves, forecast plots and a comparison of implementations.
- [ ] The report PDF builds without errors; the README explains setup and execution.
