# Assignment 07 — Report Plan

**Output:** `report/Assignment_07.tex` → `Assignment_07.pdf` (XeLaTeX).
**Author:** Lưu Anh Dũng (B23DCDK036) · Intelligent System Development · Lecturer: Assoc. Prof. Dinh Que Tran, Ph.D. (class code: ask before adding).

This file plans the report: where its structure comes from, what goes in each section, which notebook output feeds each table and figure, and which explanation ("why") each result needs.

## 1. Sources of the structure

| Source | What it decides |
|---|---|
| **Brief (from the user/lecturer)**: two datasets (e-commerce review comments, Kaggle stocks, ~100 MB each); 5-layer RNN structure as a composition of elementary functions + explanation; the same for LSTM; code in PyTorch and Keras; run, evaluate and compare LSTM vs RNN on both datasets; notebooks + report | **The section list** (§3 below): every brief item gets its own section, and §2 maps items to sections |
| Lecturer's *Deep Learning as Function Composition* tutorial (`TAILIEU/Deep_Learning_CNN_Function_Composition_Tutorial.pdf`) | The method inside the sections: write the network as $f_L\circ\dots\circ f_1$; print intermediate shapes; count parameters by hand and compare with the framework; plot train/validation loss; inspect wrong predictions; compare **accuracy + parameters + training time + behaviour**; present a new architecture as *old architecture + a mechanism that fixes a limitation* (LSTM = RNN + gated additive cell) |
| `assignment_06/report/Assignment_06.tex` | **Format only**: preamble (fontspec, booktabs, tabularx, float, listings, tikz, caption), cover-page layout, `\run{}` macro for run IDs, table style, generated `table_*.tex` files. **Do not copy** its author/class (another student) or its section list |
| `assignment_05/report/PLAN.md`, `extract_figures.py`, `make_summary.py` | Workflow: extract figures from executed notebooks, build tables with a script, never type numbers by hand |
| `assignment_06/report/make_tables.py` | Template for `report/make_tables.py` (reads `results/*.json` and `*_comparison.csv`) |

## 2. Requirement → section map

| # | Brief item | Report section | Evidence |
|---|---|---|---|
| R1 | Datasets: e-commerce comments + Kaggle stocks, moderate size | §5 Datasets | size table, EDA figures |
| R2 | 5-layer RNN as a composition of elementary functions, explained | §2, §3 | formulas, TikZ grid, numeric example, parameter count, gradient map |
| R3 | The same for LSTM | §4 | gate table, cell-path derivation, parameter count, gradient map |
| R4 | Code in PyTorch and Keras | §6 | side-by-side listings, weight mapping, verification table |
| R5 | Run and evaluate | §7–§9 | protocol, results tables, curves, confusion/ROC, forecasts |
| R6 | Compare LSTM and RNN on both datasets | §10 | cross-dataset table, accuracy/params/time/behaviour |
| R7 | Notebooks + report | Appendix A | file map, how to reproduce |

## 3. Sections

Target length 18–24 pages. Each section lists its content, its sources and the **why** it must explain. Results are interpreted only in §8–§10.

### Cover page + table of contents
Title *"Assignment 07 — 5-Layer RNN and LSTM as Compositions of Elementary Functions: PyTorch and Keras on Review Sentiment and Stock Volatility"*; course, author, ID, lecturer, date. ToC on page 2.

### Summary (½ page)
- One paragraph: 2 datasets × 2 architectures × 2 frameworks = 8 runs, same data, protocol and initialisation.
- 4 takeaways with numbers (filled from the tables):
  1. EC: LSTM beats RNN by about 1 point of accuracy and 1–3 points of negative-class F1, and the gap grows with review length.
  2. EC: TF-IDF + logistic regression (0.941) still beats both. That is a negative result, and it gets explained.
  3. ST: both beat every baseline (R²_oos ≈ 0.09 vs 0.044 for the 20-day mean). The LSTM is only marginally better, because the useful memory is short.
  4. The LSTM costs 1.8–4.8× more time per epoch. TF and PyTorch agree to within ~0.01.

### §1 Introduction
- Problem: sequence models; why a plain RNN struggles with long sequences; what the LSTM changes.
- Objectives (R1–R7), the requirement → section table (§2 above), and the run-ID convention `<EC|ST>-<TF|PT>-<RNN|LSTM>`.

### §2 Elementary functions and composition *(R2, R3 foundation)*
Source: concepts notebook §1, with its new "why" paragraphs.
- Table: matmul, add, tanh, sigmoid, ⊙, each with formula, tiny example, derivative and **role**.
- Why: depth needs a nonlinearity ($(xW_1)W_2=x(W_1W_2)$); why tanh rather than ReLU in a recurrence ($1.1^{50}\approx117$); why sigmoid for gates; why ⊙.
- The lecturer's mental model: *network = structured composition of functions*.

### §3 Structure of the 5-layer RNN *(R2)*
Source: concepts notebook §2–§4.
1. **Cell**: $h_t=\tanh(x_tW_x+h_{t-1}W_h+b)$ as $\tanh\circ\mathrm{add}(\cdot)$; numeric example ($F=2,H=2$); why weight sharing; why $h_0=0$.
2. **Stack**: layer equations, then $\hat y=g\circ\mathrm{last}\circ f^{(5)}\circ\dots\circ f^{(1)}$. TikZ figure of the 5×T grid (up = layers, right = time). Why stack, why only the last step, why $g=\sigma$ or identity.
3. **Shapes table** for both datasets, e.g. EC `(256,100)` → embedding `(256,100,64)` → 5 × `(256,100,64)` → last `(256,64)` → `(256,1)`.
4. **Parameters**: general formula, then the hand count vs PyTorch vs Keras: toy (181/201), EC (1,321,345 / 1,321,665), ST (9,537 / 9,697). Why PyTorch is +L·H (two biases).
5. **Gradient**: chain rule, $\prod \mathrm{diag}(1-h^2)W_h^\top$; vanishing/exploding scalar examples; figure *RNN gradient map* (layer × step); measured value at $T=50$, 5 layers: 8.7e-11. Why clipping fixes explosion but not vanishing.

### §4 Structure of the 5-layer LSTM *(R3)*
Source: concepts notebook §5–§6.
1. Gate table (i, f, g, o, c, h) + role + **why** each piece exists (context-dependent gates, signed candidate, additive cell, $o\odot\tanh c$).
2. The cell path: $\partial c_t/\partial c_{t-1}\approx\mathrm{diag}(f_t)$, so $0.97^{50}\approx0.22$ vs $0.6^{50}\approx8\times10^{-12}$. Framed as the lecturer's *old architecture + mechanism*: the RNN's limitation (vanishing gradient) → mechanism (gated additive memory).
3. Composition view of the 5-layer LSTM; TikZ cell diagram.
4. Parameters: 4× formula and why; toy 709/789, EC 1,445,185 / 1,446,465, ST 38,049 / 38,689. Note that in EC the embedding (1,280,000) dominates both models.
5. Gradient-flow figure: RNN 8.7e-11, LSTM bias 0 9.4e-11, LSTM bias 1 4.5e-4; why forget bias 1 ($0.73^{50}$ vs $0.5^{50}$), and why we set it in PyTorch too.
6. Summary table RNN vs LSTM (from notebook §7).

### §5 Datasets *(R1)*
Source: dataset notebooks §2–§4, `dataset/README.md`.
- **Selection rationale table**: candidates considered, why these two (size ~100 MB, real e-commerce comments, long sequences for EC; volatility clustering for ST), why not reuse A6's data.
- **EC: Amazon Unlocked Mobile reviews**: 413,840 raw rows → drop 251,349 duplicate texts and 3-star reviews → 148,124; 70.1 % positive; stratified 70/15/15. Figures: rating distribution, length by class. Why drop duplicates (leakage across splits); why drop 3 stars (ambiguous label); why MAXLEN 100 (covers 88.8 % of reviews); why a train-only vocabulary of 20k (99.3 % coverage, <unk> 0.73 %).
- **ST: S&P 500 stocks**: 1.89 M rows; 145 stocks with the full 3,768 days; features ret, hl, oc, lvol; target $|r_{t+1}|$; windows of T=60; chronological split (train to 2020-06, valid to 2022-09, test to 2024-12). Figures `ST_eda`, `ST_clustering`. Why the target is volatility, not direction (lag-1 ACF: r −0.04 vs |r| 0.215, still 0.048 at lag 60); why z-score on train only; why a chronological split.
- Pipeline figure: text → tokens → ids → padded (the worked review from EC §3).

### §6 Implementation in PyTorch and Keras *(R4)*
Source: concepts notebook §3/§5 checks, dataset notebooks §5–§6.
- Side-by-side listings (`listings`): Keras `Sequential` (Embedding → 4 × RNN with `return_sequences=True` → RNN → Dense) and the PyTorch `nn.Module` (`nn.RNN/LSTM(num_layers=5)` → `out[:, -1]` → Linear).
- Weight-name mapping table: Keras `kernel` / `recurrent_kernel` / `bias` ↔ PyTorch `weight_ih_l{k}` (transposed) / `weight_hh_l{k}` / `bias_ih + bias_hh`; gate order i, f, g, o.
- Verification table: hand-written NumPy vs PyTorch vs Keras, max |diff| ~1e-7 for RNN and LSTM.
- Initialisation copied across frameworks (xavier, orthogonal, zero bias, forget bias 1, embedding ±0.05): why it matters for a fair comparison.

### §7 Experimental setup *(R5)*
- One protocol table: H (64/32), Adam lr 1e-3, clip-norm 1, batch 256, ≤10 epochs, early stopping patience 2 on validation loss, seed 42, loss (BCE / MSE).
- Baselines: EC majority, TF-IDF + logistic regression; ST train mean, persistence, 20-day mean. Why each baseline: majority = floor, TF-IDF = strong order-free model, 20-day mean = what "memory" alone gives.
- Metrics + why: EC accuracy, macro F1, **negative-class F1** (the minority class that matters), ROC-AUC; ST RMSE/MAE in %, R²_oos vs train mean, correlation.
- Hardware/software actually used: Windows CPU run, TensorFlow 2.22.0-rc0, PyTorch 2.14.1+cpu (from the notebook outputs). Why the times are comparable only within this machine.

### §8 Results: EC review sentiment *(R5)*
- `table_ec.tex` (generated): 2 baselines + 4 runs × acc, F1-macro, F1-neg, AUC, params, s/epoch, epochs (best).
- Figures: `EC_curves` (loss), `EC_confusion_roc`, `EC_by_length`; examples where the RNN and LSTM disagree (782 of 22,219), plus 2–3 wrong predictions (the lecturer's "inspect incorrect predictions").
- **Why** paragraphs (required):
  - LSTM > RNN, and the gap widens on long reviews (>100 words: 0.855 → 0.868/0.874): negative reviews are longer and their evidence is spread out, which is where the cell path helps.
  - Validation loss rises after epoch 2–3 while training loss keeps falling: overfitting, mostly in the 1.28 M-parameter embedding trained from scratch on 104 k reviews. Early stopping keeps the best epoch.
  - **TF-IDF beats both** (0.941 / AUC 0.983): sentiment here is largely lexical ("great", "broke", "return"), so bag-of-words sees every word of every review, while the RNNs see at most 100 words and must learn embeddings from scratch. State it as measured, not hidden.
  - Short reviews are easiest for all models (≥0.945).

### §9 Results: ST stock volatility *(R5)*
- `table_st.tex` (generated): 3 baselines + 4 runs × RMSE %, MAE %, R²_oos, corr, params, s/epoch, epochs.
- Figures: `ST_curves`, `ST_forecast`, `ST_by_regime`.
- **Why** paragraphs:
  - Persistence is worse than the mean (R² −0.70): one day's |r| is a noisy estimate of tomorrow's. The 20-day mean (R² 0.044) shows that averaging the recent past works.
  - The networks double that (R² ≈ 0.09): they weight the recent past adaptively and use the range/volume features.
  - LSTM ≈ RNN (RMSE 1.296–1.299 vs 1.302–1.303): the |r| autocorrelation decays within a few weeks, so the information sits in recent steps, which the RNN can already reach. The LSTM's advantage shows mainly in the most turbulent quintile (1.752 vs 1.765).
  - Early stopping at epochs 1–4 and noisy validation loss: the validation period (mid-2020 to 2022) is a different regime from the train period (post-COVID, rate hikes). Low R² is normal for daily volatility.

### §10 Comparison: RNN vs LSTM across both datasets *(R6)*
- `table_summary.tex`: 8 runs, with columns dataset, framework, model, main score, params, s/epoch, train time.
- Four axes, following the lecturer's tutorial:
  1. **Accuracy**: LSTM gain on EC vs ST, and why it differs (long-range dependency vs short memory).
  2. **Parameters**: 4× in the recurrent part. On EC the embedding dominates (+9 % in total); on ST it is the full 4× (9.5 k → 38 k).
  3. **Time**: LSTM 2.5–4.8× slower per epoch in TF, 1.8–2.5× in PT. Why: four matmuls per step; framework kernel differences on CPU.
  4. **Behaviour**: gradient flow (§4), accuracy by length, error by regime.
- Framework comparison: TF vs PT within 0.01 on every metric. Why they are not identical: random batch order, nondeterministic kernels, different best epochs; parameters differ only by the extra bias.
- Bar charts: score and s/epoch for the 8 runs (`make_tables.py` writes `cmp_score.png`, `cmp_time.png`).

### §11 Limitations
One run per configuration (no seeds/variance); truncation to 100 words; embeddings learned from scratch (no pretrained vectors); no bidirectional/attention variants; CPU-only timing; volatility regime shift; hyperparameters not tuned per model.

### §12 Conclusion
Answer the brief directly: what the composition view explains; when the LSTM is worth its cost (EC yes, ST marginal); what to try next (pretrained embeddings, bidirectional LSTM, GRU, attention, multiple seeds).

### Appendix A: Reproducibility
File map (`dataset/`, `notebook/`, `report/`), `requirements.txt`, Kaggle download commands, run order, how to rebuild tables and figures and the PDF.

## 4. Figures and tables inventory

| ID | File | Source | Section |
|---|---|---|---|
| F1 | `rnn_grid` (TikZ) | written in LaTeX | §3 |
| F2 | `lstm_cell` (TikZ) | written in LaTeX | §4 |
| F3–F5 | `C_rnn_grad.png`, `C_lstm_grad.png`, `C_grid.png` | extracted from `00_rnn_lstm_concepts.ipynb` (it does not `savefig`) | §3–§4 |
| F6–F7 | `EC_rating.png`, `EC_length.png`, `EC_pipeline.png` | extracted from `ec_reviews.ipynb` (EDA cells not saved yet) | §5 |
| F8–F9 | `ST_eda.png`, `ST_clustering.png` | already in `report/images/` | §5 |
| F10–F12 | `EC_curves.png`, `EC_confusion_roc.png`, `EC_by_length.png` | already saved | §8 |
| F13–F15 | `ST_curves.png`, `ST_forecast.png`, `ST_by_regime.png` | already saved | §9 |
| F16–F17 | `cmp_score.png`, `cmp_time.png` | `make_tables.py` | §10 |
| T1–T4 | params, verification, shapes, protocol | written from the concepts notebook | §3, §4, §6, §7 |
| T5–T7 | `table_ec.tex`, `table_st.tex`, `table_summary.tex` | `make_tables.py` | §8–§10 |
| T8 | `table_ec_length.tex`, `table_st_regime.tex` | `make_tables.py` parses the notebook output tables | §8–§9 |

## 5. Work steps

1. Remove the leftover `notebook/results/smoke/` and `report/images/smoke/` (from the abandoned smoke run).
2. Write `report/extract_figures.py` (adapted from A5): decode the embedded PNGs of named cells into `report/images/`.
3. Write `report/make_tables.py` (adapted from A6): T5–T8, F16–F17.
4. Write `Assignment_07.tex`: A6 preamble and cover layout with this author; sections as in §3.
5. Build with `xelatex` twice; check every number against `results/*.json`; check no overfull figures.
6. Fill the notebooks' §8 *Conclusion* cells with the same findings in short form, so notebooks and report agree.
7. Commit (no co-author line) and push after review.

## 6. Writing rules

- Concise, like the notebooks: formula, then a tiny numeric example, then **why** it is so.
- Every number comes from the executed notebooks or `results/`; captions name the run ID or dataset code.
- Negative and small results are stated as measured (TF-IDF wins on EC; LSTM ≈ RNN on ST), each with its explanation.
- Terms are defined once and used consistently: step, layer, hidden state, cell state, gate, window, run ID.
- English, like assignments 05–06.
