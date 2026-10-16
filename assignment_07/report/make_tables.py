"""Builds the report's LaTeX tables and comparison charts from the executed notebooks (run after the notebooks).

Reads notebook/results/<run>.json and <code>_comparison.csv, and the accuracy-by-length (EC) and
error-by-regime (ST) tables printed in the notebooks.

    python report/make_tables.py
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).parent
NB = HERE.parent / "notebook"
RES = NB / "results"
IMG = HERE / "images"
RUNS = ["TF-RNN", "TF-LSTM", "PT-RNN", "PT-LSTM"]


def run(rid): return r"\run{" + rid + "}"


def load(code):
    cmp_ = pd.read_csv(RES / f"{code}_comparison.csv", index_col=0)
    base = cmp_[cmp_.index.str.startswith("baseline")]
    runs = {k: json.loads((RES / f"{code}-{k}.json").read_text()) for k in RUNS}
    return base, runs


def tabular(header, rows, align):
    out = [r"\begin{tabular}{" + align + "}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    for r in rows:
        out.append(r"\midrule" if r == "MID" else " & ".join(r) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def f4(x): return f"{x:.4f}".replace("-0.0000", "0.0000")
def f3(x): return f"{x:.3f}"
def thousands(x): return f"{int(x):,}".replace(",", "{,}")


def bold_best(rows, col, best):
    """Bold the best value of column `col` among the model rows (strings already formatted)."""
    vals = [float(r[col]) for r in rows if r != "MID" and r[col] != "--"]
    target = f"{best(vals):.4f}"
    for r in rows:
        if r != "MID" and r[col] == target:
            r[col] = r"\textbf{" + target + "}"


def table_ec():
    base, runs = load("EC")
    rows = []
    for name, b in base.iterrows():
        rows.append([r"\textit{" + {"baseline: majority": "majority class", "baseline: tfidf_logreg": "TF-IDF + log. regression"}[name] + "}",
                     f4(b.accuracy), f4(b.f1_macro), f4(b.f1_negative), f4(b.roc_auc), "--", "--", "--"])
    rows.append("MID")
    for k in RUNS:
        r = runs[k]
        rows.append([run("EC-" + k), f4(r["accuracy"]), f4(r["f1_macro"]), f4(r["f1_negative"]), f4(r["roc_auc"]),
                     thousands(r["params"]), f"{r['sec_per_epoch']:.0f}", f"{r['epochs']} ({r['best_epoch']})"])
    model_rows = rows[len(base) + 1:]
    for c in range(1, 5):
        bold_best(model_rows, c, max)
    return tabular(["Run", "Accuracy", "F1 macro", "F1 neg.", "ROC-AUC", "Params", "s/epoch", "Epochs (best)"], rows, "lrrrrrrr")


def table_st():
    base, runs = load("ST")
    rows = []
    for name, b in base.iterrows():
        corr = "--" if pd.isna(b["corr"]) else f3(b["corr"])
        rows.append([r"\textit{" + name.replace("baseline: ", "").replace("|r_t|", "$|r_t|$").replace("|r|", "$|r|$") + "}",
                     f4(b["RMSE (%)"]), f4(b["MAE (%)"]), f4(b["r2_oos"]), corr, "--", "--", "--"])
    rows.append("MID")
    for k in RUNS:
        r = runs[k]
        rows.append([run("ST-" + k), f4(100 * r["rmse"]), f4(100 * r["mae"]), f4(r["r2_oos"]), f3(r["corr"]),
                     thousands(r["params"]), f"{r['sec_per_epoch']:.0f}", f"{r['epochs']} ({r['best_epoch']})"])
    model_rows = rows[len(base) + 1:]
    bold_best(model_rows, 1, min); bold_best(model_rows, 2, min); bold_best(model_rows, 3, max)
    return tabular(["Run", "RMSE (\\%)", "MAE (\\%)", "$R^2_{\\text{oos}}$", "Corr.", "Params", "s/epoch", "Epochs (best)"], rows, "lrrrrrrr")


def table_summary():
    """RNN vs LSTM per dataset and framework: main score, parameters, time."""
    rows = []
    for code, score, label in [("EC", "f1_macro", "F1 macro"), ("ST", "r2_oos", "$R^2_{\\text{oos}}$")]:
        _, runs = load(code)
        for fw in ["TF", "PT"]:
            rnn, lstm = runs[f"{fw}-RNN"], runs[f"{fw}-LSTM"]
            rows.append([code, {"TF": "Keras", "PT": "PyTorch"}[fw], label,
                         f4(rnn[score]), f4(lstm[score]), f"{lstm[score] - rnn[score]:+.4f}",
                         thousands(rnn["params"]), thousands(lstm["params"]),
                         f"{rnn['sec_per_epoch']:.0f} / {lstm['sec_per_epoch']:.0f}",
                         f"{lstm['sec_per_epoch'] / rnn['sec_per_epoch']:.1f}$\\times$"])
        if code == "EC":
            rows.append("MID")
    return tabular(["Data", "Framework", "Score", "RNN", "LSTM", "$\\Delta$", "Params RNN", "Params LSTM",
                    "s/epoch R / L", "Time ratio"], rows, "lllrrrrrrr")


def notebook_table(file, first_line):
    """Parse a DataFrame printed by a notebook cell (text/plain output) into a DataFrame."""
    cells = json.loads((NB / file).read_text(encoding="utf-8"))["cells"]
    for c in cells:
        if c["cell_type"] == "code" and "".join(c["source"]).startswith(first_line):
            for o in c["outputs"]:
                t = o.get("text") or o.get("data", {}).get("text/plain")
                if t:
                    text = "".join(t)
                    lines = [l for l in text.splitlines() if l.strip()]
                    header = re.split(r"\s{2,}", lines[0].strip())
                    rows = [re.split(r"\s{2,}", l.strip()) for l in lines[1:]]
                    return pd.DataFrame(rows, columns=["group"] + header).set_index("group").astype(float)
    raise KeyError(first_line)


def table_ec_length():
    d = notebook_table("ec_reviews.ipynb", "n_te = df.n_words.values[i_te]")
    rows = []
    for g, r in d.iterrows():
        vals = [r[f"EC-{k}"] for k in RUNS]
        rows.append([g.replace(">", "$>$").replace("-", "--"), *[f3(v) for v in vals],
                     f"{(r['EC-TF-LSTM'] + r['EC-PT-LSTM'] - r['EC-TF-RNN'] - r['EC-PT-RNN']) / 2:+.3f}", thousands(r["reviews"])])
    return tabular(["Words", *[run("EC-" + k) for k in RUNS], "LSTM $-$ RNN", "Reviews"], rows, "lrrrrrr")


def table_st_regime():
    d = notebook_table("st_stocks.ipynb", 'lvl = pd.qcut(A["abs_mean20"]')
    rows = []
    for g, r in d.iterrows():
        rows.append([g, *[f4(r[f"ST-{k}"]) for k in RUNS], f4(r["20-day mean"]),
                     f"{(r['ST-TF-LSTM'] + r['ST-PT-LSTM'] - r['ST-TF-RNN'] - r['ST-PT-RNN']) / 2:+.4f}"])
    return tabular(["20-day volatility", *[run("ST-" + k) for k in RUNS], "20-day mean", "LSTM $-$ RNN"], rows, "lrrrrrr")


def charts():
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.2))
    for a, (code, score, label) in zip(ax, [("EC", "f1_macro", "test F1 macro"), ("ST", "r2_oos", "test $R^2_{oos}$")]):
        base, runs = load(code)
        names = list(runs)
        vals = [runs[k][score] for k in names]
        a.bar(names, vals, color=["C0" if k.endswith("RNN") else "C3" for k in names])
        ref = base.loc["baseline: tfidf_logreg", "f1_macro"] if code == "EC" else base.loc["baseline: 20-day mean |r|", "r2_oos"]
        a.axhline(ref, ls=":", c="k", label="TF-IDF + LogReg" if code == "EC" else "20-day mean")
        lo = min(vals + [ref]); hi = max(vals + [ref])
        a.set_ylim(lo - (hi - lo) * 0.6, hi + (hi - lo) * 0.4)
        a.set_title(f"{code}: {label}"); a.legend(fontsize=8)
        for i, v in enumerate(vals):
            a.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout(); fig.savefig(IMG / "cmp_score.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3))
    labels, vals, cols = [], [], []
    for code in ["EC", "ST"]:
        _, runs = load(code)
        for k, r in runs.items():
            labels.append(f"{code}-{k}"); vals.append(r["sec_per_epoch"]); cols.append("C0" if k.endswith("RNN") else "C3")
    ax.barh(labels[::-1], vals[::-1], color=cols[::-1])
    for i, v in enumerate(vals[::-1]):
        ax.text(v, i, f" {v:.0f}", va="center", fontsize=8)
    ax.set_xlabel("median seconds per epoch (CPU)")
    fig.tight_layout(); fig.savefig(IMG / "cmp_time.png", dpi=150); plt.close(fig)


def main():
    out = {"table_ec.tex": table_ec(), "table_st.tex": table_st(), "table_summary.tex": table_summary(),
           "table_ec_length.tex": table_ec_length(), "table_st_regime.tex": table_st_regime()}
    for name, tex in out.items():
        (HERE / name).write_text(tex)
        print("wrote", name)
    charts()
    print("wrote images/cmp_score.png, images/cmp_time.png")


if __name__ == "__main__":
    main()
