"""Builds the LaTeX tables of the report from notebook/results/*.json and *_comparison.csv (run after the notebooks)."""
import json
from pathlib import Path
import pandas as pd

HERE = Path(__file__).parent
RES = HERE.parent / "notebook" / "results"
ORDER = ["SC-RNN", "TF-RNN", "PT-RNN", "TF-LSTM", "PT-LSTM", "TF-GRU", "PT-GRU"]


def run(rid): return r"\run{" + rid + "}"


def load(code):
    cmp_ = pd.read_csv(RES / f"{code}_comparison.csv", index_col=0)
    base = cmp_[cmp_.index.str.startswith("baseline")]
    runs = {k: json.loads((RES / f"{code}-{k}.json").read_text()) for k in ORDER}
    return base, runs


def table(code, cols, header, fmt):
    base, runs = load(code)
    lines = [r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule", "Run & " + " & ".join(header) + r" \\", r"\midrule"]
    for name, row in base.iterrows():
        label = name.replace("baseline: ", "").replace("_", r"\_").replace("%", r"\%")
        vals = [fmt[c](row[c]) if c in row and pd.notna(row[c]) else "--" for c in cols]
        lines.append(r"\textit{" + label + "} & " + " & ".join(vals) + r" \\")
    lines.append(r"\midrule")
    for k in ORDER:
        r = runs[k]; flat = {**r["test"], "params": r["params"], "epochs": f"{r['epochs_run']} ({r['best_epoch']})",
                             "s_per_epoch": r["time_per_epoch_s"], "infer": r["infer_ms_per_1k"]}
        lines.append(run(f"{code}-{k}") + " & " + " & ".join(fmt[c](flat[c]) for c in cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


f4 = lambda x: f"{x:.4f}"; f1 = lambda x: f"{x:.1f}"; f3 = lambda x: f"{x:.3f}"; f2 = lambda x: f"{x:.2f}"
ec = table("EC", ["rmse_units", "mae_units", "r2_oos", "params", "epochs", "s_per_epoch", "infer"],
           ["RMSE", "MAE", r"$R^2$", "params", "epochs (best)", "s/epoch", "ms/1k"],
           dict(rmse_units=f2, mae_units=f2, r2_oos=f3, params=lambda x: f"{int(x):,}".replace(",", "{,}"), epochs=str, s_per_epoch=f1, infer=f2))
st = table("ST", ["rmse", "r2_oos", "dir_acc", "ic", "params", "epochs", "s_per_epoch", "infer"],
           ["RMSE", r"$R^2$", "dir.\\ acc.", "IC", "params", "epochs (best)", "s/epoch", "ms/1k"],
           dict(rmse=f4, r2_oos=f3, dir_acc=f3, ic=f3, params=lambda x: f"{int(x):,}".replace(",", "{,}"), epochs=str, s_per_epoch=f1, infer=f2))
(HERE / "table_ec.tex").write_text(ec, encoding="utf-8")
(HERE / "table_st.tex").write_text(st, encoding="utf-8")

# compact 14-row summary: core metric per run
rows = []
for code, key, lab in (("EC", "rmse_units", "RMSE (units)"), ("ST", "rmse", "RMSE (log return)")):
    _, runs = load(code)
    for k in ORDER:
        r = runs[k]
        rows.append((f"{code}-{k}", r["test"][key], r["test"]["r2_oos"], r["params"], r["time_per_epoch_s"]))
lines = [r"\begin{tabular}{lrrrr}", r"\toprule", r"Run & test RMSE & $R^2$ & params & s/epoch \\", r"\midrule"]
for i, (rid, a, b, p, t) in enumerate(rows):
    if i == len(ORDER): lines.append(r"\midrule")
    fa = f"{a:.2f}" if rid.startswith("EC") else f"{a:.4f}"
    lines.append(run(rid) + f" & {fa} & {b:.3f} & {p:,} & {t:.1f}".replace(",", "{,}", 0) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(HERE / "table_summary.tex").write_text("\n".join(lines).replace("{,}", ","), encoding="utf-8")
print("tables written")
