"""Builds the LaTeX tables and the comparison charts of the report from notebook/results and the executed notebooks.
Run after the notebooks:  python make_tables.py"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).parent
NB = HERE.parent / "notebook"
RES = NB / "results"
IMG = HERE / "images"
ORDER = ["TF-RNN", "PT-RNN", "TF-LSTM", "PT-LSTM"]
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130})


def run(rid): return r"\run{" + rid + "}"
def num(x): return f"{int(x):,}".replace(",", "{,}")
f4 = lambda x: f"{x:.4f}"; f3 = lambda x: f"{x:.3f}"; f2 = lambda x: f"{x:.2f}"; f1 = lambda x: f"{x:.1f}"


def load(code):
    cmp_ = pd.read_csv(RES / f"{code}_comparison.csv", index_col=0)
    base = cmp_[cmp_.index.str.startswith("baseline")]
    runs = {k: json.loads((RES / f"{code}-{k}.json").read_text()) for k in ORDER}
    return base, runs


def table(code, cols, header, fmt, base_cols=None):
    base, runs = load(code)
    lines = [r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule", "Run & " + " & ".join(header) + r" \\", r"\midrule"]
    for name, row in base.iterrows():
        label = name.replace("baseline: ", "").replace("_", r"\_")
        vals = [fmt[c](row[c]) if c in row and pd.notna(row[c]) and c in (base_cols or cols) else "--" for c in cols]
        lines.append(r"\textit{" + label + "} & " + " & ".join(vals) + r" \\")
    lines.append(r"\midrule")
    for k in ORDER:
        r = runs[k]
        flat = {**r, "epochs": f"{r['epochs']} ({r['best_epoch']})"}
        lines.append(run(f"{code}-{k}") + " & " + " & ".join(fmt[c](flat[c]) for c in cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def frame_table(df, fmt=f3, first="", index_fmt=str):
    lines = [r"\begin{tabular}{l" + "r" * len(df.columns) + "}", r"\toprule",
             first + " & " + " & ".join(c.replace("_", r"\_").replace("%", r"\%") for c in df.columns) + r" \\", r"\midrule"]
    for idx, row in df.iterrows():
        lines.append(index_fmt(idx) + " & " + " & ".join(fmt(v) if pd.notna(v) else "--" for v in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def parse_text_table(txt, row_labels, cols):
    lines = [l for l in txt.splitlines() if l.strip()]
    rows = {}
    for l in lines[1:]:
        tok = l.split()
        for lab in row_labels:
            n = len(str(lab).split())
            if " ".join(tok[:n]) == str(lab):
                rows[lab] = [float(v) for v in tok[n:]]
    return pd.DataFrame(rows, index=cols).T


# ---------- result tables ----------
ec = table("EC", ["accuracy", "f1_macro", "f1_negative", "f1_neutral", "f1_positive", "roc_auc", "logloss", "params", "sec_per_epoch", "epochs"],
           ["acc.", "F1 macro", "F1 neg.", "F1 neu.", "F1 pos.", "AUC", "log-loss", "params", "s/epoch", "epochs (best)"],
           dict(accuracy=f3, f1_macro=f3, f1_negative=f3, f1_neutral=f3, f1_positive=f3, roc_auc=f3, logloss=f3, params=num, sec_per_epoch=f1, epochs=str))
(HERE / "table_ec.tex").write_text(ec, encoding="utf-8")

# ST: the comparison csv already holds RMSE/MAE of the baselines in percent; the run JSONs hold them as fractions
base_st, runs_st = load("ST")
lines = [r"\begin{tabular}{lrrrrrrr}", r"\toprule", r"Run & RMSE (\%) & MAE (\%) & $R^2$ & corr. & params & s/epoch & epochs (best) \\", r"\midrule"]
for name, row in base_st.iterrows():
    cells = [f4(row["RMSE (%)"]), f4(row["MAE (%)"]), f3(row["r2_oos"]), f3(row["corr"]) if pd.notna(row["corr"]) else "--", "--", "--", "--"]
    lines.append(r"\textit{" + name.replace("baseline: ", "") + "} & " + " & ".join(cells) + r" \\")
lines.append(r"\midrule")
for k in ORDER:
    r = runs_st[k]
    lines.append(run(f"ST-{k}") + f" & {100 * r['rmse']:.4f} & {100 * r['mae']:.4f} & {r['r2_oos']:.3f} & {r['corr']:.3f} & {num(r['params'])} & {r['sec_per_epoch']:.1f} & {r['epochs']} ({r['best_epoch']}) \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(HERE / "table_st.tex").write_text("\n".join(lines), encoding="utf-8")

# EC: by review length and by star rating (CSV written by the notebook)
by_len = pd.read_csv(RES / "EC_by_length.csv", index_col=0)
lt = by_len.drop(columns="reviews")
lt.insert(0, "reviews", by_len["reviews"].astype(int))
lt.columns = [c.replace("EC-", "") if c != "reviews" else c for c in lt.columns]
lines = frame_table(lt.drop(columns="reviews"), f3, "words")
lines = lines.replace(r"words &", r"words & reviews &", 1)
body = lines.splitlines()
out = []
for i, l in enumerate(body):
    if i >= 4 and l.split(" & ")[0] in lt.index:
        lab = l.split(" & ")[0]
        l = lab + " & " + f"{lt.loc[lab, 'reviews']:,}".replace(",", "{,}") + " & " + l.split(" & ", 1)[1]
    out.append(l)
out[0] = r"\begin{tabular}{lr" + "r" * (lt.shape[1] - 1) + "}"
(HERE / "table_ec_length.tex").write_text("\n".join(out).replace(">", "$>$"), encoding="utf-8")

stars = pd.read_csv(RES / "EC_stars.csv", index_col=0)
stars_t = stars.drop(columns="reviews"); stars_t.columns = [c.replace("EC-", "") for c in stars_t.columns]
out = frame_table(stars_t, f3, "stars").splitlines()
out[0] = out[0]
(HERE / "table_ec_stars.tex").write_text("\n".join(out), encoding="utf-8")

# ST: by volatility regime and by year (printed tables of the executed notebook)
labels = ["calmest", "calm", "middle", "turbulent", "most turbulent"]
d = json.loads((NB / "st_stocks.ipynb").read_text(encoding="utf8"))
def printed(marker):
    for c in d["cells"]:
        if c["cell_type"] == "code" and marker in "".join(c["source"]):
            for o in c["outputs"]:
                if o["output_type"] == "display_data" and "text/plain" in o["data"] and "Figure" not in "".join(o["data"]["text/plain"]):
                    return "".join(o["data"]["text/plain"])
                if o["output_type"] == "execute_result":
                    return "".join(o["data"]["text/plain"])
    raise KeyError(marker)
RUN_COLS = ["ST-TF-RNN", "ST-TF-LSTM", "ST-PT-RNN", "ST-PT-LSTM"]            # column order of the notebook tables
reg = parse_text_table(printed("lvl = pd.qcut"), labels, RUN_COLS + ["EWMA 0.94"]); reg.columns = [c.replace("ST-", "") for c in reg.columns]
(HERE / "table_st_regime.tex").write_text(frame_table(reg, f3, "20-day volatility"), encoding="utf-8")
yr = parse_text_table(printed("by_year = pd.DataFrame"), [2022, 2023, 2024], RUN_COLS + ["EWMA 0.94", "mean target (%)"]); yr.columns = [c.replace("ST-", "") for c in yr.columns]
(HERE / "table_st_year.tex").write_text(frame_table(yr, f3, "year", index_fmt=lambda i: str(int(i))), encoding="utf-8")

# ---------- summary table (8 runs) ----------
_, ec_runs = load("EC")
rows = [r"\begin{tabular}{llrrrrr}", r"\toprule", r"Run & main metric & value & second metric & params & s/epoch & train time (s) \\", r"\midrule"]
for k in ORDER:
    r = ec_runs[k]
    rows.append(run(f"EC-{k}") + f" & macro-F1 & {r['f1_macro']:.3f} & acc. {r['accuracy']:.3f} & {num(r['params'])} & {r['sec_per_epoch']:.1f} & {r['train_seconds']:.0f} \\\\")
rows.append(r"\midrule")
for k in ORDER:
    r = runs_st[k]
    rows.append(run(f"ST-{k}") + f" & RMSE (\\%) & {100 * r['rmse']:.4f} & $R^2$ {r['r2_oos']:.3f} & {num(r['params'])} & {r['sec_per_epoch']:.1f} & {r['train_seconds']:.0f} \\\\")
rows += [r"\bottomrule", r"\end{tabular}"]
(HERE / "table_summary.tex").write_text("\n".join(rows), encoding="utf-8")

# ---------- comparison charts ----------
colors = {"RNN": "#3b6ea8", "LSTM": "#c8503c"}
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
for a, (code, key, title, scale) in zip(ax, [("EC", "f1_macro", "EC: macro-F1 (higher is better)", 1), ("ST", "r2_oos", r"ST: out-of-sample $R^2$ (higher is better)", 1)]):
    _, runs = load(code)
    vals = [runs[k][key] * scale for k in ORDER]
    bars = a.bar(ORDER, vals, color=[colors[k.split("-")[1]] for k in ORDER])
    for b, v in zip(bars, vals): a.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    a.set_title(title, fontsize=10); a.set_ylim(0, max(vals) * 1.1)
plt.tight_layout(); plt.savefig(IMG / "cmp_score.png", bbox_inches="tight"); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
for a, code in zip(ax, ["EC", "ST"]):
    _, runs = load(code)
    vals = [runs[k]["sec_per_epoch"] for k in ORDER]
    bars = a.bar(ORDER, vals, color=[colors[k.split("-")[1]] for k in ORDER])
    for b, v in zip(bars, vals): a.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    a.set_title(f"{code}: seconds per epoch (CPU)", fontsize=10)
plt.tight_layout(); plt.savefig(IMG / "cmp_time.png", bbox_inches="tight"); plt.close()
print("tables and charts written")
