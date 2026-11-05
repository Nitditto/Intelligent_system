"""Save the figures that the notebooks show but do not savefig to report/images/<name>.png.

The dataset notebooks already save their result figures; this adds the concepts figures and the EC
data-exploration figures, taken from the executed notebooks so nothing has to be re-run.

    python report/extract_figures.py
"""
import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "report" / "images"
# notebook -> {first line of the code cell: figure name}
FIGURES = {
    "00_rnn_lstm_concepts.ipynb": {
        "L, T_ = 5, 4": "C_grid",
        "def grad_map(": "C_rnn_grad",
        'g_lstm = grad_map("LSTM"': "C_lstm_grad",
    },
    "ec_reviews.ipynb": {
        'df = raw.dropna(subset=["Reviews"])': "EC_ratings",
        'df["n_words"] =': "EC_length",
        "def tokenize(text):": "EC_vocab",
    },
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for file, wanted in FIGURES.items():
        cells = json.loads((ROOT / "notebook" / file).read_text(encoding="utf-8"))["cells"]
        for cell in cells:
            if cell["cell_type"] != "code":
                continue
            src = "".join(cell["source"])
            name = next((n for key, n in wanted.items() if src.startswith(key)), None)
            images = [o["data"]["image/png"] for o in cell.get("outputs", []) if "image/png" in o.get("data", {})]
            if name and images:
                (OUT / f"{name}.png").write_bytes(base64.b64decode("".join(images[0])))
                print(f"{file}: {name}.png")


if __name__ == "__main__":
    main()
