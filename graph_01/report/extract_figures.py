"""Export the figures of notebook/01_diabetes_kg.ipynb to report/images/.

Each figure is identified by a snippet of the code cell that produced it, so the
mapping survives cells being added or moved.
  python report/extract_figures.py
"""
import base64
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebook" / "01_diabetes_kg.ipynb"
OUT = ROOT / "report" / "images"

# (code snippet, output file names in the order the cell produces them)
FIGURES = [
    ("toy = [", ["toy_graph.png"]),
    ("kept = idx[idx.relevant]", ["corpus_stats.png"]),
    ('arts.about.value_counts().plot', ["pages_per_subject.png"]),
    ("ent.label.value_counts().plot.barh", ["graph_counts.png"]),
    ('heat("HAS_COMPLICATION"', ["heat_complications.png", "heat_symptoms.png"]),
    ("Y = 1.4", ["schema.png"]),
    ('draw(G, "Diabetes Knowledge Graph', ["kg_full_nb.png"]),
    ('one_hop(G, "Đái tháo đường thai kỳ")', ["kg_gestational.png", "kg_drugs.png"]),
    ("pr_dir = nx.pagerank", ["pagerank.png"]),
    ("MATCH (a:Article)-[:MENTIONS]->(e:Entity)\nRETURN [l IN labels(e)", ["mentions.png"]),
]


def main():
    nb = nbformat.read(NB, 4)
    OUT.mkdir(parents=True, exist_ok=True)
    for snippet, names in FIGURES:
        cell = next(c for c in nb.cells if c.cell_type == "code" and snippet in c.source)
        pngs = [o["data"]["image/png"] for o in cell.outputs if "image/png" in o.get("data", {})]
        assert len(pngs) == len(names), (snippet, len(pngs))
        for name, data in zip(names, pngs):
            (OUT / name).write_bytes(base64.b64decode(data))
            print("wrote", name)


if __name__ == "__main__":
    main()
