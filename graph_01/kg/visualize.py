"""Draw the diabetes KG with NetworkX (lecture §4.2) -> report/images/.

  kg_full.png      all entity-to-entity triples, coloured by entity type
  kg_type2.png     1-hop neighbourhood of "Đái tháo đường type 2"
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "processed"
FIG = ROOT / "report" / "images"
COLORS = {"DiabetesType": "#d62728", "Symptom": "#ff7f0e", "RiskFactor": "#8c564b",
          "Complication": "#9467bd", "Test": "#17becf", "Drug": "#1f77b4", "DrugClass": "#aec7e8",
          "Food": "#2ca02c", "Lifestyle": "#bcbd22", "Organ": "#e377c2"}


def build_graph():
    G = nx.DiGraph()
    with open(DATA / "entities.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            G.add_node(r["name"], type=r["label"])
    with open(DATA / "triples.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            G.add_edge(r["subject"], r["object"], relation=r["relation"], support=int(r["support"]))
    return G


def draw(G, path, title, labels_edges=False, k=0.9):
    plt.figure(figsize=(18, 13))
    pos = nx.spring_layout(G, k=k, seed=7)
    colors = [COLORS.get(G.nodes[n].get("type"), "#999") for n in G]
    sizes = [300 + 60 * G.degree(n) for n in G]
    nx.draw_networkx_edges(G, pos, alpha=0.25, arrows=True, arrowsize=8)
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=sizes, alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=8)
    if labels_edges:
        nx.draw_networkx_edge_labels(G, pos, nx.get_edge_attributes(G, "relation"), font_size=6)
    for lab, c in COLORS.items():
        plt.scatter([], [], c=c, label=lab)
    plt.legend(loc="lower left", fontsize=10)
    plt.title(title, fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    G = build_graph()
    print(f"NetworkX graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    draw(G, FIG / "kg_full.png", "Diabetes Knowledge Graph (Vinmec)")
    center = "Đái tháo đường type 2"
    # only the edges touching the centre, so the figure shows exactly the 1-hop facts
    sub = nx.DiGraph()
    sub.add_node(center, **G.nodes[center])
    for u, v, d in [*G.out_edges(center, data=True), *G.in_edges(center, data=True)]:
        sub.add_node(u, **G.nodes[u]); sub.add_node(v, **G.nodes[v]); sub.add_edge(u, v, **d)
    draw(sub, FIG / "kg_type2.png", f"1-hop neighbourhood of {center}", labels_edges=True, k=1.5)
    print(f"Saved {FIG / 'kg_full.png'} and {FIG / 'kg_type2.png'}")


if __name__ == "__main__":
    main()
