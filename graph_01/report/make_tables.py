"""Build the LaTeX tables of the report from the data files and Neo4j -> report/table_*.tex.

Every number in the report comes from here or from the executed notebook.
  python report/make_tables.py        (Neo4j must be running and loaded)
"""
import csv
import math
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from neo4j import GraphDatabase

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
sys.path.insert(0, str(ROOT / "kg"))
import vocab  # noqa: E402

DRIVER = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "diabetes123"))
TYPE = "[l IN labels({v}) WHERE l <> 'Entity'][0]"


def cy(query, **params):
    with DRIVER.session() as s:
        return [r.data() for r in s.run(query, **params)]


def esc(x):
    s = str(x)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"),
                 ("≥", r"$\geq$"), ("→", r"$\rightarrow$")]:
        s = s.replace(a, b)
    return s


def write(name, rows):
    """rows: list of lists, already escaped where needed; written as body lines of a tabular."""
    (REPORT / f"table_{name}.tex").write_text("\n".join(" & ".join(map(str, r)) + r" \\" for r in rows) + "\n")
    print("wrote", f"table_{name}.tex", len(rows), "rows")


def fmt(n):
    return f"{n:,}"


def main():
    idx = pd.read_csv(ROOT / "dataset" / "raw" / "index.csv")
    ent = pd.read_csv(ROOT / "dataset" / "processed" / "entities.csv")
    tri = pd.read_csv(ROOT / "dataset" / "processed" / "triples.csv")
    audit = pd.read_csv(ROOT / "evaluation" / "manual_audit.csv")
    kept = idx[idx.relevant]

    # --- crawl funnel + relevance filter
    ct = pd.crosstab(idx.slug_match, idx.relevant)
    write("relevance", [[esc(s), fmt(ct.loc[s, True]), fmt(ct.loc[s, False]), fmt(ct.loc[s].sum())] for s in ["strong", "weak"]]
          + [[r"\textbf{Total}", fmt(int(idx.relevant.sum())), fmt(int((~idx.relevant).sum())), fmt(len(idx))]])
    rej = idx[~idx.relevant].dropna(subset=["title"]).head(6)
    write("rejected", [[esc(r.title), esc(r.slug_match), r.core_mentions, r.related_mentions] for r in rej.itertuples()])
    years = pd.to_datetime(kept.date_published, errors="coerce").dt.year.dropna().astype(int)
    write("volume", [
        ["Candidate URLs from the sitemap", fmt(len(idx))],
        ["Pages kept after the relevance filter", fmt(len(kept))],
        ["Official disease pages among them", fmt(int((kept.page_type == "disease").sum()))],
        ["Total words", fmt(int(kept.n_words.sum()))],
        ["Median words per page", fmt(int(kept.n_words.median()))],
        ["Shortest / longest page (words)", f"{fmt(int(kept.n_words.min()))} / {fmt(int(kept.n_words.max()))}"],
        ["Publication years", f"{years.min()}--{years.max()}"],
        ["Size of the saved JSON pages", "9.4 MB"],
    ])

    # --- vocabulary and entity types
    voc = Counter(lab for lab, _ in vocab.VOCAB.values())
    al = Counter()
    for lab, aliases in vocab.VOCAB.values():
        al[lab] += len(aliases)
    in_graph = ent.label.value_counts()
    examples = ent.groupby("label").name.apply(lambda s: ", ".join(list(s)[:3]))
    order = ["DiabetesType", "Symptom", "RiskFactor", "Complication", "Test", "Drug", "DrugClass", "Food", "Lifestyle", "Organ"]
    write("entity_types", [[t, voc.get(t, "--"), al.get(t, "--"), in_graph.get(t, 0), esc(examples.get(t, ""))] for t in order]
          + [[r"\textbf{Total}", sum(voc.values()), sum(al.values()), int(in_graph.sum()), ""]])

    # --- relation types
    rel = cy(f"""MATCH (a:Entity)-[r]->(b:Entity)
                 RETURN type(r) AS rel, {TYPE.format(v='a')} AS src, {TYPE.format(v='b')} AS dst, r.source AS source,
                        count(*) AS n ORDER BY n DESC""")
    write("relations", [[esc(r["rel"]), f"{r['src']} $\\rightarrow$ {r['dst']}", r["source"], r["n"]] for r in rel])

    # --- example triples
    ex = tri[tri.source == "vinmec"].sort_values("support", ascending=False)
    pick = pd.concat([ex[ex.relation == r].head(1) for r in ex.relation.unique()])
    write("example_triples", [[esc(r.subject), esc(r.relation), esc(r.object), r.support] for r in pick.itertuples()])

    # --- lecture checklist
    nq = sum(1 for line in (ROOT / "cypher" / "queries.cypher").read_text().splitlines() if line.startswith("// ") and line[3:4].isdigit())
    write("checklist", [
        ["Entities", r"$\geq$ 50 (Mini Project 3), $\geq$ 20 (Exercise 1)", len(ent)],
        ["Relationships", r"$\geq$ 100 (Mini Project 3), $\geq$ 30 (Exercise 1)", len(tri)],
        ["Relationship types", r"$\geq$ 5", tri.relation.nunique()],
        ["Entity types", r"$\geq$ 3", ent.label.nunique()],
        ["Cypher queries", r"$\geq$ 5", nq],
        ["Triples stored as CSV", r"\code{subject, relation, object}", r"\code{triples.csv}"],
        ["One-hop and two-hop queries", "required", "queries 12--14, \\code{graph\\_facts}"],
        ["Graph visualisation", "required", "NetworkX figures, website"],
    ])

    # --- support thresholds
    ext = tri[tri.source == "vinmec"]
    write("support", [[f"$\\geq$ {t}", int((ext.support >= t).sum()), f"{(ext.support >= t).mean() * 100:.0f}\\%"] for t in [1, 2, 3, 5, 10, 20]])

    # --- precision
    n, k = len(audit), int(audit.correct.sum())
    p = k / n
    z = 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    by = audit.groupby("relation").correct.agg(["count", "sum"])
    write("precision", [[esc(r), int(row["count"]), int(row["sum"])] for r, row in by.iterrows()]
          + [[r"\textbf{All}", n, f"\\textbf{{{k}}} ({p:.2f}; likely range {c - h:.2f}--{c + h:.2f})"]])
    write("audit_examples", [[r.id, esc(r.subject), esc(r.relation), esc(r.object), r.support, "yes" if r.correct else r"\textbf{no}"]
                             for r in audit.head(12).itertuples()])

    # --- query results
    q = cy("""MATCH (e:Entity {name: 'Metformin'})-[r]-(n:Entity)
              RETURN startNode(r).name AS s, type(r) AS rel, endNode(r).name AS o, coalesce(r.support, 0) AS sup, r.source AS src
              ORDER BY sup DESC""")
    write("q_metformin", [[esc(r["s"]), esc(r["rel"]), esc(r["o"]), r["sup"], r["src"]] for r in q])
    q = cy("""MATCH (t:DiabetesType {name: 'Đái tháo đường type 2'})-[r:HAS_COMPLICATION]->(c)-[:AFFECTS_ORGAN]->(o:Organ)
              RETURN o.name AS organ, collect(c.name) AS via, sum(r.support) AS ev ORDER BY ev DESC""")
    write("q_organs", [[esc(r["organ"]), esc(", ".join(r["via"])), r["ev"]] for r in q])
    q = cy("""MATCH (s:Symptom)
              OPTIONAL MATCH (:DiabetesType {name: 'Đái tháo đường type 1'})-[a:HAS_SYMPTOM]->(s)
              OPTIONAL MATCH (:DiabetesType {name: 'Đái tháo đường type 2'})-[b:HAS_SYMPTOM]->(s)
              WITH s, a, b WHERE a IS NOT NULL OR b IS NOT NULL
              RETURN s.name AS symptom, a.support AS t1, b.support AS t2,
                     CASE WHEN a IS NOT NULL AND b IS NOT NULL THEN 'both' WHEN a IS NOT NULL THEN 'type 1 only' ELSE 'type 2 only' END AS found
              ORDER BY found, symptom""")
    write("q_t1t2", [[esc(r["symptom"]), r["t1"] or "--", r["t2"] or "--", r["found"]] for r in q])
    q = cy("""MATCH (t:DiabetesType)-[r:HAS_SYMPTOM]->(s:Symptom) WHERE s.name IN $input
              RETURN t.name AS type, count(s) AS m, collect(s.name) AS syms, sum(r.support) AS ev
              ORDER BY m DESC, ev DESC""", input=["Khát nhiều", "Tiểu nhiều", "Sụt cân", "Nhìn mờ"])
    write("q_symptoms", [[esc(r["type"]), r["m"], esc(", ".join(r["syms"])), r["ev"]] for r in q])
    q = cy("""MATCH (t:DiabetesType)-[r:DIAGNOSED_BY]->(x:Test)
              RETURN x.name AS test, x.threshold AS th, sum(r.support) AS ev ORDER BY ev DESC""")
    write("q_tests", [[esc(r["test"]), esc(r["th"] or "--"), r["ev"]] for r in q])
    q = cy("""MATCH (:DiabetesType {name: 'Đái tháo đường thai kỳ'})-[r:RECOMMENDS_FOOD|AVOIDS_FOOD]->(f:Food)
              RETURN type(r) AS adv, f.name AS food, r.support AS n ORDER BY adv, n DESC""")
    write("q_diet", [[esc(r["adv"]), esc(r["food"]), r["n"]] for r in q])
    q = cy("""MATCH (:DiabetesType {name: 'Đái tháo đường type 2'})-[r:HAS_COMPLICATION]->(:Complication {name: 'Bệnh thận đái tháo đường'})
              UNWIND r.example_urls AS url MATCH (a:Article {url: url}) RETURN a.title AS t""")
    write("q_provenance", [[esc(r["t"])] for r in q])
    q = cy("""MATCH (a:Article)-[:MENTIONS]->(c1:Complication), (a)-[:MENTIONS]->(c2:Complication) WHERE c1.name < c2.name
              RETURN c1.name AS a, c2.name AS b, count(a) AS n ORDER BY n DESC LIMIT 6""")
    write("q_cooccur", [[esc(r["a"]), esc(r["b"]), r["n"]] for r in q])

    # --- API endpoints
    write("api", [
        [r"\code{GET /stats}", "Number of entities per type, facts per relation, articles"],
        [r"\code{GET /entities?q=\&type=}", "Search by name (prefix matches first, then by number of connections)"],
        [r"\code{GET /entities/\{name\}?min\_support=}", "One entity: type, threshold, article count and all its facts"],
        [r"\code{GET /evidence?subject=\&relation=\&object=}", "The source articles stored on one fact"],
        [r"\code{GET /entities/\{name\}/articles}", "Latest articles that mention the entity"],
        [r"\code{POST /symptom-check}", "Diabetes types ranked by matching symptoms"],
        [r"\code{GET /paths?source=\&target=}", "Shortest chains of facts between two entities (up to 4 steps)"],
        [r"\code{GET /health}", "Is the database reachable? (used for the offline banner)"],
    ])
    DRIVER.close()


if __name__ == "__main__":
    main()
