# Graph 01: Diabetes Knowledge Graph from Vinmec with Neo4j

Build a Knowledge Graph about diabetes (đái tháo đường / tiểu đường): its types, symptoms, causes, risk factors, complications, tests, drugs, diet and lifestyle advice, from public Vinmec articles. Store it in Neo4j and query it with Cypher.

```
graph_01/
├── kg/            pipeline: crawl → extract → load, plus retrieval, drawing, evaluation and Cypher queries
├── dataset/       raw/ (crawled pages + index), processed/ (entities, triples, articles, mentions CSV)
├── evaluation/    manual check of 50 sampled facts
├── notebook/      01_diabetes_kg.ipynb: every step explained and run
├── api/, web/     FastAPI service + React explorer
└── report/        Graph_01.tex / .pdf and its plan
```

| File | Role |
|---|---|
| `kg/crawl.py` | Vinmec sitemap → diabetes pages → `dataset/raw/` |
| `kg/vocab.py` | vocabulary, section rules, curated facts |
| `kg/extract.py` | pages → `dataset/processed/*.csv` |
| `kg/load.py` | CSV → Neo4j |
| `kg/retriever.py` | `graph_facts(entity, hops)` |
| `kg/visualize.py` | NetworkX figures |
| `kg/evaluate.py` | random sample of facts for the manual check |
| `kg/queries.cypher` | example Cypher queries for Neo4j Browser |

## Run

```bash
source .venv/bin/activate
python kg/crawl.py        # 1. crawl   -> dataset/raw/pages/*.json (1,096 pages)
python kg/extract.py      # 2. extract -> dataset/processed/{entities,triples,articles,mentions}.csv
docker start neo4j-diabetes || docker run -d --name neo4j-diabetes -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/diabetes123 -v "$PWD/neo4j_data:/data" neo4j:5
python kg/load.py         # 3. load into Neo4j (wipes and reloads)
```

Open http://localhost:7474 (user `neo4j`, password `diabetes123`) and run the queries in `kg/queries.cypher`.

## Graph

127 entities (DiabetesType, Symptom, RiskFactor, Complication, Test, Drug, DrugClass, Food, Lifestyle, Organ), 339 facts and 1,096 Article nodes. A fact extracted from text needs at least 2 supporting pages (3 for side effects), or must come from a Vinmec disease page; `support` on each relationship counts the pages. Facts with `source = "curated"` (drug class, affected organ, subtype) and the diagnostic thresholds are background knowledge, not extracted.

## Website

A small explorer for the graph: search any entity, see its facts with the number of supporting articles and the source links, compare diabetes types, check symptoms, and find connections between two entities.

```
browser (web/, React + Vite) ──/api──▶ api/ (FastAPI) ──Cypher, read-only──▶ Neo4j
```

```bash
docker start neo4j-diabetes                       # the graph must be loaded (python kg/load.py)
uvicorn api.main:app --reload --port 8000         # API, docs at http://localhost:8000/docs
cd web && npm install && npm run dev              # site at http://localhost:5173
```

| Page | URL |
|---|---|
| Home: search, graph size, start points | `#/` |
| Entity: facts grouped by relation, evidence links, neighbourhood, articles | `#/entity/Metformin` |
| Diabetes types side by side | `#/types/Đái tháo đường type 2` |
| Symptom checker | `#/symptoms` |
| Connections (shortest chains between two entities) | `#/paths?from=…&to=…` |

## Report

`report/Graph_01.pdf` (50 pages), built from `report/Graph_01.tex` with XeLaTeX; plan in `report/PLAN.md`.
