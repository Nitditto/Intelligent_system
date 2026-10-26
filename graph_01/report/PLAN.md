# Graph 01 — Report Plan

**Output:** `report/Graph_01.tex` → `Graph_01.pdf` (XeLaTeX, English, same layout as the earlier reports).
**Author:** Nguyễn Văn Trường (B23DCCE095) · Intelligent System Development · Lecturer: Assoc. Prof. Dinh Que Tran, Ph.D. (class code: ask before adding).

This file plans the report: where its structure comes from, what goes in each section, which file produces each table and figure, and what each result has to explain.

## 1. Sources of the structure

There is no separate written brief for graph_01. The task, as given, is: *build a diabetes Knowledge Graph with Neo4j from Vinmec data, then a website to retrieve the data*.

| Source | What it decides |
|---|---|
| Lecture *From DL to Knowledge Graph, RAG and Intelligent Agents*, **§37 Recommended Final Report Structure** | **The section list.** Its sections are: Introduction, Problem Definition, Data Collection, Deep Learning Model, Knowledge Graph Design, RAG Architecture, Agent Architecture, System Implementation, Experimental Evaluation, Error Analysis, Limitations, Conclusion. This project has no deep-learning model, RAG pipeline or agent, so those three sections are left out and the omission is stated in §1 (see the open questions in §6). |
| Same lecture, **§35 Mini Project 3: Knowledge Graph** | **The requirements to prove**: ≥ 50 entities, ≥ 100 relationships, ≥ 5 relationship types, ≥ 3 entity types, ≥ 5 Cypher queries |
| Lecture note *Knowledge Graph and RAG*, **§22 Exercise 1**, §5.1, §23.4 | Also required: triples stored as CSV `subject, relation, object`; graph construction; visualisation; one-hop and two-hop queries; bounded 1–2-hop graph retrieval |
| Same lecture, §30.2 *Knowledge Graph Failure*, §29 *Evaluation* | The error categories used in Error Analysis and Limitations (missing entities, incorrect relationships, duplicates, outdated data), and evaluating systematically rather than "it gives an answer" |
| `assignment_08/report/Assignment_08.tex` | **Format only**: preamble, cover page, table style, `make_tables.py` → `table_*.tex`, an `images/` folder. Not its section list. |
| `assignment_07/report/PLAN.md`, `extract_figures.py` | Workflow: extract figures from the executed notebook, generate tables with a script, never type numbers by hand |

## 2. Requirement → section map

| # | Requirement | Report section | Evidence |
|---|---|---|---|
| R1 | Data from Vinmec about diabetes | §3 Data Collection | crawl funnel, relevance-filter table, example page |
| R2 | A Knowledge Graph (≥ 50 entities, ≥ 100 relations, ≥ 5 relation types, ≥ 3 entity types) | §4 KG Design, §5.2 | schema figure, entity/relation tables, checklist table |
| R3 | Triples as CSV `subject, relation, object` | §4.4 | CSV excerpt |
| R4 | Stored and queried in Neo4j, ≥ 5 Cypher queries, one-hop and two-hop | §5.3, §6.2 | data model, loader listing, 6+ queries with real results |
| R5 | Visualisation | §4.5, §6 | NetworkX figures, heatmaps, neighbourhood figure, website graph |
| R6 | Graph retrieval (bounded hops) | §5.4 | `graph_facts` listing, 1-hop vs 2-hop counts |
| R7 | Website for retrieving the data | §5.5 | architecture, API table, screenshots |
| R8 | Systematic evaluation | §6 Experimental Evaluation | precision on 50 sampled facts, likely range, support thresholds |
| R9 | Error analysis and limitations | §7, §8 | error table, fix and its trade-off, failure list |

## 3. Sections

Target length: **35–45 pages**. Each section lists its content, its sources and the **why** it must explain. Findings are interpreted only in §6–§7.

**Writing rule (from the user): explain more, not less.** Every method step, figure, table and listing gets a full explanation in plain language:
- what it is
- why it was done this way
- what alternative was tried or rejected, and why that would fail
- a concrete example from the real data (an actual Vinmec sentence, heading or number)
- how to read the figure or table, and what it shows

Explain from scratch, the same way as the notebook. Formulas are kept to a minimum and always explained in words. No generic filler.

### Cover page + table of contents
Title: *"Graph 01: A Diabetes Knowledge Graph from Vinmec Health Articles with Neo4j"*. Then course, author, ID, lecturer, date, and the table of contents on page 2.

### Summary (½ page)
- One paragraph: 1,096 articles → 127 entities, 339 facts, 12 relation types, stored in Neo4j, with a web explorer.
- Four takeaways with numbers:
  1. Section headings are the key signal: they turn free text into typed facts.
  2. Requiring support from ≥ 2 articles filters out noise.
  3. Precision is 48/50 (0.96, likely range 0.87–0.99), and both errors had the same cause.
  4. The graph answers two-hop questions that no single article states, and every answer links back to its sources.

### §1 Introduction
- The problem: Vietnamese diabetes information is spread across hundreds of articles, which a person can read but a program cannot query.
- Why keyword search is not enough: synonyms ("uống nhiều" / "khát nước"), facts spread over several articles, no way to combine facts across articles.
- Why a Knowledge Graph: explicit typed facts, multi-hop questions, explainable paths, provenance.
- Objectives R1–R9 and the requirement → section table.
- Scope: Knowledge Graph + retrieval website only. No deep-learning model, RAG or agent (lecture §37 sections 4, 6 and 7 are omitted).

### §2 Problem Definition
- Input: public Vinmec articles. Output: a typed graph plus a way to query it.
- Questions the system must answer, each mapped to a query type:
  - What are the symptoms of type X? (one-hop)
  - Which organs can type 2 damage? (two-hop)
  - Which types match these symptoms? (aggregation)
  - How is A connected to B? (paths)
  - Where does this fact come from? (provenance)
- Success criteria: the lecture's minimums (R2, R4); precision measured on a random sample; every fact traceable to articles.

### §3 Data Collection (R1)
Sources: `kg/crawl.py`, `dataset/raw/index.csv`, notebook Part 2.
1. **Why Vinmec**: Vietnamese, written or reviewed by doctors, structured disease pages, ~1,000 diabetes articles.
2. **Finding pages**: the search page is rendered by JavaScript, so the sitemap is used instead (~222,000 URLs). Slug keywords are split into strong and weak; `thao-nhat` is excluded. Result: 1,197 candidates. Politeness: `robots.txt` and 1 request per second.
3. **Relevance filter**: why removing accents causes collisions (*tiểu dương*, *đại dương*), the four keep-rules, and a **table** of kept/rejected counts by slug strength (934/10 and 162/85) with rejected examples.
4. **Parsing**: section headings, paragraphs, bullets; boilerplate is cut. **Figure**: the saved type 2 disease page (sections and bullets).
5. **Volume table**: 1,096 pages, 1.13 M words, median 1,044 words per page, publication years. **Figure**: words per page and publication year.

### §4 Knowledge Graph Design (R2, R3, R5)
Sources: `kg/vocab.py`, `kg/extract.py`, notebook Parts 1, 3, 4.
1. **Triples and the graph**: what a fact is, and why typed nodes and named relations.
2. **Schema**: entity types (table: type, count, examples) and relation types (table: relation, from → to, count, meaning). **Figure**: layered schema diagram.
3. **Extraction method**, each step with its *why*:
   - the vocabulary and entity normalisation (106 entities, 320 aliases)
   - alias matching: word boundaries, longest alias first, accents kept
   - the section-heading rule (heading → allowed types table), plus the `about_disease()` check
   - choosing the subject from the title (5 diabetes types)
   - support counting (≥ 2 articles; disease pages; ≥ 3 for side effects), with the A/B/C worked example
   - fixed food polarity; side effects from single sentences with the "thiếu" exception
   - curated facts (drug class, organ, thresholds), labelled `source = curated`
   - **Figure (TikZ)**: the pipeline from page → sections → allowed types → matches → votes → filtered facts
4. **Output files**: the four CSVs and what each is for; an excerpt of `triples.csv` (R3).
5. **The resulting graph** — figures:
   - full NetworkX graph
   - type 2 one-hop neighbourhood
   - entities per type and facts per relation
   - support distribution
   - complication × type heatmap and symptom × type heatmap
   - *Why it matters*: gestational diabetes gets its own pregnancy complications, which shows the subject rule works.
6. **Mini Project 3 checklist table**: requirement vs achieved (50 vs 127, 100 vs 339, 5 vs 12, 3 vs 10, 5 vs 14 queries).

### §5 System Implementation (R4, R6, R7)
Sources: `kg/load.py`, `kg/queries.cypher`, `kg/retriever.py`, `api/`, `web/`.
1. **Architecture figure (TikZ)**: crawler → extract → CSV → Neo4j → API → web, and notebook / Cypher Browser.
2. **File map table**: each folder and its role.
3. **Neo4j**: why a graph database rather than tables. The data model (two labels per node, evidence stored on relationships, Article nodes). Loader listing: constraint, `UNWIND` + `MERGE`, and why each is used.
4. **Graph retrieval**: `graph_facts` listing; 1 hop gives 5 facts and 2 hops give 230 for *Hạ đường huyết* (lecture §23.4).
5. **Website**:
   - Why an API in between: the browser never touches the database; read-only queries.
   - Endpoint table: `/stats`, `/entities`, `/entities/{name}`, `/evidence`, `/symptom-check`, `/paths`.
   - Screenshots: home, an entity page with sources open, diabetes-type tabs, symptom checker with results, connections.
   - Design choices: shareable URLs, the evidence filter, plain-language relation names, loading/error states.
6. **Code listings** (short, the core only): vocabulary entry, `allowed_labels` / `about_disease`, the support filter, the loader, one API endpoint.

### §6 Experimental Evaluation (R4, R8)
Sources: notebook Parts 6, 8, 9; `evaluation/manual_audit.csv`; `kg/evaluate.py`.
1. **Query results** (Mini Project 3 ≥ 5 queries). Six queries, each with its Cypher, a results table and one line of interpretation:
   - one-hop Metformin
   - two-hop organs damaged by type 2
   - type 1 vs type 2 symptoms (the type-1-only ones are the signs of ketoacidosis)
   - symptom checker
   - risk factor → organ path
   - provenance
2. **Graph analysis**: degree; directed vs undirected PageRank, and why directed PageRank rewards dead-end nodes; article mentions.
3. **Fact precision**: protocol (50 random facts, seed 42, judged against all supporting articles); 48/50 = 0.96, likely range 0.87–0.99; per-relation table.
4. **Support thresholds**: facts remaining at support ≥ 1/2/3/5/10 (301/295/252/161/70).

### §7 Error Analysis
- The two errors: a fact about a complication attached to the disease. Table: fact | what the article actually said.
- The cause: broad heading keywords ("nguyên nhân gây ngứa", "yếu tố nguy cơ gây tai biến").
- The fix (`about_disease`) and its effect: 345 → 339 facts. Table of the 6 removed facts, 2 correct removals and 4 true facts lost. The precision/recall trade-off, and why `support` is kept on every fact.
- Other observed errors: "Insulin → nhiễm toan ceton" (a cause mistaken for a side effect); the *táo* / *táo bón* alias collision; general-node dominance; weak co-occurrence ranking.

### §8 Limitations
Lecture §30.2 categories, plus:
- the fixed vocabulary, so recall is unknown
- subjects chosen from titles only
- fixed food polarity
- a single source with articles of different ages
- curated facts
- precision judged against the source, not against clinical guidelines
- not medical advice

### §9 Conclusion and Future Work
- What was achieved: R1–R9 with numbers.
- Future work: measuring recall on annotated pages; extending the vocabulary semi-automatically; extracting sentence-level relations with an LLM; connecting to the assignment_08 RAG pipeline (KG + RAG); scheduled re-crawls for freshness.

### Appendix A — Reproduction
Commands in order (crawl → extract → load → notebook → API → web) and the file map.

## 4. Figures and tables: where each comes from

| Item | Produced by | File in `report/images/` |
|---|---|---|
| Pipeline diagram, system architecture | TikZ in the `.tex` | — |
| Schema diagram, full graph, type 2 neighbourhood, heatmaps, support histogram, entities/relations bars, PageRank, words/years | `report/extract_figures.py` (PNG outputs of `notebook/01_diabetes_kg.ipynb`) | `schema.png`, `kg_full.png`, `kg_type2.png`, `heat_complications.png`, `heat_symptoms.png`, `support_hist.png`, … |
| Website screenshots (5) | Playwright against the running site (`report/take_screenshots.py`) | `web_home.png`, `web_entity.png`, `web_types.png`, `web_symptoms.png`, `web_paths.png` |
| Crawl funnel, relevance filter, entity types, relations, support thresholds, precision per relation, query results | `report/make_tables.py` (reads `dataset/raw/index.csv`, `dataset/processed/*.csv`, `evaluation/manual_audit.csv`, Neo4j) | `table_*.tex` |

Every number in the text is taken from these generated tables or from executed notebook outputs. None are typed by hand.

## 5. Work order

1. `extract_figures.py`: export the notebook figures to `report/images/`.
2. `take_screenshots.py`: website screenshots (API + site running).
3. `make_tables.py` → `table_*.tex`.
4. Write `Graph_01.tex` section by section, following §3.
5. Compile with XeLaTeX twice, check every figure, table and reference, and the page count.
6. Final check: every number matches the notebook and evaluation files; requirement map R1–R9 complete.

## 6. Decisions

1. **Structure:** lecture §37 without the deep-learning, RAG and agent sections, plus two extra sections: **§6 Visualisation** (all graph figures, each explained) and **§8 Demo** (website and Neo4j screenshots, walked through as user scenarios). Final order: Introduction, Problem Definition, Data Collection, Knowledge Graph Design, System Implementation, Visualisation, Experimental Evaluation, Demo, Error Analysis, Limitations, Conclusion.
2. **Cover page:** taken from `assignment_08/report/Assignment_08.tex`, with the author set to Lưu Anh Dũng (B23DCDK036). The class code is left out because it is not known.
3. **Screenshots:** only the website (no Neo4j Browser).
4. **Language:** English throughout.
