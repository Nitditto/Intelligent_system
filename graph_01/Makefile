# Diabetes Knowledge Graph: one command per step.
# Each step writes a file; make skips a step when that file is newer than its inputs.
# Steps without an output file (Neo4j load, notebook run) leave a marker in .make/.
#   make all       everything that is out of date: setup -> extract -> load -> notebook -> report
#   make app       API + website together (Ctrl+C stops both)
#   make help      list every target

PY      := .venv/bin/python
NEO4J   := neo4j-diabetes
NB      := notebook/01_diabetes_kg.ipynb
STAMP   := .make

RAW       := dataset/raw/index.csv
# triples.csv stands for all four CSV files, which extract.py writes together
PROCESSED := dataset/processed/triples.csv
VENV      := $(STAMP)/venv
PDF       := report/Graph_01.pdf

.PHONY: help setup crawl extract neo4j load notebook api web app report all clean

help:  ## list targets
	@grep -E '^[a-z]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-9s %s\n", $$1, $$2}'

# ---------------------------------------------------------------- environment
setup: $(VENV) web/node_modules  ## install Python and web dependencies (only if missing or changed)

$(VENV): requirements.txt
	test -d .venv || python3 -m venv .venv
	$(PY) -m pip install -q -r requirements.txt
	@mkdir -p $(STAMP) && touch $@

web/node_modules: web/package.json
	cd web && npm install
	touch web/node_modules

# ---------------------------------------------------------------- pipeline
crawl: $(RAW)  ## download Vinmec pages (only if never crawled; the crawler itself resumes)

$(RAW): | $(VENV)
	$(PY) kg/crawl.py

extract: $(PROCESSED)  ## pages -> dataset/processed/*.csv (when the rules or pages change)

$(PROCESSED): $(RAW) kg/extract.py kg/vocab.py | $(VENV)
	$(PY) kg/extract.py

neo4j:  ## start Docker and the Neo4j container if they are not running
	@docker info >/dev/null 2>&1 || { open -a Docker; echo "Starting Docker..."; until docker info >/dev/null 2>&1; do sleep 2; done; }
	@docker start $(NEO4J) >/dev/null 2>&1 || docker run -d --name $(NEO4J) -p 7474:7474 -p 7687:7687 \
		-e NEO4J_AUTH=neo4j/diabetes123 -v "$(CURDIR)/neo4j_data:/data" neo4j:5
	@until curl -s localhost:7474 >/dev/null; do sleep 1; done

load: $(STAMP)/loaded  ## CSV -> Neo4j (when the CSV files change)

$(STAMP)/loaded: $(PROCESSED) kg/load.py
	@$(MAKE) -s neo4j
	$(PY) kg/load.py
	@mkdir -p $(STAMP) && touch $@

notebook: $(STAMP)/notebook  ## execute the notebook (when the graph or the notebook changes)

$(STAMP)/notebook: $(STAMP)/loaded $(NB) $(wildcard kg/*.py)
	$(PY) -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 $(NB)
	@touch $@

report: $(PDF)  ## compile the report (when the .tex, tables or images change)

$(PDF): report/Graph_01.tex $(wildcard report/table_*.tex) $(wildcard report/images/*.png)
	cd report && xelatex -interaction=nonstopmode Graph_01.tex >/dev/null && xelatex -interaction=nonstopmode Graph_01.tex | grep "Output written"
	cd report && rm -f Graph_01.aux Graph_01.log Graph_01.out Graph_01.toc

all: setup extract load notebook report  ## run every step that is out of date

# ---------------------------------------------------------------- website
api: neo4j  ## FastAPI on :8000
	.venv/bin/uvicorn api.main:app --reload --port 8000

web: web/node_modules  ## website on :5173
	cd web && npm run dev

app: neo4j web/node_modules  ## API + website together
	@trap 'kill 0' INT TERM; \
	.venv/bin/uvicorn api.main:app --port 8000 & \
	(cd web && npm run dev) & \
	wait

clean:  ## remove caches, LaTeX build files and markers (next make reloads and reruns)
	rm -rf $(STAMP) kg/__pycache__ api/__pycache__ web/dist
	cd report && rm -f *.aux *.log *.out *.toc
