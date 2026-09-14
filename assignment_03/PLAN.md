# Assignment 03 — Report Plan

This file plans the **final report**: what sections it has, what goes in each section,
and — for every per-app section — exactly what to pull out of that app's notebook to
write it. It does not plan the notebooks themselves; that lives in each app's own
`<app>/PLAN.md` (`diabetes/PLAN.md`, `house_price/PLAN.md`,
`customer_behaviour/PLAN.md`), which this file references rather than repeats.

## Why this structure is an assumption, and what it's based on

Neither `REQUIREMENT.md` nor either PDF in `materials/` says anything about a report:
no required sections, no page count, no cover-page format, no submission format. This
is different from assignment_02, where a report guide existed per app plus a combined
final deliverable. `REQUIREMENT.md`'s own header states *"Project layout follows
assignment_02/ where the slides don't say otherwise"* — the report is exactly a case
where the slides say nothing, so the plan below reuses **assignment_02's actual report
shape** (recovered from its git history, commit `aaad172`, `Report_Deliverable.md`) as
the template, trimmed of what's now out of scope and extended with what
`REQUIREMENT.md` explicitly asks for this round.

**Treat this as a draft to confirm, not a settled requirement** — check with the
professor if there's a chance to ask before finalizing. In the meantime this is
concrete enough to write against, which beats waiting on an unanswerable question.

What changed relative to assignment_02's report:
- **Removed**: everything about `api/`, `web/`, `mobile/` deployment, screenshots, and
  the deployment-architecture section — `REQUIREMENT.md` explicitly puts those out of
  scope for assignment_03 ("No API, web, or mobile layer is required... `api/`,
  `web/`, `mobile/` folders from assignment_02 are out of scope").
- **Added**: a from-scratch DL section per app (architecture, forward/backward
  walkthrough, loss curve, DL-specific metrics), a 4-model comparison per app (3
  classical + 1 DL, where assignment_02 only ever compared classical models against
  each other), and a framework-level explanation of the lecture's central equation —
  this is the part that satisfies the instinct that the report should "walk through
  the construction of the deep learning model, explain what it's doing, how it works,
  and why."
- **Kept**: cover page fields, the mandatory data-representation summary table, the
  cross-application comparison table, the reproducibility section — all still directly
  applicable.

---

## Section-by-section plan

### 1. Cover page
Same fields as assignment_02: student name, student ID, class, team, lecturer (Dinh
Que Tran), semester. Nothing app-specific — write once.

### 2. Executive summary
One paragraph stating what was built (3 intelligent systems, each now compared across
3 classical ML models + 1 from-scratch NumPy DL model) plus one summary table:

| Application | Task | Dataset | Rows (vs. assignment_02) | Winning model |
|---|---|---|---|---|
| Diabetes | binary classification | BRFSS 2019+2020 | 815,711 (vs. 253,680, 3.2x) | *(fill in after notebook run)* |
| House price | regression | USA real estate (600K sample) | 600,000 (vs. 238,924, 2.5x) | *(fill in after notebook run)* |
| Customer behaviour | binary classification | Flipkart reviews | 364,401 (vs. 116,264, 3.1x) | *(fill in after notebook run)* |

Source for the dataset/row numbers: each app's own `PLAN.md` §Dataset. The "winning
model" column can't be filled until the notebooks actually run — leave it as a
placeholder in the report draft, not a guess.

### 3. Connection to Lecture 03 — Representation Learning and Deep Learning
This section exists so the per-app sections don't each have to re-derive the same
math. Write once, here:
- Restate the lecture's central equation: `ŷ = (f_L ∘ f_{L-1} ∘ ... ∘ f_1)(x)`.
- Restate "Deep Learning = Function Composition + Representation Learning +
  Optimization" (the tutorial PDF's own closing summary, §56 of that deck).
- Explain forward propagation, loss, backpropagation via the chain rule, and gradient
  descent **once, in words**, at the level of "what is happening and why," not just
  symbols — this is the generic version of what each per-app DL subsection will
  instantiate concretely.
- State explicitly what makes this "deep learning from scratch": no autograd, no
  framework — every `∂L/∂W` is derived and coded by hand, per `REQUIREMENT.md`'s
  constraint checklist.

This section is what most directly answers "explain what it's doing, how it works, and
why" — written once at the framework level, then referenced (not repeated) in each
app's DL subsection.

### 4. Per-application sections (×3): Diabetes, House Price, Customer Behaviour

Each app gets an identical subsection skeleton. Everything in it is pulled from that
app's `<app>/PLAN.md` (the "what/why" reasoning, written before the notebook existed)
and from the app's executed notebook (the actual numbers, plots, and trained-model
behavior, which don't exist until the notebook runs). Do not write the numeric parts
from memory or estimate — pull them from the executed notebook's output cells.

**4.x.1 Problem description** — one paragraph. Pull from `<app>/PLAN.md` §Problem,
lightly reworded for report prose rather than planning-note style.

**4.x.2 Dataset** — source, row count, size vs. assignment_02, why this dataset
satisfies the "bigger" rule, target variable. Pull from `<app>/PLAN.md` §Dataset and
§Target. Add the *actual* post-cleaning row count from the notebook (raw rows almost
always shrink after dropping nulls/duplicates — report both raw and clean counts, like
assignment_02's report did: "253,680 → 229,781").

**4.x.3 Data understanding & cleaning** — what the EDA found (2-4 plots with
Observation/Interpretation, pulled directly from the notebook's own EDA
Observation/Interpretation markdown cells — don't re-interpret the plots separately in
the report; the notebook cells already state the interpretation, so quote/paraphrase
them), and what cleaning was applied. Also report how each of the "Open questions" in
`<app>/PLAN.md` got resolved (e.g. diabetes: was `year` kept or dropped; house_price:
was `prev_sold_date` engineered or dropped; customer_behaviour: was `Summary`
concatenated in) — these were left open in the plan specifically so the EDA could
decide, so the report is where that decision gets recorded with its reason.

**4.x.4 Representation** — what raw columns became what feature vector, and why. Pull
from `<app>/PLAN.md` §Representation. This is where the report earns its keep
pedagogically — `REQUIREMENT.md` and the lecture both frame representation choice as
the actual point of the exercise, not just accuracy, so this subsection should be
written with real explanation, not just a column list.

**4.x.5 Classical ML models** — which 3, why (pull from `<app>/PLAN.md` §"3 classical
ML models"), and their metrics (pull from the notebook's metrics-table cell, verbatim
numbers, not re-typed from memory).

**4.x.6 From-scratch deep learning model** — the section that most needs care:
- Architecture: state `d → h1 → h2 → 1` with the actual `d` for this app (pull from
  `<app>/PLAN.md` §"From-scratch DL model"), and *why* those widths (also pulled from
  there — each app's plan already states the width reasoning).
- Forward propagation walkthrough: one short paragraph per layer, stating what it
  computes and why that activation was chosen there (ReLU for hidden layers = keeps
  the network nonlinear and cheap to differentiate; sigmoid/linear at the output =
  matches the task type). Reference section 3's generic explanation rather than
  re-deriving the chain rule from zero.
- Backpropagation walkthrough: explain *in words* what each gradient represents (e.g.
  "`dZ3 = ŷ - y` is the error signal at the output; layer 2 asks how much it
  contributed to that error via `dH2 = dZ3 · W3ᵀ`, then applies the ReLU derivative to
  zero out the contribution of any neuron that was inactive during the forward pass").
  This is the part that demonstrates understanding, not just implementation — it's
  worth writing carefully.
- Loss curve: embed the plot, describe what a healthy vs. unhealthy curve looks like
  and which this is.
- DL metrics: pull from the notebook, same as the classical models.

**4.x.7 4-model comparison** — embed both bar charts (primary metric; secondary
metrics), then the written comparison. Pull the written comparison from the notebook's
own "written comparison" markdown cell (each app's cell plan has one, e.g. diabetes
cell 41) rather than re-writing it independently — the notebook cell is where the
comparison was actually reasoned through against real numbers, so the report should
transcribe/expand it, not contradict or duplicate it from scratch.

### 5. Cross-application comparison
One table, all 3 apps × 4 models, primary metric only (Accuracy for the two
classification apps, R² for house_price) — the mandatory summary `REQUIREMENT.md`
asks for ("A top-level summary table/notebook comparing all 3 datasets × 4 models is
expected in the final report"). Pull every cell from the three per-app results tables
already produced in section 4 — this table is a transposition, not new analysis.

### 6. Discussion — what changed from assignment_02
One paragraph per app plus a short synthesis: bigger data (state the actual factor:
3.2x / 2.5x / 3.1x), a genuinely new representation ingredient this round (the
from-scratch DL model), what got harder (training time on 600K-800K rows with a
pure-NumPy loop — no GPU, no vectorized framework tricks beyond what NumPy itself
gives), and whether the DL model actually beat the classical models or not — report
the real answer from section 4/5's numbers, whichever way it goes. A DL model *not*
winning is a legitimate, useful finding here (small-network-vs-tree-ensemble on
tabular data, or vocabulary-capped DL vs. full-vocabulary linear models on text, are
both known limitations worth naming rather than glossing over).

**Specific point to include, discovered while building the diabetes and house_price
notebooks — write this as an Optimization-pillar finding, not a representation/data-
cleaning aside:** plain gradient descent (`W -= lr * dW`, no per-parameter adaptive
scaling, no Adam, nothing a framework quietly handles) has no defense against a
feature sitting on a wildly different numeric scale than the rest of the input. This
showed up independently in two apps — diabetes's `year` column (left as the literal
2019/2020 integer, dwarfing every standardized/small-integer feature next to it) and
house_price's `acre_lot`/`house_size`/`city_freq` (all heavy-tailed counts/measurements
whose raw scale produced double-digit z-scores after standardization) — and in both
cases it caused the from-scratch network to diverge or collapse to a trivial
majority/mean prediction, while every classical model trained on the exact same
feature matrix was unaffected (`sklearn`'s solvers and the tree ensembles are either
scale-invariant or robust to this kind of ill-conditioning). The reason this belongs
in the report at all is specifically *because* it is invisible when you call
`model.fit()` on a packaged implementation — it was only discoverable by building the
optimizer by hand, which is the actual point of the from-scratch requirement. Lead
with that framing (what building the optimizer yourself exposes that a framework
would silently absorb) and use the two concrete fixes as supporting evidence, not the
headline. See `PROGRESS.md`'s "Bugs found and fixed" section for the exact numbers
(loss trajectories, before/after z-scores) to cite here once writing this section.

### 7. Conclusion
Short — 1 paragraph. What was built, what was learned about representation learning
and from-scratch implementation specifically (not a restatement of section 6).

### 8. Reproducibility
Environment (Python version, key package versions, OS), `RANDOM_SEED = 42` statement,
and per-app "how to re-run the notebook end-to-end" instructions (`jupyter nbconvert
--execute ...`, matching assignment_02's style) — no API/web/mobile run commands this
round, since those don't exist.

---

## What this file deliberately does not cover

- Notebook cell layout, dataset preprocessing detail, model/architecture choices and
  their reasoning — all of that lives in `diabetes/PLAN.md`, `house_price/PLAN.md`,
  `customer_behaviour/PLAN.md`. This file only says what the report *does with* that
  material once the notebooks exist.
- Numbers that don't exist yet (metrics, winning models, actual post-cleaning row
  counts) — those get filled in from the executed notebooks, not estimated here.

## Repo structure (for reference — constructed alongside this plan)

```
assignment_03/
  PLAN.md                         <- this file (report plan)
  README.md
  REQUIREMENT.md
  materials/
    intel_sys_dev_slide_03_deepLeaning_1.pdf
    int_sys_dev_slide_03_basicML_deepLearning_04.09.pdf
  diabetes/
    PLAN.md         <- app-specific implementation plan
    data/           DATASET_2019.csv, DATASET_2020.csv
    notebook/       .gitkeep   (holds diabetes.ipynb once written)
    model/          .gitkeep   (holds model_pipeline.joblib, dl_weights.npz,
                                 feature_names.joblib, input_schema.json)
  house_price/
    PLAN.md         <- app-specific implementation plan
    data/           realtor-data-sample.csv
    notebook/       .gitkeep
    model/          .gitkeep
  customer_behaviour/
    PLAN.md         <- app-specific implementation plan
    data/           flipkart_reviews.csv
    notebook/       .gitkeep
    model/          .gitkeep
  report/
    .gitkeep        (holds Assignment_03.md / .pdf once written, per this plan)
```

No `api/`, `web/`, `mobile/` folders — out of scope per `REQUIREMENT.md`.
