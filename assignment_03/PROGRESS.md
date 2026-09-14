# Assignment 03 — Progress Tracker

Purpose: this task spans multiple long sessions (3 notebooks, 12 trained models, a
from-scratch NumPy DL implementation per app, plus a final report). This file is the
resume point — read it first in any new session before doing anything else.

## Scope (confirmed with user)

All 3 apps are in scope: `diabetes`, `house_price`, `customer_behaviour` — not just
customer_behaviour. Each app needs its own executed `.ipynb` with 3 classical ML
models + 1 from-scratch NumPy DL model, evaluated and compared. Then a final report
across all 3 apps (`report/Assignment_03.md`/`.pdf`).

**Key directive from user**: visualizations must NOT be concentrated only in the EDA
section. Every notebook must carry real plots in the classical-model section, the
DL-model section, AND the comparison section — not just a metrics table followed by
one loss curve at the end.

## What's been done

1. Read `REQUIREMENT.md` (assignment brief), root `PLAN.md` (report plan), and all
   3 per-app `PLAN.md` files.
2. **Rewrote all 3 per-app `PLAN.md` cell-by-cell plans** to add visualization cells
   outside the EDA section. Each plan's "Conventions" section now also says the DL
   training loop tracks accuracy (classification) or MAE (regression) per epoch, not
   just loss, so the loss curve becomes a paired dual-curve plot.

   New plots added per app (beyond the original ~4 EDA plots + 2 comparison bar
   charts + 1 loss curve + 1 DL confusion matrix):
   - **diabetes** (`diabetes/PLAN.md`, now 51 cells): +EDA5 correlation heatmap;
     classical section +confusion-matrix grid (1x3), +ROC overlay (3 models),
     +Random Forest feature-importance bar chart; DL section: loss curve →
     loss+accuracy dual curve, +weight histogram (`W1`), +DL feature-signal bar chart
     (mean `|W1|` per feature, comparable to the RF importance chart); comparison
     +ROC overlay all 4 models.
   - **house_price** (`house_price/PLAN.md`, now 54 cells): +EDA5 correlation
     heatmap; classical section +predicted-vs-actual scatter grid (1x3), +residual
     plot grid (1x3), +feature-importance/coefficient panel; DL section: loss curve
     → loss+MAE dual curve, +predicted-vs-actual scatter, +weight histogram, +DL
     feature-signal bar chart; comparison +predicted-vs-actual 2x2 small multiples
     (all 4 models, shared dollar axis).
   - **customer_behaviour** (`customer_behaviour/PLAN.md`, now 53 cells): +EDA5
     top-unigram/bigram frequency chart (recommend vs. not); classical section
     +confusion-matrix grid (1x3), +ROC overlay (3 models), +top-TF-IDF-terms
     diverging bar chart (Logistic Regression coefficients); DL section: loss curve
     → loss+accuracy dual curve, +weight histogram, +DL top-TF-IDF-terms bar chart
     (mean `|W1|` per term); comparison +ROC overlay all 4 models.

   Also fixed cell-number cross-references in each "Open questions" section that
   shifted because of the new EDA5 cell.

3. Data status: **all 3 datasets are present** — confirmed by directly listing each
   `data/` folder (an earlier `ls` at the very start of the session showed
   diabetes/house_price data folders as empty, but that was stale/timing — re-checked
   and all 3 CSVs exist with real sizes):
   `customer_behaviour/data/flipkart_reviews.csv` (58MB),
   `diabetes/data/DATASET_2019.csv` (113MB) + `DATASET_2020.csv` (105MB),
   `house_price/data/realtor-data-sample.csv` (48MB). No further Kaggle download
   needed. (Kaggle CLI 2.2.4 is installed with a `KGAT_...` token at
   `~/.kaggle/access_token`, usable via `KAGGLE_API_TOKEN` env var if a re-download is
   ever needed.)

## How notebooks are built

Not hand-typed cell by cell. A Python generator script (using `nbformat`)
builds the full `.ipynb` in one shot from a list of markdown/code cell
sources, then `jupyter nbconvert --to notebook --execute --inplace` (using
the `assignment_03` kernel, `--ExecutePreprocessor.timeout=1800`) runs it
end to end and writes outputs back into the file. Generator scripts live in
the session scratchpad
(`.../scratchpad/build_customer_behaviour_nb.py`,
`build_diabetes_nb.py`, `build_house_price_nb.py` once written) — **not**
committed to the repo, just a build tool. If a cell errors, fix the
generator script (not the .ipynb directly, since re-running nbconvert would
overwrite manual edits to the .ipynb anyway) and regenerate + re-execute.
One exception: the "written comparison" markdown cell's real numbers are
patched directly into the executed .ipynb with a small nbformat script
after execution (cheaper than a full re-run just to fill in one markdown
cell) — see how this was done for customer_behaviour cell 58 as the
pattern to repeat for diabetes and house_price.

Real data always turns out to differ from what a plan written before seeing
it assumed (column dtypes, schema differences between files, sentinel
values, vocabulary sizes). When that happens, the notebook's markdown cells
document the actual finding and decision inline — the plan is a starting
point, not a spec to force-fit the data into.

## Environment

Global `python` (scoop, 3.14.7) has a broken pandas (`pandas._libs.pandas_parser`
missing — mismatched/corrupt install, do not use it directly). A working venv exists
at `assignment_03/.venv` (created fresh, same as assignment_02's per-app pattern but
at the assignment root since there's no api/web/mobile split this round — matches
user's explicit instruction). Installed: numpy 2.5.3, pandas 3.0.5, scikit-learn
1.9.1, scipy 1.18.1, matplotlib 3.11.2, seaborn 0.13.2, jupyter, nbconvert 7.16.6,
nbformat, nbclient, ipykernel. Registered as Jupyter kernel name `assignment_03`
(display name "assignment_03 (.venv)") via
`.venv/Scripts/python.exe -m ipykernel install --user --name=assignment_03`.

**Always use `assignment_03/.venv/Scripts/python.exe` (or activate the venv) for
anything Python in this assignment — never the bare `python` command.** Each app's
own `requirements.txt` (numpy, pandas, scikit-learn, matplotlib/seaborn — per
REQUIREMENT.md, no DL framework) still needs to be written per app dir, matching
what's actually imported in that app's notebook.

## Notebook independence (user directive, third correction)

Each app's notebook must stand alone as a self-contained deliverable — no
notebook should reference another notebook in this assignment ("the
diabetes notebook", "the other two notebooks", "unlike in
customer_behaviour"), and none should reference **assignment_02** either
("the model assignment_02 already used for this app", "the assignment_02
report... found a weak R-squared"). Both kinds of reference got introduced
during the narration pass and were found and removed from all three
builder scripts (a plain single-line grep missed one instance that was
split across a line break in the source — worth doing a
newline-collapsed sweep, not just a line-based grep, when checking for
this again). If a model choice happens to continue something from a prior
assignment, or an architecture choice happens to parallel another app in
this same assignment, explain the reasoning on its own terms (why this
model fits this data/task) rather than by pointing outside the notebook.
customer_behaviour and diabetes were already executed when this was
caught; the fix was patched directly into the executed .ipynb (markdown
text only, no code changed, so no need for a costly full re-run) rather
than regenerating from the builder script. house_price was fixed in the
builder script before its first execution.

**This slipped again afterward**: when patching diabetes's written-comparison
cell with real post-execution numbers, the first draft named
"the customer_behaviour notebook" directly, in a sentence about a
similar linear-separability finding. Caught and fixed immediately (see
diabetes.ipynb's final markdown cell for the corrected version). Lesson:
the notebook-independence check needs to be re-applied to *every* new
piece of prose written after execution too (written-comparison patches
especially), not just to the original builder-script sweep — it is an
easy thing to slip back into when a genuinely similar finding shows up
across apps and the natural instinct is to point at where it was seen
before.

## Major finding: Review text was near-total train/test leakage (customer_behaviour)

User got suspicious of the ~0.99 F1 scores across all 4 customer_behaviour
models and asked for an explanation. Investigated directly rather than
guessing:
- **99.1% of test-set `Review` texts appear verbatim in the training set.**
  `Review` is not diverse free text — it is overwhelmingly a tiny set of
  one/two-word stock phrases repeated tens of thousands of times each
  ("Wonderful" x18,567, "Awesome" x11,501, "Great product" x11,300, ...).
  Only 3,837 unique Review texts exist across 363,233 rows (~1%).
  Combined with the earlier documented decision to keep exact-duplicate
  rows (justified then as "real signal," which is still true for the
  dataset as a whole), a random 80/20 split puts copies of the same short
  phrase on both sides ~99% of the time — the "test set" was mostly asking
  "have you seen this exact string before," not testing generalization to
  new language. Not classical leakage (the target never leaks into
  features), but the near-perfect scores substantially overstate real
  generalization.
- **User's fix, verified before implementing**: use `Summary` instead of
  `Review` as the TF-IDF input. Checked the actual numbers first:
  `Summary` present on 99.45% of rows (negligible loss from requiring it),
  165,842 unique Summary texts out of 361,229 non-null rows (~46% unique,
  vs. Review's ~1%), average 8.2 words vs. Review's 1.9. Train/test exact-
  text overlap drops from 99.1% to 59.7% with Summary — still non-trivial
  (common one-word summaries like "Good"/"Nice" still repeat a lot), but a
  real improvement, not just a cosmetic one.
- **Implemented**: switched `build_customer_behaviour_nb.py`'s TF-IDF
  fitting (both classical and DL vectorizers) from `Review` to `Summary`;
  changed the cleaning-section valid-row filter from `Review.notna()` to
  `Summary.notna()`; rewrote EDA plot 5 into a "how repetitive is each text
  field" comparison (prints + bar chart of the actual uniqueness numbers
  above) that now serves as the evidentiary basis for the representation
  choice, followed by the top-terms-by-class chart re-targeted at
  `Summary`; updated `input_schema.json`'s documented inputs and notes.
  Regenerated (80 cells) and re-executing as of this checkpoint — **check
  the notebook directly for final post-Summary-switch metrics if
  resuming**, and re-patch the "written comparison" markdown cell with
  those real numbers once execution finishes (same pattern as before).
- **Not yet done, flagged for later if it matters**: even 59.7% overlap
  is still substantial. A fully rigorous fix would additionally use a
  grouped train/test split (group by exact Summary text, e.g.
  `GroupShuffleSplit`, so no exact string spans both splits) rather than a
  plain random split. This was offered to the user as an option but not
  chosen this round — the Summary swap was the requested and implemented
  fix. Worth raising again when writing the report's limitations section,
  and worth doing the same repetition check on diabetes/house_price's
  categorical-heavy fields if similar suspicion arises there (unlikely,
  since those two apps don't have a free-text column at all).

## Resource-contention lesson: kill abandoned benchmark scripts

While iterating on house_price's DL learning rate, several standalone
benchmark scripts (`/tmp/bench_house_price_dl.py` and variants) were
launched via `run_in_background` to test candidate learning rates before
committing to one in the actual notebook. Once `LEARNING_RATE=0.01` was
chosen and the real notebook execution was launched, the earlier
benchmark script (testing `[0.01, 0.005, 0.001]` over 3000 epochs each)
was never explicitly stopped — it kept running in the background,
competing for CPU with the actual notebook's kernel process and making
the real execution take far longer than it should have (confirmed via
`Get-Process python | Select CPU`: one abandoned process had accumulated
far more CPU-seconds than the real one). Killed with
`Stop-Process -Id <pid> -Force` once identified via
`wmic process where "ProcessId=<pid>" get CommandLine`.

**Lesson**: once a benchmark/diagnostic script's purpose is served (a
hyperparameter is chosen, a bug is confirmed fixed), explicitly verify it
has actually finished or kill it — don't just move on and let it run
unsupervised in the background, especially right before launching the
real, expensive notebook execution that will compete with it for the
same CPU. Check `Get-Process python -ErrorAction SilentlyContinue |
Select Id,StartTime,CPU` if an execution seems to be taking unexpectedly
long, and cross-reference against `wmic process where
"ProcessId=<id>" get CommandLine` to identify which process is which
before killing anything.

## Bugs found and fixed before/during execution

- **diabetes DL model collapsed to majority-class prediction.** Root cause:
  `year` was left as the raw integer 2019/2020 instead of being mapped to
  0/1, so it entered the feature matrix at a scale of ~2020, dwarfing every
  other standardized/small-integer feature and blowing up the forward pass
  (sigmoid overflow at epoch 1, loss=17.49). Fixed in `build_diabetes_nb.py`
  by mapping `year` to a binary flag right where the target is derived
  (with an explanation of why binary, not one-hot, is correct for an
  exactly-2-level field — one-hot would just add a redundant mirror column,
  the dummy-variable trap; one-hot is still correct and used for genuinely
  unordered 3+-level fields like race/marital/employment/personal-doctor).
  Learning-rate re-benchmark after the fix (see below) is what determines
  the final `LEARNING_RATE`/`EPOCHS` values baked into the notebook.
- **house_price DL model would have hit the same class of bug.** Found by
  inspection before ever executing: `city_freq` (a raw count that can run
  into the thousands for popular cities) was going into the feature matrix
  unscaled, next to one-hot flags and already-standardized fields. Fixed in
  `build_house_price_nb.py` by folding `city_freq` into the same
  `StandardScaler` step as the other continuous columns (cell 21), and
  updated the downstream feature-assembly cell (23) and the representation
  markdown (17) to match. This was caught and fixed *before* the first
  execution of house_price, so no wasted run there.

**Lesson for any remaining or future notebook work in this assignment**:
before finalizing a from-scratch DL section, explicitly check the actual
numeric range of every column entering the network (not just whether it
was "supposed to be" scaled/encoded) — a single unscaled or mis-encoded
column is enough to blow up or degenerate full-batch gradient descent even
when every classical model trained on the exact same matrix works fine
(tree models don't care about scale; sklearn's LogisticRegression's
solvers are far more robust to it than plain vanilla gradient descent is).

## Checklist (update after every meaningful step)

- [x] Read REQUIREMENT.md, root PLAN.md, all 3 app PLAN.md files.
- [x] Revise all 3 app PLAN.md files to spread visualizations beyond EDA.
- [x] Write this PROGRESS.md before starting notebook work.
- [x] Confirm `diabetes/data/DATASET_2019.csv` + `DATASET_2020.csv` present.
- [x] Confirm `house_price/data/realtor-data-sample.csv` present.
- [x] Create `assignment_03/.venv` with working numpy/pandas/sklearn/matplotlib/
      seaborn/jupyter stack (global Python's pandas install is broken) + register
      Jupyter kernel `assignment_03`.
- [x] Build + execute `customer_behaviour/notebook/customer_behaviour.ipynb`
      (80 cells, COMPLETE, built via a generator script — see "How notebooks
      are built" below). **Superseded an earlier Review-text version** — see
      "Major finding: Review text was near-total train/test leakage" above
      for why. Final version uses `Summary` as the TF-IDF input, not
      `Review`. All 4 models trained, 0 cell errors, artifacts persisted,
      reload-and-verify passed. Real (honest, post-leakage-fix) results:
      Linear SVM wins by F1 (0.9550), Logistic Regression (0.9523) and the
      from-scratch DL model (0.9506) close behind, Multinomial NB lowest
      (0.9402, but with the highest recall of all four at 0.9900 — a real
      precision/recall trade-off tied to its independence assumption and
      not seeing the Price feature). Also wrote
      `customer_behaviour/requirements.txt`.
- [x] Build + execute `diabetes/notebook/diabetes.ipynb` (79 cells,
      COMPLETE). All 4 models trained, 0 cell errors, `year` scale bug
      fixed (mapped to 0/1), `LEARNING_RATE=0.1`/`EPOCHS=300` chosen from
      direct benchmarking, per-function DL narration + subsection headers +
      NumPy syntax explanations added, artifacts persisted and
      reload-verified. Real results: Gradient Boosting wins by F1 (0.2712),
      Logistic Regression close behind (0.2507), From-scratch DL (0.1990)
      beats Random Forest (0.1494) but trails the other two — written
      comparison cell patched with this reasoning (imbalanced target
      suppressing recall for every model at the 0.5 threshold; DL landing
      near LogReg since both learn smooth boundaries over the same encoded
      space while GB/RF carve axis-aligned regions instead). Wrote
      `diabetes/requirements.txt`.
      Key real-data findings baked into this notebook (differ from the original
      `diabetes/PLAN.md` estimate, which was written before inspecting the raw
      columns):
      - 2019 and 2020 files do NOT share the same schema: `high_bp_00` and
        `high_cholesterol_00` exist only in 2019. Both dropped after concat
        (documented in a markdown cell, tied to the year-as-feature question).
      - The `_00`-suffixed columns are a mix of already-ordinal numeric codes
        (education_00, income_00, gen_health_00, l_checkup_00 — paired with a
        redundant string column each, e.g. `general_health_00` string vs.
        `gen_health_00` int) and genuinely unordered nominal strings (race_00,
        marital_status_00, employment_status_00, has_personal_doctor_00, sex_00)
        that were NOT anticipated as needing one-hot in the original plan text
        (which assumed all `_00` columns were pre-encoded). One-hot is applied
        to the genuinely nominal ones only; ordinal ones stay as passthrough
        integers, matching the plan's original "no one-hot" rule for ordinal
        survey fields.
      - `avg_drinks_p_day_00` and `poor_health_days_00` are majority-missing by
        survey design (conditional questions), not data-quality noise — dropped.
      - `weight_00 == 999` and `bmi_00 > 100` are BRFSS sentinel/implausible
        values, cleaned to NaN then median-imputed.
      - Actual `d` (post one-hot) is computed at runtime, not hardcoded 33 —
        the DL architecture (`h1 = d`, `h2 = max(d//2, 8)`) adapts automatically.
      - Full-batch gradient descent used (not mini-batch) — dense feature
        matrix is small enough (~815K x ~45, comfortably fits in memory).
- [ ] Build `house_price/notebook/house_price.ipynb` per the revised 54-cell plan.
- [ ] Execute house_price notebook end-to-end, verify, persist artifacts,
      reload-verify.
- [ ] Collect actual metrics/winning-model per app from executed notebooks (do NOT
      guess/estimate — pull from notebook output cells only, per root PLAN.md's own
      rule).
- [ ] Write `report/Assignment_03.md` following root `PLAN.md`'s section-by-section
      plan (cover page → exec summary → Lecture 03 framework section → 3 per-app
      sections → cross-app comparison → discussion → conclusion → reproducibility).
- [ ] Convert report to `.docx`/`.pdf` deliverable (check how assignment_02 did this
      — pandoc, per recent commit history — and match style: Times New Roman,
      plain-ASCII per assignment_02's last commits).
- [ ] Each app's `requirements.txt` (numpy, pandas, scikit-learn, matplotlib/seaborn
      — no DL framework) — verify present/correct.
- [ ] Final pass against `REQUIREMENT.md`'s "Constraints checklist" (no
      tensorflow/torch/keras imports, manual forward/backward/update, RANDOM_SEED=42
      everywhere, no leakage, metrics appropriate per task type).

## Notation and concept explanations added (user directives, fourth, fifth, sixth)

Two more rounds of "explain this more" feedback, both now applied to all
three builder scripts:
- **Variable notation** (`X`, `W1`, `b1`, `Z1`, `A1`, the trailing layer
  digit, the `d` prefix in backward): added as a dedicated "### A note on
  notation" markdown cell right before the `forward` function in each
  notebook's DL section — that is the first place every symbol appears
  together, so it is the natural place to define them all at once rather
  than scattering partial explanations.
- **What He initialization actually is** (not just the formula): added to
  each notebook's "Weight initialization" cell — explains the
  vanishing/exploding-signal problem plain random init has across stacked
  layers, what `fan_in` means concretely per layer, why the `2` in
  `sqrt(2/fan_in)` specifically compensates for ReLU zeroing out ~half its
  input, and the contrast with Xavier/Glorot's `1/fan_in` for smoother
  activations. Named after Kaiming He (2015).

A third addition followed the same pattern: a "### How these pieces
actually train something" synthesis explaining the four-step
forward→loss(spectator)→backward→update cycle, what persists across
iterations (only W/b) vs. what's ephemeral (cache, gradients), and why
repetition is the entire training mechanism — inserted right before each
notebook's "Training loop" cell.

A fourth addition, requested as a step-by-step numbered list rather than
a paragraph (matching how the user phrases things themselves): a 6-point
"what does ReLU's clipping actually buy you" explanation in the ReLU
cell of each notebook — why any nonlinearity is needed at all (stacked
linear layers collapse into one linear layer without it), what ReLU's
on/off-switch behavior specifically does, why that lets many simple units
approximate a complex boundary together, and the cheap-derivative /
vanishing-gradient contrast with sigmoid.

**Important correction from the user**: the dying-ReLU failure mode must
NOT be added as a generic point in this ReLU explanation and must NOT be
propagated into diabetes's or customer_behaviour's ReLU cells at all —
those two models never hit that failure. It stays in exactly two places:
(1) house_price's own "Training loop" cell, where it explains that app's
specific hyperparameter history, and (2) the report plan (root
`PLAN.md`'s Discussion section addition, already written) as an
Optimization-pillar finding. A first draft mistakenly added a 7th point
about dying ReLU to customer_behaviour's ReLU cell and was corrected
before customer_behaviour's next patch — keep this same scoping if
editing any of these cells again.

A sixth and seventh addition, both requested directly:
- The forward-propagation cell now includes an explicit arrow-chain
  pipeline diagram (`X -> Z1 = X@W1+b1 -> A1 = relu(Z1) -> ...`)
  reinforcing "multiply first, activate second" as a visual reference,
  not just prose.
- The backward-propagation explanation was rewritten from "what this
  code does" (mechanical, operation-by-operation) into "what this means"
  (a numbered list framed around who is to blame for the final mistake,
  and how that blame gets distributed backward layer by layer: `dZ3`
  is the signed error itself, `dW3` asks how much each hidden unit's
  connection deserves blame, `db3` is the average systematic offset,
  `dA2` redistributes blame onto hidden units by connection strength,
  `dZ2` gates that blame to only units that were actually switched on,
  and `dW1`/`dW2` repeat the pattern one layer further back). Same
  pattern applied to all three notebooks, with house_price's version
  adapted for the MSE/no-sigmoid case and explicitly tying `dZ2 = dA2 *
  relu_derivative(...)` to the dying-ReLU incident described in that
  app's own training-loop cell (appropriately scoped to house_price
  only, per the earlier correction).
- Following that, the "How these pieces actually train something"
  four-step cycle (forward/loss/backward/update) was reformatted from a
  prose paragraph into an explicit numbered list across all three
  notebooks, matching the user's own preferred style.

A fifth addition to the same ReLU cell: an explicit "could this be any
random function?" point (now point 3 of what was a 6-point list, making
it 7), answering the user's direct question — no, it has to be
deterministic, differentiable almost everywhere, and numerically
well-behaved; ReLU is one member of a family of valid choices (sigmoid,
tanh, Leaky ReLU, GELU, etc.) that satisfy those constraints, not the
only mathematically valid option.

**Status as of this checkpoint** (seven markdown-only patches total:
notation key, He-init explanation, training-cycle synthesis [now a
numbered list], 7-point ReLU explanation, forward-pipeline diagram,
meaning-focused backward walkthrough, plus the written-comparison
re-patch):
- **diabetes.ipynb**: all seven patched in directly via nbformat — DONE,
  no further action needed here.
- **customer_behaviour.ipynb**: same seven patches all applied directly
  via nbformat too (after its second-latest execution, 86 cells, 0
  errors, written-comparison re-patched with real numbers since
  regeneration resets it to the placeholder) — DONE.
- **house_price.ipynb**: builder script (`build_house_price_nb.py`) has
  all seven edits plus the tuned DL hyperparameters
  (`LEARNING_RATE=0.01, EPOCHS=2000`, chosen after finding a dying-ReLU
  collapse at higher rates — see "Bugs found and fixed" above). The
  currently-running execution was started before ALL SEVEN of these
  markdown edits, so once it finishes: (1) check for 0 errors, (2)
  extract the real DL metrics/loss trajectory, (3) patch in all seven
  pieces directly via nbformat, using the diabetes/customer_behaviour
  patches above as the template (search for each cell's old text as it
  originally read — the exact `old`/`new` strings used are in the
  `patch_*.py` scripts left in the scratchpad directory from this
  session if still present) — do NOT regenerate/rerun again just for
  markdown, that would waste another ~15-20 min for no code-level
  change, (4) patch the written-comparison placeholder with real
  numbers, (5) write `house_price/requirements.txt` (not yet created).
  **Also note**: this execution ran much longer than expected because an
  abandoned benchmark script was still competing for CPU in the
  background (see "Resource-contention lesson" above) — it has since
  been killed, so the notebook execution should proceed at normal speed
  from this point on if it hasn't already finished.

## Cell-level detail (user directive, second correction)

First pass at customer_behaviour and diabetes still fell short in one
specific way even after the "natural prose" fix below: the classical-model
and from-scratch-DL sections were mostly bare code cells run back to back
with no markdown narrating what each step does or why, while the EDA
section had generous prose. Concretely, the problems flagged were:
- Model-fitting cells (Logistic Regression, Linear SVM, Random Forest,
  Gradient Boosting, etc.) had no markdown before them explaining why that
  model, what its hyperparameters mean, or what to expect — just
  instantiate-and-fit code back to back with the next model's code.
- The entire from-scratch DL section (`relu`, `sigmoid`,
  `binary_cross_entropy`, weight init, `forward`, `backward`, the training
  loop) was a sequence of code cells with only inline code comments, not
  markdown cells narrating each function: what it computes, why the formula
  is what it is, why this hyperparameter value.

**The fix going forward**: every code cell in the classical-model and DL
sections needs a markdown cell immediately before (or after, whichever
reads more naturally) it that explains, in the same natural-prose style as
the EDA section, what that specific piece of code is about to do and why —
not just restating the code, but narrating the reasoning a person would
walk through if they were building this by hand. This applies to: each
individual model's fit cell, the predictions/metrics cells, and every
function-definition cell in the DL section (relu, sigmoid, loss function,
weight init, forward, backward, training loop hyperparameters). The
existing EDA and comparison sections were already at the right density and
do not need to be redone.

## Writing style for notebooks and report (user directive)

All markdown prose in every notebook, and the final report, must read as
natural writing:
- No em dashes anywhere in prose (code comments should avoid them too).
- Avoid short, clipped, "telegraphic" sentences and formulaic templates —
  the first draft of customer_behaviour's markdown cells used a rigid
  "**Observation:** ... **Interpretation:** ..." bolded-label pattern with
  heavy em-dash use; this was rewritten into flowing explanatory paragraphs
  before execution, and that rewritten version is the pattern to match going
  forward for diabetes, house_price, and the final report.
- Prioritize explaining *what the code is doing and why* (the reasoning,
  the trade-off, the decision process) over simply *reporting that something
  was done* — narrate the thinking, not just the outcome. E.g. instead of
  "Cleaned Price column. Result: N rows.", write about why the raw column
  needed cleaning, what was actually wrong with it, and what the numbers
  turned out to mean.
- This applies to every markdown cell (EDA interpretation, cleaning
  decisions, representation reasoning, DL architecture explanation,
  comparison write-ups) and to the final report's prose sections.

## Conventions to keep consistent across all 3 notebooks

- `RANDOM_SEED = 42` everywhere (splits, shuffles, estimators, DL weight init).
- Train/test split 80/20, computed once, reused identically for classical + DL models.
- Scaling/vectorization fit on train only, applied to test (no leakage).
- He-init: `np.random.randn(fan_in, fan_out) * sqrt(2/fan_in)`, biases zero.
- One logical operation per cell (function def, plot, fit, explanation) — literal
  cell-by-cell plans in each `<app>/PLAN.md` are not guidelines, follow them as-is
  (renumbered versions now current as of this file).
- DL training loop tracks loss + a second metric (accuracy or MAE) per epoch for
  dual-curve plots — this is the mechanism that satisfies the "not just EDA
  visualizations" directive inside the DL section.
- No `tensorflow`/`torch`/`keras` imports anywhere; scikit-learn only for the 3
  classical models, never for the DL model itself.

## Notes / decisions carried over from planning

- customer_behaviour: Naive Bayes fits on TF-IDF text block only (non-negative
  requirement), not on Price — documented as a deliberate scope difference from
  LogReg/SVM which use TF-IDF ‖ Price.
- customer_behaviour DL model input is capped at `max_features=5000` TF-IDF
  (separate vectorizer from the classical models' full-vocab one) — stated
  engineering trade-off for a dense NumPy forward/backward pass.
- house_price DL model has **no output activation** (linear, regression) — the one
  place the architecture deliberately differs from the diabetes/customer_behaviour
  sigmoid+BCE pattern.
- Winning-model columns in report tables must stay as placeholders until real
  notebook numbers exist — root `PLAN.md` explicitly forbids guessing these.

## Status update: house_price finished, diabetes class-balancing in flight

**house_price.ipynb: DONE.** Execution completed clean (0 errors, 82 cells,
LEARNING_RATE=0.01, EPOCHS=2000, no dying ReLU). All seven conceptual-explanation
patches confirmed present (ReLU 7-point list, MSE loss, He-init full explanation,
notation key, forward-pipeline diagram, meaning-focused backprop walkthrough,
four-step training-cycle list) — a full diff against a dry-run build of the
builder script confirmed the executed notebook now matches it exactly except for
the intentionally-patched written-comparison cell. Real numbers: Random Forest
wins on every metric (RMSE 1,004,048 / MAE 225,937 / R2 0.3980), Gradient Boosting
second (R2 0.2397), From-scratch DL third (R2 -1.1044), Linear Regression worst
(R2 -48.3632). The negative R2 values are explained in the written-comparison cell:
price is extremely heavy-tailed (median ~$315K, max $875M), `expm1` amplifies
log-space error into huge dollar error for the priciest listings, and tree models
are protected because leaf predictions can't extrapolate past training ranges the
way an unbounded linear output (Linear Regression and the DL model's final layer)
can. `house_price/requirements.txt` still needs to be written (not yet created).

**customer_behaviour.ipynb: header + prose fixes applied.** User caught two issues
via live review: (1) the DL section was missing `###` subheaders that diabetes/
house_price both have (ReLU, Sigmoid, BCE, Weight init, Backprop, Inference,
Metrics) — patched into both the executed notebook and `build_customer_behaviour_nb.py`.
(2) The representation-section markdown still had a stale, factually backwards
leftover sentence claiming "the final representation uses Review alone" (a relic
from before the Review-to-Summary switch) — rewritten to correctly state Review is
dropped entirely, and to state the actual 416,770 (classical) vs 5,000 (DL)
TF-IDF vocabulary sizes explicitly, tying that gap to why the from-scratch DL model
trails the classical models on this app.

**Em-dash sweep**: found and fixed literal U+2014 em-dash characters that had crept
into all three builder scripts (5 each) and into already-executed diabetes.ipynb
(1) and customer_behaviour.ipynb (2) — replaced with `--` everywhere per the
standing no-em-dash rule. house_price.ipynb was already clean after its patches.

**diabetes F1 scores investigated — root cause found and fix in progress.**
User asked why every diabetes model's F1 was terrible (GB 0.2712, LogReg 0.2507,
DL 0.1990, RF 0.1494) and whether it's fixable. Diagnosis (verified with a
standalone script, not yet the notebook): target is imbalanced (~15.6% positive),
all 4 models use the default 0.5 threshold with no reweighting. Sweeping the
threshold on a held-out LogisticRegression fit took F1 from 0.232 to 0.466;
`class_weight="balanced"` at the default threshold got to 0.456, essentially the
same ceiling. Secondary, smaller factor: `high_bp_00` and `high_cholesterol_00`
exist only in the 2019 survey file and get silently dropped when 2019+2020 are
concatenated (schema mismatch) — normally top BRFSS diabetes predictors; adding
them back on the 2019 subset alone lifted AUC 0.806->0.819, F1 0.241->0.292, but
this is a smaller effect than the threshold/weighting fix and has NOT been acted
on (would require re-deriving a combined-schema strategy; out of scope for now).

Checked `REQUIREMENT.md`: DL loss is specified only as "binary cross-entropy," no
stance on weighting either way, so a weighted variant doesn't violate the spec.
User initially asked for threshold tuning across all 4 models, documented (why +
how) in the notebook, then changed direction mid-implementation: **implement
balanced class weighting instead**, for both the classical models and the
from-scratch DL model. Threshold-tuning work was abandoned before any code was
written into the notebook (only a plan existed, no cells inserted) — safe to
ignore/forget.

**What was implemented for class balancing (in `build_diabetes_nb.py`, then
written out to `diabetes/notebook/diabetes.ipynb`, execution launched but not yet
confirmed complete as of this checkpoint — check background task `btjwx4lvi` /
file `diabetes/notebook/diabetes.ipynb`'s mtime and cell outputs first thing on
resume):**
- Added `from sklearn.utils.class_weight import compute_sample_weight` to imports.
- New markdown+code cell right after the "## Classical machine learning models"
  header: explains WHY (average-loss-dominated-by-majority-class mechanism, not
  just "imbalance exists") and HOW (`compute_sample_weight("balanced", y_train)`,
  the `n_total/(n_classes*n_in_class)` formula, printed per-class weight values).
- `log_reg.fit`, `random_forest.fit`, `gradient_boosting.fit` all now pass
  `sample_weight=sample_weights_train`.
- `binary_cross_entropy(y_true, y_pred, weights=None, eps=1e-9)`: new optional
  `weights` param, multiplies per-row loss before the mean (this is a proper
  weighted mean with no extra renormalization needed, since
  `compute_sample_weight("balanced", ...)` is scaled so `sum(weights) == n_rows`).
  Markdown updated with a short paragraph explaining this mirrors the classical
  models' `sample_weight`.
- `backward(y, cache, W2, W3, weights=None)`: `dZ3 = (cache["A3"] - y) * weights`
  when weights given; every downstream gradient (`dW3`, `db3`, `dA2`, ... `dW1`)
  already propagates correctly with no other changes needed, since they're all
  just linear functions of `dZ3`. Backprop markdown's numbered walkthrough point 1
  got one added sentence explaining the weight multiply.
- Training loop cell: added `sample_weights_col = sample_weights_train.reshape(-1, 1)`,
  passes `weights=sample_weights_col` into both `backward(...)` and the
  `binary_cross_entropy(...)` call used for the printed/plotted training loss.

**Still pending, in this exact order, on resume:**
1. Check background task `btjwx4lvi` (nbconvert executing diabetes.ipynb) —
   read its output file, confirm 0 cell errors.
2. Pull the new real metrics for all 4 models (classical results table, DL
   metrics table, combined_results) — expect meaningfully higher recall/F1 than
   the pre-fix numbers logged above, at some precision cost.
3. Patch the "### So which model actually wins, and why" markdown cell (currently
   still has the OLD pre-fix numbers: GB 0.2712/LogReg 0.2507/DL 0.1990/RF 0.1494)
   with the new real numbers and updated reasoning — both in the executed
   notebook (nbformat patch or accept the fresh execution's version) and confirm
   `build_diabetes_nb.py`'s copy matches.
4. Re-verify no em-dashes were reintroduced by this new prose (grep for U+2014).
5. Re-verify model artifacts (`diabetes/model/model_pipeline.joblib`, `dl_weights.npz`)
   got re-saved correctly at the end of the notebook (they're saved by cells near
   the end, after `combined_results` — should happen automatically as part of the
   same execution run).
6. Still outstanding from before this checkpoint: write `house_price/requirements.txt`.
7. Longer-term, not started: the final report (`report/Assignment_03.md`/`.pdf`).

## Status update: diabetes class-balancing confirmed working, fully closed out

Background task `btjwx4lvi` completed clean (0 errors, 81 cells). Balanced sample
weighting worked as predicted: F1 nearly doubled for every model and all four
converged tightly together, a striking change from the pre-fix spread.

Real post-fix numbers:
- Logistic Regression: Accuracy 0.7236, Precision 0.3318, Recall 0.7617, F1 0.4622
- Random Forest: Accuracy 0.7146, Precision 0.3272, Recall 0.7861, F1 0.4621
- Gradient Boosting: Accuracy 0.7149, Precision 0.3279, Recall 0.7893, F1 0.4633
- From-scratch DL: Accuracy 0.7343, Precision 0.3388, Recall 0.7401, F1 0.4648 (winner by F1)

All four now sit within 0.003 F1 of each other (previously spread from 0.1494 to
0.2712), which is itself the interesting finding: most of what separated the four
models before was really differences in how aggressively each one's *default*
training leaned toward the majority class, not real differences in separability.
Recall rose from an 8.5-17.9% pre-fix range to 74.0-78.9%; the cost was precision
dropping from 0.53-0.61 to ~0.33 and accuracy dropping from ~85% to the low 70s —
a real, expected trade-off from balanced weighting, not a free lunch.

Patched the "### So which model actually wins, and why" cell in the executed
notebook with this real analysis (`patch_diabetes_comparison.py`, scratchpad).
Left `build_diabetes_nb.py`'s copy of that cell as the placeholder, matching the
existing convention already used in the house_price and customer_behaviour
builder scripts (per this file's earlier note: winning-model text stays a
placeholder in the builder/plan files, only the executed notebook gets real
numbers).

Verified: 0 em-dashes in the new prose; model artifacts re-saved correctly
(`model_pipeline.joblib` picked Gradient Boosting as best-by-F1 among the three
classical models, `dl_weights.npz` saved, reload-and-verify passed).

Also fixed, live user catch: house_price's Data Understanding section had a
sentence referencing "the real estate dataset used in the previous assignment"
(a cross-assignment reference violating the standing rule) — rewritten to justify
the log1p-target choice with its own concrete example ($50k miss on a $300k home
vs. a $5M home) instead of pointing at assignment_02. Fixed in both the executed
notebook and `build_house_price_nb.py`. Swept all three executed notebooks for
similar stray "previous assignment" / "assignment_02" / "the diabetes notebook"
etc. references afterward — the only other two hits ("the other two...") were
legitimate in-notebook references to sibling classical models, not cross-notebook
references, so no further action needed.

`house_price/requirements.txt` written (same template as diabetes/customer_behaviour).

**Remaining work, all three per-app notebooks are now functionally complete:**
- Final report (`report/Assignment_03.md`/`.pdf`) across all 3 apps, per root
  `PLAN.md` — not yet started. This is the only major piece left in the assignment.
