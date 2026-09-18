# Assignment 04 — Progress Tracker

Purpose: this task spans multiple long sessions (3 notebooks, 9 trained models across
3 implementations each, a from-scratch NumPy MLP and two from-scratch NumPy CNNs, plus
a final report and a slide deck). This file is the resume point. Read it first in any
new session before doing anything else, then check the checklist at the bottom for
what's actually done vs. still pending.

Scaffolding files (`REQUIREMENT.md`, this file, `PLAN.md`, `SLIDES_REQUIREMENT.md`,
every `<app>/PLAN.md`) are **not deliverables** — per the user, they get cleaned up at
the end, same as assignment_03. They exist so the notebook/report/slides can be built
without re-deciding settled questions every session. Their own prose doesn't need to
be clean; the notebooks, report, and slides do.

## Scope

All 3 apps: `diabetes` (MLP), `fashion_mnist` (CNN), `cifar10` (CNN). Each app builds
the **same architecture 3 times** (scratch NumPy, Keras, PyTorch) and compares them,
per `REQUIREMENT.md`. Then a final report (`report/Assignment_04.md`/`.pdf`) and a
slide deck (`slides/`), per `PLAN.md` and `SLIDES_REQUIREMENT.md` respectively.

## Lessons carried over from assignment_03 (read before writing any notebook code)

Extracted from assignment_03's own `PROGRESS.md` (now removed from that assignment's
final commit, but recovered from git history at commit `1a8638f`). These are
concrete mistakes that cost real time/tokens last time — apply the fix up front this
round rather than rediscovering each one:

1. **Notebook independence is a hard rule, checked more than once.** No notebook may
   reference another notebook in this assignment ("the diabetes notebook," "unlike
   in fashion_mnist") or any prior assignment ("assignment_03," "the model
   assignment_03 used"). This slipped through *multiple* times last round, including
   in a markdown patch written *after* the original independence sweep — the check
   has to be re-applied to every new piece of prose, not just the first draft. If a
   choice happens to parallel something from another app or a prior assignment,
   justify it on this app's own terms instead of pointing outside the notebook. A
   plain line-based grep for cross-references can miss a reference split across a
   line wrap — do a newline-collapsed sweep before considering a notebook clean.
   `cifar10/PLAN.md`'s cell 1 already has a reminder about this for its own scratch
   derivation; apply the same discipline to every markdown cell in all 3 notebooks,
   not just that one.
2. **Notebooks are built by a generator script, not hand-typed.** Write a Python
   script (using `nbformat`) that constructs the full `.ipynb` from a list of
   markdown/code cell sources in one shot, then execute it end to end with
   `jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800`
   using this assignment's registered kernel. If a cell errors or needs a content
   fix, fix the generator script and regenerate + re-execute — don't hand-edit the
   executed `.ipynb` directly, since a later regeneration would silently overwrite
   that edit. The one accepted exception: patching a "written comparison" cell's
   prose with real post-execution numbers via a small standalone `nbformat` patch
   script is cheaper than a full re-run, as long as no code cell changes.
   Generator/patch scripts live in the session scratchpad, not committed to the repo.
3. **A single unscaled column can break a from-scratch network even when every
   classical/framework model on the same matrix trains fine.** Root-caused twice
   last round: a raw calendar year (`2019`/`2020`) and a raw city-frequency count
   both dwarfed the surrounding standardized/small-integer features and either
   overflowed the sigmoid or would have degenerated training. This is why
   `diabetes/PLAN.md`'s representation section now says explicitly not to leave
   `year` unscaled — the same check (print min/max of every column in the final
   feature matrix right before training) is worth doing in the two image apps too,
   even though pixel values there are already normalized to `[0,1]` and so less
   likely to hit this.
4. **Class imbalance can make every implementation collapse toward the majority
   class**, producing near-worthless F1 despite a plausible-looking accuracy number.
   `diabetes/PLAN.md`'s cell 18a now computes balanced class weights up front and
   applies them identically across all 3 implementations, rather than discovering
   collapsed metrics after three separate training runs and having to retrofit a
   fix into all three. Check whether `fashion_mnist`/`cifar10` need the same
   treatment once their actual class distributions are confirmed (both are
   documented as balanced benchmark datasets, so this is unlikely to apply, but
   verify rather than assume).
5. **Kill abandoned background scripts before launching a real, expensive execution.**
   A forgotten benchmark script competing for CPU in the background made a real
   notebook execution take far longer than expected last round. On Windows:
   `Get-Process python -ErrorAction SilentlyContinue | Select Id,StartTime,CPU` to
   spot a process that's accumulated unexpectedly high CPU time, then
   `wmic process where "ProcessId=<id>" get CommandLine` to identify what it's
   actually running before killing it with `Stop-Process -Id <id> -Force`.
6. **Every code cell in the model-building/scratch sections needs a markdown cell
   narrating the reasoning, not just the EDA section.** Last round's first draft had
   EDA sections with generous prose and model-fitting/DL sections that were bare
   code back to back. The fix, now baked into `REQUIREMENT.md`'s "Notebook writing
   style" section for this round: every function-definition, model-fit, and metrics
   cell gets a markdown cell explaining what it does and why, in the same
   reasoning-first prose style as the EDA section.
7. **Specific narration cells worth including in every scratch section** (these
   came from repeated "explain this more" rounds last time, so building them in
   from the start saves the back-and-forth): a notation-key cell before `forward`
   (what `X`, `W1`, `Z1`, `A1`, the trailing digit, the `d` prefix each mean), a
   "what He init actually does and why" explanation (not just the formula) in the
   weight-init cell, a step-by-step "why this nonlinearity" explanation in the
   activation cell, an explicit shape-annotated pipeline diagram in the forward
   cell, a meaning-focused (not just mechanical) walkthrough of what each gradient
   represents in the backward cell, and a numbered forward→loss→backward→update
   cycle summary before the training loop.
8. **No em dashes anywhere in notebook or report prose** (code comments too) —
   already in `REQUIREMENT.md`'s writing-style section; worth a literal grep sweep
   (`U+2014`) before considering any notebook finished, since it crept back in
   multiple times last round despite being a standing rule.
9. **Winning-model / real-metric placeholders stay placeholders until a notebook
   has actually executed.** Never guess or estimate a number that belongs in the
   report — pull it from the executed notebook's own output cells.
10. **BRFSS-specific**: sentinel/placeholder codes (e.g. `999` for "don't know") and
    a non-identical column schema across per-year files are expected, not
    hypothetical — `diabetes/PLAN.md` cells 4/4a/6a/14a now plan for this explicitly
    given the 6-year (vs. assignment_03's 2-year) source this round.

## Environment

`assignment_04/.venv`, created from `C:\Users\minorin\scoop\apps\python310\current\python.exe`
(Python 3.10.11) — **not** the global scoop `python` (3.14.7), which lacks working
TensorFlow/PyTorch wheels at that version (and assignment_03's notes flag its pandas
install as broken besides). Always use `assignment_04/.venv/Scripts/python.exe` (or
an activated venv) for anything Python in this assignment.

Installed (confirmed working, verified via direct import):
- Core stack: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `jupyter`,
  `nbconvert`, `nbformat`, `nbclient`, `ipykernel`.
- `tensorflow` 2.21.0, `torch` 2.14.0+cpu, `torchvision` 0.29.0+cpu — CPU wheels
  (torch/torchvision via `--index-url https://download.pytorch.org/whl/cpu`),
  confirmed importable (`torch.cuda.is_available()` correctly reports `False`, as
  expected for a CPU build).
- GPU note: `nvidia-smi` shows an NVIDIA GPU is present on this machine, but CPU-only
  wheels were installed deliberately, for simplicity and so the 3-implementation
  timing comparison isn't confounded by one implementation running on a different
  device class than the others. Revisit only if scratch-CNN or full-dataset training
  time turns out to be a real blocker (see lesson 3/5 above on checking actual
  wall-clock time before assuming a problem).
- Jupyter kernel `assignment_04` registered (display name "assignment_04 (.venv)").
- `kaggle` 1.7.4.5 is installed in this venv too, but **do not use it for
  downloading** — it needs Python >=3.11 for the newer `KAGGLE_API_TOKEN`-only auth
  flow the `KGAT_...` token at `~/.kaggle/access_token` uses; this venv is 3.10.11,
  so its `kaggle` install falls back to the older username+key flow and fails with
  `KeyError: 'username'`. The **global** scoop Python
  (`C:\Users\minorin\scoop\apps\python\current\Scripts\kaggle.exe`) has `kaggle`
  2.2.4, which supports the token correctly — that's what was actually used for the
  download below. Kaggle CLI is a one-off download tool, not part of the trained
  pipeline, so this mismatch doesn't need fixing; just remember to use the global
  `kaggle.exe`, not this venv's, if a re-download is ever needed:
  `export KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token)` then
  `"C:\Users\minorin\scoop\apps\python\current\Scripts\kaggle.exe" datasets download ...`.

## Data status

- `diabetes/data/`: **downloaded and confirmed.** `DATASET_2016.csv` through
  `DATASET_2020.csv` from `spandanjit2005/brfss-diabetes-indicator-dataset` (Kaggle),
  via the global Kaggle CLI (see above). Real row counts confirmed via `wc -l`: 2016
  483,230; 2017 447,952; 2018 434,232; 2019 416,661; 2020 399,050 — combined
  2,181,125 rows (2.67x assignment_03's 815,711), ~557MB total. A 6th year (2015,
  439,881 rows) was also downloaded, checked, and then deleted: 6 years landed at
  2,621,006 rows (3.2x, over the 2-3x target band) at ~700MB, so 5 years was kept
  instead as the better-centered and smaller choice. `REQUIREMENT.md` and
  `diabetes/PLAN.md` both already reflect the final 5-year decision.
- `fashion_mnist/data/`, `cifar10/data/`: no manual download needed — loaded via
  `tf.keras.datasets`/`torchvision.datasets` built-in loaders on first notebook run,
  per `REQUIREMENT.md`.

## Quality review round (user read diabetes.ipynb directly, found real problems)

The user read `diabetes.ipynb` cell by cell and found a list of concrete issues,
several of which are worth remembering as general lessons, not just one-off fixes:

1. **A planning artifact leaked into the notebook** ("`wc -l` on the raw files
   earlier suggested..."). The notebook must never reference how *this plan* was
   built, only what the notebook itself does.
2. **Vague forward references that never land.** "This notebook comes back to
   this later" is only acceptable if the very next cell or two actually delivers;
   otherwise state the fact directly now. Added as a standing rule in
   `REQUIREMENT.md`'s writing-style section.
3. **A markdown cell's own section header swallowed the wrong content.** A
   cell's `## Data cleaning` header was placed *before* the paragraph that should
   have closed out the previous section (explaining the year-vs-diabetes-rate
   chart). The fix was structural: split into two cells, observation first, new
   header second. Lesson: when a markdown cell mixes "wrap up the previous plot"
   with "start the next section," check whether the header landed on the wrong
   side of that mix.
4. **Cross-project/cross-notebook references kept slipping back in**, including
   ones *I introduced myself while fixing other issues* in this same round (e.g.
   fixing the em-dash rule while writing "the same way the diabetes MLP's own
   gradient check does"). Sweeping once is not enough — re-sweep after every
   editing pass, including passes whose purpose was something else entirely.
   Found and fixed 5 additional disguised cross-references this round beyond the
   ones already fixed earlier (none containing the literal word "diabetes" or an
   app name — phrases like "the plain MLP the other tabular-data notebook uses,"
   "the other two notebooks in this project," "used throughout this project," "an
   assumption carried over from a 1-channel implementation elsewhere").
5. **"It's not A, it's B" framing reads as illogical.** State the true claim
   directly ("B, because...") instead of denying something first and explaining
   second. Added as a standing rule in `REQUIREMENT.md`.
6. **Stating a fact without the underlying "why."** Several sections (the
   scale-danger paragraph, the channel-dimension/channel-ordering paragraphs, the
   convolution explanation) stated *what* happens without explaining *why* it
   works that way or *why it matters*. Fixed by restructuring conceptual
   explanations project-wide into: mechanical steps (numbered list where the
   steps are genuinely sequential) → a small worked numeric example → the "why
   this is useful" reasoning in prose → the general formula/code. Applied to:
   ReLU/sigmoid, binary cross-entropy, He init, backward pass, gradient check
   (diabetes); convolution, max pooling, softmax cross-entropy, forward/backward
   composition, gradient check (fashion_mnist, cifar10).
7. **Visualizations were concentrated only in the final comparison section.**
   Added a loss+accuracy dual-curve plot immediately after each of the 3
   implementations finishes training, not just a combined plot at the end.
   Required adding per-epoch train-accuracy tracking to PyTorch's loop in all 3
   notebooks (reusing already-computed batch logits, no extra forward pass).
8. **Combined comparison plot silently hid one line under another** when two
   implementations' values were nearly identical (a solid line drawn on top of
   another solid line at the same values). Fixed with deliberate line-style
   variation (thicker+semi-transparent / dashed / dotted) so all three stay
   visible even when overlapping, not just spaced further apart.
9. **User's stance on the two summary tables**: keep the comparison table
   *and* add charts alongside it (not one instead of the other) — added a
   precision/recall/F1 grouped bar chart to complement the already-present
   accuracy and params/time charts, table stays displayed. The separate
   component-by-implementation table (slide 28's own table) stays as-is; it's a
   named deliverable from the source lecture material, not something added
   unilaterally, so "feels redundant" doesn't mean drop it.

All of the above were applied to `build_diabetes_nb.py`, `build_fashion_mnist_nb.py`,
and `build_cifar10_nb.py` (proactively, since the template was shared — the user only
read diabetes directly, but the same issues existed in the other two and would have
surfaced again on review). All 3 notebooks were regenerated and are being
re-executed as of this checkpoint; **check execution status on resume** before
trusting any metrics currently written into `PLAN.md`/`SLIDES_REQUIREMENT.md` --
none of the added visualizations/tracking change the actual training math, so the
metrics should come out identical to before, but this must be confirmed against
the freshly executed notebooks, not assumed.

## Bug: cifar10.ipynb was executed from a stale (un-regenerated) script version

After fixing the redundant "Same check as the other two implementations" markdown
cell in `build_cifar10_nb.py`, `fashion_mnist.ipynb` was explicitly regenerated before
its next execution, but `cifar10.ipynb` was not — it was executed directly from
whatever `.ipynb` was already on disk, left over from an earlier batch regeneration
that predated the redundant-line fix. Execution succeeded with 0 errors (a stale
script isn't a broken script, just an outdated one), so nothing caught this
automatically. **Caught by re-running the full 3-notebook cross-reference/redundancy
sweep after cifar10 finished**, which flagged the leftover line at cell 43 even
though it had already been removed from the generator script itself.

**Lesson**: after editing a generator script, regenerate *that specific notebook*
before its next execution, every time — don't assume a notebook is current just
because its script was fixed at some point earlier in the session. A stale
`.ipynb` executes cleanly (no errors) and looks identical in every automated check
except the one that actually greps for the specific text that was supposed to be
removed. Re-confirmed by comparing cell counts: the stale file had 61 cells, a
fresh regeneration from the current script produced 60.

Regenerated fresh (60 cells, confirmed missing the redundant line) and re-executing
as of this checkpoint.

## Bug: hardcoded cell index in a patch script corrupted fashion_mnist.ipynb

`patch_fashion_mnist_comparison.py` and `patch_cifar10_comparison.py` were both
written early, when each notebook was much smaller, and hardcoded the written-
comparison cell's index (`nb.cells[50]`, `nb.cells[43]`) instead of searching for it
by content. After the quality-review round added many more cells (per-implementation
plots, restructured explanations), those indices no longer pointed at the placeholder
markdown cell — they pointed at an unrelated **code** cell (the one building
`comparison_table`). Running `patch_fashion_mnist_comparison.py` overwrote that code
cell's `source` with prose text without changing its `cell_type`, which silently
destroyed the `comparison_table` assignment. Every downstream cell that referenced
`comparison_table` (the accuracy chart, the precision/recall/F1 chart, the params/time
chart) would have failed on any future re-run, and the actual placeholder cell (now
several positions later) was left untouched, still showing the placeholder text.

**Caught by the user directly reading the notebook** ("cifar doesn't have the section
of which implementation wins"), not by an automated check — `nbconvert`'s error count
stayed at 0 because the corrupted cell was never re-executed after being corrupted
(the patch runs *after* execution, so a broken code cell just sits there with its old,
now-mismatched output, not a fresh error).

**Fix**: rewrote both patch scripts to find the target cell dynamically (`cell_type ==
"markdown" and "placeholder and gets rewritten" in cell["source"]`) instead of by
index, matching the pattern `patch_diabetes_comparison.py` already used. Also made
`patch_cifar10_cleanup.py` (the download-progress-bar trimmer) search for its target
cell by content (`"cifar10.load_data()" in cell["source"]`) instead of a hardcoded
index, for the same reason. **Lesson for next time**: never hardcode a cell index in a
patch script that runs after the generator script might still change cell counts —
search by content every time, even for a "one-off" patch, since "one-off" patches keep
getting re-run after regeneration.

`fashion_mnist.ipynb` was regenerated fresh (restoring the correct `comparison_table`
cell and the placeholder in its correct position) and is being re-executed as of this
checkpoint. `cifar10.ipynb` has not been executed yet in this round, so it was never
actually corrupted, just at risk from the same stale hardcoded index in its own patch
script, which is now fixed before first use.

## Bugs found and fixed

- **CNN scratch backward pass double-divided by batch size, shrinking every
  dense-layer gradient by an extra factor of `1/N`.** Found by the fashion_mnist
  notebook's own gradient-check cell: `K1`/`K2` (conv layers) matched the numerical
  gradient to ~1e-10 relative error, but `W1` (dense layer) failed at ~50% relative
  error. Root cause: `softmax_cross_entropy_backward` divides `dlogits` by `N` once
  (correct, since the loss is a batch mean), but `linear_backward` *also* divided
  `dW`/`db` by `x.shape[0]` — a second, redundant `/N` that `conv2d_backward` never
  had (it just sums, no division), so conv gradients were correct and dense-layer
  gradients were wrong by exactly `1/N`. Confirmed by reproducing the exact failure
  standalone and testing the fix numerically before touching the generator scripts
  (see the session's diagnostic script in scratchpad if still present). Diabetes's
  MLP was unaffected — it uses a different, internally-consistent convention
  (`dZ3 = ŷ - y` unnormalized, `/m` applied once per layer in each `dW`/`db`, no
  pre-normalization at the loss step), so nothing there needed changing.
  **Fix**: `linear_backward` no longer divides by `m`; `db` uses `np.sum` instead of
  `np.mean`. Applied to both `build_fashion_mnist_nb.py` and `build_cifar10_nb.py`
  (cifar10 had the identical bug, never executed yet at the time it was caught, so
  no wasted run there). Verified via the standalone reproduction that after the fix,
  analytic and numerical gradients match to ~8 significant figures.
- **Notebook-independence violations, caught in my own generated code before
  execution.** `fashion_mnist.ipynb`'s and `cifar10.ipynb`'s builder scripts
  contained multiple sentences referencing each other or the diabetes notebook
  directly ("unlike the diabetes notebook's CSV files," "the same way Fashion-MNIST
  did," "the other CNN notebook," "the other notebook's Keras model," etc. — 8
  instances in fashion_mnist, 8 in cifar10 counting the generic "other
  notebook"/"other image notebook" phrasing). This is the exact mistake flagged as a
  repeated problem in assignment_03's own retrospective, and it recurred here
  despite `REQUIREMENT.md` and both `PLAN.md` files stating the rule explicitly
  before any notebook code was written. **Lesson, worth repeating for next time**:
  writing the rule down in the planning docs is not sufficient on its own; each
  generated notebook needs its own explicit grep sweep (`diabetes`, the other app
  names, `other notebook`, `other .* notebook`) before considering it done, every
  time, not just once. Fixed by rewriting every flagged sentence to justify itself
  on that notebook's own terms before regenerating and re-executing.

## Minor known issue, not yet fixed (low priority)

Diabetes's gradient-check cell (cell 60) checked `W1[0,0]`, `W2[0,0]`, `W3[0,0]` and
got `analytic=0, numerical=0, rel_error=0` for all three — technically passing, but
uninformative, since that specific 5-row sample happened to land in a ReLU dead zone
for those exact units (all-negative pre-activation across all 5 sampled rows, so the
true gradient really is 0 there, not just close to it). The check still ran and still
"passed," and the model went on to train correctly (see the real, sensible converged
metrics above), so this isn't blocking anything, but the check would be more useful
evidence of correctness if it verified an entry with a genuinely nonzero gradient. Not
fixed yet; revisit if there's time before final cleanup, low priority since the
fashion_mnist/cifar10 checks (multiple entries, larger network) did catch a real bug
and are much more informative already.

## Checklist (update after every meaningful step)

- [x] Read `REQUIREMENT.md`, root `PLAN.md`, all 3 app `PLAN.md` files (already
      written this session, before this checkpoint).
- [x] Apply notebook-independence, no-em-dash, and writing-style rules into
      `REQUIREMENT.md` and all per-app `PLAN.md` files (done this session).
- [x] Apply assignment_03 lessons (schema intersection, sentinel values, year-scale
      bug, class imbalance) into `diabetes/PLAN.md` (done this session, see cells
      4/4a/6a/14a/15/17/18/18a and the updated §Representation).
- [x] Write this `PROGRESS.md`.
- [x] Create `assignment_04/.venv` (Python 3.10.11) with core scientific stack.
- [x] Confirm `tensorflow`/`torch`/`torchvision` install finished and import
      correctly (TF 2.21.0, torch 2.14.0+cpu, torchvision 0.29.0+cpu).
- [x] Register Jupyter kernel `assignment_04`.
- [x] Download diabetes data from Kaggle, confirm actual row counts, trim from 6
      years to 5 (2016-2020) to land centered in the 2-3x band at a smaller total
      size (2,181,125 rows, ~557MB) — done this session.
- [x] Build + execute `diabetes/notebook/diabetes.ipynb` — **DONE**. 96 cells,
      0 errors, all 3 models trained and persisted, reload-verify passed. Built via
      `build_diabetes_nb.py` (im2col not needed here, it's an MLP) in the session
      scratchpad. Real results: PyTorch wins by accuracy (0.782), Keras (0.753),
      scratch (0.731); all 3 at exactly 2,177 params (confirms identical
      architecture). Two real findings patched into the notebook's written-
      comparison cell (via `patch_diabetes_comparison.py`) and copied into root
      `PLAN.md`/`SLIDES_REQUIREMENT.md`: (1) PyTorch's `pos_weight` only reweights
      the positive class's loss term, unlike scratch/Keras's symmetric per-class
      weighting, so PyTorch's loss value (~0.87) isn't on the same scale as the
      other two (~0.51) despite comparable accuracy — a real cross-framework
      mechanics gap, not a bug; (2) PyTorch was the slowest implementation (345.7s
      vs. Keras 29.1s, scratch 75.1s) due to per-batch autograd overhead that a
      2,177-parameter network can't amortize away.
- [x] **Pulled diabetes's real findings into root `PLAN.md`'s executive-summary
      table and into `SLIDES_REQUIREMENT.md` slides 10/20/21/23** — done this
      session, right after the notebook finished, per the "don't make the report/
      deck writer reopen the notebook" instruction. Repeat this same step for
      `fashion_mnist` and `cifar10` right after each of those finishes too.
- [x] Build + execute `fashion_mnist/notebook/fashion_mnist.ipynb` — **DONE**. 57
      cells, 0 errors (after the double-`/N` bug fix, see "Bugs found and fixed"),
      all 3 models trained and persisted, reload-verify passed. Real results:
      scratch wins by accuracy (0.897), PyTorch (0.866), Keras (0.860); all 3 at
      exactly 421,642 params. Scratch training time (1,136.1s) is ~15x Keras
      (76.2s) and ~6x PyTorch (192.5s) — the opposite speed ranking from diabetes's
      MLP, where scratch was faster than PyTorch. Written-comparison cell patched
      with this real analysis; findings copied into root `PLAN.md`/
      `SLIDES_REQUIREMENT.md` (slides 11/20/21). Notebook-independence + em-dash
      sweep run twice (once on the generator script, once post-execution — caught
      and fixed one more self-introduced "diabetes notebook" reference in the
      written-comparison patch itself, confirming the lesson that every new piece
      of prose needs its own sweep, not just the first draft).
- [x] Build + execute `cifar10/notebook/cifar10.ipynb` — **DONE**. 50 cells, 0
      errors (same im2col/linear_backward fix applied before first execution, so
      no wasted run). Gradient check passed including the 3-input-channel `K1`
      entries specifically. Real results: Keras wins by accuracy (0.610),
      scratch (0.603), PyTorch (0.570); all 3 at exactly 545,098 params. Scratch
      training time (1,981.2s) is ~20x Keras (98.6s) and ~8x PyTorch (243.9s) —
      the widest gap of all 3 apps, pointing at input size/channel count (not
      parameter count) as the driver. One cosmetic issue found and fixed
      post-execution: loading CIFAR-10 triggered a one-time download whose
      progress bar was captured as ~20,000 tiny output chunks; trimmed via a
      direct nbformat patch (cosmetic only, no re-run needed). Written-comparison
      cell patched with real analysis; findings copied into root `PLAN.md`/
      `SLIDES_REQUIREMENT.md` (slides 12/20/21).
- [x] **Notebook-independence + em-dash sweep across all 3 executed notebooks —
      DONE, all 3 confirmed clean** (`diabetes.ipynb` 96 cells, `fashion_mnist
      .ipynb` 57 cells, `cifar10.ipynb` 50 cells, 0 errors and 0 cross-references
      in all 3, verified with a single script checking all three at once).
- [x] Collect real metrics/winning-implementation per app from executed notebooks
      only (never guessed) — done, in root `PLAN.md`'s executive summary table.

## FINAL STATE as of this checkpoint — all 3 notebooks genuinely done

After the above, the user did a direct quality review of `diabetes.ipynb` and found
several real problems (see "Quality review round" and the two "Bug:" sections above
for the full list and the fixes). That round touched every notebook's generator
script and required 2 extra regenerate+execute cycles for `fashion_mnist.ipynb` (one
to fix a patch-script corruption) and `cifar10.ipynb` (one for the same corruption
risk, one because it had been executed from a stale un-regenerated script). **Final
verified state, confirmed by a single script checking all 3 notebooks together**
(0 errors, 0 cross-references, 0 em dashes, 0 leftover redundant lines, every code
cell executed, `component_table`/`comparison_table` intact, reload-verify passes)
immediately before reporting completion to the user:

| Notebook | Cells | Errors | Scratch / Keras / PyTorch accuracy | Params |
|---|---|---|---|---|
| `diabetes.ipynb` | 104 | 0 | 0.731 / 0.753 / 0.782 | 2,177 |
| `fashion_mnist.ipynb` | 62 | 0 | 0.897 / 0.860 / 0.866 | 421,642 |
| `cifar10.ipynb` | 60 | 0 | 0.603 / 0.610 / 0.570 | 545,098 |

All metrics identical across every regenerate/re-execute cycle (confirms the
quality-review edits never touched training math, only explanations/visualizations,
as intended). Root `PLAN.md` and `SLIDES_REQUIREMENT.md` already reflect these exact
numbers from before the quality-review round and do not need further updates.

**If resuming after this checkpoint**: the notebooks are done. Do not re-run them
again without a specific reason (a new content edit) — re-running costs 5-35 minutes
per notebook for no benefit if nothing changed. Next task is the final report.

- [ ] Write `report/Assignment_04.md`/`.pdf` per root `PLAN.md`.
- [x] ~~Build the slide deck~~ — **out of scope**: user confirmed someone else
      builds the actual deck; my job is keeping `SLIDES_REQUIREMENT.md` accurate
      with real findings (done for all 3 apps as of this checkpoint), not
      producing a `.pptx`/`.pdf` file.
- [ ] Final pass against `REQUIREMENT.md`'s constraints checklist.
- [ ] Delete scaffolding files (`REQUIREMENT.md`, `PLAN.md`, `SLIDES_REQUIREMENT.md`,
      every `<app>/PLAN.md`, this file) once everything above is done and verified —
      matching assignment_03's own final cleanup commit. Do not delete early.
