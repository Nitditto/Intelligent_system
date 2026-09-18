# Assignment 04 — Presentation Slides Requirement

This file plans the **presentation deck** — a separate deliverable from `report/`,
required alongside it per the verbal brief: *"a presentation slides with all the
definitions, model development process and demonstration."* It outlines what goes on
each slide and how each should look, so the deck can be built directly against this
file once the notebooks (and their real numbers/plots) exist. Like `PLAN.md`, this is
a plan to build against, not a settled, professor-confirmed spec — neither PDF in
`materials/` says anything about a required presentation, so this is reasoned from the
verbal brief plus the lecture's own structure.

Output file: `slides/Assignment_04_Slides.pptx` (or `.pdf` if built from LaTeX Beamer/
Marp instead — pick whichever the report tooling makes easiest; content plan below is
format-agnostic).

## Audience & tone

Written for a classroom presentation to the lecturer (Dinh Que Tran) and classmates —
not a leave-behind document. Slides carry the spine of the argument in short phrases
and visuals; the presenter explains verbally. Follow the source deck's own visual
language where practical (`materials/intel_sys_dev_slide_04_compare_models_CNN.pdf`
uses a plain white background, a dark blue header bar, a light-blue "boxed" callout
for key takeaways, and monospace code blocks on a light-grey panel) — reusing that
language makes the deck feel like a natural continuation of the lecture rather than a
disconnected report-to-slides dump.

## Global design rules

- **One idea per slide.** If a slide needs a scroll or a font size below ~24pt to fit,
  split it.
- **Title + one visual (chart, diagram, code snippet, or table) + a short takeaway
  line**, not paragraphs of prose. Move detailed reasoning into speaker notes, not
  slide body text.
- **Consistent color coding across the whole deck** for the three implementations,
  reused on every chart/table/diagram that compares them:
  - Scratch (NumPy) — orange/tan (matches the source slide deck's "Scratch" boxes)
  - Keras/TensorFlow — light blue
  - PyTorch — light green
  This mirrors the exact color scheme already used in
  `intel_sys_dev_slide_04_compare_models_CNN.pdf` (slide 2's three boxes), so reusing
  it is both less design work and immediately legible to an audience that just saw the
  same lecture.
- **Consistent color coding for the three apps** (diabetes / fashion_mnist / cifar10)
  wherever they appear side by side — pick 3 distinct colors different from the
  implementation palette above so the two coding schemes never clash on the same
  slide.
- **Code snippets**: light-grey panel, monospace font, syntax-relevant coloring, no
  more than ~10-12 lines visible at once — same visual pattern as the source deck's
  own code slides (e.g. slide 9, 13, 14).
- **Every chart follows one design system** (axis labels, legend placement, font)
  across the whole deck — do not let different slides look like they came from
  different tools.
- Page numbers in the footer (`n / total`), consistent with the source lecture deck's
  own footer style.

## Slide-by-slide outline

### Section A — Title & framing (2 slides)

**1. Title slide**
- Title: the assignment's framing question, e.g. "Comparing CNN & MLP
  Implementations: Scratch vs. Keras vs. PyTorch."
- Subtitle: course name, assignment number, date.
- Student name/ID, class, lecturer name — same cover-page fields as the report.
- Visual: reuse the source deck's title-slide layout (dark blue banner, centered
  title) for immediate visual continuity with the lecture.

**2. Agenda / central question**
- Restate slide 2's central question from the lecture material verbatim: *"What is
  the same, and what is different?"* across the three implementations — this frames
  the entire deck's narrative arc.
- A simple 4-item agenda: Definitions → Architectures → Implementation & Training →
  Comparison & Demo.

### Section B — Definitions (4-6 slides)

This section exists because the verbal brief explicitly asks for "all the
definitions" — treat it as a compressed, visual restatement of the tutorial's own
core definitions (`Deep_Learning_CNN_Function_Composition_Tutorial.pdf` §1-17), not a
copy of the report's prose.

**3. Deep learning as function composition**
- The central equation from the tutorial (§7): `F(x) = f_3 ∘ f_2 ∘ f_1(x)`.
- One visual: a horizontal pipeline diagram, boxes labeled `f1 → f2 → f3`, arrows
  between, matching the visual language of slide 8 in the lecture deck
  (`x → f_conv → x1 → f_relu → x2 → ...`).
- Takeaway line: "Deep learning = composing simple functions into a complex one."

**4. The building blocks: neuron, activation, layer**
- Three small diagrams/equations, one row each: `z = wᵀx + b` (neuron), `ReLU(x) =
  max(0,x)` (activation, with the tiny numeric example from tutorial §5.1), `f(x) =
  ReLU(Wx+b)` (layer).
- Keep every equation exactly as the tutorial states it — this slide's job is
  correctness and brevity, not paraphrase.

**5. From vectors to images**
- Contrast `x ∈ ℝ^d` (tabular) vs. `X ∈ ℝ^{C×H×W}` (image), reusing the lecture's own
  diabetes-vs-MNIST framing (slide 3-5).
- Visual: side-by-side icon/diagram — a flat feature list next to a 3-channel image
  grid — making the "why a different architecture" argument visual rather than
  textual.

**6. Convolution, pooling, and why they exist**
- Definitions: convolution (`Y(i,j) = Σ K(u,v)X(i+u,j+v)`, tutorial §9.1), max pooling
  (tutorial §13), local connectivity + weight sharing (tutorial §9.3-9.5) as the two
  reasons convolution beats a fully-connected layer on images.
- Visual: the tutorial's own small numeric convolution example (§9.2, the 3×3 input /
  2×2 kernel producing `Y(1,1) = -4`) worked out on-slide, since it's small enough to
  show fully and is more convincing than the formula alone.

**7. CNN architecture recipe**
- The canonical block: `Conv → ReLU → Pool → Flatten → Dense`, exactly as stated on
  lecture slide 6 and tutorial §16.
- Visual: shape-progression diagram for one concrete case (reuse whichever image app's
  actual shape trace is available once notebooks run — e.g. `3×32×32 → 32×32×32 →
  32×16×16 → 64×16×16 → 64×8×8 → 4096 → 128 → 10` for CIFAR-10), so the definition
  slide already previews real numbers from this project rather than only the generic
  lecture example.

**8. Same architecture, three abstraction levels** *(bridge slide into Section C)*
- The concept↔API correspondence table from lecture slide 7/18 (Convolution /
  Activation / Pooling / Flatten / Dense / Loss / Optimizer × conceptual purpose ×
  typical API name), trimmed to the rows relevant to this project.
- Takeaway line, verbatim from lecture slide 18: *"Different names do not imply
  different machine-learning concepts."*

### Section C — Architectures & model development process (6-9 slides)

**9. The three applications at a glance**
- One slide, 3-column layout (one column per app: diabetes / fashion_mnist /
  cifar10), each showing: a representative data sample (a small tabular row snippet
  or an example image), the task, the architecture family (MLP or CNN).
- This is the deck's map for the rest of Section C and D — every later "per-app" slide
  should visually rhyme with this one's 3-column layout.

**10. Diabetes — architecture**
- The MLP diagram, with the notebook's confirmed real numbers:
  `50 → Linear → ReLU(32) → Linear → ReLU(16) → Linear → Sigmoid → ŷ` (2,177
  parameters total, confirmed matching across all 3 implementations).
- One line stating *why* an MLP and not a CNN here (slide 20's own reasoning: no
  spatial neighborhood between tabular features).

**11. Fashion-MNIST — architecture**
- The CNN diagram with the confirmed real shape trace, reusing lecture slide 21's
  layout style: `(B,1,28,28) → (B,32,28,28) → (B,32,14,14) → (B,64,14,14) →
  (B,64,7,7) → (B,3136) → (B,128) → (B,10)`, 421,642 parameters total (confirmed
  matching across all 3 implementations).

**12. CIFAR-10 — architecture**
- The CNN diagram with the confirmed real shape trace, reusing tutorial §16's
  `SmallCNN` structure and shape progression: `(B,3,32,32) → (B,32,32,32) →
  (B,32,16,16) → (B,64,16,16) → (B,64,8,8) → (B,4096) → (B,128) → (B,10)`,
  545,098 parameters total (confirmed matching across all 3 implementations).

**13. Model development process — the training cycle**
- One universal diagram, reused for all three apps: `Forward → Loss → Gradient →
  Update` (lecture slide 11), with the scratch/Keras/PyTorch code-shape comparison
  from lecture slide 26 condensed into 3 short columns (Manual forward/Manual
  loss/Manual gradient/Manual update/Manual training loop vs. Model definition/
  compile()/fit() vs. nn.Module/loss function/loss.backward()/optimizer.step()).
- This slide is the deck's answer to "model development process" from the verbal
  brief — one clear diagram rather than three redundant ones.

**14. From scratch — what the student actually wrote**
- A trimmed, presentation-sized code snippet: the `conv2d`/`relu`/`max_pool2d`/
  `linear` function signatures (no bodies, or a 3-4 line body at most) plus the
  `cnn_forward`/`mlp_forward` composition function, styled like lecture slide 9-10.
- Takeaway line: "CNN = composition of functions" (lecture slide 10).

**15. Keras — what disappeared**
- Side-by-side: the scratch training loop (collapsed to ~4 lines: forward/loss/
  gradient/update) next to Keras's `model.compile()` + `model.fit()` call.
- Takeaway line, from lecture slide 12: "The convolution and pooling loops have
  disappeared... because they are encapsulated by layers."

**16. PyTorch — explicit but modular**
- The `nn.Module` class skeleton (`__init__`/`forward`) next to the explicit training
  loop (`zero_grad → forward → loss → backward → step`), styled like lecture slide 14
  and 16.
- Takeaway line: PyTorch sits between scratch's full visibility and Keras's full
  convenience.

**17. (Optional, if time allows) Fairness rule**
- One slide, large centered text, exactly lecture slide 29's box: *"Same Dataset +
  Same Split + Same Architecture + Comparable Hyperparameters."*
- One line explaining why this matters: without it, a difference in accuracy could
  come from an unfair setup rather than from the framework itself.

### Section D — Comparison & demonstration (6-8 slides)

**18. Component-by-implementation table — one app as example**
- Reproduce lecture slide 28's exact table structure, filled in for one
  representative app (pick whichever is most illustrative once notebooks are done —
  likely CIFAR-10, since it has every row including Convolution/Pooling).
- Note in speaker notes that the full table for all 3 apps is in the report.

**19. Training curves, live comparison**
- One chart per app (or one combined if it stays legible): loss/accuracy vs. epoch,
  3 lines per chart (one per implementation, using the deck's fixed color coding).
- Takeaway: curves should look near-identical if the architecture and hyperparameters
  really are held fixed — call out any implementation that visibly diverges and why.

**20. Accuracy & parameter-count comparison**
- Grouped bar chart, 3 apps × 3 implementations, accuracy on the y-axis (or two
  panels: accuracy, and parameter count) — the deck's single most "comparison-dense"
  slide, so give it the most space and the clearest legend.
- Diabetes's real numbers, confirmed: accuracy 0.731 (scratch) / 0.753 (Keras) /
  0.782 (PyTorch), all 3 at exactly 2,177 parameters. Fashion-MNIST's real numbers,
  confirmed: accuracy 0.897 (scratch) / 0.860 (Keras) / 0.866 (PyTorch), all 3 at
  exactly 421,642 parameters — note scratch *wins* here, the opposite ranking from
  diabetes, a good talking point for "these are close enough to call the same
  architecture, not a ranking that holds app to app." CIFAR-10's real numbers,
  confirmed: accuracy 0.603 (scratch) / 0.610 (Keras) / 0.570 (PyTorch), all 3 at
  exactly 545,098 parameters — Keras wins narrowly here, absolute accuracy is much
  lower than Fashion-MNIST's (expected: harder task, no augmentation, only 12
  epochs of plain SGD).

**21. Training time comparison**
- Bar chart, 3 apps × 3 implementations, training time (seconds/minutes) on the
  y-axis — a separate slide from accuracy so each claim gets its own visual instead of
  being crowded onto one chart.
- Diabetes's real numbers: Keras 29.1s (fastest), scratch 75.1s, PyTorch 345.7s
  (slowest, >4x scratch despite the network having only 2,177 parameters) — a good
  candidate for the speaker notes to flag PyTorch's per-batch autograd overhead as a
  small-model artifact, not a general "PyTorch is slow" claim.
- Fashion-MNIST's real numbers tell the opposite story: Keras 76.2s (fastest),
  PyTorch 192.5s, scratch 1,136.1s (slowest by far, ~15x Keras, ~6x PyTorch). This
  is the deck's strongest example of the abstraction-level trade-off actually
  costing something real: scratch's `im2col`-vectorized convolution is still plain
  NumPy from a Python loop, with none of a framework's operator-level optimization
  for convolution specifically, and it shows up directly in wall-clock time once
  the model has actual convolutions (unlike diabetes's MLP, where scratch was
  comparably fast to PyTorch). CIFAR-10's real numbers widen the gap further:
  Keras 98.6s (fastest), PyTorch 243.9s, scratch 1,981.2s (slowest, ~20x Keras,
  ~8x PyTorch) — a clean 3-point progression across the deck's three apps (MLP:
  scratch comparable → grayscale CNN: scratch ~15x slower → RGB CNN: scratch
  ~20x slower) worth building as its own "the gap widens with the actual
  convolution workload" narrative arc across slides 20-21.

**22. Confusion matrices**
- 3-panel figure (one per implementation) for one app, or a compact multi-app grid if
  space allows — pick the app with the most interesting class-confusion pattern once
  results exist (classification apps only, i.e. any of the 3 since none is regression
  this round).

**23. Which framework hid the most? Which gave the most control?**
- Direct answer to lecture slide 28's own question, stated as this project's actual
  finding (not restated from the lecture) — a short ranked list or 3-column
  scratch/Keras/PyTorch comparison of "lines of code," "explicit steps visible," and
  "ease of modification," backed by what was observed building all 9 models.
- Concrete example already confirmed from the diabetes app, worth using here
  directly: all 3 implementations offer "balanced class weighting" under a
  same-sounding name, but Keras's `class_weight` and the scratch implementation's
  manual `weights` argument both reweight *every* row (both classes), while
  PyTorch's `BCEWithLogitsLoss(pos_weight=...)` only reweights the positive class's
  term — same feature, same intent, genuinely different mechanics underneath. This
  is a real, first-hand instance of lecture slide 18's "different names do not
  imply different machine-learning concepts" cutting the other way too: sometimes
  the same name *does* hide a real mechanical difference, and the only way to know
  is to check.

**24. Live demonstration slide**
- Not a static content slide — a placeholder/cue slide for switching to a live
  notebook or a short recorded clip: load one trained model per implementation
  (for one app, likely whichever trained fastest to demo live, e.g. Fashion-MNIST),
  run a prediction on a held-out sample, show the predicted class + confidence for
  all 3 implementations side by side.
- Include a fallback: a screenshot/GIF of the same demonstration embedded on the
  slide, in case live execution isn't practical during the presentation.

**25. Conclusion**
- 3-4 bullet takeaways, one per major claim the deck made: (1) same architecture,
  different abstraction level — confirmed by near-identical accuracy across
  implementations; (2) scratch gives full visibility at the cost of manual
  implementation of every gradient; (3) Keras trades visibility for development
  speed; (4) PyTorch sits in between.

**26. Questions slide**
- Minimal: "Questions?" + contact/repo link, matching a typical closing-slide
  convention.

## What this file deliberately does not cover

- The report's full written analysis, per-app EDA narrative, and reproducibility
  instructions — those live in `PLAN.md` and the eventual `report/`. The slides
  summarize; they do not duplicate the report's prose.
- Exact chart data and screenshots — those don't exist until the notebooks
  (`diabetes/notebook/`, `fashion_mnist/notebook/`, `cifar10/notebook/`) are written
  and executed. This file fixes *what* each slide shows and *why*, not the final
  numbers.
- Speaker-note scripts — left to whoever presents, informed by the per-slide
  takeaway lines above.
