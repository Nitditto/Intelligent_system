---
name: frontend-ui
description: >-
  UI/design conventions for the web clients in this repo (assignment_02/*/web —
  React 18 + Vite + plain CSS, no framework). Load before creating or restyling any
  web client, adding a component, or when a UI "looks ugly / inconsistent". Covers
  design tokens, spacing/type scale, component patterns (card, button, field, pill,
  meter), dark mode, states (hover/focus/disabled/loading/error/empty) and a11y.
---

# Frontend UI conventions

The web clients (`assignment_02/diabetes/web`, `assignment_02/customer_behaviour/web`,
future `house_price/web`) are **React 18 + Vite + hand-written CSS** — no Tailwind, no
component library. These rules keep them looking like one product.

## 0. Golden rules

1. **Tokens, not literals.** Every colour, space, radius, shadow, font-size comes from a
   CSS custom property defined once in `:root` (see §1). No raw hex or px in component
   CSS except `1px` borders and `0`.
2. **One stylesheet of truth.** All styling lives in `src/styles.css` (or
   `src/styles/*.css` imported once). No CSS-in-JS. Inline `style={{}}` is allowed
   **only** for a value computed at runtime (a bar width, a transform).
3. **Mobile-first.** Base rules target ~360px; widen with `@media (min-width: …)`.
   The page body never scrolls horizontally.
4. **Every interactive state is designed:** default, `:hover`, `:focus-visible`,
   `:active`, `:disabled`, loading, error, and the empty/first-load state.
5. **Accessible by default:** real `<label>` per field, visible focus ring, text
   contrast ≥ 4.5:1, respects `prefers-reduced-motion`, hit targets ≥ 40px.
6. **Semantic class names**, lowercase, hyphenated, component-scoped
   (`.result-card`, `.result-card__meter`). No utility soup, no deep selectors
   (max 2 levels).

## 1. Design tokens (put at the top of `styles.css`)

```css
:root {
  color-scheme: light dark;

  /* neutrals — cool grey ramp */
  --c-bg:        #f7f8fa;
  --c-surface:   #ffffff;
  --c-surface-2: #f1f3f7;
  --c-border:    #e3e7ee;
  --c-text:      #1a2233;
  --c-text-soft: #5b6577;
  --c-text-faint:#8a93a6;

  /* brand + semantic */
  --c-accent:      #3b6ef2;
  --c-accent-ink:  #ffffff;
  --c-accent-weak: #eaf0fe;
  --c-good:        #128a5b;
  --c-good-weak:   #e2f4ec;
  --c-warn:        #b7791f;
  --c-bad:         #d1453b;
  --c-bad-weak:    #fceceb;

  /* type — a modular-ish scale */
  --f-sans: "Inter", ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  --f-mono: ui-monospace, "SF Mono", "Cascadia Code", "JetBrains Mono", monospace;
  --t-xs: 12px; --t-sm: 13px; --t-md: 15px; --t-lg: 18px; --t-xl: 22px; --t-2xl: 28px;
  --lh: 1.55;

  /* space — 4px base scale: 1=4 2=8 3=12 4=16 5=20 6=24 8=32 10=40 12=48 */
  --s-1: 4px; --s-2: 8px; --s-3: 12px; --s-4: 16px; --s-5: 20px;
  --s-6: 24px; --s-8: 32px; --s-10: 40px; --s-12: 48px;

  /* shape + depth */
  --r-sm: 6px; --r-md: 10px; --r-lg: 14px; --r-pill: 999px;
  --sh-1: 0 1px 2px rgba(16,24,40,.06), 0 1px 3px rgba(16,24,40,.10);
  --sh-2: 0 4px 12px rgba(16,24,40,.08), 0 2px 4px rgba(16,24,40,.06);

  --ring: 0 0 0 3px color-mix(in srgb, var(--c-accent) 35%, transparent);
  --container: 1080px;
}

@media (prefers-color-scheme: dark) {
  :root {
    --c-bg:#0f1420; --c-surface:#161c2b; --c-surface-2:#1e2536; --c-border:#2a3346;
    --c-text:#e8ecf4; --c-text-soft:#a9b2c5; --c-text-faint:#7b8499;
    --c-accent:#5b8bff; --c-accent-weak:#1b2740;
    --c-good:#3ecf8e; --c-good-weak:#14301f; --c-bad:#f2635a; --c-bad-weak:#331b1a;
    --sh-1: 0 1px 2px rgba(0,0,0,.4); --sh-2: 0 6px 20px rgba(0,0,0,.45);
  }
}
```

Load Inter in `index.html` (`<head>`), with a real fallback already in `--f-sans`:

```html
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet"
  href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap">
```

## 2. Base / reset

```css
* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body {
  margin: 0; background: var(--c-bg); color: var(--c-text);
  font: 400 var(--t-md)/var(--lh) var(--f-sans);
  -webkit-font-smoothing: antialiased;
}
h1,h2,h3 { margin: 0 0 var(--s-3); line-height: 1.25; font-weight: 600; }
h1 { font-size: var(--t-2xl); letter-spacing: -0.02em; }
h2 { font-size: var(--t-lg); }
p  { margin: 0 0 var(--s-3); }
a  { color: var(--c-accent); text-underline-offset: 2px; }
code, kbd { font-family: var(--f-mono); font-size: .92em;
  background: var(--c-surface-2); padding: 1px 5px; border-radius: var(--r-sm); }
:focus-visible { outline: none; box-shadow: var(--ring); border-radius: var(--r-sm); }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
```

## 3. Layout

- Page wrapper: `max-width: var(--container); margin-inline: auto; padding: var(--s-6) var(--s-5) var(--s-12);`
- Two-column work area: CSS grid, `grid-template-columns: 1.1fr 1fr; gap: var(--s-5);`
  collapsing to one column at `max-width: 900px`.
- A sticky right column on wide screens is fine (`position: sticky; top: var(--s-6);`).
- Section rhythm: stack blocks with `gap` on the flex/grid parent, **not** margins on
  children.

## 4. Components

### Card
```css
.card {
  background: var(--c-surface); border: 1px solid var(--c-border);
  border-radius: var(--r-lg); box-shadow: var(--sh-1);
  padding: var(--s-6);
}
.card > h2 { font-size: var(--t-sm); text-transform: uppercase; letter-spacing: .06em;
  color: var(--c-text-faint); margin-bottom: var(--s-4); }
```

### Button
- One primary per view (`.btn--primary`), the rest `.btn--ghost`.
- Height 40px, `border-radius: var(--r-md)`, `font-weight: 600`, `font-size: var(--t-sm)`.
- `:disabled` → `opacity:.55; cursor:not-allowed`. Loading = spinner + label, stays
  the same width (no layout shift).
```css
.btn { display:inline-flex; align-items:center; gap:var(--s-2); justify-content:center;
  height:40px; padding:0 var(--s-4); border:1px solid transparent; border-radius:var(--r-md);
  font:600 var(--t-sm)/1 var(--f-sans); cursor:pointer; transition:background .12s, box-shadow .12s; }
.btn--primary { background:var(--c-accent); color:var(--c-accent-ink); }
.btn--primary:hover { background: color-mix(in srgb, var(--c-accent) 88%, #000); }
.btn--ghost { background:var(--c-surface); border-color:var(--c-border); color:var(--c-text); }
.btn--ghost:hover { background:var(--c-surface-2); }
.btn:disabled { opacity:.55; cursor:not-allowed; }
```

### Field (input / select / textarea)
- Always `<label class="field"><span class="field__label">…</span><input…></label>`.
- Control height 40px (textarea min 84px), `border:1px solid var(--c-border)`,
  `border-radius: var(--r-md)`, `background: var(--c-surface)`.
- `:focus` → `border-color: var(--c-accent); box-shadow: var(--ring);`.
- Helper text `.field__note` in `--t-xs` / `--c-text-faint`; error text in `--c-bad`.
- Required marker: a `*` in `--c-bad` after the label.
- Group related fields in a `.field-group` with a small heading; two-up on wide,
  one-up on narrow (`grid-template-columns: repeat(auto-fit, minmax(200px, 1fr))`).

### Pill / badge
```css
.pill { display:inline-flex; align-items:center; gap:var(--s-1);
  padding:3px var(--s-2); border-radius:var(--r-pill); font-size:var(--t-xs); font-weight:600;
  background:var(--c-surface-2); color:var(--c-text-soft); }
.pill--good { background:var(--c-good-weak); color:var(--c-good); }
.pill--bad  { background:var(--c-bad-weak);  color:var(--c-bad); }
```

### Meter / gauge (probability, score, confidence)
- Track: `height: 10px; border-radius: var(--r-pill); background: var(--c-surface-2);`
- Fill: width set inline from the value; colour semantic (good/warn/bad) or a gradient.
- Mark the decision threshold with a 2px vertical rule and label both ends in `--t-xs`.
- Always show the numeric value next to it — the bar is decoration, the number is truth.

### Verdict block (the headline result)
- Big, tinted panel: `background: var(--c-good-weak | --c-bad-weak)`, `border-radius: var(--r-md)`,
  `padding: var(--s-4) var(--s-5)`. Title in `--t-lg`/700 in the semantic colour,
  confidence in `--t-sm`/`--c-text-soft` on the right.

## 5. States you must implement

| State | Rule |
|---|---|
| **Empty / first load** | A short helper sentence in `--c-text-soft`, never a blank pane. |
| **Loading** | Inline spinner + text; disable the submit; don't blank existing content. |
| **Error** | Red inline message near the cause (or a top banner for conn(ection); actionable text ("Start the API on :8000"). Never a bare stack trace. |
| **Offline API** | A dismissible banner at the top; keep the form usable. |
| **Disabled** | Reduced opacity + `not-allowed`; explain why if not obvious. |

## 6. Data display

- Numbers: format with a helper (`toFixed`, thousands separators, `%`); align
  right in tables. Show units.
- Chips for lists of short tokens (keywords, tags, factors), not comma strings.
- Key→value rows: a two-column grid, label in `--c-text-soft`, value in `--c-text`
  600. Don't use `<table>` for < 3 columns of pairs.
- Colour never carries meaning alone — pair with an icon, a word, or position.

### 6a. Make predictions and metrics legible (non-negotiable for these apps)

A screen that shows a model's output must be understandable by someone who has never
seen the dataset. Assume no domain knowledge.

- **State what is predicted, in one plain sentence, near the top** — not the raw target
  name. "estimates whether the customer will leave a positive (4–5★) or negative (1–3★)
  review", not "predicts `satisfied`".
- **Lead with a plain verdict**, then the number. "This customer will probably be
  unhappy" before "P = 0.07".
- **Every displayed number gets: a plain label + (if it's a probability / score /
  threshold) a one-line "how to read it".** "Chance of a positive review: 7% — we flag
  anything below 50%." Never show a bare `p_satisfied`, `confidence`, `threshold`,
  `ROC-AUC` with no gloss.
- **Don't show two competing percentages** (e.g. `confidence` and `p_satisfied`) without
  saying how they differ — usually just pick one.
- **Explain each feature/signal you surface**: label it in plain words and add a short
  "why this matters" note ("Arriving after the promised date is the biggest cause of a
  bad review").
- **Give the user the "so what"** — a suggested action or interpretation, not just a
  score.
- **Keep model provenance, but humanise it.** A small footer is fine —
  "LogisticRegression on delivery + comment features, flags orders below 50%" — not
  "`clf=LogisticRegression threshold=0.5 seed=42`" as the whole story.
- Offer a collapsible **"How it works"** (`<details>`): what it was trained on, what
  drives the prediction, that inference is server-side.

### 6b. Contribution / SHAP chart (linear models)

For a linear model, the exact per-feature pull is `phi_j = coef_j·(x_j − mean_j)` — no
`shap` library. Persist the transformed-feature means + `coef`/`intercept` from the
notebook; the API returns `{base_p, final_p, items:[{label, kind, effect}], other_effect}`.

- **Diverging bar chart**: a centre axis; bar length ∝ `|effect|` (log-odds);
  left/`--c-bad` = pushes toward the bad outcome, right/`--c-good` = toward the good one.
- Sort rows by `|effect|` desc; cap at ~10–12; fold the rest into an "other smaller
  factors" row so the decomposition stays exact.
- **Contextual labels from the raw value**, not the column name: `delivery_delay_days>0`
  → "Delivered 12 days late"; one-hot → "Category: furniture_decor"; a text token →
  `comment: "atrasado"`. Scaled numerics → "Order value: above/below average".
- **Waterfall line** underneath: `base 63% → −50 pts → 13%`. State plainly what `base`
  is (a class-balanced model's neutral point is *not* the dataset base rate — say so).
- Suppress phantom rows: an empty text field still yields tiny `coef·(0−mean)` pulls —
  fold those into "other", don't render `comment: "…"` for a comment that isn't there.

### 6c. Multi-step wizard (full-viewport)

When a form has >8 fields or the user asked for "next-next": a wizard, not one long page.

- **Shell**: `.app{min-height:100%;display:flex;flex-direction:column}` with a **sticky
  top bar** (`position:sticky;top:0`); the middle `.stage` is `flex:1;overflow-y:auto`.
  `html,body,#root{height:100%}`.
- **Progress = a horizontal stepper at the top of the content**, not a bottom bar
  (`① Label ── ② Label ── …`): connector line between items, current gets an accent
  ring, done steps get a ✓ and are **clickable to jump back**, future steps are dim and
  disabled. On ≤620px hide the labels (keep numbered circles) and show a
  "Step N of M · Label" caption. A sticky bottom bar on a short page floats in
  whitespace — avoid it.
- **Back / Next live inline at the foot of the step panel** (a `.step__nav` with a
  top border, `justify-content:space-between`): `← Back` (disabled on step 0) and
  `Next →` that is **disabled until that step's required fields are valid** (`title=`
  says why). On the last step the primary button becomes the real action ("Predict"),
  not "Next".
- **One state object** for all fields across all steps (`values`); steps read/write it.
  Lift the step index to the parent so shortcuts (e.g. "load an example") can jump.
- Each step = a small left **visual** (an icon tile, a live mini-SVG) + a right field
  grid (`repeat(auto-fit, minmax(210px, 1fr))`). The visual reacts to the inputs.
- End with a **read-back summary step** (label → value grid per section) + an
  "assumed (not asked): …" line for any hardcoded/hidden fields, then the action button.
- Collapses to one column at ~760px; the sticky footer must still fit at 360px
  (shrink button padding, hide secondary labels).

## 7. Do / Don't

- ✅ `gap` for spacing between siblings · ❌ stacked `margin-bottom`
- ✅ `color-mix()` / token references for hover shades · ❌ new hex values
- ✅ `rem`/`px` from the scale · ❌ arbitrary `13.5px`, `#4a72e1`
- ✅ 2–3 font sizes per screen · ❌ five
- ✅ one accent colour · ❌ three competing brand colours
- ✅ `prefers-color-scheme` dark variant · ❌ light-only hard-coded `#fff`
- ✅ components in `src/components/*.jsx`, one concern each · ❌ 300-line `App.jsx`

## 8. Checklist before calling a UI "done"

- [ ] No raw hex / off-scale px in component CSS (grep for `#` and stray `px`)
- [ ] Tab through every control — focus ring visible, order logical
- [ ] 360px wide: no horizontal scroll, nothing clipped
- [ ] Dark mode: readable, no white flashes
- [ ] Loading + error + empty states all render sensibly
- [ ] Primary action obvious; one per screen
- [ ] Numbers formatted, units shown, colours have a non-colour cue
