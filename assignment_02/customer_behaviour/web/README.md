# Order-satisfaction — Web app (React + Vite)

A full-viewport, multi-step client over the FastAPI service. It runs **no model in
the browser** — the prediction is one `POST /predict` call.

```
Browser (React wizard)  ->  REST API  ->  build_features + preprocessing + LogisticRegression
                        <-  JSON       <-  verdict + P(satisfied) + linear-SHAP contributions
```

## The flow

A sticky top bar (title · **Load a real order** picker · light/dark/auto theme · API
status) sits above the **wizard**: a horizontal stepper (`① Product ─ ② Payment ─
③ Delivery & review ─ ④ Review`; done steps get a ✓ and are clickable to jump back),
the step in a panel card, and `← Back` / `Next →` (→ `Predict this order` on step 4)
inline at the foot of that card.

| Step | Left (hardcoded visual) | Right (fields) | `Next` needs |
|---|---|---|---|
| **1 Product** | category emoji tile | category · order value · quantity · seller count · photo count · weight | order value **> 0** |
| **2 Payment** | 💳 + region | payment method · instalments · shipping paid · total charged · customer state | shipping paid |
| **3 Delivery & review** | live SVG delivery timeline (ordered → promised → delivered) | order date · promised date · actual date · review comment (+ PT example chips) | the 3 dates |
| **4 Review & predict** | — | read-back of everything + the "assumed" line + big **Predict** button | — |

Two fields are **not asked for** and filled server-side (`GET /questions` →
`fixed_inputs`): listing description length (607) and payment-methods-used (1).

**Load a real order** picks one of ~40 real Olist orders from `GET /samples`, fills all
steps, and jumps to step 4 so you can review then predict.

## Result screen (full width)

- **Verdict** — "Positive/Negative review likely", the % chance of *that* outcome, the
  main driver in one sentence.
- **Gauge** — `P(satisfied)` with the 50% cut-off tick and a 0→100 scale.
- **"Why this prediction" — a diverging bar chart** of the linear-SHAP contributions:
  each factor's pull in log-odds, red = toward a bad review, green = toward a good one,
  with a waterfall line `63% → −50 pts → 13%` (`base_p` = the model's class-balanced
  neutral point; the dataset positive rate 79% is noted).
- **Signals** — delivery lateness, total delivery time, comment presence, category/region,
  each with a "why it matters" note.
- **Words in the comment** — the TF-IDF terms pulling each way (only when a comment exists).
- **Suggested action** + a "Start over".

## Run

```bash
# API first: from customer_behaviour/  ->  uvicorn api.main:app --port 8000
cd web
npm install
npm run dev            # http://localhost:5174   (proxies /api/* to :8000)
```

`CB_API_URL=http://host:8000 npm run dev` to point elsewhere.
`VITE_API_BASE=https://host npm run build` for a static build (`web/dist/`).

## Layout

```
web/src/
  main.jsx  App.jsx                 shell: load meta/samples/model, hold values + step + result
  api.js                            fetch wrapper
  lib/order.js                      timestamp <-> "days late / delivery days" maths
  lib/labels.js                     category -> emoji
  components/
    Wizard.jsx                      step shell, footer nav, summary step
    Field.jsx                       one /questions field (number/choice/date/text + example chips)
    CategoryTile.jsx  DeliveryTimeline.jsx     the per-step left visuals
    ShapChart.jsx                   diverging-bar contribution chart
    ResultScreen.jsx               full-screen result (verdict + gauge + SHAP + signals)
    ExamplePicker.jsx  ThemeToggle.jsx
  styles.css                        one token-based sheet, light + dark
```

## Screenshots for the report (IDs W1–W5)

1. **W1** — step 1 (Product) with the category tile and fields.
2. **W2** — step 3 (Delivery & review) with the live timeline showing a late delivery.
3. **W3** — step 4 (Review & predict) summary.
4. **W4** — result screen: negative verdict + the SHAP diverging-bar chart + waterfall line.
5. **W5** — `http://localhost:8000/docs` `POST /predict` try-it (evidence the app calls the API).
