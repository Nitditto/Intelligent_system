# Product-recommendation — Web app (React + Vite)

A full-viewport, 3-step client over the FastAPI service. It runs **no model in the
browser** — the prediction is one `POST /predict` call.

```
Browser (React wizard)  ->  REST API  ->  build_features + preprocessing + LogisticRegression
                        <-  JSON       <-  verdict + P(recommend) + linear-SHAP contributions
```

Target: **`is_recommended`** — did the reviewer tick "recommends this product". Dataset:
~104k Sephora skincare reviews (product-id shard 500–750).

## The flow

A sticky top bar (title · **Load a real review** picker · light/dark/auto theme · API
status) sits above the **wizard**: a horizontal stepper (`① Your skin profile ─
② The product ─ ③ The review`; done steps get a ✓ and are clickable to jump back),
the step's fields in a grid card, `← Back` / `Next →` (→ **Predict** on step 3) at the
foot.

| Step | Fields | `Next` needs |
|---|---|---|
| **1 Your skin profile** | skin type · skin tone · eye colour · hair colour | — |
| **2 The product** | category · brand · price · “loves” · total reviews | price **> 0** |
| **3 The review** | review title · review text (+ real-review example chips) | review text |

Fields the form does not ask for are filled server-side (`GET /questions` →
`fixed_inputs`): the edition flags, the community feedback-vote counts (a new review has
none), and the ingredient / highlight counts (imputed to the dataset median).

**Load a real review** picks one of ~42 real Sephora reviews from `GET /samples`, fills
the form, and jumps to step 3.

## Result screen (full width)

- **Verdict** — "This customer would / would not recommend the product", plus the %
  chance of a recommendation.
- **Meter** — `P(recommend)` with the 50% cut-off tick and a 0→100 scale, and a
  plain-language "read it as…" line.
- **What the model saw** — skin type, category, brand, price tier, product “loves”,
  review length — with a note that the structured block alone reaches ROC-AUC ~0.8 and
  the review text takes it to ~0.96.
- **Words in the review that moved the call** — the TF-IDF terms pulling each way.
- **"Why this prediction"** — a diverging bar chart of the linear-SHAP contributions:
  each factor's pull in log-odds, red = toward "won't recommend", green = toward
  "recommends", with a waterfall line.
- **How to use this** + a "How this works" `<details>` (including the §14a note that the
  review text is co-authored with the recommend tick).

## Run

```bash
# API first: from customer_behaviour/  ->  python -m uvicorn api.main:app --port 8000
cd web
npm install
npm run dev            # http://localhost:5174   (proxies /api/* to :8000)
```

`CB_API_URL=http://host:8000 npm run dev` to point elsewhere.
`VITE_API_BASE=https://host npm run build` for a static build (`web/dist/`).

## Layout

```
web/src/
  main.jsx  App.jsx                 shell: load meta/samples/model, hold values + result
  api.js                            fetch wrapper
  lib/sephora.js                    presentational label maps
  components/
    Wizard.jsx                      3-step shell + stepper + footer nav
    Field.jsx                       one /questions field (choice/number/text/textarea)
    ShapChart.jsx                   diverging-bar contribution chart
    ResultScreen.jsx                full-screen result (verdict + meter + signals + SHAP)
    ExamplePicker.jsx  ThemeToggle.jsx
  styles.css                        one token-based sheet, light + dark
```

## Screenshots for the report (IDs W1–W5)

1. **W1** — step 1 (skin profile).
2. **W2** — step 3 (the review) with an example loaded.
3. **W3** — result screen: "won't recommend" verdict + the meter.
4. **W4** — result screen: the SHAP diverging-bar chart + waterfall line + review-term chips.
5. **W5** — `http://localhost:8000/docs` `POST /predict` try-it (evidence the app calls the API).
