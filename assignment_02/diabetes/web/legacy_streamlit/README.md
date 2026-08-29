# Diabetes Screening — Web app (Streamlit)

A thin client over the FastAPI service. It does **not** load the model or run
inference itself — every prediction is a call to `POST /predict`.

## Run

```bash
# 1. start the API (separate terminal, from diabetes/)
uvicorn api.main:app --port 8000

# 2. start the web app (from diabetes/)
streamlit run web/app.py
```

Open http://localhost:8501.

Point at a non-default API with an environment variable:

```bash
DIABETES_API_URL=http://192.168.1.20:8000 streamlit run web/app.py
```

## Pages

| Page | What it shows |
|---|---|
| **1 · Screening** | the questionnaire, rendered from `GET /questions`; every item has a *Not sure* option; height + weight compute BMI. Submit → `POST /predict?include=explain,similar,whatif,counterfactual`. |
| **2 · Result** | risk % + Low/Moderate/High band, completeness + uncertainty banner, SHAP factor bars ("why this score"), the 5 most similar survey respondents and how many had diabetes, what-if bars + a BMI sweep chart, and the nearest counterfactual sentence with the not-medical-advice caveat. |
| **3 · History** | past screenings for the current `session_id` (`GET /history`). |
| **4 · Dashboard** | operator view: two threshold sliders with a live precision / recall / flag-rate / per-1000 trade-off from `GET /threshold-curve`, a recall-vs-flag-rate curve, a recommendation helper (`GET /threshold-recommend`), **Save** via `POST /config/thresholds` (needs the admin key), and the live monitoring metrics (`GET /metrics`). |

## Screenshots for the report

Take these against a running system (see `../Report_Deliverable.md` for captions):

1. Page 1 with a filled questionnaire.
2. Page 2 top — risk band + SHAP factor bars.
3. Page 2 lower — similar cases + what-if + counterfactual.
4. Page 4 — threshold dashboard with the trade-off panel.
