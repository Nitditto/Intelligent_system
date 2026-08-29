# Build progress — Diabetes application

Tracking file so the build can resume after a context reset. Update the checkboxes and
the "Resume here" pointer as work proceeds. Spec: `APP_DESIGN.md`. Assignment layout:
`materials/intel_sys_dev_assignment_02_final.pdf` Appendix A.

Environment: **Anaconda base**, run everything with `F:/anaconda/python.exe`.
Installed for this project: `fastapi 0.141`, `uvicorn 0.40`, `shap 0.52`,
`streamlit 1.58`, `sklearn 1.9`, `pandas 3.0`, `pydantic 2.13`.

---

## Order of work (one component at a time)

### 1. FastAPI backend — `diabetes/api/`   ✅ DONE (2026-08-29)
- [x] `requirements.txt`
- [x] `api/config.py`  `api/schema.py`  `api/inference.py`  `api/validate.py`
- [x] `api/explain.py`  `api/similarity.py`  `api/whatif.py`  `api/monitoring.py`  `api/dashboard.py`
- [x] `api/main.py`  `api/__init__.py`  `api/build_artifacts.py`  `api/README.md`
- [x] `.gitignore` (ignores api/artifacts, api/runtime, model/holdout_scores.joblib)
- [x] ran `python api/build_artifacts.py` → holdout_scores.joblib, neighbor_index.joblib,
      shap_background.joblib, bmi_by_age_sex.joblib, runtime/thresholds.json (seed {0.45, 0.6})
- [x] smoke-tested via `fastapi.testclient`: /predict (+explain/similar/whatif/counterfactual),
      sparse input, BMI clamp, 422 on bad input, /threshold-curve, /threshold-recommend,
      /config + /config/thresholds (admin key + 401), /metrics, /history, /questions — all pass.
      Sample: full profile → prob 0.8496, band High (matches notebook inference test).

**To run:** from `diabetes/`, `F:/anaconda/python.exe -m uvicorn api.main:app --port 8000`
(docs at http://localhost:8000/docs). If `api/artifacts/` is missing, first run
`F:/anaconda/python.exe api/build_artifacts.py`.

### 2. Web client — `diabetes/web/`   ✅ REBUILT in React + Vite (2026-08-29)
Was Streamlit; rebuilt after UX feedback (looked like a checklist, not a web app;
awkward "not sure" checkboxes; SHAP as a static bar chart; over-stuffed dashboard;
irrelevant model-internals text). Old Streamlit files kept in `web/legacy_streamlit/`.
- [x] `web/package.json` `vite.config.js` `index.html` `.env.example` `.gitignore`
- [x] `src/api.js` — fetch wrapper (`/api` proxy in dev, `VITE_API_BASE` in prod)
- [x] `src/App.jsx` — single page, two views (Screening / Operator), localStorage session
- [x] `src/components/` — Questionnaire (yes/no rest at "No", no checkboxes),
      ResultPanel, RiskHeadline, **ForcePlot** (hand-built interactive SVG SHAP force
      plot with hover tooltips), SimilarCases, WhatYouCanChange (per-habit deltas +
      live BMI slider + de-duped counterfactual sentence), HistoryPanel,
      Dashboard (one referral-cut-off slider + "per 1,000" readout + trade-off curve
      + help-me-choose + guarded Save + collapsible recent activity), TradeoffCurve,
      Sparkline, Collapsible, Header
- [x] `web/README.md` rewritten
- [x] `npm install` (110 pkgs) + `npm run build` clean (45 modules); `npm run dev`
      serves, `/api` proxy reaches the API, JSX transforms 200
- [x] all API response shapes re-verified live: /predict (+explain 23 factors, base
      0.5 / pred 0.83, similar, whatif, counterfactual), /config, /threshold-curve
      (49 pts, per_1000), /threshold-recommend, /metrics, /history

### 3. Flutter mobile — `diabetes/mobile/`   ✅ DONE (2026-08-29, source only)
- [x] `mobile/pubspec.yaml` (deps: http, cupertino_icons; flutter_lints)
- [x] `mobile/lib/main.dart` (ApiClient instance, sessionId, bandColor helper)
- [x] `mobile/lib/api_client.dart` (base url via --dart-define=API_URL, default 10.0.2.2:8000)
- [x] `mobile/lib/models.dart` (Question, PredictResult, ShapFactor, SimilarCase, WhatIfFactor, HistoryItem)
- [x] `mobile/lib/screens/`: questionnaire_screen · result_screen · whatif_screen · history_screen
- [x] `mobile/analysis_options.yaml`, `mobile/.gitignore`, `mobile/README.md`
- [x] brace-balance sanity check passes on all .dart files
- [ ] **USER MUST DO:** `cd mobile && flutter create . && flutter pub get && flutter run`
      (no Flutter SDK on this machine — code is compile-ready, not compiled/run here).
      Android cleartext http: add `android:usesCleartextTraffic="true"` (see mobile/README.md).

### 4. Report outline — `diabetes/Report_Deliverable.md`   ✅ DONE (2026-08-29)
- [x] section-by-section outline for Part A §4.1–§4.10 + Reproducibility + Appendix D/E
- [x] 📸 placeholders with captions; consolidated screenshot checklist (N1–N8, W1–W5, M1–M5, X1)
- [x] key executed numbers quoted inline so the report can cite them directly

---

## ✅ ALL FOUR COMPONENTS BUILT.

Remaining = user manual steps (no code left to write):
1. `cd mobile && flutter create . && flutter pub get && flutter run` (no Flutter SDK here)
2. take the screenshots listed in `Report_Deliverable.md` → "Screenshot checklist"
3. write the report from `Report_Deliverable.md`
4. (optional) add a cell to notebook §19 saving `model/holdout_scores.joblib`, so it
   comes from the notebook instead of `api/build_artifacts.py`

### How to run the full system
```
cd diabetes
F:/anaconda/python.exe api/build_artifacts.py         # once, if api/artifacts/ missing
F:/anaconda/python.exe -m uvicorn api.main:app --port 8000
npm --prefix web install && npm --prefix web run dev  # separate terminal -> :5173
```

## Notes / decisions log

- Model artifacts already exist from the notebook: `model/model_pipeline.joblib`,
  `model/feature_names.joblib`, `model/input_schema.json`. Chosen model = RandomForest,
  seed 42, sklearn 1.9.0.
- `model/holdout_scores.joblib` does NOT exist yet — `build_artifacts.py` regenerates it
  by re-fitting a train-only RF with the notebook's specs (n_estimators=300, max_depth=12,
  min_samples_leaf=20, class_weight="balanced_subsample") and scoring val+test.
- 23 model features, order in `feature_names.joblib`:
  14 binary, then GenHlth/Age/Education/Income, then BMI/MentHlth/PhysHlth, then
  TotalUnhealthyDays, CardioRisk.
- Modifiable factors for what-if: BMI, Smoker(→0), HvyAlcoholConsump(→0),
  PhysActivity(→1), Fruits(→1), Veggies(→1).
- `shap` and `fastapi` were pip-installed into anaconda base this session.
