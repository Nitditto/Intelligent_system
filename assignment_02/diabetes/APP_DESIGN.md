# Diabetes Screening Application — Structure & Workflow

Design notes for the deployable app that wraps `model/model_pipeline.joblib`
(random forest + preprocessing) produced by `notebook/diabetes.ipynb`.

The guiding idea: **all intelligence lives in the API**. The web and mobile apps are
thin clients that render what the API returns, so neither has to be complex and the
"interesting" features are shared by both.

---

## 1. What the dataset implies for the app

The data is a **health questionnaire** (CDC BRFSS 2015). Three consequences drive the design:

| Dataset fact | App consequence |
|---|---|
| Every feature is a survey answer; respondents drop "don't know / refused" | The app must *be* a questionnaire, and must accept **"Not sure / prefer not to say"** for any question. Unanswered → `null` → median-imputed server-side (the imputer is already in the pipeline). |
| Nobody knows their own BMI | Ask **height + weight**, compute `BMI` on the client/server. Never show a raw "BMI" field. |
| Deployed model is a **random forest**, no other model is stored | "5 most similar cases" is done with **random-forest proximity** (shared-leaf frequency) — the model's *own* similarity measure — not a separate k-NN model. See §5. |
| Features split into *modifiable* (BMI, activity, diet, smoking, heavy drinking) and *fixed* (age, sex, prior stroke/heart disease, income) | The what-if / counterfactual feature only ever varies **modifiable** factors. It must be labelled as *model association, not medical advice*. See §6. |
| Cross-sectional survey, ~230k US adults, income/education are predictive | Ethics/limitations section is mandatory (screening aid ≠ diagnosis; associational ≠ causal; monitor flag-rate fairness). See §9. |

---

## 2. Personas & flows

- **Clinic / community screening** (primary): a health worker enters answers from a short
  interview, reads the result to the person, discusses modifiable factors.
- **Self-screening** (web): a member of the public answers for themselves.

Both use the same API. The mobile app targets the first persona; the web app serves both.

---

## 3. The questionnaire

21 model features, grouped into plain-language sections. Every question offers
**Yes / No / Not sure** (or a value + "prefer not to say").

| Section | Questions asked | Maps to |
|---|---|---|
| About you | Age band (dropdown 18–24 … 80+); Sex; Highest education; Household income band | `Age`, `Sex`, `Education`, `Income` |
| Body | Height; Weight (units toggle) | → compute `BMI` |
| Conditions you've been told you have | High blood pressure? High cholesterol? Ever had a stroke? Ever had heart disease or a heart attack? Serious difficulty walking / climbing stairs? | `HighBP`, `HighChol`, `Stroke`, `HeartDiseaseorAttack`, `DiffWalk` |
| Checks & access to care | Cholesterol checked in last 5 years? Any health-care coverage? In the last year, needed a doctor but couldn't afford one? | `CholCheck`, `AnyHealthcare`, `NoDocbcCost` |
| Lifestyle | Smoked ≥100 cigarettes in your life? Heavy drinker (≥14/wk men, ≥7/wk women)? Any physical activity in last 30 days? Eat fruit daily? Eat vegetables daily? | `Smoker`, `HvyAlcoholConsump`, `PhysActivity`, `Fruits`, `Veggies` |
| How you've felt (last 30 days) | General health (Excellent→Poor, 5 levels); # days mental health not good (0–30); # days physical health not good (0–30) | `GenHlth`, `MentHlth`, `PhysHlth` |

**Question metadata** (label, help text, type, options, `modifiable: bool`,
`impute_ok: bool`) lives in `api/config.py` as a single list so the client can render
the form from it and the report can quote it.

### Missing / "unsure" handling

1. Client sends `null` for any unanswered / "not sure" field.
2. `POST /predict` accepts all fields as optional **except** `Age` and `Sex`
   (identity anchors). Height/weight optional → if absent, `BMI` is imputed too.
3. The pipeline's `SimpleImputer(strategy="median")` fills every `null` with the
   training-set median. No client-side imputation.
4. API returns a **completeness score** = `1 − (#imputed / 21)` and the list of
   imputed fields.
5. **Uncertainty from missingness**: for each imputed binary field, re-run
   `predict_proba` with that field forced to 0 and to 1; report the resulting
   min–max band. Result shows e.g. *"Estimated risk 0.62 (could be 0.55–0.70
   depending on 2 unanswered questions)."*
6. If completeness < 0.8, the result screen shows a caution banner.

### BMI when height/weight are missing

`BMI` is the one field where a plain training-set median is a poor fallback: this
dataset over-represents people with diabetes, who have a higher BMI, so the pooled
median is nudged upward. Because the training data has **no missing `BMI`**,
imputation only ever happens at inference, so the app is free to use a better
estimate. In priority order:

1. **Compute it** from the height and weight questions. This should be the usual case.
2. **Fallback — a conditional lookup**: median `BMI` among the **non-diabetic**
   training rows (`Diabetes_binary == 0`), grouped by (`Age` band × `Sex`). This is
   "typical BMI for a healthy person of your age and sex". It conditions on the two
   variables that most explain BMI and, by filtering to the negative class, removes
   the diabetes skew — with no external data source to license or cite.
3. **Optional upgrade** — replace that table with NHANES / CDC national mean `BMI`
   by age group and sex if the report needs a formal general-population reference.
   Adds a dependency and a citation for a small gain over step 2.

Do the `BMI` fill in the API layer, before the pipeline, so the pipeline's
`SimpleImputer` stays the catch-all for the other fields. Mark the row `BMI_imputed`
and widen the uncertainty band by also scoring at the age/sex 25th and 75th `BMI`
percentiles. A `BMI` that *is* supplied but lies outside the training range is a
separate case, handled by the clamp-with-warning guard in §8.1.

---

## 4. What the API produces per screening

| Output | Source | Shown as |
|---|---|---|
| Risk probability + band (Low / Moderate / High) | `pipeline.predict_proba`; band from configurable thresholds | gauge + label + the uncertainty band from §3 |
| Top contributing factors | SHAP `TreeExplainer` on the imputed vector | signed horizontal bars with plain-language labels ("High blood pressure +0.11", "Eats vegetables daily −0.03") |
| 5 most similar respondents | RF proximity (§5) | de-identified profile chips + *"4 of your 5 closest matches had diabetes"* |
| Modifiable-factor what-ifs | re-scored vectors (§6) | toggles/sliders with live Δrisk; BMI curve |
| Nearest realistic counterfactual | greedy search over modifiable factors (§6) | one sentence: *"If physically active and BMI 30 instead of 41, the estimate drops from High (0.68) to Moderate (0.49)."* |
| Session history + cohort metrics | request log (§7) | history list; small monitoring dashboard |

---

## 5. "5 most similar cases" with a random forest

The deployed model *is* a random forest, so use its built-in notion of similarity —
**proximity** = how often two samples fall in the same leaf across all trees. This is
supervised (reflects what the model learned) and needs no extra model.

### Build once — in `api/build_artifacts.py` (§11.2)

```
pipe   = joblib.load("model/model_pipeline.joblib")
rf     = pipe.named_steps["clf"]
Xtr    = engineer(train_df)[MODEL_FEATURES]              # ~230k rows
# stratified subsample for a fast, memory-bounded index
idx    = stratified_sample(Xtr.index, n=30_000, by=y_train)
Xsub   = pipe.named_steps["prep"].transform(Xtr.loc[idx])
leaves = rf.apply(Xsub)                                  # (30000, n_estimators) int32
profiles = build_profile_strings(train_df.loc[idx])     # "F, 55–59, BMI 34, HighBP, inactive"
outcomes = y_train.loc[idx].to_numpy()

joblib.dump({"leaves": leaves, "outcomes": outcomes,
             "profiles": profiles, "n_estimators": rf.n_estimators},
            "api/artifacts/neighbor_index.joblib")
```

### Query at inference — `api/similarity.py`

```
def similar_cases(raw_vec, k=5):
    x_pre   = prep.transform(engineer(raw_vec)[MODEL_FEATURES])
    q_leaf  = rf.apply(x_pre)[0]                         # (n_estimators,)
    prox    = (index["leaves"] == q_leaf).mean(axis=1)   # (30000,) in [0,1]
    top     = np.argsort(prox)[::-1][:k]
    return {
        "method": "random-forest proximity (shared-leaf frequency)",
        "neighbors": [{"profile": index["profiles"][i],
                       "outcome": "diabetes/pre-diabetes" if index["outcomes"][i] else "no diabetes",
                       "proximity": round(float(prox[i]), 3)} for i in top],
        "n_with_diabetes": int(index["outcomes"][top].sum()),
    }
```

Fallback (simpler, if proximity is too slow/large): `sklearn.neighbors.NearestNeighbors`
on `prep.transform(Xsub)` with Euclidean distance — same response shape, `method` field
set accordingly.

**Privacy note for the report:** the training data is public-domain CC0 survey data with
no identifiers; still, only a de-identified one-line profile and the outcome are shown,
never a raw row.

---

## 6. What-if and counterfactual explanations

Complements SHAP: SHAP says *why the current score is what it is*; what-if says
*how the score moves if a modifiable factor changes*.

**Modifiable factors** (the only ones the feature is allowed to vary):
`BMI` (via weight), `Smoker`, `HvyAlcoholConsump`, `PhysActivity`, `Fruits`, `Veggies`.
Engineered features (`TotalUnhealthyDays`, `CardioRisk`) recompute automatically.

**Fixed factors** (never varied): `Age`, `Sex`, `Education`, `Income`, `Stroke`,
`HeartDiseaseorAttack`, `HighBP`, `HighChol`, `CholCheck`, `AnyHealthcare`, `NoDocbcCost`,
`DiffWalk`, `GenHlth`, `MentHlth`, `PhysHlth`. (`HighBP` / `HighChol` are downstream
health states — excluded so the app never implies "just don't have high blood pressure".)

### `POST /whatif`

```
def whatif(raw_vec):
    base = predict_proba(raw_vec)
    out  = {"base_risk": base, "factors": [], "bmi_sweep": []}

    for f, healthier_value in [("Smoker", 0), ("HvyAlcoholConsump", 0),
                               ("PhysActivity", 1), ("Fruits", 1), ("Veggies", 1)]:
        if raw_vec[f] == healthier_value:
            continue                                     # already at the healthy value
        v = {**raw_vec, f: healthier_value}
        out["factors"].append({"feature": f, "from": raw_vec[f], "to": healthier_value,
                               "new_risk": predict_proba(v),
                               "delta": predict_proba(v) - base})

    # 1-D BMI sweep from current down to 25 (and up a little, for "or vice versa")
    for bmi in np.arange(min(raw_vec["BMI"], 25), max(raw_vec["BMI"], 45) + 1, 1):
        out["bmi_sweep"].append({"bmi": float(bmi),
                                 "risk": predict_proba({**raw_vec, "BMI": float(bmi)})})
    return out
```

Client renders each `factors[]` entry as *"Stop heavy drinking → risk 0.61 → 0.55 (−0.06)"*
and `bmi_sweep` as a line chart with the current BMI marked. Because the sweep spans both
directions, it shows *"BMI 80 → 60: risk 0.74 → 0.58"* and the reverse.

### `POST /counterfactual` — smallest change that lowers the band

Greedy search: repeatedly apply the single modifiable change with the largest risk
reduction until the score drops below the "High" threshold or no modifiable factor
remains.

```
def counterfactual(raw_vec, target_band="Moderate"):
    cur, changes = dict(raw_vec), []
    while band(predict_proba(cur)) worse_than target_band:
        best = argmax over modifiable f of (risk_drop from setting f to its healthy value)
        if best gives no improvement: break
        changes.append(best); cur[best.feature] = best.healthy_value
    return {"changes": changes, "from_risk": predict_proba(raw_vec),
            "to_risk": predict_proba(cur),
            "feasible": band(predict_proba(cur)) == target_band}
```

**Mandatory caveat** (returned in the payload and shown on screen):
> These figures show how the model's estimate changes given patterns in survey data.
> They are not a prediction of your future health and are not medical advice.

---

## 7. Monitoring, history & operator dashboard

### 7.1 Request log & history

`api/monitoring.py` writes every request to a small SQLite table
(`ts, session_id, features_json, imputed_fields, probability, band`). Powers:

- `GET /history?session=…` — this session's past screenings.
- `GET /metrics` — count, live flag rate, band distribution, mean risk by `Age` band,
  share of screenings that hit the `BMI` clamp or fell below the completeness threshold
  (§8.1), mean completeness. This page shows **what actually happened** over time.

### 7.2 Terms

The model outputs `p = predict_proba(x)[1]` ∈ [0, 1] — the estimated probability that
the person has diabetes or pre-diabetes. Turning that number into an action needs
**cut-points**:

| Term | Meaning |
|---|---|
| `high_cut` | If `p ≥ high_cut` the result is **High** → "refer for a confirmatory blood test now". |
| `moderate_cut` | If `moderate_cut ≤ p < high_cut` the result is **Moderate** → "re-screen in a year / lifestyle advice". Below `moderate_cut` is **Low**. Constraint: `0 ≤ moderate_cut ≤ high_cut ≤ 1`. |
| **flag rate** | The share of screened people who land in **High** (i.e. `p ≥ high_cut`). It is the operational cost knob: a lower `high_cut` raises the flag rate, catches more true cases (higher recall) but generates more false alarms (lower precision) and more confirmatory-test load. A clinic that can run 200 blood tests a month needs a `high_cut` that flags ≈ 200 people. |
| **holdout scores** | The array of `(y_true, p)` pairs produced when the notebook scored the final model on data it did not train on (validation for tuning, test for the honest read-out). Saved as `model/holdout_scores.joblib`. Because the labels are known, the dashboard can recompute precision / recall / confusion / flag rate at **any** candidate threshold instantly, without re-running the model. |

The notebook currently uses a single 0.5 cut for its binary prediction; the app
generalises that to two operator-tunable cuts.

### 7.3 Dashboard page

**Why this is a page and not a constant.** The model produces a probability; the
cut-points that turn it into *Low / Moderate / High* are a **deployment policy**, not
a modelling choice. The right values depend on things the notebook cannot know: how
many confirmatory blood tests the clinic can run per month, the local diabetes
prevalence, and how much worse a missed case is than a false alarm. Two clinics
running the same model legitimately want different cut-points, and the person who
should choose them is a clinic lead, not the model author. The dashboard exposes that
choice to a non-ML operator and shows the consequence of each setting before it is
committed.

**Controls.** Two sliders bound to `moderate_cut` and `high_cut` (enforcing
`moderate_cut ≤ high_cut`), plus optional presets ("Capacity-limited",
"High-sensitivity", "Balanced").

**Worked example of moving the slider.** Suppose `high_cut` starts at 0.50 and the
holdout preview shows recall 0.58, precision 0.44, flag rate 23%. The operator drags
`high_cut` down to 0.34; the panel updates in place to recall 0.90, precision 0.34,
flag rate 41%. In words: dropping the cut-point catches 9 in 10 true cases instead of
6 in 10, but now 4 in 10 people screened are referred and two-thirds of those
referrals are false alarms. The operator weighs that against the clinic's test
capacity and clicks **Save**, which persists the new cut-points for every subsequent
screening.

**Live trade-off panel** — recomputed on every slider move from
`model/holdout_scores.joblib`:

- Confusion matrix at `high_cut` (treating **High** as positive): TN / FP / FN / TP.
- Precision, Recall (sensitivity), Specificity, F1.
- Flag rate `(FP + TP) / N`, and expected absolute numbers, e.g. *"per 1,000 screened
  → ≈ 410 flagged, ≈ 155 true cases caught, ≈ 30 missed, ≈ 255 false alarms."*
- ROC curve and PR curve with a dot at the current operating point.
- Histogram of `p` for positives vs negatives with both cut-lines drawn — shows how
  much the classes overlap at that cut.

**Recommendation helper.** The operator picks an objective — a target recall, a target
flag rate, maximum F1, or Youden's J — and the page scans thresholds on the holdout
scores for the value that meets it: *"Target recall 90% → `high_cut` = 0.34, flag rate
41%, precision 0.34."*

**Persistence & audit.** `POST /config/thresholds {moderate_cut, high_cut}` (admin-key
guarded) writes `api/runtime/thresholds.json`; `inference.band()` reads it per request. `GET /config` returns the current values plus
who set them and when. Every change is logged so a shift in the live flag rate on the
monitoring page can be traced to a policy change. Changing thresholds never touches
the model artifact and is reversible instantly.

**Guardrails.** Clamp to sane ranges; warn if the implied flag rate exceeds a
capacity figure the operator enters; warn if recall drops below a floor (e.g.
*"< 0.60 recall — this configuration misses 40% of cases"*).

Monitoring (7.1) and the dashboard (7.3) are complementary: tune on the dashboard
against a labelled sample, deploy, then watch the live flag rate on monitoring — if it
drifts away from the holdout preview, that is itself a distribution-shift signal.

---

## 8. Input guards & other extras

### 8.1 Input-range guard and thin-input annotation

Generalising to *new individuals* is the point of the model, so the app does not try
to detect and refuse every input that differs from the training data. Only two
concrete situations are handled, because both make the returned probability less
trustworthy in a way the model itself cannot signal.

**1. Out-of-range `BMI` — clamp with a visible warning.** Every feature except `BMI`
is a bounded code (0/1, `GenHlth` 1–5, `Age` 1–13, the day-counts 0–30, …) and is
range-checked by the request schema, so it cannot carry an artificial value. `BMI` is
computed from free-text height and weight and is the one feature that can. If it falls
outside the range actually seen in training, `[12.0, 98.0]`, it is clamped to the
nearest bound and the response carries a warning:

> BMI value capped to the maximum value in our dataset (98.0).
> BMI value capped to the minimum value in our dataset (12.0).

Clamping — rather than rejecting — keeps the screening usable when a value is merely a
unit mix-up or a typo, while the warning tells the operator the prediction used a
boundary value rather than the number entered. This is an inference-time guard only;
it does not change how the model was trained (the notebook keeps all training `BMI`
values as-is, and Appendix A of the notebook shows clamping does not move the metrics).

**2. Many "not sure" answers — annotate the result.** When a large share of the 21
questions are left blank, the prediction rests mostly on imputed medians rather than
on the person's own answers. The completeness score from §3 already measures this; if
it is low, the result screen adds a line such as *"7 of 21 answers were left blank and
filled with typical values — treat this estimate as indicative only"*, alongside the
uncertainty band. No separate model is needed.

Both signals are logged so `/metrics` can report the share of screenings that hit the
`BMI` clamp or came in below the completeness threshold; a rising share is a sign the
questionnaire or the intake process needs attention. A full statistical
out-of-distribution detector (RF-proximity distance, or an `IsolationForest` on the
preprocessed training sample) is possible but is not worth the extra artifact here,
given that the feature space is otherwise fully bounded.

### 8.2 Batch mode

`POST /predict/batch` accepts a CSV and returns predictions + bands as a file, for a
health worker screening a list.

---

## 9. Limitations & ethics (for the report and an in-app "About" page)

- Screening aid only — not a diagnosis; a positive flag means "refer for a blood test".
- The model is **associational**: it learned correlations from a one-time survey.
  What-if numbers are how the *model* responds, not proven causal effects.
- Training population is US adults, 2015; transfer to other populations is untested.
  The model has a bounded **domain of validity** — inside it, generalising to new
  people is exactly the job. The §8.1 guards cover the two edge cases that most affect
  this app: an out-of-range `BMI` and a mostly-blank questionnaire.
- Imputed answers lower certainty — the result surfaces completeness and an uncertainty
  band.
- `Income` and `Education` are predictive; `/metrics` should be watched for disparate
  flag rates across those bands.

---

## 10. API surface

All `POST` endpoints take the same body: the 21 raw questionnaire fields, every one
nullable except `Age` and `Sex` (plus optional `height`, `weight`, `session_id`).

| Endpoint | Returns |
|---|---|
| `POST /predict` | `risk`, `band`, `probability`, `uncertainty_band`, `completeness`, `imputed_fields`, `warnings[]` (e.g. `bmi_capped_max`, `low_completeness`) |
| `POST /explain` | `shap: [{feature, label, value, direction}]`, `base_value` |
| `POST /similar` | `neighbors: [{profile, outcome, proximity}]`, `n_with_diabetes`, `method` |
| `POST /whatif` | `base_risk`, `factors: [{feature, from, to, new_risk, delta}]`, `bmi_sweep: [{bmi, risk}]` |
| `POST /counterfactual` | `changes: [{feature, from, to}]`, `from_risk`, `to_risk`, `feasible` |
| `GET /history` | list of past predictions for the session |
| `GET /metrics` | monitoring aggregates (live flag rate, band distribution, …) |
| `GET /threshold-curve` | holdout precision / recall / flag rate / confusion at a grid of `high_cut` values, for the dashboard |
| `POST /config/thresholds` | set `{moderate_cut, high_cut}` (admin-key guarded); persisted for all later `/predict` calls |
| `GET /config` | current thresholds + who set them and when |
| `GET /model-info` | model name, version, training date, feature schema, current thresholds |
| `GET /healthz` | liveness |

Convenience: `POST /predict?include=explain,similar,whatif` returns everything in one
response so the mobile app makes a single call.

---

## 11. Repository layout

Follows Appendix A of the assignment (`data/`, `notebook/`, `model/`, `api/`, `web/`,
`mobile/`, `requirements.txt`). Appendix A allows either `preprocessor.joblib` +
`model.joblib` **or** a single `model_pipeline.joblib`; this project uses the single
combined pipeline that the notebook already saves in §22.

The one addition to Appendix A is the **split between graded deliverables and
auxiliary artifacts** (see §11.1): `model/` holds only the persisted pipeline that
the assignment grades; everything the extra features need is regenerable and lives
under `api/artifacts/` and `api/runtime/`.

```
diabetes/
├── data/
│   └── diabetes_012_health_indicators_BRFSS2015.csv
├── notebook/
│   └── diabetes.ipynb
├── model/                          ← the persisted ML pipeline (GRADED — Appendix A, Part X)
│   ├── model_pipeline.joblib       preprocessing + feature transform + model   (notebook §22)
│   ├── feature_names.joblib        ordered feature contract                    (notebook §22)
│   ├── input_schema.json           raw-input contract + valid ranges           (notebook §22)
│   └── holdout_scores.joblib       (y_true, p) on val + test — evaluation record (notebook §19)
├── api/
│   ├── main.py            FastAPI app; routes; CORS for Flutter Web
│   ├── schema.py          Pydantic: QuestionnaireInput (Optional fields), response models
│   ├── config.py          question metadata, modifiable list, default thresholds
│   ├── validate.py        schema range checks + BMI clamp to [12.0, 98.0] + warnings (§8.1)
│   ├── inference.py       load pipeline; engineer(); predict; band(); completeness/uncertainty
│   ├── explain.py         SHAP TreeExplainer wrapper + label mapping
│   ├── similarity.py      RF-proximity query (loads api/artifacts/neighbor_index.joblib)
│   ├── whatif.py          modifiable-factor deltas, BMI sweep, counterfactual search
│   ├── monitoring.py      request-log + /metrics aggregation
│   ├── dashboard.py       /threshold-curve, /config, /config/thresholds (reads model/holdout_scores)
│   ├── build_artifacts.py ONE-OFF packaging step: model/ + data/ → api/artifacts/*  (see §11.2)
│   ├── artifacts/         serving-only, regenerable — NOT graded, git-ignorable
│   │   ├── neighbor_index.joblib     leaves + outcomes + profiles
│   │   ├── shap_background.joblib    training sample for TreeExplainer
│   │   └── ood_detector.joblib       optional
│   └── runtime/           mutable app state — git-ignored
│       ├── thresholds.json           {moderate_cut, high_cut, set_by, set_at}
│       └── requests.sqlite           request log
├── web/                   Streamlit multipage (or React)
│   ├── app.py             1 Questionnaire · 2 Result · 3 History · 4 Dashboard
│   └── api_client.py
├── mobile/                Flutter — questionnaire stepper → result → what-if → history
├── requirements.txt
└── README.md             reproduce: run notebook → run api/build_artifacts.py → run api
```

### 11.1 What counts as the "persisted pipeline"

Part X of the assignment says to save *preprocessing, feature transformation, the
trained model, and model configuration where necessary*. That is exactly
`model/model_pipeline.joblib` (imputer + scaler + random forest in one object) plus
`feature_names.joblib` and `input_schema.json` as its configuration. `holdout_scores.joblib`
is kept in `model/` too because the notebook produces it and it is an immutable record
of the model's evaluated performance.

`neighbor_index.joblib`, `shap_background.joblib` and `ood_detector.joblib` are **not**
preprocessing and **not** the model — they are indexes *derived from* the trained
pipeline to power the similar-cases and explanation features. They are regenerable at
any time from `model/` + `data/`, so they live under `api/artifacts/` and can be
git-ignored. `thresholds.json` and `requests.sqlite` are mutable runtime state, not
artifacts, and live under `api/runtime/`.

### 11.2 Where the artifacts are built

| File | Built by | Why there |
|---|---|---|
| `model_pipeline.joblib`, `feature_names.joblib`, `input_schema.json` | **notebook** §22 | these are the graded deliverable; §22 "Model persistence" already writes them |
| `holdout_scores.joblib` | **notebook** §19 | one line — `y_val/proba_val` and `y_test/proba_test` already exist there |
| `neighbor_index.joblib`, `shap_background.joblib`, `ood_detector.joblib` | **`api/build_artifacts.py`** | they need `shap` and heavy full-training-data passes that do not belong in the 23-section ML-experiment notebook; a separate packaging step mirrors real train → package → serve flow |
| `thresholds.json` | seeded by `build_artifacts.py` (e.g. the max-F1 threshold on validation), then edited by the dashboard at runtime | mutable policy, not a training artifact |

Name it `build_artifacts.py`, **not** `setup.py` — `setup.py` means a setuptools
packaging script and would be misleading. Document it in the README as step 2 of the
reproduce chain (§12).

---

## 12. End-to-end workflow

```
① notebook/diabetes.ipynb        — Data → Clean → Represent → Learn → Evaluate → Persist
     saves  model/model_pipeline.joblib, model/feature_names.joblib, model/input_schema.json   (§22)
     saves  model/holdout_scores.joblib = (y_true, p) on val + test rows                       (§19)

② python api/build_artifacts.py  — Package (run once, after the notebook)
     loads model/model_pipeline.joblib + data/*.csv
     → api/artifacts/neighbor_index.joblib   (RF leaf matrix for 30k stratified rows + outcomes + profiles)
     → api/artifacts/shap_background.joblib  (sample for TreeExplainer)
     → api/artifacts/ood_detector.joblib     (optional)
     → api/runtime/thresholds.json           (seed: max-F1 threshold on validation)

③ uvicorn api.main:app           — Deploy
     loads pipeline + artifacts once into memory; reads api/runtime/thresholds.json per request

④ client renders the questionnaire from config.py
     collects Yes / No / Not-sure  + height/weight
     builds JSON: null for every "not sure" / unanswered field

⑤ POST /predict?include=explain,similar,whatif
     validate (Age, Sex required; schema range checks; clamp BMI to [12.0, 98.0] with a
       warning; fill missing BMI from the age×sex healthy-population table)
     → engineer() → pipeline (imputes remaining nulls) → predict_proba
     → SHAP on the imputed vector
     → RF proximity → 5 neighbours + outcome tally
     → what-if deltas + BMI sweep + counterfactual
     → append request to api/runtime/requests.sqlite

⑥ client renders result
     risk gauge + band + uncertainty/completeness banner
     SHAP factor bars (plain language)
     "4 of your 5 closest matches had diabetes"  + profile chips
     what-if toggles/sliders with live Δrisk; BMI curve
     counterfactual sentence + medical-advice caveat

⑦ History screen → GET /history
   Dashboard screen → GET /metrics (live) + GET /threshold-curve (holdout preview);
                      POST /config/thresholds writes api/runtime/thresholds.json,
                      which inference.band() reads on the next request
```

---

## 13. Build order (suggested)

1. `inference.py` + `validate.py` (schema range checks, BMI clamp + warnings,
   completeness score) + `POST /predict` + `schema.py` with nullable fields — the
   minimum that satisfies the assignment.
2. Streamlit questionnaire + result screen against `/predict`.
3. Flutter questionnaire + result screen against the same endpoint (screenshots for the
   mobile deliverable).
4. `explain.py` + `/explain`; add the factor-bar chart to both clients.
5. `api/build_artifacts.py` + `similarity.py` + `/similar`; add the "similar cases" panel.
6. `whatif.py` + `/whatif` + `/counterfactual`; add sliders/toggles + BMI curve.
7. `monitoring.py` + `/history` + `/metrics`; history list (mobile) + live monitoring (web).
8. `dashboard.py` + `/threshold-curve` + `/config/thresholds`; save `holdout_scores.joblib`
   from the notebook (§19); build the Dashboard page (sliders + trade-off panel +
   recommendation helper). `inference.band()` reads `api/runtime/thresholds.json`.
9. Optional: statistical OOD detector, batch mode.
10. Write the limitations/ethics section into the report and an in-app "About" page.
