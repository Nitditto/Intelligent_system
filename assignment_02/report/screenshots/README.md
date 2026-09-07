# Screenshot evidence - Assignment 02

All three stacks were run locally and captured on 2026-09-07.
Backends: FastAPI + Uvicorn (`python -m uvicorn api.main:app`). Web: React + Vite dev
server. Mobile: the Flutter app built for web (`flutter build web`) and served statically,
rendered at a 390 x 844 device viewport (`deviceScaleFactor` 3).

| Suffix | App | Backend port | Web port | Mobile served |
|---|---|---|---|---|
| `-D` | Diabetes (`diabetes/`) | 8000 | 5176 | `diabetes/mobile` -> 8091 |
| `-H` | House price (`house_price/`) | 8002 | 5175 | `house_price/mobile` -> 8090 |
| `-C` | Customer behaviour (`customer_behaviour/`) | 8000 | 5174 | `customer_behaviour/mobile` -> 8092 |

## Web application evidence (Appendix D / report Section 13.3)

| File | Report ID | Shows |
|---|---|---|
| `W1-D_input.png` | W1-D | Diabetes - health questionnaire with a valid set of answers entered |
| `W2-D_result.png` | W2-D | Diabetes - result: **82 % High risk**, "refer for a confirmatory blood test", SHAP force plot, nearest survey respondents, what-if BMI slider |
| `W3-D_docs.png` | W-docs-D | Diabetes - FastAPI `/docs`, `POST /predict` expanded (request/response schema) |
| `W1-H_input.png` | W1-H | House price - step 1 of the valuation wizard, "District 7 Apartment (HCMC)" preset filled |
| `W2-H_result.png` | W2-H | House price - **Predicted valuation 9.90 tỷ VNĐ**, low/high range, SHAP contribution waterfall, plain-language summary |
| `W3-H_docs.png` | W-docs-H | House price - FastAPI `/docs`, `POST /predict` expanded |
| `W1-C_input.png` | W1-C | Customer behaviour - step 1 (skin profile) of the 3-step wizard, "API connected" |
| `W2-C_result.png` | W2-C | Customer behaviour - **WON'T RECOMMEND, P(recommend) = 0 %**, probability meter, "what the model saw", review terms, linear-SHAP chart |
| `W3-C_docs.png` | W-docs-C | Customer behaviour - FastAPI `/docs`, `POST /predict` expanded |

## Mobile application evidence (Appendix E / report Section 14.3)

| File | Report ID | Shows |
|---|---|---|
| `M1-D_input.png` | M1-D | Diabetes - Flutter input screen (About you / Body / Conditions) |
| `M2-D_result.png` | M2-D | Diabetes - Flutter "Result" screen: **27 % Low**, range 15-46 %, "Why this score" contributions, similar respondents (`POST /predict` 200) |
| `M1-H_input.png` | M1-H | House price - Flutter input screen, Rạch Giá preset, "API Ready (RandomForest)" |
| `M2-H_result.png` | M2-H | House price - Flutter valuation result: **~ 3.01 tỷ VNĐ** (38.3 M VND/m^2), unit-price card, analysis text (`POST /predict` 200) |
| `M1-C_input.png` | M1-C | Customer behaviour - Flutter step 1 (skin profile), "API connected" |
| `M2-C_result.png` | M2-C | Customer behaviour - Flutter "Prediction" screen: **WON'T RECOMMEND, 0 %**, probability meter, "what the model saw", review terms (`POST /predict` 200) |

The `POST /predict 200` line in each mobile run (captured from the browser network log
while driving the built Flutter client) is the evidence that the mobile app talks to the
deployed service - `Training != Inference`.

## Notebook figures (report Section 4-Section 6 EDA / evaluation, IDs N1-N9 per app)

**Plots** - `image/png` outputs pulled straight from the executed notebooks
(`sec<N>` in the filename is the source section):

- `NB_D_0*` - diabetes: Section 9 outliers, Section 10 EDA grid, Section 10 correlation heatmap, Section 19 confusion + ROC.
- `NB_H_0*`/`NB_H_1*` - house price: Section 9 outliers; Section 10.1 distribution set (price / area / unit price);
  Section 10.2 correlation + eta^2 views; Section 10.3 missingness / skew / QQ; Section 19 predicted-vs-actual + residuals.
- `NB_C_0*` - customer behaviour: Section 9 outliers, Section 10 EDA grid, Section 19 confusion + ROC.
- `NB_C_NB_appendixB_kmeans_a/b` - App 3 Appendix B: K-Means elbow/silhouette + segment profile / PCA scatter.

**Cell captures** - `NB_<app>_N<n>_*` are full notebook cells (code + rendered table/text
output), rendered via `nbconvert --to html` and clipped. One set per app:

| File stem | Report ID | Section | Shows |
|---|---|---|---|
| `N1_sec4_inspect` | N1 | Section 4 Dataset inspection | `df.shape` / `df.head` / `df.info` / `df.describe` |
| `N2_sec5_quality` | N2 | Section 5 Data-quality analysis | NaN / duplicate / invalid / imbalance table |
| `N4_sec12_representation` | N4 | Section 12 Data representation | one raw record -> its feature vector -> `X` shape / dtype (Part IV requirement) |
| `N7_sec18_comparison` | N7 | Section 18 Model comparison | the 5-/8-model metric table + winner |
| `N6_sec10_topics_mi` (App 3 only) | N6-C | Section 10 | topic-bucket recommend rates + mutual-information table |
| `N9_sec20_error_analysis` (App 3 only) | N9-C | Section 20 | false-negative / false-positive feature comparison |
