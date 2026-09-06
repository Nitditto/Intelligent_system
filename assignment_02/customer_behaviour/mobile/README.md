# Product-recommendation — Mobile app (Flutter)

A thin **client** of the FastAPI service. It trains and runs **no model** — it
collects one review, `POST`s it to `/predict`, and shows the response.

Target: **`is_recommended`** — will the reviewer tick "recommends this product".

The UI follows the same design language as the web client (`.claude/skills/frontend-ui`):
tokenised colours (light **and** dark, `ThemeMode.system`), a 4px spacing scale, one
accent colour, and hand-built components — no default Material look.

## Screens

1. **Review form** — a **3-step wizard** (Your skin profile → The product → The review)
   with a horizontal stepper (done steps tappable to jump back). Fields render from
   `GET /questions`, grouped into cards; a prefilled sample so it works immediately;
   a "Load a real review" action (top-right) opens a sheet of real reviews from
   `GET /samples`; the review step offers example chips. `← Back` / `Predict` in a
   sticky footer; `Predict` is disabled until the step's required field is set.
2. **Result** — a tinted **verdict block** ("would / would not recommend"), a
   `P(recommend)` **meter** with the 50% cut-off rule and a plain "read it as…" line,
   a **"what the model saw"** key→value card, **review-term chips** (toward / against),
   a **"how to use this"** block, a diverging-bar **contribution chart** (linear-SHAP)
   with a waterfall line, and a collapsible **"How this works"**.

## Run

```bash
# API first: from customer_behaviour/  ->  python -m uvicorn api.main:app --port 8000
cd mobile
flutter clean          # needed after pulling — the package was renamed & restructured
flutter pub get
flutter run --dart-define=API_URL=http://10.0.2.2:8000   # Android emulator -> host
```

`10.0.2.2` is the Android emulator's alias for the host's `localhost`. On a real
device pass your machine's LAN IP. A quick desktop preview:
`flutter run -d chrome --dart-define=API_URL=http://localhost:8000`.

## Layout

```
mobile/lib/
  main.dart                     MaterialApp — light + dark theme, ThemeMode.system
  theme.dart                    design tokens: AppColors ThemeExtension + Sp / Rad / Ty
  api_client.dart               REST client (/healthz /questions /samples /model-info /predict)
  models.dart                   FormFieldSpec · TermPull · PredictResult
  widgets/
    primitives.dart             AppCard · SectionLabel · AppButton · Pill · AppField
    result_parts.dart           StepperBar · VerdictBlock · Meter · KvGrid · ChipRow
                                · ContribChart · HowItWorks · OfflineBanner
  screens/
    review_form_screen.dart     the 3-step wizard (ReviewFormScreen)
    result_screen.dart          the prediction screen
```

## Screenshots for the report (IDs M1–M4)

1. **M1** — review form, step 1 (skin profile) with the stepper.
2. **M2** — review form, step 3 (the review) with an example loaded.
3. **M3** — result screen: verdict block + `P(recommend)` meter + "what the model saw".
4. **M4** — result screen scrolled to the contribution chart + review-term chips.
