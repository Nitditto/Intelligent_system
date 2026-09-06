# Product-recommendation — Mobile app (Flutter)

A thin **client** of the FastAPI service. It trains and runs **no model** — it
collects one review, `POST`s it to `/predict`, and shows the response.

Target: **`is_recommended`** — will the reviewer tick "recommends this product".

## Screens

1. **Review form** — fields grouped by section, rendered from `GET /questions`
   (skin profile · the product · the review). A prefilled "oily skin, broke me
   out" sample; tap an example chip on the review field to swap in a real review.
   Floating **Predict** button → `POST /predict`.
2. **Result** — verdict ("Would / would not recommend"), `P(recommend)` bar with
   the 50% cut-off, the profile/product signals, the review terms pulling each
   way, a plain-language interpretation, and a model/`§14a` footer.

## Run

```bash
# API first: from customer_behaviour/  ->  python -m uvicorn api.main:app --port 8000
cd mobile
flutter pub get
flutter run --dart-define=API_URL=http://10.0.2.2:8000   # Android emulator -> host
```

`10.0.2.2` is the Android emulator's alias for the host's `localhost`. On a real
device pass your machine's LAN IP.

## Layout

```
mobile/lib/
  main.dart                     app shell + verdict colour helper
  api_client.dart               REST client (/healthz /questions /model-info /predict)
  models.dart                   FormFieldSpec · TermPull · PredictResult
  screens/
    order_form_screen.dart      the review form (ReviewFormScreen)
    result_screen.dart          the prediction screen
```

## Screenshots for the report (IDs M1–M3)

1. **M1** — review form with the sample loaded.
2. **M2** — result screen: "won't recommend" verdict + `P(recommend)` bar.
3. **M3** — result screen scrolled to the review-term chips + interpretation.
