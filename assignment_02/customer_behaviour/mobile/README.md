# Order-satisfaction — Mobile app (Flutter)

A thin **client** of the FastAPI service. It trains and runs **no model** — it
collects one order, `POST`s it to `/predict`, and shows the response.

```
Mobile UI  ->  REST API  ->  build_features + preprocessing + LogisticRegression
           <-  JSON       <-  prediction + confidence + signals
```

Two screens:

| Screen | Contents |
|---|---|
| **Order form** (`order_form_screen.dart`) | fields rendered from `GET /questions` — number / choice / date / free-text. Prefilled with a "late + damaged" sample. Client-side validation (required fields, numeric ranges). A **Predict satisfaction** button. Header shows API status + the model name from `GET /model-info`. |
| **Result** (`result_screen.dart`) | satisfied / at-risk verdict, a **confidence** value, a `P(satisfied)` bar with the decision cut-off marked, the delivery / comment **signals**, the comment **terms** pushing each way, and a one-paragraph interpretation. "Edit the order" returns to the form. |

## Prerequisites

- Flutter 3.19+ (`flutter --version`)
- The API reachable from the device/emulator. On the **Android emulator**,
  `http://10.0.2.2:8000` is the host's localhost (the default in `api_client.dart`).

## Run

```bash
cd mobile
flutter create .            # generate android/ ios/ … for this package (first time only)
flutter pub get
flutter run --dart-define=API_URL=http://10.0.2.2:8000
```

For a physical device on the same Wi-Fi, use the host's LAN IP:

```bash
flutter run --dart-define=API_URL=http://192.168.1.20:8000
```

Android also needs cleartext HTTP for a plain `http://` URL — after `flutter create .`,
add `android:usesCleartextTraffic="true"` to the `<application>` tag in
`android/app/src/main/AndroidManifest.xml` (or point `API_URL` at an `https` tunnel).

## Layout

```
mobile/
  pubspec.yaml   analysis_options.yaml
  lib/
    main.dart                    app + theme + verdict colour
    api_client.dart              health / questions / model-info / predict
    models.dart                  FormFieldSpec, PredictResult
    screens/
      order_form_screen.dart     the input form (from /questions) + validation + submit
      result_screen.dart         the prediction view
```

## Screenshots for the report (IDs M1–M4)

1. **M1** — the order form filled in (the prefilled sample).
2. **M2** — the result for that order: *at risk*, low `P(satisfied)`, "days late" signal.
3. **M3** — edit the delivery date to before the promised date, clear the comment,
   predict again → *likely satisfied*.
4. **M4** — a snippet showing the request going to `API_URL` (e.g. the run console, or
   the API's Uvicorn access log with the `POST /predict` line) as evidence the app
   calls the REST service.
