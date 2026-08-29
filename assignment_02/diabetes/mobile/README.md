# Diabetes Screening — Mobile app (Flutter)

A REST client of the diabetes screening API. It runs **no model on the device** —
every screen calls the same endpoints the web app uses.

```
Mobile UI  ->  REST API  ->  preprocessing + model  ->  prediction  ->  Mobile UI
```

## Prerequisites

- Flutter SDK 3.19+ (`flutter --version`)
- The API running and reachable from the device/emulator
  (`uvicorn api.main:app --port 8000` from `diabetes/`)

## First-time setup

`lib/` and the config files are in the repo; the platform folders are not. Generate
them once:

```bash
cd mobile
flutter create .          # adds android/ ios/ etc. without touching lib/
flutter pub get
```

## Run

The API base URL is a compile-time constant. Defaults to `http://10.0.2.2:8000`
(the Android emulator's alias for the host's localhost).

```bash
# Android emulator, API on the host machine
flutter run

# real device / different host — pass the address the phone can reach
flutter run --dart-define=API_URL=http://192.168.1.20:8000

# release APK for submission
flutter build apk --dart-define=API_URL=http://<api-host>:8000
```

## Screens

| Screen | Content |
|---|---|
| **Questionnaire** | grouped questions rendered from `GET /questions`; Yes / No / **Not sure** for every item; height + weight → BMI; submit → `POST /predict?include=explain,similar,whatif,counterfactual`. |
| **Result** | risk % + Low/Moderate/High band card, completeness + uncertainty note, warnings, SHAP factor bars, "N of 5 similar respondents had diabetes" + profile list, the nearest-counterfactual sentence with the not-medical-advice caveat. |
| **What could change** | list of modifiable-factor risk deltas, plus a BMI slider that re-scores live against the API. |
| **History** | this launch's screenings from `GET /history`. |

## Screenshots for the report

1. Questionnaire — a partly filled form.
2. Result — risk band card + SHAP factor bars.
3. Result (scrolled) — similar cases + counterfactual sentence.
4. What could change — modifiable-factor list + BMI slider.

## Notes

- `lib/main.dart` holds the `ApiClient` instance, the per-launch `sessionId`, and the
  shared `bandColor()` helper.
- Network access on Android needs the `INTERNET` permission, which `flutter create`
  adds to `android/app/src/main/AndroidManifest.xml` by default. For a cleartext
  `http://` API on Android 9+, also set `android:usesCleartextTraffic="true"` on the
  `<application>` tag (fine for a localhost/LAN demo).
