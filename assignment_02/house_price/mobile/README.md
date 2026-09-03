# House Price Mobile Application (Flutter)

Cross-platform mobile client for the **House-Price Prediction Service** (Assignment 02 · Application 2).
Built with **Flutter 3.x / Dart**, utilizing Material 3 design and Provider state management.

---

## 1. Architecture: Training ≠ Inference

```
Flutter Mobile Client
       ↓ (HTTP POST /predict)
FastAPI Backend Service (Port 8002)
       ↓
Loaded Preprocessing Pipeline (model_pipeline.joblib)
       ↓
Saved RandomForest Model (log1p prediction)
       ↓ (expm1 inversion)
Prediction Response (JSON)
       ↓
Flutter Mobile Presentation (Tỷ / Triệu VNĐ & Interpretation)
```

The mobile client contains no ML model weights or tensor operations; it strictly acts as a thin REST client.

---

## 2. API Host Addressing

By default, the app targets the FastAPI service on port **8002**:

| Target Environment | Base URL | Notes |
|---|---|---|
| **Android Emulator** | `http://10.0.2.2:8002` | Standard Android alias to host machine loopback |
| **iOS Simulator / macOS** | `http://localhost:8002` | Shares network stack directly with host |
| **Physical Phone (Wi-Fi)** | `http://<YOUR_LAN_IP>:8002` | Start FastAPI with `--host 0.0.0.0 --port 8002` |
| **Ngrok Tunnel** | `https://<subdomain>.ngrok-free.app` | For remote demonstrations |

*Tip: Tap the server status badge in the top-right corner of the app bar to switch the API Base URL at runtime.*

---

## 3. How to Run

### Prerequisites
- Flutter SDK (≥ 3.19.0)
- Android Studio / Xcode / Android Emulator

### Run Commands
```bash
# 1. Start the FastAPI backend first (from Assignment_2/):
uvicorn house_price.api.main:app --reload --host 0.0.0.0 --port 8002

# 2. In another terminal, run Flutter:
cd house_price/mobile
flutter pub get
flutter run
```

---

## 4. Deliverable Evidence Checklist (Report Appendix E)
- [ ] Mobile input screen filled screenshot
- [ ] Prediction result screen screenshot (showing formatted price, price/m², model badge, interpretation)
- [ ] Proof of API communication: Uvicorn access log showing `POST /predict 200 OK` from `10.0.2.2` or device IP
