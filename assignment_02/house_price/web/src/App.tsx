import React, { useState, useEffect } from "react";
import { Navbar } from "./components/Navbar";
import { PredictionForm } from "./components/PredictionForm";
import { ResultCard } from "./components/ResultCard";
import { ErrorBanner } from "./components/ErrorBanner";
import { fetchHealth, predictPrice } from "./services/api";
import type { ListingInput, PredictionResponse, HealthResponse } from "./types";

export const App: React.FC = () => {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthLoading, setHealthLoading] = useState(false);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [lastInput, setLastInput] = useState<ListingInput | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");

  const checkServerHealth = async () => {
    setHealthLoading(true);
    try {
      const res = await fetchHealth();
      setHealth(res);
    } catch {
      setHealth(null);
    } finally {
      setHealthLoading(false);
    }
  };

  useEffect(() => {
    checkServerHealth();
  }, []);

  const handlePredict = async (formData: ListingInput) => {
    setSubmitting(true);
    setErrorMessage("");
    setLastInput(formData);

    try {
      const res = await predictPrice(formData);
      setPrediction(res);
    } catch (err: any) {
      if (err.response) {
        if (err.response.status === 422) {
          const detail = err.response.data?.detail;
          if (Array.isArray(detail)) {
            const msgs = detail.map((d: any) => `${d.loc?.join(".")}: ${d.msg}`).join("; ");
            setErrorMessage(`Validation error: ${msgs}`);
          } else {
            setErrorMessage(`Validation error: ${JSON.stringify(detail)}`);
          }
        } else if (err.response.status === 503) {
          setErrorMessage("Model pipeline not loaded on server. Please verify model_pipeline.joblib.");
        } else {
          setErrorMessage(`Server error (${err.response.status}): ${err.response.data?.detail || "Unknown error"}`);
        }
      } else if (err.request) {
        setErrorMessage("Cannot connect to API server at http://localhost:8002. Please ensure the backend is running.");
      } else {
        setErrorMessage(`Error: ${err.message}`);
      }
    } finally {
      setSubmitting(false);
    }
  };

  const handleReset = () => {
    setPrediction(null);
    setLastInput(null);
    setErrorMessage("");
  };

  return (
    <div className="app-layout">
      <Navbar
        health={health}
        loading={healthLoading}
        onRefreshHealth={checkServerHealth}
      />

      <main className="main-content">
        <div className="content-container">
          <ErrorBanner
            message={errorMessage}
            onDismiss={() => setErrorMessage("")}
          />

          <div className="dashboard-grid">
            <div className="grid-col-form">
              <PredictionForm
                onSubmit={handlePredict}
                loading={submitting}
                onReset={handleReset}
              />
            </div>

            <div className="grid-col-result">
              <ResultCard result={prediction} input={lastInput} />
            </div>
          </div>
        </div>
      </main>

      <footer className="footer">
        <div className="footer-container">
          <p>© 2026 Vietnam Real Estate Valuation. All estimates are for informational purposes.</p>
        </div>
      </footer>
    </div>
  );
};

export default App;
