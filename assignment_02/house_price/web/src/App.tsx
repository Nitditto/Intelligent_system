import React, { useState, useEffect } from "react";
import { Navbar } from "./components/Navbar";
import { PredictionForm } from "./components/PredictionForm";
import { ResultCard } from "./components/ResultCard";
import { ErrorBanner } from "./components/ErrorBanner";
import { fetchHealth, predictPrice } from "./services/api";
import type { ListingInput, PredictionResponse, HealthResponse } from "./types";

const DEFAULT_FORM_DATA: ListingInput = {
  Area: 78.7,
  Width: 4.0,
  Length: 19.6,
  Bedrooms: 3,
  Bathrooms: 2,
  Floors: 2,
  "Alley Width": 3.5,
  "Agent Listing Count": 1,
  "Property Type": "Nhà riêng",
  Position: "Đường chính",
  Direction: "Nam",
  "Road Type": "Đường nhựa",
  Province: "an-giang",
  "Agent Role": "Chính chủ",
  ward: "Phường An Hòa",
  district: "Rạch Giá",
};

const STEPS_NAV = [
  { num: 1, title: "Dimensions", desc: "Area & Lot Size" },
  { num: 2, title: "Living Space", desc: "Rooms & Floors" },
  { num: 3, title: "Location", desc: "City & Position" },
  { num: 4, title: "Attributes", desc: "Type & Access" },
  { num: 5, title: "Valuation & SHAP", desc: "Model Prediction" },
];

export const App: React.FC = () => {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthLoading, setHealthLoading] = useState(false);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [formData, setFormData] = useState<ListingInput>(DEFAULT_FORM_DATA);
  const [submitting, setSubmitting] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const [currentStep, setCurrentStep] = useState(1);

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

  const handlePredict = async (dataToPredict: ListingInput) => {
    setSubmitting(true);
    setErrorMessage("");

    try {
      const res = await predictPrice(dataToPredict);
      setPrediction(res);
      setCurrentStep(5);
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
    setFormData(DEFAULT_FORM_DATA);
    setErrorMessage("");
    setCurrentStep(1);
  };

  const handleStepClick = (stepNum: number) => {
    if (stepNum === 5 && !prediction) {
      handlePredict(formData);
      return;
    }
    setCurrentStep(stepNum);
  };

  return (
    <div className="app-layout">
      <Navbar
        health={health}
        loading={healthLoading}
        onRefreshHealth={checkServerHealth}
      />

      <main className="main-content">
        <div className="content-container-wizard">
          <ErrorBanner
            message={errorMessage}
            onDismiss={() => setErrorMessage("")}
          />

          {/* Stepper Progress Navigation Bar */}
          <div className="wizard-stepper-bar">
            {STEPS_NAV.map((s) => {
              const isCurrent = currentStep === s.num;
              const isCompleted = (prediction !== null && s.num === 5) || currentStep > s.num;
              return (
                <button
                  key={s.num}
                  type="button"
                  className={`wizard-step-tab ${isCurrent ? "active" : ""} ${
                    isCompleted ? "completed" : ""
                  }`}
                  onClick={() => handleStepClick(s.num)}
                >
                  <span className="step-badge">
                    {s.num === 5 && prediction ? "✓" : currentStep > s.num ? "✓" : `0${s.num}`}
                  </span>
                  <div className="step-tab-text">
                    <span className="tab-title">{s.title}</span>
                    <span className="tab-desc">{s.desc}</span>
                  </div>
                </button>
              );
            })}
          </div>

          {/* Step Views: Steps 1-4 = Input Wizard, Step 5 = Result Dashboard at the Last Place */}
          {currentStep <= 4 ? (
            <div className="wizard-stage-container animate-step">
              <PredictionForm
                formData={formData}
                onFormDataChange={setFormData}
                onSubmit={handlePredict}
                loading={submitting}
                onReset={handleReset}
                currentStep={currentStep}
                onStepChange={setCurrentStep}
              />
            </div>
          ) : (
            <div className="wizard-stage-container animate-step">
              <ResultCard
                result={prediction}
                input={formData}
                onJumpToStep={setCurrentStep}
                onReset={handleReset}
              />
            </div>
          )}
        </div>
      </main>

      <footer className="footer">
        <div className="footer-container">
          <p>© 2026 REALVAL · Vietnam Real Estate Valuation Engine. Powered by Random Forest Regressor.</p>
        </div>
      </footer>
    </div>
  );
};

export default App;
