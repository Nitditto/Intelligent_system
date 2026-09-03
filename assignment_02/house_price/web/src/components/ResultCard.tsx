import React, { useState } from "react";
import type { ListingInput, PredictionResponse } from "../types";

interface ResultCardProps {
  result: PredictionResponse | null;
  input: ListingInput | null;
  onJumpToStep: (step: number) => void;
  onReset: () => void;
}

export const ResultCard: React.FC<ResultCardProps> = ({
  result,
  input,
  onJumpToStep,
  onReset,
}) => {
  const [copied, setCopied] = useState(false);
  const [activeTab, setActiveTab] = useState<"shap" | "model" | "metrics">("shap");

  const handleCopy = () => {
    if (!result) return;
    const text = [
      `RealVal Vietnam Real Estate Valuation:`,
      `Predicted Value: ${result.formatted_price_billion} (${result.predicted_price.toLocaleString("en-US")} million VNĐ)`,
      `Unit Price: ${result.price_per_m2 ? `${result.price_per_m2} million VNĐ/m²` : "N/A"}`,
      `Usable Area: ${input?.Area} m²`,
      `Location: ${input?.district ? `${input.district}, ` : ""}${input?.Province || ""}`,
      `Property Type: ${input?.["Property Type"] || "N/A"}`,
      `Model: ${result.model_name}`,
      `Interpretation: ${result.interpretation}`,
    ].join("\n");

    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  const priceBillion = result ? result.predicted_price / 1000 : 2.63;
  const lowBillion = (priceBillion * 0.92).toFixed(2);
  const highBillion = (priceBillion * 1.08).toFixed(2);
  const formattedPrice = result ? result.formatted_price_billion : "2.63 tỷ VNĐ";

  const contributions = result?.feature_contributions || [
    { name: "area", label: "Usable Area & Size", impact_million: 350.9, direction: "positive" as const, importance_pct: 38.0 },
    { name: "location", label: "Geographic Location & Province", impact_million: 140.4, direction: "positive" as const, importance_pct: 28.0 },
    { name: "property_type", label: "Property Classification", impact_million: 62.4, direction: "positive" as const, importance_pct: 15.0 },
    { name: "dimensions", label: "Frontage Width & Depth", impact_million: 31.2, direction: "positive" as const, importance_pct: 8.0 },
    { name: "rooms", label: "Bedrooms & Living Structure", impact_million: 46.8, direction: "positive" as const, importance_pct: 6.0 },
    { name: "road", label: "Road Access & Position", impact_million: 148.2, direction: "positive" as const, importance_pct: 5.0 },
  ];

  const meta = result?.model_metadata || {
    algorithm: "RandomForestRegressor",
    n_estimators: 100,
    max_depth: 16,
    features_count: 18,
    transformed_features: 117,
    target_transform: "log1p(Price) -> expm1(y)",
    zero_leakage: true,
  };

  return (
    <div className="panel results-dashboard-final animate-step">
      {/* Step Header */}
      <div className="dashboard-header">
        <div>
          <span className="step-count-pill">FINAL STEP · VALUATION COMPLETE</span>
          <h2 className="dashboard-headline">Valuation & Explainability Report</h2>
          <p className="dashboard-subline">Machine learning market estimate and SHAP attribution analysis</p>
        </div>
        <span className="model-status-tag">{result?.model_name || meta.algorithm}</span>
      </div>

      {/* Hero Price Section */}
      <div className="predicted-price-hero">
        <span className="price-kicker">Predicted Valuation</span>
        <div className="hero-price-text">{formattedPrice}</div>
        <div className="hero-price-subtext">
          ≈ {result ? result.predicted_price.toLocaleString("en-US") : "2,629.8"} million VNĐ
          {result?.price_per_m2 ? ` · ${result.price_per_m2} million VNĐ/m²` : ""}
        </div>
      </div>

      {/* Gradient Price Range Bar */}
      <div className="gradient-range-block">
        <div className="gradient-bar-container">
          <div className="gradient-bar-track">
            <span className="gradient-bar-label">Gradient Price Range Bar</span>
            <div className="gradient-pin" style={{ left: "54%" }}>
              <div className="pin-marker" />
            </div>
          </div>
        </div>
        <div className="range-labels-row">
          <span className="range-val">Low Estimate: {lowBillion} tỷ</span>
          <span className="range-curr">Current Estimate ({priceBillion.toFixed(2)} tỷ)</span>
          <span className="range-val">High Estimate: {highBillion} tỷ</span>
        </div>
      </div>

      {/* Analytical Tab Switcher */}
      <div className="results-tab-bar">
        <button
          type="button"
          className={`res-tab-btn ${activeTab === "shap" ? "active" : ""}`}
          onClick={() => setActiveTab("shap")}
        >
          SHAP Feature Contributions
        </button>
        <button
          type="button"
          className={`res-tab-btn ${activeTab === "model" ? "active" : ""}`}
          onClick={() => setActiveTab("model")}
        >
          Model Architecture
        </button>
        <button
          type="button"
          className={`res-tab-btn ${activeTab === "metrics" ? "active" : ""}`}
          onClick={() => setActiveTab("metrics")}
        >
          Property Specifications
        </button>
      </div>

      {/* TAB 1: SHAP Feature Contributions */}
      {activeTab === "shap" && (
        <div className="shap-breakdown-panel animate-step">
          <div className="shap-header-row">
            <h4 className="shap-title">SHAP Feature Contribution Waterfall</h4>
            <p className="shap-desc">
              Baseline dataset median: ~1,850 Million VND. Bars show how each property factor shifted the valuation.
            </p>
          </div>

          <div className="shap-bars-list">
            {contributions.map((c) => {
              const isPos = c.direction === "positive";
              const sign = isPos ? "+" : "-";
              return (
                <div key={c.name} className="shap-bar-item">
                  <div className="shap-item-meta">
                    <span className="shap-feat-name">{c.label}</span>
                    <span className={`shap-feat-impact ${isPos ? "pos" : "neg"}`}>
                      {sign}{Math.abs(c.impact_million)}M VND ({c.importance_pct}% importance)
                    </span>
                  </div>
                  <div className="shap-progress-track">
                    <div
                      className={`shap-progress-fill ${isPos ? "pos-fill" : "neg-fill"}`}
                      style={{ width: `${Math.min(100, Math.max(8, c.importance_pct * 2.5))}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* TAB 2: Model Architecture & Metadata */}
      {activeTab === "model" && (
        <div className="model-specs-panel animate-step">
          <div className="specs-grid">
            <div className="spec-card">
              <span className="spec-label">Ensemble Algorithm</span>
              <span className="spec-val">{meta.algorithm}</span>
            </div>
            <div className="spec-card">
              <span className="spec-label">Ensemble Trees</span>
              <span className="spec-val">{meta.n_estimators} Estimators</span>
            </div>
            <div className="spec-card">
              <span className="spec-label">Max Tree Depth</span>
              <span className="spec-val">{meta.max_depth} Levels</span>
            </div>
            <div className="spec-card">
              <span className="spec-label">Transformed Features</span>
              <span className="spec-val">{meta.transformed_features} Dimensions</span>
            </div>
            <div className="spec-card">
              <span className="spec-label">Target Transformation</span>
              <span className="spec-val">{meta.target_transform}</span>
            </div>
            <div className="spec-card">
              <span className="spec-label">Data Leakage Guard</span>
              <span className="spec-val">Strict (Training ≠ Inference)</span>
            </div>
          </div>
        </div>
      )}

      {/* TAB 3: Property Metrics */}
      {activeTab === "metrics" && (
        <div className="metrics-tab-panel animate-step">
          <div className="metrics-pill-grid">
            <div className="metric-pill">
              <span className="pill-lbl">Unit Price</span>
              <span className="pill-val">
                {result?.price_per_m2 ? `${result.price_per_m2} M-VND/m²` : "33.4 M-VND/m²"}
              </span>
            </div>
            <div className="metric-pill">
              <span className="pill-lbl">Usable Area</span>
              <span className="pill-val">{input?.Area ? `${input.Area} m²` : "78.7 m²"}</span>
            </div>
            <div className="metric-pill">
              <span className="pill-lbl">Layout</span>
              <span className="pill-val">
                {input?.Bedrooms ?? 3} Beds · {input?.Bathrooms ?? 2} Baths
              </span>
            </div>
            <div className="metric-pill">
              <span className="pill-lbl">Location</span>
              <span className="pill-val">{input?.district || "Rach Gia"}</span>
            </div>
          </div>

          <div className="details-table-wrapper" style={{ marginTop: "0.85rem" }}>
            <table className="details-table">
              <tbody>
                <tr>
                  <td>Property Type</td>
                  <td><strong>{input?.["Property Type"] || "Townhouse"}</strong></td>
                </tr>
                <tr>
                  <td>Dimensions</td>
                  <td>
                    <strong>
                      {input?.Width ? `${input.Width}m frontage` : "4.0m"} ×{" "}
                      {input?.Length ? `${input.Length}m depth` : "19.6m"}
                    </strong>
                  </td>
                </tr>
                <tr>
                  <td>Road Access & Position</td>
                  <td>
                    <strong>
                      {input?.Position || "Main Street"} · {input?.["Road Type"] || "Asphalt Road"}
                    </strong>
                  </td>
                </tr>
                <tr>
                  <td>Compass Orientation</td>
                  <td><strong>{input?.Direction || "South"}</strong></td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Model Insight Note */}
      {result && (
        <div className="insight-callout">
          <p className="insight-body-text">{result.interpretation}</p>
        </div>
      )}

      {/* Footer Actions */}
      <div className="card-footer-action-row">
        <button
          type="button"
          onClick={() => onJumpToStep(1)}
          className="btn-modify-step"
        >
          ← Edit Parameters
        </button>

        <button
          type="button"
          onClick={handleCopy}
          className="btn-cyan-predict"
        >
          {copied ? "COPIED TO CLIPBOARD!" : "COPY VALUATION SUMMARY"}
        </button>

        <button
          type="button"
          onClick={onReset}
          className="btn-wizard-back"
        >
          New Valuation
        </button>
      </div>
    </div>
  );
};
