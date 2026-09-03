import React, { useState } from "react";
import type { ListingInput, PredictionResponse } from "../types";

interface ResultCardProps {
  result: PredictionResponse | null;
  input: ListingInput | null;
}

export const ResultCard: React.FC<ResultCardProps> = ({ result, input }) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    if (!result) return;
    const text = [
      `Vietnam Real Estate Price Estimate:`,
      `Price: ${result.formatted_price_billion} (${result.predicted_price.toLocaleString("en-US")} million VNĐ)`,
      `Unit Price: ${result.price_per_m2 ? `${result.price_per_m2} million VNĐ/m²` : "N/A"}`,
      `Area: ${input?.Area} m²`,
      `Location: ${input?.district ? `${input.district}, ` : ""}${input?.Province || ""}`,
      `Property Type: ${input?.["Property Type"] || "N/A"}`,
      `Model: ${result.model_name}`,
    ].join("\n");

    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  if (!result) {
    return (
      <div className="card result-card empty-state">
        <div className="empty-content">
          <span className="badge-standby">Awaiting Input</span>
          <h3 className="empty-heading">No Valuation Calculated Yet</h3>
          <p className="empty-sub">
            Fill in the property details on the left, or select one of the quick presets above, then click <strong>Calculate Price Estimate</strong>.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="card result-card active-result">
      {/* Top Tag & Model Info */}
      <div className="result-top-bar">
        <span className="tag-valuation">Estimated Market Price</span>
        <span className="tag-model">{result.model_name}</span>
      </div>

      {/* Main Hero Number */}
      <div className="hero-price-container">
        <div className="hero-price">{result.formatted_price_billion}</div>
        <div className="hero-price-alt">
          ≈ {result.predicted_price.toLocaleString("en-US")} million VNĐ
        </div>
      </div>

      {/* Core Metrics Grid */}
      <div className="metric-tiles">
        <div className="metric-tile">
          <span className="tile-label">Unit Price</span>
          <span className="tile-value">
            {result.price_per_m2 ? `${result.price_per_m2.toLocaleString("en-US")} tr/m²` : "—"}
          </span>
        </div>
        <div className="metric-tile">
          <span className="tile-label">Usable Area</span>
          <span className="tile-value">{input?.Area ? `${input.Area} m²` : "—"}</span>
        </div>
        <div className="metric-tile">
          <span className="tile-label">Bedrooms / Baths</span>
          <span className="tile-value">
            {input?.Bedrooms ?? "—"} BR / {input?.Bathrooms ?? "—"} BA
          </span>
        </div>
        <div className="metric-tile">
          <span className="tile-label">Structure</span>
          <span className="tile-value">{input?.Floors ? `${input.Floors} Floors` : "—"}</span>
        </div>
      </div>

      {/* Property Parameter Summary */}
      <div className="details-box">
        <h4 className="box-title">Subject Property Summary</h4>
        <div className="details-list">
          <div className="detail-item">
            <span className="detail-k">Location:</span>
            <span className="detail-v">
              {input?.district ? `${input.district}, ` : ""}
              {input?.Province || "Surveyed Area"}
            </span>
          </div>
          <div className="detail-item">
            <span className="detail-k">Property Type:</span>
            <span className="detail-v">{input?.["Property Type"] || "Residential"}</span>
          </div>
          <div className="detail-item">
            <span className="detail-k">Dimensions:</span>
            <span className="detail-v">
              {input?.Width ? `${input.Width}m width` : "—"}
              {input?.Length ? ` × ${input.Length}m depth` : ""}
            </span>
          </div>
          <div className="detail-item">
            <span className="detail-k">Access:</span>
            <span className="detail-v">
              {input?.Position || "Main road"} · {input?.["Road Type"] || "Asphalt"}
            </span>
          </div>
        </div>
      </div>

      {/* Model Note */}
      <div className="insight-box">
        <div className="insight-title">Model Evaluation Note</div>
        <p className="insight-body">{result.interpretation}</p>
        <p className="insight-foot">
          * Calculated by pre-trained machine learning regression model. Estimates are for analytical guidance.
        </p>
      </div>

      {/* Copy Action */}
      <div className="action-row">
        <button
          type="button"
          onClick={handleCopy}
          className="btn-copy"
        >
          {copied ? "Copied to clipboard" : "Copy Estimate Summary"}
        </button>
      </div>
    </div>
  );
};
