import React from "react";
import type { HealthResponse } from "../types";

interface NavbarProps {
  health: HealthResponse | null;
  loading: boolean;
  onRefreshHealth: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({ health, loading, onRefreshHealth }) => {
  const isHealthy = health?.model_loaded === true;

  return (
    <header className="navbar">
      <div className="nav-container">
        <div className="nav-brand-group">
          <div className="brand-badge">REALVAL</div>
          <div className="brand-info">
            <h1 className="brand-title">Vietnam Real Estate Valuation</h1>
            <span className="brand-subtitle">AI-Powered Automated Valuation Model</span>
          </div>
        </div>

        <div className="nav-actions">
          <div className="nav-pill-tabs">
            <span className="nav-tab active">Predictor</span>
            <span className="nav-tab">Market Insights</span>
          </div>

          <button
            type="button"
            onClick={onRefreshHealth}
            className={`api-status-pill ${isHealthy ? "online" : "offline"}`}
            title="Click to refresh connection status"
            disabled={loading}
          >
            <span className="status-dot" />
            <span className="status-label">
              {loading
                ? "Connecting..."
                : isHealthy
                ? `API Active (${health?.model_name || "RandomForest"})`
                : "API Offline"}
            </span>
          </button>
        </div>
      </div>
    </header>
  );
};
