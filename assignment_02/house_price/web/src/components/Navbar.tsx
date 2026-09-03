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
        <div className="nav-brand">
          <span className="brand-title">Vietnam House Price Estimator</span>
        </div>

        <div className="nav-actions">
          <button
            type="button"
            onClick={onRefreshHealth}
            className={`status-pill ${isHealthy ? "status-online" : "status-offline"}`}
            title="Click to check API status"
            disabled={loading}
          >
            <span className="status-dot" />
            <span>
              {loading
                ? "Connecting..."
                : isHealthy
                ? "API Connected"
                : "API Disconnected"}
            </span>
          </button>
        </div>
      </div>
    </header>
  );
};
