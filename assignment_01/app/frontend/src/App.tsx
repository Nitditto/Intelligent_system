import { useEffect, useState } from 'react';
import { PredictionForm } from './components/PredictionForm';
import { ResultsList } from './components/ResultsList';
import { ModelDetailsPanel } from './components/ModelDetailsPanel';
import axios from 'axios';
import { Activity, BarChart3, Database, HeartPulse, Moon, ShieldCheck, Sun } from 'lucide-react';

export type ModelSummary = {
  model: string;
  accuracy: number;
};

export type PredictionResponse = {
  id: string;
  results: ModelSummary[];
};

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

function App() {
  const [loading, setLoading] = useState(false);
  const [resultId, setResultId] = useState<string | null>(null);
  const [summaries, setSummaries] = useState<ModelSummary[]>([]);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [isDarkMode, setIsDarkMode] = useState(() => {
    const dark = localStorage.getItem('theme') !== 'light';
    // Applied synchronously (before first paint) to avoid a light-theme flash
    // when the user's stored preference is dark.
    document.documentElement.classList.toggle('dark', dark);
    return dark;
  });

  useEffect(() => {
    document.documentElement.classList.toggle('dark', isDarkMode);
    localStorage.setItem('theme', isDarkMode ? 'dark' : 'light');
  }, [isDarkMode]);

  const handlePredict = async (data: Record<string, number>) => {
    setLoading(true);
    setResultId(null);
    setSummaries([]);
    setSelectedModel(null);
    try {
      const response = await axios.post<PredictionResponse>(`${API_URL}/api/predict`, data);
      setResultId(response.data.id);
      setSummaries(response.data.results);
    } catch (error) {
      console.error('Prediction failed:', error);
      alert('Prediction failed. Ensure the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  const topAccuracy = summaries.length > 0
    ? `${(Math.max(...summaries.map(summary => summary.accuracy)) * 100).toFixed(1)}%`
    : '--';

  return (
    <div className="app-shell min-h-screen relative overflow-hidden">
      <div className="mx-auto flex w-full max-w-7xl flex-col gap-6 px-4 py-5 sm:px-6 lg:px-8">
        <header className="top-bar relative z-10 flex items-center justify-between gap-4">
          <div className="flex min-w-0 items-center gap-3">
            <div className="brand-mark">
              <HeartPulse className="h-6 w-6 text-white" />
            </div>
            <div className="min-w-0">
              <p className="text-xs font-semibold uppercase tracking-wide text-[var(--app-primary)]">AI Clinical Lab</p>
              <h1 className="truncate text-xl font-bold text-[var(--app-text)] sm:text-2xl">
                Diabetes Risk Predictor
              </h1>
            </div>
          </div>
          <button
            type="button"
            onClick={() => setIsDarkMode(prev => !prev)}
            className="theme-toggle shrink-0 rounded-lg p-3 transition-transform hover:-translate-y-0.5 active:translate-y-0"
            aria-label={isDarkMode ? 'Switch to light mode' : 'Switch to dark mode'}
            title={isDarkMode ? 'Light mode' : 'Dark mode'}
          >
            {isDarkMode ? <Sun className="w-5 h-5 text-amber-400" /> : <Moon className="w-5 h-5 text-primary-600" />}
          </button>
        </header>

        <section className="hero-band relative z-10">
          <div className="hero-copy">
            <div className="status-pill">
              <span className="status-dot" />
              Backend endpoint: {API_URL.replace(/^https?:\/\//, '')}
            </div>
            <h2>Compare six trained models for one patient profile.</h2>
            <p>
              Enter the clinical indicators, run the prediction, then inspect accuracy,
              confidence, and SHAP feature impact for each model.
            </p>
          </div>

          <div className="metric-grid">
            <div className="metric-tile">
              <ShieldCheck className="h-5 w-5 text-[var(--app-success)]" />
              <span className="metric-value">6</span>
              <span className="metric-label">Models</span>
            </div>
            <div className="metric-tile">
              <Database className="h-5 w-5 text-[var(--app-primary)]" />
              <span className="metric-value">8</span>
              <span className="metric-label">Features</span>
            </div>
            <div className="metric-tile">
              <BarChart3 className="h-5 w-5 text-[var(--app-accent)]" />
              <span className="metric-value">{topAccuracy}</span>
              <span className="metric-label">Top accuracy</span>
            </div>
          </div>
        </section>

        <main className="relative z-10 grid w-full grid-cols-1 gap-5 lg:grid-cols-12 items-start">
          <section className="lg:col-span-5">
            <PredictionForm onSubmit={handlePredict} loading={loading} />
          </section>
          <section className="lg:col-span-7">
            {resultId && summaries.length > 0 ? (
              <div className="results-stage">
                <ResultsList
                  summaries={summaries}
                  onSelectModel={setSelectedModel}
                  selectedModel={selectedModel}
                />
              </div>
            ) : (
              <div className="empty-state glass-panel">
                <div className="empty-icon">
                  <Activity className="h-7 w-7" />
                </div>
                <div>
                  <h2>Ready for analysis</h2>
                  <p>
                    Submit the patient features to generate ranked model results and
                    open detailed SHAP explanations for each one.
                  </p>
                </div>
                <div className="empty-steps">
                  <span>1. Enter profile</span>
                  <span>2. Predict outcome</span>
                  <span>3. Review model insights</span>
                </div>
              </div>
            )}
          </section>
        </main>
      </div>

      <ModelDetailsPanel
        isOpen={!!selectedModel}
        onClose={() => setSelectedModel(null)}
        model={selectedModel}
        resultId={resultId}
        apiUrl={API_URL}
      />
    </div>
  );
}

export default App;
