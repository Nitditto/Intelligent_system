import { useEffect, useState } from 'react';
import { PredictionForm } from './components/PredictionForm';
import { ResultsList } from './components/ResultsList';
import { ModelDetailsPanel } from './components/ModelDetailsPanel';
import { ModelDocumentation } from './components/ModelDocumentation';
import { DatasetInsights } from './components/DatasetInsights';
import { AuditHistory } from './components/AuditHistory';
import type { HistoryRecord } from './components/AuditHistory';
import { ClinicalReport } from './components/ClinicalReport';
import axios from 'axios';
import { 
  Activity, 
  BarChart3, 
  Database, 
  HeartPulse, 
  Moon, 
  ShieldCheck, 
  Sun, 
  BookOpen, 
  LayoutGrid, 
  Download, 
  ChevronRight,
  History,
  FileText
} from 'lucide-react';
import clsx from 'clsx';

export type ModelSummary = {
  model: string;
  accuracy: number;
};

export type PredictionResponse = {
  id: string;
  results: ModelSummary[];
};

export type DetailedResult = {
  model: string;
  prediction: number;
  confidence: number;
  accuracy: number;
  shap_values: { feature: string; value: number }[];
};

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const formatModelName = (name: string) => {
  return name.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
};

function App() {
  const [loading, setLoading] = useState(false);
  const [resultId, setResultId] = useState<string | null>(null);
  const [summaries, setSummaries] = useState<ModelSummary[]>([]);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [submittedFeatures, setSubmittedFeatures] = useState<Record<string, number> | null>(null);
  const [detailedResults, setDetailedResults] = useState<Record<string, DetailedResult>>({});
  const [restoreValues, setRestoreValues] = useState<Record<string, number> | null>(null);
  
  // Navigation State
  const [activeTab, setActiveTab] = useState<'dashboard' | 'insights' | 'documentation' | 'history' | 'report'>('dashboard');

  // History State
  const [history, setHistory] = useState<HistoryRecord[]>(() => {
    const stored = localStorage.getItem('screening_history');
    return stored ? JSON.parse(stored) : [];
  });

  // activeRecordId state (for clinical report viewer)
  const [activeRecordId, setActiveRecordId] = useState<string | null>(null);

  useEffect(() => {
    localStorage.setItem('screening_history', JSON.stringify(history));
  }, [history]);

  const [isDarkMode, setIsDarkMode] = useState(() => {
    const dark = localStorage.getItem('theme') !== 'light';
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
    setDetailedResults({});
    setSubmittedFeatures(data);
    try {
      const response = await axios.post<PredictionResponse>(`${API_URL}/api/predict`, data);
      setResultId(response.data.id);
      setSummaries(response.data.results);
      
      // Fetch details for each model for the comparison matrix
      const detailsPromises = response.data.results.map(summary =>
        axios.get<DetailedResult>(`${API_URL}/api/result/${response.data.id}?model=${summary.model}`)
      );
      const detailsResponses = await Promise.all(detailsPromises);
      const newDetailsMap: Record<string, DetailedResult> = {};
      detailsResponses.forEach(res => {
        newDetailsMap[res.data.model] = res.data;
      });
      setDetailedResults(newDetailsMap);

      // Compute consensus statistics
      let diabeticCount = 0;
      detailsResponses.forEach(res => {
        if (res.data.prediction === 1) diabeticCount++;
      });
      const healthyCount = detailsResponses.length - diabeticCount;
      const isDiabeticVerdict = diabeticCount >= healthyCount;
      const percentage = isDiabeticVerdict ? (diabeticCount / detailsResponses.length) * 100 : (healthyCount / detailsResponses.length) * 100;
      const verdict = isDiabeticVerdict ? 'Diabetic Risk' : 'Healthy';

      // Save to Audit History
      const newRecord: HistoryRecord = {
        id: response.data.id,
        timestamp: new Date().toISOString(),
        patientFeatures: data,
        consensus: {
          diabeticCount,
          healthyCount,
          percentage,
          verdict
        }
      };
      setHistory(prev => [newRecord, ...prev]);
      setActiveRecordId(response.data.id);
    } catch (error) {
      console.error('Prediction failed:', error);
      alert('Prediction failed. Ensure the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  const handleRestoreProfile = (features: Record<string, number>) => {
    setRestoreValues(features);
    setActiveTab('dashboard');
  };

  const handleViewReport = async (recordId: string) => {
    setActiveRecordId(recordId);
    const record = history.find(h => h.id === recordId);
    if (record) {
      setSubmittedFeatures(record.patientFeatures);
      setResultId(record.id);
      
      const models = ['random_forest', 'xgboost', 'knn', 'logistic_regression', 'svm_linear', 'svm_rbf'];
      try {
        const detailsPromises = models.map(modelName =>
          axios.get<DetailedResult>(`${API_URL}/api/result/${recordId}?model=${modelName}`)
        );
        const detailsResponses = await Promise.all(detailsPromises);
        const newDetailsMap: Record<string, DetailedResult> = {};
        detailsResponses.forEach(res => {
          newDetailsMap[res.data.model] = res.data;
        });
        setDetailedResults(newDetailsMap);
        
        // Also rebuild summaries
        const newSummaries = models.map(modelName => ({
          model: modelName,
          accuracy: newDetailsMap[modelName].accuracy
        })).sort((a, b) => b.accuracy - a.accuracy);
        setSummaries(newSummaries);
        
        setActiveTab('report');
      } catch (err) {
        console.error('Failed to load past result:', err);
        alert('Failed to load report from server. The backend server might have restarted, clearing its in-memory database.');
      }
    }
  };

  const handleExportReport = () => {
    if (!resultId || !submittedFeatures || Object.keys(detailedResults).length === 0) return;
    
    let text = `==================================================\n`;
    text += `       DIABETES RISK CLINICAL ASSESSMENT REPORT      \n`;
    text += `==================================================\n`;
    text += `Assessment ID: ${resultId}\n`;
    text += `Date: ${new Date().toLocaleString()}\n\n`;
    
    text += `--- PATIENT CLINICAL DATA ---\n`;
    Object.entries(submittedFeatures).forEach(([key, val]) => {
      text += `${key.replace(/([A-Z])/g, ' $1').trim()}: ${val}\n`;
    });
    text += `\n`;
    
    text += `--- MODEL PREDICTIONS COMPARISON ---\n`;
    text += `Rank | Model | Prediction | Confidence | Test Accuracy\n`;
    text += `--------------------------------------------------\n`;
    summaries.forEach((summary, idx) => {
      const detail = detailedResults[summary.model];
      const predText = detail ? (detail.prediction === 1 ? 'Diabetic' : 'Healthy') : 'N/A';
      const confText = detail ? `${(detail.confidence * 100).toFixed(1)}%` : 'N/A';
      text += `#${idx + 1} | ${formatModelName(summary.model)} | ${predText} | ${confText} | ${(summary.accuracy * 100).toFixed(2)}%\n`;
    });
    text += `\n`;
    
    text += `--- FINAL VERDICT (Selected Model: Random Forest) ---\n`;
    const rf = detailedResults['random_forest'];
    if (rf) {
      text += `Decision: ${rf.prediction === 1 ? 'DIABETIC RISK SIGNAL DETECTED' : 'HEALTHY PROFILE'}\n`;
      text += `Confidence Level: ${(rf.confidence * 100).toFixed(1)}%\n\n`;
      text += `Top Driver Metrics (SHAP): \n`;
      const sortedShap = [...rf.shap_values].sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
      sortedShap.slice(0, 3).forEach(s => {
        text += `- ${s.feature}: ${s.value > 0 ? '+' : ''}${s.value.toFixed(4)} (${s.value > 0 ? 'Elevates risk' : 'Suppresses risk'})\n`;
      });
    } else {
      text += `Random Forest model details unavailable.\n`;
    }
    
    text += `\n==================================================\n`;
    text += `Disclaimer: This screening report is generated by traditional machine learning algorithms and serves as a preliminary diagnostic assistant. It should not be used as a replacement for standard clinical tests (HbA1c / Oral Glucose tolerance tests).\n`;
    text += `==================================================\n`;
    
    const element = document.createElement("a");
    const file = new Blob([text], {type: 'text/plain'});
    element.href = URL.createObjectURL(file);
    element.download = `diabetes_screening_report_${resultId.slice(0, 8)}.txt`;
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  const getConsensus = () => {
    const total = Object.keys(detailedResults).length;
    if (total === 0) return { diabeticCount: 0, healthyCount: 0, percentage: 0, verdict: 'N/A' };
    
    let diabeticCount = 0;
    Object.values(detailedResults).forEach(r => {
      if (r.prediction === 1) diabeticCount++;
    });
    
    const healthyCount = total - diabeticCount;
    const isDiabeticVerdict = diabeticCount >= healthyCount;
    const percentage = isDiabeticVerdict ? (diabeticCount / total) * 100 : (healthyCount / total) * 100;
    
    return {
      diabeticCount,
      healthyCount,
      percentage,
      verdict: isDiabeticVerdict ? 'Diabetic Risk' : 'Healthy'
    };
  };

  const consensus = getConsensus();
  const topAccuracy = summaries.length > 0
    ? `${(Math.max(...summaries.map(summary => summary.accuracy)) * 100).toFixed(1)}%`
    : '--';

  const activeRecord = history.find(h => h.id === activeRecordId) || null;

  return (
    <div className="app-shell min-h-screen relative overflow-hidden pb-12">
      <div className="mx-auto flex w-full max-w-7xl flex-col gap-6 px-4 py-5 sm:px-6 lg:px-8">
        
        {/* Header Tabs Navigation */}
        <header className="top-bar relative z-10 flex flex-col md:flex-row md:items-center justify-between gap-4 shadow-lg no-print">
          <div className="flex min-w-0 items-center gap-3">
            <div className="brand-mark shrink-0">
              <HeartPulse className="h-6 w-6 text-white" />
            </div>
            <div className="min-w-0">
              <p className="text-xs font-bold uppercase tracking-wider text-[var(--app-primary)]">AI Clinical Lab</p>
              <h1 className="truncate text-lg font-black tracking-tight text-[var(--app-text)] sm:text-xl">
                Diabetes Risk Diagnostic Dashboard
              </h1>
            </div>
          </div>

          {/* Navigation Bar */}
          <div className="flex bg-[var(--app-bg)] rounded-xl p-1 border border-[var(--app-border)] text-xs font-bold uppercase tracking-wider gap-1 self-center md:self-auto overflow-x-auto max-w-full">
            <button
              onClick={() => setActiveTab('dashboard')}
              className={clsx(
                'px-3.5 py-2 rounded-lg transition-all flex items-center gap-1.5 cursor-pointer whitespace-nowrap',
                activeTab === 'dashboard' ? 'bg-[var(--app-surface-solid)] text-[var(--app-primary)] shadow-sm' : 'text-[var(--app-text-muted)] hover:text-[var(--app-text)]'
              )}
            >
              <LayoutGrid className="w-4 h-4" />
              Predictor
            </button>
            <button
              onClick={() => setActiveTab('history')}
              className={clsx(
                'px-3.5 py-2 rounded-lg transition-all flex items-center gap-1.5 cursor-pointer whitespace-nowrap',
                activeTab === 'history' ? 'bg-[var(--app-surface-solid)] text-[var(--app-primary)] shadow-sm' : 'text-[var(--app-text-muted)] hover:text-[var(--app-text)]'
              )}
            >
              <History className="w-4 h-4" />
              History ({history.length})
            </button>
            <button
              onClick={() => setActiveTab('report')}
              className={clsx(
                'px-3.5 py-2 rounded-lg transition-all flex items-center gap-1.5 cursor-pointer whitespace-nowrap',
                activeTab === 'report' ? 'bg-[var(--app-surface-solid)] text-[var(--app-primary)] shadow-sm' : 'text-[var(--app-text-muted)] hover:text-[var(--app-text)]'
              )}
            >
              <FileText className="w-4 h-4" />
              Report
            </button>
            <button
              onClick={() => setActiveTab('insights')}
              className={clsx(
                'px-3.5 py-2 rounded-lg transition-all flex items-center gap-1.5 cursor-pointer whitespace-nowrap',
                activeTab === 'insights' ? 'bg-[var(--app-surface-solid)] text-[var(--app-primary)] shadow-sm' : 'text-[var(--app-text-muted)] hover:text-[var(--app-text)]'
              )}
            >
              <Database className="w-4 h-4" />
              Insights
            </button>
            <button
              onClick={() => setActiveTab('documentation')}
              className={clsx(
                'px-3.5 py-2 rounded-lg transition-all flex items-center gap-1.5 cursor-pointer whitespace-nowrap',
                activeTab === 'documentation' ? 'bg-[var(--app-surface-solid)] text-[var(--app-primary)] shadow-sm' : 'text-[var(--app-text-muted)] hover:text-[var(--app-text)]'
              )}
            >
              <BookOpen className="w-4 h-4" />
              Models
            </button>
          </div>

          <button
            type="button"
            onClick={() => setIsDarkMode(prev => !prev)}
            className="theme-toggle rounded-xl p-2.5 transition-transform hover:-translate-y-0.5 active:translate-y-0 cursor-pointer self-end md:self-auto"
            aria-label={isDarkMode ? 'Switch to light mode' : 'Switch to dark mode'}
            title={isDarkMode ? 'Light mode' : 'Dark mode'}
          >
            {isDarkMode ? <Sun className="w-4 h-4 text-amber-400" /> : <Moon className="w-4 h-4 text-indigo-600" />}
          </button>
        </header>

        {/* Dynamic Pages */}
        {activeTab === 'insights' && <DatasetInsights />}
        {activeTab === 'documentation' && <ModelDocumentation />}
        {activeTab === 'history' && (
          <AuditHistory 
            history={history} 
            onRestore={handleRestoreProfile} 
            onViewReport={handleViewReport} 
          />
        )}
        {activeTab === 'report' && (
          <ClinicalReport 
            activeRecord={activeRecord} 
            detailedResults={detailedResults} 
            onExportTxt={handleExportReport} 
          />
        )}
        
        {activeTab === 'dashboard' && (
          <>
            {/* Hero Banner */}
            <section className="hero-band relative z-10 shadow-md no-print">
              <div className="hero-copy">
                <div className="status-pill">
                  <span className="status-dot" />
                  Service endpoint: {API_URL.replace(/^https?:\/\//, '')}
                </div>
                <h2>Predict patient risk across six candidate models.</h2>
                <p>
                  Enter medical markers in the live form to run parallel inferences. 
                  Inspect diagnostics, consensus metrics, and detailed SHAP attributions for each classifier.
                </p>
              </div>

              <div className="metric-grid">
                <div className="metric-tile shadow-sm">
                  <ShieldCheck className="h-5 w-5 text-[var(--app-success)]" />
                  <span className="metric-value">6</span>
                  <span className="metric-label">Classifiers</span>
                </div>
                <div className="metric-tile shadow-sm">
                  <Database className="h-5 w-5 text-[var(--app-primary)]" />
                  <span className="metric-value">8</span>
                  <span className="metric-label">Indicators</span>
                </div>
                <div className="metric-tile shadow-sm">
                  <BarChart3 className="h-5 w-5 text-[var(--app-accent)]" />
                  <span className="metric-value">{topAccuracy}</span>
                  <span className="metric-label">Top accuracy</span>
                </div>
              </div>
            </section>

            {/* Content Body */}
            <main className="relative z-10 grid w-full grid-cols-1 gap-6 lg:grid-cols-12 items-start no-print">
              <section className="lg:col-span-5">
                <PredictionForm 
                  onSubmit={handlePredict} 
                  loading={loading} 
                  restoreValues={restoreValues} 
                />
              </section>
              
              <section className="lg:col-span-7 space-y-6">
                {resultId && summaries.length > 0 ? (
                  <div className="space-y-6">
                    {/* Olympic Podium Ranked Leaderboard */}
                    <div className="results-stage shadow-lg">
                      <ResultsList
                        summaries={summaries}
                        onSelectModel={setSelectedModel}
                        selectedModel={selectedModel}
                      />
                    </div>

                    {/* Decision matrix table */}
                    <div className="glass-panel p-5 space-y-4 shadow-lg">
                      <div className="flex items-center justify-between border-b border-[var(--app-border)] pb-3">
                        <div>
                          <h3 className="text-sm font-bold text-[var(--app-text)] flex items-center gap-1.5">
                            <Activity className="w-4 h-4 text-[var(--app-primary)]" />
                            Consensus &amp; Decision Matrix
                          </h3>
                          <p className="text-xs text-[var(--app-text-muted)]">Detailed predictions comparison</p>
                        </div>
                        <div className="flex items-center gap-2">
                          <button
                            onClick={() => setActiveTab('report')}
                            className="inline-flex items-center gap-1 px-3 py-1.5 rounded-lg border border-transparent bg-[var(--app-primary)] text-white hover:bg-[var(--app-primary-strong)] text-xs font-bold transition-colors cursor-pointer"
                          >
                            <FileText className="w-3.5 h-3.5" />
                            Open Report Page
                          </button>
                          <button
                            onClick={handleExportReport}
                            className="export-button px-3 py-1.5 rounded-lg text-xs font-bold flex items-center gap-1.5 transition-colors cursor-pointer"
                          >
                            <Download className="w-3.5 h-3.5" />
                            Export TXT
                          </button>
                        </div>
                      </div>

                      {/* Consensus bar */}
                      {consensus.verdict !== 'N/A' && (
                        <div className="p-3 bg-[var(--app-bg)] rounded-xl border border-[var(--app-border)] space-y-2">
                          <div className="flex justify-between text-xs font-bold">
                            <span className="text-[var(--app-text-muted)]">Consensus Verdict:</span>
                            <span className={clsx(
                              consensus.verdict === 'Healthy' ? 'text-emerald-500' : 'text-rose-500'
                            )}>
                              {consensus.percentage.toFixed(0)}% {consensus.verdict} ({consensus.verdict === 'Healthy' ? consensus.healthyCount : consensus.diabeticCount}/6 models)
                            </span>
                          </div>
                          <div className="consensus-strip">
                            <span 
                              style={{ width: `${(consensus.healthyCount / 6) * 100}%` }} 
                              className="bg-emerald-500" 
                            />
                            <span 
                              style={{ width: `${(consensus.diabeticCount / 6) * 100}%` }} 
                              className="bg-rose-500" 
                            />
                          </div>
                        </div>
                      )}

                      {/* Comparison table */}
                      <div className="overflow-x-auto">
                        <table className="matrix-table text-xs">
                          <thead>
                            <tr>
                              <th>Model</th>
                              <th>Accuracy</th>
                              <th>Prediction</th>
                              <th>Confidence</th>
                              <th className="text-right">Action</th>
                            </tr>
                          </thead>
                          <tbody>
                            {summaries.map((summary) => {
                              const detail = detailedResults[summary.model];
                              return (
                                <tr key={summary.model} className="hover:bg-[var(--app-primary-soft)] transition-colors">
                                  <td className="font-semibold text-[var(--app-text)]">{formatModelName(summary.model)}</td>
                                  <td className="text-[var(--app-text-muted)] font-mono">{(summary.accuracy * 100).toFixed(1)}%</td>
                                  <td>
                                    {detail ? (
                                      <span className={clsx(
                                        'matrix-verdict-pill text-[10px]',
                                        detail.prediction === 1 ? 'matrix-verdict-positive' : 'matrix-verdict-negative'
                                      )}>
                                        {detail.prediction === 1 ? 'Diabetic' : 'Healthy'}
                                      </span>
                                    ) : (
                                      <span className="text-[var(--app-text-muted)]">Loading...</span>
                                    )}
                                  </td>
                                  <td className="font-mono text-[var(--app-text-muted)]">
                                    {detail ? `${(detail.confidence * 100).toFixed(1)}%` : '--'}
                                  </td>
                                  <td className="text-right">
                                    <button 
                                      onClick={() => setSelectedModel(summary.model)}
                                      className="text-[var(--app-primary)] hover:text-[var(--app-primary-strong)] font-bold inline-flex items-center gap-0.5 cursor-pointer"
                                    >
                                      Explain
                                      <ChevronRight className="w-3.5 h-3.5" />
                                    </button>
                                  </td>
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="empty-state glass-panel shadow-lg">
                    <div className="empty-icon">
                      <Activity className="h-7 w-7" />
                    </div>
                    <div>
                      <h2>Ready for Clinical Inference</h2>
                      <p>
                        Submit patient indicators to generate parallel predictions, view model consensus levels, 
                        and open explainable SHAP dashboards.
                      </p>
                    </div>
                    <div className="empty-steps">
                      <span>1. Load Patient Profile</span>
                      <span>2. Process Inference</span>
                      <span>3. Audit Attributions</span>
                    </div>
                  </div>
                )}
              </section>
            </main>
          </>
        )}
      </div>

      <ModelDetailsPanel
        isOpen={!!selectedModel}
        onClose={() => setSelectedModel(null)}
        model={selectedModel}
        resultId={resultId}
        apiUrl={API_URL}
        patientFeatures={submittedFeatures}
      />
    </div>
  );
}

export default App;
