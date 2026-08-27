import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import {
  X,
  Loader2,
  Target,
  Percent,
  Award,
  TrendingUp,
  TrendingDown,
  Activity,
  AlertCircle,
  FileText,
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Legend,
} from 'recharts';

interface SHAPFeature {
  feature: string;
  value: number;
}

interface DetailedModelResult {
  model: string;
  prediction: number;
  confidence: number;
  accuracy: number;
  shap_values: SHAPFeature[];
}

interface ModelDetailsPanelProps {
  isOpen: boolean;
  onClose: () => void;
  model: string | null;
  resultId: string | null;
  apiUrl: string;
  patientFeatures: Record<string, number> | null;
}

const formatModelName = (name: string) => {
  return name.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
};

const CustomSHAPTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload as SHAPFeature;
    const isPositive = data.value > 0;
    return (
      <div className="rounded-xl border border-[var(--app-border)] bg-[var(--app-surface-solid)] p-3.5 text-xs shadow-2xl min-w-[210px] space-y-1.5">
        <div className="flex items-center justify-between border-b border-[var(--app-border)] pb-1.5">
          <span className="font-semibold text-[var(--app-text)]">{data.feature}</span>
          <span
            className="rounded px-1.5 py-0.5 font-mono font-medium"
            style={{ background: isPositive ? 'rgba(99, 102, 241, 0.15)' : 'rgba(6, 182, 212, 0.15)', color: isPositive ? '#6366f1' : '#06b6d4' }}
          >
            {data.value > 0 ? `+${data.value.toFixed(4)}` : data.value.toFixed(4)}
          </span>
        </div>
        <p className="muted-text leading-relaxed">
          {isPositive ? (
            <span className="flex items-center gap-1 font-medium" style={{ color: '#6366f1' }}>
              <TrendingUp className="w-3.5 h-3.5" /> Pushes toward Diabetic
            </span>
          ) : (
            <span className="flex items-center gap-1 font-medium" style={{ color: '#06b6d4' }}>
              <TrendingDown className="w-3.5 h-3.5" /> Pushes toward Non-Diabetic
            </span>
          )}
        </p>
      </div>
    );
  }
  return null;
};

export const ModelDetailsPanel: React.FC<ModelDetailsPanelProps> = ({
  isOpen,
  onClose,
  model,
  resultId,
  apiUrl,
  patientFeatures,
}) => {
  const [data, setData] = useState<DetailedModelResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen && model && resultId) {
      setLoading(true);
      setError(null);
      axios.get<DetailedModelResult>(`${apiUrl}/api/result/${resultId}?model=${model}`)
        .then(res => {
          const sortedData = {
            ...res.data,
            shap_values: [...res.data.shap_values].sort((a, b) => Math.abs(b.value) - Math.abs(a.value)),
          };
          setData(sortedData);
        })
        .catch(err => {
          console.error(err);
          setError('Could not load the model details. Is the backend running?');
        })
        .finally(() => setLoading(false));
    }
  }, [isOpen, model, resultId, apiUrl]);

  // Normalize patient values to 0-100 range for the Radar Chart
  const normalize = (feature: string, val: number) => {
    const bounds: Record<string, { min: number; max: number }> = {
      Pregnancies: { min: 0, max: 15 },
      Glucose: { min: 40, max: 200 },
      BloodPressure: { min: 40, max: 122 },
      SkinThickness: { min: 0, max: 80 },
      Insulin: { min: 0, max: 400 },
      BMI: { min: 15, max: 60 },
      DiabetesPedigreeFunction: { min: 0.08, max: 2.0 },
      Age: { min: 21, max: 80 },
    };
    const b = bounds[feature] || { min: 0, max: 100 };
    const scaled = ((val - b.min) / (b.max - b.min)) * 100;
    return Math.max(0, Math.min(100, scaled));
  };

  const getRadarData = () => {
    if (!patientFeatures) return [];

    // Feature averages from the training dataset (Section 6 EDA, notebook-derived)
    const nonDiabeticAverages: Record<string, number> = {
      Pregnancies: 3.30, Glucose: 110.0, BloodPressure: 68.2, SkinThickness: 19.7,
      Insulin: 68.8, BMI: 30.3, DiabetesPedigreeFunction: 0.430, Age: 31.2,
    };
    const diabeticAverages: Record<string, number> = {
      Pregnancies: 4.87, Glucose: 141.3, BloodPressure: 70.8, SkinThickness: 22.2,
      Insulin: 100.3, BMI: 35.1, DiabetesPedigreeFunction: 0.551, Age: 37.1,
    };

    const rows: { label: string; key: string }[] = [
      { label: 'Glucose', key: 'Glucose' },
      { label: 'BMI', key: 'BMI' },
      { label: 'Age', key: 'Age' },
      { label: 'Blood Pressure', key: 'BloodPressure' },
      { label: 'Pregnancies', key: 'Pregnancies' },
      { label: 'Pedigree', key: 'DiabetesPedigreeFunction' },
    ];

    return rows.map(({ label, key }) => ({
      feature: label,
      Patient: Math.round(normalize(key, patientFeatures[key])),
      HealthyAvg: Math.round(normalize(key, nonDiabeticAverages[key])),
      DiabeticAvg: Math.round(normalize(key, diabeticAverages[key])),
    }));
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 z-40 bg-slate-950/60 backdrop-blur-sm"
          />

          <motion.div
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ type: 'spring', damping: 30, stiffness: 240 }}
            className="fixed right-0 top-0 bottom-0 z-50 flex w-full max-w-2xl flex-col border-l border-[var(--app-border)] bg-[var(--app-surface-solid)] shadow-2xl"
          >
            {/* Header */}
            <div className="sticky top-0 z-10 flex items-center justify-between gap-4 border-b border-[var(--app-border)] p-6 bg-[var(--app-surface-solid)]">
              <div className="flex min-w-0 items-center gap-3">
                <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-[var(--app-primary-soft)] text-[var(--app-primary)]">
                  <Activity className="w-5 h-5" />
                </div>
                <div className="min-w-0">
                  <div className="flex flex-wrap items-center gap-2">
                    <h2 className="truncate text-xl font-bold text-[var(--app-text)]">
                      {data ? formatModelName(data.model) : 'Model Analysis'}
                    </h2>
                    <span className="shrink-0 rounded-full border border-[var(--app-border)] bg-[var(--app-primary-soft)] px-2.5 py-0.5 text-[10px] font-bold uppercase tracking-wider text-[var(--app-primary-strong)]">
                      Explainable AI
                    </span>
                  </div>
                  <p className="mt-0.5 text-xs text-[var(--app-text-muted)]">Prediction breakdown &amp; population comparison</p>
                </div>
              </div>

              <button
                onClick={onClose}
                className="theme-toggle shrink-0 rounded-xl p-2.5 transition-colors cursor-pointer"
                aria-label="Close panel"
              >
                <X className="w-5 h-5 text-[var(--app-text-muted)]" />
              </button>
            </div>

            {/* Content Body */}
            <div className="flex-1 overflow-y-auto p-6 space-y-6">
              {loading ? (
                <div className="flex h-96 flex-col items-center justify-center gap-3 text-[var(--app-primary)]">
                  <Loader2 className="w-8 h-8 animate-spin" />
                  <span className="text-sm text-[var(--app-text-muted)] font-medium">Computing metric attributions...</span>
                </div>
              ) : error ? (
                <div className="flex h-96 flex-col items-center justify-center rounded-2xl border border-[var(--app-danger)] bg-[var(--app-danger-soft)] p-6 text-center">
                  <AlertCircle className="mb-2 h-10 w-10 text-[var(--app-danger)]" />
                  <p className="text-sm font-semibold text-[var(--app-danger)]">{error}</p>
                </div>
              ) : data ? (
                <>
                  {/* Stat cards */}
                  <div className="grid grid-cols-3 gap-4">
                    <div
                      className="rounded-2xl border p-4 shadow-sm flex flex-col justify-between"
                      style={{
                        background: data.prediction === 1 ? 'var(--app-danger-soft)' : 'var(--app-success-soft)',
                        borderColor: data.prediction === 1 ? 'var(--app-danger)' : 'var(--app-success)',
                      }}
                    >
                      <div className="flex items-center justify-between text-xs font-bold text-[var(--app-text-muted)] uppercase">
                        Prediction
                        <Target className="w-4 h-4" style={{ color: data.prediction === 1 ? 'var(--app-danger)' : 'var(--app-success)' }} />
                      </div>
                      <div className="mt-4">
                        <span className="text-lg font-black tracking-tight" style={{ color: data.prediction === 1 ? 'var(--app-danger)' : 'var(--app-success)' }}>
                          {data.prediction === 1 ? 'Diabetic' : 'Healthy'}
                        </span>
                        <p className="mt-1 text-[10px] text-[var(--app-text-muted)] font-medium">
                          {data.prediction === 1 ? 'Elevated glycemic index' : 'Glucose stable'}
                        </p>
                      </div>
                    </div>

                    <div className="rounded-2xl border border-[var(--app-border)] bg-[var(--app-primary-soft)] p-4 shadow-sm flex flex-col justify-between">
                      <div className="flex items-center justify-between text-xs font-bold text-[var(--app-text-muted)] uppercase">
                        Confidence
                        <Percent className="w-4 h-4 text-[var(--app-primary)]" />
                      </div>
                      <div className="mt-4">
                        <span className="text-lg font-black tracking-tight text-[var(--app-primary-strong)]">
                          {(data.confidence * 100).toFixed(1)}%
                        </span>
                        <p className="mt-1 text-[10px] text-[var(--app-text-muted)] font-medium">Class probability</p>
                      </div>
                    </div>

                    <div className="rounded-2xl border border-[var(--app-border)] bg-[var(--app-surface)] p-4 shadow-sm flex flex-col justify-between">
                      <div className="flex items-center justify-between text-xs font-bold text-[var(--app-text-muted)] uppercase">
                        Accuracy
                        <Award className="w-4 h-4 text-[var(--app-accent)]" />
                      </div>
                      <div className="mt-4">
                        <span className="text-lg font-black tracking-tight text-[var(--app-accent)]">
                          {(data.accuracy * 100).toFixed(1)}%
                        </span>
                        <p className="mt-1 text-[10px] text-[var(--app-text-muted)] font-medium">Test score</p>
                      </div>
                    </div>
                  </div>

                  {/* SHAP Chart */}
                  <div className="rounded-2xl border border-[var(--app-border)] bg-[var(--app-surface-solid)] p-5 shadow-md space-y-4">
                    <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[var(--app-border)] pb-3">
                      <div>
                        <h3 className="text-sm font-bold text-[var(--app-text)] flex items-center gap-1.5">
                          <FileText className="w-4 h-4 text-[var(--app-primary)]" />
                          SHAP Feature Attribution
                        </h3>
                        <p className="text-xs text-[var(--app-text-muted)]">Indicates feature influence on output log-odds</p>
                      </div>
                      <div className="flex items-center gap-3 text-[10px] font-bold uppercase tracking-wider">
                        <div className="flex items-center gap-1.5" style={{ color: '#6366f1' }}>
                          <span className="h-2.5 w-2.5 rounded-full" style={{ background: '#6366f1' }} />
                          Raises Risk
                        </div>
                        <div className="flex items-center gap-1.5" style={{ color: '#06b6d4' }}>
                          <span className="h-2.5 w-2.5 rounded-full" style={{ background: '#06b6d4' }} />
                          Lowers Risk
                        </div>
                      </div>
                    </div>

                    <div className="h-[280px] w-full">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={data.shap_values} layout="vertical" margin={{ top: 5, right: 25, left: 10, bottom: 5 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="var(--app-border)" horizontal={false} />
                          <XAxis type="number" stroke="var(--app-text-muted)" fontSize={10} tickLine={false} />
                          <YAxis dataKey="feature" type="category" width={110} stroke="var(--app-text-muted)" fontSize={10} tickLine={false} axisLine={false} />
                          <RechartsTooltip content={<CustomSHAPTooltip />} cursor={{ fill: 'var(--app-primary-soft)' }} />
                          <ReferenceLine x={0} stroke="var(--app-text-muted)" strokeWidth={1.5} />
                          <Bar dataKey="value" radius={[4, 4, 4, 4]}>
                            {data.shap_values.map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={entry.value > 0 ? '#6366f1' : '#06b6d4'} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  {/* Radar Chart comparing patient to population averages */}
                  {patientFeatures && (
                    <div className="rounded-2xl border border-[var(--app-border)] bg-[var(--app-surface-solid)] p-5 shadow-md space-y-4">
                      <div>
                        <h3 className="text-sm font-bold text-[var(--app-text)] flex items-center gap-1.5">
                          <Activity className="w-4 h-4 text-[var(--app-primary)]" />
                          Cohort Comparison (Normalized Metrics)
                        </h3>
                        <p className="text-xs text-[var(--app-text-muted)]">
                          Patient indices mapped from 0 to 100 relative to healthy and diabetic population means.
                        </p>
                      </div>

                      <div className="h-[300px] w-full flex items-center justify-center">
                        <ResponsiveContainer width="100%" height="100%">
                          <RadarChart cx="50%" cy="50%" outerRadius="75%" data={getRadarData()}>
                            <PolarGrid stroke="var(--app-border)" />
                            <PolarAngleAxis dataKey="feature" stroke="var(--app-text-muted)" fontSize={10} />
                            <PolarRadiusAxis angle={30} domain={[0, 100]} stroke="var(--app-border)" fontSize={9} />
                            <Radar name="This Patient" dataKey="Patient" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.25} />
                            <Radar name="Healthy Population Avg" dataKey="HealthyAvg" stroke="#06b6d4" fill="#06b6d4" fillOpacity={0.1} />
                            <Radar name="Diabetic Population Avg" dataKey="DiabeticAvg" stroke="#6366f1" fill="#6366f1" fillOpacity={0.1} />
                            <Legend wrapperStyle={{ fontSize: 11 }} />
                            <RechartsTooltip contentStyle={{ background: 'var(--app-surface-solid)', borderColor: 'var(--app-border)', borderRadius: '12px', fontSize: 11 }} />
                          </RadarChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  )}
                </>
              ) : null}
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
};
