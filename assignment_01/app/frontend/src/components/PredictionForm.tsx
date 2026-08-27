import React, { useState, useEffect } from 'react';
import { Loader2, SendHorizontal, Info, BookOpen, ChevronDown, ChevronUp } from 'lucide-react';

interface PredictionFormProps {
  onSubmit: (data: Record<string, number>) => void;
  loading: boolean;
  restoreValues: Record<string, number> | null;
}

const defaultValues = {
  Pregnancies: 6,
  Glucose: 148,
  BloodPressure: 72,
  SkinThickness: 35,
  Insulin: 0,
  BMI: 33.6,
  DiabetesPedigreeFunction: 0.627,
  Age: 50,
};

const presets = {
  low: {
    label: 'Low-Risk Patient',
    values: {
      Pregnancies: 1,
      Glucose: 85,
      BloodPressure: 66,
      SkinThickness: 29,
      Insulin: 0,
      BMI: 22.4,
      DiabetesPedigreeFunction: 0.351,
      Age: 22,
    }
  },
  borderline: {
    label: 'Borderline Patient',
    values: {
      Pregnancies: 3,
      Glucose: 130,
      BloodPressure: 78,
      SkinThickness: 23,
      Insulin: 95,
      BMI: 28.5,
      DiabetesPedigreeFunction: 0.523,
      Age: 34,
    }
  },
  high: {
    label: 'High-Risk Patient',
    values: {
      Pregnancies: 8,
      Glucose: 183,
      BloodPressure: 90,
      SkinThickness: 0,
      Insulin: 0,
      BMI: 39.5,
      DiabetesPedigreeFunction: 1.288,
      Age: 51,
    }
  }
};

const featureGuidelines: Record<string, { desc: string; normal: string; watch: string; high: string; tips: string }> = {
  Pregnancies: {
    desc: 'Number of times pregnant. Highly correlated with gestational diabetes risks.',
    normal: '0-2 times',
    watch: '3-5 times',
    high: '6+ times',
    tips: 'Monitor blood sugar levels closely during and post-pregnancy.'
  },
  Glucose: {
    desc: 'Plasma glucose concentration 2 hours in an oral glucose tolerance test.',
    normal: '< 100 mg/dL',
    watch: '100 - 125 mg/dL (Prediabetes)',
    high: '>= 126 mg/dL (Diabetic level)',
    tips: 'Reduce simple carbohydrates, increase dietary fiber, and exercise regularly.'
  },
  BloodPressure: {
    desc: 'Diastolic blood pressure (mm Hg). High BP increases vascular complications.',
    normal: '< 80 mmHg',
    watch: '80 - 89 mmHg (Elevated)',
    high: '>= 90 mmHg (Hypertension)',
    tips: 'Lower sodium intake, manage stress levels, and perform aerobic exercise.'
  },
  SkinThickness: {
    desc: 'Triceps skin fold thickness (mm) to estimate subcutaneous body fat.',
    normal: '10 - 25 mm',
    watch: '26 - 32 mm',
    high: '>= 33 mm (Elevated body fat)',
    tips: 'Integrate full-body strength training to alter muscle-to-fat composition.'
  },
  Insulin: {
    desc: '2-Hour serum insulin (mu U/ml). Zero values represent missing readings in the dataset.',
    normal: '15 - 120 mu U/ml',
    watch: '121 - 160 mu U/ml',
    high: '> 160 mu U/ml (Insulin Resistance)',
    tips: 'Consult a physician. Soluble fiber can help stabilize insulin curves.'
  },
  BMI: {
    desc: 'Body Mass Index (weight in kg / height in m²). Indicator of obesity.',
    normal: '< 25.0 (Healthy weight)',
    watch: '25.0 - 29.9 (Overweight)',
    high: '>= 30.0 (Obese range)',
    tips: 'Focus on calorie-deficit diet, active steps (10k/day), and weight management.'
  },
  DiabetesPedigreeFunction: {
    desc: 'Genetic predisposition score based on family history. Higher values mean higher hereditary risk.',
    normal: '< 0.40',
    watch: '0.40 - 0.75 (Moderate family risk)',
    high: '>= 0.76 (High family risk)',
    tips: 'Regular screenings are critical since genetic likelihood is elevated.'
  },
  Age: {
    desc: 'Age of the subject. Metabolic rate changes and risk profile naturally shifts.',
    normal: '21 - 30 years',
    watch: '31 - 45 years',
    high: '>= 46 years (Higher surveillance recommended)',
    tips: 'Perform annual comprehensive medical checkups.'
  }
};

export const PredictionForm: React.FC<PredictionFormProps> = ({ onSubmit, loading, restoreValues }) => {
  const [formData, setFormData] = useState<Record<string, number>>(defaultValues);
  const [showRefGuide, setShowRefGuide] = useState(false);

  useEffect(() => {
    if (restoreValues) {
      setFormData(restoreValues);
    }
  }, [restoreValues]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData(prev => ({ ...prev, [e.target.name]: parseFloat(e.target.value) || 0 }));
  };

  const handleLoadPreset = (presetKey: 'low' | 'borderline' | 'high') => {
    setFormData(presets[presetKey].values);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(formData);
  };

  // Get real-time medical indicator badge
  const getRangeBadge = (key: string, val: number) => {
    if (key === 'Glucose') {
      if (val < 100) return <span className="range-badge range-badge-normal">Normal</span>;
      if (val < 126) return <span className="range-badge range-badge-watch">Prediabetes</span>;
      return <span className="range-badge range-badge-high">Diabetic Level</span>;
    }
    if (key === 'BMI') {
      if (val < 25) return <span className="range-badge range-badge-normal">Normal Weight</span>;
      if (val < 30) return <span className="range-badge range-badge-watch">Overweight</span>;
      return <span className="range-badge range-badge-high">Obese</span>;
    }
    if (key === 'BloodPressure') {
      if (val < 80) return <span className="range-badge range-badge-normal">Normal</span>;
      if (val < 90) return <span className="range-badge range-badge-watch">Elevated</span>;
      return <span className="range-badge range-badge-high">High BP</span>;
    }
    if (key === 'Age') {
      if (val <= 30) return <span className="range-badge range-badge-normal">Age Group 1</span>;
      if (val <= 45) return <span className="range-badge range-badge-watch">Age Group 2</span>;
      return <span className="range-badge range-badge-high">Age Group 3</span>;
    }
    return null;
  };

  return (
    <div className="glass-panel h-full p-5 sm:p-6 space-y-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wide text-[var(--app-primary)]">Input</p>
          <h2 className="mt-1 text-xl font-bold text-[var(--app-text)]">Patient Features</h2>
        </div>
        <span className="rounded-lg bg-[var(--app-primary-soft)] px-3 py-1 text-xs font-semibold text-[var(--app-primary-strong)] flex items-center gap-1">
          <Info className="w-3.5 h-3.5" />
          Pre-processed Live Input
        </span>
      </div>

      {/* Preset Profiles */}
      <div className="space-y-2">
        <label className="text-xs font-bold uppercase tracking-wider text-[var(--app-text-muted)]">Test Presets</label>
        <div className="preset-row">
          <button 
            type="button" 
            onClick={() => handleLoadPreset('low')}
            className="preset-chip preset-chip-low"
          >
            <span className="dot" />
            {presets.low.label}
          </button>
          <button 
            type="button" 
            onClick={() => handleLoadPreset('borderline')}
            className="preset-chip preset-chip-borderline"
          >
            <span className="dot" />
            {presets.borderline.label}
          </button>
          <button 
            type="button" 
            onClick={() => handleLoadPreset('high')}
            className="preset-chip preset-chip-high"
          >
            <span className="dot" />
            {presets.high.label}
          </button>
        </div>
      </div>

      {/* Inputs Form */}
      <form onSubmit={handleSubmit} className="space-y-6">
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          {Object.keys(defaultValues).map((key) => (
            <div key={key} className="flex flex-col gap-1.5">
              <div className="flex items-center justify-between">
                <label htmlFor={key} className="text-xs font-semibold text-[var(--app-text)]">
                  {key.replace(/([A-Z])/g, ' $1').trim()}
                </label>
                {getRangeBadge(key, formData[key])}
              </div>
              <input
                type="number"
                step="any"
                id={key}
                name={key}
                value={formData[key]}
                onChange={handleChange}
                className="surface-control rounded-xl px-3.5 py-2.5 text-sm font-semibold transition-all shadow-sm"
                required
              />
            </div>
          ))}
        </div>

        <button
          type="submit"
          disabled={loading}
          className="primary-gradient flex w-full items-center justify-center gap-2 rounded-xl px-6 py-3.5 font-bold text-white shadow-lg shadow-primary-500/20 transition-all hover:-translate-y-0.5 active:translate-y-0 disabled:cursor-not-allowed disabled:opacity-50 text-sm cursor-pointer"
        >
          {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : <SendHorizontal className="w-5 h-5" />}
          {loading ? 'Analyzing Profile...' : 'Predict Diabetes Risk'}
        </button>
      </form>

      {/* Clinical Feature Reference Accordion */}
      <div className="border-t border-[var(--app-border)] pt-4">
        <button
          onClick={() => setShowRefGuide(prev => !prev)}
          className="flex w-full items-center justify-between text-xs font-bold uppercase tracking-wider text-[var(--app-text-muted)] hover:text-[var(--app-primary)] transition-colors"
        >
          <span className="flex items-center gap-1.5">
            <BookOpen className="w-4 h-4" />
            Clinical Feature Reference Guide
          </span>
          {showRefGuide ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </button>

        {showRefGuide && (
          <div className="mt-4 space-y-4 max-h-[300px] overflow-y-auto pr-1 text-xs">
            {Object.entries(featureGuidelines).map(([name, guide]) => (
              <div key={name} className="p-3 bg-[var(--app-bg)] rounded-xl border border-[var(--app-border)] space-y-1.5">
                <div className="font-bold text-[var(--app-text)]">{name.replace(/([A-Z])/g, ' $1').trim()}</div>
                <p className="text-[var(--app-text-muted)] leading-relaxed">{guide.desc}</p>
                <div className="grid grid-cols-3 gap-2 pt-1 border-t border-[var(--app-border)]">
                  <div>
                    <span className="text-[10px] font-semibold text-emerald-500 block">Normal</span>
                    <span className="font-mono font-medium">{guide.normal}</span>
                  </div>
                  <div>
                    <span className="text-[10px] font-semibold text-amber-500 block">Watch</span>
                    <span className="font-mono font-medium">{guide.watch}</span>
                  </div>
                  <div>
                    <span className="text-[10px] font-semibold text-rose-500 block">Elevated</span>
                    <span className="font-mono font-medium">{guide.high}</span>
                  </div>
                </div>
                <div className="bg-[var(--app-surface-solid)] border border-[var(--app-border)] p-2 rounded-lg mt-1">
                  <span className="text-[10px] font-bold text-[var(--app-primary)] block uppercase">Health Tip:</span>
                  <p className="text-[var(--app-text-muted)] mt-0.5">{guide.tips}</p>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
