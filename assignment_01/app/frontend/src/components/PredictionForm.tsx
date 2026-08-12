import React, { useState } from 'react';
import { Loader2 } from 'lucide-react';

interface PredictionFormProps {
  onSubmit: (data: Record<string, number>) => void;
  loading: boolean;
}

const defaultValues = {
  Pregnancies: 6,
  Glucose: 148,
  BloodPressure: 72,
  SkinThickness: 35,
  Insulin: 0,
  BMI: 33.6,
  DiabetesPedigreeFunction: 0.627,
  Age: 50
};

export const PredictionForm: React.FC<PredictionFormProps> = ({ onSubmit, loading }) => {
  const [formData, setFormData] = useState<Record<string, number>>(defaultValues);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData(prev => ({ ...prev, [e.target.name]: parseFloat(e.target.value) || 0 }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(formData);
  };

  return (
    <div className="glass-panel p-6 h-full">
      <h2 className="text-2xl font-bold mb-6 text-primary-100">Patient Features</h2>
      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {Object.entries(defaultValues).map(([key, _]) => (
            <div key={key} className="flex flex-col gap-1">
              <label htmlFor={key} className="text-sm text-slate-300 font-medium">
                {key.replace(/([A-Z])/g, ' $1').trim()}
              </label>
              <input
                type="number"
                step="any"
                id={key}
                name={key}
                value={formData[key]}
                onChange={handleChange}
                className="bg-slate-800/50 border border-slate-700 rounded-lg px-4 py-2 text-slate-100 focus:outline-none focus:ring-2 focus:ring-primary-500/50 transition-all"
                required
              />
            </div>
          ))}
        </div>
        
        <button
          type="submit"
          disabled={loading}
          className="w-full mt-6 bg-gradient-to-r from-primary-600 to-primary-500 hover:from-primary-500 hover:to-primary-400 text-white font-bold py-3 px-6 rounded-xl shadow-lg shadow-primary-500/25 transition-all transform hover:-translate-y-0.5 active:translate-y-0 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
        >
          {loading && <Loader2 className="w-5 h-5 animate-spin" />}
          {loading ? 'Analyzing...' : 'Predict Outcome'}
        </button>
      </form>
    </div>
  );
};
