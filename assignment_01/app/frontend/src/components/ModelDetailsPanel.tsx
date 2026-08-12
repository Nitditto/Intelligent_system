import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Loader2, Target, Percent } from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell
} from 'recharts';
import clsx from 'clsx';

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
}

const formatModelName = (name: string) => {
  return name.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
};

export const ModelDetailsPanel: React.FC<ModelDetailsPanelProps> = ({ isOpen, onClose, model, resultId, apiUrl }) => {
  const [data, setData] = useState<DetailedModelResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen && model && resultId) {
      setLoading(true);
      setError(null);
      axios.get<DetailedModelResult>(`${apiUrl}/api/result/${resultId}?model=${model}`)
        .then(res => {
          // Sort SHAP values by absolute magnitude for better chart display
          const sortedData = {
            ...res.data,
            shap_values: res.data.shap_values.sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
          };
          setData(sortedData);
        })
        .catch(err => {
          console.error(err);
          setError('Failed to load model details.');
        })
        .finally(() => setLoading(false));
    }
  }, [isOpen, model, resultId, apiUrl]);

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 bg-slate-900/60 backdrop-blur-sm z-40"
          />
          <motion.div
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ type: 'spring', damping: 25, stiffness: 200 }}
            className="fixed right-0 top-0 bottom-0 w-full max-w-xl bg-surface border-l border-slate-700 shadow-2xl z-50 overflow-y-auto"
          >
            <div className="p-6">
              <button
                onClick={onClose}
                className="absolute top-6 right-6 p-2 rounded-full hover:bg-slate-800 text-slate-400 hover:text-white transition-colors"
              >
                <X className="w-6 h-6" />
              </button>

              {loading ? (
                <div className="flex flex-col items-center justify-center h-[80vh] text-primary-400 gap-4">
                  <Loader2 className="w-10 h-10 animate-spin" />
                  <p>Loading model insights...</p>
                </div>
              ) : error ? (
                <div className="flex items-center justify-center h-[80vh] text-red-400">
                  <p>{error}</p>
                </div>
              ) : data ? (
                <div className="space-y-8 mt-4">
                  <div>
                    <h2 className="text-3xl font-bold text-slate-100">{formatModelName(data.model)}</h2>
                    <p className="text-slate-400 mt-1">Detailed Analysis</p>
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div className="glass-panel p-4 flex flex-col gap-2 relative overflow-hidden">
                      <div className="absolute top-0 right-0 p-4 opacity-10">
                        <Target className="w-16 h-16" />
                      </div>
                      <span className="text-sm text-slate-400 font-medium">Prediction</span>
                      <span className={clsx(
                        "text-2xl font-bold",
                        data.prediction === 1 ? "text-red-400" : "text-green-400"
                      )}>
                        {data.prediction === 1 ? 'Diabetes' : 'No Diabetes'}
                      </span>
                    </div>
                    <div className="glass-panel p-4 flex flex-col gap-2 relative overflow-hidden">
                      <div className="absolute top-0 right-0 p-4 opacity-10">
                        <Percent className="w-16 h-16" />
                      </div>
                      <span className="text-sm text-slate-400 font-medium">Confidence</span>
                      <span className="text-2xl font-bold text-primary-400">
                        {(data.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>

                  <div className="glass-panel p-6">
                    <h3 className="text-lg font-semibold text-slate-200 mb-6">Feature Contributions (SHAP Values)</h3>
                    <div className="h-[400px] w-full">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart
                          data={data.shap_values}
                          layout="vertical"
                          margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                        >
                          <CartesianGrid strokeDasharray="3 3" stroke="#334155" horizontal={false} />
                          <XAxis type="number" stroke="#94a3b8" />
                          <YAxis 
                            dataKey="feature" 
                            type="category" 
                            width={120} 
                            stroke="#94a3b8"
                            tick={{ fill: '#cbd5e1', fontSize: 12 }} 
                          />
                          <Tooltip 
                            contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', color: '#f1f5f9' }}
                            itemStyle={{ color: '#d946ef' }}
                            cursor={{ fill: '#334155', opacity: 0.4 }}
                            formatter={(value: number) => [value.toFixed(4), 'Impact']}
                          />
                          <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                            {data.shap_values.map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={entry.value > 0 ? '#ef4444' : '#3b82f6'} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                    <p className="text-xs text-slate-500 mt-4 text-center">
                      <span className="text-red-400 font-semibold">Red</span> pushes prediction towards Diabetes. 
                      <span className="text-blue-400 font-semibold ml-2">Blue</span> pushes towards No Diabetes.
                    </p>
                  </div>
                </div>
              ) : null}
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
};
