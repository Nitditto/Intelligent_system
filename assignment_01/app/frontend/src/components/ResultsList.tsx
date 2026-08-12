import React from 'react';
import type { ModelSummary } from '../App';
import { ChevronRight, Award } from 'lucide-react';
import clsx from 'clsx';
import { motion } from 'framer-motion';

interface ResultsListProps {
  summaries: ModelSummary[];
  onSelectModel: (model: string) => void;
  selectedModel: string | null;
}

const formatModelName = (name: string) => {
  return name.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
};

export const ResultsList: React.FC<ResultsListProps> = ({ summaries, onSelectModel, selectedModel }) => {
  return (
    <div className="glass-panel p-6 h-full flex flex-col">
      <h2 className="text-2xl font-bold mb-6 text-primary-100 flex items-center gap-2">
        <Award className="w-6 h-6 text-yellow-500" />
        Model Rankings
      </h2>
      <div className="flex-1 overflow-y-auto pr-2 space-y-3">
        {summaries.map((summary, idx) => (
          <motion.div 
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: idx * 0.1 }}
            key={summary.model}
          >
            <button
              onClick={() => onSelectModel(summary.model)}
              className={clsx(
                "w-full text-left p-4 rounded-xl transition-all border flex items-center justify-between group",
                selectedModel === summary.model
                  ? "bg-primary-900/40 border-primary-500 shadow-lg shadow-primary-900/20"
                  : "bg-slate-800/30 border-slate-700 hover:bg-slate-800 hover:border-slate-600"
              )}
            >
              <div className="flex items-center gap-4">
                <div className={clsx(
                  "flex items-center justify-center w-8 h-8 rounded-full font-bold text-sm",
                  idx === 0 ? "bg-yellow-500/20 text-yellow-400" :
                  idx === 1 ? "bg-slate-300/20 text-slate-300" :
                  idx === 2 ? "bg-amber-700/20 text-amber-500" : "bg-slate-700 text-slate-400"
                )}>
                  #{idx + 1}
                </div>
                <div>
                  <h3 className="font-semibold text-slate-100">{formatModelName(summary.model)}</h3>
                  <p className="text-sm text-slate-400">Accuracy: {(summary.accuracy * 100).toFixed(2)}%</p>
                </div>
              </div>
              <ChevronRight className={clsx(
                "w-5 h-5 transition-transform",
                selectedModel === summary.model ? "text-primary-400" : "text-slate-500 group-hover:text-slate-300 group-hover:translate-x-1"
              )} />
            </button>
          </motion.div>
        ))}
      </div>
    </div>
  );
};
