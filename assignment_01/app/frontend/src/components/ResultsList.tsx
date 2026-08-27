import React from 'react';
import type { ModelSummary } from '../App';
import { ChevronRight, Trophy, TreePine, Waypoints, Sigma } from 'lucide-react';
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

const getModelFamily = (name: string) => {
  if (name === 'random_forest' || name === 'xgboost') return 'Tree Ensemble';
  if (name === 'knn') return 'Instance-Based';
  return 'Linear / Kernel';
};

const familyIcon = (family: string) => {
  if (family === 'Tree Ensemble') return TreePine;
  if (family === 'Instance-Based') return Waypoints;
  return Sigma;
};

export const ResultsList: React.FC<ResultsListProps> = ({ summaries, onSelectModel, selectedModel }) => {
  const runnerUps = summaries.slice(3);

  return (
    <div className="h-full p-5 sm:p-6 flex flex-col space-y-6">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between border-b border-[var(--app-border)] pb-4">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wide text-[var(--app-primary)]">Output</p>
          <h2 className="mt-1 flex items-center gap-2 text-xl font-bold text-[var(--app-text)]">
            <Trophy className="w-5 h-5 text-amber-500" />
            Classifier Leaderboard
          </h2>
        </div>
        <p className="text-xs text-[var(--app-text-muted)] font-medium">Select a model card to audit details &amp; SHAP values</p>
      </div>

      {/* Top 3 — equal-height rank cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        {summaries.slice(0, 3).map((summary, idx) => {
          const family = getModelFamily(summary.model);
          const FamilyIcon = familyIcon(family);
          const isFirst = idx === 0;
          return (
            <motion.button
              key={summary.model}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: idx * 0.08, type: 'spring', stiffness: 140 }}
              onClick={() => onSelectModel(summary.model)}
              className={clsx(
                'rank-card text-left cursor-pointer group',
                isFirst && 'rank-card-best',
                selectedModel === summary.model && 'ring-2 ring-[var(--app-primary)]'
              )}
            >
              <div className="flex items-center justify-between">
                <span className={clsx('rank-pill', isFirst && 'rank-pill-best')}>#{idx + 1}</span>
                <FamilyIcon className={clsx('w-4 h-4', isFirst ? 'text-amber-500' : 'text-[var(--app-text-muted)]')} />
              </div>

              <h3 className="mt-3 text-sm font-bold text-[var(--app-text)] leading-snug line-clamp-2 min-h-[2.4rem]">
                {formatModelName(summary.model)}
              </h3>
              <span className="text-[10px] font-semibold text-[var(--app-text-muted)] uppercase tracking-wide">
                {family}
              </span>

              {isFirst && (
                <span className="mt-2 inline-flex items-center gap-1 text-[10px] font-bold text-amber-600 bg-amber-500/10 px-2 py-0.5 rounded-full w-fit">
                  <Trophy className="w-2.5 h-2.5" />
                  Best Predictor
                </span>
              )}

              <div className="mt-4 pt-3 border-t border-[var(--app-border)] flex items-end justify-between">
                <div>
                  <div className={clsx('text-lg font-black font-mono', isFirst ? 'text-amber-600' : 'text-[var(--app-primary-strong)]')}>
                    {(summary.accuracy * 100).toFixed(1)}%
                  </div>
                  <p className="text-[9px] text-[var(--app-text-muted)] uppercase tracking-wider font-semibold">Test Accuracy</p>
                </div>
                <ChevronRight className="w-4 h-4 text-[var(--app-text-muted)] group-hover:text-[var(--app-primary)] group-hover:translate-x-0.5 transition-transform" />
              </div>
            </motion.button>
          );
        })}
      </div>

      {/* Runner Ups (Ranks 4-6) */}
      {runnerUps.length > 0 && (
        <div className="space-y-2 pt-4">
          <label className="text-[10px] font-bold uppercase tracking-wider text-[var(--app-text-muted)] block">Runner-Up Classifiers</label>
          <div className="space-y-2">
            {runnerUps.map((summary, idx) => {
              const rank = idx + 4;
              return (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: rank * 0.05 }}
                  key={summary.model}
                >
                  <button
                    onClick={() => onSelectModel(summary.model)}
                    className={clsx(
                      'ranking-button w-full text-left p-3.5 rounded-xl transition-all flex items-center justify-between gap-3 group cursor-pointer hover:scale-[1.005]',
                      selectedModel === summary.model && 'ranking-button-selected shadow-md shadow-primary-900/10'
                    )}
                  >
                    <div className="flex min-w-0 items-center gap-3">
                      <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-[var(--app-primary-soft)] text-[var(--app-text-muted)] font-black text-xs">
                        #{rank}
                      </div>
                      <div className="min-w-0">
                        <h3 className="truncate font-semibold text-xs text-[var(--app-text)]">{formatModelName(summary.model)}</h3>
                        <div className="mt-1.5 flex items-center gap-2">
                          <div className="accuracy-track">
                            <span style={{ width: `${Math.round(summary.accuracy * 100)}%` }} />
                          </div>
                          <p className="text-xs font-bold text-[var(--app-primary-strong)] font-mono">
                            {(summary.accuracy * 100).toFixed(1)}%
                          </p>
                        </div>
                      </div>
                    </div>
                    <ChevronRight className={clsx(
                      'w-4 h-4 shrink-0 transition-transform text-[var(--app-text-muted)] group-hover:text-[var(--app-primary)] group-hover:translate-x-0.5'
                    )} />
                  </button>
                </motion.div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};
