import React, { useState } from 'react';
import { Cpu, Shield, Award as AwardIcon, HelpCircle, CheckCircle, AlertTriangle, ChevronRight } from 'lucide-react';
import clsx from 'clsx';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';

interface ModelInfo {
  name: string;
  displayName: string;
  family: 'Tree Ensemble' | 'Linear / Kernel' | 'Instance-Based';
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  description: string;
  representation: string;
  criterion: string;
  assumptions: string;
  strengths: string[];
  weaknesses: string[];
}

const modelsData: ModelInfo[] = [
  {
    name: 'random_forest',
    displayName: 'Random Forest',
    family: 'Tree Ensemble',
    accuracy: 76.62, precision: 0.696, recall: 0.593, f1: 0.640,
    description: 'An ensemble method that builds multiple independent decision trees and merges their predictions. It uses bootstrap aggregating (bagging) and random feature selection to prevent overfitting.',
    representation: 'Raw numerical feature vectors (does not require scaling, handles raw non-linear distributions naturally).',
    criterion: 'Minimizing node impurity (using Gini impurity or Entropy) to find optimal feature splits.',
    assumptions: 'Assumes that combining multiple uncorrelated weak decision tree models reduces overall variance without increasing bias.',
    strengths: [
      'Extremely robust to outliers and noisy data.',
      'Captures complex non-linear feature interactions without manual scaling.',
      'Less prone to overfitting compared to single decision trees.',
    ],
    weaknesses: [
      'Computationally heavy for large tree ensembles.',
      'Essentially a black box, making step-by-step decision explanation difficult.',
    ],
  },
  {
    name: 'xgboost',
    displayName: 'XGBoost',
    family: 'Tree Ensemble',
    accuracy: 74.68, precision: 0.653, recall: 0.593, f1: 0.621,
    description: 'A sequential ensemble technique that trains trees one by one, where each tree is built to correct the errors of the preceding ones using gradient descent optimization.',
    representation: 'Raw numerical feature vectors (natively handles missing values and does not require scaling).',
    criterion: 'Minimizing a regularized cross-entropy loss function via gradient descent.',
    assumptions: 'Assumes residual errors of prior iterations can be modeled and corrected by subsequent weak trees.',
    strengths: [
      'State-of-the-art performance on tabular datasets.',
      'Built-in L1 and L2 regularization to prevent overfitting.',
      'Handles missing values automatically.',
    ],
    weaknesses: [
      'Highly sensitive to hyperparameter tuning.',
      'Can easily overfit the training dataset if early stopping is not configured.',
    ],
  },
  {
    name: 'knn',
    displayName: 'K-Nearest Neighbors',
    family: 'Instance-Based',
    accuracy: 74.03, precision: 0.635, recall: 0.611, f1: 0.623,
    description: 'An instance-based (lazy) learner that does not construct an explicit model during training. It classifies new points by taking a majority vote of the k-nearest data samples in the multi-dimensional space.',
    representation: 'Standardized numerical feature vectors (scaling is mandatory, otherwise large-range variables dominate).',
    criterion: 'Spatial distance metric (usually Euclidean distance).',
    assumptions: 'Assumes points close to each other in the feature space are highly likely to share the same class.',
    strengths: [
      'Simple to understand and implement.',
      'No explicit training phase required; updates dynamically with new data.',
      'Captures localized boundary changes effectively.',
    ],
    weaknesses: [
      'Extremely slow at prediction time on massive datasets.',
      'Suffers from the curse of dimensionality and is highly sensitive to irrelevant features.',
    ],
  },
  {
    name: 'logistic_regression',
    displayName: 'Logistic Regression',
    family: 'Linear / Kernel',
    accuracy: 70.78, precision: 0.600, recall: 0.500, f1: 0.545,
    description: 'A linear classification model that models the probability of a binary output by fitting the log-odds ratio to a linear combination of input features.',
    representation: 'Standardized numerical feature vectors (benefits from scaling for optimization stability).',
    criterion: 'Maximizing the likelihood of the training data (minimizing log-loss / cross-entropy).',
    assumptions: 'Assumes a linear relationship between features and the log-odds of the outcome. Assumes minimal multicollinearity.',
    strengths: [
      'Very fast to train and evaluate.',
      'Highly interpretable (coefficients represent the direct weight of each feature).',
      'Outputs well-calibrated class probabilities.',
    ],
    weaknesses: [
      'Cannot capture non-linear relationships without manual feature cross-engineering.',
      'Underperforms on highly complex medical datasets.',
    ],
  },
  {
    name: 'svm_linear',
    displayName: 'SVM (Linear Kernel)',
    family: 'Linear / Kernel',
    accuracy: 70.78, precision: 0.605, recall: 0.481, f1: 0.536,
    description: 'A margin classifier that finds the optimal straight hyperplane separating the binary classes while maximizing the margin of safety between the boundary and support vectors.',
    representation: 'Standardized numerical feature vectors.',
    criterion: 'Maximizing the margin of separation while minimizing classification errors (hinge loss with L2 regularization).',
    assumptions: 'Assumes the classes are linearly separable (or soft-margin separable) in the feature space.',
    strengths: [
      'Highly effective in high-dimensional spaces.',
      'Robust against individual outliers that do not touch the margin boundary.',
    ],
    weaknesses: [
      'Does not natively output probabilities (requires Platt scaling).',
      'Cannot separate complex curved patterns.',
    ],
  },
  {
    name: 'svm_rbf',
    displayName: 'SVM (RBF Kernel)',
    family: 'Linear / Kernel',
    accuracy: 69.48, precision: 0.581, recall: 0.463, f1: 0.515,
    description: 'An advanced SVM model that projects input features into a higher-dimensional space using the Radial Basis Function kernel, finding linear boundaries in that transformed space.',
    representation: 'Standardized numerical feature vectors.',
    criterion: 'Maximizing margin in the kernel-transformed high-dimensional feature space.',
    assumptions: 'Assumes points grouped closer together in the transformed space share the same outcome.',
    strengths: [
      'Capable of modeling extremely complex non-linear classification boundaries.',
      'Handles complicated overlaps between patient features.',
    ],
    weaknesses: [
      'Difficult to interpret (support vectors are in high dimensions).',
      'Computationally expensive to train and tune parameters (C, gamma).',
    ],
  },
];

const familyColor: Record<ModelInfo['family'], string> = {
  'Tree Ensemble': '#10b981',
  'Instance-Based': '#f59e0b',
  'Linear / Kernel': '#6366f1',
};

const CustomBarTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    const m = payload[0].payload as ModelInfo;
    return (
      <div className="rounded-xl border border-[var(--app-border)] bg-[var(--app-surface-solid)] p-3 text-xs shadow-2xl min-w-[170px] space-y-1">
        <p className="font-semibold text-[var(--app-text)]">{m.displayName}</p>
        <p className="muted-text">F1: <span className="font-mono font-semibold text-[var(--app-text)]">{m.f1.toFixed(3)}</span></p>
        <p className="muted-text">Accuracy: <span className="font-mono font-semibold text-[var(--app-text)]">{m.accuracy.toFixed(2)}%</span></p>
      </div>
    );
  }
  return null;
};

export const ModelDocumentation: React.FC = () => {
  const [selected, setSelected] = useState<string>('random_forest');
  const model = modelsData.find(m => m.name === selected) ?? modelsData[0];
  const sortedByF1 = [...modelsData].sort((a, b) => b.f1 - a.f1);

  return (
    <div className="space-y-6">
      <div className="rounded-2xl bg-[var(--app-surface)] border border-[var(--app-border)] p-6 shadow-lg backdrop-blur-md">
        <h2 className="text-2xl font-black tracking-tight text-[var(--app-text)] flex items-center gap-2">
          <Cpu className="w-6 h-6 text-[var(--app-primary)]" />
          Model Performance Hub
        </h2>
        <p className="mt-2 text-sm text-[var(--app-text-muted)] max-w-2xl">
          Six traditional ML classifiers trained on the Pima Indian Diabetes dataset. Select a model below to
          review its math, assumptions, and where it wins or loses against the others.
        </p>
      </div>

      {/* Comparison chart */}
      <div className="glass-panel p-5">
        <div className="mb-3 flex items-center justify-between">
          <h3 className="text-sm font-semibold text-[var(--app-text)]">F1-score comparison (test set)</h3>
          <div className="flex items-center gap-3 text-[11px] muted-text">
            {(['Tree Ensemble', 'Linear / Kernel', 'Instance-Based'] as const).map(f => (
              <span key={f} className="flex items-center gap-1.5">
                <span className="h-2.5 w-2.5 rounded-full" style={{ background: familyColor[f] }} />
                {f}
              </span>
            ))}
          </div>
        </div>
        <div className="h-[220px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={sortedByF1} layout="vertical" margin={{ top: 5, right: 25, left: 10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--app-border)" horizontal={false} />
              <XAxis type="number" domain={[0, 0.8]} stroke="var(--app-text-muted)" fontSize={11} tickLine={false} />
              <YAxis dataKey="displayName" type="category" width={130} stroke="var(--app-text-muted)" fontSize={11} tickLine={false} axisLine={false} />
              <Tooltip content={<CustomBarTooltip />} cursor={{ fill: 'var(--app-primary-soft)' }} />
              <Bar
                dataKey="f1"
                radius={[4, 4, 4, 4]}
                onClick={(data: any) => setSelected(data.name)}
                cursor="pointer"
              >
                {sortedByF1.map(m => (
                  <Cell
                    key={m.name}
                    fill={familyColor[m.family]}
                    fillOpacity={m.name === selected ? 1 : 0.45}
                    stroke={m.name === selected ? familyColor[m.family] : 'transparent'}
                    strokeWidth={2}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Selector + detail explorer */}
      <div className="grid grid-cols-1 gap-5 lg:grid-cols-12 items-start">
        <div className="lg:col-span-4 glass-panel p-3 space-y-1.5">
          {sortedByF1.map((m, idx) => (
            <button
              key={m.name}
              onClick={() => setSelected(m.name)}
              className={clsx(
                'ranking-button w-full text-left p-3 rounded-lg transition-all flex items-center justify-between gap-2',
                selected === m.name && 'ranking-button-selected'
              )}
            >
              <div className="flex min-w-0 items-center gap-2.5">
                <span
                  className="flex h-6 w-6 shrink-0 items-center justify-center rounded-md text-[10px] font-bold text-white"
                  style={{ background: familyColor[m.family] }}
                >
                  {idx + 1}
                </span>
                <div className="min-w-0">
                  <p className="truncate text-sm font-semibold text-[var(--app-text)]">{m.displayName}</p>
                  <p className="text-[10px] muted-text">{m.family} &middot; F1 {m.f1.toFixed(3)}</p>
                </div>
              </div>
              <ChevronRight className={clsx('w-4 h-4 shrink-0', selected === m.name ? 'text-[var(--app-primary)]' : 'text-[var(--app-text-muted)]')} />
            </button>
          ))}
        </div>

        <div className="lg:col-span-8 glass-panel p-6">
          <div className="flex items-start justify-between gap-3 border-b border-[var(--app-border)] pb-4 mb-4">
            <div>
              <span
                className="inline-block mb-1.5 text-[11px] font-semibold px-2 py-0.5 rounded-full text-white"
                style={{ background: familyColor[model.family] }}
              >
                {model.family}
              </span>
              <h3 className="text-lg font-bold text-[var(--app-text)]">{model.displayName}</h3>
              {model.name === 'random_forest' && (
                <p className="mt-0.5 text-xs font-semibold text-[var(--app-primary)]">Auto-selected as final model (highest F1)</p>
              )}
            </div>
            <div className="flex flex-col items-end shrink-0">
              <div className="text-xs font-semibold muted-text">Test F1</div>
              <div className="text-lg font-black text-[var(--app-primary)]">{model.f1.toFixed(3)}</div>
            </div>
          </div>

          <p className="text-sm text-[var(--app-text-muted)] leading-relaxed mb-4">{model.description}</p>

          <div className="grid grid-cols-4 gap-2 text-center text-xs mb-4">
            {[
              { label: 'Accuracy', value: `${model.accuracy.toFixed(2)}%` },
              { label: 'Precision', value: model.precision.toFixed(3) },
              { label: 'Recall', value: model.recall.toFixed(3) },
              { label: 'F1-score', value: model.f1.toFixed(3) },
            ].map(stat => (
              <div key={stat.label} className="rounded-lg border border-[var(--app-border)] p-2" style={{ background: 'var(--app-surface)' }}>
                <div className="font-medium muted-text">{stat.label}</div>
                <div className="font-bold text-[var(--app-text)]">{stat.value}</div>
              </div>
            ))}
          </div>

          <div className="space-y-3 border-t border-[var(--app-border)] pt-4 mb-4 text-xs">
            <div className="grid grid-cols-4 gap-2">
              <span className="col-span-1 font-semibold text-[var(--app-text)] flex items-center gap-1">
                <Shield className="w-3.5 h-3.5 text-[var(--app-accent)]" /> Representation
              </span>
              <span className="col-span-3 text-[var(--app-text-muted)]">{model.representation}</span>
            </div>
            <div className="grid grid-cols-4 gap-2">
              <span className="col-span-1 font-semibold text-[var(--app-text)] flex items-center gap-1">
                <AwardIcon className="w-3.5 h-3.5 text-[var(--app-accent)]" /> Criterion
              </span>
              <span className="col-span-3 text-[var(--app-text-muted)]">{model.criterion}</span>
            </div>
            <div className="grid grid-cols-4 gap-2">
              <span className="col-span-1 font-semibold text-[var(--app-text)] flex items-center gap-1">
                <HelpCircle className="w-3.5 h-3.5 text-[var(--app-accent)]" /> Assumptions
              </span>
              <span className="col-span-3 text-[var(--app-text-muted)]">{model.assumptions}</span>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 border-t border-[var(--app-border)] pt-4 text-xs">
            <div>
              <h4 className="font-bold text-[var(--app-success)] flex items-center gap-1 mb-2">
                <CheckCircle className="w-3.5 h-3.5" /> Strengths
              </h4>
              <ul className="list-disc pl-4 space-y-1 text-[var(--app-text-muted)]">
                {model.strengths.map((s, i) => <li key={i}>{s}</li>)}
              </ul>
            </div>
            <div>
              <h4 className="font-bold text-[var(--app-danger)] flex items-center gap-1 mb-2">
                <AlertTriangle className="w-3.5 h-3.5" /> Weaknesses
              </h4>
              <ul className="list-disc pl-4 space-y-1 text-[var(--app-text-muted)]">
                {model.weaknesses.map((w, i) => <li key={i}>{w}</li>)}
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
