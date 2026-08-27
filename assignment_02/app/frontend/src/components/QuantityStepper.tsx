import React from 'react';
import { Minus, Plus } from 'lucide-react';

interface QuantityStepperProps {
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
}

export const QuantityStepper: React.FC<QuantityStepperProps> = ({ value, onChange, min = 0, max = 20 }) => {
  return (
    <div className="flex items-center rounded-xl border border-stone-200 dark:border-navy-600 bg-white dark:bg-navy-900/60 overflow-hidden shadow-sm">
      <button
        type="button"
        onClick={() => onChange(Math.max(min, value - 1))}
        disabled={value <= min}
        aria-label="Giảm"
        className="w-10 h-11 shrink-0 flex items-center justify-center text-stone-500 dark:text-stone-400 hover:bg-stone-100 dark:hover:bg-navy-800 disabled:opacity-30 disabled:cursor-not-allowed transition-colors cursor-pointer"
      >
        <Minus className="w-4 h-4" />
      </button>
      <div className="flex-1 text-center text-sm font-bold text-stone-900 dark:text-stone-100 select-none">{value}</div>
      <button
        type="button"
        onClick={() => onChange(Math.min(max, value + 1))}
        disabled={value >= max}
        aria-label="Tăng"
        className="w-10 h-11 shrink-0 flex items-center justify-center text-stone-500 dark:text-stone-400 hover:bg-stone-100 dark:hover:bg-navy-800 disabled:opacity-30 disabled:cursor-not-allowed transition-colors cursor-pointer"
      >
        <Plus className="w-4 h-4" />
      </button>
    </div>
  );
};
