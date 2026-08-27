import React, { useEffect, useRef, useState } from 'react';
import axios from 'axios';
import { SlidersHorizontal, TrendingUp, TrendingDown, Minus, RotateCcw, Maximize, Signpost, Layers, BedDouble, Bath } from 'lucide-react';
import { pickBestModel } from '../utils';

interface WhatIfSimulatorProps {
  baseForm: Record<string, any>;
  basePrice: number;
  baseModelName: string;
  apiBase: string;
}

const formatModelName = (name: string) => name.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');

const SLIDERS: { key: string; label: string; icon: React.ElementType; min: number; max: number; step: number; unit?: string }[] = [
  { key: 'Area', label: 'Diện tích', icon: Maximize, min: 15, max: 500, step: 1, unit: 'm²' },
  { key: 'Frontage', label: 'Mặt tiền', icon: Signpost, min: 1, max: 20, step: 0.5, unit: 'm' },
  { key: 'Access_Road', label: 'Ngõ vào', icon: Signpost, min: 1, max: 20, step: 0.5, unit: 'm' },
  { key: 'Floors', label: 'Số tầng', icon: Layers, min: 1, max: 10, step: 1 },
  { key: 'Bedrooms', label: 'Phòng ngủ', icon: BedDouble, min: 1, max: 10, step: 1 },
  { key: 'Bathrooms', label: 'Phòng tắm', icon: Bath, min: 1, max: 10, step: 1 },
];

export const WhatIfSimulator: React.FC<WhatIfSimulatorProps> = ({ baseForm, basePrice, baseModelName, apiBase }) => {
  const [values, setValues] = useState<Record<string, number>>({});
  const [baseValues, setBaseValues] = useState<Record<string, number>>({});
  const [price, setPrice] = useState(basePrice);
  const [modelName, setModelName] = useState(baseModelName);
  const [loading, setLoading] = useState(false);
  const debounceRef = useRef<number | undefined>(undefined);
  const skipNextFetch = useRef(false);

  // Re-sync sliders whenever a fresh base prediction comes in (no need to re-fetch, we already have basePrice)
  useEffect(() => {
    skipNextFetch.current = true;
    const snapshot = {
      Area: baseForm.Area,
      Frontage: baseForm.Frontage,
      Access_Road: baseForm.Access_Road,
      Floors: baseForm.Floors,
      Bedrooms: baseForm.Bedrooms,
      Bathrooms: baseForm.Bathrooms,
    };
    setValues(snapshot);
    setBaseValues(snapshot);
    setPrice(basePrice);
    setModelName(baseModelName);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [baseForm, basePrice, baseModelName]);

  useEffect(() => {
    if (Object.keys(values).length === 0) return;
    if (skipNextFetch.current) {
      skipNextFetch.current = false;
      return;
    }
    window.clearTimeout(debounceRef.current);
    debounceRef.current = window.setTimeout(async () => {
      setLoading(true);
      try {
        const res = await axios.post(`${apiBase}/predict`, { ...baseForm, ...values });
        const best = pickBestModel(res.data.results);
        if (best) {
          setPrice(best.price);
          setModelName(best.model);
        }
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    }, 400);
    return () => window.clearTimeout(debounceRef.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [values]);

  if (Object.keys(values).length === 0) return null;

  const delta = price - basePrice;
  const deltaPct = basePrice ? (delta / basePrice) * 100 : 0;
  const isUp = delta > 0.005;
  const isDown = delta < -0.005;
  const isDirty = SLIDERS.some(s => values[s.key] !== baseValues[s.key]);
  const handleReset = () => setValues({ ...baseValues });

  return (
    <div className="glass-card p-6 sm:p-8">
      <div className="flex flex-wrap items-center justify-between gap-4 mb-2">
        <h3 className="text-lg font-bold text-navy-800 dark:text-stone-100 flex items-center gap-2">
          <SlidersHorizontal className="w-5 h-5 text-brand-500" />
          Thử nghiệm What-if
        </h3>

        <button
          type="button"
          onClick={handleReset}
          disabled={!isDirty}
          className="inline-flex items-center gap-1.5 text-xs font-semibold text-stone-500 dark:text-stone-400 hover:text-brand-600 dark:hover:text-brand-400 disabled:opacity-30 disabled:cursor-not-allowed transition-colors cursor-pointer"
        >
          <RotateCcw className="w-3.5 h-3.5" />
          Khôi phục gốc
        </button>
      </div>
      <p className="text-xs text-stone-400 dark:text-stone-500 mb-6">Kéo các thanh dưới đây để xem giá trị thay đổi theo thời gian thực (mô hình {formatModelName(modelName)}).</p>

      {/* Live price readout */}
      <div className="flex flex-wrap items-center gap-4 mb-6 rounded-2xl border border-stone-100 dark:border-navy-700 bg-stone-50 dark:bg-navy-900/50 p-5">
        <div>
          <div className="text-[11px] font-semibold text-stone-400 dark:text-stone-500 uppercase tracking-wider">Giá gốc</div>
          <div className="text-lg font-bold text-stone-500 dark:text-stone-400">{basePrice.toFixed(2)} tỷ</div>
        </div>
        <div className={`text-2xl font-light ${isUp ? 'text-red-400' : isDown ? 'text-emerald-400' : 'text-stone-300 dark:text-stone-600'}`}>→</div>
        <div className="flex-1 min-w-[140px]">
          <div className="text-[11px] font-semibold text-stone-400 dark:text-stone-500 uppercase tracking-wider">Giá dự đoán mới</div>
          <div className="text-3xl font-extrabold text-navy-800 dark:text-white flex items-center gap-2">
            {price.toFixed(2)} tỷ
            {loading && <span className="w-4 h-4 rounded-full border-2 border-brand-500 border-t-transparent animate-spin" />}
          </div>
        </div>
        <div className={`flex items-center gap-1.5 px-3 py-2 rounded-xl text-sm font-bold ${
          isUp ? 'bg-red-50 dark:bg-red-500/10 text-red-600 dark:text-red-400'
          : isDown ? 'bg-emerald-50 dark:bg-emerald-500/10 text-emerald-600 dark:text-emerald-400'
          : 'bg-stone-100 dark:bg-navy-800 text-stone-400 dark:text-stone-500'
        }`}>
          {isUp ? <TrendingUp className="w-4 h-4" /> : isDown ? <TrendingDown className="w-4 h-4" /> : <Minus className="w-4 h-4" />}
          {delta >= 0 ? '+' : ''}{delta.toFixed(2)} tỷ ({deltaPct >= 0 ? '+' : ''}{deltaPct.toFixed(1)}%)
        </div>
      </div>

      {/* Slider controls */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {SLIDERS.map(s => {
          const Icon = s.icon;
          const val = values[s.key] ?? s.min;
          const pct = ((val - s.min) / (s.max - s.min)) * 100;
          const changed = val !== baseValues[s.key];
          return (
            <div
              key={s.key}
              className={`rounded-2xl border p-4 transition-colors ${
                changed
                  ? 'border-brand-300 dark:border-brand-500/40 bg-brand-50/60 dark:bg-brand-900/10'
                  : 'border-stone-100 dark:border-navy-700 bg-white dark:bg-navy-900/40'
              }`}
            >
              <div className="flex items-center justify-between mb-3">
                <span className="flex items-center gap-1.5 text-xs font-semibold text-stone-500 dark:text-stone-400">
                  <Icon className="w-3.5 h-3.5 text-brand-500" />
                  {s.label}
                </span>
                <span className="text-sm font-extrabold text-navy-800 dark:text-stone-100 tabular-nums">
                  {val}{s.unit ? ` ${s.unit}` : ''}
                </span>
              </div>
              <input
                type="range"
                min={s.min}
                max={s.max}
                step={s.step}
                value={val}
                onChange={(e) => setValues(v => ({ ...v, [s.key]: parseFloat(e.target.value) }))}
                className="range-slider"
                style={{ '--pct': `${pct}%` } as React.CSSProperties}
              />
              <div className="flex justify-between mt-1.5 text-[10px] font-medium text-stone-300 dark:text-stone-600">
                <span>{s.min}{s.unit ? ` ${s.unit}` : ''}</span>
                <span>{s.max}{s.unit ? ` ${s.unit}` : ''}</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
