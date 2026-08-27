import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Building, MapPin, Ruler, Compass, Search, Home, ChevronRight, CheckCircle2, Sun, Moon, Crown, Percent, TrendingUp, Target, LayoutGrid, History, Database, Cpu, FileText } from 'lucide-react';
import localProvinces from './locations.json';
import { ModelDetailsPanel } from './components/ModelDetailsPanel';
import { WhatIfSimulator } from './components/WhatIfSimulator';
import { QuantityStepper } from './components/QuantityStepper';
import { PredictionHistory } from './components/PredictionHistory';
import { PropertyReport } from './components/PropertyReport';
import { DatasetInsights } from './components/DatasetInsights';
import { ModelExplainer } from './components/ModelExplainer';
import { pickBestModel, type ModelResult } from './utils';
import { Browser } from '@capacitor/browser';

export type HouseForm = {
  StreetAddress: string;
  City: string;
  District: string;
  Area: number;
  Frontage: number;
  Access_Road: number;
  House_direction: string;
  Balcony_direction: string;
  Floors: number;
  Bedrooms: number;
  Bathrooms: number;
  Legal_status: string;
  Furniture_state: string;
};

export type HistoryRecord = {
  id: string;
  resultId: string;
  timestamp: string;
  form: HouseForm;
  results: ModelResult[];
};

type Tab = 'predict' | 'history' | 'insights' | 'models' | 'report';

const TABS: { id: Tab; label: string; icon: React.ElementType }[] = [
  { id: 'predict', label: 'Dự đoán giá', icon: LayoutGrid },
  { id: 'history', label: 'Lịch sử', icon: History },
  { id: 'insights', label: 'Thống kê dữ liệu', icon: Database },
  { id: 'models', label: 'Mô hình', icon: Cpu },
  { id: 'report', label: 'Báo cáo', icon: FileText },
];

const HISTORY_STORAGE_KEY = 'house_price_history';

axios.defaults.headers.common['ngrok-skip-browser-warning'] = 'true';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8001/api';

type Theme = 'light' | 'dark';

function getInitialTheme(): Theme {
  if (typeof window === 'undefined') return 'light';
  const stored = window.localStorage.getItem('theme');
  if (stored === 'light' || stored === 'dark') return stored;
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

function getInitialHistory(): HistoryRecord[] {
  if (typeof window === 'undefined') return [];
  try {
    const stored = window.localStorage.getItem(HISTORY_STORAGE_KEY);
    return stored ? JSON.parse(stored) : [];
  } catch {
    return [];
  }
}

function App() {
  const [theme, setTheme] = useState<Theme>(getInitialTheme);
  const [locations, setLocations] = useState<Record<string, string[]>>({});
  const [cities, setCities] = useState<string[]>([]);
  const [activeTab, setActiveTab] = useState<Tab>('predict');
  const [history, setHistory] = useState<HistoryRecord[]>(getInitialHistory);
  const [activeReportId, setActiveReportId] = useState<string | null>(null);
  const resultsRef = React.useRef<HTMLDivElement>(null);

  const [form, setForm] = useState({
    StreetAddress: '',
    City: 'Thành phố Hà Nội',
    District: 'Quận Hai Bà Trưng',
    Area: 50,
    Frontage: 4,
    Access_Road: 3,
    House_direction: 'Đông - Nam',
    Balcony_direction: 'Đông - Nam',
    Floors: 3,
    Bedrooms: 3,
    Bathrooms: 2,
    Legal_status: 'Sổ đỏ/ Sổ hồng',
    Furniture_state: 'Cơ bản'
  });

  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<any[]>([]);
  const [searchResults, setSearchResults] = useState<any[]>([]);
  const [searchLoading, setSearchLoading] = useState(false);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [resultId, setResultId] = useState<string | null>(null);
  const [isPanelOpen, setIsPanelOpen] = useState(false);
  const [submittedForm, setSubmittedForm] = useState<typeof form | null>(null);

  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark');
    window.localStorage.setItem('theme', theme);
  }, [theme]);

  useEffect(() => {
    window.localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(history.slice(0, 50)));
  }, [history]);

  useEffect(() => {
    // Process localProvinces JSON
    const locMap: Record<string, string[]> = {};
    const cityList: string[] = [];
    localProvinces.forEach((prov: any) => {
      cityList.push(prov.name);
      locMap[prov.name] = prov.districts.map((d: any) => d.name);
    });

    setLocations(locMap);
    setCities(cityList);
    if (cityList.length > 0) {
      const firstCity = cityList[0];
      setForm(f => ({ ...f, City: firstCity, District: locMap[firstCity]?.[0] || '' }));
    }
  }, []);

  const handleCityChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const city = e.target.value;
    setForm(f => ({
      ...f,
      City: city,
      District: locations[city]?.[0] || ''
    }));
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target;
    setForm(f => ({
      ...f,
      [name]: type === 'number' ? parseFloat(value) || 0 : value
    }));
  };

  const handlePredict = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setResults([]);
    setSearchResults([]);
    try {
      const res = await axios.post(`${API_BASE}/predict`, form);
      // Sort so the highest-R² (most accurate) model is first
      res.data.results.sort((a: any, b: any) => b.r2 - a.r2);
      setResults(res.data.results);
      setResultId(res.data.id);
      setSubmittedForm(form);

      // Auto trigger search using the most accurate model's price
      if (res.data.results.length > 0) {
        const bestPrice = pickBestModel(res.data.results).price;
        fetchRealHouses(bestPrice);

        const record: HistoryRecord = {
          id: crypto.randomUUID(),
          resultId: res.data.id,
          timestamp: new Date().toISOString(),
          form,
          results: res.data.results
        };
        setHistory(h => [record, ...h].slice(0, 50));
      }
      
      // Auto-scroll to results on mobile / layout wrap
      setTimeout(() => {
        resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 100);
    } catch (err) {
      console.error(err);
      alert("Lỗi khi dự đoán.");
    } finally {
      setLoading(false);
    }
  };

  const fetchRealHouses = async (price: number) => {
    setSearchLoading(true);
    try {
      const res = await axios.get(`${API_BASE}/search`, {
        params: {
          city: form.City,
          district: form.District,
          price: Math.round(price * 100) / 100,
          area: form.Area,
          floors: form.Floors,
          bedrooms: form.Bedrooms
        }
      });
      setSearchResults(res.data);
    } catch (err) {
      console.error(err);
    } finally {
      setSearchLoading(false);
    }
  };

  const bestModel = results.length > 0 ? pickBestModel(results) : null;
  const activeRecord = history.find(r => r.id === activeReportId) || null;

  const handleRestore = (record: HistoryRecord) => {
    setForm(record.form);
    setResults([...record.results].sort((a, b) => b.r2 - a.r2));
    setResultId(record.resultId);
    setSubmittedForm(record.form);
    setSearchResults([]);
    setActiveTab('predict');
  };

  const handleViewReport = (recordId: string) => {
    setActiveReportId(recordId);
    setActiveTab('report');
  };

  const handleExportReport = (appraiserName: string, agencyName: string, notes: string, detail: any) => {
    if (!activeRecord) return;
    const best = pickBestModel(activeRecord.results);
    const f = activeRecord.form;
    const lines = [
      `BÁO CÁO ĐỊNH GIÁ BẤT ĐỘNG SẢN`,
      `Đơn vị: ${agencyName}`,
      `Định giá viên: ${appraiserName}`,
      `Ngày: ${new Date(activeRecord.timestamp).toLocaleString()}`,
      ``,
      `--- THÔNG TIN BẤT ĐỘNG SẢN ---`,
      `Vị trí: ${f.District}, ${f.City}`,
      `Diện tích: ${f.Area} m² | Mặt tiền: ${f.Frontage} m | Đường vào: ${f.Access_Road} m`,
      `Số tầng: ${f.Floors} | Phòng ngủ: ${f.Bedrooms} | Phòng tắm: ${f.Bathrooms}`,
      `Pháp lý: ${f.Legal_status} | Nội thất: ${f.Furniture_state}`,
      ``,
      `--- KẾT QUẢ 6 MÔ HÌNH ---`,
      ...[...activeRecord.results].sort((a, b) => b.r2 - a.r2).map(r =>
        `${r.model}: ${r.price.toFixed(2)} tỷ (R²=${r.r2.toFixed(3)}, MAPE=${r.mape.toFixed(1)}%)`
      ),
      ``,
      `--- KẾT LUẬN ---`,
      `Giá đề xuất: ${best.price.toFixed(2)} tỷ VND (mô hình ${best.model}, độ tin cậy ${(best.confidence * 100).toFixed(1)}%)`,
      ``,
      `--- GHI CHÚ ĐỊNH GIÁ VIÊN ---`,
      notes,
    ];
    if (detail?.shap_values?.length) {
      lines.push('', '--- YẾU TỐ ẢNH HƯỞNG (SHAP) ---');
      detail.shap_values
        .filter((x: any) => x.feature !== 'Giá trị nền (Base Value)')
        .sort((a: any, b: any) => Math.abs(b.value) - Math.abs(a.value))
        .slice(0, 5)
        .forEach((x: any) => lines.push(`${x.feature}: ${x.value > 0 ? '+' : ''}${x.value.toFixed(2)} tỷ`));
    }
    const blob = new Blob([lines.join('\n')], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `bao-cao-dinh-gia-${activeRecord.id.slice(0, 8)}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="min-h-screen pb-20">
      <div
        className="app-hero relative overflow-hidden bg-gradient-to-br from-navy-700 via-navy-800 to-navy-900 dark:from-navy-900 dark:via-navy-950 dark:to-black pb-32 pt-12"
        style={{
          WebkitMaskImage: 'linear-gradient(to bottom, black 0%, black 55%, transparent 100%)',
          maskImage: 'linear-gradient(to bottom, black 0%, black 55%, transparent 100%)'
        }}
      >
        <div className="absolute inset-0 pointer-events-none overflow-hidden">
          <div className="absolute -top-32 -left-20 w-[28rem] h-[28rem] bg-brand-400/20 rounded-full blur-[100px]" />
          <div className="absolute -bottom-40 -right-16 w-[26rem] h-[26rem] bg-navy-300/10 rounded-full blur-[100px]" />
        </div>
        <div className="relative max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-start justify-between gap-4">
            <div>
              <span className="inline-flex items-center gap-2 text-brand-400 text-xs font-bold uppercase tracking-widest bg-white/5 border border-brand-400/30 rounded-full px-3 py-1 mb-4">
                <Home className="w-3.5 h-3.5" /> Định giá bất động sản bằng AI
              </span>
              <h1 className="text-4xl font-extrabold text-white tracking-tight flex items-center gap-3">
                AI Định Giá Bất Động Sản
              </h1>
              <p className="mt-4 text-navy-200 max-w-2xl text-lg">
                Sử dụng 6 mô hình Machine Learning tiên tiến để dự đoán giá nhà chính xác và tìm kiếm các bất động sản tương tự trên thị trường.
              </p>
            </div>
            <button
              type="button"
              onClick={() => setTheme(t => (t === 'dark' ? 'light' : 'dark'))}
              aria-label="Chuyển giao diện sáng/tối"
              title="Chuyển giao diện sáng/tối"
              className="shrink-0 p-2.5 rounded-full bg-white/10 hover:bg-white/20 border border-white/20 text-white transition-colors cursor-pointer"
            >
              {theme === 'dark' ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
            </button>
          </div>
        </div>
      </div>

      <main className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 -mt-16">
        {/* Navigation Bar - Responsive glassmorphic tabs selector */}
        <nav className="app-nav no-print mb-6 p-1.5 flex items-center gap-1.5 overflow-x-auto bg-white/70 dark:bg-navy-800/60 backdrop-blur-md border border-stone-200/50 dark:border-navy-700/50 rounded-2xl shadow-lg">
          {TABS.map(tab => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                type="button"
                onClick={() => setActiveTab(tab.id)}
                className={`shrink-0 inline-flex items-center gap-1.5 px-4 py-2.5 rounded-xl text-sm font-semibold transition-colors cursor-pointer ${
                  isActive
                    ? 'bg-brand-500 text-white shadow-sm'
                    : 'text-stone-600 dark:text-stone-300 hover:bg-stone-100 dark:hover:bg-navy-700/40'
                }`}
              >
                <Icon className="w-4 h-4" />
                {tab.label}
              </button>
            );
          })}
        </nav>

        {activeTab === 'history' && (
          <PredictionHistory history={history} onRestore={handleRestore} onViewReport={handleViewReport} />
        )}

        {activeTab === 'insights' && (
          <DatasetInsights apiBase={API_BASE} />
        )}

        {activeTab === 'models' && (
          <ModelExplainer />
        )}

        {activeTab === 'report' && (
          <PropertyReport activeRecord={activeRecord} apiBase={API_BASE} onExportTxt={handleExportReport} />
        )}

        {activeTab === 'predict' && (
        <>
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start mb-8">
          {/* Left: input form */}
          <div className="lg:col-span-5 lg:sticky lg:top-6">
            <div className="glass-card p-6 sm:p-8">
              <form onSubmit={handlePredict} className="space-y-6">
                <div className="space-y-6">

                  {/* Location */}
                  <div className="space-y-4">
                    <h3 className="text-lg font-semibold text-stone-800 dark:text-stone-100 flex items-center gap-2 border-b border-stone-200 dark:border-navy-700 pb-2">
                      <MapPin className="w-5 h-5 text-brand-500" /> Khu vực
                    </h3>
                    <div className="grid grid-cols-1 gap-4">
                      <div>
                        <label className="label-text">Địa chỉ (Số nhà, đường...)</label>
                        <input type="text" name="StreetAddress" value={form.StreetAddress} onChange={handleChange} className="input-field" placeholder="Ví dụ: 123 Đường Nguyễn Văn Linh" />
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-4 mt-4">
                      <div>
                        <label className="label-text">Tỉnh/Thành phố</label>
                        <select name="City" value={form.City} onChange={handleCityChange} className="input-field">
                          {cities.map(c => <option key={c} value={c}>{c}</option>)}
                        </select>
                      </div>
                      <div>
                        <label className="label-text">Quận/Huyện</label>
                        <select name="District" value={form.District} onChange={handleChange} className="input-field">
                          {locations[form.City]?.map(d => <option key={d} value={d}>{d}</option>)}
                        </select>
                      </div>
                    </div>
                  </div>

                  {/* Specs */}
                  <div className="space-y-4">
                    <h3 className="text-lg font-semibold text-stone-800 dark:text-stone-100 flex items-center gap-2 border-b border-stone-200 dark:border-navy-700 pb-2">
                      <Ruler className="w-5 h-5 text-brand-500" /> Kích thước & Số phòng
                    </h3>
                    <div className="grid grid-cols-3 gap-4">
                      <div>
                        <label className="label-text">Diện tích (m²)</label>
                        <input type="number" name="Area" value={form.Area} onChange={handleChange} className="input-field" step="0.1" />
                      </div>
                      <div>
                        <label className="label-text">Mặt tiền (m)</label>
                        <input type="number" name="Frontage" value={form.Frontage} onChange={handleChange} className="input-field" step="0.1" />
                      </div>
                      <div>
                        <label className="label-text">Ngõ vào (m)</label>
                        <input type="number" name="Access_Road" value={form.Access_Road} onChange={handleChange} className="input-field" step="0.1" />
                      </div>
                      <div>
                        <label className="label-text">Số tầng</label>
                        <QuantityStepper value={form.Floors} onChange={(v) => setForm(f => ({ ...f, Floors: v }))} min={1} max={20} />
                      </div>
                      <div>
                        <label className="label-text">Phòng ngủ</label>
                        <QuantityStepper value={form.Bedrooms} onChange={(v) => setForm(f => ({ ...f, Bedrooms: v }))} min={1} max={15} />
                      </div>
                      <div>
                        <label className="label-text">Phòng tắm</label>
                        <QuantityStepper value={form.Bathrooms} onChange={(v) => setForm(f => ({ ...f, Bathrooms: v }))} min={1} max={15} />
                      </div>
                    </div>
                  </div>

                  {/* Other Specs */}
                  <div className="space-y-4">
                    <h3 className="text-lg font-semibold text-stone-800 dark:text-stone-100 flex items-center gap-2 border-b border-stone-200 dark:border-navy-700 pb-2">
                      <Compass className="w-5 h-5 text-brand-500" /> Thông tin khác
                    </h3>
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <label className="label-text">Hướng nhà</label>
                        <select name="House_direction" value={form.House_direction} onChange={handleChange} className="input-field">
                          <option value="Đông">Đông</option>
                          <option value="Tây">Tây</option>
                          <option value="Nam">Nam</option>
                          <option value="Bắc">Bắc</option>
                          <option value="Đông - Nam">Đông - Nam</option>
                          <option value="Tây - Nam">Tây - Nam</option>
                          <option value="Đông - Bắc">Đông - Bắc</option>
                          <option value="Tây - Bắc">Tây - Bắc</option>
                        </select>
                      </div>
                      <div>
                        <label className="label-text">Hướng ban công</label>
                        <select name="Balcony_direction" value={form.Balcony_direction} onChange={handleChange} className="input-field">
                          <option value="Đông">Đông</option>
                          <option value="Tây">Tây</option>
                          <option value="Nam">Nam</option>
                          <option value="Bắc">Bắc</option>
                          <option value="Đông - Nam">Đông - Nam</option>
                          <option value="Tây - Nam">Tây - Nam</option>
                          <option value="Đông - Bắc">Đông - Bắc</option>
                          <option value="Tây - Bắc">Tây - Bắc</option>
                        </select>
                      </div>
                      <div>
                        <label className="label-text">Pháp lý</label>
                        <select name="Legal_status" value={form.Legal_status} onChange={handleChange} className="input-field">
                          <option value="Have certificate">Đã có sổ đỏ/sổ hồng</option>
                          <option value="Waiting for certificate">Đang chờ sổ</option>
                          <option value="Sale contract">Hợp đồng mua bán</option>
                        </select>
                      </div>
                      <div>
                        <label className="label-text">Nội thất</label>
                        <select name="Furniture_state" value={form.Furniture_state} onChange={handleChange} className="input-field">
                          <option value="Full">Đầy đủ</option>
                          <option value="Basic">Cơ bản</option>
                          <option value="None">Không có</option>
                        </select>
                      </div>
                    </div>
                  </div>

                </div>

                <div className="pt-4">
                  <button
                    type="submit"
                    disabled={loading}
                    className="w-full px-8 py-4 bg-brand-600 hover:bg-brand-700 text-white rounded-xl font-bold text-lg shadow-lg hover:shadow-xl transition-all disabled:opacity-50 flex items-center justify-center gap-2"
                  >
                    {loading ? <span className="animate-spin rounded-full h-5 w-5 border-2 border-white border-t-transparent"></span> : <Building className="w-6 h-6" />}
                    Dự đoán giá trị
                  </button>
                </div>
              </form>
            </div>
          </div>

          {/* Right: results + ranking board */}
          <div ref={resultsRef} className="lg:col-span-7">
            {loading ? (
              <div className="glass-card p-16 flex flex-col items-center justify-center gap-3 text-stone-500 dark:text-stone-400">
                <div className="animate-spin rounded-full h-10 w-10 border-4 border-brand-500 border-t-transparent" />
                Đang chạy 6 mô hình dự đoán...
              </div>
            ) : results.length > 0 && bestModel ? (
              <div>
                <h2 className="text-2xl font-bold text-stone-800 dark:text-stone-100 mb-6 flex items-center gap-2">
                  <CheckCircle2 className="w-6 h-6 text-green-500" />
                  Kết quả dự đoán từ 6 mô hình
                </h2>

                {/* Featured: most accurate model */}
                <div
                  onClick={() => { setSelectedModel(bestModel.model); setIsPanelOpen(true); }}
                  className="group glass-card p-6 sm:p-8 mb-4 cursor-pointer ring-1 ring-brand-400/30 transition-shadow relative overflow-hidden"
                  style={{ boxShadow: '0 0 50px -15px rgba(194,138,42,0.35)' }}
                >
                  <div className="absolute -right-10 -top-10 w-40 h-40 bg-brand-400/10 rounded-full blur-2xl pointer-events-none" />
                  <div className="relative flex flex-col sm:flex-row sm:items-end sm:justify-between gap-6">
                    <div>
                      <div className="inline-flex items-center gap-1.5 text-xs font-bold text-brand-700 dark:text-brand-400 bg-brand-50 dark:bg-brand-900/20 border border-brand-200 dark:border-brand-500/30 rounded-full px-3 py-1 mb-3">
                        <Crown className="w-3.5 h-3.5" /> Mô hình chính xác nhất (R² cao nhất)
                      </div>
                      <div className="text-sm font-semibold text-stone-500 dark:text-stone-400 uppercase tracking-wider">{bestModel.model.replace('_', ' ')}</div>
                      <div className="text-4xl sm:text-5xl font-extrabold text-navy-800 dark:text-white mt-1">{bestModel.price.toFixed(2)} tỷ</div>
                    </div>
                    <div className="flex gap-6 shrink-0">
                      <div className="text-center">
                        <Percent className="w-4 h-4 text-brand-500 mx-auto mb-1" />
                        <div className="font-bold text-stone-800 dark:text-stone-100">{(bestModel.confidence * 100).toFixed(1)}%</div>
                        <div className="text-[11px] text-stone-400 dark:text-stone-500">Tin cậy</div>
                      </div>
                      <div className="text-center">
                        <TrendingUp className="w-4 h-4 text-brand-500 mx-auto mb-1" />
                        <div className="font-bold text-stone-800 dark:text-stone-100">{bestModel.r2.toFixed(3)}</div>
                        <div className="text-[11px] text-stone-400 dark:text-stone-500">R²</div>
                      </div>
                      <div className="text-center">
                        <Target className="w-4 h-4 text-brand-500 mx-auto mb-1" />
                        <div className="font-bold text-stone-800 dark:text-stone-100">{bestModel.mape.toFixed(1)}%</div>
                        <div className="text-[11px] text-stone-400 dark:text-stone-500">MAPE</div>
                      </div>
                    </div>
                  </div>
                  <div className="relative mt-5 pt-4 border-t border-stone-100 dark:border-navy-700 flex items-center justify-between gap-3">
                    <span className="text-sm text-stone-400 dark:text-stone-500">Nhấn để xem chi tiết</span>
                    <span className="inline-flex items-center gap-1.5 text-sm font-semibold text-white bg-brand-600 group-hover:bg-brand-700 rounded-full pl-4 pr-3 py-2 transition-colors">
                      Xem phân tích SHAP <ChevronRight className="w-4 h-4" />
                    </span>
                  </div>
                </div>

                {/* Leaderboard: remaining models ranked by R² */}
                <div className="glass-card divide-y divide-stone-100 dark:divide-navy-700 overflow-hidden">
                  {results.map((r, i) => (
                    <div
                      key={i}
                      onClick={() => { setSelectedModel(r.model); setIsPanelOpen(true); }}
                      className="group flex items-center gap-4 p-4 cursor-pointer hover:bg-stone-50 dark:hover:bg-navy-800/60 transition-colors"
                    >
                      <div className={`w-7 h-7 shrink-0 rounded-full flex items-center justify-center text-xs font-bold ${i === 0 ? 'bg-brand-500 text-white' : 'bg-stone-100 dark:bg-navy-700 text-stone-500 dark:text-stone-400'}`}>
                        {i + 1}
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="font-semibold text-stone-800 dark:text-stone-100 uppercase text-sm tracking-wide truncate">{r.model.replace('_', ' ')}</div>
                        <div className="text-xs text-stone-400 dark:text-stone-500">R²: {r.r2.toFixed(4)} · MAPE: {r.mape.toFixed(1)}% · Tin cậy: {(r.confidence * 100).toFixed(1)}%</div>
                      </div>
                      <div className="text-right shrink-0">
                        <div className="font-bold text-stone-800 dark:text-stone-100">{r.price.toFixed(2)} tỷ</div>
                      </div>
                      <ChevronRight className="w-4 h-4 text-stone-300 dark:text-stone-600 shrink-0 group-hover:translate-x-1 group-hover:text-brand-500 transition-all" />
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="glass-card p-16 flex flex-col items-center justify-center gap-4 text-center h-full">
                <div className="w-16 h-16 rounded-full bg-brand-50 dark:bg-brand-900/20 text-brand-500 flex items-center justify-center">
                  <Home className="w-8 h-8" />
                </div>
                <div>
                  <h3 className="text-lg font-bold text-stone-700 dark:text-stone-200 mb-1.5">Chưa có kết quả dự đoán</h3>
                  <p className="text-sm text-stone-500 dark:text-stone-400 max-w-sm">Điền thông tin bất động sản bên trái rồi nhấn "Dự đoán giá trị" để xem kết quả từ 6 mô hình Machine Learning.</p>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* What-if Simulator */}
        {bestModel && submittedForm && (
          <div className="mb-8">
            <WhatIfSimulator
              baseForm={submittedForm}
              basePrice={bestModel.price}
              baseModelName={bestModel.model}
              apiBase={API_BASE}
            />
          </div>
        )}

        {/* Real Houses Search */}
        {(searchLoading || results.length > 0) && (
          <div className="mt-8">
            <h2 className="text-2xl font-bold text-stone-800 dark:text-stone-100 mb-6 flex items-center gap-2">
              <Search className="w-6 h-6 text-brand-500" />
              Nhà đất thực tế đang bán
            </h2>
            {searchLoading ? (
              <div className="flex justify-center p-12">
                <div className="animate-spin rounded-full h-12 w-12 border-4 border-brand-500 border-t-transparent"></div>
              </div>
            ) : searchResults.length === 0 ? (
              <div className="bg-stone-50 dark:bg-navy-800/40 border border-stone-200 dark:border-navy-700 rounded-2xl p-12 text-center">
                <div className="w-16 h-16 bg-stone-200 dark:bg-navy-700 rounded-full flex items-center justify-center mx-auto mb-4">
                  <Search className="w-8 h-8 text-stone-400 dark:text-stone-500" />
                </div>
                <h3 className="text-lg font-bold text-stone-700 dark:text-stone-200 mb-2">Không tìm thấy bất động sản phù hợp</h3>
                <p className="text-stone-500 dark:text-stone-400">Hệ thống đã thử tìm trên các trang batdongsan.com.vn, bds68 nhưng hiện không có tin đăng nào khớp hoàn toàn với vị trí và mức giá dự đoán. Vui lòng thử lại với diện tích hoặc vị trí khác.</p>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {searchResults.map((item, idx) => (
                  <a key={idx} href={item.link} onClick={(e) => { e.preventDefault(); Browser.open({ url: item.link }).catch(() => window.open(item.link, '_blank')); }} className="group glass-card overflow-hidden hover:shadow-2xl transition-all flex flex-col">
                    <div className="h-48 overflow-hidden bg-stone-200 dark:bg-navy-700 relative">
                      {item.image && item.image !== "No Images" ? (
                        <img
                          src={item.image}
                          alt="House"
                          className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-500"
                          onError={(e) => {
                            e.currentTarget.style.display = 'none';
                            if (e.currentTarget.nextElementSibling) {
                              (e.currentTarget.nextElementSibling as HTMLElement).style.display = 'flex';
                            }
                          }}
                        />
                      ) : null}
                      <div className="absolute inset-0 bg-stone-200 dark:bg-navy-700 text-stone-500 dark:text-stone-400 flex items-center justify-center font-medium" style={{ display: (!item.image || item.image === "No Images") ? 'flex' : 'none' }}>
                        No Images
                      </div>
                      <div className="absolute bottom-2 right-2 bg-black/60 text-white text-xs px-2 py-1 rounded backdrop-blur-sm z-10">Xem chi tiết</div>
                    </div>
                    <div className="p-5 flex-1 flex flex-col">
                      <h3 className="font-semibold text-stone-800 dark:text-stone-100 line-clamp-2 mb-2 group-hover:text-brand-600 dark:group-hover:text-brand-400 transition-colors">{item.title}</h3>
                      <p className="text-sm text-stone-500 dark:text-stone-400 line-clamp-3 mb-4 flex-1">{item.snippet}</p>
                      <div className="mt-auto flex items-center text-brand-600 dark:text-brand-400 font-medium text-sm">
                        Truy cập trang <ChevronRight className="w-4 h-4 ml-1 group-hover:translate-x-1 transition-transform" />
                      </div>
                    </div>
                  </a>
                ))}
              </div>
            )}
          </div>
        )}
        </>
        )}
      </main>

      <ModelDetailsPanel
        isOpen={isPanelOpen}
        onClose={() => setIsPanelOpen(false)}
        model={selectedModel}
        resultId={resultId}
        apiUrl={API_BASE.replace('/api', '')}
      />
    </div>
  );
}

export default App;
