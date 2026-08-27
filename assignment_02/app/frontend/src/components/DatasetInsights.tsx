import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter, ZAxis } from 'recharts';
import { Database, Layers, PieChart, AlertTriangle, Loader2 } from 'lucide-react';

interface BucketEntry { label: string; avgPrice: number; count: number }
interface HistogramEntry { range: string; count: number }
interface DatasetStats {
  totalListings: number;
  featureDimensions: number;
  districtCount: number;
  priceRange: { min: number; max: number };
  missingPct: Record<string, number>;
  priceHistogram: HistogramEntry[];
  topDistricts: BucketEntry[];
  byLegalStatus: BucketEntry[];
  byFurnitureState: BucketEntry[];
  areaPriceSample: { area: number; price: number }[];
}

interface DatasetInsightsProps {
  apiBase: string;
}

export const DatasetInsights: React.FC<DatasetInsightsProps> = ({ apiBase }) => {
  const [stats, setStats] = useState<DatasetStats | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    axios.get<DatasetStats>(`${apiBase}/dataset-stats`)
      .then(res => setStats(res.data))
      .catch(err => console.error(err))
      .finally(() => setLoading(false));
  }, [apiBase]);

  if (loading) {
    return (
      <div className="glass-card p-16 flex flex-col items-center justify-center gap-3 text-stone-500 dark:text-stone-400">
        <Loader2 className="w-8 h-8 animate-spin text-brand-500" />
        Đang tải thống kê dữ liệu thực tế...
      </div>
    );
  }

  if (!stats) {
    return (
      <div className="glass-card p-16 text-center text-stone-500 dark:text-stone-400">
        Không thể tải thống kê dữ liệu.
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Overview */}
      <div className="glass-card p-6">
        <h2 className="text-2xl font-bold text-stone-800 dark:text-stone-100 flex items-center gap-2">
          <Database className="w-6 h-6 text-brand-500" />
          Khám phá bộ dữ liệu bất động sản Việt Nam
        </h2>
        <p className="mt-2 text-sm text-stone-500 dark:text-stone-400 max-w-2xl">
          Thống kê được tính trực tiếp từ {stats.totalListings.toLocaleString('vi-VN')} tin đăng thực tế dùng để huấn luyện mô hình — không phải dữ liệu minh họa.
        </p>

        <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="rounded-xl border border-stone-200 dark:border-navy-700 bg-white dark:bg-navy-800/60 p-4 shadow-sm">
            <div className="text-xs font-semibold text-stone-500 dark:text-stone-400 uppercase">Số tin đăng</div>
            <div className="mt-1 text-xl font-bold text-stone-800 dark:text-stone-100">{stats.totalListings.toLocaleString('vi-VN')}</div>
          </div>
          <div className="rounded-xl border border-stone-200 dark:border-navy-700 bg-white dark:bg-navy-800/60 p-4 shadow-sm">
            <div className="text-xs font-semibold text-stone-500 dark:text-stone-400 uppercase">Số chiều đặc trưng</div>
            <div className="mt-1 text-xl font-bold text-stone-800 dark:text-stone-100">{stats.featureDimensions}</div>
          </div>
          <div className="rounded-xl border border-stone-200 dark:border-navy-700 bg-white dark:bg-navy-800/60 p-4 shadow-sm">
            <div className="text-xs font-semibold text-stone-500 dark:text-stone-400 uppercase">Số quận/huyện</div>
            <div className="mt-1 text-xl font-bold text-stone-800 dark:text-stone-100">{stats.districtCount}</div>
          </div>
          <div className="rounded-xl border border-stone-200 dark:border-navy-700 bg-white dark:bg-navy-800/60 p-4 shadow-sm">
            <div className="text-xs font-semibold text-stone-500 dark:text-stone-400 uppercase">Khoảng giá</div>
            <div className="mt-1 text-xl font-bold text-brand-600 dark:text-brand-400">{stats.priceRange.min} - {stats.priceRange.max} tỷ</div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Price histogram */}
        <div className="glass-card p-5 lg:col-span-6 space-y-4">
          <div>
            <h3 className="text-base font-bold text-stone-800 dark:text-stone-100 flex items-center gap-1.5 mb-2">
              <PieChart className="w-5 h-5 text-brand-500" />
              Phân phối giá nhà (tỷ VND)
            </h3>
            <p className="text-xs text-stone-500 dark:text-stone-400">Số lượng tin đăng theo từng khoảng giá.</p>
          </div>
          <div className="h-[260px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={stats.priceHistogram} margin={{ top: 10, right: 10, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--tw-color-stone-200, #e7e5e4)" />
                <XAxis dataKey="range" fontSize={11} tickLine={false} />
                <YAxis fontSize={11} width={40} tickLine={false} />
                <Tooltip contentStyle={{ borderRadius: '12px' }} />
                <Bar dataKey="count" name="Số tin đăng" fill="#c28a2a" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Top districts */}
        <div className="glass-card p-5 lg:col-span-6 space-y-4">
          <div>
            <h3 className="text-base font-bold text-stone-800 dark:text-stone-100 flex items-center gap-1.5 mb-2">
              <Layers className="w-5 h-5 text-brand-500" />
              Top 10 quận/huyện nhiều tin đăng nhất
            </h3>
            <p className="text-xs text-stone-500 dark:text-stone-400">Giá trung bình (tỷ VND) theo từng khu vực.</p>
          </div>
          <div className="h-[260px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={stats.topDistricts} layout="vertical" margin={{ top: 5, right: 25, left: 10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                <XAxis type="number" fontSize={11} tickLine={false} />
                <YAxis dataKey="label" type="category" width={90} fontSize={10} tickLine={false} axisLine={false} />
                <Tooltip contentStyle={{ borderRadius: '12px' }} formatter={(v: any) => [`${v} tỷ`, 'Giá TB']} />
                <Bar dataKey="avgPrice" name="Giá trung bình" fill="#4a5f7d" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* By legal status / furniture */}
        <div className="glass-card p-5 lg:col-span-6 space-y-4">
          <div>
            <h3 className="text-base font-bold text-stone-800 dark:text-stone-100 flex items-center gap-1.5 mb-2">
              <Layers className="w-5 h-5 text-brand-500" />
              Giá trung bình theo pháp lý &amp; nội thất
            </h3>
            <p className="text-xs text-stone-500 dark:text-stone-400">So sánh giữa các trạng thái pháp lý và mức độ nội thất.</p>
          </div>
          <div className="h-[260px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={[...stats.byLegalStatus, ...stats.byFurnitureState]} margin={{ top: 10, right: 10, left: 0, bottom: 30 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="label" fontSize={9} tickLine={false} angle={-25} textAnchor="end" interval={0} />
                <YAxis fontSize={11} width={36} tickLine={false} />
                <Tooltip contentStyle={{ borderRadius: '12px' }} formatter={(v: any) => [`${v} tỷ`, 'Giá TB']} />
                <Bar dataKey="avgPrice" name="Giá trung bình" fill="#c28a2a" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Area vs Price scatter */}
        <div className="glass-card p-5 lg:col-span-6 space-y-4">
          <div>
            <h3 className="text-base font-bold text-stone-800 dark:text-stone-100 flex items-center gap-1.5 mb-2">
              <Database className="w-5 h-5 text-brand-500" />
              Diện tích vs. Giá
            </h3>
            <p className="text-xs text-stone-500 dark:text-stone-400">Mẫu ngẫu nhiên 200 tin đăng từ bộ dữ liệu.</p>
          </div>
          <div className="h-[260px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 15, right: 15, bottom: 5, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" dataKey="area" name="Diện tích" unit=" m²" fontSize={10} tickLine={false} domain={['dataMin - 5', 'dataMax + 5']} />
                <YAxis type="number" dataKey="price" name="Giá" unit=" tỷ" fontSize={10} tickLine={false} width={44} domain={['dataMin - 1', 'dataMax + 1']} />
                <ZAxis type="number" range={[40, 40]} />
                <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ borderRadius: '12px', fontSize: 11 }} />
                <Legend verticalAlign="top" height={28} wrapperStyle={{ fontSize: 11 }} />
                <Scatter name="Tin đăng" data={stats.areaPriceSample} fill="#c28a2a" shape="circle" />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Missingness */}
        <div className="glass-card p-5 lg:col-span-12 space-y-4">
          <div>
            <h3 className="text-base font-bold text-stone-800 dark:text-stone-100 flex items-center gap-1.5 mb-2">
              <AlertTriangle className="w-5 h-5 text-brand-500" />
              Tỷ lệ dữ liệu thiếu theo cột
            </h3>
            <p className="text-xs text-stone-500 dark:text-stone-400">
              Nhiều thuộc tính bị thiếu đáng kể trong dữ liệu gốc — được xử lý bằng trung vị (số) hoặc "Unknown" (phân loại) trước khi huấn luyện.
            </p>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
            {Object.entries(stats.missingPct).filter(([, v]) => v > 0).sort((a, b) => b[1] - a[1]).map(([col, pct]) => (
              <div key={col} className="rounded-xl border border-stone-200 dark:border-navy-700 bg-white dark:bg-navy-800/60 p-3">
                <div className="text-[10px] font-semibold text-stone-500 dark:text-stone-400 uppercase truncate" title={col}>{col}</div>
                <div className="mt-1 h-1.5 w-full bg-stone-100 dark:bg-navy-900/60 rounded-full overflow-hidden">
                  <div className="h-full bg-brand-500 rounded-full" style={{ width: `${pct}%` }} />
                </div>
                <div className="mt-1 text-sm font-bold text-stone-800 dark:text-stone-100">{pct}%</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
