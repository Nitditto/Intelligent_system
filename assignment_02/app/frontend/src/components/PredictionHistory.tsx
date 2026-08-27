import React from 'react';
import { History, RefreshCw, FileText, ChevronRight, Calendar, Crown } from 'lucide-react';
import { pickBestModel } from '../utils';
import type { HistoryRecord } from '../App';

interface PredictionHistoryProps {
  history: HistoryRecord[];
  onRestore: (record: HistoryRecord) => void;
  onViewReport: (recordId: string) => void;
}

export const PredictionHistory: React.FC<PredictionHistoryProps> = ({ history, onRestore, onViewReport }) => {
  return (
    <div className="space-y-6">
      <div className="glass-card p-6">
        <h2 className="text-2xl font-bold text-stone-800 dark:text-stone-100 flex items-center gap-2">
          <History className="w-6 h-6 text-brand-500" />
          Lịch sử dự đoán
        </h2>
        <p className="mt-2 text-sm text-stone-500 dark:text-stone-400 max-w-2xl">
          Nhật ký các bất động sản đã được định giá trong phiên làm việc này. Khôi phục lại thông tin vào biểu mẫu hoặc xem báo cáo chi tiết.
        </p>
      </div>

      {history.length === 0 ? (
        <div className="glass-card p-12 text-center flex flex-col items-center justify-center gap-4">
          <div className="w-12 h-12 rounded-full bg-brand-50 dark:bg-brand-900/20 text-brand-500 flex items-center justify-center">
            <Calendar className="w-6 h-6" />
          </div>
          <div>
            <h3 className="text-base font-bold text-stone-800 dark:text-stone-100">Chưa có lịch sử</h3>
            <p className="text-xs text-stone-500 dark:text-stone-400 mt-1 max-w-xs">
              Thực hiện một dự đoán ở tab "Dự đoán giá" để bắt đầu ghi nhận lịch sử.
            </p>
          </div>
        </div>
      ) : (
        <div className="glass-card divide-y divide-stone-100 dark:divide-navy-700 overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse text-xs">
              <thead>
                <tr className="bg-stone-50 dark:bg-navy-900/40 text-stone-400 dark:text-stone-500 font-bold uppercase tracking-wider">
                  <th className="p-4">Thời gian</th>
                  <th className="p-4">Vị trí &amp; thông số</th>
                  <th className="p-4">Mô hình tốt nhất</th>
                  <th className="p-4 text-right">Thao tác</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-stone-100 dark:divide-navy-700">
                {history.map((record) => {
                  const best = pickBestModel(record.results);
                  const dateStr = new Date(record.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
                  const dateDay = new Date(record.timestamp).toLocaleDateString([], { month: 'short', day: 'numeric' });

                  return (
                    <tr key={record.id} className="hover:bg-stone-50 dark:hover:bg-navy-800/60 transition-colors">
                      <td className="p-4 whitespace-nowrap">
                        <div className="font-semibold text-stone-800 dark:text-stone-100">{dateStr}</div>
                        <div className="text-[10px] text-stone-400 dark:text-stone-500 font-medium mt-0.5">{dateDay}</div>
                      </td>
                      <td className="p-4">
                        <div className="font-semibold text-stone-800 dark:text-stone-100 truncate max-w-[220px]">{record.form.District}, {record.form.City}</div>
                        <div className="flex flex-wrap gap-2 text-[10px] font-semibold mt-1">
                          <span className="bg-stone-100 dark:bg-navy-900/60 px-2 py-0.5 rounded border border-stone-200 dark:border-navy-700 text-stone-600 dark:text-stone-300">
                            {record.form.Area} m²
                          </span>
                          <span className="bg-stone-100 dark:bg-navy-900/60 px-2 py-0.5 rounded border border-stone-200 dark:border-navy-700 text-stone-600 dark:text-stone-300">
                            {record.form.Bedrooms} PN
                          </span>
                          <span className="bg-stone-100 dark:bg-navy-900/60 px-2 py-0.5 rounded border border-stone-200 dark:border-navy-700 text-stone-600 dark:text-stone-300">
                            {record.form.Floors} tầng
                          </span>
                        </div>
                      </td>
                      <td className="p-4 whitespace-nowrap">
                        <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full font-bold text-[10px] bg-brand-50 dark:bg-brand-900/20 text-brand-700 dark:text-brand-400 border border-brand-200 dark:border-brand-500/30">
                          <Crown className="w-3 h-3" />
                          {best.price.toFixed(2)} tỷ ({best.model.replace('_', ' ')})
                        </span>
                      </td>
                      <td className="p-4 text-right space-x-2 whitespace-nowrap">
                        <button
                          onClick={() => onRestore(record)}
                          className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg border border-stone-200 dark:border-navy-600 bg-white dark:bg-navy-900/60 text-stone-700 dark:text-stone-200 hover:border-brand-500 hover:text-brand-600 font-bold transition-colors cursor-pointer"
                          title="Khôi phục thông tin vào biểu mẫu"
                        >
                          <RefreshCw className="w-3.5 h-3.5" />
                          Khôi phục
                        </button>
                        <button
                          onClick={() => onViewReport(record.id)}
                          className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg border border-transparent bg-brand-600 text-white hover:bg-brand-700 font-bold transition-colors cursor-pointer"
                          title="Mở báo cáo định giá chi tiết"
                        >
                          <FileText className="w-3.5 h-3.5" />
                          Báo cáo
                          <ChevronRight className="w-3 h-3" />
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};
