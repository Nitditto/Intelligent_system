import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { FileText, Printer, FileDown, Building2, MapPin, User, Clipboard, Loader2 } from 'lucide-react';
import { pickBestModel } from '../utils';
import type { HistoryRecord } from '../App';

interface SHAPFeature {
  feature: string;
  value: number;
}

interface DetailedResult {
  model: string;
  prediction: number;
  confidence: number;
  r2: number;
  mape: number;
  base_value: number;
  shap_values: SHAPFeature[];
}

interface PropertyReportProps {
  activeRecord: HistoryRecord | null;
  apiBase: string;
  onExportTxt: (appraiserName: string, agencyName: string, notes: string, detail: DetailedResult | null) => void;
}

const formatModelName = (name: string) =>
  name.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');

export const PropertyReport: React.FC<PropertyReportProps> = ({ activeRecord, apiBase, onExportTxt }) => {
  const [appraiserName, setAppraiserName] = useState('Nguyễn Văn An');
  const [agencyName, setAgencyName] = useState('Sàn Giao Dịch BĐS Kim Cương');
  const [notes, setNotes] = useState(
    'Bất động sản có vị trí thuận lợi, phù hợp nhu cầu an cư. Đề xuất tham khảo thêm các tin đăng thực tế cùng khu vực trước khi ra quyết định giao dịch.'
  );
  const [detail, setDetail] = useState<DetailedResult | null>(null);
  const [loading, setLoading] = useState(false);

  const best = activeRecord ? pickBestModel(activeRecord.results) : null;

  useEffect(() => {
    setDetail(null);
    if (activeRecord && best) {
      setLoading(true);
      axios.get<DetailedResult>(`${apiBase}/result/${activeRecord.resultId}?model=${best.model}`)
        .then(res => setDetail(res.data))
        .catch(err => console.error(err))
        .finally(() => setLoading(false));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeRecord?.id]);

  if (!activeRecord || !best) {
    return (
      <div className="glass-card p-12 text-center flex flex-col items-center justify-center gap-4">
        <div className="w-12 h-12 rounded-full bg-red-50 dark:bg-red-500/10 text-red-500 flex items-center justify-center">
          <FileText className="w-6 h-6" />
        </div>
        <div>
          <h3 className="text-base font-bold text-stone-800 dark:text-stone-100">Chưa có báo cáo nào được chọn</h3>
          <p className="text-xs text-stone-500 dark:text-stone-400 mt-1 max-w-xs mx-auto">
            Hãy thực hiện dự đoán ở tab "Dự đoán giá" hoặc chọn một bản ghi từ "Lịch sử" để mở báo cáo.
          </p>
        </div>
      </div>
    );
  }

  const f = activeRecord.form;
  const featureItems = (detail?.shap_values || []).filter(x => x.feature !== 'Giá trị nền (Base Value)');

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
      {/* Controls */}
      <div className="lg:col-span-4 glass-card p-5 space-y-5 no-print">
        <div className="border-b border-stone-100 dark:border-navy-700 pb-3">
          <h3 className="text-sm font-bold text-stone-800 dark:text-stone-100 flex items-center gap-1.5">
            <Clipboard className="w-4 h-4 text-brand-500" />
            Tùy chỉnh báo cáo
          </h3>
          <p className="text-xs text-stone-500 dark:text-stone-400 mt-0.5">Điền thông tin định giá viên trước khi in/xuất</p>
        </div>

        <div className="space-y-4 text-xs font-semibold">
          <div className="flex flex-col gap-1.5">
            <label className="text-stone-500 dark:text-stone-400 uppercase">Định giá viên</label>
            <div className="relative">
              <User className="absolute left-3.5 top-3 w-4 h-4 text-stone-400" />
              <input
                type="text"
                value={appraiserName}
                onChange={(e) => setAppraiserName(e.target.value)}
                className="input-field w-full pl-10"
              />
            </div>
          </div>

          <div className="flex flex-col gap-1.5">
            <label className="text-stone-500 dark:text-stone-400 uppercase">Đơn vị / Sàn giao dịch</label>
            <div className="relative">
              <MapPin className="absolute left-3.5 top-3 w-4 h-4 text-stone-400" />
              <input
                type="text"
                value={agencyName}
                onChange={(e) => setAgencyName(e.target.value)}
                className="input-field w-full pl-10"
              />
            </div>
          </div>

          <div className="flex flex-col gap-1.5">
            <label className="text-stone-500 dark:text-stone-400 uppercase">Ghi chú &amp; khuyến nghị</label>
            <textarea
              rows={5}
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              className="input-field w-full resize-none"
            />
          </div>
        </div>

        <div className="grid grid-cols-2 gap-3 pt-2">
          <button
            onClick={() => window.print()}
            className="bg-brand-600 hover:bg-brand-700 text-white font-bold py-3 rounded-xl flex items-center justify-center gap-1.5 text-xs shadow-md transition-colors cursor-pointer"
          >
            <Printer className="w-4 h-4" />
            In / PDF
          </button>
          <button
            onClick={() => onExportTxt(appraiserName, agencyName, notes, detail)}
            className="border border-stone-200 dark:border-navy-600 text-stone-700 dark:text-stone-200 font-bold py-3 rounded-xl flex items-center justify-center gap-1.5 text-xs hover:border-brand-500 hover:text-brand-600 transition-colors cursor-pointer"
          >
            <FileDown className="w-4 h-4" />
            Xuất TXT
          </button>
        </div>
      </div>

      {/* Printable report sheet */}
      <div className="lg:col-span-8 print-container rounded-2xl border border-stone-200 dark:border-navy-700 bg-white p-8 text-stone-800 shadow-xl space-y-8 min-h-[29.7cm] flex flex-col justify-between">
        <div className="space-y-6">
          <div className="flex justify-between items-start border-b-2 border-navy-800 pb-5">
            <div className="flex items-center gap-2">
              <div className="h-10 w-10 bg-navy-800 rounded-xl flex items-center justify-center text-white shrink-0 shadow-md">
                <Building2 className="w-5 h-5" />
              </div>
              <div>
                <h2 className="text-base font-black text-navy-900 uppercase tracking-tight">Báo cáo định giá bất động sản</h2>
                <p className="text-[10px] font-bold text-stone-500 tracking-wider uppercase mt-0.5">{agencyName}</p>
              </div>
            </div>
            <div className="text-right text-[10px] text-stone-500 font-bold uppercase tracking-wider space-y-0.5">
              <div>Mã hồ sơ: {activeRecord.id.slice(0, 8)}</div>
              <div>Ngày: {new Date(activeRecord.timestamp).toLocaleDateString()}</div>
              <div>Giờ: {new Date(activeRecord.timestamp).toLocaleTimeString()}</div>
            </div>
          </div>

          <div className="text-center pt-2">
            <h1 className="text-lg font-black text-navy-900 uppercase tracking-wider">Định giá bằng Trí tuệ nhân tạo</h1>
            <p className="text-[10px] font-semibold text-stone-500 mt-1 uppercase">6 mô hình Machine Learning độc lập</p>
          </div>

          <div className="space-y-3">
            <h3 className="text-xs font-bold uppercase tracking-wider text-navy-900 border-b border-stone-200 pb-1.5">
              1. Thông tin bất động sản
            </h3>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs">
              <div className="p-3 bg-stone-50 rounded-xl border border-stone-200">
                <span className="text-[9px] font-bold text-stone-500 block uppercase">Vị trí</span>
                <span className="font-bold text-navy-900 mt-0.5 block truncate">{f.District}</span>
              </div>
              <div className="p-3 bg-stone-50 rounded-xl border border-stone-200">
                <span className="text-[9px] font-bold text-stone-500 block uppercase">Diện tích</span>
                <span className="font-bold text-navy-900 mt-0.5 block">{f.Area} m²</span>
              </div>
              <div className="p-3 bg-stone-50 rounded-xl border border-stone-200">
                <span className="text-[9px] font-bold text-stone-500 block uppercase">Mặt tiền</span>
                <span className="font-bold text-navy-900 mt-0.5 block">{f.Frontage} m</span>
              </div>
              <div className="p-3 bg-stone-50 rounded-xl border border-stone-200">
                <span className="text-[9px] font-bold text-stone-500 block uppercase">Số tầng</span>
                <span className="font-bold text-navy-900 mt-0.5 block">{f.Floors}</span>
              </div>
              <div className="p-3 bg-stone-50 rounded-xl border border-stone-200">
                <span className="text-[9px] font-bold text-stone-500 block uppercase">Phòng ngủ</span>
                <span className="font-bold text-navy-900 mt-0.5 block">{f.Bedrooms}</span>
              </div>
              <div className="p-3 bg-stone-50 rounded-xl border border-stone-200">
                <span className="text-[9px] font-bold text-stone-500 block uppercase">Phòng tắm</span>
                <span className="font-bold text-navy-900 mt-0.5 block">{f.Bathrooms}</span>
              </div>
              <div className="p-3 bg-stone-50 rounded-xl border border-stone-200">
                <span className="text-[9px] font-bold text-stone-500 block uppercase">Pháp lý</span>
                <span className="font-bold text-navy-900 mt-0.5 block truncate">{f.Legal_status}</span>
              </div>
              <div className="p-3 bg-stone-50 rounded-xl border border-stone-200">
                <span className="text-[9px] font-bold text-stone-500 block uppercase">Nội thất</span>
                <span className="font-bold text-navy-900 mt-0.5 block truncate">{f.Furniture_state}</span>
              </div>
            </div>
          </div>

          <div className="space-y-3">
            <h3 className="text-xs font-bold uppercase tracking-wider text-navy-900 border-b border-stone-200 pb-1.5">
              2. So sánh dự đoán từ 6 mô hình
            </h3>
            <div className="border border-stone-200 rounded-xl overflow-hidden">
              <table className="w-full text-left border-collapse text-xs">
                <thead>
                  <tr className="bg-stone-100 border-b border-stone-200 text-stone-500 font-bold uppercase tracking-wider">
                    <th className="p-3">Mô hình</th>
                    <th className="p-3">Giá dự đoán</th>
                    <th className="p-3">R²</th>
                    <th className="p-3 text-right">MAPE</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-stone-200">
                  {[...activeRecord.results].sort((a, b) => b.r2 - a.r2).map((r) => (
                    <tr key={r.model}>
                      <td className="p-3 font-semibold text-navy-900">{formatModelName(r.model)}</td>
                      <td className="p-3 font-mono text-stone-700">{r.price.toFixed(2)} tỷ</td>
                      <td className="p-3 font-mono text-stone-600">{r.r2.toFixed(3)}</td>
                      <td className="p-3 font-mono text-stone-600 text-right">{r.mape.toFixed(1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="space-y-4">
            <h3 className="text-xs font-bold uppercase tracking-wider text-navy-900 border-b border-stone-200 pb-1.5">
              3. Kết luận định giá &amp; Yếu tố ảnh hưởng
            </h3>
            <div className="border rounded-xl p-4 flex gap-4 items-start bg-brand-50 border-brand-200 text-brand-900">
              <Building2 className="w-6 h-6 shrink-0 mt-0.5 text-brand-600" />
              <div>
                <span className="text-xs font-black uppercase tracking-wider">Giá đề xuất: {best.price.toFixed(2)} tỷ VND</span>
                <p className="text-[11px] font-medium leading-relaxed mt-1 text-stone-600">
                  Mức giá trên được lấy từ mô hình <strong>{formatModelName(best.model)}</strong> (R²={best.r2.toFixed(3)}, độ chính xác MAPE={best.mape.toFixed(1)}%) — mô hình có độ phù hợp cao nhất trong 6 mô hình được huấn luyện, với độ tin cậy {(best.confidence * 100).toFixed(1)}%.
                </p>
              </div>
            </div>

            {loading ? (
              <div className="flex items-center gap-2 text-xs text-stone-500">
                <Loader2 className="w-4 h-4 animate-spin" /> Đang tải phân tích SHAP...
              </div>
            ) : featureItems.length > 0 ? (
              <div className="space-y-2 text-xs">
                <span className="font-bold text-stone-600 uppercase text-[10px] tracking-wider block">Top yếu tố tác động (SHAP):</span>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                  {[...featureItems].sort((a, b) => Math.abs(b.value) - Math.abs(a.value)).slice(0, 3).map((item) => (
                    <div key={item.feature} className="p-3 border border-stone-200 bg-stone-50 rounded-xl">
                      <span className="font-bold text-navy-900 block truncate" title={item.feature}>{item.feature}</span>
                      <span className={`text-[10px] font-bold block mt-1.5 uppercase ${item.value > 0 ? 'text-red-600' : 'text-blue-600'}`}>
                        {item.value > 0 ? `Tăng giá (+${item.value.toFixed(2)} tỷ)` : `Giảm giá (${item.value.toFixed(2)} tỷ)`}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            ) : null}
          </div>

          <div className="space-y-3">
            <h3 className="text-xs font-bold uppercase tracking-wider text-navy-900 border-b border-stone-200 pb-1.5">
              4. Khuyến nghị của định giá viên
            </h3>
            <p className="text-xs leading-relaxed text-stone-700 bg-stone-50 p-4 border border-stone-200 rounded-xl italic whitespace-pre-wrap">
              "{notes}"
            </p>
          </div>
        </div>

        <div className="flex justify-between items-end border-t border-stone-200 pt-8 mt-12 text-xs">
          <div className="text-stone-500 leading-relaxed max-w-[65%]">
            <span className="font-bold text-navy-900 uppercase text-[9px] tracking-wider block">Miễn trừ trách nhiệm:</span>
            Báo cáo này được tạo tự động bằng các mô hình học máy dựa trên dữ liệu lịch sử, chỉ mang tính chất tham khảo. Vui lòng đối chiếu với khảo sát thực địa và tư vấn pháp lý trước khi ra quyết định giao dịch.
          </div>
          <div className="text-center w-[180px]">
            <div className="border-b border-stone-400 h-10 w-full" />
            <span className="font-bold text-navy-900 mt-2 block">{appraiserName}</span>
            <span className="text-[9px] font-semibold text-stone-400 uppercase tracking-wider mt-0.5 block">Chữ ký định giá viên</span>
          </div>
        </div>
      </div>
    </div>
  );
};
