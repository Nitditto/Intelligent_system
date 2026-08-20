import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { X, Loader2, Target, Percent, Info, HelpCircle, TrendingUp, CheckCircle2 } from 'lucide-react';

interface SHAPFeature {
  feature: string;
  value: number;
}

interface DetailedModelResult {
  model: string;
  prediction: number;
  confidence: number;
  r2: number;
  mape: number;
  base_value: number;
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

const translateFeatureName = (name: string) => {
  // Translate features to Vietnamese for premium UX
  const translations: Record<string, string> = {
    'Area': 'Diện tích',
    'Frontage': 'Mặt tiền',
    'Access Road': 'Độ rộng đường vào',
    'Floors': 'Số tầng',
    'Bedrooms': 'Số phòng ngủ',
    'Bathrooms': 'Số phòng vệ sinh',
    'House direction_Đông': 'Hướng nhà: Đông',
    'House direction_Tây': 'Hướng nhà: Tây',
    'House direction_Nam': 'Hướng nhà: Nam',
    'House direction_Bắc': 'Hướng nhà: Bắc',
    'House direction_Đông - Nam': 'Hướng nhà: Đông Nam',
    'House direction_Tây - Nam': 'Hướng nhà: Tây Nam',
    'House direction_Đông - Bắc': 'Hướng nhà: Đông Bắc',
    'House direction_Tây - Bắc': 'Hướng nhà: Tây Bắc',
    'House direction_Unknown': 'Hướng nhà: Không xác định',
    'Furniture state_Full': 'Nội thất: Đầy đủ',
    'Furniture state_Basic': 'Nội thất: Cơ bản',
    'Furniture state_None': 'Nội thất: Trống',
    'Furniture state_Unknown': 'Nội thất: Không xác định',
    'Balcony direction_Đông': 'Hướng ban công: Đông',
    'Balcony direction_Tây': 'Hướng ban công: Tây',
    'Balcony direction_Nam': 'Hướng ban công: Nam',
    'Balcony direction_Bắc': 'Hướng ban công: Bắc',
    'Balcony direction_Đông - Nam': 'Hướng ban công: Đông Nam',
    'Balcony direction_Tây - Nam': 'Hướng ban công: Tây Nam',
    'Balcony direction_Đông - Bắc': 'Hướng ban công: Đông Bắc',
    'Balcony direction_Tây - Bắc': 'Hướng ban công: Tây Bắc',
    'Legal status_Sổ đỏ/ Sổ hồng': 'Pháp lý: Sổ đỏ/Sổ hồng',
    'Legal status_Hợp đồng mua bán': 'Pháp lý: Hợp đồng mua bán',
    'Legal status_Đang chờ sổ': 'Pháp lý: Đang chờ sổ',
    'Furniture state_Đầy đủ': 'Nội thất: Đầy đủ',
    'Furniture state_Cơ bản': 'Nội thất: Cơ bản',
    'Furniture state_Thô': 'Nội thất: Bàn giao thô',
    'Furniture state_Không có nội thất': 'Nội thất: Không có',
    'Giá trị nền (Base Value)': 'Giá trị nền thị trường'
  };

  if (translations[name]) return translations[name];
  // If it starts with District_
  if (name.startsWith('District_')) {
    return `Quận: ${name.replace('District_', '')}`;
  }
  return name;
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
          setData(res.data);
        })
        .catch(err => {
          console.error(err);
          setError('Không thể tải chi tiết phân tích mô hình.');
        })
        .finally(() => setLoading(false));
    }
  }, [isOpen, model, resultId, apiUrl]);

  if (!isOpen) return null;

  // Split SHAP values into positive and negative
  const shapValues = data?.shap_values || [];
  const pos = shapValues.filter(x => x.value > 0);
  const neg = shapValues.filter(x => x.value < 0);

  const totalPos = pos.reduce((sum, x) => sum + x.value, 0);
  const totalNeg = Math.abs(neg.reduce((sum, x) => sum + x.value, 0));
  
  // Calculate a reasonable visual scale
  const maxDelta = Math.max(totalPos, totalNeg, 0.5);

  return (
    <>
      {/* Backdrop */}
      <div
        onClick={onClose}
        className="fixed inset-0 bg-slate-900/40 backdrop-blur-sm z-[100] transition-opacity duration-300"
      />

      {/* Drawer */}
      <div className="fixed right-0 top-0 bottom-0 w-full max-w-xl bg-white shadow-2xl border-l border-slate-200 z-[101] overflow-y-auto flex flex-col transition-transform duration-300 transform translate-x-0">
        {/* Header */}
        <div className="p-6 border-b border-slate-100 flex items-center justify-between sticky top-0 bg-white/90 backdrop-blur-md z-10">
          <div>
            <h2 className="text-xl font-bold text-slate-800">
              {model ? formatModelName(model) : ''}
            </h2>
            <p className="text-sm text-slate-500">Phân tích đóng góp đặc trưng (SHAP Values)</p>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-slate-100 text-slate-400 hover:text-slate-700 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 flex-1 space-y-6">
          {loading ? (
            <div className="flex flex-col items-center justify-center h-[50vh] text-brand-600 gap-3">
              <Loader2 className="w-8 h-8 animate-spin" />
              <p className="text-slate-500 text-sm">Đang tính toán SHAP và giải thích mô hình...</p>
            </div>
          ) : error ? (
            <div className="bg-red-50 text-red-700 p-4 rounded-xl text-center text-sm border border-red-100">
              {error}
            </div>
          ) : data ? (
            <div className="space-y-6">
              {/* Summary Stats */}
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-slate-50 border border-slate-100 p-4 rounded-xl flex items-center gap-3">
                  <div className="p-3 bg-brand-100 text-brand-700 rounded-lg">
                    <Target className="w-5 h-5" />
                  </div>
                  <div>
                    <div className="text-xs text-slate-500">Giá trị dự báo</div>
                    <div className="text-sm font-bold text-slate-800">{data.prediction.toFixed(2)} tỷ</div>
                  </div>
                </div>

                <div className="bg-slate-50 border border-slate-100 p-4 rounded-xl flex items-center gap-3">
                  <div className="p-3 bg-green-100 text-green-700 rounded-lg">
                    <Percent className="w-5 h-5" />
                  </div>
                  <div>
                    <div className="text-xs text-slate-500">Độ tin cậy</div>
                    <div className="text-sm font-bold text-slate-800">{(data.confidence * 100).toFixed(1)}%</div>
                  </div>
                </div>

                <div className="bg-slate-50 border border-slate-100 p-4 rounded-xl flex items-center gap-3">
                  <div className="p-3 bg-blue-100 text-blue-700 rounded-lg">
                    <TrendingUp className="w-5 h-5" />
                  </div>
                  <div>
                    <div className="text-xs text-slate-500">Hệ số R²</div>
                    <div className="text-sm font-bold text-slate-800">{data.r2.toFixed(4)}</div>
                  </div>
                </div>

                <div className="bg-slate-50 border border-slate-100 p-4 rounded-xl flex items-center gap-3">
                  <div className="p-3 bg-purple-100 text-purple-700 rounded-lg">
                    <CheckCircle2 className="w-5 h-5" />
                  </div>
                  <div>
                    <div className="text-xs text-slate-500">Chính xác (MAPE)</div>
                    <div className="text-sm font-bold text-slate-800">{data.mape.toFixed(2)}%</div>
                  </div>
                </div>
              </div>

              {/* Force Plot Section */}
              <div className="border border-slate-150 rounded-2xl p-5 bg-white shadow-sm space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-md font-bold text-slate-800 flex items-center gap-1.5">
                    SHAP Stacked Force Plot
                    <span className="cursor-help" title="Lực đẩy từ giá gốc: Đỏ đẩy giá cao lên, Xanh kéo giá thấp xuống.">
                      <HelpCircle className="w-4 h-4 text-slate-400" />
                    </span>
                  </h3>
                </div>

                {/* Plot component */}
                <div className="relative mt-4">
                  {/* Label for stacks */}
                  <div className="flex justify-between text-xs text-slate-500 font-semibold mb-2">
                    <span className="text-blue-600">Lực kéo giảm giá (-)</span>
                    <span className="text-red-600">Lực đẩy tăng giá (+)</span>
                  </div>

                  {/* Horizontal Bar container */}
                  <div className="relative h-14 bg-slate-100 rounded-xl overflow-hidden flex border border-slate-200/60 shadow-inner">
                    {/* Negative (left) stack */}
                    <div className="w-1/2 flex flex-row-reverse items-stretch h-full">
                      {neg.map((item, idx) => {
                        const valAbs = Math.abs(item.value);
                        const pct = (valAbs / maxDelta) * 50; // max 50%
                        return (
                          <div
                            key={idx}
                            style={{ width: `${Math.max(2, pct)}%` }}
                            className="bg-blue-500 hover:bg-blue-600 transition-colors cursor-pointer relative group flex items-center justify-center text-white text-[9px] font-bold border-r border-blue-400"
                          >
                            {pct > 12 && (
                              <span className="truncate px-1 select-none">
                                {translateFeatureName(item.feature).split(':')[0]}
                              </span>
                            )}
                            
                            {/* Rich Tooltip */}
                            <div className="absolute bottom-full mb-2 hidden group-hover:flex flex-col bg-slate-900/95 backdrop-blur-sm text-white text-xs rounded-lg p-2.5 z-[150] shadow-xl border border-slate-800 min-w-48 leading-relaxed">
                              <span className="font-semibold text-slate-300">{translateFeatureName(item.feature)}</span>
                              <span className="text-blue-400 font-bold mt-1">Đóng góp: {item.value.toFixed(3)} tỷ VNĐ</span>
                            </div>
                          </div>
                        );
                      })}
                    </div>

                    {/* Positive (right) stack */}
                    <div className="w-1/2 flex items-stretch h-full">
                      {pos.map((item, idx) => {
                        const pct = (item.value / maxDelta) * 50; // max 50%
                        return (
                          <div
                            key={idx}
                            style={{ width: `${Math.max(2, pct)}%` }}
                            className="bg-red-500 hover:bg-red-600 transition-colors cursor-pointer relative group flex items-center justify-center text-white text-[9px] font-bold border-l border-red-400"
                          >
                            {pct > 12 && (
                              <span className="truncate px-1 select-none">
                                {translateFeatureName(item.feature).split(':')[0]}
                              </span>
                            )}
                            
                            {/* Rich Tooltip */}
                            <div className="absolute bottom-full mb-2 hidden group-hover:flex flex-col bg-slate-900/95 backdrop-blur-sm text-white text-xs rounded-lg p-2.5 z-[150] shadow-xl border border-slate-800 min-w-48 leading-relaxed">
                              <span className="font-semibold text-slate-300">{translateFeatureName(item.feature)}</span>
                              <span className="text-red-400 font-bold mt-1">Đóng góp: +{item.value.toFixed(3)} tỷ VNĐ</span>
                            </div>
                          </div>
                        );
                      })}
                    </div>

                    {/* Central Base line */}
                    <div className="absolute top-0 bottom-0 left-1/2 w-0.5 bg-slate-800 z-10">
                      <div className="absolute -top-1 -left-1 w-2.5 h-2.5 rounded-full bg-slate-800" />
                    </div>
                  </div>

                  {/* Summary of forces */}
                  <div className="flex items-center justify-between mt-3 text-xs">
                    <div className="text-blue-600 font-bold flex items-center gap-1">
                      <span className="w-2 h-2 rounded-full bg-blue-500 inline-block"></span>
                      Tổng lực giảm: -{totalNeg.toFixed(2)} tỷ
                    </div>
                    <div className="text-red-600 font-bold flex items-center gap-1">
                      <span className="w-2 h-2 rounded-full bg-red-500 inline-block"></span>
                      Tổng lực tăng: +{totalPos.toFixed(2)} tỷ
                    </div>
                  </div>
                </div>
              </div>

              {/* Explanatory text */}
              <div className="bg-slate-50 border border-slate-100 rounded-xl p-4 flex gap-3 text-xs text-slate-500">
                <Info className="w-5 h-5 text-slate-400 shrink-0" />
                <div className="space-y-1">
                  <p className="font-semibold text-slate-700">Biểu đồ SHAP Force Plot giải thích thế nào?</p>
                  <p>Mỗi đặc trưng của bất động sản bạn nhập (ví dụ: diện tích lớn, số phòng ngủ, vị trí quận nội thành) sẽ hoạt động như một "lực đẩy" (Màu Đỏ) hoặc "lực kéo" (Màu Xanh Dương) làm thay đổi giá trị nhà so với giá trung vị gốc, tạo nên mức giá dự đoán cuối cùng.</p>
                </div>
              </div>

              {/* Feature list table */}
              <div className="space-y-3">
                <h4 className="text-sm font-bold text-slate-800">Danh sách các yếu tố ảnh hưởng mạnh nhất</h4>
                <div className="border border-slate-100 rounded-xl overflow-hidden divide-y divide-slate-100 text-xs">
                  {shapValues.map((item, idx) => (
                    <div key={idx} className="flex justify-between items-center p-3 hover:bg-slate-50">
                      <span className="font-medium text-slate-700">{translateFeatureName(item.feature)}</span>
                      <span className={`font-bold ${item.value > 0 ? 'text-red-600' : 'text-blue-600'}`}>
                        {item.value > 0 ? `+${item.value.toFixed(3)}` : item.value.toFixed(3)} tỷ
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ) : null}
        </div>
      </div>
    </>
  );
};
