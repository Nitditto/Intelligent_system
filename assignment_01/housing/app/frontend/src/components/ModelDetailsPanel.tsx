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

  // Base value is carried as a pseudo-feature; split it out so it doesn't skew the
  // feature-contribution totals or dominate the waterfall bars.
  const shapValues = data?.shap_values || [];
  const baseValueItem = shapValues.find(x => x.feature === 'Giá trị nền (Base Value)');
  const featureItems = shapValues.filter(x => x !== baseValueItem);

  const pos = featureItems.filter(x => x.value > 0);
  const neg = featureItems.filter(x => x.value < 0);
  const totalPos = pos.reduce((sum, x) => sum + x.value, 0);
  const totalNeg = Math.abs(neg.reduce((sum, x) => sum + x.value, 0));
  const maxAbs = Math.max(...featureItems.map(x => Math.abs(x.value)), 0.01);

  return (
    <>
      {/* Backdrop */}
      <div
        onClick={onClose}
        className="fixed inset-0 bg-stone-900/40 backdrop-blur-sm z-[100] transition-opacity duration-300"
      />

      {/* Drawer */}
      <div className="fixed right-0 top-0 bottom-0 w-full max-w-xl bg-white dark:bg-navy-900 shadow-2xl border-l border-stone-200 dark:border-navy-700 z-[101] overflow-y-auto flex flex-col transition-transform duration-300 transform translate-x-0">
        {/* Header */}
        <div className="p-6 border-b border-stone-100 dark:border-navy-700 flex items-center justify-between sticky top-0 bg-white/90 dark:bg-navy-900/90 backdrop-blur-md z-10">
          <div>
            <h2 className="text-xl font-bold text-stone-800 dark:text-stone-100">
              {model ? formatModelName(model) : ''}
            </h2>
            <p className="text-sm text-stone-500 dark:text-stone-400">Phân tích đóng góp đặc trưng (SHAP Values)</p>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-stone-100 dark:hover:bg-navy-800 text-stone-400 hover:text-stone-700 dark:hover:text-stone-200 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 flex-1 space-y-6">
          {loading ? (
            <div className="flex flex-col items-center justify-center h-[50vh] text-brand-600 dark:text-brand-400 gap-3">
              <Loader2 className="w-8 h-8 animate-spin" />
              <p className="text-stone-500 dark:text-stone-400 text-sm">Đang tính toán SHAP và giải thích mô hình...</p>
            </div>
          ) : error ? (
            <div className="bg-red-50 dark:bg-red-500/10 text-red-700 dark:text-red-400 p-4 rounded-xl text-center text-sm border border-red-100 dark:border-red-500/30">
              {error}
            </div>
          ) : data ? (
            <div className="space-y-6">
              {/* Summary Stats */}
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-stone-50 dark:bg-navy-800/60 border border-stone-100 dark:border-navy-700 p-4 rounded-xl flex items-center gap-3">
                  <div className="p-3 bg-brand-100 dark:bg-brand-900/40 text-brand-700 dark:text-brand-400 rounded-lg">
                    <Target className="w-5 h-5" />
                  </div>
                  <div>
                    <div className="text-xs text-stone-500 dark:text-stone-400">Giá trị dự báo</div>
                    <div className="text-sm font-bold text-stone-800 dark:text-stone-100">{data.prediction.toFixed(2)} tỷ</div>
                  </div>
                </div>

                <div className="bg-stone-50 dark:bg-navy-800/60 border border-stone-100 dark:border-navy-700 p-4 rounded-xl flex items-center gap-3">
                  <div className="p-3 bg-brand-100 dark:bg-brand-900/40 text-brand-700 dark:text-brand-400 rounded-lg">
                    <Percent className="w-5 h-5" />
                  </div>
                  <div>
                    <div className="text-xs text-stone-500 dark:text-stone-400">Độ tin cậy</div>
                    <div className="text-sm font-bold text-stone-800 dark:text-stone-100">{(data.confidence * 100).toFixed(1)}%</div>
                  </div>
                </div>

                <div className="bg-stone-50 dark:bg-navy-800/60 border border-stone-100 dark:border-navy-700 p-4 rounded-xl flex items-center gap-3">
                  <div className="p-3 bg-brand-100 dark:bg-brand-900/40 text-brand-700 dark:text-brand-400 rounded-lg">
                    <TrendingUp className="w-5 h-5" />
                  </div>
                  <div>
                    <div className="text-xs text-stone-500 dark:text-stone-400">Hệ số R²</div>
                    <div className="text-sm font-bold text-stone-800 dark:text-stone-100">{data.r2.toFixed(4)}</div>
                  </div>
                </div>

                <div className="bg-stone-50 dark:bg-navy-800/60 border border-stone-100 dark:border-navy-700 p-4 rounded-xl flex items-center gap-3">
                  <div className="p-3 bg-brand-100 dark:bg-brand-900/40 text-brand-700 dark:text-brand-400 rounded-lg">
                    <CheckCircle2 className="w-5 h-5" />
                  </div>
                  <div>
                    <div className="text-xs text-stone-500 dark:text-stone-400">Chính xác (MAPE)</div>
                    <div className="text-sm font-bold text-stone-800 dark:text-stone-100">{data.mape.toFixed(2)}%</div>
                  </div>
                </div>
              </div>

              {/* Feature contribution waterfall */}
              <div className="border border-stone-200 dark:border-navy-700 rounded-2xl p-5 bg-white dark:bg-navy-800/60 shadow-sm space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-md font-bold text-stone-800 dark:text-stone-100 flex items-center gap-1.5">
                    Yếu tố ảnh hưởng đến giá
                    <span className="cursor-help" title="Mỗi thanh thể hiện mức độ một đặc trưng đẩy giá tăng (đỏ) hoặc kéo giá giảm (xanh) so với giá trị nền thị trường.">
                      <HelpCircle className="w-4 h-4 text-stone-400 dark:text-stone-500" />
                    </span>
                  </h3>
                </div>

                {baseValueItem && (
                  <div className="flex items-center justify-between text-xs bg-stone-50 dark:bg-navy-900/50 rounded-lg px-3 py-2">
                    <span className="text-stone-500 dark:text-stone-400">Giá trị nền thị trường</span>
                    <span className="font-bold text-stone-700 dark:text-stone-200">{baseValueItem.value.toFixed(2)} tỷ</span>
                  </div>
                )}

                <div className="flex justify-between text-[11px] font-semibold text-stone-400 dark:text-stone-500 px-[8.5rem]">
                  <span className="text-blue-600 dark:text-blue-400">← Giảm giá</span>
                  <span className="text-red-600 dark:text-red-400">Tăng giá →</span>
                </div>

                <div className="space-y-2.5">
                  {featureItems.map((item, idx) => {
                    const isPos = item.value > 0;
                    const pct = Math.max(3, (Math.abs(item.value) / maxAbs) * 100);
                    return (
                      <div key={idx} className="flex items-center gap-3">
                        <span
                          className="w-28 sm:w-32 shrink-0 text-right text-[11px] leading-tight text-stone-600 dark:text-stone-300 truncate"
                          title={translateFeatureName(item.feature)}
                        >
                          {translateFeatureName(item.feature)}
                        </span>
                        <div className="flex-1 relative h-5">
                          <div className="absolute inset-y-0 left-1/2 w-px bg-stone-300 dark:bg-navy-600" />
                          <div
                            className={`absolute inset-y-0 rounded-sm ${isPos ? 'left-1/2 bg-red-500' : 'right-1/2 bg-blue-500'}`}
                            style={{ width: `${pct / 2}%` }}
                          />
                        </div>
                        <span className={`w-14 shrink-0 text-right text-xs font-bold ${isPos ? 'text-red-600 dark:text-red-400' : 'text-blue-600 dark:text-blue-400'}`}>
                          {isPos ? '+' : ''}{item.value.toFixed(2)}
                        </span>
                      </div>
                    );
                  })}
                </div>

                <div className="flex items-center justify-between pt-3 border-t border-stone-100 dark:border-navy-700 text-xs">
                  <div className="text-blue-600 dark:text-blue-400 font-bold flex items-center gap-1">
                    <span className="w-2 h-2 rounded-full bg-blue-500 inline-block"></span>
                    Tổng lực giảm: -{totalNeg.toFixed(2)} tỷ
                  </div>
                  <div className="text-red-600 dark:text-red-400 font-bold flex items-center gap-1">
                    <span className="w-2 h-2 rounded-full bg-red-500 inline-block"></span>
                    Tổng lực tăng: +{totalPos.toFixed(2)} tỷ
                  </div>
                </div>
              </div>

              {/* Explanatory text */}
              <div className="bg-stone-50 dark:bg-navy-800/60 border border-stone-100 dark:border-navy-700 rounded-xl p-4 flex gap-3 text-xs text-stone-500 dark:text-stone-400">
                <Info className="w-5 h-5 text-stone-400 dark:text-stone-500 shrink-0" />
                <div className="space-y-1">
                  <p className="font-semibold text-stone-700 dark:text-stone-200">Biểu đồ này giải thích thế nào?</p>
                  <p>Mỗi đặc trưng của bất động sản bạn nhập (ví dụ: diện tích lớn, số phòng ngủ, vị trí quận nội thành) sẽ hoạt động như một "lực đẩy" (Màu Đỏ) hoặc "lực kéo" (Màu Xanh Dương) làm thay đổi giá trị nhà so với giá trị nền thị trường, tạo nên mức giá dự đoán cuối cùng.</p>
                </div>
              </div>
            </div>
          ) : null}
        </div>
      </div>
    </>
  );
};
