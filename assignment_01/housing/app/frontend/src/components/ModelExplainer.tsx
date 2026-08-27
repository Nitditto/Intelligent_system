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
  family: 'Tree Ensemble' | 'Kernel / Margin' | 'Instance-Based' | 'Linear';
  mae: number;
  r2: number;
  mape: number;
  description: string;
  representation: string;
  criterion: string;
  assumptions: string;
  strengths: string[];
  weaknesses: string[];
}

// Real metrics, ported 1:1 from Report.md Section 8 (Results table) and
// matching the values hardcoded server-side in main.py's model_r2_scores / model_mape_accuracies.
const modelsData: ModelInfo[] = [
  {
    name: 'xgboost',
    displayName: 'XGBoost',
    family: 'Tree Ensemble',
    mae: 1.0498, r2: 0.5954, mape: 78.91,
    description: 'Kỹ thuật ensemble tuần tự, huấn luyện các cây quyết định lần lượt, mỗi cây được xây dựng để sửa lỗi của các cây trước đó thông qua gradient descent.',
    representation: 'Vector đặc trưng one-hot chưa chuẩn hóa (không cần scaling, tự xử lý được giá trị thiếu).',
    criterion: 'Tối thiểu hóa hàm mất mát bình phương có điều chuẩn (regularized squared-error) bằng gradient boosting.',
    assumptions: 'Giả định rằng sai số dư (residual) từ các cây trước chứa tín hiệu có thể học được bởi các cây tiếp theo.',
    strengths: [
      'Độ chính xác hàng đầu trên dữ liệu dạng bảng (tabular).',
      'Xử lý tốt không gian đặc trưng thưa (one-hot 342 quận).',
      'Có cơ chế điều chuẩn (regularization) tích hợp giúp giảm overfitting.',
    ],
    weaknesses: [
      'Nhiều siêu tham số cần tinh chỉnh (learning_rate, max_depth, n_estimators).',
      'Khó diễn giải nhất trong 6 mô hình.',
    ],
  },
  {
    name: 'random_forest',
    displayName: 'Random Forest',
    family: 'Tree Ensemble',
    mae: 1.0738, r2: 0.5596, mape: 77.96,
    description: 'Phương pháp ensemble xây dựng nhiều cây quyết định độc lập và tổng hợp kết quả dự đoán bằng cách lấy trung bình (bagging + random feature selection).',
    representation: 'Vector đặc trưng one-hot chưa chuẩn hóa (cây không yêu cầu scaling).',
    criterion: 'Tối thiểu hóa phương sai (squared error) tại mỗi điểm chia, trung bình hóa qua các cây được lấy mẫu bootstrap.',
    assumptions: 'Ít giả định phân phối; giả định rằng việc trung bình hóa nhiều cây ít tương quan sẽ giảm phương sai.',
    strengths: [
      'Bắt được các quan hệ phi tuyến và tương tác đặc trưng (VD: diện tích × quận).',
      'Bền vững trước outlier và các cột one-hot không liên quan.',
    ],
    weaknesses: [
      'Tốn bộ nhớ/tính toán hơn với ensemble lớn.',
      'Kém diễn giải hơn một cây đơn hoặc mô hình tuyến tính.',
    ],
  },
  {
    name: 'svr_rbf',
    displayName: 'SVR (RBF Kernel)',
    family: 'Kernel / Margin',
    mae: 1.1195, r2: 0.5249, mape: 77.54,
    description: 'Mô hình SVR nâng cao, chiếu đặc trưng đầu vào lên không gian nhiều chiều hơn bằng kernel RBF, tìm ranh giới tuyến tính trong không gian đã biến đổi đó.',
    representation: 'Vector đặc trưng đã chuẩn hóa (scaling bắt buộc).',
    criterion: 'Tối đa hóa biên (margin) trong không gian đặc trưng đã biến đổi qua kernel, dung sai ε (ε-insensitive loss).',
    assumptions: 'Các điểm gần nhau trong không gian đã biến đổi có xu hướng có giá tương tự.',
    strengths: [
      'Mô hình hóa được ranh giới phi tuyến phức tạp mà không cần feature engineering thủ công.',
      'Là mô hình phi cây tốt nhất trong thử nghiệm (R²=0.5249).',
    ],
    weaknesses: [
      'Tốn kém khi huấn luyện/tinh chỉnh trên tập dữ liệu lớn (30k dòng).',
      'Khó diễn giải (support vector nằm trong không gian nhiều chiều).',
    ],
  },
  {
    name: 'knn',
    displayName: 'K-Nearest Neighbors',
    family: 'Instance-Based',
    mae: 1.2747, r2: 0.4108, mape: 74.37,
    description: 'Học dựa trên thực thể (lazy learner), không xây dựng mô hình tường minh khi huấn luyện. Dự đoán bằng trung bình có trọng số khoảng cách của k tin đăng gần nhất (k=9, distance-weighted, tối ưu qua GridSearchCV).',
    representation: 'Vector đặc trưng đã chuẩn hóa (bắt buộc vì KNN dựa vào khoảng cách).',
    criterion: 'Khoảng cách Euclidean trong không gian 367 chiều đã chuẩn hóa.',
    assumptions: 'Các tin đăng gần nhau trong không gian đặc trưng có giá tương tự nhau.',
    strengths: [
      'Đơn giản, không cần pha huấn luyện tường minh.',
      'Bắt được cấu trúc phi tuyến cục bộ.',
    ],
    weaknesses: [
      'Dự đoán chậm trên tập dữ liệu lớn.',
      'Chịu "lời nguyền chiều dữ liệu" (curse of dimensionality) do 367 chiều, phần lớn là one-hot thưa.',
    ],
  },
  {
    name: 'svr_linear',
    displayName: 'SVR (Linear Kernel)',
    family: 'Kernel / Margin',
    mae: 1.3147, r2: 0.3765, mape: 73.94,
    description: 'Mặt phẳng giá tuyến tính giữ hầu hết các điểm trong biên ε, tương tự Linear Regression nhưng dùng hàm mất mát dựa trên biên (hinge-like) thay vì bình phương sai số.',
    representation: 'Vector đặc trưng đã chuẩn hóa.',
    criterion: 'Hàm mất mát ε-insensitive bình phương, điều chuẩn bởi tham số C.',
    assumptions: 'Bề mặt giá xấp xỉ tuyến tính trong không gian đặc trưng đã mã hóa.',
    strengths: [
      'Bền vững trước outlier hơn hồi quy bình phương tối thiểu thông thường.',
      'Mở rộng tốt hơn kernel RBF trên tập dữ liệu lớn.',
    ],
    weaknesses: [
      'Cùng hạn chế tuyến tính như Linear Regression.',
      'Không mô hình hóa được hiệu ứng phi tuyến hoặc tương tác đặc trưng.',
    ],
  },
  {
    name: 'linear_regression',
    displayName: 'Linear Regression',
    family: 'Linear',
    mae: 1.3001, r2: 0.3735, mape: 74.02,
    description: 'Mô hình tuyến tính học một trọng số (hệ số) cho mỗi chiều đặc trưng và một hệ số chặn, tối thiểu hóa sai số bình phương trung bình giữa giá dự đoán và thực tế (trong không gian log-giá).',
    representation: 'Vector đặc trưng đã chuẩn hóa; mục tiêu ở không gian log-giá.',
    criterion: 'Tối thiểu hóa sai số bình phương trung bình (MSE) giữa log-giá dự đoán và thực tế.',
    assumptions: 'Quan hệ tuyến tính giữa đặc trưng và log(Giá), đa cộng tuyến hạn chế, phần dư gần như đồng phương sai.',
    strengths: [
      'Huấn luyện và đánh giá rất nhanh.',
      'Hệ số dễ diễn giải — là điểm tham chiếu vững chắc cho các mô hình phức tạp hơn.',
    ],
    weaknesses: [
      'Không bắt được hiệu ứng giá phi tuyến (VD: lợi ích giảm dần theo diện tích) hoặc tương tác đặc trưng (quận × pháp lý).',
      'Bị ảnh hưởng bởi sai lệch khi biến đổi ngược log1p→expm1 (xem Thí nghiệm 3 trong Report.md).',
    ],
  },
];

const familyColor: Record<ModelInfo['family'], string> = {
  'Tree Ensemble': '#10b981',
  'Kernel / Margin': '#6366f1',
  'Instance-Based': '#f59e0b',
  'Linear': '#94a3b8',
};

const CustomBarTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    const m = payload[0].payload as ModelInfo;
    return (
      <div className="rounded-xl border border-stone-200 dark:border-navy-700 bg-white dark:bg-navy-800 p-3 text-xs shadow-2xl min-w-[170px] space-y-1">
        <p className="font-semibold text-stone-800 dark:text-stone-100">{m.displayName}</p>
        <p className="text-stone-500 dark:text-stone-400">R²: <span className="font-mono font-semibold text-stone-800 dark:text-stone-100">{m.r2.toFixed(3)}</span></p>
        <p className="text-stone-500 dark:text-stone-400">MAPE: <span className="font-mono font-semibold text-stone-800 dark:text-stone-100">{m.mape.toFixed(2)}%</span></p>
      </div>
    );
  }
  return null;
};

export const ModelExplainer: React.FC = () => {
  const [selected, setSelected] = useState<string>('xgboost');
  const model = modelsData.find(m => m.name === selected) ?? modelsData[0];
  const sortedByR2 = [...modelsData].sort((a, b) => b.r2 - a.r2);

  return (
    <div className="space-y-6">
      <div className="glass-card p-6">
        <h2 className="text-2xl font-bold text-stone-800 dark:text-stone-100 flex items-center gap-2">
          <Cpu className="w-6 h-6 text-brand-500" />
          Trung tâm giải thích mô hình
        </h2>
        <p className="mt-2 text-sm text-stone-500 dark:text-stone-400 max-w-2xl">
          6 mô hình hồi quy được huấn luyện trên bộ dữ liệu bất động sản Việt Nam. Chọn một mô hình bên dưới để xem cách nó hoạt động,
          giả định và điểm mạnh/yếu so với các mô hình còn lại.
        </p>
      </div>

      {/* Comparison chart */}
      <div className="glass-card p-5">
        <div className="mb-3 flex items-center justify-between flex-wrap gap-2">
          <h3 className="text-sm font-semibold text-stone-800 dark:text-stone-100">So sánh hệ số R² (tập kiểm tra)</h3>
          <div className="flex items-center gap-3 text-[11px] text-stone-500 dark:text-stone-400 flex-wrap">
            {(['Tree Ensemble', 'Kernel / Margin', 'Instance-Based', 'Linear'] as const).map(f => (
              <span key={f} className="flex items-center gap-1.5">
                <span className="h-2.5 w-2.5 rounded-full" style={{ background: familyColor[f] }} />
                {f}
              </span>
            ))}
          </div>
        </div>
        <div className="h-[220px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={sortedByR2} layout="vertical" margin={{ top: 5, right: 25, left: 10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" horizontal={false} />
              <XAxis type="number" domain={[0, 0.65]} fontSize={11} tickLine={false} />
              <YAxis dataKey="displayName" type="category" width={130} fontSize={11} tickLine={false} axisLine={false} />
              <Tooltip content={<CustomBarTooltip />} cursor={{ fill: 'rgba(194,138,42,0.08)' }} />
              <Bar
                dataKey="r2"
                radius={[4, 4, 4, 4]}
                onClick={(data: any) => setSelected(data.name)}
                cursor="pointer"
              >
                {sortedByR2.map(m => (
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
        <div className="lg:col-span-4 glass-card p-3 space-y-1.5">
          {sortedByR2.map((m, idx) => (
            <button
              key={m.name}
              onClick={() => setSelected(m.name)}
              className={clsx(
                'w-full text-left p-3 rounded-lg transition-all flex items-center justify-between gap-2 cursor-pointer',
                selected === m.name ? 'bg-brand-50 dark:bg-brand-900/20 ring-1 ring-brand-400/40' : 'hover:bg-stone-50 dark:hover:bg-navy-800/60'
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
                  <p className="truncate text-sm font-semibold text-stone-800 dark:text-stone-100">{m.displayName}</p>
                  <p className="text-[10px] text-stone-500 dark:text-stone-400">{m.family} &middot; R² {m.r2.toFixed(3)}</p>
                </div>
              </div>
              <ChevronRight className={clsx('w-4 h-4 shrink-0', selected === m.name ? 'text-brand-500' : 'text-stone-400 dark:text-stone-500')} />
            </button>
          ))}
        </div>

        <div className="lg:col-span-8 glass-card p-6">
          <div className="flex items-start justify-between gap-3 border-b border-stone-100 dark:border-navy-700 pb-4 mb-4">
            <div>
              <span
                className="inline-block mb-1.5 text-[11px] font-semibold px-2 py-0.5 rounded-full text-white"
                style={{ background: familyColor[model.family] }}
              >
                {model.family}
              </span>
              <h3 className="text-lg font-bold text-stone-800 dark:text-stone-100">{model.displayName}</h3>
              {model.name === 'xgboost' && (
                <p className="mt-0.5 text-xs font-semibold text-brand-600 dark:text-brand-400">Mô hình chính xác nhất (R² cao nhất)</p>
              )}
            </div>
            <div className="flex flex-col items-end shrink-0">
              <div className="text-xs font-semibold text-stone-500 dark:text-stone-400">R² kiểm tra</div>
              <div className="text-lg font-black text-brand-600 dark:text-brand-400">{model.r2.toFixed(3)}</div>
            </div>
          </div>

          <p className="text-sm text-stone-600 dark:text-stone-400 leading-relaxed mb-4">{model.description}</p>

          <div className="grid grid-cols-3 gap-2 text-center text-xs mb-4">
            {[
              { label: 'MAE (tỷ)', value: model.mae.toFixed(3) },
              { label: 'R²', value: model.r2.toFixed(3) },
              { label: 'MAPE Acc.', value: `${model.mape.toFixed(2)}%` },
            ].map(stat => (
              <div key={stat.label} className="rounded-lg border border-stone-200 dark:border-navy-700 p-2 bg-stone-50 dark:bg-navy-900/40">
                <div className="font-medium text-stone-500 dark:text-stone-400">{stat.label}</div>
                <div className="font-bold text-stone-800 dark:text-stone-100">{stat.value}</div>
              </div>
            ))}
          </div>

          <div className="space-y-3 border-t border-stone-100 dark:border-navy-700 pt-4 mb-4 text-xs">
            <div className="grid grid-cols-4 gap-2">
              <span className="col-span-1 font-semibold text-stone-800 dark:text-stone-100 flex items-center gap-1">
                <Shield className="w-3.5 h-3.5 text-brand-500" /> Biểu diễn
              </span>
              <span className="col-span-3 text-stone-500 dark:text-stone-400">{model.representation}</span>
            </div>
            <div className="grid grid-cols-4 gap-2">
              <span className="col-span-1 font-semibold text-stone-800 dark:text-stone-100 flex items-center gap-1">
                <AwardIcon className="w-3.5 h-3.5 text-brand-500" /> Tiêu chí học
              </span>
              <span className="col-span-3 text-stone-500 dark:text-stone-400">{model.criterion}</span>
            </div>
            <div className="grid grid-cols-4 gap-2">
              <span className="col-span-1 font-semibold text-stone-800 dark:text-stone-100 flex items-center gap-1">
                <HelpCircle className="w-3.5 h-3.5 text-brand-500" /> Giả định
              </span>
              <span className="col-span-3 text-stone-500 dark:text-stone-400">{model.assumptions}</span>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 border-t border-stone-100 dark:border-navy-700 pt-4 text-xs">
            <div>
              <h4 className="font-bold text-emerald-600 dark:text-emerald-400 flex items-center gap-1 mb-2">
                <CheckCircle className="w-3.5 h-3.5" /> Điểm mạnh
              </h4>
              <ul className="list-disc pl-4 space-y-1 text-stone-500 dark:text-stone-400">
                {model.strengths.map((s, i) => <li key={i}>{s}</li>)}
              </ul>
            </div>
            <div>
              <h4 className="font-bold text-red-600 dark:text-red-400 flex items-center gap-1 mb-2">
                <AlertTriangle className="w-3.5 h-3.5" /> Điểm yếu
              </h4>
              <ul className="list-disc pl-4 space-y-1 text-stone-500 dark:text-stone-400">
                {model.weaknesses.map((w, i) => <li key={i}>{w}</li>)}
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
