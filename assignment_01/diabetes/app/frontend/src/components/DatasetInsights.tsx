import React, { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter, ZAxis } from 'recharts';
import { Database, Filter, Layers, PieChart, Activity } from 'lucide-react';

const averagesData = [
  { name: 'Glucose (mg/dL)', NonDiabetic: 110.0, Diabetic: 141.3 },
  { name: 'BMI (kg/m²)', NonDiabetic: 30.3, Diabetic: 35.1 },
  { name: 'Age (Years)', NonDiabetic: 31.2, Diabetic: 37.1 },
  { name: 'Blood Pressure', NonDiabetic: 68.2, Diabetic: 70.8 },
  { name: 'Insulin (mu U/ml)', NonDiabetic: 68.8, Diabetic: 100.3 }
];

const sampleScatterData = [
  {"Glucose":98,"BMI":34.0,"Age":43,"Outcome":0},
  {"Glucose":112,"BMI":35.7,"Age":21,"Outcome":0},
  {"Glucose":108,"BMI":30.8,"Age":21,"Outcome":0},
  {"Glucose":107,"BMI":24.6,"Age":34,"Outcome":0},
  {"Glucose":136,"BMI":29.9,"Age":50,"Outcome":0},
  {"Glucose":103,"BMI":37.7,"Age":55,"Outcome":0},
  {"Glucose":71,"BMI":20.4,"Age":22,"Outcome":0},
  {"Glucose":117,"BMI":33.8,"Age":44,"Outcome":0},
  {"Glucose":154,"BMI":31.3,"Age":37,"Outcome":0},
  {"Glucose":147,"BMI":33.7,"Age":65,"Outcome":0},
  {"Glucose":111,"BMI":27.5,"Age":40,"Outcome":1},
  {"Glucose":179,"BMI":34.2,"Age":60,"Outcome":0},
  {"Glucose":148,"BMI":30.9,"Age":29,"Outcome":1},
  {"Glucose":96,"BMI":33.6,"Age":43,"Outcome":0},
  {"Glucose":88,"BMI":28.4,"Age":22,"Outcome":0},
  {"Glucose":125,"BMI":33.3,"Age":28,"Outcome":1},
  {"Glucose":84,"BMI":37.2,"Age":28,"Outcome":0},
  {"Glucose":86,"BMI":30.2,"Age":24,"Outcome":0},
  {"Glucose":183,"BMI":28.4,"Age":36,"Outcome":1},
  {"Glucose":140,"BMI":42.6,"Age":24,"Outcome":1},
  {"Glucose":104,"BMI":28.8,"Age":48,"Outcome":0},
  {"Glucose":88,"BMI":32.0,"Age":29,"Outcome":0},
  {"Glucose":106,"BMI":36.6,"Age":45,"Outcome":0},
  {"Glucose":96,"BMI":33.2,"Age":21,"Outcome":0},
  {"Glucose":129,"BMI":35.9,"Age":39,"Outcome":0},
  {"Glucose":184,"BMI":35.5,"Age":41,"Outcome":1},
  {"Glucose":109,"BMI":25.0,"Age":27,"Outcome":0},
  {"Glucose":100,"BMI":19.5,"Age":28,"Outcome":0},
  {"Glucose":102,"BMI":45.5,"Age":23,"Outcome":1},
  {"Glucose":89,"BMI":30.4,"Age":38,"Outcome":0},
  {"Glucose":162,"BMI":53.2,"Age":25,"Outcome":1},
  {"Glucose":146,"BMI":38.5,"Age":67,"Outcome":1},
  {"Glucose":184,"BMI":37.0,"Age":31,"Outcome":1},
  {"Glucose":167,"BMI":32.3,"Age":30,"Outcome":1},
  {"Glucose":109,"BMI":42.7,"Age":54,"Outcome":0},
  {"Glucose":152,"BMI":34.2,"Age":33,"Outcome":1},
  {"Glucose":197,"BMI":34.7,"Age":62,"Outcome":1},
  {"Glucose":109,"BMI":32.5,"Age":38,"Outcome":1},
  {"Glucose":121,"BMI":26.5,"Age":62,"Outcome":0},
  {"Glucose":131,"BMI":43.2,"Age":26,"Outcome":1}
];

const ageDistributionData = [
  { range: '21-25', NonDiabetic: 198, Diabetic: 32 },
  { range: '26-30', NonDiabetic: 86, Diabetic: 45 },
  { range: '31-35', NonDiabetic: 51, Diabetic: 40 },
  { range: '36-40', NonDiabetic: 39, Diabetic: 36 },
  { range: '41-45', NonDiabetic: 24, Diabetic: 32 },
  { range: '46-50', NonDiabetic: 21, Diabetic: 26 },
  { range: '51-55', NonDiabetic: 15, Diabetic: 24 },
  { range: '56+', NonDiabetic: 16, Diabetic: 15 }
];

// 9x9 Pearson Correlation Matrix Data
const heatmapVariables = ['Preg', 'Gluc', 'BP', 'Skin', 'Ins', 'BMI', 'Pedig', 'Age', 'Outc'];
const heatmapFullNames = {
  Preg: 'Pregnancies',
  Gluc: 'Glucose',
  BP: 'Blood Pressure',
  Skin: 'Skin Thickness',
  Ins: 'Insulin',
  BMI: 'Body Mass Index',
  Pedig: 'Diabetes Pedigree',
  Age: 'Age',
  Outc: 'Outcome (Diabetes)'
};

const correlationMatrix: Record<string, Record<string, number>> = {
  Preg: { Preg: 1.0, Gluc: 0.13, BP: 0.14, Skin: -0.08, Ins: -0.07, BMI: 0.02, Pedig: -0.03, Age: 0.54, Outc: 0.22 },
  Gluc: { Preg: 0.13, Gluc: 1.0, BP: 0.15, Skin: 0.06, Ins: 0.33, BMI: 0.22, Pedig: 0.14, Age: 0.26, Outc: 0.47 },
  BP: { Preg: 0.14, Gluc: 0.15, BP: 1.0, Skin: 0.21, Ins: 0.09, BMI: 0.28, Pedig: 0.04, Age: 0.24, Outc: 0.07 },
  Skin: { Preg: -0.08, Gluc: 0.06, BP: 0.21, Skin: 1.0, Ins: 0.44, BMI: 0.39, Pedig: 0.18, Age: -0.11, Outc: 0.07 },
  Ins: { Preg: -0.07, Gluc: 0.33, BP: 0.09, Skin: 0.44, Ins: 1.0, BMI: 0.2, Pedig: 0.19, Age: -0.04, Outc: 0.13 },
  BMI: { Preg: 0.02, Gluc: 0.22, BP: 0.28, Skin: 0.39, Ins: 0.2, BMI: 1.0, Pedig: 0.14, Age: 0.04, Outc: 0.29 },
  Pedig: { Preg: -0.03, Gluc: 0.14, BP: 0.04, Skin: 0.18, Ins: 0.19, BMI: 0.14, Pedig: 1.0, Age: 0.03, Outc: 0.17 },
  Age: { Preg: 0.54, Gluc: 0.26, BP: 0.24, Skin: -0.11, Ins: -0.04, BMI: 0.04, Pedig: 0.03, Age: 1.0, Outc: 0.24 },
  Outc: { Preg: 0.22, Gluc: 0.47, BP: 0.07, Skin: 0.07, Ins: 0.13, BMI: 0.29, Pedig: 0.17, Age: 0.24, Outc: 1.0 }
};

export const DatasetInsights: React.FC = () => {
  const [filterOutcome, setFilterOutcome] = useState<'all' | 'healthy' | 'diabetic'>('all');

  const filteredScatter = sampleScatterData.filter(d => {
    if (filterOutcome === 'healthy') return d.Outcome === 0;
    if (filterOutcome === 'diabetic') return d.Outcome === 1;
    return true;
  });

  const healthyScatter = filteredScatter.filter(d => d.Outcome === 0);
  const diabeticScatter = filteredScatter.filter(d => d.Outcome === 1);

  // Helper to color cell background according to Pearson coefficient
  const getCellBg = (val: number) => {
    if (val === 1.0) return 'rgba(30, 41, 59, 0.9)'; // Diagonal
    if (val > 0) {
      // Scale positive correlations to indigo
      return `rgba(99, 102, 241, ${val * 0.95})`;
    } else {
      // Scale negative correlations to rose
      return `rgba(244, 63, 94, ${Math.abs(val) * 0.95})`;
    }
  };

  const getCellTextColor = (val: number) => {
    if (Math.abs(val) > 0.4) return '#ffffff';
    return 'var(--app-text)';
  };

  const getCorrelationDescription = (v1: string, v2: string, r: number) => {
    const name1 = heatmapFullNames[v1 as keyof typeof heatmapFullNames];
    const name2 = heatmapFullNames[v2 as keyof typeof heatmapFullNames];
    if (v1 === v2) return `${name1} correlates perfectly with itself (1.00).`;
    
    let strength = 'negligible';
    const absVal = Math.abs(r);
    if (absVal > 0.5) strength = 'strong';
    else if (absVal > 0.3) strength = 'moderate';
    else if (absVal > 0.1) strength = 'weak';

    const direction = r > 0 ? 'positive' : 'negative';
    let text = `${name1} has a ${strength} ${direction} correlation of ${r.toFixed(2)} with ${name2}.`;
    
    if (v2 === 'Outc') {
      text += ` This indicates that higher ${name1} measurements statistically ${r > 0 ? 'increase' : 'decrease'} diabetes probability in the patient cohort.`;
    }
    return text;
  };

  return (
    <div className="space-y-6">
      {/* Overview Banner */}
      <div className="rounded-2xl bg-[var(--app-surface)] border border-[var(--app-border)] p-6 shadow-lg backdrop-blur-md">
        <h2 className="text-2xl font-black tracking-tight text-[var(--app-text)] flex items-center gap-2">
          <Database className="w-6 h-6 text-[var(--app-primary)]" />
          Pima Indian Diabetes Dataset Explorer
        </h2>
        <p className="mt-2 text-sm text-[var(--app-text-muted)] max-w-2xl">
          Visualizing the statistics of the 768 patient cases. This exploratory view showcases how glucose level, 
          body mass index, and age serve as major statistical markers for predicting diabetes.
        </p>

        {/* Info stats */}
        <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="rounded-xl border border-[var(--app-border)] bg-[var(--app-surface-solid)] p-4 shadow-sm">
            <div className="text-xs font-semibold text-[var(--app-text-muted)] uppercase">Observations</div>
            <div className="mt-1 text-xl font-bold text-[var(--app-text)]">768 Patients</div>
          </div>
          <div className="rounded-xl border border-[var(--app-border)] bg-[var(--app-surface-solid)] p-4 shadow-sm">
            <div className="text-xs font-semibold text-[var(--app-text-muted)] uppercase">Features</div>
            <div className="mt-1 text-xl font-bold text-[var(--app-text)]">8 Clinical Indicators</div>
          </div>
          <div className="rounded-xl border border-[var(--app-border)] bg-[var(--app-surface-solid)] p-4 shadow-sm">
            <div className="text-xs font-semibold text-[var(--app-text-muted)] uppercase">Outcome Type</div>
            <div className="mt-1 text-xl font-bold text-[var(--app-text)]">Binary Classifier</div>
          </div>
          <div className="rounded-xl border border-[var(--app-border)] bg-[var(--app-surface-solid)] p-4 shadow-sm">
            <div className="text-xs font-semibold text-[var(--app-text-muted)] uppercase">Diabetes Ratio</div>
            <div className="mt-1 text-xl font-bold text-[var(--app-danger)]">34.9% positive</div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Averages comparison */}
        <div className="rounded-2xl border border-[var(--app-border)] bg-[var(--app-surface-solid)] p-5 shadow-md lg:col-span-6 space-y-4">
          <div>
            <h3 className="text-base font-bold text-[var(--app-text)] flex items-center gap-1.5 mb-2">
              <PieChart className="w-5 h-5 text-[var(--app-primary)]" />
              Outcome-wise Feature Averages
            </h3>
            <p className="text-xs text-[var(--app-text-muted)]">
              Comparison of average values between healthy and diabetic cohorts. Note the significant difference in glucose levels.
            </p>
          </div>
          <div className="h-[260px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={averagesData} margin={{ top: 10, right: 10, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--app-border)" />
                <XAxis dataKey="name" stroke="var(--app-text-muted)" fontSize={10} tickLine={false} />
                <YAxis stroke="var(--app-text-muted)" fontSize={11} width={36} tickLine={false} />
                <Tooltip contentStyle={{ background: 'var(--app-surface-solid)', borderColor: 'var(--app-border)', borderRadius: '12px' }} />
                <Legend verticalAlign="top" height={28} wrapperStyle={{ fontSize: 11 }} />
                <Bar dataKey="NonDiabetic" name="Non-Diabetic" fill="#06b6d4" radius={[4, 4, 0, 0]} />
                <Bar dataKey="Diabetic" name="Diabetic" fill="#6366f1" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Age distribution */}
        <div className="rounded-2xl border border-[var(--app-border)] bg-[var(--app-surface-solid)] p-5 shadow-md lg:col-span-6 space-y-4">
          <div>
            <h3 className="text-base font-bold text-[var(--app-text)] flex items-center gap-1.5 mb-2">
              <Layers className="w-5 h-5 text-[var(--app-primary)]" />
              Patient Age &amp; Diabetes Ratio
            </h3>
            <p className="text-xs text-[var(--app-text-muted)]">
              Number of healthy vs diabetic cases across age groups. Diabetes rates tend to rise in the 26-45 range.
            </p>
          </div>
          <div className="h-[260px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={ageDistributionData} stackOffset="sign" margin={{ top: 10, right: 10, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--app-border)" />
                <XAxis dataKey="range" stroke="var(--app-text-muted)" fontSize={10} tickLine={false} />
                <YAxis stroke="var(--app-text-muted)" fontSize={11} width={32} tickLine={false} />
                <Tooltip contentStyle={{ background: 'var(--app-surface-solid)', borderColor: 'var(--app-border)', borderRadius: '12px' }} />
                <Legend verticalAlign="top" height={28} wrapperStyle={{ fontSize: 11 }} />
                <Bar dataKey="NonDiabetic" name="Non-Diabetic" fill="#06b6d4" stackId="a" radius={[0, 0, 0, 0]} />
                <Bar dataKey="Diabetic" name="Diabetic" fill="#6366f1" stackId="a" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Pearson Correlation Heatmap */}
        <div className="rounded-2xl border border-[var(--app-border)] bg-[var(--app-surface-solid)] p-5 shadow-md lg:col-span-6 space-y-4">
          <div>
            <h3 className="text-base font-bold text-[var(--app-text)] flex items-center gap-1.5 mb-2">
              <Activity className="w-5 h-5 text-[var(--app-primary)]" />
              Pearson Correlation Matrix
            </h3>
            <p className="text-xs text-[var(--app-text-muted)]">
              Pearson correlation coefficients between variables. Hover over the cells to read the correlation strength and clinical meaning.
            </p>
          </div>

          <div className="w-full flex flex-col items-center">
            {/* Column Headers */}
            <div className="grid grid-cols-10 w-full mb-1">
              <div className="col-span-1" />
              {heatmapVariables.map((v) => (
                <div key={v} className="col-span-1 text-[9px] font-black text-[var(--app-text-muted)] uppercase text-center truncate px-0.5" title={heatmapFullNames[v as keyof typeof heatmapFullNames]}>
                  {v}
                </div>
              ))}
            </div>

            {/* Matrix Grid Rows */}
            <div className="space-y-1 w-full">
              {heatmapVariables.map((rowVar) => (
                <div key={rowVar} className="grid grid-cols-10 items-center w-full">
                  {/* Row Header */}
                  <div className="col-span-1 text-[9px] font-black text-[var(--app-text-muted)] uppercase truncate pr-1" title={heatmapFullNames[rowVar as keyof typeof heatmapFullNames]}>
                    {rowVar}
                  </div>
                  
                  {/* Row Cells */}
                  {heatmapVariables.map((colVar) => {
                    const corrVal = correlationMatrix[rowVar][colVar];
                    return (
                      <div 
                        key={colVar} 
                        className="col-span-1 p-0.5"
                      >
                        <div 
                          className="heatmap-cell"
                          style={{
                            backgroundColor: getCellBg(corrVal),
                            color: getCellTextColor(corrVal)
                          }}
                        >
                          {corrVal.toFixed(2)}
                          
                          {/* Hover Tooltip */}
                          <div className="heatmap-tooltip">
                            <span className="font-bold text-[var(--app-primary-strong)] text-[10px] block border-b border-slate-700 pb-1 mb-1">
                              Pearson Correlation Matrix
                            </span>
                            <span className="leading-relaxed">
                              {getCorrelationDescription(rowVar, colVar, corrVal)}
                            </span>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              ))}
            </div>

            {/* Heatmap Legend */}
            <div className="flex justify-between items-center w-full mt-4 text-[9px] font-bold text-[var(--app-text-muted)] border-t border-[var(--app-border)] pt-2.5">
              <div className="flex items-center gap-1">
                <span className="h-2.5 w-2.5 rounded bg-rose-500" />
                Negative Correlation
              </div>
              <div className="flex items-center gap-1">
                <span className="h-2.5 w-2.5 rounded bg-indigo-500" />
                Positive Correlation
              </div>
              <div className="flex items-center gap-1">
                <span className="h-2.5 w-2.5 rounded bg-slate-800" />
                Diagonal (1.0)
              </div>
            </div>
          </div>
        </div>

        {/* Scatter Plot: Glucose vs BMI */}
        <div className="rounded-2xl border border-[var(--app-border)] bg-[var(--app-surface-solid)] p-5 shadow-md lg:col-span-6 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <h3 className="text-base font-bold text-[var(--app-text)] flex items-center gap-1.5">
                <Filter className="w-5 h-5 text-[var(--app-primary)]" />
                Glucose vs. BMI Scatter Distribution
              </h3>
              <p className="text-xs text-[var(--app-text-muted)] mt-1">
                Sample patient distribution — diabetic points cluster toward higher Glucose and BMI.
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2 bg-[var(--app-bg)] rounded-lg p-1 border border-[var(--app-border)] text-[10px] font-black uppercase tracking-wider w-fit">
            <button
              onClick={() => setFilterOutcome('all')}
              className={`px-2.5 py-1 rounded-md transition-colors cursor-pointer ${filterOutcome === 'all' ? 'bg-[var(--app-surface-solid)] text-[var(--app-primary)] shadow-sm' : 'text-[var(--app-text-muted)]'}`}
            >
              All
            </button>
            <button
              onClick={() => setFilterOutcome('healthy')}
              className={`px-2.5 py-1 rounded-md transition-colors cursor-pointer ${filterOutcome === 'healthy' ? 'bg-[var(--app-surface-solid)] shadow-sm' : 'text-[var(--app-text-muted)]'}`}
              style={filterOutcome === 'healthy' ? { color: '#06b6d4' } : undefined}
            >
              Healthy
            </button>
            <button
              onClick={() => setFilterOutcome('diabetic')}
              className={`px-2.5 py-1 rounded-md transition-colors cursor-pointer ${filterOutcome === 'diabetic' ? 'bg-[var(--app-surface-solid)] shadow-sm' : 'text-[var(--app-text-muted)]'}`}
              style={filterOutcome === 'diabetic' ? { color: '#6366f1' } : undefined}
            >
              Diabetic
            </button>
          </div>

          <div className="h-[260px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 15, right: 15, bottom: 5, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--app-border)" />
                <XAxis
                  type="number" dataKey="Glucose" name="Glucose" unit=" mg/dL"
                  stroke="var(--app-text-muted)" fontSize={10} tickLine={false}
                  domain={['dataMin - 10', 'dataMax + 10']}
                />
                <YAxis
                  type="number" dataKey="BMI" name="BMI" unit=" kg/m²"
                  stroke="var(--app-text-muted)" fontSize={10} tickLine={false} width={48}
                  domain={['dataMin - 5', 'dataMax + 5']}
                />
                <ZAxis type="number" range={[40, 40]} />
                <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ background: 'var(--app-surface-solid)', borderColor: 'var(--app-border)', borderRadius: '12px', fontSize: 11 }} />
                <Legend verticalAlign="top" height={28} wrapperStyle={{ fontSize: 11 }} />
                {filterOutcome !== 'diabetic' && (
                  <Scatter name="Healthy" data={healthyScatter} fill="#06b6d4" shape="circle" />
                )}
                {filterOutcome !== 'healthy' && (
                  <Scatter name="Diabetic" data={diabeticScatter} fill="#6366f1" shape="circle" />
                )}
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
};
