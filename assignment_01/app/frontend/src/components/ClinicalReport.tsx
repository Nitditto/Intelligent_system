import React, { useState } from 'react';
import type { DetailedResult } from '../App';
import type { HistoryRecord } from './AuditHistory';
import { FileText, Printer, FileDown, ShieldCheck, HeartPulse, User, MapPin, Clipboard } from 'lucide-react';
import clsx from 'clsx';

interface ClinicalReportProps {
  activeRecord: HistoryRecord | null;
  detailedResults: Record<string, DetailedResult>;
  onExportTxt: () => void;
}

const formatModelName = (name: string) =>
  name.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');

export const ClinicalReport: React.FC<ClinicalReportProps> = ({
  activeRecord,
  detailedResults,
  onExportTxt
}) => {
  const [doctorName, setDoctorName] = useState('Dr. Alex Carter, MD');
  const [clinicName, setClinicName] = useState('Metro Health Diagnostic Center');
  const [patientNotes, setPatientNotes] = useState(
    'Patient presents with borderline indications. Recommend monitoring HbA1c levels annually. Moderate adjustment to dietary sugar and cardiovascular exercise (30 mins/day) suggested.'
  );

  if (!activeRecord) {
    return (
      <div className="rounded-2xl border border-[var(--app-border)] bg-[var(--app-surface-solid)] p-12 text-center shadow-md flex flex-col items-center justify-center gap-4">
        <div className="w-12 h-12 rounded-full bg-[var(--app-danger-soft)] text-[var(--app-danger)] flex items-center justify-center">
          <FileText className="w-6 h-6" />
        </div>
        <div>
          <h3 className="text-base font-bold text-[var(--app-text)]">No Active Session Loaded</h3>
          <p className="text-xs text-[var(--app-text-muted)] mt-1 max-w-xs mx-auto">
            Please run a patient prediction on the "Risk Predictor" tab or select an entry from "Audit History" to open this report workspace.
          </p>
        </div>
      </div>
    );
  }

  const f = activeRecord.patientFeatures;
  const isHealthy = activeRecord.consensus.verdict === 'Healthy';
  const rfResult = detailedResults['random_forest']; // Selected final model

  const handlePrint = () => {
    window.print();
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
      {/* Workspace editor side panel */}
      <div className="lg:col-span-4 rounded-2xl border border-[var(--app-border)] bg-[var(--app-surface-solid)] p-5 shadow-md space-y-5 no-print">
        <div className="border-b border-[var(--app-border)] pb-3">
          <h3 className="text-sm font-bold text-[var(--app-text)] flex items-center gap-1.5">
            <Clipboard className="w-4 h-4 text-[var(--app-primary)]" />
            Report Workspace Controls
          </h3>
          <p className="text-xs text-[var(--app-text-muted)] mt-0.5">Customize the clinical printout metadata</p>
        </div>

        <div className="space-y-4 text-xs font-semibold">
          <div className="flex flex-col gap-1.5">
            <label className="text-[var(--app-text-muted)] uppercase">Attending Clinician</label>
            <div className="relative">
              <User className="absolute left-3.5 top-3 w-4 h-4 text-[var(--app-text-muted)]" />
              <input
                type="text"
                value={doctorName}
                onChange={(e) => setDoctorName(e.target.value)}
                className="surface-control w-full pl-10 pr-3.5 py-2.5 rounded-xl font-bold transition-all shadow-sm text-xs"
              />
            </div>
          </div>

          <div className="flex flex-col gap-1.5">
            <label className="text-[var(--app-text-muted)] uppercase">Clinic / Hospital Facility</label>
            <div className="relative">
              <MapPin className="absolute left-3.5 top-3 w-4 h-4 text-[var(--app-text-muted)]" />
              <input
                type="text"
                value={clinicName}
                onChange={(e) => setClinicName(e.target.value)}
                className="surface-control w-full pl-10 pr-3.5 py-2.5 rounded-xl font-bold transition-all shadow-sm text-xs"
              />
            </div>
          </div>

          <div className="flex flex-col gap-1.5">
            <label className="text-[var(--app-text-muted)] uppercase">Clinical Comments &amp; Diagnosis Notes</label>
            <textarea
              rows={5}
              value={patientNotes}
              onChange={(e) => setPatientNotes(e.target.value)}
              className="surface-control w-full px-3.5 py-2.5 rounded-xl font-medium transition-all shadow-sm text-xs leading-relaxed resize-none"
              placeholder="Write diagnosis recommendations..."
            />
          </div>
        </div>

        <div className="grid grid-cols-2 gap-3 pt-2">
          <button
            onClick={handlePrint}
            className="primary-gradient text-white font-bold py-3 rounded-xl flex items-center justify-center gap-1.5 text-xs shadow-md shadow-primary-500/25 hover:-translate-y-0.5 active:translate-y-0 transition-transform cursor-pointer"
          >
            <Printer className="w-4 h-4" />
            Print / PDF
          </button>
          <button
            onClick={onExportTxt}
            className="export-button font-bold py-3 rounded-xl flex items-center justify-center gap-1.5 text-xs hover:-translate-y-0.5 active:translate-y-0 transition-transform cursor-pointer"
          >
            <FileDown className="w-4 h-4" />
            TXT Report
          </button>
        </div>
      </div>

      {/* Styled Printable Medical Report sheet */}
      <div className="lg:col-span-8 print-container rounded-2xl border border-[var(--app-border)] bg-white p-8 text-slate-800 shadow-xl space-y-8 min-h-[29.7cm] flex flex-col justify-between">
        <div className="space-y-6">
          {/* Medical Letterhead */}
          <div className="flex justify-between items-start border-b-2 border-slate-900 pb-5">
            <div className="flex items-center gap-2">
              <div className="h-10 w-10 bg-indigo-600 rounded-xl flex items-center justify-center text-white shrink-0 shadow-md">
                <HeartPulse className="w-5 h-5" />
              </div>
              <div>
                <h2 className="text-base font-black text-slate-900 uppercase tracking-tight">Clinical Screening Laboratory</h2>
                <p className="text-[10px] font-bold text-slate-500 tracking-wider uppercase mt-0.5">{clinicName}</p>
              </div>
            </div>
            <div className="text-right text-[10px] text-slate-500 font-bold uppercase tracking-wider space-y-0.5">
              <div>Session ID: {activeRecord.id.slice(0, 8)}</div>
              <div>Date: {new Date(activeRecord.timestamp).toLocaleDateString()}</div>
              <div>Time: {new Date(activeRecord.timestamp).toLocaleTimeString()}</div>
            </div>
          </div>

          <div className="text-center pt-2">
            <h1 className="text-lg font-black text-slate-900 uppercase tracking-wider">Diabetes Risk Diagnostic Assessment</h1>
            <p className="text-[10px] font-semibold text-slate-500 mt-1 uppercase">Evaluated using Multi-Classifier Machine Learning Suite</p>
          </div>

          {/* Patient Details & Clinical indicators grid */}
          <div className="space-y-3">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-900 border-b border-slate-200 pb-1.5">
              1. Patient Clinical Metrics
            </h3>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs">
              <div className="p-3 bg-slate-50 rounded-xl border border-slate-200">
                <span className="text-[9px] font-bold text-slate-500 block uppercase">Age</span>
                <span className="font-bold text-slate-900 mt-0.5 block">{f.Age} Years</span>
              </div>
              <div className="p-3 bg-slate-50 rounded-xl border border-slate-200">
                <span className="text-[9px] font-bold text-slate-500 block uppercase">Pregnancies</span>
                <span className="font-bold text-slate-900 mt-0.5 block">{f.Pregnancies} Times</span>
              </div>
              <div className="p-3 bg-slate-50 rounded-xl border border-slate-200">
                <span className="text-[9px] font-bold text-slate-500 block uppercase">Glucose Level</span>
                <span className="font-bold text-slate-900 mt-0.5 block">{f.Glucose} mg/dL</span>
              </div>
              <div className="p-3 bg-slate-50 rounded-xl border border-slate-200">
                <span className="text-[9px] font-bold text-slate-500 block uppercase">Body Mass Index</span>
                <span className="font-bold text-slate-900 mt-0.5 block">{f.BMI} kg/m²</span>
              </div>
              <div className="p-3 bg-slate-50 rounded-xl border border-slate-200">
                <span className="text-[9px] font-bold text-slate-500 block uppercase">Diastolic BP</span>
                <span className="font-bold text-slate-900 mt-0.5 block">{f.BloodPressure} mmHg</span>
              </div>
              <div className="p-3 bg-slate-50 rounded-xl border border-slate-200">
                <span className="text-[9px] font-bold text-slate-500 block uppercase">Skin Thickness</span>
                <span className="font-bold text-slate-900 mt-0.5 block">{f.SkinThickness} mm</span>
              </div>
              <div className="p-3 bg-slate-50 rounded-xl border border-slate-200">
                <span className="text-[9px] font-bold text-slate-500 block uppercase">Serum Insulin</span>
                <span className="font-bold text-slate-900 mt-0.5 block">{f.Insulin} mu U/ml</span>
              </div>
              <div className="p-3 bg-slate-50 rounded-xl border border-slate-200">
                <span className="text-[9px] font-bold text-slate-500 block uppercase">Family Pedigree</span>
                <span className="font-bold text-slate-900 mt-0.5 block">{f.DiabetesPedigreeFunction.toFixed(3)}</span>
              </div>
            </div>
          </div>

          {/* Model inferences comparison */}
          <div className="space-y-3">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-900 border-b border-slate-200 pb-1.5">
              2. Classifier Multi-Inference Results
            </h3>
            <div className="border border-slate-200 rounded-xl overflow-hidden">
              <table className="w-full text-left border-collapse text-xs">
                <thead>
                  <tr className="bg-slate-100 border-b border-slate-200 text-slate-500 font-bold uppercase tracking-wider">
                    <th className="p-3">Model</th>
                    <th className="p-3">Prediction</th>
                    <th className="p-3">Confidence</th>
                    <th className="p-3 text-right">Accuracy</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-200">
                  {Object.values(detailedResults).map((r) => (
                    <tr key={r.model}>
                      <td className="p-3 font-semibold text-slate-900">{formatModelName(r.model)}</td>
                      <td className="p-3">
                        <span className={clsx(
                          'px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wide',
                          r.prediction === 1 ? 'bg-rose-100 text-rose-700' : 'bg-emerald-100 text-emerald-700'
                        )}>
                          {r.prediction === 1 ? 'Diabetic' : 'Healthy'}
                        </span>
                      </td>
                      <td className="p-3 font-mono text-slate-600">{(r.confidence * 100).toFixed(1)}%</td>
                      <td className="p-3 font-mono text-slate-600 text-right">{(r.accuracy * 100).toFixed(1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Final selected model verdict & SHAP */}
          <div className="space-y-4">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-900 border-b border-slate-200 pb-1.5">
              3. Diagnostic Assessment &amp; AI Explainer
            </h3>
            <div className={clsx(
              'border rounded-xl p-4 flex gap-4 items-start',
              isHealthy ? 'bg-emerald-50 border-emerald-200 text-emerald-900' : 'bg-rose-50 border-rose-200 text-rose-900'
            )}>
              <ShieldCheck className={clsx('w-6 h-6 shrink-0 mt-0.5', isHealthy ? 'text-emerald-600' : 'text-rose-600')} />
              <div>
                <span className="text-xs font-black uppercase tracking-wider">Consensus Verdict: {activeRecord.consensus.verdict}</span>
                <p className="text-[11px] font-medium leading-relaxed mt-1 text-slate-600">
                  The clinical screening models converged on a **{activeRecord.consensus.verdict}** assessment with an consensus rate of **{activeRecord.consensus.percentage.toFixed(0)}%** ({isHealthy ? activeRecord.consensus.healthyCount : activeRecord.consensus.diabeticCount}/6 models).
                  {rfResult && ` The primary predictor model (Random Forest, Acc=${(rfResult.accuracy*100).toFixed(1)}%) predicts a ${(rfResult.confidence*100).toFixed(1)}% probability of ${rfResult.prediction === 1 ? 'Diabetes' : 'Healthy status'}.`}
                </p>
              </div>
            </div>

            {/* Top SHAP Attributions */}
            {rfResult && (
              <div className="space-y-2 text-xs">
                <span className="font-bold text-slate-600 uppercase text-[10px] tracking-wider block">Top Decision Drivers (Random Forest SHAP):</span>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                  {[...rfResult.shap_values].sort((a,b)=>Math.abs(b.value)-Math.abs(a.value)).slice(0,3).map((item) => (
                    <div key={item.feature} className="p-3 border border-slate-200 bg-slate-50 rounded-xl">
                      <span className="font-bold text-slate-900 block">{item.feature.replace(/([A-Z])/g, ' $1').trim()}</span>
                      <span className={clsx(
                        'text-[10px] font-bold block mt-1.5 uppercase',
                        item.value > 0 ? 'text-rose-600' : 'text-emerald-600'
                      )}>
                        {item.value > 0 ? `Raises Risk (+${item.value.toFixed(3)})` : `Lowers Risk (${item.value.toFixed(3)})`}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Clinician Notes section */}
          <div className="space-y-3">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-900 border-b border-slate-200 pb-1.5">
              4. Attending Clinician Recommendations
            </h3>
            <p className="text-xs leading-relaxed text-slate-700 bg-slate-50 p-4 border border-slate-200 rounded-xl italic whitespace-pre-wrap">
              "{patientNotes}"
            </p>
          </div>
        </div>

        {/* Doctor Signature Block */}
        <div className="flex justify-between items-end border-t border-slate-200 pt-8 mt-12 text-xs">
          <div className="text-slate-500 leading-relaxed max-w-[65%]">
            <span className="font-bold text-slate-900 uppercase text-[9px] tracking-wider block">Legal Medical Disclaimer:</span>
            This screening document is generated using advanced artificial intelligence and machine learning pipelines. It serves strictly as an auxiliary diagnostic aid. Confirmation via standard laboratory diagnostics (e.g., fasting HbA1c, OGTT) by a licensed clinician is mandatory before establishing any medical therapy.
          </div>
          <div className="text-center w-[180px]">
            <div className="border-b border-slate-400 h-10 w-full" />
            <span className="font-bold text-slate-900 mt-2 block">{doctorName}</span>
            <span className="text-[9px] font-semibold text-slate-400 uppercase tracking-wider mt-0.5 block">Authorized Signature</span>
          </div>
        </div>
      </div>
    </div>
  );
};
