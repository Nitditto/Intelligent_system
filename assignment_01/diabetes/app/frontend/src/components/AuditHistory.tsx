import React from 'react';
import { History, RefreshCw, FileText, ChevronRight, Calendar } from 'lucide-react';
import clsx from 'clsx';

export type HistoryRecord = {
  id: string;
  timestamp: string;
  patientFeatures: Record<string, number>;
  consensus: {
    diabeticCount: number;
    healthyCount: number;
    percentage: number;
    verdict: string;
  };
};

interface AuditHistoryProps {
  history: HistoryRecord[];
  onRestore: (features: Record<string, number>) => void;
  onViewReport: (recordId: string) => void;
}

export const AuditHistory: React.FC<AuditHistoryProps> = ({ history, onRestore, onViewReport }) => {
  return (
    <div className="space-y-6">
      <div className="rounded-2xl bg-[var(--app-surface)] border border-[var(--app-border)] p-6 shadow-lg backdrop-blur-md">
        <h2 className="text-2xl font-black tracking-tight text-[var(--app-text)] flex items-center gap-2">
          <History className="w-6 h-6 text-[var(--app-primary)]" />
          Patient Screening History
        </h2>
        <p className="mt-2 text-sm text-[var(--app-text-muted)] max-w-2xl">
          Audit trail of clinical patient profiles evaluated during the current session. 
          Use this audit log to reload previous patient parameters or generate detailed clinical printouts.
        </p>
      </div>

      {history.length === 0 ? (
        <div className="rounded-2xl border border-[var(--app-border)] bg-[var(--app-surface-solid)] p-12 text-center shadow-md flex flex-col items-center justify-center gap-4">
          <div className="w-12 h-12 rounded-full bg-[var(--app-primary-soft)] text-[var(--app-primary)] flex items-center justify-center">
            <Calendar className="w-6 h-6" />
          </div>
          <div>
            <h3 className="text-base font-bold text-[var(--app-text)]">No History Found</h3>
            <p className="text-xs text-[var(--app-text-muted)] mt-1 max-w-xs">
              Complete patient profiles in the "Risk Predictor" tab to populate this audit registry.
            </p>
          </div>
        </div>
      ) : (
        <div className="rounded-2xl border border-[var(--app-border)] bg-[var(--app-surface-solid)] shadow-md overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse text-xs">
              <thead>
                <tr className="bg-[var(--app-bg)] text-[var(--app-text-muted)] font-bold uppercase tracking-wider border-b border-[var(--app-border)]">
                  <th className="p-4">Time</th>
                  <th className="p-4">Patient Indicators</th>
                  <th className="p-4">Consensus Outcome</th>
                  <th className="p-4 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[var(--app-border)]">
                {history.map((record) => {
                  const f = record.patientFeatures;
                  const dateStr = new Date(record.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
                  const dateDay = new Date(record.timestamp).toLocaleDateString([], { month: 'short', day: 'numeric' });
                  
                  return (
                    <tr key={record.id} className="hover:bg-[var(--app-primary-soft)] transition-colors">
                      {/* Timestamp */}
                      <td className="p-4 whitespace-nowrap">
                        <div className="font-semibold text-[var(--app-text)]">{dateStr}</div>
                        <div className="text-[10px] text-[var(--app-text-muted)] font-medium mt-0.5">{dateDay}</div>
                      </td>

                      {/* Patient features summary */}
                      <td className="p-4">
                        <div className="flex flex-wrap gap-2 text-[10px] font-semibold">
                          <span className="bg-[var(--app-bg)] px-2 py-0.5 rounded border border-[var(--app-border)] text-[var(--app-text)]">
                            Age: {f.Age}
                          </span>
                          <span className="bg-[var(--app-bg)] px-2 py-0.5 rounded border border-[var(--app-border)] text-[var(--app-text)]">
                            Glucose: {f.Glucose}
                          </span>
                          <span className="bg-[var(--app-bg)] px-2 py-0.5 rounded border border-[var(--app-border)] text-[var(--app-text)]">
                            BMI: {f.BMI}
                          </span>
                          <span className="bg-[var(--app-bg)] px-2 py-0.5 rounded border border-[var(--app-border)] text-[var(--app-text)]">
                            BP: {f.BloodPressure}
                          </span>
                        </div>
                      </td>

                      {/* Outcome Verdict */}
                      <td className="p-4 whitespace-nowrap">
                        <span className={clsx(
                          'inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full font-bold text-[10px]',
                          record.consensus.verdict === 'Healthy' 
                            ? 'bg-emerald-500/10 text-emerald-500 border border-emerald-500/20' 
                            : 'bg-rose-500/10 text-rose-500 border border-rose-500/20'
                        )}>
                          <span className={clsx('h-1.5 w-1.5 rounded-full', record.consensus.verdict === 'Healthy' ? 'bg-emerald-500' : 'bg-rose-500')} />
                          {record.consensus.verdict} ({record.consensus.percentage.toFixed(0)}% Consensus)
                        </span>
                      </td>

                      {/* Actions */}
                      <td className="p-4 text-right space-x-2 whitespace-nowrap">
                        <button
                          onClick={() => onRestore(record.patientFeatures)}
                          className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg border border-[var(--app-border)] bg-[var(--app-surface-solid)] text-[var(--app-text)] hover:border-[var(--app-primary)] hover:text-[var(--app-primary)] font-bold transition-colors cursor-pointer"
                          title="Restore parameters to the main prediction form"
                        >
                          <RefreshCw className="w-3.5 h-3.5" />
                          Reload
                        </button>
                        <button
                          onClick={() => onViewReport(record.id)}
                          className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg border border-transparent bg-[var(--app-primary)] text-white hover:bg-[var(--app-primary-strong)] font-bold transition-colors cursor-pointer"
                          title="Open structured clinical report workspace"
                        >
                          <FileText className="w-3.5 h-3.5" />
                          Report
                          <ChevronRight className="w-3 w-3" />
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
