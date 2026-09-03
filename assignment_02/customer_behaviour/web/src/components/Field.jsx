import React from 'react'

// One labelled input, driven by a /questions field spec.
export default function Field({ f, value, onChange, invalid }) {
  const wide = f.type === 'text'
  return (
    <label className={`field${wide ? ' field--wide' : ''}${invalid ? ' field--invalid' : ''}`}>
      <span className="field__label">
        {f.label}
        {f.required && <span className="req" title="required">*</span>}
      </span>

      {f.type === 'choice' && (
        <select value={value ?? ''} onChange={(e) => onChange(f.field, e.target.value)}>
          <option value="">— not set —</option>
          {f.options.map((o) => (
            <option key={o} value={o}>{o === '__other__' ? 'other' : o}</option>
          ))}
        </select>
      )}

      {f.type === 'number' && (
        <input
          type="number" step="any" min={f.min} max={f.max} inputMode="decimal"
          value={value ?? ''}
          onChange={(e) => onChange(f.field, e.target.value)}
        />
      )}

      {f.type === 'date' && (
        <input
          type="datetime-local"
          value={value ?? ''}
          onChange={(e) => onChange(f.field, e.target.value)}
        />
      )}

      {f.type === 'text' && (
        <>
          <textarea
            rows={3}
            placeholder="e.g. Produto chegou com atraso…"
            value={value ?? ''}
            onChange={(e) => onChange(f.field, e.target.value)}
          />
          {f.examples?.length > 0 && (
            <div className="examples">
              <span className="examples__lbl">Try:</span>
              {f.examples.map(([ptxt, en]) => (
                <button
                  type="button" key={ptxt} className="examples__chip" title={en}
                  onClick={() => onChange(f.field, ptxt)}
                >
                  {ptxt.length > 40 ? ptxt.slice(0, 38) + '…' : ptxt}
                </button>
              ))}
            </div>
          )}
        </>
      )}

      {f.note && <span className="field__note">{f.note}</span>}
    </label>
  )
}
