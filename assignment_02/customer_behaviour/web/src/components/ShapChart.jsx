import React from 'react'

export default function ShapChart({ contributions }) {
  if (!contributions || !contributions.items || !contributions.items.length) {
    return null
  }

  const { base_p, final_p, items, other_effect } = contributions
  const rows = [...items]
  if (other_effect && Math.abs(other_effect) >= 0.01) {
    rows.push({ label: 'Other smaller factors combined', kind: 'other', effect: other_effect })
  }

  const maxAbs = Math.max(...rows.map((r) => Math.abs(r.effect)), 0.4)
  const width = (e) => `${Math.max(4, (Math.abs(e) / maxAbs) * 100)}%`
  const sign = (e) => (e >= 0 ? `+${e.toFixed(2)}` : e.toFixed(2))

  const basePct = Math.round(base_p * 100)
  const finalPct = Math.round(final_p * 100)
  const netDelta = finalPct - basePct

  return (
    <div className="shap-box">
      <div className="shap-box__head">
        <h3 className="shap-box__title">Why this prediction — each factor&apos;s pull</h3>
        <p className="shap-box__sub">
          <span className="shap-legend shap-legend--neg" /> pulls toward a bad review &nbsp;·&nbsp;
          <span className="shap-legend shap-legend--pos" /> pulls toward a good one
        </p>
      </div>

      <div className="shap-rows">
        {rows.map((it, idx) => {
          const neg = it.effect < 0
          return (
            <div key={idx} className={`shap-row${it.kind === 'other' ? ' shap-row--other' : ''}`}>
              <div className="shap-row__label">
                <span className={`shap-tag shap-tag--${it.kind === 'text' ? 'txt' : it.kind === 'other' ? 'other' : 'tab'}`}>
                  {it.kind === 'text' ? 'comment' : it.kind === 'other' ? 'other' : 'order'}
                </span>
                <span className="shap-row__name" title={it.label}>{it.label}</span>
              </div>
              <div className="shap-row__track">
                <span
                  className={`shap-bar shap-bar--${neg ? 'neg' : 'pos'}`}
                  style={{ width: width(it.effect) }}
                />
              </div>
              <span className={`shap-row__val ${neg ? 'text-bad' : 'text-good'}`}>{sign(it.effect)}</span>
            </div>
          )
        })}
      </div>

      <div className="shap-waterfall">
        <div className="shap-wf-step">
          <span className="shap-wf-lbl">Neutral starting point</span>
          <span className="shap-wf-val">{basePct}%</span>
        </div>
        <span className="shap-wf-arrow">→</span>
        <div className="shap-wf-step">
          <span className="shap-wf-lbl">Net pull of the factors</span>
          <span className={`shap-wf-val ${netDelta >= 0 ? 'text-good' : 'text-bad'}`}>
            {netDelta >= 0 ? `+${netDelta}` : netDelta} pts
          </span>
        </div>
        <span className="shap-wf-arrow">→</span>
        <div className="shap-wf-step shap-wf-step--final">
          <span className="shap-wf-lbl">Final chance of a good review</span>
          <span className="shap-wf-val">{finalPct}%</span>
        </div>
      </div>

      <p className="shap-footnote">
        Bars are each factor&apos;s effect in log-odds. The starting point ({basePct}%) is the
        model&apos;s neutral prediction — lower than the {Math.round((contributions.dataset_base_rate ?? 0.79) * 100)}%
        dataset average because the model treats the rarer &ldquo;bad review&rdquo; class as
        equally important.
      </p>
    </div>
  )
}
