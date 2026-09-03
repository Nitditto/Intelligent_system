import React from 'react'

const pct = (x) => `${Math.round(x * 100)}%`
const signed = (x) => `${x >= 0 ? '+' : '−'}${Math.abs(x).toFixed(2)}`

// Diverging bar chart of the linear-model contributions (linear SHAP).
// Each bar length = |log-odds effect|; left/red pushes toward a bad review,
// right/green toward a good one.
export default function ShapChart({ contributions }) {
  const { base_p, final_p, dataset_base_rate, items = [], other_effect = 0, note } = contributions
  if (!items.length) return null

  const maxAbs = Math.max(...items.map((i) => Math.abs(i.effect)), Math.abs(other_effect), 0.01)
  const w = (e) => `${(Math.abs(e) / maxAbs) * 48}%`
  const deltaPts = Math.round((final_p - base_p) * 100)

  const rows = [...items]
  if (Math.abs(other_effect) >= 0.01) {
    rows.push({ label: `${'other smaller factors'}`, kind: 'other', effect: other_effect })
  }

  return (
    <div className="shap">
      <div className="shap__head">Why this prediction — each factor's pull</div>

      <div className="shap__rows">
        {rows.map((it, i) => {
          const neg = it.effect < 0
          return (
            <div className="shap-row" key={i}>
              <span className="shap-row__lbl" title={it.label}>
                {it.kind === 'text' && <span className="shap-row__tag">comment</span>}
                {it.label}
              </span>
              <span className="shap-track">
                <i className="shap-track__axis" />
                <i
                  className={`shap-bar ${neg ? 'shap-bar--neg' : 'shap-bar--pos'}`}
                  style={neg ? { right: '50%', width: w(it.effect) } : { left: '50%', width: w(it.effect) }}
                />
              </span>
              <span className={`shap-row__val ${neg ? 'is-neg' : 'is-pos'}`}>{signed(it.effect)}</span>
            </div>
          )
        })}
      </div>

      <div className="shap__foot">
        <div className="shap__waterfall">
          <b>{pct(base_p)}</b>
          <span className="shap__arrow">→</span>
          <span className={deltaPts < 0 ? 'is-neg' : 'is-pos'}>
            {deltaPts >= 0 ? '+' : '−'}{Math.abs(deltaPts)} pts
          </span>
          <span className="shap__arrow">→</span>
          <b>{pct(final_p)}</b>
        </div>
        <p className="shap__note">
          Starts at the model&apos;s neutral point (<b>{pct(base_p)}</b> — it uses balanced class
          weights, so lower than the {pct(dataset_base_rate)} dataset positive rate); the factors
          above move it to <b>{pct(final_p)}</b>. Bar length is each factor&apos;s effect in
          log-odds.
        </p>
      </div>
    </div>
  )
}
