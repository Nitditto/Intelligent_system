import RiskHeadline from './RiskHeadline.jsx'
import ForcePlot from './ForcePlot.jsx'
import SimilarCases from './SimilarCases.jsx'
import WhatYouCanChange from './WhatYouCanChange.jsx'

export default function ResultPanel({ result, questions, answers, payload, sessionId }) {
  const qByField = Object.fromEntries((questions || []).map((q) => [q.field, q]))

  function describeAnswer(feature) {
    const q = qByField[feature]
    if (!q) return null
    const v = answers?.[feature]
    if (v === '' || v == null) return null
    if (q.type === 'yesno') return Number(v) === 1 ? 'Yes' : 'No'
    if (q.type === 'choice') {
      const opt = q.options.find(([code]) => Number(code) === Number(v))
      return opt ? opt[1] : null
    }
    return String(v)
  }

  const ex = result.explain && result.explain.factors ? result.explain : null
  const answered = Math.round((result.completeness ?? 1) * 21)

  return (
    <div>
      <RiskHeadline result={result} />

      {result.warnings?.map((w, i) => (
        <div className="notice warn" key={i}>
          {w}
        </div>
      ))}
      {answered < 21 && (
        <p className="muted small" style={{ marginTop: -8, marginBottom: 18 }}>
          You answered {answered} of 21 questions. Blanks were filled with typical
          values.
        </p>
      )}

      {ex && (
        <div className="card">
          <h2>Why this estimate</h2>
          <p className="card-hint">
            Each block is one of your answers. Red blocks pushed the estimate up from
            the population average; blue blocks pulled it down.
          </p>
          <ForcePlot
            factors={ex.all_factors || ex.factors}
            baseValue={ex.base_value}
            output={ex.predicted_probability ?? result.probability}
            describe={describeAnswer}
          />
        </div>
      )}

      <SimilarCases similar={result.similar} />

      <WhatYouCanChange
        whatif={result.whatif}
        counterfactual={result.counterfactual}
        payload={payload}
        sessionId={sessionId}
      />
    </div>
  )
}
