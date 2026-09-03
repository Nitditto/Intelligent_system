import React from 'react'
import ShapChart from './ShapChart.jsx'

const pct = (x) => `${Math.round(x * 100)}%`

function Signal({ status, label, value, note }) {
  return (
    <div className="sig">
      <span className={`sig__dot sig__dot--${status}`} aria-hidden="true">
        {status === 'bad' ? '!' : status === 'good' ? '✓' : status === 'warn' ? '•' : '–'}
      </span>
      <div className="sig__body">
        <div className="sig__line">
          <span className="sig__label">{label}</span>
          <span className="sig__value">{value}</span>
        </div>
        {note && <div className="sig__note">{note}</div>}
      </div>
    </div>
  )
}

export default function ResultScreen({ result, model, onRestart }) {
  const { prediction, p_satisfied, threshold, signals, contributions } = result
  const good = prediction === 'satisfied'
  const outcomeChance = good ? p_satisfied : 1 - p_satisfied
  const late = signals.late
  const delay = signals.days_vs_promise

  const textItems = (contributions?.items || []).filter((i) => i.kind === 'text')
  const negWords = textItems.filter((i) => i.effect < 0).map((i) => i.label.replace(/^comment: "|"$/g, ''))
  const posWords = textItems.filter((i) => i.effect > 0).map((i) => i.label.replace(/^comment: "|"$/g, ''))

  let reason
  if (late && delay != null) reason = `the order reached the customer ~${delay} day${delay === 1 ? '' : 's'} after the promised date`
  else if (!late && (signals.delivery_days ?? 0) > 20) reason = `delivery was on time but slow overall (${signals.delivery_days} days)`
  else if (!good && negWords.length) reason = `the wording of the customer's comment`
  else if (good && !late) reason = `the order arrived on time with nothing negative in the comment`
  else reason = `a mix of smaller factors`

  return (
    <div className="result-screen">
      <div className="result-screen__grid">
        {/* left: verdict + gauge + signals + action */}
        <div className="rs-col">
          <div className={`verdict verdict--${good ? 'good' : 'bad'}`}>
            <div className="verdict__stars">{good ? '★★★★' : '★☆☆'}</div>
            <div className="verdict__title">
              {good ? 'Positive review likely' : 'Negative review likely'}
            </div>
            <div className="verdict__lead">
              About <b>{pct(outcomeChance)}</b> chance this customer leaves a{' '}
              <b>{good ? '4–5★ (positive)' : '1–3★ (negative)'}</b> review.
            </div>
            <div className="verdict__why">Main driver: {reason}.</div>
          </div>

          <div className="gauge">
            <div className="gauge__top">
              <span className="gauge__label">Chance of a positive (4–5★) review</span>
              <span className="gauge__num">{pct(p_satisfied)}</span>
            </div>
            <div className="meter" role="img" aria-label={`Chance of a positive review ${pct(p_satisfied)}`}>
              <div className="meter__fill" style={{ width: `${p_satisfied * 100}%` }} />
              <div className="meter__cut" style={{ left: `${threshold * 100}%` }} />
            </div>
            <div className="gauge__scale">
              <span>0% · certain bad review</span>
              <span>100% · certain good review</span>
            </div>
            <div className="gauge__hint">
              Tick = the <b>{pct(threshold)}</b> cut-off; below it the order is flagged as at-risk —
              this one is {p_satisfied < threshold ? 'flagged' : 'not flagged'}.
            </div>
          </div>

          <div className="sig-block">
            <div className="sig-block__head">What the model looked at</div>
            <Signal status={late ? 'bad' : 'good'} label="Delivery vs. the promised date"
              value={delay == null ? 'unknown' : late ? `${delay} days late` : `${Math.abs(delay)} days early`}
              note={delay == null ? undefined : late
                ? 'Arriving after the promised date is the strongest cause of a bad review.'
                : 'Beating the promised date makes a good review much more likely.'} />
            <Signal status={(signals.delivery_days ?? 0) > 20 ? 'warn' : 'neutral'}
              label="Total time, order to doorstep" value={`${signals.delivery_days ?? '—'} days`} />
            <Signal status={signals.has_comment ? 'warn' : 'neutral'}
              label="Customer wrote a comment" value={signals.has_comment ? 'yes' : 'no'}
              note={signals.has_comment
                ? 'Customers who bother to write are more often unhappy — words shown in the chart.'
                : 'No comment text; the prediction rests on the order details only.'} />
            <Signal status="neutral" label="Product type · customer region"
              value={`${signals.category_group} · ${signals.customer_region}`} />
          </div>

          {(negWords.length || posWords.length) > 0 && (
            <div className="sig-block">
              <div className="sig-block__head">Words in the comment the model reacts to</div>
              {negWords.length > 0 && (
                <div className="terms">
                  <span className="terms__label is-bad">pulls toward a bad review</span>
                  <div className="chips">{negWords.map((t) => <span className="chip" key={t}>{t}</span>)}</div>
                </div>
              )}
              {posWords.length > 0 && (
                <div className="terms">
                  <span className="terms__label is-good">pulls toward a good review</span>
                  <div className="chips">{posWords.map((t) => <span className="chip" key={t}>{t}</span>)}</div>
                </div>
              )}
            </div>
          )}

          <div className={`todo todo--${good ? 'good' : 'bad'}`}>
            <span className="todo__k">Suggested action</span>
            {good
              ? 'Nothing — let the normal post-purchase flow run.'
              : 'Reach out to this customer or offer a goodwill voucher before they post the review, and check the shipment.'}
          </div>
        </div>

        {/* right: SHAP chart */}
        <div className="rs-col">
          {contributions?.items?.length
            ? <ShapChart contributions={contributions} />
            : <p className="muted">This model doesn&apos;t expose a per-feature breakdown.</p>}
        </div>
      </div>

      <div className="result-screen__foot">
        {model && (
          <span className="foot">
            model: <b>{model.chosen_model}</b> ({model.representation}) · flags orders below{' '}
            {Math.round((model.decision_threshold ?? 0.5) * 100)}% · scikit-learn {model.sklearn_version},
            seed {model.random_seed}
          </span>
        )}
        <button type="button" className="btn btn--ghost" onClick={onRestart}>Start over</button>
      </div>
    </div>
  )
}
