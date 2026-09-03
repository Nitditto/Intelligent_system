import React, { useMemo } from 'react'
import Field from './Field.jsx'
import CategoryTile from './CategoryTile.jsx'
import DeliveryTimeline from './DeliveryTimeline.jsx'
import { deriveTimeline } from '../lib/order.js'

function Stepper({ labels, step, onJump }) {
  return (
    <ol className="stepper">
      {labels.map((label, i) => {
        const state = i < step ? 'done' : i === step ? 'current' : 'todo'
        const clickable = i < step
        return (
          <li key={label} className={`stepper__item is-${state}`}>
            {i > 0 && <span className="stepper__line" aria-hidden="true" />}
            <button
              type="button" className="stepper__btn" disabled={!clickable}
              aria-current={state === 'current' ? 'step' : undefined}
              onClick={() => clickable && onJump(i)}
            >
              <span className="stepper__dot">{state === 'done' ? '✓' : i + 1}</span>
              <span className="stepper__label">{label}</span>
            </button>
          </li>
        )
      })}
    </ol>
  )
}

export default function Wizard({ meta, values, setValues, step, setStep, onPredict, busy }) {
  const sections = meta.wizard_steps
  const N = sections.length // step indices 0..N-1 are field steps; step N is the summary
  const isSummary = step >= N
  const labels = [...sections, 'Review']

  const set = (k, v) => setValues((s) => ({ ...s, [k]: v }))
  const fieldsFor = (sec) => meta.fields.filter((f) => f.section === sec)

  const stepValid = useMemo(() => {
    if (isSummary) return true
    return fieldsFor(sections[step]).every((f) => {
      if (!f.required) return true
      const v = values[f.field]
      if (v === '' || v == null) return false
      if (f.field === 'price_total' && !(Number(v) > 0)) return false
      return true
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [step, values, meta])

  const missingInvalid = (f) => {
    if (!f.required) return false
    const v = values[f.field]
    if (v === '' || v == null) return true
    if (f.field === 'price_total' && !(Number(v) > 0)) return true
    return false
  }

  function leftVisual() {
    if (step === 0) return <CategoryTile category={values.category} />
    if (sections[step] === 'Payment') {
      return (
        <div className="viz-tile">
          <div className="viz-tile__glyph" aria-hidden="true">💳</div>
          <div className="viz-tile__name">
            {values.main_payment_type ? values.main_payment_type.replace(/_/g, ' ') : 'payment'}
          </div>
          <div className="viz-tile__cap">
            {values.customer_state ? `ships to ${values.customer_state}` : 'customer region'}
          </div>
        </div>
      )
    }
    return <DeliveryTimeline values={values} />
  }

  return (
    <div className="wiz">
      <Stepper labels={labels} step={step} onJump={setStep} />
      <p className="wiz__count">
        Step {Math.min(step + 1, N + 1)} of {N + 1} · <b>{labels[Math.min(step, N)]}</b>
      </p>

      <div className="wiz-panel">
        {isSummary ? (
          <Summary meta={meta} values={values} />
        ) : (
          <div className="step__body">
            <aside className="step__viz">{leftVisual()}</aside>
            <div className="step__fields">
              <h2 className="step__title">{sections[step]}</h2>
              <div className="fieldgrid">
                {fieldsFor(sections[step]).map((f) => (
                  <Field
                    key={f.field} f={f} value={values[f.field]} onChange={set}
                    invalid={missingInvalid(f)}
                  />
                ))}
              </div>
            </div>
          </div>
        )}

        <div className="step__nav">
          <button
            type="button" className="btn btn--ghost"
            onClick={() => setStep((s) => Math.max(0, s - 1))} disabled={step === 0}
          >
            ← Back
          </button>
          {isSummary ? (
            <button type="button" className="btn btn--primary btn--lg" onClick={onPredict} disabled={busy}>
              {busy && <span className="spinner" aria-hidden="true" />}
              {busy ? 'Predicting…' : 'Predict this order'}
            </button>
          ) : (
            <button
              type="button" className="btn btn--primary"
              onClick={() => setStep((s) => s + 1)} disabled={!stepValid}
              title={stepValid ? '' : 'Fill the required fields first'}
            >
              Next →
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

function Summary({ meta, values }) {
  const fmt = (f) => {
    const v = values[f.field]
    if (v === '' || v == null) return '—'
    if (f.type === 'date') return String(v).replace('T', ' ')
    if (f.field === 'review_comment_message') return v.length > 90 ? v.slice(0, 88) + '…' : v
    return String(v)
  }
  const { daysLate, deliveryDays } = deriveTimeline(values)
  return (
    <div className="summary">
      <h2 className="step__title">Review &amp; predict</h2>
      {Number.isFinite(daysLate) && (
        <p className={`summary__lead ${daysLate > 0.5 ? 'is-bad' : 'is-good'}`}>
          Delivery: <b>{daysLate > 0.5 ? `${daysLate} days late` : `${Math.abs(daysLate)} days early`}</b>
          {Number.isFinite(deliveryDays) ? ` · ${deliveryDays} days door to door` : ''}
        </p>
      )}
      {meta.wizard_steps.map((sec) => (
        <div className="summary__group" key={sec}>
          <div className="summary__group-head">{sec}</div>
          <dl className="summary__kv">
            {meta.fields.filter((f) => f.section === sec).map((f) => (
              <div key={f.field}>
                <dt>{f.label}</dt>
                <dd>{fmt(f)}</dd>
              </div>
            ))}
          </dl>
        </div>
      ))}
      <p className="summary__assume">
        Assumed (not asked): listing description length ={' '}
        <b>{meta.fixed_inputs?.product_desc_len}</b> chars · payment methods used ={' '}
        <b>{meta.fixed_inputs?.n_payment_types}</b>. Blank optional fields use dataset-typical values.
      </p>
    </div>
  )
}
