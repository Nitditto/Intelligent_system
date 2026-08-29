import { useMemo, useState } from 'react'

// yes/no answers default to "No" (0) - the natural resting state for a screening
// form. Optional dropdowns default to "skip". Mental/physical bad-day counts
// default to 0. Nothing here uses a "not sure" checkbox.
function initialAnswers(questions) {
  const a = {}
  for (const q of questions) {
    if (q.type === 'yesno') a[q.field] = 0
    else if (q.type === 'choice') a[q.field] = ''
    else if (q.field === 'MentHlth' || q.field === 'PhysHlth') a[q.field] = 0
    else a[q.field] = ''
  }
  return a
}

export default function Questionnaire({ questions, busy, onSubmit }) {
  const [answers, setAnswers] = useState(() => initialAnswers(questions))
  const [touched, setTouched] = useState(false)

  const sections = useMemo(() => {
    const m = new Map()
    for (const q of questions) {
      if (!m.has(q.section)) m.set(q.section, [])
      m.get(q.section).push(q)
    }
    return [...m.entries()]
  }, [questions])

  const set = (field, value) => setAnswers((s) => ({ ...s, [field]: value }))

  const missingRequired = ['Age', 'Sex'].filter((f) => answers[f] === '' || answers[f] == null)

  function submit(e) {
    e.preventDefault()
    setTouched(true)
    if (missingRequired.length) {
      document.getElementById('field-' + missingRequired[0])?.scrollIntoView({
        behavior: 'smooth',
        block: 'center',
      })
      return
    }
    const payload = {}
    for (const q of questions) {
      const v = answers[q.field]
      if (q.type === 'yesno') {
        payload[q.field] = Number(v)
      } else if (q.type === 'choice') {
        if (v !== '' && v != null) payload[q.field] = Number(v)
      } else {
        if (v !== '' && v != null && !Number.isNaN(Number(v))) payload[q.field] = Number(v)
      }
    }
    onSubmit(payload, answers)
  }

  return (
    <form className="card" onSubmit={submit}>
      <h2>Health questionnaire</h2>
      <p className="card-hint">
        Answer what applies to you. Anything you leave blank is filled with a typical
        value and the result is marked as less certain. Fields marked * are required.
      </p>

      {sections.map(([name, qs]) => (
        <div key={name}>
          <div className="section-label">{name}</div>
          <div className="grid-2">
            {qs.map((q) => (
              <Field
                key={q.field}
                q={q}
                value={answers[q.field]}
                invalid={touched && missingRequired.includes(q.field)}
                onChange={(v) => set(q.field, v)}
              />
            ))}
          </div>
        </div>
      ))}

      <div className="spacer" />
      <button className="btn primary" type="submit" disabled={busy}>
        {busy ? 'Scoring…' : 'See my estimate'}
      </button>
    </form>
  )
}

function Field({ q, value, invalid, onChange }) {
  const wide = q.type === 'yesno' // give yes/no questions the full row - labels are long
  return (
    <div
      id={'field-' + q.field}
      className={
        'field' +
        (q.required ? ' required' : '') +
        (invalid ? ' invalid' : '')
      }
      style={wide ? { gridColumn: '1 / -1' } : undefined}
    >
      <label htmlFor={'input-' + q.field}>{q.label}</label>

      {q.type === 'yesno' && (
        <div className="toggle" role="group" aria-label={q.label}>
          <button
            type="button"
            className={Number(value) === 1 ? 'on' : ''}
            onClick={() => onChange(1)}
          >
            Yes
          </button>
          <button
            type="button"
            className={Number(value) === 0 ? 'on' : ''}
            onClick={() => onChange(0)}
          >
            No
          </button>
        </div>
      )}

      {q.type === 'choice' && (
        <select
          id={'input-' + q.field}
          value={value}
          onChange={(e) => onChange(e.target.value)}
        >
          <option value="">
            {q.required ? 'Select…' : 'Prefer not to say'}
          </option>
          {q.options.map(([code, text]) => (
            <option key={code} value={code}>
              {text}
            </option>
          ))}
        </select>
      )}

      {q.type === 'number' && (
        <input
          id={'input-' + q.field}
          type="number"
          inputMode="decimal"
          value={value}
          min={q.min}
          max={q.max}
          placeholder={
            q.field === 'BMI'
              ? 'leave blank to compute from height & weight'
              : q.min != null
                ? `${q.min}–${q.max}`
                : ''
          }
          onChange={(e) => onChange(e.target.value)}
        />
      )}

      {q.note && <div className="note">{q.note}</div>}
    </div>
  )
}
