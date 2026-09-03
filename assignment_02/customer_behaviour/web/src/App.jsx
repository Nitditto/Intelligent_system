import React, { useEffect, useState } from 'react'
import { api, ApiError } from './api.js'
import Wizard from './components/Wizard.jsx'
import ResultScreen from './components/ResultScreen.jsx'
import ThemeToggle from './components/ThemeToggle.jsx'
import ExamplePicker from './components/ExamplePicker.jsx'
import { exampleToValues } from './lib/order.js'

const FALLBACK = {
  category: 'bed_bath_table', price_total: 129.9, freight_total: 18.3, payment_value_total: 148.2,
  n_items: 1, n_sellers: 1, main_payment_type: 'credit_card', max_installments: 3, customer_state: 'SP',
  product_weight_g: 1200, product_photos_qty: 3,
  order_purchase_timestamp: '2018-05-01T10:00', order_estimated_delivery_date: '2018-05-20T00:00',
  order_delivered_customer_date: '2018-05-31T14:00',
  review_comment_message: 'Produto chegou muito atrasado e a embalagem estava danificada.',
}

export default function App() {
  const [meta, setMeta] = useState(null)
  const [model, setModel] = useState(null)
  const [samples, setSamples] = useState(null)
  const [values, setValues] = useState(FALLBACK)
  const [step, setStep] = useState(0)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)
  const [apiUp, setApiUp] = useState(null)

  useEffect(() => {
    api.health().then(() => setApiUp(true)).catch(() => setApiUp(false))
    api.questions().then(setMeta).catch((e) => setError(e.message))
    api.modelInfo().then(setModel).catch(() => {})
    api.samples().then((s) => {
      setSamples(s)
      const seed = s.examples?.find((e) => e._true_score <= 2 && e.review_comment_message) || s.examples?.[0]
      if (seed) setValues(exampleToValues(seed, s.defaults))
    }).catch(() => {})
  }, [])

  const nSteps = meta ? meta.wizard_steps.length : 3

  async function onPredict() {
    setError(null)
    setBusy(true)
    const payload = {}
    for (const [k, v] of Object.entries(values)) {
      if (v === '' || v == null) continue
      payload[k] = k.includes('_date') || k.includes('timestamp') ? String(v).replace('T', ' ') : v
    }
    for (const f of meta.fields) {
      if (f.type === 'number' && payload[f.field] != null) payload[f.field] = Number(payload[f.field])
    }
    try {
      setResult(await api.predict(payload))
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  function loadExample(ex) {
    setError(null)
    setResult(null)
    setValues(exampleToValues(ex, samples?.defaults))
    setStep(nSteps) // jump to the review/summary step
  }

  function restart() {
    setResult(null)
    setStep(0)
  }

  return (
    <div className="app">
      <header className="topbar">
        <div className="topbar__title">
          <span className="topbar__mark">◱</span>
          <span>Order satisfaction predictor</span>
        </div>
        <div className="topbar__actions">
          {samples?.examples?.length > 0 && !result && (
            <ExamplePicker examples={samples.examples} onPick={loadExample} />
          )}
          <ThemeToggle />
          <span className={`pill ${apiUp === false ? 'pill--bad' : apiUp ? 'pill--good' : ''}`}>
            {apiUp === false ? 'API offline' : apiUp ? 'API connected' : '…'}
          </span>
        </div>
      </header>

      <main className="stage">
        {apiUp === false && (
          <div className="banner">
            <span>The API isn&apos;t reachable.</span>
            <span>From <code>customer_behaviour/</code> run <code>uvicorn api.main:app --port 8000</code>, then reload.</span>
          </div>
        )}
        {error && <p className="err err--center">{error}</p>}

        {result ? (
          <ResultScreen result={result} model={model} onRestart={restart} />
        ) : meta ? (
          <Wizard
            meta={meta}
            values={values}
            setValues={setValues}
            step={step}
            setStep={setStep}
            onPredict={onPredict}
            busy={busy}
          />
        ) : (
          <p className="muted stage__loading">Loading…</p>
        )}
      </main>
    </div>
  )
}
