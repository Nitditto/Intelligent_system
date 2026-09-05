import React, { useEffect, useState } from 'react'
import { api, ApiError } from './api.js'
import Wizard from './components/Wizard.jsx'
import ResultScreen from './components/ResultScreen.jsx'
import { exampleToValues } from './lib/order.js'

const FALLBACK = {
  category: 'bed_bath_table',
  price_total: 129.9,
  freight_total: 18.3,
  payment_value_total: 148.2,
  n_items: 1,
  n_sellers: 1,
  main_payment_type: 'credit_card',
  max_installments: 3,
  customer_state: 'SP',
  product_weight_g: 1200,
  product_photos_qty: 3,
  product_desc_len: 850,
  order_purchase_timestamp: '2018-05-01T10:00',
  order_estimated_delivery_date: '2018-05-20T00:00',
  order_delivered_customer_date: '2018-05-31T14:00',
  review_comment_message: 'Produto chegou muito atrasado e a embalagem estava danificada.',
}

export default function App() {
  const [meta, setMeta] = useState(null)
  const [model, setModel] = useState(null)
  const [samples, setSamples] = useState(null)
  const [values, setValues] = useState(FALLBACK)
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

  async function handleSubmit() {
    setError(null)
    setBusy(true)

    const payload = {}
    for (const [k, v] of Object.entries(values)) {
      if (v === '' || v == null) continue
      payload[k] = k.includes('_date') || k.includes('timestamp')
        ? String(v).replace('T', ' ')
        : v
    }

    if (meta?.fields) {
      for (const f of meta.fields) {
        if (f.type === 'number' && payload[f.field] != null) {
          payload[f.field] = Number(payload[f.field])
        }
      }
    }

    try {
      const res = await api.predict(payload)
      setResult(res)
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  function handleRestart() {
    setResult(null)
    setError(null)
  }

  return (
    <div className="app-viewport">
      {result ? (
        <ResultScreen
          result={result}
          onRestart={handleRestart}
          model={model}
        />
      ) : (
        <Wizard
          values={values}
          setValues={setValues}
          samples={samples}
          apiUp={apiUp}
          onSubmit={handleSubmit}
          busy={busy}
          error={error}
          setError={setError}
        />
      )}
    </div>
  )
}
