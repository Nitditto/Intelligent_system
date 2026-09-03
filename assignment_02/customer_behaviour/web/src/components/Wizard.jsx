import React, { useState } from 'react'
import ThemeToggle from './ThemeToggle.jsx'
import ExamplePicker from './ExamplePicker.jsx'
import StepProduct from '../steps/StepProduct.jsx'
import StepPayment from '../steps/StepPayment.jsx'
import StepDelivery from '../steps/StepDelivery.jsx'
import StepReview from '../steps/StepReview.jsx'
import { STEP_DEFINITIONS, STEP_REQUIRED_FIELDS } from '../lib/labels.js'
import { exampleToValues } from '../lib/order.js'

export default function Wizard({
  values,
  setValues,
  samples,
  apiUp,
  onSubmit,
  busy,
  error,
  setError,
}) {
  const [step, setStep] = useState(0)
  const [stepErrors, setStepErrors] = useState({})

  function handleLoadExample(ex) {
    setError(null)
    setStepErrors({})
    setValues(exampleToValues(ex, samples?.defaults))
    setStep(3) // Jump directly to Review & Predict step for instant preview
  }

  function validateCurrentStep() {
    const requiredKeys = STEP_REQUIRED_FIELDS[step] || []
    const errs = {}
    let valid = true

    for (const key of requiredKeys) {
      const val = values[key]
      if (val === '' || val == null) {
        errs[key] = 'This field is required'
        valid = false
      }
    }

    setStepErrors(errs)
    return valid
  }

  function handleNext() {
    setError(null)
    if (!validateCurrentStep()) {
      return
    }
    if (step < STEP_DEFINITIONS.length - 1) {
      setStep(s => s + 1)
      setStepErrors({})
    } else {
      onSubmit()
    }
  }

  function handleBack() {
    setError(null)
    setStepErrors({})
    if (step > 0) {
      setStep(s => s - 1)
    }
  }

  const currentStepDef = STEP_DEFINITIONS[step]

  return (
    <div className="wiz-shell">
      {/* 1. Fixed Top Bar */}
      <header className="wiz-hdr">
        <div className="wiz-hdr__left">
          <h1 className="wiz-hdr__title">Order Satisfaction Predictor</h1>
          <span className="wiz-hdr__sub">Full-Viewport Wizard</span>
        </div>

        <div className="wiz-hdr__right">
          {samples?.examples?.length > 0 && (
            <ExamplePicker examples={samples.examples} onPick={handleLoadExample} />
          )}
          <ThemeToggle />
          <span className={`pill ${apiUp === false ? 'pill--bad' : apiUp ? 'pill--good' : ''}`}>
            {apiUp === false ? 'API offline' : apiUp ? 'API connected' : 'connecting…'}
          </span>
        </div>
      </header>

      {apiUp === false && (
        <div className="banner banner--top">
          <span>API isn&apos;t reachable. Run <code>uvicorn api.main:app --port 8000</code> in <code>customer_behaviour/</code>.</span>
        </div>
      )}

      {/* 2. Scrollable Step Body */}
      <main className="wiz-body">
        <div className="wiz-step-hdr">
          <div className="wiz-step-badge">Step {step + 1} of 4</div>
          <h2 className="wiz-step-title">{currentStepDef.title}</h2>
          <p className="wiz-step-sub">{currentStepDef.sub}</p>
        </div>

        <div className="wiz-step-content">
          {step === 0 && <StepProduct values={values} setValues={setValues} errors={stepErrors} />}
          {step === 1 && <StepPayment values={values} setValues={setValues} errors={stepErrors} />}
          {step === 2 && <StepDelivery values={values} setValues={setValues} errors={stepErrors} />}
          {step === 3 && (
            <StepReview
              values={values}
              onSubmit={onSubmit}
              busy={busy}
              error={error}
            />
          )}
        </div>
      </main>

      {/* 3. Sticky Footer Navigation */}
      <footer className="wiz-foot">
        <div className="wiz-foot__container">
          <button
            type="button"
            className="btn btn--ghost"
            disabled={step === 0 || busy}
            onClick={handleBack}
          >
            ← Back
          </button>

          {/* Progress Dots */}
          <div className="wiz-dots">
            {STEP_DEFINITIONS.map((def, idx) => (
              <button
                key={idx}
                type="button"
                className={`wiz-dot ${idx === step ? 'is-active' : idx < step ? 'is-done' : ''}`}
                title={def.title}
                onClick={() => {
                  if (idx < step || validateCurrentStep()) {
                    setStep(idx)
                  }
                }}
              >
                <span className="wiz-dot__num">{idx + 1}</span>
              </button>
            ))}
          </div>

          <button
            type="button"
            className={`btn ${step === STEP_DEFINITIONS.length - 1 ? 'btn--primary btn--glow' : 'btn--primary'}`}
            disabled={busy}
            onClick={handleNext}
          >
            {step === STEP_DEFINITIONS.length - 1 ? (
              busy ? 'Running model…' : '⚡ Predict Order'
            ) : (
              'Next Step →'
            )}
          </button>
        </div>
      </footer>
    </div>
  )
}
