import { useEffect, useMemo, useRef, useState } from 'react'
import { api } from './api.js'
import Header from './components/Header.jsx'
import Questionnaire from './components/Questionnaire.jsx'
import ResultPanel from './components/ResultPanel.jsx'
import HistoryPanel from './components/HistoryPanel.jsx'
import Collapsible from './components/Collapsible.jsx'
import Dashboard from './components/Dashboard.jsx'

function useSessionId() {
  return useMemo(() => {
    const KEY = 'dx-session'
    let id = null
    try {
      id = localStorage.getItem(KEY)
    } catch {
      /* ignore */
    }
    if (!id) {
      id = 'web-' + Math.random().toString(36).slice(2, 10)
      try {
        localStorage.setItem(KEY, id)
      } catch {
        /* ignore */
      }
    }
    return id
  }, [])
}

export default function App() {
  const sessionId = useSessionId()
  const [view, setView] = useState('screen')
  const [apiOk, setApiOk] = useState(true)
  const [questions, setQuestions] = useState(null)
  const [loadErr, setLoadErr] = useState(null)

  const [busy, setBusy] = useState(false)
  const [submitErr, setSubmitErr] = useState(null)
  const [result, setResult] = useState(null)
  const [answers, setAnswers] = useState(null)
  const [payload, setPayload] = useState(null)
  const [historyKey, setHistoryKey] = useState(0)

  const resultRef = useRef(null)

  useEffect(() => {
    api
      .health()
      .then(() => setApiOk(true))
      .catch(() => setApiOk(false))
    api
      .questions()
      .then((r) => setQuestions(r.questions))
      .catch((e) => setLoadErr(e.message))
  }, [])

  async function handleSubmit(nextPayload, nextAnswers) {
    setBusy(true)
    setSubmitErr(null)
    try {
      const full = { ...nextPayload, session_id: sessionId }
      const res = await api.predict(full)
      setResult(res)
      setAnswers(nextAnswers)
      setPayload(full)
      setHistoryKey((k) => k + 1)
      setApiOk(true)
      setTimeout(
        () => resultRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }),
        60,
      )
    } catch (e) {
      setSubmitErr(e.message)
      if (e.kind === 'network') setApiOk(false)
    } finally {
      setBusy(false)
    }
  }

  return (
    <>
      <Header view={view} setView={setView} apiOk={apiOk} />

      {view === 'operator' ? (
        <Dashboard />
      ) : (
        <div className="page">
          <h1>How likely is it that you have diabetes?</h1>
          <p className="lead">
            Answer a short health questionnaire and get an estimate from a model trained
            on a large public health survey, plus what drove the number and what could
            change it. It does not replace a test or a doctor.
          </p>

          {!apiOk && (
            <div className="notice err">
              Can't reach the screening service. Start the API with{' '}
              <code>uvicorn api.main:app --port 8000</code> and reload.
            </div>
          )}
          {loadErr && <div className="notice err">Could not load the form: {loadErr}</div>}

          {questions && (
            <Questionnaire questions={questions} busy={busy} onSubmit={handleSubmit} />
          )}

          {submitErr && <div className="notice err">{submitErr}</div>}

          <div ref={resultRef} />
          {result && (
            <ResultPanel
              result={result}
              questions={questions}
              answers={answers}
              payload={payload}
              sessionId={sessionId}
            />
          )}

          <div className="card">
            <Collapsible summary="Your earlier screenings" open={!!result}>
              <HistoryPanel sessionId={sessionId} refreshKey={historyKey} />
            </Collapsible>
          </div>
        </div>
      )}
    </>
  )
}
