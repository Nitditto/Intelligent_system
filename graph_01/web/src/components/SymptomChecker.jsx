import { useState } from 'react'
import { api } from '../lib/api.js'
import { href } from '../lib/route.js'
import { articles } from '../lib/labels.js'
import { useAsync } from '../lib/useAsync.js'
import { Empty, ErrorNote, Spinner } from './Feedback.jsx'

const EXAMPLE = ['Khát nhiều', 'Tiểu nhiều', 'Sụt cân', 'Nhìn mờ']

// Pick symptoms -> which diabetes types the articles associate with them.
export default function SymptomChecker() {
  const symptoms = useAsync(() => api.search('', 'Symptom', 50), [])
  const [picked, setPicked] = useState([])
  const [result, setResult] = useState({ loading: false, data: null, error: null })

  const toggle = (name) => setPicked((p) => (p.includes(name) ? p.filter((x) => x !== name) : [...p, name]))
  const check = (list = picked) => {
    setResult((r) => ({ ...r, loading: true, error: null }))
    api.symptomCheck(list, 2)
      .then((data) => setResult({ loading: false, data, error: null }))
      .catch((error) => setResult({ loading: false, data: null, error }))
  }

  return (
    <div className="stack">
      <div>
        <h1>Symptom checker</h1>
        <p className="lead">
          Choose the symptoms you are interested in. The graph lists which types of diabetes Vinmec articles link to them,
          ranked by how many of your symptoms match.
        </p>
      </div>
      <p className="notice">This shows what the articles say. It is not a diagnosis: diabetes is confirmed by blood tests such as HbA1c.</p>

      <section className="card">
        <h2>Symptoms</h2>
        {symptoms.loading && <Spinner />}
        <ErrorNote error={symptoms.error} />
        <div className="chips" role="group" aria-label="Symptoms">
          {symptoms.data?.map((s) => (
            <button key={s.name} type="button" className="chip" aria-pressed={picked.includes(s.name)} onClick={() => toggle(s.name)}>
              {picked.includes(s.name) && <span aria-hidden="true">✓</span>}
              {s.name}
            </button>
          ))}
        </div>
        <div className="actions">
          <button type="button" className="btn btn--primary" disabled={!picked.length || result.loading} onClick={() => check()}
            title={picked.length ? undefined : 'Choose at least one symptom'}>
            {result.loading && <span className="spinner spinner--ink" aria-hidden="true" />}
            Check {picked.length ? `${picked.length} symptom${picked.length > 1 ? 's' : ''}` : ''}
          </button>
          <button type="button" className="btn btn--ghost" onClick={() => { setPicked(EXAMPLE); check(EXAMPLE) }}>Try an example</button>
          {picked.length > 0 && <button type="button" className="btn btn--ghost" onClick={() => { setPicked([]); setResult({ loading: false, data: null, error: null }) }}>Clear</button>}
        </div>
      </section>

      <ErrorNote error={result.error} />
      {result.data && (
        <section className="card" aria-live="polite">
          <h2>Diabetes types linked to these symptoms</h2>
          {result.data.length === 0 && <Empty>No diabetes type is linked to these symptoms by at least two articles.</Empty>}
          {result.data.length > 0 && (
            <div className="verdict">
              <p className="verdict__title">Most strongly linked to: {result.data[0].type}</p>
              <p className="verdict__note">
                It matches {result.data[0].matched.length} of your {picked.length} symptoms. “Backed by” counts the Vinmec articles that state each link,
                so a bigger number means the link is repeated more often, not that it is more likely for you.
              </p>
            </div>
          )}
          <ol className="rank">
            {result.data.map((r) => {
              const share = r.matched.length / picked.length
              return (
                <li key={r.type} className="rank__item">
                  <div className="rank__head">
                    <a href={href('entity', r.type)} className="rank__name">{r.type}</a>
                    <span className="rank__score">{r.matched.length} of {picked.length} symptoms</span>
                  </div>
                  <span className="meter meter--wide" aria-hidden="true">
                    <span className="meter__track"><span className="meter__fill" style={{ width: `${share * 100}%` }} /></span>
                  </span>
                  <p className="rank__detail">Matches: {r.matched.join(', ')} · backed by {articles(r.evidence)}</p>
                </li>
              )
            })}
          </ol>
          <p className="hint">Ties are ordered by the number of supporting articles. General diabetes usually leads because most articles do not name a type.</p>
        </section>
      )}
    </div>
  )
}
