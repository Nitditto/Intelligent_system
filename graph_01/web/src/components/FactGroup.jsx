import { useState } from 'react'
import { api } from '../lib/api.js'
import { href } from '../lib/route.js'
import { relVerb, typeClass } from '../lib/labels.js'
import { ErrorNote, Spinner } from './Feedback.jsx'
import SupportMeter from './SupportMeter.jsx'
import TypePill from './TypePill.jsx'

// One relation seen from the current entity, e.g. "Symptoms" or "Used to treat".
export default function FactGroup({ title, facts, focus }) {
  const f0 = facts[0]
  const otherType = f0.subject === focus ? f0.object_type : f0.subject_type
  return (
    <section className={`card fact-group ${typeClass(otherType)}`}>
      <h2>{title} <span className="fact-group__count">{facts.length}</span></h2>
      <ul className="fact-list">
        {facts.map((f) => <FactRow key={`${f.subject}|${f.relation}|${f.object}`} fact={f} focus={focus} />)}
      </ul>
    </section>
  )
}

function FactRow({ fact, focus }) {
  const [open, setOpen] = useState(false)
  const [state, setState] = useState({ loading: false, data: null, error: null })
  const out = fact.subject === focus
  const other = out ? fact.object : fact.subject
  const otherType = out ? fact.object_type : fact.subject_type
  const extracted = fact.source !== 'curated'

  const toggle = () => {
    setOpen((o) => !o)
    if (!state.data && !state.loading) {
      setState({ loading: true, data: null, error: null })
      api.evidence(fact)
        .then((data) => setState({ loading: false, data, error: null }))
        .catch((error) => setState({ loading: false, data: null, error }))
    }
  }

  return (
    <li className="fact">
      <div className="fact__main">
        <a className="fact__name" href={href('entity', other)}>{other}</a>
        <TypePill type={otherType} />
        <SupportMeter support={fact.support} source={fact.source} />
        {extracted && (
          <button type="button" className="btn btn--ghost btn--sm" aria-expanded={open} onClick={toggle}>
            {open ? 'Hide sources' : 'Sources'}
          </button>
        )}
      </div>
      {open && (
        <div className="fact__evidence">
          <p className="fact__sentence">“{fact.subject} {relVerb(fact.relation)} {fact.object}” is stated in these Vinmec articles:</p>
          {state.loading && <Spinner label="Loading sources…" />}
          <ErrorNote error={state.error} />
          {state.data && (
            <ul className="source-list">
              {state.data.map((a) => (
                <li key={a.url}><a href={a.url} target="_blank" rel="noreferrer">{a.title}</a></li>
              ))}
            </ul>
          )}
        </div>
      )}
    </li>
  )
}
