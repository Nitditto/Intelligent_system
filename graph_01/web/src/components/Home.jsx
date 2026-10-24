import { api } from '../lib/api.js'
import { go, href } from '../lib/route.js'
import { TYPES, fmtInt, typeClass } from '../lib/labels.js'
import { useAsync } from '../lib/useAsync.js'
import { ErrorNote } from './Feedback.jsx'
import SearchBox from './SearchBox.jsx'
import { DIABETES_TYPES } from './TypesView.jsx'

const QUICK = ['Metformin', 'HbA1c', 'Bệnh võng mạc', 'Khát nhiều', 'Thừa cân, béo phì', 'Đồ ngọt, bánh kẹo']

export default function Home() {
  const stats = useAsync(() => api.stats(), [])
  return (
    <div className="stack stack--lg">
      <section className="hero">
        <h1>What do Vinmec articles say about diabetes?</h1>
        <p className="lead">
          A knowledge graph of symptoms, causes, complications, tests, drugs and diet advice, extracted from
          {stats.data ? ` ${fmtInt(stats.data.articles)} ` : ' '}Vinmec health articles. Every fact links back to its sources.
        </p>
        <SearchBox size="lg" onSelect={(e) => go('entity', e.name)} />
        <div className="chips chips--center">
          {QUICK.map((q) => <a key={q} className="chip" href={href('entity', q)}>{q}</a>)}
        </div>
      </section>

      <ErrorNote error={stats.error} />
      {stats.data && (
        <section className="tiles" aria-label="Graph size">
          <Tile value={stats.data.facts} label="facts" />
          <Tile value={stats.data.nodes.reduce((s, n) => s + n.n, 0)} label="entities" />
          <Tile value={stats.data.articles} label="source articles" />
          <Tile value={stats.data.relations.length} label="kinds of relation" />
        </section>
      )}

      <section className="stack">
        <h2 className="section-title">Start from a diabetes type</h2>
        <div className="cards">
          {DIABETES_TYPES.map((t) => (
            <a key={t.name} href={href('types', t.name)} className="card card--link">
              <strong>{t.short}</strong>
              <span className="card__note">{t.note}</span>
            </a>
          ))}
        </div>
      </section>

      {stats.data && (
        <section className="stack">
          <h2 className="section-title">Browse by kind</h2>
          <div className="chips">
            {stats.data.nodes.map((n) => (
              <span key={n.name} className={`chip chip--static ${typeClass(n.name)}`}>
                <span className="dot" aria-hidden="true" />{TYPES[n.name]?.plural || n.name} · {n.n}
              </span>
            ))}
          </div>
        </section>
      )}

      <details className="card how">
        <summary>How this graph was built</summary>
        <ol>
          <li>Diabetes articles were collected from the vinmec.com sitemap and filtered by how often they really discuss diabetes.</li>
          <li>Each article was split into sections. A section heading such as “Triệu chứng” or “Điều trị” decides what kind of fact may be read from it.</li>
          <li>Mentions of known symptoms, drugs, tests, foods… were matched in those sections and counted per article.</li>
          <li>A fact is kept only when at least two different articles state it (or an official Vinmec disease page does).</li>
        </ol>
        <p>On a random sample of 50 facts, 48 were correct. Health information only — not medical advice.</p>
      </details>
    </div>
  )
}

function Tile({ value, label }) {
  return (
    <div className="tile">
      <span className="tile__value">{fmtInt(value)}</span>
      <span className="tile__label">{label}</span>
    </div>
  )
}
