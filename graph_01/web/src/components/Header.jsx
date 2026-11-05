import { go, href } from '../lib/route.js'
import SearchBox from './SearchBox.jsx'

const NAV = [
  ['types', 'Diabetes types'],
  ['symptoms', 'Symptom checker'],
  ['paths', 'Connections'],
]

export default function Header({ view }) {
  return (
    <header className="topbar">
      <div className="topbar__inner">
        <a href={href('home')} className="brand">
          <svg viewBox="0 0 32 32" aria-hidden="true" className="brand__logo">
            <path d="M16 16 5 7M16 16l11-7M16 16 9 27" />
            <circle cx="16" cy="16" r="5" /><circle cx="5" cy="7" r="3" /><circle cx="27" cy="9" r="3" /><circle cx="9" cy="27" r="3" />
          </svg>
          <span>Diabetes KG</span>
        </a>
        <nav className="nav" aria-label="Main">
          {NAV.map(([v, label]) => (
            <a key={v} href={href(v)} className="nav__link" aria-current={view === v ? 'page' : undefined}>{label}</a>
          ))}
        </nav>
        {view !== 'home' && (
          <div className="topbar__search">
            <SearchBox size="sm" placeholder="Search…" onSelect={(e) => go('entity', e.name)} />
          </div>
        )}
      </div>
    </header>
  )
}
