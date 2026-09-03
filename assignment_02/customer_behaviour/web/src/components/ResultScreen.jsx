import React from 'react'
import ShapChart from './ShapChart.jsx'

export default function ResultScreen({ result, onRestart }) {
  if (!result) return null

  const isSatisfied = result.prediction === 'satisfied'
  const pSat = Math.round(result.p_satisfied * 100)
  const pDissat = 100 - pSat
  const { signals, contributions } = result

  // Extract text sentiment items from contributions
  const textItems = contributions?.items?.filter(it => it.kind === 'text') || []
  const badWords = textItems.filter(it => it.effect < 0).map(it => it.label.replace('comment: "', '').replace('"', ''))
  const goodWords = textItems.filter(it => it.effect > 0).map(it => it.label.replace('comment: "', '').replace('"', ''))

  return (
    <div className="res-dashboard">
      {/* Top Navigation Bar */}
      <header className="res-nav">
        <div className="res-nav__left">
          <button type="button" className="btn btn--ghost" onClick={onRestart}>
            ← Predict Another Order
          </button>
        </div>

        <div className="res-nav__right">
          <span className="pill pill--model">
            🤖 Model: <b>{result.model}</b>
          </span>
          <span className="pill pill--thresh">
            🎯 Cutoff: <b>{Math.round(result.threshold * 100)}%</b>
          </span>
        </div>
      </header>

      {/* Main Dashboard Grid */}
      <main className="res-grid">
        {/* LEFT COLUMN */}
        <div className="res-col res-col--left">
          {/* 1. Hero Verdict Card */}
          <div className={`hero-card ${isSatisfied ? 'hero-card--good' : 'hero-card--bad'}`}>
            <div className="hero-card__top">
              <div className="hero-card__stars">
                {isSatisfied ? '★★★★★' : '★☆☆☆☆'}
              </div>
              <span className={`hero-card__status-badge ${isSatisfied ? 'hero-card__status-badge--good' : 'hero-card__status-badge--bad'}`}>
                {isSatisfied ? '✨ SATISFACTION LIKELY' : '⚠️ AT-RISK ORDER'}
              </span>
            </div>

            <h2 className="hero-card__title">
              {isSatisfied ? 'Positive Review Likely (4–5★)' : 'Negative Review Likely (1–3★)'}
            </h2>

            <p className="hero-card__subtitle">
              About <b>{isSatisfied ? pSat : pDissat}% chance</b> of a{' '}
              <b>{isSatisfied ? '4–5★ (positive)' : '1–3★ (negative)'}</b> review.
            </p>
          </div>

          {/* 2. Arc Gauge Card */}
          <div className="card arc-gauge-card">
            <div className="arc-gauge">
              <div className="arc-gauge__svg-wrapper">
                <svg viewBox="0 0 180 100" className="arc-gauge__svg" aria-hidden="true">
                  {/* track */}
                  <path d="M 18 90 A 72 72 0 0 1 162 90" fill="none"
                        stroke="var(--c-surface-2)" strokeWidth="11" strokeLinecap="round"
                        pathLength="100" />
                  {/* fill = P(satisfied) */}
                  <path d="M 18 90 A 72 72 0 0 1 162 90" fill="none"
                        stroke={isSatisfied ? 'var(--c-good)' : 'var(--c-bad)'}
                        strokeWidth="11" strokeLinecap="butt"
                        pathLength="100" strokeDasharray={`${pSat} 100`}
                        className="arc-gauge__meter" />
                  {/* 50% decision cut-off marker */}
                  <path d="M 18 90 A 72 72 0 0 1 162 90" fill="none"
                        stroke="var(--c-text)" strokeWidth="13"
                        pathLength="100" strokeDasharray="0.8 100" strokeDashoffset="-49.6"
                        opacity="0.45" />
                </svg>

                <div className="arc-gauge__overlay">
                  <span className="arc-gauge__num">{pSat}%</span>
                  <span className="arc-gauge__lbl">chance of a good review</span>
                </div>
              </div>

              <div className="arc-gauge__footnote">
                <span>0% · bad</span>
                <span className="arc-gauge__cutoff-tag">↑ 50% cut-off</span>
                <span>good · 100%</span>
              </div>
            </div>
          </div>

          {/* 3. Key Order Signals Grid */}
          <div className="card signals-card">
            <h4 className="card-section-title">KEY SIGNALS</h4>
            
            <div className="signals-grid-2x2">
              <div className="sig-tile">
                <div className="sig-tile__icon">⚡</div>
                <div className="sig-tile__info">
                  <span className="sig-tile__lbl">Delivery vs. Promised</span>
                  <span className={`sig-tile__val ${signals?.late ? 'text-bad' : 'text-good'}`}>
                    {signals?.days_vs_promise != null
                      ? signals.days_vs_promise > 0
                        ? `${signals.days_vs_promise} days late`
                        : `${Math.abs(signals.days_vs_promise)} days early`
                      : '—'}
                  </span>
                </div>
              </div>

              <div className="sig-tile">
                <div className="sig-tile__icon">📦</div>
                <div className="sig-tile__info">
                  <span className="sig-tile__lbl">Transit Time</span>
                  <span className="sig-tile__val">
                    {signals?.delivery_days != null ? `${signals.delivery_days} days` : '—'}
                  </span>
                </div>
              </div>

              <div className="sig-tile">
                <div className="sig-tile__icon">💬</div>
                <div className="sig-tile__info">
                  <span className="sig-tile__lbl">Review Comment</span>
                  <span className="sig-tile__val">
                    {signals?.has_comment ? 'Present' : 'None'}
                  </span>
                </div>
              </div>

              <div className="sig-tile">
                <div className="sig-tile__icon">📍</div>
                <div className="sig-tile__info">
                  <span className="sig-tile__lbl">Category & Region</span>
                  <span className="sig-tile__val">
                    {signals?.category_group || 'bed_bath_table'} · {signals?.customer_region || 'Southeast'}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* 4. Words Reaction Card */}
          {(badWords.length > 0 || goodWords.length > 0) && (
            <div className="card words-card">
              <h4 className="card-section-title">COMMENT WORDS</h4>
              
              {badWords.length > 0 && (
                <div className="words-group">
                  <span className="words-group__lbl text-bad">🔴 BAD REVIEW SIGNALS</span>
                  <div className="words-chips">
                    {badWords.map((word, i) => (
                      <span key={i} className="word-chip word-chip--bad">{word}</span>
                    ))}
                  </div>
                </div>
              )}

              {goodWords.length > 0 && (
                <div className="words-group">
                  <span className="words-group__lbl text-good">🟢 GOOD REVIEW SIGNALS</span>
                  <div className="words-chips">
                    {goodWords.map((word, i) => (
                      <span key={i} className="word-chip word-chip--good">{word}</span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* 5. Retention Playbook */}
          <div className={`retention-playbook ${isSatisfied ? 'retention-playbook--good' : 'retention-playbook--bad'}`}>
            <div className="retention-playbook__head">
              <span className="retention-playbook__icon">{isSatisfied ? '✅' : '🚨'}</span>
              <span className="retention-playbook__title">SUGGESTED ACTION</span>
            </div>
            <p className="retention-playbook__text">
              {isSatisfied
                ? 'Standard order fulfillment workflow.'
                : 'Reach out to customer or offer a goodwill voucher before review is posted.'}
            </p>
          </div>
        </div>

        {/* RIGHT COLUMN (SHAP CHART) */}
        <div className="res-col res-col--right">
          <div className="card shap-executive-card">
            <ShapChart contributions={contributions} />
          </div>
        </div>
      </main>
    </div>
  )
}
