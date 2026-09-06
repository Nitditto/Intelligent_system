import React from 'react'
import ShapChart from './ShapChart.jsx'
import { titleCase } from '../lib/sephora.js'

export default function ResultScreen({ result, model, onRestart }) {
  if (!result) return null

  const rec = result.prediction === 'recommend'
  const pRec = Math.round(result.p_recommend * 100)
  const { signals = {}, contributions, review_terms } = result

  const toward = (review_terms?.toward || []).map((t) => t.term)
  const against = (review_terms?.against || []).map((t) => t.term)

  return (
    <div className="res-dashboard">
      <header className="res-nav">
        <div className="res-nav__left">
          <button type="button" className="btn btn--ghost" onClick={onRestart}>
            ← Score another review
          </button>
        </div>
        <div className="res-nav__right">
          <span className="pill">Model: <b>{result.model}</b></span>
          <span className="pill">Cut-off: <b>{Math.round(result.threshold * 100)}%</b></span>
        </div>
      </header>

      <main className="res-grid">
        <div className="res-col res-col--left">
          {/* verdict */}
          <div className={`hero-card ${rec ? 'hero-card--good' : 'hero-card--bad'}`}>
            <span className={`hero-card__status-badge ${rec ? 'hero-card__status-badge--good' : 'hero-card__status-badge--bad'}`}>
              {rec ? '👍 RECOMMENDS' : '👎 WON’T RECOMMEND'}
            </span>
            <h2 className="hero-card__title">
              {rec
                ? 'This customer would probably recommend it'
                : 'This customer probably would not recommend it'}
            </h2>
            <p className="hero-card__subtitle">
              Chance of a recommendation: <b>{pRec}%</b> — we call it “recommends” above{' '}
              {Math.round(result.threshold * 100)}%.
            </p>
          </div>

          {/* probability meter */}
          <div className="card">
            <h4 className="card-section-title">Chance of a recommendation</h4>
            <div className="meter">
              <div className="meter__track">
                <span
                  className={`meter__fill ${rec ? 'meter__fill--good' : 'meter__fill--bad'}`}
                  style={{ width: `${pRec}%` }}
                />
                <span className="meter__threshold" style={{ left: `${Math.round(result.threshold * 100)}%` }} />
              </div>
              <div className="meter__scale">
                <span>0% · won’t recommend</span>
                <span>{Math.round(result.threshold * 100)}% cut-off</span>
                <span>recommends · 100%</span>
              </div>
            </div>
            <p className="card-hint">
              {pRec} of 100 reviewers who wrote this ticked <i>recommend</i>.
            </p>
          </div>

          {/* signals */}
          <div className="card signals-card">
            <h4 className="card-section-title">What the model saw</h4>
            <div className="kv-grid">
              <span className="kv__k">Skin type</span>
              <span className="kv__v">{signals.skin_type ? titleCase(signals.skin_type) : '—'}</span>
              <span className="kv__k">Category</span>
              <span className="kv__v">{signals.category || '—'}</span>
              <span className="kv__k">Brand</span>
              <span className="kv__v">{signals.brand || '—'}</span>
              <span className="kv__k">Price</span>
              <span className="kv__v">
                {signals.price_usd != null ? `$${signals.price_usd}` : '—'}
                {signals.price_tier ? ` · ${signals.price_tier}` : ''}
              </span>
              <span className="kv__k">Product “loves”</span>
              <span className="kv__v">
                {signals.product_loves != null ? Number(signals.product_loves).toLocaleString() : '—'}
              </span>
              <span className="kv__k">Review length</span>
              <span className="kv__v">{signals.review_tokens} words</span>
            </div>
            <p className="card-hint">
              Structured signals alone reach ROC-AUC ~0.8; the review text takes it to ~0.96.
            </p>
          </div>

          {/* review terms */}
          {(toward.length > 0 || against.length > 0) && (
            <div className="card words-card">
              <h4 className="card-section-title">Review words that moved the call</h4>
              {against.length > 0 && (
                <div className="words-group">
                  <span className="words-group__lbl text-bad">▼ toward “won’t recommend”</span>
                  <div className="words-chips">
                    {against.map((w, i) => (
                      <span key={i} className="word-chip word-chip--bad">{w}</span>
                    ))}
                  </div>
                </div>
              )}
              {toward.length > 0 && (
                <div className="words-group">
                  <span className="words-group__lbl text-good">▲ toward “recommends”</span>
                  <div className="words-chips">
                    {toward.map((w, i) => (
                      <span key={i} className="word-chip word-chip--good">{w}</span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* interpretation */}
          <div className={`retention-playbook ${rec ? 'retention-playbook--good' : 'retention-playbook--bad'}`}>
            <div className="retention-playbook__head">
              <span className="retention-playbook__icon">{rec ? '✅' : '⚠️'}</span>
              <span className="retention-playbook__title">How to use this</span>
            </div>
            <p className="retention-playbook__text">
              {rec
                ? 'Text and profile agree — safe to surface for similar skin types.'
                : 'Review reads negative — flag for review-consistency QA.'}
            </p>
          </div>

          <details className="how-it-works">
            <summary>How this works</summary>
            <p>
              ~104k Sephora skincare reviews. <b>Logistic Regression</b> over the skin profile +
              product + a TF-IDF of the review text. Inference is server-side; the review text is
              written with the recommend tick, so it partly leaks the outcome (§14a).
            </p>
          </details>
        </div>

        <div className="res-col res-col--right">
          <div className="card">
            <ShapChart contributions={contributions} />
          </div>
        </div>
      </main>
    </div>
  )
}
