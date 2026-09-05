import React, { useMemo } from 'react'
import { PT_SENTIMENT_KEYWORDS } from '../lib/labels.js'

export default function ReviewSentimentCard({ comment, customerState }) {
  const text = comment || ''

  const highlightedElements = useMemo(() => {
    if (!text.trim()) {
      return <span className="review-empty">No review comment entered (optional text).</span>
    }

    const words = text.split(/(\s+|[.,!?;:]+)/)
    return words.map((chunk, idx) => {
      const lower = chunk.toLowerCase()
      const isPos = PT_SENTIMENT_KEYWORDS.positive.some(kw => lower.includes(kw))
      const isNeg = PT_SENTIMENT_KEYWORDS.negative.some(kw => lower.includes(kw))

      if (isNeg) {
        return <mark key={idx} className="hl-term hl-term--neg">{chunk}</mark>
      }
      if (isPos) {
        return <mark key={idx} className="hl-term hl-term--pos">{chunk}</mark>
      }
      return chunk
    })
  }, [text])

  const posCount = PT_SENTIMENT_KEYWORDS.positive.filter(kw => text.toLowerCase().includes(kw)).length
  const negCount = PT_SENTIMENT_KEYWORDS.negative.filter(kw => text.toLowerCase().includes(kw)).length

  return (
    <div className="review-card">
      <div className="review-card__header">
        <div className="review-card__user">
          <div className="review-card__avatar">👤</div>
          <div>
            <div className="review-card__username">Verified Customer ({customerState || 'SP'})</div>
            <div className="review-card__date">Post-delivery review comment</div>
          </div>
        </div>
        <div className="review-card__badge">PT-BR Natural Language</div>
      </div>

      <div className="review-card__body">
        <p className="review-card__text">{highlightedElements}</p>
      </div>

      <div className="review-card__analysis">
        <span className="review-analysis-lbl">Quick keyword scan</span>
        <div className="review-analysis-chips">
          {posCount === 0 && negCount === 0 ? (
            <span className="chip">
              {text.trim() ? 'No obvious sentiment words — the model reads the full text' : 'No comment to scan'}
            </span>
          ) : (
            <>
              {posCount > 0 && <span className="chip chip--good">🟢 {posCount} positive</span>}
              {negCount > 0 && <span className="chip chip--bad">🔴 {negCount} negative</span>}
            </>
          )}
        </div>
      </div>
    </div>
  )
}
