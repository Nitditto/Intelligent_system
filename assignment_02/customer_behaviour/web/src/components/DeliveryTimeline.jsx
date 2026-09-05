import React from 'react'
import { calculateTiming } from '../lib/order.js'

export default function DeliveryTimeline({ values }) {
  const { purchase, promised, actual, deliveryDays, daysLate, isLate, isValid } = calculateTiming(values)

  return (
    <div className="timeline-box">
      <div className="timeline-box__head">
        <h4>Delivery Timeline & Status</h4>
        {isValid && (
          <span className={`timeline-badge ${isLate ? 'timeline-badge--bad' : 'timeline-badge--good'}`}>
            {isLate
              ? `⚠️ ${Math.abs(daysLate)} day(s) LATE`
              : daysLate < 0
              ? `✅ ${Math.abs(daysLate)} day(s) EARLY`
              : '✅ Delivered on promised date'}
          </span>
        )}
      </div>

      {isValid ? (
        <>
          <div className="timeline-svg-container">
            <svg viewBox="0 0 320 80" className="timeline-svg">
              <line x1="40" y1="40" x2="280" y2="40" className="timeline-line-bg" />
              <line
                x1="40" y1="40"
                x2={isLate ? '280' : '220'}
                y2="40"
                className={`timeline-line-fill ${isLate ? 'timeline-line-fill--late' : 'timeline-line-fill--good'}`}
              />

              <g transform="translate(40, 40)">
                <circle r="12" className="timeline-node timeline-node--placed" />
                <text y="4" textAnchor="middle" className="timeline-node-icon">🛒</text>
                <text y="28" textAnchor="middle" className="timeline-lbl">Placed</text>
                <text y="40" textAnchor="middle" className="timeline-date">{purchase ? purchase.toLocaleDateString() : ''}</text>
              </g>

              <g transform="translate(180, 40)">
                <circle r="12" className="timeline-node timeline-node--promised" />
                <text y="4" textAnchor="middle" className="timeline-node-icon">📅</text>
                <text y="28" textAnchor="middle" className="timeline-lbl">Promised</text>
                <text y="40" textAnchor="middle" className="timeline-date">{promised ? promised.toLocaleDateString() : ''}</text>
              </g>

              <g transform="translate(280, 40)">
                <circle r="12" className={`timeline-node ${isLate ? 'timeline-node--bad' : 'timeline-node--good'}`} />
                <text y="4" textAnchor="middle" className="timeline-node-icon">{isLate ? '📦' : '🚚'}</text>
                <text y="28" textAnchor="middle" className="timeline-lbl">Delivered</text>
                <text y="40" textAnchor="middle" className="timeline-date">{actual ? actual.toLocaleDateString() : ''}</text>
              </g>
            </svg>
          </div>

          <div className="timeline-stats">
            <div className="timeline-stat">
              <span className="timeline-stat-lbl">Transit Duration</span>
              <span className="timeline-stat-val">{deliveryDays} days</span>
            </div>
            <div className="timeline-stat">
              <span className="timeline-stat-lbl">Delay vs Promise</span>
              <span className={`timeline-stat-val ${isLate ? 'text-bad' : 'text-good'}`}>
                {daysLate > 0 ? `+${daysLate} days` : `${daysLate} days`}
              </span>
            </div>
          </div>
        </>
      ) : (
        <p className="muted">Enter valid purchase, promised, and delivery dates to see timeline.</p>
      )}
    </div>
  )
}
