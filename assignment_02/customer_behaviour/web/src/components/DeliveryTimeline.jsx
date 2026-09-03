import React from 'react'
import { deriveTimeline } from '../lib/order.js'

// Small SVG: Ordered ── Promised ── Delivered, positioned on a time axis, with a
// live "X days late / early" read-out. Updates as the dates change.
export default function DeliveryTimeline({ values }) {
  const { daysLate, deliveryDays } = deriveTimeline(values)
  const ok = Number.isFinite(daysLate) && Number.isFinite(deliveryDays) && deliveryDays > 0

  // axis spans 0 .. max(deliveryDays, promised-day) with a little headroom
  const promisedDay = deliveryDays - daysLate
  const span = Math.max(deliveryDays, promisedDay, 1) * 1.15
  const x = (d) => 8 + (Math.max(0, d) / span) * 184 // 8..192 in a 200-wide viewBox
  const late = daysLate > 0.5

  return (
    <div className="viz-tile">
      <div className="viz-tile__name">Delivery timeline</div>
      {ok ? (
        <>
          <svg viewBox="0 0 200 96" className="timeline" role="img"
               aria-label={`Delivered ${late ? daysLate + ' days late' : Math.abs(daysLate) + ' days early'}`}>
            <line x1="8" y1="34" x2="192" y2="34" className="tl-axis" />
            {/* promised */}
            <line x1={x(promisedDay)} y1="24" x2={x(promisedDay)} y2="44" className="tl-promised" />
            <text x={x(promisedDay)} y="58" className="tl-lbl" textAnchor="middle">promised</text>
            {/* ordered */}
            <circle cx={x(0)} cy="34" r="4" className="tl-dot" />
            <text x={x(0)} y="20" className="tl-lbl" textAnchor="middle">ordered</text>
            {/* delivered */}
            <circle cx={x(deliveryDays)} cy="34" r="5"
                    className={late ? 'tl-dot tl-dot--bad' : 'tl-dot tl-dot--good'} />
            <text x={x(deliveryDays)} y="78" className="tl-lbl" textAnchor="middle">delivered</text>
          </svg>
          <div className={`viz-tile__cap ${late ? 'is-bad' : 'is-good'}`}>
            {late
              ? `${daysLate} day${daysLate === 1 ? '' : 's'} late · ${deliveryDays} days total`
              : `${Math.abs(daysLate)} day${Math.abs(daysLate) === 1 ? '' : 's'} early · ${deliveryDays} days total`}
          </div>
        </>
      ) : (
        <div className="viz-tile__cap">Fill the three dates to see the timeline.</div>
      )}
    </div>
  )
}
