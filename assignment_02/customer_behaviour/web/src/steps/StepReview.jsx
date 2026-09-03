import React from 'react'
import ReviewSentimentCard from '../components/ReviewSentimentCard.jsx'
import { CATEGORY_MAP } from '../lib/labels.js'
import { calculateTiming } from '../lib/order.js'

export default function StepReview({ values, onSubmit, busy, error }) {
  const timing = calculateTiming(values)
  const catLabel = CATEGORY_MAP[values.category]?.label || values.category || 'bed_bath_table'
  const price = Number(values.price_total || 0)
  const freight = Number(values.freight_total || 0)
  const total = Number(values.payment_value_total || (price + freight))

  return (
    <div className="step-grid">
      {/* Left visual */}
      <div className="step-grid__visual">
        <ReviewSentimentCard
          comment={values.review_comment_message}
          customerState={values.customer_state}
        />
      </div>

      {/* Right input summary & Predict CTA */}
      <div className="step-grid__form">
        <div className="receipt-card">
          <div className="receipt-head">
            <span className="receipt-title">📋 ORDER E-RECEIPT</span>
            <span className="receipt-badge">Ready for Inference</span>
          </div>

          <div className="receipt-body">
            <div className="receipt-item">
              <span className="receipt-lbl">Category:</span>
              <span className="receipt-val">{catLabel}</span>
            </div>
            <div className="receipt-item">
              <span className="receipt-lbl">Item Value:</span>
              <span className="receipt-val">R$ {price.toFixed(2)}</span>
            </div>
            <div className="receipt-item">
              <span className="receipt-lbl">Freight Paid:</span>
              <span className="receipt-val">R$ {freight.toFixed(2)}</span>
            </div>
            <div className="receipt-item">
              <span className="receipt-lbl">Total Charged:</span>
              <span className="receipt-val"><b>R$ {total.toFixed(2)}</b></span>
            </div>
            <div className="receipt-item">
              <span className="receipt-lbl">Payment Method:</span>
              <span className="receipt-val">{values.main_payment_type || 'credit_card'} ({values.max_installments || 1}x)</span>
            </div>
            <div className="receipt-item">
              <span className="receipt-lbl">Customer Region:</span>
              <span className="receipt-val">{values.customer_state || 'SP'}</span>
            </div>
            <div className="receipt-item">
              <span className="receipt-lbl">Delivery Delay:</span>
              <span className={`receipt-val ${timing.isLate ? 'text-bad' : 'text-good'}`}>
                {timing.isLate ? `⚠️ ${timing.daysLate} day(s) LATE` : '✅ On time / early'}
              </span>
            </div>
            <div className="receipt-item">
              <span className="receipt-lbl">Review Comment:</span>
              <span className="receipt-val receipt-val--text">
                {values.review_comment_message ? `"${values.review_comment_message}"` : 'None'}
              </span>
            </div>
          </div>

          <div className="receipt-assumptions">
            💡 <b>Fixed Assumptions:</b> <code>product_desc_len = 607</code> (dataset median), <code>n_payment_types = 1</code>.
          </div>
        </div>

        {error && <div className="banner">{error}</div>}

        <p className="receipt-cta-hint">
          Everything look right? Press <b>⚡ Predict Order</b> below to run the model.
        </p>
      </div>
    </div>
  )
}
