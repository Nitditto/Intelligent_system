import React from 'react'
import { STATE_TO_REGION } from '../lib/labels.js'

export default function PaymentCard({ values }) {
  const method = values.main_payment_type || 'credit_card'
  const price = Number(values.price_total || 0)
  const freight = Number(values.freight_total || 0)
  const total = Number(values.payment_value_total || (price + freight))
  const maxInst = Number(values.max_installments || 1)
  const instVal = maxInst > 0 ? (total / maxInst).toFixed(2) : total.toFixed(2)
  const state = values.customer_state || 'SP'
  const region = STATE_TO_REGION[state] || 'Southeast'

  const freightRatio = price > 0 ? ((freight / price) * 100).toFixed(1) : 0

  const getMethodTitle = (m) => {
    switch (m) {
      case 'boleto': return 'Brazilian Bank Slip (Boleto)'
      case 'voucher': return 'Store Voucher / Credit'
      case 'debit_card': return 'Debit Card Payment'
      default: return 'Credit Card Payment'
    }
  }

  const getMethodIcon = (m) => {
    switch (m) {
      case 'boleto': return '📄'
      case 'voucher': return '🎟️'
      case 'debit_card': return '💳'
      default: return '💳'
    }
  }

  return (
    <div className="pay-card-wrapper">
      {/* Glassmorphic Credit / Payment Card Mockup */}
      <div className={`pay-card pay-card--${method}`}>
        <div className="pay-card__top">
          <span className="pay-card__chip">💳 CHIP</span>
          <span className="pay-card__brand">{getMethodIcon(method)} {method.toUpperCase()}</span>
        </div>
        <div className="pay-card__number">•••• •••• •••• 2026</div>
        <div className="pay-card__bottom">
          <div className="pay-card__holder">
            <span className="pay-card__lbl">CUSTOMER REGION</span>
            <span className="pay-card__val">📍 {state} — {region}</span>
          </div>
          <div className="pay-card__amount">
            <span className="pay-card__lbl">TOTAL CHARGED</span>
            <span className="pay-card__val">R$ {total.toFixed(2)}</span>
          </div>
        </div>
      </div>

      {/* Instalments & Freight Ratio Details */}
      <div className="pay-details">
        <div className="pay-detail-row">
          <span className="pay-detail-lbl">Payment Plan</span>
          <span className="pay-detail-val">
            {maxInst > 1 ? `${maxInst}x of R$ ${instVal}/mo` : 'Paid in full (1x)'}
          </span>
        </div>
        <div className="pay-detail-row">
          <span className="pay-detail-lbl">Method Type</span>
          <span className="pay-detail-val">{getMethodTitle(method)}</span>
        </div>
      </div>

      {/* Freight Ratio Visual Bar */}
      <div className="freight-ratio-box">
        <div className="freight-ratio-box__top">
          <span>Freight-to-Price Ratio</span>
          <b>{freightRatio}%</b>
        </div>
        <div className="freight-ratio-box__bar">
          <div
            className="freight-ratio-box__fill"
            style={{ width: `${Math.min(freightRatio, 100)}%` }}
          />
        </div>
      </div>
    </div>
  )
}
