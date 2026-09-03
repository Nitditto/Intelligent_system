import React from 'react'
import PaymentCard from '../components/PaymentCard.jsx'
import { STATE_TO_REGION } from '../lib/labels.js'

const PAYMENT_METHODS = [
  { value: 'credit_card', label: 'Credit Card' },
  { value: 'boleto', label: 'Boleto (Brazilian Bank Slip)' },
  { value: 'voucher', label: 'Store Voucher / Credit' },
  { value: 'debit_card', label: 'Debit Card' },
]

export default function StepPayment({ values, setValues, errors }) {
  const onChange = (k, v) => setValues(s => ({ ...s, [k]: v }))

  return (
    <div className="step-grid">
      <div className="step-grid__visual">
        <PaymentCard values={values} />
      </div>

      <div className="step-grid__form">
        <div className="form">
          <div className="field">
            <label className="field__label">
              Shipping paid / Freight (R$) <span className="req">*</span>
            </label>
            <input
              type="number"
              min="0"
              step="0.01"
              placeholder="e.g. 18.30"
              value={values.freight_total ?? ''}
              className={errors.freight_total ? 'is-invalid' : ''}
              onChange={(e) => onChange('freight_total', e.target.value)}
            />
            {errors.freight_total && <span className="field__err">{errors.freight_total}</span>}
          </div>

          <div className="form-row-2">
            <div className="field">
              <label className="field__label">Payment method</label>
              <select
                value={values.main_payment_type || 'credit_card'}
                onChange={(e) => onChange('main_payment_type', e.target.value)}
              >
                {PAYMENT_METHODS.map(m => (
                  <option key={m.value} value={m.value}>{m.label}</option>
                ))}
              </select>
            </div>

            <div className="field">
              <label className="field__label">Instalments</label>
              <input
                type="number"
                min="1"
                max="24"
                placeholder="1"
                value={values.max_installments ?? ''}
                onChange={(e) => onChange('max_installments', e.target.value)}
              />
            </div>
          </div>

          <div className="form-row-2">
            <div className="field">
              <label className="field__label">Total charged (R$)</label>
              <input
                type="number"
                min="0"
                step="0.01"
                placeholder="Leave blank for price + freight"
                value={values.payment_value_total ?? ''}
                onChange={(e) => onChange('payment_value_total', e.target.value)}
              />
            </div>

            <div className="field">
              <label className="field__label">Customer State (Brazil)</label>
              <select
                value={values.customer_state || 'SP'}
                onChange={(e) => onChange('customer_state', e.target.value)}
              >
                {Object.keys(STATE_TO_REGION).map(st => (
                  <option key={st} value={st}>
                    {st} ({STATE_TO_REGION[st]})
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
