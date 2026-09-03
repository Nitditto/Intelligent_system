import React from 'react'
import DeliveryTimeline from '../components/DeliveryTimeline.jsx'

const PORTUGUESE_EXAMPLES = [
  { pt: "Produto excelente, chegou antes do prazo. Recomendo!", en: "Great product, arrived before deadline!" },
  { pt: "Entrega rápida e produto conforme o anúncio.", en: "Fast delivery, product as advertised." },
  { pt: "Até agora não recebi o produto.", en: "I still haven't received the product." },
  { pt: "Veio com defeito e a caixa estava toda amassada.", en: "Defective product, crushed box." },
  { pt: "Produto diferente do que foi anunciado no site.", en: "Different from what was advertised." },
]

export default function StepDelivery({ values, setValues, errors }) {
  const onChange = (k, v) => setValues(s => ({ ...s, [k]: v }))

  return (
    <div className="step-grid">
      <div className="step-grid__visual">
        <DeliveryTimeline values={values} />
      </div>

      <div className="step-grid__form">
        <div className="form">
          <div className="field">
            <label className="field__label">
              Order purchase date <span className="req">*</span>
            </label>
            <input
              type="datetime-local"
              value={values.order_purchase_timestamp ? String(values.order_purchase_timestamp).slice(0, 16) : ''}
              className={errors.order_purchase_timestamp ? 'is-invalid' : ''}
              onChange={(e) => onChange('order_purchase_timestamp', e.target.value)}
            />
            {errors.order_purchase_timestamp && <span className="field__err">{errors.order_purchase_timestamp}</span>}
          </div>

          <div className="form-row-2">
            <div className="field">
              <label className="field__label">
                Promised delivery date <span className="req">*</span>
              </label>
              <input
                type="datetime-local"
                value={values.order_estimated_delivery_date ? String(values.order_estimated_delivery_date).slice(0, 16) : ''}
                className={errors.order_estimated_delivery_date ? 'is-invalid' : ''}
                onChange={(e) => onChange('order_estimated_delivery_date', e.target.value)}
              />
              {errors.order_estimated_delivery_date && <span className="field__err">{errors.order_estimated_delivery_date}</span>}
            </div>

            <div className="field">
              <label className="field__label">
                Actual delivery date <span className="req">*</span>
              </label>
              <input
                type="datetime-local"
                value={values.order_delivered_customer_date ? String(values.order_delivered_customer_date).slice(0, 16) : ''}
                className={errors.order_delivered_customer_date ? 'is-invalid' : ''}
                onChange={(e) => onChange('order_delivered_customer_date', e.target.value)}
              />
              {errors.order_delivered_customer_date && <span className="field__err">{errors.order_delivered_customer_date}</span>}
            </div>
          </div>

          <div className="field field--wide">
            <label className="field__label">Customer review comment (optional text)</label>
            <textarea
              rows={3}
              placeholder="e.g. Produto excelente, chegou antes do prazo..."
              value={values.review_comment_message || ''}
              onChange={(e) => onChange('review_comment_message', e.target.value)}
            />

            <div className="examples">
              {PORTUGUESE_EXAMPLES.map((ex, idx) => (
                <button
                  key={idx}
                  type="button"
                  className="examples__chip"
                  title={ex.en}
                  onClick={() => onChange('review_comment_message', ex.pt)}
                >
                  💬 {ex.pt.slice(0, 32)}…
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
