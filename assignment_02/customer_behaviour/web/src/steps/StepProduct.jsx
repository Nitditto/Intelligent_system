import React from 'react'
import CategoryTile from '../components/CategoryTile.jsx'
import { CATEGORY_MAP } from '../lib/labels.js'

export default function StepProduct({ values, setValues, errors }) {
  const onChange = (k, v) => setValues(s => ({ ...s, [k]: v }))

  return (
    <div className="step-grid">
      <div className="step-grid__visual">
        <CategoryTile
          category={values.category}
          priceTotal={values.price_total}
          nItems={values.n_items}
        />
      </div>

      <div className="step-grid__form">
        <div className="form">
          <div className="field">
            <label className="field__label">Product category</label>
            <select
              value={values.category || 'bed_bath_table'}
              onChange={(e) => onChange('category', e.target.value)}
            >
              {Object.entries(CATEGORY_MAP).map(([key, info]) => (
                <option key={key} value={key}>
                  {info.emoji} {info.label}
                </option>
              ))}
            </select>
          </div>

          <div className="field">
            <label className="field__label">
              Order value (R$) <span className="req">*</span>
            </label>
            <input
              type="number"
              min="0"
              step="0.01"
              placeholder="e.g. 129.90"
              value={values.price_total ?? ''}
              className={errors.price_total ? 'is-invalid' : ''}
              onChange={(e) => onChange('price_total', e.target.value)}
            />
            {errors.price_total && <span className="field__err">{errors.price_total}</span>}
          </div>

          <div className="form-row-2">
            <div className="field">
              <label className="field__label">Item quantity (units)</label>
              <input
                type="number"
                min="1"
                placeholder="1"
                value={values.n_items ?? ''}
                onChange={(e) => onChange('n_items', e.target.value)}
              />
            </div>

            <div className="field">
              <label className="field__label">Number of sellers</label>
              <input
                type="number"
                min="1"
                placeholder="1"
                value={values.n_sellers ?? ''}
                onChange={(e) => onChange('n_sellers', e.target.value)}
              />
            </div>
          </div>

          <div className="form-row-2">
            <div className="field">
              <label className="field__label">Photos in listing</label>
              <input
                type="number"
                min="0"
                placeholder="3"
                value={values.product_photos_qty ?? ''}
                onChange={(e) => onChange('product_photos_qty', e.target.value)}
              />
            </div>

            <div className="field">
              <label className="field__label">Product weight (grams)</label>
              <input
                type="number"
                min="0"
                placeholder="1200"
                value={values.product_weight_g ?? ''}
                onChange={(e) => onChange('product_weight_g', e.target.value)}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
