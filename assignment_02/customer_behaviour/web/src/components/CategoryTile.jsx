import React from 'react'
import { CATEGORY_MAP } from '../lib/labels.js'

export default function CategoryTile({ category, priceTotal, nItems }) {
  const catInfo = CATEGORY_MAP[category] || CATEGORY_MAP.__other__

  return (
    <div className="cat-tile">
      <div className="cat-tile__badge">{catInfo.emoji}</div>
      <h3 className="cat-tile__title">{catInfo.label}</h3>
      <p className="cat-tile__desc">{catInfo.desc}</p>
      
      <div className="cat-tile__stats">
        <div className="cat-tile__stat">
          <span className="cat-tile__stat-lbl">Price</span>
          <span className="cat-tile__stat-val">R$ {Number(priceTotal || 0).toFixed(2)}</span>
        </div>
        <div className="cat-tile__stat">
          <span className="cat-tile__stat-lbl">Quantity</span>
          <span className="cat-tile__stat-val">{nItems || 1} unit(s)</span>
        </div>
      </div>
    </div>
  )
}
