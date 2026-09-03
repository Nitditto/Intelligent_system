import React from 'react'
import { categoryEmoji, prettyCategory } from '../lib/labels.js'

// The "hardcoded example image for the category" tile from the wireframe.
export default function CategoryTile({ category }) {
  return (
    <div className="viz-tile">
      <div className="viz-tile__glyph" aria-hidden="true">{categoryEmoji(category)}</div>
      <div className="viz-tile__name">{prettyCategory(category)}</div>
      <div className="viz-tile__cap">example product type</div>
    </div>
  )
}
