// Hardcoded example glyph per product category (the "ảnh ví dụ" tile).
export const CATEGORY_EMOJI = {
  bed_bath_table: '🛏️',
  health_beauty: '🧴',
  sports_leisure: '⚽',
  computers_accessories: '🖥️',
  furniture_decor: '🛋️',
  housewares: '🍽️',
  watches_gifts: '⌚',
  telephony: '📱',
  auto: '🚗',
  toys: '🧸',
  cool_stuff: '🎁',
  garden_tools: '🪴',
  perfumery: '🌸',
  baby: '🍼',
  electronics: '🔌',
  __other__: '📦',
}

export const categoryEmoji = (c) => CATEGORY_EMOJI[c] || '📦'
export const prettyCategory = (c) => (c === '__other__' || !c ? 'other / unspecified' : c.replace(/_/g, ' '))
