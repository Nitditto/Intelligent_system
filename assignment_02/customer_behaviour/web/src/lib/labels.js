// Label maps, emojis, Portuguese sentiment keywords, and wizard helpers

export const CATEGORY_MAP = {
  bed_bath_table: { emoji: '🛏️', label: 'Bed, Bath & Table', desc: 'Linens, towels, bedroom & bathroom accessories' },
  health_beauty: { emoji: '💄', label: 'Health & Beauty', desc: 'Cosmetics, skincare, supplements' },
  sports_leisure: { emoji: '⚽', label: 'Sports & Leisure', desc: 'Fitness gear, outdoor equipment, sports' },
  furniture_decor: { emoji: '🪑', label: 'Furniture & Decor', desc: 'Home furnishings, wall art, lamps' },
  computers_accessories: { emoji: '💻', label: 'Computers & Tech', desc: 'Laptops, keyboards, computer parts' },
  housewares: { emoji: '🍽️', label: 'Housewares', desc: 'Kitchenware, cookware, utensils' },
  watches_gifts: { emoji: '⌚', label: 'Watches & Gifts', desc: 'Wristwatches, gift items, jewelry' },
  telephony: { emoji: '📱', label: 'Telephony', desc: 'Cellphones, chargers, cases' },
  auto: { emoji: '🚗', label: 'Automotive', desc: 'Car accessories, tools, auto parts' },
  toys: { emoji: '🧸', label: 'Toys & Games', desc: 'Action figures, games, kids toys' },
  cool_stuff: { emoji: '🎁', label: 'Cool Stuff', desc: 'Novelty gifts, collectibles' },
  garden_tools: { emoji: '🪴', label: 'Garden & Tools', desc: 'Plants, lawn care, hand tools' },
  baby: { emoji: '👶', label: 'Baby Products', desc: 'Strollers, baby clothing, care' },
  electronics: { emoji: '🎧', label: 'Electronics', desc: 'Audio, gadgets, home electronics' },
  perfumery: { emoji: '🧪', label: 'Perfumery', desc: 'Fragrances, perfumes, colognes' },
  __other__: { emoji: '📦', label: 'Other Categories', desc: 'General merchandise & miscellaneous' },
}

export const STATE_TO_REGION = {
  AC: 'North', AM: 'North', AP: 'North', PA: 'North', RO: 'North', RR: 'North', TO: 'North',
  AL: 'Northeast', BA: 'Northeast', CE: 'Northeast', MA: 'Northeast', PB: 'Northeast',
  PE: 'Northeast', PI: 'Northeast', RN: 'Northeast', SE: 'Northeast',
  DF: 'Centre-West', GO: 'Centre-West', MS: 'Centre-West', MT: 'Centre-West',
  ES: 'Southeast', MG: 'Southeast', RJ: 'Southeast', SP: 'Southeast',
  PR: 'South', RS: 'South', SC: 'South',
}

export const STEP_DEFINITIONS = [
  { id: 'product', title: 'Product Details', sub: 'Category, value, weight & seller info' },
  { id: 'payment', title: 'Payment & Region', sub: 'Payment method, freight & customer location' },
  { id: 'delivery', title: 'Delivery & Review', sub: 'Order dates, delivery timing & customer review' },
  { id: 'review', title: 'Review & Predict', sub: 'Final summary & satisfaction prediction' },
]

export const STEP_REQUIRED_FIELDS = {
  0: ['price_total'],
  1: ['freight_total'],
  2: ['order_purchase_timestamp', 'order_estimated_delivery_date', 'order_delivered_customer_date'],
  3: [],
}

export const PT_SENTIMENT_KEYWORDS = {
  positive: [
    'excelente', 'recomendo', 'ótimo', 'otimo', 'perfeito', 'rápido', 'rapido',
    'satisfeito', 'lindo', 'bom', 'boa', 'chegou antes', 'qualidade', 'parabéns',
    'parabens', 'gostei', 'top', '10', 'originais', 'cumpre', 'maravilhoso',
  ],
  negative: [
    'atrasado', 'atraso', 'demorou', 'defeito', 'danificado', 'péssimo', 'pessimo',
    'ruim', 'errado', 'amassado', 'diferente', 'faltou', 'não recebi', 'nao recebi',
    'decepção', 'decepcao', 'quebrado', 'lixo', 'propaganda enganosa', 'incompleto',
  ],
}
