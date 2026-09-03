// Helpers for turning the "Quick mode" inputs (days late, total delivery days)
// into the three raw timestamps the API expects, and back.

export const DAY_MS = 86400000

// "2018-05-01T10:00"  or  "2018-05-01 10:00:00"  -> Date
export function parseLocal(s) {
  if (!s) return null
  const [d, t = '00:00'] = String(s).replace('T', ' ').split(' ')
  const [Y, M, D] = d.split('-').map(Number)
  const [h, m] = t.split(':').map(Number)
  const dt = new Date(Y, (M || 1) - 1, D || 1, h || 0, m || 0)
  return Number.isNaN(dt.getTime()) ? null : dt
}

// Date -> "YYYY-MM-DDTHH:mm" (what <input type="datetime-local"> wants)
export function fmtLocal(dt) {
  const p = (n) => String(n).padStart(2, '0')
  return `${dt.getFullYear()}-${p(dt.getMonth() + 1)}-${p(dt.getDate())}T${p(dt.getHours())}:${p(dt.getMinutes())}`
}

export function toLocalInput(s) {
  const dt = parseLocal(s)
  return dt ? fmtLocal(dt) : ''
}

const round1 = (x) => Math.round(x * 10) / 10

// {daysLate (actual - promised), deliveryDays (order -> doorstep)} from the 3 dates
export function deriveTimeline(values) {
  const purchase = parseLocal(values.order_purchase_timestamp)
  const promised = parseLocal(values.order_estimated_delivery_date)
  const actual = parseLocal(values.order_delivered_customer_date)
  return {
    daysLate: purchase && promised && actual ? round1((actual - promised) / DAY_MS) : 0,
    deliveryDays: purchase && actual ? round1((actual - purchase) / DAY_MS) : 12,
  }
}

export function calculateTiming(values) {
  const purchase = parseLocal(values.order_purchase_timestamp)
  const promised = parseLocal(values.order_estimated_delivery_date)
  const actual = parseLocal(values.order_delivered_customer_date)
  const isValid = Boolean(purchase && promised && actual)
  
  const daysLate = isValid ? round1((actual - promised) / DAY_MS) : 0
  const deliveryDays = isValid ? round1((actual - purchase) / DAY_MS) : 0
  
  return {
    purchase,
    promised,
    actual,
    deliveryDays,
    daysLate,
    isLate: daysLate > 0,
    isValid,
  }
}


// Recompute promised + actual from the two quick numbers, keeping the purchase
// date fixed. actual = purchase + deliveryDays ; promised = actual - daysLate.
export function applyTimeline(values, { daysLate, deliveryDays }) {
  let purchase = parseLocal(values.order_purchase_timestamp)
  if (!purchase) purchase = new Date(2018, 4, 1, 10, 0)
  const actual = new Date(purchase.getTime() + Number(deliveryDays) * DAY_MS)
  const promised = new Date(actual.getTime() - Number(daysLate) * DAY_MS)
  return {
    ...values,
    order_purchase_timestamp: fmtLocal(purchase),
    order_delivered_customer_date: fmtLocal(actual),
    order_estimated_delivery_date: fmtLocal(promised),
  }
}

// example order from /samples -> plain values map the form/payload can use
export function exampleToValues(example, defaults = {}) {
  const out = { ...defaults }
  for (const [k, v] of Object.entries(example)) {
    if (k.startsWith('_')) continue
    out[k] = k.includes('_date') || k.includes('timestamp') ? toLocalInput(v) : v ?? ''
  }
  return out
}
