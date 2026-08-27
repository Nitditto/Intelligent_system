export interface ModelResult {
  model: string;
  price: number;
  confidence: number;
  r2: number;
  mape: number;
}

/** Picks the model with the highest R² (goodness of fit) among the returned results. */
export function pickBestModel(results: ModelResult[]): ModelResult {
  return results.reduce((best, r) => (r.r2 > best.r2 ? r : best), results[0]);
}
