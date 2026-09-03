class PredictionResult {
  final double predictedPrice;
  final double? pricePerM2;
  final String formattedPriceBillion;
  final String currency;
  final String modelName;
  final String interpretation;

  PredictionResult({
    required this.predictedPrice,
    this.pricePerM2,
    required this.formattedPriceBillion,
    required this.currency,
    required this.modelName,
    required this.interpretation,
  });

  factory PredictionResult.fromJson(Map<String, dynamic> json) {
    return PredictionResult(
      predictedPrice: (json['predicted_price'] as num).toDouble(),
      pricePerM2: json['price_per_m2'] != null ? (json['price_per_m2'] as num).toDouble() : null,
      formattedPriceBillion: json['formatted_price_billion'] as String? ?? '',
      currency: json['currency'] as String? ?? 'million VND',
      modelName: json['model_name'] as String? ?? 'RandomForest',
      interpretation: json['interpretation'] as String? ?? '',
    );
  }
}
