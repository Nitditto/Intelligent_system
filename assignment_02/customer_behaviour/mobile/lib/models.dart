// Data models mirroring the customer-behaviour API's JSON.

class FormFieldSpec {
  final String field;
  final String type; // "number" | "choice" | "date" | "text"
  final String label;
  final bool required;
  final num? min;
  final num? max;
  final String? note;
  final List<String> options;

  FormFieldSpec({
    required this.field,
    required this.type,
    required this.label,
    required this.required,
    this.min,
    this.max,
    this.note,
    this.options = const [],
  });

  factory FormFieldSpec.fromJson(Map<String, dynamic> j) => FormFieldSpec(
        field: j['field'] as String,
        type: j['type'] as String,
        label: j['label'] as String,
        required: j['required'] == true,
        min: j['min'] as num?,
        max: j['max'] as num?,
        note: j['note'] as String?,
        options: ((j['options'] as List?) ?? [])
            .map((e) => e.toString())
            .toList(),
      );
}

class PredictResult {
  final String prediction; // "satisfied" | "dissatisfied"
  final double confidence;
  final double pSatisfied;
  final double threshold;
  final List<String> termsNegative;
  final List<String> termsPositive;
  final Map<String, dynamic> signals;
  final String model;
  final String representation;

  PredictResult({
    required this.prediction,
    required this.confidence,
    required this.pSatisfied,
    required this.threshold,
    required this.termsNegative,
    required this.termsPositive,
    required this.signals,
    required this.model,
    required this.representation,
  });

  bool get good => prediction == 'satisfied';

  factory PredictResult.fromJson(Map<String, dynamic> j) {
    final terms = (j['review_terms'] as Map?)?.cast<String, dynamic>() ?? {};
    return PredictResult(
      prediction: j['prediction'] as String,
      confidence: (j['confidence'] as num).toDouble(),
      pSatisfied: (j['p_satisfied'] as num).toDouble(),
      threshold: (j['threshold'] as num).toDouble(),
      termsNegative: ((terms['toward_dissatisfied'] as List?) ?? [])
          .map((e) => e.toString())
          .toList(),
      termsPositive: ((terms['toward_satisfied'] as List?) ?? [])
          .map((e) => e.toString())
          .toList(),
      signals: (j['signals'] as Map?)?.cast<String, dynamic>() ?? {},
      model: (j['model'] ?? '') as String,
      representation: (j['representation'] ?? '') as String,
    );
  }
}
