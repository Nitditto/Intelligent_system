// REST client for the customer-behaviour API. The mobile app performs no
// inference itself — it enters a review, POSTs it, and shows the response.
//
// Base URL: pass --dart-define=API_URL=http://<host>:8000 at run time.
// Default 10.0.2.2 is the Android emulator's alias for the host's localhost.

import 'dart:convert';
import 'package:http/http.dart' as http;

import 'models.dart';

class ApiException implements Exception {
  final String message;
  ApiException(this.message);
  @override
  String toString() => message;
}

class ApiClient {
  static const String baseUrl =
      String.fromEnvironment('API_URL', defaultValue: 'http://10.0.2.2:8000');

  final http.Client _http = http.Client();

  Uri _u(String path) => Uri.parse('$baseUrl$path');

  Future<bool> health() async {
    try {
      final r =
          await _http.get(_u('/healthz')).timeout(const Duration(seconds: 5));
      return r.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  Future<List<FormFieldSpec>> questions() async {
    final r = await _http.get(_u('/questions'));
    _check(r);
    final body = jsonDecode(r.body) as Map<String, dynamic>;
    return (body['fields'] as List)
        .map((q) => FormFieldSpec.fromJson((q as Map).cast<String, dynamic>()))
        .toList();
  }

  Future<Map<String, dynamic>> modelInfo() async {
    final r = await _http.get(_u('/model-info'));
    _check(r);
    return jsonDecode(r.body) as Map<String, dynamic>;
  }

  /// `{ "defaults": {...}, "examples": [ {...review..., "_label", "_recommended"} ] }`
  Future<Map<String, dynamic>> samples() async {
    final r = await _http.get(_u('/samples'));
    _check(r);
    return jsonDecode(r.body) as Map<String, dynamic>;
  }

  Future<PredictResult> predict(Map<String, dynamic> payload) async {
    final r = await _http.post(
      _u('/predict'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode(payload),
    );
    if (r.statusCode == 422) {
      throw ApiException('The server rejected the input:\n${r.body}');
    }
    _check(r);
    return PredictResult.fromJson(jsonDecode(r.body) as Map<String, dynamic>);
  }

  void _check(http.Response r) {
    if (r.statusCode >= 400) {
      throw ApiException('API error ${r.statusCode}: ${r.body}');
    }
  }
}
