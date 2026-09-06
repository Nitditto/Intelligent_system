import 'package:flutter/material.dart';

import 'api_client.dart';
import 'screens/order_form_screen.dart';

void main() => runApp(const RecommendationApp());

final ApiClient api = ApiClient();

class RecommendationApp extends StatelessWidget {
  const RecommendationApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Product recommendation',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorSchemeSeed: const Color(0xFF2F6FED),
        useMaterial3: true,
      ),
      home: const ReviewFormScreen(),
    );
  }
}

Color verdictColor(bool good) =>
    good ? const Color(0xFF1F9D55) : const Color(0xFFD64545);
