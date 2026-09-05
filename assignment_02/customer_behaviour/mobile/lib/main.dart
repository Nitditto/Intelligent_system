import 'package:flutter/material.dart';

import 'api_client.dart';
import 'screens/order_form_screen.dart';

void main() => runApp(const OrderSatisfactionApp());

final ApiClient api = ApiClient();

class OrderSatisfactionApp extends StatelessWidget {
  const OrderSatisfactionApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Order satisfaction',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorSchemeSeed: const Color(0xFF2F6FED),
        useMaterial3: true,
      ),
      home: const OrderFormScreen(),
    );
  }
}

Color verdictColor(bool good) =>
    good ? const Color(0xFF1F9D55) : const Color(0xFFD64545);
