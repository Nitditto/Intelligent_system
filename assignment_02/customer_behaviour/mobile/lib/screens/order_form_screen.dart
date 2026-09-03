import 'package:flutter/material.dart';
import 'package:intl/intl.dart';

import '../main.dart';
import '../models.dart';
import 'result_screen.dart';

/// One screen: enter an order, submit, see the prediction (on the next screen).
class OrderFormScreen extends StatefulWidget {
  const OrderFormScreen({super.key});

  @override
  State<OrderFormScreen> createState() => _OrderFormScreenState();
}

class _OrderFormScreenState extends State<OrderFormScreen> {
  final _formKey = GlobalKey<FormState>();
  List<FormFieldSpec>? _fields;
  String? _loadError;
  bool? _apiUp;
  String _modelName = '';
  bool _busy = false;

  // field -> current value (String for text/choice/number, DateTime for date)
  final Map<String, dynamic> _values = {};

  static final _fmt = DateFormat('yyyy-MM-dd');

  // Prefilled "late + damaged" sample so the form is usable immediately.
  static final Map<String, dynamic> _sample = {
    'category': 'bed_bath_table',
    'price_total': '129.90',
    'freight_total': '18.30',
    'payment_value_total': '148.20',
    'n_items': '1',
    'n_sellers': '1',
    'main_payment_type': 'credit_card',
    'max_installments': '3',
    'customer_state': 'SP',
    'product_weight_g': '1200',
    'product_photos_qty': '3',
    'product_desc_len': '850',
    'order_purchase_timestamp': DateTime(2018, 5, 1),
    'order_estimated_delivery_date': DateTime(2018, 5, 20),
    'order_delivered_customer_date': DateTime(2018, 5, 31),
    'review_comment_message':
        'Produto chegou muito atrasado e a embalagem estava danificada.',
  };

  @override
  void initState() {
    super.initState();
    _values.addAll(_sample);
    _load();
  }

  Future<void> _load() async {
    api.health().then((v) => setState(() => _apiUp = v));
    api.modelInfo().then((m) {
      setState(() => _modelName = (m['chosen_model'] ?? '').toString());
    }).catchError((_) {});
    try {
      final f = await api.questions();
      setState(() => _fields = f);
    } catch (e) {
      setState(() => _loadError = '$e');
    }
  }

  Map<String, dynamic> _buildPayload() {
    final p = <String, dynamic>{};
    for (final f in _fields!) {
      final v = _values[f.field];
      if (v == null || (v is String && v.trim().isEmpty)) continue;
      if (v is DateTime) {
        p[f.field] = '${_fmt.format(v)} 12:00:00';
      } else if (f.type == 'number') {
        p[f.field] = num.tryParse(v.toString());
      } else {
        p[f.field] = v;
      }
    }
    return p;
  }

  Future<void> _submit() async {
    if (!_formKey.currentState!.validate()) return;
    setState(() => _busy = true);
    try {
      final result = await api.predict(_buildPayload());
      if (!mounted) return;
      Navigator.of(context).push(MaterialPageRoute(
        builder: (_) => ResultScreen(result: result),
      ));
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context)
          .showSnackBar(SnackBar(content: Text('$e')));
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Will this order get a good review?'),
        bottom: PreferredSize(
          preferredSize: const Size.fromHeight(24),
          child: Padding(
            padding: const EdgeInsets.only(bottom: 6, left: 16, right: 16),
            child: Align(
              alignment: Alignment.centerLeft,
              child: Text(
                _apiUp == false
                    ? 'API offline — run  uvicorn api.main:app --port 8000'
                    : _apiUp == true
                        ? 'API · ${_modelName.isEmpty ? "ready" : _modelName}'
                        : 'connecting…',
                style: TextStyle(
                  fontSize: 12,
                  color: _apiUp == false
                      ? Colors.red.shade100
                      : Colors.white70,
                ),
              ),
            ),
          ),
        ),
      ),
      body: _loadError != null
          ? _ErrorView(message: _loadError!, onRetry: _load)
          : _fields == null
              ? const Center(child: CircularProgressIndicator())
              : Form(
                  key: _formKey,
                  child: ListView(
                    padding: const EdgeInsets.fromLTRB(16, 12, 16, 96),
                    children: [
                      const Text(
                        'Predicts review_score ≥ 4 from delivery, payment and '
                        'product details plus the customer comment. Nothing runs '
                        'on the device — this is a POST /predict call.',
                        style: TextStyle(color: Colors.black54, fontSize: 13),
                      ),
                      const SizedBox(height: 12),
                      for (final f in _fields!) _buildField(f),
                    ],
                  ),
                ),
      floatingActionButton: _fields == null
          ? null
          : FloatingActionButton.extended(
              onPressed: _busy ? null : _submit,
              icon: _busy
                  ? const SizedBox(
                      width: 18,
                      height: 18,
                      child: CircularProgressIndicator(
                          strokeWidth: 2, color: Colors.white))
                  : const Icon(Icons.assessment_outlined),
              label: Text(_busy ? 'Predicting…' : 'Predict satisfaction'),
            ),
    );
  }

  Widget _buildField(FormFieldSpec f) {
    final pad = const EdgeInsets.only(bottom: 14);
    switch (f.type) {
      case 'choice':
        return Padding(
          padding: pad,
          child: DropdownButtonFormField<String>(
            value: _values[f.field] as String?,
            isExpanded: true,
            decoration: _dec(f),
            items: [
              const DropdownMenuItem(value: null, child: Text('—')),
              ...f.options.map(
                (o) => DropdownMenuItem(value: o, child: Text(o)),
              ),
            ],
            validator: (v) =>
                f.required && (v == null) ? 'Required' : null,
            onChanged: (v) => setState(() => _values[f.field] = v),
          ),
        );
      case 'date':
        final dt = _values[f.field] as DateTime?;
        return Padding(
          padding: pad,
          child: InputDecorator(
            decoration: _dec(f),
            child: Row(
              children: [
                Expanded(
                  child: Text(dt == null ? '—' : _fmt.format(dt)),
                ),
                TextButton(
                  onPressed: () async {
                    final picked = await showDatePicker(
                      context: context,
                      initialDate: dt ?? DateTime(2018, 5, 1),
                      firstDate: DateTime(2016),
                      lastDate: DateTime(2020),
                    );
                    if (picked != null) {
                      setState(() => _values[f.field] = picked);
                    }
                  },
                  child: const Text('Pick'),
                ),
              ],
            ),
          ),
        );
      case 'text':
        return Padding(
          padding: pad,
          child: TextFormField(
            initialValue: _values[f.field] as String?,
            maxLines: 3,
            decoration: _dec(f),
            onChanged: (v) => _values[f.field] = v,
          ),
        );
      default: // number
        return Padding(
          padding: pad,
          child: TextFormField(
            initialValue: _values[f.field]?.toString(),
            keyboardType:
                const TextInputType.numberWithOptions(decimal: true),
            decoration: _dec(f),
            validator: (v) {
              if (f.required && (v == null || v.trim().isEmpty)) {
                return 'Required';
              }
              if (v != null && v.trim().isNotEmpty) {
                final n = num.tryParse(v);
                if (n == null) return 'Enter a number';
                if (f.min != null && n < f.min!) return 'Min ${f.min}';
                if (f.max != null && n > f.max!) return 'Max ${f.max}';
              }
              return null;
            },
            onChanged: (v) => _values[f.field] = v,
          ),
        );
    }
  }

  InputDecoration _dec(FormFieldSpec f) => InputDecoration(
        labelText: f.label + (f.required ? ' *' : ''),
        helperText: f.note,
        helperMaxLines: 4,
        border: const OutlineInputBorder(),
        isDense: true,
      );
}

class _ErrorView extends StatelessWidget {
  final String message;
  final VoidCallback onRetry;
  const _ErrorView({required this.message, required this.onRetry});

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(Icons.cloud_off, size: 40, color: Colors.grey),
            const SizedBox(height: 12),
            Text(message, textAlign: TextAlign.center),
            const SizedBox(height: 12),
            FilledButton(onPressed: onRetry, child: const Text('Retry')),
          ],
        ),
      ),
    );
  }
}
