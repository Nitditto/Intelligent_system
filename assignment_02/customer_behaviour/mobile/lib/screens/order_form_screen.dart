import 'package:flutter/material.dart';

import '../main.dart';
import '../models.dart';
import 'result_screen.dart';

/// One screen: enter a review, submit, see the prediction (on the next screen).
class ReviewFormScreen extends StatefulWidget {
  const ReviewFormScreen({super.key});

  @override
  State<ReviewFormScreen> createState() => _ReviewFormScreenState();
}

class _ReviewFormScreenState extends State<ReviewFormScreen> {
  final _formKey = GlobalKey<FormState>();
  List<FormFieldSpec>? _fields;
  String? _loadError;
  bool? _apiUp;
  String _modelName = '';
  bool _busy = false;

  final Map<String, dynamic> _values = {};

  // Prefilled "oily skin, broke me out" sample so the form is usable immediately.
  static final Map<String, dynamic> _sample = {
    'skin_type': 'oily',
    'skin_tone': 'light',
    'eye_color': 'brown',
    'hair_color': 'brown',
    'secondary_category': 'Moisturizers',
    'brand_name': 'Skinfix',
    'price_usd': '32',
    'loves_count': '21000',
    'reviews': '1800',
    'review_title': 'Not for oily skin',
    'review_text':
        'Broke me out within a week and felt greasy all day. Smells strongly '
        'of perfume. Wanted to love it but returned it.',
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
      if (f.type == 'number') {
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
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('$e')));
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final fields = _fields;
    final sections = <String>[];
    if (fields != null) {
      for (final f in fields) {
        if (!sections.contains(f.section)) sections.add(f.section);
      }
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text('Will this customer recommend it?'),
        bottom: PreferredSize(
          preferredSize: const Size.fromHeight(22),
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
                  color:
                      _apiUp == false ? Colors.red.shade100 : Colors.white70,
                ),
              ),
            ),
          ),
        ),
      ),
      body: _loadError != null
          ? _ErrorView(message: _loadError!, onRetry: _load)
          : fields == null
              ? const Center(child: CircularProgressIndicator())
              : Form(
                  key: _formKey,
                  child: ListView(
                    padding: const EdgeInsets.fromLTRB(16, 12, 16, 96),
                    children: [
                      const Text(
                        'Predicts is_recommended from the reviewer’s skin '
                        'profile + the product + the review text. Nothing runs '
                        'on the device — this is a POST /predict call.',
                        style: TextStyle(color: Colors.black54, fontSize: 13),
                      ),
                      const SizedBox(height: 12),
                      for (final sec in sections) ...[
                        Padding(
                          padding: const EdgeInsets.only(top: 6, bottom: 8),
                          child: Text(
                            sec.toUpperCase(),
                            style: const TextStyle(
                              fontSize: 11,
                              fontWeight: FontWeight.w800,
                              letterSpacing: 0.6,
                              color: Colors.black45,
                            ),
                          ),
                        ),
                        for (final f in fields.where((x) => x.section == sec))
                          _buildField(f),
                      ],
                    ],
                  ),
                ),
      floatingActionButton: fields == null
          ? null
          : FloatingActionButton.extended(
              onPressed: _busy ? null : _submit,
              icon: _busy
                  ? const SizedBox(
                      width: 18,
                      height: 18,
                      child: CircularProgressIndicator(
                          strokeWidth: 2, color: Colors.white))
                  : const Icon(Icons.reviews_outlined),
              label: Text(_busy ? 'Scoring…' : 'Predict'),
            ),
    );
  }

  Widget _buildField(FormFieldSpec f) {
    const pad = EdgeInsets.only(bottom: 14);
    switch (f.type) {
      case 'choice':
        return Padding(
          padding: pad,
          child: DropdownButtonFormField<String>(
            initialValue: _values[f.field] as String?,
            isExpanded: true,
            decoration: _dec(f),
            items: [
              const DropdownMenuItem(value: null, child: Text('—')),
              ...f.options
                  .map((o) => DropdownMenuItem(value: o, child: Text(o))),
            ],
            validator: (v) => f.required && v == null ? 'Required' : null,
            onChanged: (v) => setState(() => _values[f.field] = v),
          ),
        );
      case 'text':
        return Padding(
          padding: pad,
          child: TextFormField(
            initialValue: _values[f.field] as String?,
            decoration: _dec(f),
            onChanged: (v) => _values[f.field] = v,
          ),
        );
      case 'textarea':
        return Padding(
          padding: pad,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              TextFormField(
                initialValue: _values[f.field] as String?,
                maxLines: 5,
                decoration: _dec(f),
                validator: (v) => f.required && (v == null || v.trim().isEmpty)
                    ? 'Required'
                    : null,
                onChanged: (v) => setState(() => _values[f.field] = v),
              ),
              if (f.examples.isNotEmpty)
                Padding(
                  padding: const EdgeInsets.only(top: 8),
                  child: Wrap(
                    spacing: 6,
                    runSpacing: 6,
                    children: f.examples
                        .map((ex) => ActionChip(
                              label: Text(ex[0],
                                  style: const TextStyle(fontSize: 12)),
                              onPressed: () => setState(() {
                                _values['review_title'] = ex[0];
                                _values[f.field] = ex[1];
                              }),
                            ))
                        .toList(),
                  ),
                ),
            ],
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
