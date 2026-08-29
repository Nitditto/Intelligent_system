import 'package:flutter/material.dart';

import '../api_client.dart';
import '../main.dart';
import '../models.dart';
import 'history_screen.dart';
import 'result_screen.dart';

/// Screen 1 — the questionnaire. Rendered from `GET /questions`; every item has a
/// "Not sure" option, and a blank answer is sent to the API as null.
class QuestionnaireScreen extends StatefulWidget {
  const QuestionnaireScreen({super.key});

  @override
  State<QuestionnaireScreen> createState() => _QuestionnaireScreenState();
}

class _QuestionnaireScreenState extends State<QuestionnaireScreen> {
  late Future<List<Question>> _questionsF;
  final Map<String, dynamic> _answers = {}; // field -> value or null
  bool _submitting = false;

  @override
  void initState() {
    super.initState();
    _questionsF = api.questions();
  }

  Future<void> _submit(List<Question> questions) async {
    if (_answers['Age'] == null || _answers['Sex'] == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Age group and Sex are required.')),
      );
      return;
    }
    final payload = <String, dynamic>{'session_id': sessionId};
    _answers.forEach((k, v) {
      if (v != null) payload[k] = v;
    });
    setState(() => _submitting = true);
    try {
      final result = await api.predict(payload);
      if (!mounted) return;
      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (_) => ResultScreen(result: result, payload: payload),
        ),
      );
    } on ApiException catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context)
            .showSnackBar(SnackBar(content: Text(e.message)));
      }
    } finally {
      if (mounted) setState(() => _submitting = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Diabetes screening'),
        actions: [
          IconButton(
            icon: const Icon(Icons.history),
            tooltip: 'History',
            onPressed: () => Navigator.push(
              context,
              MaterialPageRoute(builder: (_) => const HistoryScreen()),
            ),
          ),
        ],
      ),
      body: FutureBuilder<List<Question>>(
        future: _questionsF,
        builder: (context, snap) {
          if (snap.hasError) {
            return _ErrorView(
              message: 'Could not reach the API at ${ApiClient.baseUrl}.\n'
                  'Start it with:  uvicorn api.main:app --port 8000',
              onRetry: () => setState(() => _questionsF = api.questions()),
            );
          }
          if (!snap.hasData) {
            return const Center(child: CircularProgressIndicator());
          }
          final questions = snap.data!;
          final sections = <String, List<Question>>{};
          for (final q in questions) {
            sections.putIfAbsent(q.section, () => []).add(q);
          }
          return ListView(
            padding: const EdgeInsets.all(16),
            children: [
              const Text(
                'Answer what you can. Choose "Not sure" for anything you do not '
                'know — the estimate still works, with a note that it is less certain.',
              ),
              const SizedBox(height: 12),
              for (final entry in sections.entries) ...[
                _SectionHeader(entry.key),
                for (final q in entry.value) _questionField(q),
                const SizedBox(height: 8),
              ],
              const SizedBox(height: 8),
              FilledButton.icon(
                onPressed: _submitting ? null : () => _submit(questions),
                icon: _submitting
                    ? const SizedBox(
                        width: 18,
                        height: 18,
                        child: CircularProgressIndicator(strokeWidth: 2))
                    : const Icon(Icons.calculate),
                label: const Text('Estimate diabetes risk'),
              ),
              const SizedBox(height: 32),
            ],
          );
        },
      ),
    );
  }

  Widget _questionField(Question q) {
    switch (q.type) {
      case 'yesno':
        final current = _answers[q.field];
        return Padding(
          padding: const EdgeInsets.symmetric(vertical: 6),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(q.label),
              const SizedBox(height: 4),
              SegmentedButton<String>(
                showSelectedIcon: false,
                segments: const [
                  ButtonSegment(value: 'yes', label: Text('Yes')),
                  ButtonSegment(value: 'no', label: Text('No')),
                  ButtonSegment(value: 'na', label: Text('Not sure')),
                ],
                selected: {
                  current == 1
                      ? 'yes'
                      : current == 0
                          ? 'no'
                          : 'na'
                },
                onSelectionChanged: (s) => setState(() {
                  _answers[q.field] =
                      s.first == 'yes' ? 1 : (s.first == 'no' ? 0 : null);
                }),
              ),
            ],
          ),
        );

      case 'choice':
        return Padding(
          padding: const EdgeInsets.symmetric(vertical: 6),
          child: DropdownButtonFormField<dynamic>(
            value: _answers[q.field],
            isExpanded: true,
            decoration: InputDecoration(
              labelText: q.label,
              border: const OutlineInputBorder(),
            ),
            items: [
              const DropdownMenuItem(value: null, child: Text('Not sure')),
              for (final o in q.options)
                DropdownMenuItem(value: o[0], child: Text('${o[1]}')),
            ],
            onChanged: (v) => setState(() => _answers[q.field] = v),
          ),
        );

      default: // number
        final na = _answers['${q.field}__na'] == true;
        return Padding(
          padding: const EdgeInsets.symmetric(vertical: 6),
          child: Row(
            children: [
              Expanded(
                child: TextFormField(
                  enabled: !na,
                  keyboardType: TextInputType.number,
                  decoration: InputDecoration(
                    labelText: q.label,
                    helperText: q.note,
                    border: const OutlineInputBorder(),
                  ),
                  onChanged: (t) =>
                      _answers[q.field] = t.trim().isEmpty ? null : num.tryParse(t),
                ),
              ),
              const SizedBox(width: 8),
              Column(
                children: [
                  const Text('N/A', style: TextStyle(fontSize: 11)),
                  Switch(
                    value: na,
                    onChanged: (v) => setState(() {
                      _answers['${q.field}__na'] = v;
                      if (v) _answers[q.field] = null;
                    }),
                  ),
                ],
              ),
            ],
          ),
        );
    }
  }
}

class _SectionHeader extends StatelessWidget {
  final String text;
  const _SectionHeader(this.text);
  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.only(top: 16, bottom: 4),
        child: Text(text,
            style: Theme.of(context).textTheme.titleMedium?.copyWith(
                fontWeight: FontWeight.bold)),
      );
}

class _ErrorView extends StatelessWidget {
  final String message;
  final VoidCallback onRetry;
  const _ErrorView({required this.message, required this.onRetry});
  @override
  Widget build(BuildContext context) => Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Icon(Icons.cloud_off, size: 40),
              const SizedBox(height: 12),
              Text(message, textAlign: TextAlign.center),
              const SizedBox(height: 12),
              OutlinedButton(onPressed: onRetry, child: const Text('Retry')),
            ],
          ),
        ),
      );
}
