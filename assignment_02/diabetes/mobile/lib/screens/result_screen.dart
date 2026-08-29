import 'package:flutter/material.dart';

import '../main.dart';
import '../models.dart';
import 'whatif_screen.dart';

/// Screen 2 — the result: risk band, SHAP factor bars, similar cases, and the
/// counterfactual sentence. A button opens the interactive what-if screen.
class ResultScreen extends StatelessWidget {
  final PredictResult result;
  final Map<String, dynamic> payload;
  const ResultScreen({super.key, required this.result, required this.payload});

  @override
  Widget build(BuildContext context) {
    final r = result;
    final pct = (r.probability * 100).round();
    return Scaffold(
      appBar: AppBar(title: const Text('Result')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          _RiskCard(pct: pct, band: r.band, label: r.bandLabel,
              uncertainty: r.uncertaintyBand),
          for (final w in r.warnings)
            Padding(
              padding: const EdgeInsets.only(top: 8),
              child: Row(children: [
                const Icon(Icons.warning_amber, size: 18),
                const SizedBox(width: 6),
                Expanded(child: Text(w, style: const TextStyle(fontSize: 12))),
              ]),
            ),
          const SizedBox(height: 8),
          Text(
            'Completeness ${(r.completeness * 100).round()}% · BMI source: ${r.bmiSource}\n'
            'This is a screening aid, not a diagnosis.',
            style: const TextStyle(fontSize: 12, color: Colors.black54),
          ),
          const Divider(height: 32),

          // --- SHAP ---
          Text('Why this score', style: Theme.of(context).textTheme.titleMedium),
          const SizedBox(height: 8),
          if (r.factors.isEmpty)
            const Text('Explanation not available.',
                style: TextStyle(color: Colors.black54))
          else
            ...r.factors.take(8).map((f) => _FactorBar(
                  label: f.label,
                  value: f.value,
                  maxAbs: r.factors
                      .map((x) => x.value.abs())
                      .fold<double>(0.01, (a, b) => a > b ? a : b),
                )),
          const SizedBox(height: 4),
          const Text('Red raises the estimate, blue lowers it.',
              style: TextStyle(fontSize: 12, color: Colors.black54)),
          const Divider(height: 32),

          // --- similar cases ---
          Text('People similar to you',
              style: Theme.of(context).textTheme.titleMedium),
          const SizedBox(height: 8),
          if (r.similarAvailable) ...[
            Text(
              '${r.similarWithDiabetes} of your ${r.similarK} closest matches in the '
              'survey data had diabetes or pre-diabetes.',
            ),
            const SizedBox(height: 8),
            ...r.neighbors.map((n) => ListTile(
                  dense: true,
                  contentPadding: EdgeInsets.zero,
                  leading: Icon(
                    n.outcome.startsWith('diabetes')
                        ? Icons.circle
                        : Icons.circle_outlined,
                    size: 14,
                    color: n.outcome.startsWith('diabetes')
                        ? Colors.red
                        : Colors.green,
                  ),
                  title: Text(n.profile, style: const TextStyle(fontSize: 13)),
                  subtitle: Text(
                      '${n.outcome} · proximity ${n.proximity.toStringAsFixed(2)}',
                      style: const TextStyle(fontSize: 11)),
                )),
          ] else
            const Text('Similar-case index not available on the API.',
                style: TextStyle(color: Colors.black54)),
          const Divider(height: 32),

          // --- counterfactual + what-if link ---
          Text('What could change the estimate',
              style: Theme.of(context).textTheme.titleMedium),
          const SizedBox(height: 8),
          if (r.counterfactual != null &&
              (r.counterfactual!['changes'] as List).isNotEmpty)
            _CounterfactualText(r.counterfactual!),
          const SizedBox(height: 12),
          OutlinedButton.icon(
            icon: const Icon(Icons.tune),
            label: const Text('Explore modifiable factors'),
            onPressed: () => Navigator.push(
              context,
              MaterialPageRoute(
                builder: (_) => WhatIfScreen(result: r, payload: payload),
              ),
            ),
          ),
          if (r.caveat != null) ...[
            const SizedBox(height: 12),
            Text('⚠️ ${r.caveat}',
                style: const TextStyle(fontSize: 12, color: Colors.black54)),
          ],
          const SizedBox(height: 32),
        ],
      ),
    );
  }
}

class _RiskCard extends StatelessWidget {
  final int pct;
  final String band;
  final String label;
  final List<double>? uncertainty;
  const _RiskCard(
      {required this.pct,
      required this.band,
      required this.label,
      required this.uncertainty});

  @override
  Widget build(BuildContext context) {
    final c = bandColor(band);
    return Card(
      color: c.withOpacity(0.12),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              crossAxisAlignment: CrossAxisAlignment.baseline,
              textBaseline: TextBaseline.alphabetic,
              children: [
                Text('$pct%',
                    style: TextStyle(
                        fontSize: 40, fontWeight: FontWeight.bold, color: c)),
                const SizedBox(width: 12),
                Text(band,
                    style: TextStyle(
                        fontSize: 20, fontWeight: FontWeight.w600, color: c)),
              ],
            ),
            const SizedBox(height: 4),
            Text(label),
            if (uncertainty != null)
              Padding(
                padding: const EdgeInsets.only(top: 4),
                child: Text(
                  'Could range ${(uncertainty![0] * 100).round()}%–'
                  '${(uncertainty![1] * 100).round()}% depending on unanswered questions.',
                  style: const TextStyle(fontSize: 12, color: Colors.black54),
                ),
              ),
          ],
        ),
      ),
    );
  }
}

class _FactorBar extends StatelessWidget {
  final String label;
  final double value;
  final double maxAbs;
  const _FactorBar(
      {required this.label, required this.value, required this.maxAbs});

  @override
  Widget build(BuildContext context) {
    final frac = (value.abs() / maxAbs).clamp(0.0, 1.0);
    final positive = value > 0;
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 3),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('$label  (${value >= 0 ? '+' : ''}${value.toStringAsFixed(3)})',
              style: const TextStyle(fontSize: 12)),
          const SizedBox(height: 2),
          Align(
            alignment: Alignment.centerLeft,
            child: FractionallySizedBox(
              widthFactor: frac == 0 ? 0.01 : frac,
              child: Container(
                height: 10,
                decoration: BoxDecoration(
                  color: positive ? Colors.red.shade400 : Colors.blue.shade400,
                  borderRadius: BorderRadius.circular(4),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _CounterfactualText extends StatelessWidget {
  final Map<String, dynamic> cf;
  const _CounterfactualText(this.cf);
  @override
  Widget build(BuildContext context) {
    final changes = (cf['changes'] as List)
        .map((c) => '${c['label']} → ${c['to']}')
        .join(', ');
    final feasible = cf['feasible'] == true;
    final fromPct = ((cf['from_risk'] as num) * 100).round();
    final toPct = ((cf['to_risk'] as num) * 100).round();
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.blue.shade50,
        borderRadius: BorderRadius.circular(8),
      ),
      child: Text(
        'If $changes, the estimate moves from $fromPct% (${cf['from_band']}) '
        'to $toPct% (${cf['to_band']}) — '
        '${feasible ? 'would reach' : 'would still not reach'} the '
        '${cf['target_band']} band.',
        style: const TextStyle(fontSize: 13),
      ),
    );
  }
}
