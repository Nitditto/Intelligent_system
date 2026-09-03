import 'package:flutter/material.dart';

import '../main.dart';
import '../models.dart';

/// Shows one PredictResult: verdict, confidence, P(satisfied) bar, the delivery
/// / comment signals that drove it, comment terms, and a short interpretation.
class ResultScreen extends StatelessWidget {
  final PredictResult result;
  const ResultScreen({super.key, required this.result});

  String _pct(double x) => '${(x * 100).toStringAsFixed(1)}%';

  @override
  Widget build(BuildContext context) {
    final good = result.good;
    final color = verdictColor(good);
    final s = result.signals;
    final daysVsPromise = s['days_vs_promise'];
    final late = s['late'] == true;

    return Scaffold(
      appBar: AppBar(title: const Text('Prediction')),
      body: ListView(
        padding: const EdgeInsets.fromLTRB(16, 16, 16, 40),
        children: [
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: color.withOpacity(0.10),
              borderRadius: BorderRadius.circular(12),
            ),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Flexible(
                  child: Text(
                    good
                        ? 'Likely satisfied'
                        : 'At risk — likely a bad review',
                    style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                        color: color),
                  ),
                ),
                Text('${_pct(result.confidence)} conf.',
                    style: const TextStyle(color: Colors.black54)),
              ],
            ),
          ),
          const SizedBox(height: 16),

          Text('P(satisfied) = ${_pct(result.pSatisfied)}',
              style: const TextStyle(fontWeight: FontWeight.w600)),
          const SizedBox(height: 6),
          ClipRRect(
            borderRadius: BorderRadius.circular(999),
            child: Stack(
              children: [
                LinearProgressIndicator(
                  value: result.pSatisfied,
                  minHeight: 12,
                  backgroundColor: const Color(0xFFEDF0F4),
                  valueColor: AlwaysStoppedAnimation(color),
                ),
                Positioned(
                  left: MediaQuery.of(context).size.width *
                      result.threshold *
                      0.86,
                  child: Container(width: 2, height: 12, color: Colors.black),
                ),
              ],
            ),
          ),
          Padding(
            padding: const EdgeInsets.only(top: 4),
            child: Text('decision cut-off ${_pct(result.threshold)}',
                style:
                    const TextStyle(fontSize: 12, color: Colors.black54)),
          ),
          const SizedBox(height: 18),

          const Text('Signals', style: TextStyle(fontWeight: FontWeight.w700)),
          const SizedBox(height: 6),
          _row(
            'Delivery vs. promise',
            daysVsPromise == null
                ? 'unknown'
                : late
                    ? '$daysVsPromise days late'
                    : '${daysVsPromise.abs()} days early',
            late ? verdictColor(false) : verdictColor(true),
          ),
          _row('Total delivery time', '${s['delivery_days'] ?? '—'} days', null),
          _row('Customer left a comment', s['has_comment'] == true ? 'yes' : 'no',
              null),
          _row('Category / region',
              '${s['category_group']} · ${s['customer_region']}', null),
          const SizedBox(height: 16),

          if (result.termsNegative.isNotEmpty || result.termsPositive.isNotEmpty)
            _Terms(neg: result.termsNegative, pos: result.termsPositive),

          const SizedBox(height: 8),
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: const Color(0xFFF4F6F9),
              border: const Border(
                  left: BorderSide(color: Color(0xFF2F6FED), width: 3)),
            ),
            child: Text(
              good
                  ? 'The order looks on track — on-time delivery and no strong '
                      'negative language in the comment. No proactive support '
                      'action needed.'
                  : 'This order matches the pattern of purchases that end in a '
                      '1–3 star review — most often driven by late delivery '
                      'relative to the promised date and confirmed by the '
                      'comment. Consider a proactive support contact before the '
                      'review is posted.',
              style: const TextStyle(fontSize: 13),
            ),
          ),
          const SizedBox(height: 20),
          Text(
            'model: ${result.model} (${result.representation})  ·  '
            'POST /predict',
            style: const TextStyle(fontSize: 11, color: Colors.black45),
          ),
          const SizedBox(height: 16),
          Center(
            child: OutlinedButton.icon(
              onPressed: () => Navigator.of(context).pop(),
              icon: const Icon(Icons.edit_outlined),
              label: const Text('Edit the order'),
            ),
          ),
        ],
      ),
    );
  }

  Widget _row(String k, String v, Color? valueColor) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 3),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Expanded(flex: 5, child: Text(k, style: const TextStyle(fontSize: 13))),
            Expanded(
              flex: 4,
              child: Text(v,
                  style: TextStyle(
                      fontSize: 13,
                      fontWeight: FontWeight.w600,
                      color: valueColor)),
            ),
          ],
        ),
      );
}

class _Terms extends StatelessWidget {
  final List<String> neg;
  final List<String> pos;
  const _Terms({required this.neg, required this.pos});

  @override
  Widget build(BuildContext context) {
    Widget chips(String title, List<String> items, Color c) => Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(title,
                style: TextStyle(
                    fontSize: 11,
                    color: c,
                    fontWeight: FontWeight.w700,
                    letterSpacing: 0.4)),
            const SizedBox(height: 4),
            Wrap(
              spacing: 6,
              runSpacing: 6,
              children: items
                  .map((t) => Chip(
                        label: Text(t, style: const TextStyle(fontSize: 12)),
                        visualDensity: VisualDensity.compact,
                        materialTapTargetSize:
                            MaterialTapTargetSize.shrinkWrap,
                      ))
                  .toList(),
            ),
            const SizedBox(height: 10),
          ],
        );

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        if (neg.isNotEmpty)
          chips('COMMENT TERMS PUSHING NEGATIVE', neg, verdictColor(false)),
        if (pos.isNotEmpty)
          chips('TERMS PUSHING POSITIVE', pos, verdictColor(true)),
      ],
    );
  }
}
