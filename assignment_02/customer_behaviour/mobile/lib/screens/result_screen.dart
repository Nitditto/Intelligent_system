import 'package:flutter/material.dart';

import '../main.dart';
import '../models.dart';

/// Shows one PredictResult: verdict, P(recommend) bar, the profile/product
/// signals, the review terms that moved it, and a short interpretation.
class ResultScreen extends StatelessWidget {
  final PredictResult result;
  const ResultScreen({super.key, required this.result});

  String _pct(double x) => '${(x * 100).toStringAsFixed(1)}%';
  String? _s(dynamic v) => (v == null) ? null : v.toString();

  @override
  Widget build(BuildContext context) {
    final good = result.good;
    final color = verdictColor(good);
    final s = result.signals;

    return Scaffold(
      appBar: AppBar(title: const Text('Prediction')),
      body: ListView(
        padding: const EdgeInsets.fromLTRB(16, 16, 16, 40),
        children: [
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: color.withValues(alpha: 0.10),
              borderRadius: BorderRadius.circular(12),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  good
                      ? 'Would probably recommend'
                      : 'Probably would not recommend',
                  style: TextStyle(
                      fontSize: 18, fontWeight: FontWeight.bold, color: color),
                ),
                const SizedBox(height: 4),
                Text(
                  'Chance of a recommendation: ${_pct(result.pRecommend)} '
                  '(we call it “recommends” above ${_pct(result.threshold)}).',
                  style: const TextStyle(color: Colors.black54, fontSize: 13),
                ),
              ],
            ),
          ),
          const SizedBox(height: 16),

          const Text('Chance of a recommendation',
              style: TextStyle(fontWeight: FontWeight.w600)),
          const SizedBox(height: 6),
          ClipRRect(
            borderRadius: BorderRadius.circular(999),
            child: Stack(
              children: [
                LinearProgressIndicator(
                  value: result.pRecommend,
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
                style: const TextStyle(fontSize: 12, color: Colors.black54)),
          ),
          const SizedBox(height: 18),

          const Text('What the model saw',
              style: TextStyle(fontWeight: FontWeight.w700)),
          const SizedBox(height: 6),
          _row('Skin type', _s(s['skin_type']) ?? '—', null),
          _row('Category', _s(s['category']) ?? '—', null),
          _row('Brand', _s(s['brand']) ?? '—', null),
          _row(
            'Price',
            s['price_usd'] == null
                ? '—'
                : '\$${s['price_usd']}'
                    '${s['price_tier'] != null ? ' · ${s['price_tier']}' : ''}',
            null,
          ),
          _row('Product “loves”', _s(s['product_loves']) ?? '—', null),
          _row('Review length', '${s['review_tokens'] ?? '—'} words', null),
          const SizedBox(height: 16),

          if (result.termsAgainst.isNotEmpty || result.termsToward.isNotEmpty)
            _Terms(against: result.termsAgainst, toward: result.termsToward),

          const SizedBox(height: 8),
          Container(
            padding: const EdgeInsets.all(12),
            decoration: const BoxDecoration(
              color: Color(0xFFF4F6F9),
              border: Border(
                  left: BorderSide(color: Color(0xFF2F6FED), width: 3)),
            ),
            child: Text(
              good
                  ? 'Text and profile agree the customer is satisfied — safe to '
                      'surface this review for similar skin types.'
                  : 'The review reads negative — flag for review-consistency QA '
                      'and check whether the product page over-promises for this '
                      'skin type.',
              style: const TextStyle(fontSize: 13),
            ),
          ),
          const SizedBox(height: 16),
          Text(
            'model: ${result.model} (${result.representation})  ·  POST /predict\n'
            'The review text is written alongside the recommend tick, so it '
            'strongly signals the outcome — see notebook §14a.',
            style: const TextStyle(fontSize: 11, color: Colors.black45),
          ),
          const SizedBox(height: 16),
          Center(
            child: OutlinedButton.icon(
              onPressed: () => Navigator.of(context).pop(),
              icon: const Icon(Icons.edit_outlined),
              label: const Text('Edit the review'),
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
            Expanded(
                flex: 5,
                child: Text(k, style: const TextStyle(fontSize: 13))),
            Expanded(
              flex: 5,
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
  final List<TermPull> against;
  final List<TermPull> toward;
  const _Terms({required this.against, required this.toward});

  @override
  Widget build(BuildContext context) {
    Widget chips(String title, List<TermPull> items, Color c) => Column(
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
                        label: Text(t.term,
                            style: const TextStyle(fontSize: 12)),
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
        const Text('Words in the review that moved the call',
            style: TextStyle(fontWeight: FontWeight.w700)),
        const SizedBox(height: 8),
        if (against.isNotEmpty)
          chips('TOWARD “WON’T RECOMMEND”', against, verdictColor(false)),
        if (toward.isNotEmpty)
          chips('TOWARD “RECOMMENDS”', toward, verdictColor(true)),
      ],
    );
  }
}
