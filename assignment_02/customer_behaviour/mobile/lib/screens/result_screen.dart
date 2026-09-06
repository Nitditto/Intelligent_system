import 'package:flutter/material.dart';

import '../models.dart';
import '../theme.dart';
import '../widgets/primitives.dart';
import '../widgets/result_parts.dart';

class ResultScreen extends StatelessWidget {
  final PredictResult result;
  final Map<String, dynamic> modelInfo;
  const ResultScreen(
      {super.key, required this.result, this.modelInfo = const {}});

  String _pct(double x) => '${(x * 100).round()}%';
  String? _s(dynamic v) => v?.toString();

  @override
  Widget build(BuildContext context) {
    final c = context.c;
    final good = result.good;
    final s = result.signals;
    final pRec = (result.pRecommend * 100).round();

    final kv = <(String, String)>[
      ('Skin type', _s(s['skin_type']) ?? '—'),
      ('Category', _s(s['category']) ?? '—'),
      ('Brand', _s(s['brand']) ?? '—'),
      (
        'Price',
        s['price_usd'] == null
            ? '—'
            : '\$${s['price_usd']}'
                '${s['price_tier'] != null ? ' · ${s['price_tier']}' : ''}'
      ),
      ('Product “loves”', _s(s['product_loves']) ?? '—'),
      ('Review length', '${s['review_tokens'] ?? '—'} words'),
    ];

    final against = result.termsAgainst.map((t) => t.term).toList();
    final toward = result.termsToward.map((t) => t.term).toList();
    final framing = (modelInfo['framing'] ?? '').toString();

    return Scaffold(
      appBar: AppBar(title: const Text('Prediction')),
      body: ListView(
        padding: const EdgeInsets.fromLTRB(Sp.s4, Sp.s4, Sp.s4, Sp.s8),
        children: [
          VerdictBlock(
            good: good,
            badge: good ? 'RECOMMENDS' : "WON’T RECOMMEND",
            title: good
                ? 'This customer would probably recommend the product'
                : 'This customer probably would not recommend the product',
            subtitle:
                'The model puts the chance the reviewer ticks “recommend” at $pRec%. '
                'We call it “recommends” above ${_pct(result.threshold)}.',
          ),
          const SizedBox(height: Sp.s4),

          // meter
          AppCard(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const SectionLabel('Chance of a recommendation'),
                const SizedBox(height: Sp.s3),
                Meter(
                  value: result.pRecommend,
                  threshold: result.threshold,
                  good: good,
                  leftLabel: 'won’t',
                  rightLabel: 'recommends',
                ),
                const SizedBox(height: Sp.s3),
                Text(
                  'Read it as: $pRec times out of 100, a reviewer who wrote this — '
                  'with this profile and product — ticked “recommend”.',
                  style: TextStyle(fontSize: Ty.xs, color: c.textSoft, height: 1.5),
                ),
              ],
            ),
          ),
          const SizedBox(height: Sp.s4),

          // signals
          AppCard(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const SectionLabel('What the model saw'),
                const SizedBox(height: Sp.s3),
                KvGrid(kv),
                const SizedBox(height: Sp.s3),
                Text(
                  'Skin type is the main structured signal — a product that suits dry '
                  'skin often disappoints oily reviewers. The structured block alone '
                  'reaches ROC-AUC ~0.8; the review text takes it to ~0.96.',
                  style: TextStyle(fontSize: Ty.xs, color: c.textSoft, height: 1.5),
                ),
              ],
            ),
          ),

          if (against.isNotEmpty || toward.isNotEmpty) ...[
            const SizedBox(height: Sp.s4),
            AppCard(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const SectionLabel('Words in the review that moved the call'),
                  const SizedBox(height: Sp.s3),
                  if (against.isNotEmpty) ...[
                    Text('▼ toward “won’t recommend”',
                        style: TextStyle(
                            fontSize: 11,
                            fontWeight: FontWeight.w800,
                            color: c.bad)),
                    const SizedBox(height: 6),
                    ChipRow(against, good: false),
                    const SizedBox(height: Sp.s3),
                  ],
                  if (toward.isNotEmpty) ...[
                    Text('▲ toward “recommends”',
                        style: TextStyle(
                            fontSize: 11,
                            fontWeight: FontWeight.w800,
                            color: c.good)),
                    const SizedBox(height: 6),
                    ChipRow(toward, good: true),
                  ],
                ],
              ),
            ),
          ],

          const SizedBox(height: Sp.s4),
          // interpretation
          Container(
            padding: const EdgeInsets.all(Sp.s4),
            decoration: BoxDecoration(
              color: good ? c.goodWeak : c.badWeak,
              borderRadius: Rad.rMd,
              border: Border(
                  left: BorderSide(color: good ? c.good : c.bad, width: 4)),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(children: [
                  Icon(good ? Icons.check_circle : Icons.warning_amber_rounded,
                      size: 16, color: good ? c.good : c.bad),
                  const SizedBox(width: 6),
                  Text('HOW TO USE THIS',
                      style: TextStyle(
                          fontSize: 11,
                          fontWeight: FontWeight.w800,
                          letterSpacing: 0.5,
                          color: c.textFaint)),
                ]),
                const SizedBox(height: Sp.s2),
                Text(
                  good
                      ? 'Text and profile agree the customer is satisfied — safe to '
                          'surface this review for similar skin types.'
                      : 'The review reads negative — flag for review-consistency QA '
                          'and check whether the product page over-promises for this '
                          'skin type.',
                  style: TextStyle(fontSize: Ty.sm, color: c.text, height: 1.5),
                ),
              ],
            ),
          ),

          if (result.contributions != null) ...[
            const SizedBox(height: Sp.s4),
            AppCard(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const SectionLabel('Why this prediction — each factor’s pull'),
                  const SizedBox(height: Sp.s3),
                  ContribChart(result.contributions!),
                ],
              ),
            ),
          ],

          const SizedBox(height: Sp.s4),
          HowItWorks(
            'Trained on ~104k Sephora skincare reviews (product-id shard 500–750). '
            'One Logistic Regression over the reviewer’s skin profile + the product '
            '(category, brand, price, popularity) and a TF-IDF of the review title + '
            'body. Inference runs server-side (POST /predict). '
            '${framing.isEmpty ? '' : framing}',
          ),

          const SizedBox(height: Sp.s5),
          Center(
            child: AppButton(
              label: 'Score another review',
              icon: Icons.arrow_back,
              onPressed: () => Navigator.of(context).pop(),
            ),
          ),
          const SizedBox(height: Sp.s3),
          Text(
            '${result.model} · ${result.representation}',
            textAlign: TextAlign.center,
            style: TextStyle(fontSize: 10, color: c.textFaint),
          ),
        ],
      ),
    );
  }
}
