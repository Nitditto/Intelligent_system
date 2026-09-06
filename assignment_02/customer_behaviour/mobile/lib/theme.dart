import 'package:flutter/material.dart';

/// Design tokens ported from `.claude/skills/frontend-ui` so the mobile client
/// reads as the same product as the web client. Colours resolve per brightness
/// through an [AppColors] `ThemeExtension`; spacing / radius / type are constants.

// ---------------------------------------------------------------- spacing scale
class Sp {
  static const double s1 = 4, s2 = 8, s3 = 12, s4 = 16, s5 = 20, s6 = 24,
      s8 = 32, s10 = 40, s12 = 48;
}

class Rad {
  static const double sm = 6, md = 10, lg = 14, pill = 999;
  static const rSm = BorderRadius.all(Radius.circular(sm));
  static const rMd = BorderRadius.all(Radius.circular(md));
  static const rLg = BorderRadius.all(Radius.circular(lg));
  static const rPill = BorderRadius.all(Radius.circular(pill));
}

// ---------------------------------------------------------------- type scale
class Ty {
  static const double xs = 12, sm = 13, md = 15, lg = 18, xl = 22, xxl = 28;
  static const double lh = 1.5;
}

// ---------------------------------------------------------------- colour tokens
@immutable
class AppColors extends ThemeExtension<AppColors> {
  final Color bg, surface, surface2, border;
  final Color text, textSoft, textFaint;
  final Color accent, accentInk, accentWeak;
  final Color good, goodWeak, warn, warnWeak, bad, badWeak;

  const AppColors({
    required this.bg,
    required this.surface,
    required this.surface2,
    required this.border,
    required this.text,
    required this.textSoft,
    required this.textFaint,
    required this.accent,
    required this.accentInk,
    required this.accentWeak,
    required this.good,
    required this.goodWeak,
    required this.warn,
    required this.warnWeak,
    required this.bad,
    required this.badWeak,
  });

  static const light = AppColors(
    bg: Color(0xFFF7F8FA),
    surface: Color(0xFFFFFFFF),
    surface2: Color(0xFFF1F3F7),
    border: Color(0xFFE3E7EE),
    text: Color(0xFF1A2233),
    textSoft: Color(0xFF5B6577),
    textFaint: Color(0xFF8A93A6),
    accent: Color(0xFF3B6EF2),
    accentInk: Color(0xFFFFFFFF),
    accentWeak: Color(0xFFEAF0FE),
    good: Color(0xFF128A5B),
    goodWeak: Color(0xFFE2F4EC),
    warn: Color(0xFFB7791F),
    warnWeak: Color(0xFFFBF0DC),
    bad: Color(0xFFD1453B),
    badWeak: Color(0xFFFCECEB),
  );

  static const dark = AppColors(
    bg: Color(0xFF0F1420),
    surface: Color(0xFF161C2B),
    surface2: Color(0xFF1E2536),
    border: Color(0xFF2A3346),
    text: Color(0xFFE8ECF4),
    textSoft: Color(0xFFA9B2C5),
    textFaint: Color(0xFF7B8499),
    accent: Color(0xFF5B8BFF),
    accentInk: Color(0xFF0F1420),
    accentWeak: Color(0xFF1B2740),
    good: Color(0xFF3ECF8E),
    goodWeak: Color(0xFF14301F),
    warn: Color(0xFFE0A458),
    warnWeak: Color(0xFF33260F),
    bad: Color(0xFFF2635A),
    badWeak: Color(0xFF331B1A),
  );

  @override
  AppColors copyWith({
    Color? bg,
    Color? surface,
    Color? surface2,
    Color? border,
    Color? text,
    Color? textSoft,
    Color? textFaint,
    Color? accent,
    Color? accentInk,
    Color? accentWeak,
    Color? good,
    Color? goodWeak,
    Color? warn,
    Color? warnWeak,
    Color? bad,
    Color? badWeak,
  }) =>
      AppColors(
        bg: bg ?? this.bg,
        surface: surface ?? this.surface,
        surface2: surface2 ?? this.surface2,
        border: border ?? this.border,
        text: text ?? this.text,
        textSoft: textSoft ?? this.textSoft,
        textFaint: textFaint ?? this.textFaint,
        accent: accent ?? this.accent,
        accentInk: accentInk ?? this.accentInk,
        accentWeak: accentWeak ?? this.accentWeak,
        good: good ?? this.good,
        goodWeak: goodWeak ?? this.goodWeak,
        warn: warn ?? this.warn,
        warnWeak: warnWeak ?? this.warnWeak,
        bad: bad ?? this.bad,
        badWeak: badWeak ?? this.badWeak,
      );

  @override
  AppColors lerp(ThemeExtension<AppColors>? other, double t) {
    if (other is! AppColors) return this;
    Color m(Color a, Color b) => Color.lerp(a, b, t)!;
    return AppColors(
      bg: m(bg, other.bg),
      surface: m(surface, other.surface),
      surface2: m(surface2, other.surface2),
      border: m(border, other.border),
      text: m(text, other.text),
      textSoft: m(textSoft, other.textSoft),
      textFaint: m(textFaint, other.textFaint),
      accent: m(accent, other.accent),
      accentInk: m(accentInk, other.accentInk),
      accentWeak: m(accentWeak, other.accentWeak),
      good: m(good, other.good),
      goodWeak: m(goodWeak, other.goodWeak),
      warn: m(warn, other.warn),
      warnWeak: m(warnWeak, other.warnWeak),
      bad: m(bad, other.bad),
      badWeak: m(badWeak, other.badWeak),
    );
  }
}

extension AppColorsX on BuildContext {
  AppColors get c => Theme.of(this).extension<AppColors>()!;
}

// ---------------------------------------------------------------- ThemeData
ThemeData buildAppTheme(Brightness brightness) {
  final isDark = brightness == Brightness.dark;
  final ac = isDark ? AppColors.dark : AppColors.light;
  final base = ThemeData(brightness: brightness, useMaterial3: true);

  return base.copyWith(
    scaffoldBackgroundColor: ac.bg,
    extensions: [ac],
    colorScheme: base.colorScheme.copyWith(
      primary: ac.accent,
      onPrimary: ac.accentInk,
      surface: ac.surface,
      onSurface: ac.text,
      error: ac.bad,
    ),
    appBarTheme: AppBarTheme(
      backgroundColor: ac.surface,
      foregroundColor: ac.text,
      elevation: 0,
      scrolledUnderElevation: 0.5,
      centerTitle: false,
      titleTextStyle: TextStyle(
        color: ac.text,
        fontSize: Ty.lg,
        fontWeight: FontWeight.w700,
        letterSpacing: -0.2,
      ),
      shape: Border(bottom: BorderSide(color: ac.border)),
    ),
    textTheme: base.textTheme.apply(bodyColor: ac.text, displayColor: ac.text),
    dividerColor: ac.border,
    splashFactory: NoSplash.splashFactory,
  );
}
