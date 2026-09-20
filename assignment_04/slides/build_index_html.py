#!/usr/bin/env python3
"""
HTML Presentation Generator for CNN Presentation Deck (45 Slides)
Converts extracted_slides_clean.json into a modern, responsive, 16:9 interactive presentation deck.
"""

import json
import html
import re

with open('/Users/abc/Documents/study/intelligent-systems/intelligent_system_assignments/assignment_04/slides/extracted_slides_clean.json') as f:
    slides = json.load(f)

def highlight_python(code_str):
    """Simple syntax highlighter for Python code."""
    lines = code_str.splitlines()
    highlighted_lines = []
    
    keywords = {'def', 'class', 'import', 'from', 'as', 'return', 'super', 'if', 'else', 'for', 'in', 'while', 'True', 'False', 'None'}
    builtins = {'print', 'len', 'range', 'int', 'float', 'str', 'list', 'dict', 'set', 'tuple'}
    
    for line_idx, line in enumerate(lines, 1):
        # Escape HTML first
        esc_line = html.escape(line)
        
        # Highlight comments
        if '#' in esc_line:
            c_pos = esc_line.find('#')
            before_c = esc_line[:c_pos]
            after_c = esc_line[c_pos:]
            # Only if before_c is not inside a quote
            hl_comment = f'<span class="tok-comment">{after_c}</span>'
            esc_line = before_c + hl_comment
        
        # Highlight strings in the part before comment
        # Highlight def func_name / class ClassName
        def repl_func(m):
            return f'<span class="tok-kw">{m.group(1)}</span> <span class="tok-fn">{m.group(2)}</span>'
        esc_line = re.sub(r'\b(def|class)\s+([a-zA-Z_0-9]+)\b', repl_func, esc_line)
        
        # Highlight standalone keywords
        for kw in keywords:
            esc_line = re.sub(rf'\b({kw})\b', r'<span class="tok-kw">\1</span>', esc_line)
            
        # Highlight numbers
        esc_line = re.sub(r'(?<![a-zA-Z_])(\b\d+(\.\d+)?\b)', r'<span class="tok-num">\1</span>', esc_line)
        
        line_num = f'<span class="line-num">{line_idx:2d}</span>'
        highlighted_lines.append(f'<div class="code-line">{line_num}<span class="line-code">{esc_line}</span></div>')
        
    return '\n'.join(highlighted_lines)

def get_pillar_badge(slide):
    num = slide['num']
    if num in [1, 2]:
        return ('badge-agenda', 'Agenda & Overview')
    elif 3 <= num <= 6:
        return ('badge-pillar-1', 'Pillar 01: Overview & Scope')
    elif 7 <= num <= 16:
        return ('badge-pillar-2', 'Pillar 02: Concept & Mechanics')
    elif 17 <= num <= 33:
        return ('badge-pillar-3', 'Pillar 03: Model Evolution')
    elif 34 <= num <= 39:
        return ('badge-pillar-4', 'Pillar 04: Vectorized Implementation')
    else:
        return ('badge-pillar-4', 'Pillar 04: Empirical Applications')

html_parts = []

html_parts.append('''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
  <title>Convolutional Neural Networks (CNN) — Foundations, Architectural Evolution & Implementation</title>
  
  <!-- Modern Google Fonts -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Fira+Code:wght@400;500;600&display=swap" rel="stylesheet">
  
  <style>
    :root {
      --bg-stage: #F8FAFC;
      --card-bg: #FFFFFF;
      --border-color: #E2E8F0;
      --border-subtle: #EDF2F7;
      --text-main: #0F172A;
      --text-body: #334155;
      --text-muted: #64748B;
      
      --primary: #4338CA;
      --primary-light: #EEF2FF;
      --tech-blue: #0284C7;
      --blue-light: #F0F9FF;
      --emerald: #059669;
      --emerald-light: #ECFDF5;
      --amber: #D97706;
      --amber-light: #FFFBEB;
      --violet: #7C3AED;
      --violet-light: #F5F3FF;
      --coral: #EA580C;
      --coral-light: #FFF7ED;
      --crimson: #E11D48;
      
      --term-bg: #0F172A;
      --term-header: #1E293B;
      --term-border: #334155;
      --term-text: #E2E8F0;

      --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
      --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.07), 0 2px 4px -1px rgba(0, 0, 0, 0.04);
      --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.08), 0 4px 6px -2px rgba(0, 0, 0, 0.03);
      --shadow-card: 0 2px 8px rgba(0,0,0,0.04);
      --shadow-hero: 0 20px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.03);
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    body {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: radial-gradient(circle at 50% 20%, #1E293B 0%, #0B0F19 100%);
      color: var(--text-main);
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      min-height: 100vh;
      overflow: hidden;
      user-select: none;
      -webkit-font-smoothing: antialiased;
    }

    /* Top Progress Bar */
    #progress-container {
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 4px;
      background: rgba(255, 255, 255, 0.1);
      z-index: 1000;
    }
    #progress-bar {
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, var(--primary), var(--tech-blue), var(--violet), var(--emerald));
      transition: width 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* Deck Viewport (16:9 aspect ratio standard) */
    #deck-viewport {
      position: relative;
      width: 96vw;
      max-width: 1360px;
      height: 54vw;
      max-height: 765px;
      background-color: var(--bg-stage);
      border-radius: 12px;
      box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5), 0 0 0 1px rgba(255, 255, 255, 0.1);
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }

    /* Slides Container */
    .slide {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      display: flex;
      flex-direction: column;
      opacity: 0;
      visibility: hidden;
      transition: opacity 0.3s cubic-bezier(0.4, 0, 0.2, 1), transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
      transform: scale(0.985);
      background-color: var(--bg-stage);
      z-index: 1;
    }

    .slide.active {
      opacity: 1;
      visibility: visible;
      transform: scale(1);
      z-index: 10;
    }

    /* Slide Header */
    .slide-header {
      padding: 18px 36px 12px 36px;
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      border-bottom: 1px solid var(--border-color);
      background: #FFFFFF;
      flex-shrink: 0;
    }
    .slide-header-left {
      display: flex;
      flex-direction: column;
      gap: 3px;
    }
    .slide-title {
      font-size: 1.45rem;
      font-weight: 700;
      color: var(--text-main);
      letter-spacing: -0.025em;
      line-height: 1.25;
    }
    .slide-subtitle {
      font-size: 0.92rem;
      font-weight: 500;
      color: var(--text-muted);
    }
    .slide-header-right {
      display: flex;
      align-items: center;
      gap: 12px;
      flex-shrink: 0;
    }

    /* Pillar Badges */
    .pillar-badge {
      font-size: 0.75rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      padding: 5px 12px;
      border-radius: 20px;
    }
    .badge-agenda { background: #F1F5F9; color: #475569; border: 1px solid #CBD5E1; }
    .badge-pillar-1 { background: var(--primary-light); color: var(--primary); border: 1px solid #C7D2FE; }
    .badge-pillar-2 { background: var(--blue-light); color: var(--tech-blue); border: 1px solid #BAE6FD; }
    .badge-pillar-3 { background: var(--violet-light); color: var(--violet); border: 1px solid #DDD6FE; }
    .badge-pillar-4 { background: var(--emerald-light); color: var(--emerald); border: 1px solid #A7F3D0; }

    /* Slide Body */
    .slide-body {
      flex: 1;
      padding: 20px 36px;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }

    /* Slide Footer */
    .slide-footer {
      height: 36px;
      padding: 0 36px;
      border-top: 1px solid var(--border-color);
      display: flex;
      align-items: center;
      justify-content: space-between;
      font-size: 0.78rem;
      color: var(--text-muted);
      background-color: #FFFFFF;
      flex-shrink: 0;
    }
    .footer-course {
      font-weight: 500;
    }
    .footer-num {
      font-weight: 600;
      color: var(--text-main);
      font-family: var(--font-mono);
    }

    /* Layout Containers */
    .split-layout {
      display: grid;
      height: 100%;
      gap: 22px;
      align-items: stretch;
    }
    .split-left {
      grid-template-columns: 58% 42%;
    }
    .split-right {
      grid-template-columns: 42% 58%;
    }

    /* Bento Cards */
    .bento-card {
      background: var(--card-bg);
      border: 1px solid var(--border-color);
      border-radius: 10px;
      padding: 22px 24px;
      box-shadow: var(--shadow-card);
      display: flex;
      flex-direction: column;
      justify-content: center;
      overflow: hidden;
    }
    .bento-card-title {
      font-size: 1.15rem;
      font-weight: 700;
      color: var(--tech-blue);
      margin-bottom: 16px;
      padding-bottom: 8px;
      border-bottom: 1.5px solid var(--border-subtle);
    }

    /* Bullets List */
    .bullet-list {
      list-style: none;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }
    .bullet-item {
      display: flex;
      align-items: flex-start;
      gap: 10px;
      font-size: 0.93rem;
      line-height: 1.48;
      color: var(--text-body);
    }
    .bullet-dot {
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: var(--tech-blue);
      margin-top: 8px;
      flex-shrink: 0;
    }
    .bullet-lead {
      font-weight: 700;
      color: var(--text-main);
    }

    /* Image Panels */
    .image-panel {
      background: #FFFFFF;
      border: 1px solid var(--border-color);
      border-radius: 10px;
      padding: 12px;
      box-shadow: var(--shadow-card);
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      overflow: hidden;
    }
    .image-panel img {
      max-width: 100%;
      max-height: 380px;
      object-fit: contain;
      border-radius: 6px;
    }
    .image-caption {
      margin-top: 8px;
      font-size: 0.78rem;
      font-weight: 600;
      color: var(--text-muted);
      text-align: center;
      background: #F8FAFC;
      padding: 4px 10px;
      border-radius: 4px;
      border: 1px solid var(--border-color);
      width: 100%;
    }

    /* Split Dual Images */
    .dual-images-stack {
      display: flex;
      flex-direction: column;
      gap: 12px;
      height: 100%;
    }
    .dual-images-stack .image-panel {
      flex: 1;
      padding: 8px;
    }
    .dual-images-stack .image-panel img {
      max-height: 165px;
    }

    /* Slide 1: Hero Card Title */
    .hero-container {
      background: #FFFFFF;
      border: 1.5px solid var(--border-color);
      border-radius: 12px;
      height: 100%;
      padding: 24px 36px;
      box-shadow: var(--shadow-hero);
      display: flex;
      flex-direction: column;
      justify-content: space-between;
    }
    .hero-tag {
      font-size: 0.85rem;
      font-weight: 800;
      letter-spacing: 0.08em;
      color: var(--primary);
      text-transform: uppercase;
    }
    .hero-title {
      font-size: 2.2rem;
      font-weight: 800;
      color: var(--text-main);
      letter-spacing: -0.03em;
      line-height: 1.2;
      margin-top: 4px;
    }
    .hero-subtitle {
      font-size: 1.1rem;
      font-weight: 500;
      color: var(--text-muted);
      margin-top: 4px;
    }
    .hero-image-box {
      margin: 12px 0;
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 10px;
      background: #FAFAFA;
      display: flex;
      flex-direction: column;
      align-items: center;
      box-shadow: inset 0 2px 4px rgba(0,0,0,0.02);
    }
    .hero-image-box img {
      max-height: 180px;
      object-fit: contain;
    }
    .hero-meta {
      border-top: 1px solid var(--border-color);
      padding-top: 12px;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .hero-authors {
      font-size: 0.92rem;
      font-weight: 700;
      color: var(--text-main);
    }
    .hero-course {
      font-size: 0.82rem;
      color: var(--text-muted);
    }

    /* Slide 2: Agenda Bento Grid */
    .agenda-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      grid-template-rows: 1fr 1fr;
      gap: 16px;
      height: 100%;
    }
    .agenda-card {
      background: #FFFFFF;
      border: 1px solid var(--border-color);
      border-radius: 10px;
      padding: 14px 18px;
      box-shadow: var(--shadow-card);
      display: grid;
      grid-template-columns: 1.15fr 1fr;
      gap: 14px;
      align-items: center;
    }
    .agenda-badge {
      font-size: 0.72rem;
      font-weight: 800;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      margin-bottom: 4px;
    }
    .agenda-title {
      font-size: 1.05rem;
      font-weight: 700;
      color: var(--text-main);
      line-height: 1.25;
      margin-bottom: 2px;
    }
    .agenda-domain {
      font-size: 0.78rem;
      font-style: italic;
      color: var(--text-muted);
      margin-bottom: 6px;
    }
    .agenda-desc {
      font-size: 0.8rem;
      color: var(--text-body);
      line-height: 1.35;
    }
    .agenda-preview-img {
      width: 100%;
      height: 100%;
      max-height: 110px;
      object-fit: contain;
      border-radius: 6px;
      border: 1px solid var(--border-color);
      background: #F8FAFC;
      padding: 4px;
    }

    /* Triptych Layout (Slide 34, Slide 40) */
    .triptych-container {
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 18px;
      height: 100%;
    }
    .triptych-container {
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 22px;
      height: 100%;
    }
    .triptych-card {
      background: #FFFFFF;
      border: 1px solid var(--border-color);
      border-radius: 10px;
      padding: 24px 24px;
      box-shadow: var(--shadow-card);
      display: flex;
      flex-direction: column;
      justify-content: flex-start;
      overflow: hidden;
    }
    .triptych-card.has-img {
      padding: 16px 20px 14px 20px;
      justify-content: space-between;
    }
    .triptych-badge {
      font-size: 0.82rem;
      font-weight: 800;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      margin-bottom: 6px;
    }
    .triptych-title {
      font-size: 1.28rem;
      font-weight: 700;
      color: var(--text-main);
      margin-bottom: 18px;
      padding-bottom: 10px;
      border-bottom: 1.5px solid var(--border-subtle);
      line-height: 1.25;
    }
    .triptych-card.has-img .triptych-title {
      font-size: 1.15rem;
      margin-bottom: 12px;
      padding-bottom: 6px;
    }
    .triptych-bullet-list {
      list-style: none;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    .triptych-card.has-img .triptych-bullet-list {
      gap: 10px;
    }
    .triptych-bullet-item {
      display: flex;
      align-items: flex-start;
      gap: 10px;
      font-size: 0.95rem;
      line-height: 1.52;
      color: var(--text-body);
    }
    .triptych-card.has-img .triptych-bullet-item {
      font-size: 0.88rem;
      line-height: 1.4;
    }
    .triptych-bullet-lead {
      font-weight: 700;
      color: var(--text-main);
    }
    .triptych-img-box {
      margin-top: 12px;
      border: 1px solid var(--border-color);
      border-radius: 8px;
      overflow: hidden;
      background: #F8FAFC;
      display: flex;
      align-items: center;
      justify-content: center;
      height: 180px;
      box-shadow: inset 0 1px 3px rgba(0,0,0,0.02);
      flex-shrink: 0;
    }
    .triptych-img {
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
      border-radius: 4px;
      transition: transform 0.25s ease;
    }
    .triptych-badge {
      font-size: 0.75rem;
      font-weight: 800;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      margin-bottom: 4px;
    }
    .triptych-title {
      font-size: 1.15rem;
      font-weight: 700;
      color: var(--text-main);
      margin-bottom: 14px;
      padding-bottom: 8px;
      border-bottom: 1.5px solid var(--border-subtle);
    }

    /* Tables (Slide 35, Slide 43) */
    .table-card {
      background: #FFFFFF;
      border: 1px solid var(--border-color);
      border-radius: 10px;
      padding: 16px;
      box-shadow: var(--shadow-card);
      height: 100%;
      display: flex;
      flex-direction: column;
      overflow: auto;
    }
    .rosetta-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.84rem;
      height: 100%;
    }
    .rosetta-table th {
      background: #F1F5F9;
      color: var(--text-main);
      font-weight: 700;
      padding: 12px 16px;
      border-bottom: 2px solid var(--border-color);
      text-align: left;
    }
    .rosetta-table td {
      padding: 12px 16px;
      border-bottom: 1px solid var(--border-color);
      color: var(--text-body);
      vertical-align: middle;
      font-family: var(--font-mono);
      font-size: 0.82rem;
      white-space: pre-line;
      line-height: 1.35;
    }
    .rosetta-table td:first-child {
      font-family: var(--font-sans);
      font-weight: 600;
      color: var(--text-main);
      font-size: 0.88rem;
    }
    .rosetta-table tr:nth-child(even) td {
      background-color: #F8FAFC;
    }

    /* Terminal Code Blocks */
    .terminal-window {
      background: var(--term-bg);
      border-radius: 10px;
      overflow: hidden;
      box-shadow: var(--shadow-md), 0 0 0 1px rgba(255, 255, 255, 0.05);
      display: flex;
      flex-direction: column;
      height: 100%;
    }
    .terminal-titlebar {
      background: var(--term-header);
      padding: 10px 16px;
      display: flex;
      align-items: center;
      gap: 12px;
      border-bottom: 1px solid var(--term-border);
      flex-shrink: 0;
    }
    .terminal-dots {
      display: flex;
      gap: 6px;
    }
    .dot {
      width: 11px;
      height: 11px;
      border-radius: 50%;
    }
    .dot-red { background: #EF4444; }
    .dot-yellow { background: #F59E0B; }
    .dot-green { background: #10B981; }
    .terminal-filename {
      font-family: var(--font-mono);
      font-size: 0.8rem;
      color: #94A3B8;
      font-weight: 500;
    }
    .terminal-code-body {
      padding: 14px 16px;
      overflow: auto;
      flex: 1;
      font-family: var(--font-mono);
      font-size: 0.8rem;
      line-height: 1.45;
      color: var(--term-text);
    }
    .code-line {
      display: flex;
      gap: 14px;
    }
    .line-num {
      color: #475569;
      user-select: none;
      width: 20px;
      text-align: right;
      flex-shrink: 0;
    }
    .line-code {
      white-space: pre;
    }

    /* Syntax Highlight Tokens */
    .tok-kw { color: #C084FC; font-weight: 600; }
    .tok-fn { color: #67E8F9; }
    .tok-comment { color: #64748B; font-style: italic; }
    .tok-num { color: #FBBF24; }

    /* Floating Navigation Controls */
    #nav-controls {
      position: fixed;
      bottom: 24px;
      display: flex;
      align-items: center;
      gap: 12px;
      background: rgba(15, 23, 42, 0.88);
      backdrop-filter: blur(14px);
      -webkit-backdrop-filter: blur(14px);
      padding: 8px 18px;
      border-radius: 40px;
      border: 1px solid rgba(255, 255, 255, 0.15);
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.45);
      z-index: 1000;
      transition: opacity 0.4s cubic-bezier(0.4, 0, 0.2, 1), transform 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    #nav-controls.autohide {
      opacity: 0;
      transform: translateY(28px);
      pointer-events: none;
    }
    #nav-controls:hover {
      opacity: 1 !important;
      transform: translateY(0) !important;
      pointer-events: auto !important;
    }
    .nav-btn {
      background: rgba(255, 255, 255, 0.1);
      border: 1px solid rgba(255, 255, 255, 0.15);
      color: #FFFFFF;
      text-decoration: none;
      padding: 7px 14px;
      border-radius: 20px;
      cursor: pointer;
      font-size: 0.82rem;
      font-weight: 600;
      transition: all 0.2s ease;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      white-space: nowrap;
    }
    .nav-btn:hover {
      background: rgba(255, 255, 255, 0.25);
      color: #FFFFFF;
      transform: translateY(-1px);
    }
    .nav-btn-download {
      background: linear-gradient(135deg, #4338CA 0%, #0284C7 100%);
      border: 1px solid rgba(255, 255, 255, 0.35);
      box-shadow: 0 2px 8px rgba(2, 132, 199, 0.3);
    }
    .nav-btn-download:hover {
      background: linear-gradient(135deg, #3730A3 0%, #0369A1 100%);
      box-shadow: 0 4px 14px rgba(2, 132, 199, 0.5);
    }
    .nav-btn:active {
      transform: translateY(0);
    }
    #slide-select {
      background: rgba(255, 255, 255, 0.1);
      border: 1px solid rgba(255, 255, 255, 0.2);
      color: #FFFFFF;
      padding: 6px 12px;
      border-radius: 18px;
      font-size: 0.8rem;
      font-weight: 500;
      outline: none;
      cursor: pointer;
      max-width: 260px;
    }
    #slide-select option {
      background: #1E293B;
      color: #FFFFFF;
    }
    .nav-counter {
      color: #94A3B8;
      font-size: 0.82rem;
      font-family: var(--font-mono);
      font-weight: 600;
      padding: 0 6px;
    }

    /* Grid Overview Modal */
    #overview-modal {
      position: fixed;
      top: 0;
      left: 0;
      width: 100vw;
      height: 100vh;
      background: rgba(11, 15, 25, 0.95);
      backdrop-filter: blur(14px);
      -webkit-backdrop-filter: blur(14px);
      z-index: 2000;
      opacity: 0;
      visibility: hidden;
      transition: all 0.3s ease;
      display: flex;
      flex-direction: column;
      padding: 30px;
    }
    #overview-modal.active {
      opacity: 1;
      visibility: visible;
    }
    .overview-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      color: white;
      margin-bottom: 24px;
      padding-bottom: 16px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    .overview-header h2 {
      font-size: 1.5rem;
      font-weight: 700;
    }
    .close-btn {
      background: rgba(255, 255, 255, 0.1);
      border: none;
      color: white;
      font-size: 1.2rem;
      width: 36px;
      height: 36px;
      border-radius: 50%;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .overview-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
      gap: 16px;
      overflow-y: auto;
      padding-right: 8px;
    }
    .overview-card {
      background: rgba(30, 41, 59, 0.7);
      border: 1.5px solid rgba(255, 255, 255, 0.1);
      border-radius: 8px;
      padding: 14px;
      cursor: pointer;
      transition: all 0.2s ease;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .overview-card:hover {
      border-color: var(--tech-blue);
      transform: translateY(-2px);
      background: rgba(30, 41, 59, 0.95);
    }
    .overview-card.current {
      border-color: var(--primary);
      box-shadow: 0 0 15px rgba(67, 56, 202, 0.5);
      background: rgba(67, 56, 202, 0.2);
    }
    .overview-card-num {
      font-size: 0.75rem;
      font-weight: 700;
      color: var(--tech-blue);
      font-family: var(--font-mono);
    }
    .overview-card-title {
      color: white;
      font-size: 0.88rem;
      font-weight: 600;
      line-height: 1.3;
    }
    .overview-card-tag {
      font-size: 0.7rem;
      color: #94A3B8;
    }

    /* Keybind Hint Overlay */
    .key-hint {
      display: inline-block;
      padding: 2px 6px;
      background: rgba(255, 255, 255, 0.15);
      border-radius: 4px;
      font-size: 0.7rem;
      font-family: var(--font-mono);
      margin-left: 4px;
    }

    /* Print / PDF Export Styles */
    @media print {
      @page {
        size: 16in 9in; /* 16:9 presentation aspect ratio */
        margin: 0;
      }

      html, body {
        background: #FFFFFF !important;
        color: #0F172A !important;
        overflow: visible !important;
        height: auto !important;
        min-height: auto !important;
        display: block !important;
        -webkit-print-color-adjust: exact !important;
        print-color-adjust: exact !important;
      }

      #progress-container,
      #nav-controls,
      #overview-modal {
        display: none !important;
      }

      #deck-viewport {
        width: 100% !important;
        max-width: 100% !important;
        height: auto !important;
        max-height: none !important;
        box-shadow: none !important;
        border-radius: 0 !important;
        background: transparent !important;
        overflow: visible !important;
        display: block !important;
      }

      .slide {
        position: relative !important;
        top: auto !important;
        left: auto !important;
        width: 100vw !important;
        height: 100vh !important;
        opacity: 1 !important;
        visibility: visible !important;
        transform: none !important;
        page-break-after: always !important;
        break-after: page !important;
        display: flex !important;
        flex-direction: column !important;
        background: #F8FAFC !important;
        box-sizing: border-box !important;
        overflow: hidden !important;
      }

      .slide-header, .slide-footer {
        background: #FFFFFF !important;
        -webkit-print-color-adjust: exact !important;
        print-color-adjust: exact !important;
      }

      .bento-card, .image-panel, .triptych-card, .table-card, .hero-container, .agenda-card {
        box-shadow: none !important;
        border: 1px solid #E2E8F0 !important;
        -webkit-print-color-adjust: exact !important;
        print-color-adjust: exact !important;
      }

      .terminal-window {
        background: #0F172A !important;
        -webkit-print-color-adjust: exact !important;
        print-color-adjust: exact !important;
      }
    }

    /* Slide 45: Executive Closing Synthesis */
    .closing-synthesis-container {
      display: flex;
      flex-direction: column;
      gap: 14px;
      height: 100%;
    }
    .closing-pillars-grid {
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 16px;
      flex: 1;
    }
    .closing-pillar-card {
      background: #FFFFFF;
      border: 1px solid var(--border-color);
      border-radius: 10px;
      padding: 16px 18px;
      box-shadow: var(--shadow-card);
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      overflow: hidden;
    }
    .pillar-card-indigo { border-top: 3.5px solid var(--primary); }
    .pillar-card-coral { border-top: 3.5px solid var(--coral); }
    .pillar-card-emerald { border-top: 3.5px solid var(--emerald); }

    .closing-pillar-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 6px;
    }
    .closing-pillar-badge {
      font-size: 0.72rem;
      font-weight: 800;
      letter-spacing: 0.06em;
      padding: 3px 8px;
      border-radius: 12px;
    }
    .badge-indigo { background: var(--primary-light); color: var(--primary); }
    .badge-coral { background: var(--coral-light); color: var(--coral); }
    .badge-emerald { background: var(--emerald-light); color: var(--emerald); }

    .closing-pillar-stat {
      font-family: var(--font-mono);
      font-size: 0.76rem;
      font-weight: 700;
      color: var(--text-muted);
    }
    .closing-pillar-title {
      font-size: 1.08rem;
      font-weight: 700;
      color: var(--text-main);
      margin-bottom: 6px;
    }
    .closing-pillar-desc {
      font-size: 0.82rem;
      color: var(--text-body);
      line-height: 1.4;
      margin-bottom: 8px;
    }
    .closing-pillar-metrics {
      display: flex;
      flex-direction: column;
      gap: 4px;
      margin-bottom: 8px;
    }
    .metric-chip {
      display: flex;
      justify-content: space-between;
      background: #F8FAFC;
      border: 1px solid #E2E8F0;
      padding: 4px 8px;
      border-radius: 6px;
      font-size: 0.78rem;
    }
    .chip-name {
      font-weight: 600;
      color: var(--text-main);
    }
    .chip-val {
      font-family: var(--font-mono);
      font-weight: 700;
      color: var(--primary);
    }
    .closing-bullet-list {
      list-style: none;
      font-size: 0.8rem;
      color: var(--text-body);
      line-height: 1.4;
      display: flex;
      flex-direction: column;
      gap: 6px;
      margin-bottom: 8px;
    }
    .spectrum-rows {
      display: flex;
      flex-direction: column;
      gap: 6px;
      margin-bottom: 8px;
    }
    .spectrum-row {
      display: flex;
      flex-direction: column;
      gap: 2px;
      font-size: 0.78rem;
    }
    .framework-tag {
      font-size: 0.7rem;
      font-weight: 700;
      padding: 1px 6px;
      border-radius: 4px;
      display: inline-block;
      width: fit-content;
    }
    .tag-scratch { background: rgba(199, 124, 72, 0.15); color: #C77C48; }
    .tag-keras { background: rgba(76, 114, 176, 0.15); color: #4C72B0; }
    .tag-torch { background: rgba(85, 168, 104, 0.15); color: #059669; }

    .closing-pillar-footer {
      font-size: 0.75rem;
      font-style: italic;
      color: var(--text-muted);
      border-top: 1px dashed #E2E8F0;
      padding-top: 6px;
    }

    .closing-banner {
      background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
      color: #FFFFFF;
      border-radius: 10px;
      padding: 12px 20px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border: 1px solid #334155;
      box-shadow: 0 4px 12px rgba(0,0,0,0.15);
      flex-shrink: 0;
    }
    .banner-tag {
      font-size: 0.68rem;
      font-weight: 800;
      letter-spacing: 0.08em;
      color: #38BDF8;
      text-transform: uppercase;
    }
    .banner-axiom {
      font-size: 0.95rem;
      font-weight: 700;
      color: #FFFFFF;
      margin-top: 2px;
    }
    .banner-subtext {
      font-size: 0.76rem;
      color: #94A3B8;
      margin-top: 2px;
    }
    .qa-badge {
      display: flex;
      align-items: center;
      gap: 10px;
      background: rgba(255, 255, 255, 0.08);
      border: 1px solid rgba(255, 255, 255, 0.18);
      padding: 8px 14px;
      border-radius: 8px;
      flex-shrink: 0;
    }
    .qa-icon { font-size: 1.4rem; }
    .qa-title { font-size: 0.88rem; font-weight: 700; color: #FFFFFF; }
    .qa-sub { font-size: 0.72rem; color: #38BDF8; }
  </style>
</head>
<body>

  <!-- Top Progress Bar -->
  <div id="progress-container">
    <div id="progress-bar"></div>
  </div>

  <!-- Deck Viewport -->
  <main id="deck-viewport">
''')

for s in slides:
    num = s['num']
    active_cls = ' active' if num == 1 else ''
    layout = s.get('layout')
    badge_cls, badge_text = get_pillar_badge(s)
    title = html.escape(s.get('title', ''))
    subtitle = html.escape(s.get('subtitle', ''))
    card_title = html.escape(s.get('card_title', 'Core Insights'))

    html_parts.append(f'    <!-- Slide {num:02d} -->')
    html_parts.append(f'    <section class="slide{active_cls}" id="slide-{num}" data-slide="{num}">')
    
    # 1. Slide 1: Hero Card Title Slide
    if layout == 'hero_title':
        tag = html.escape(s.get('tag', ''))
        authors_html = ' &nbsp;|&nbsp; '.join(html.escape(a) for a in s.get('authors', []))
        meta = html.escape(s.get('metadata', ''))
        caption = html.escape(s.get('caption', 'A Typical End-to-End CNN Pipeline'))
        img_src = s.get('image', 'assets/typical_cnn.png')
        
        html_parts.append(f'''      <div class="slide-body" style="padding: 28px;">
        <div class="hero-container">
          <div>
            <div class="hero-tag">{tag}</div>
            <h1 class="hero-title">{title}</h1>
            <p class="hero-subtitle">{subtitle}</p>
          </div>
          <div class="hero-image-box">
            <img src="{img_src}" alt="{caption}">
            <div class="image-caption">{caption}</div>
          </div>
          <div class="hero-meta">
            <div class="hero-authors">Authors: {authors_html}</div>
            <div class="hero-course">{meta}</div>
          </div>
        </div>
      </div>
      <footer class="slide-footer">
        <span class="footer-course">Post & Telecommunications Institute of Technology (PTIT)</span>
        <span class="footer-num">{num} / 45</span>
      </footer>''')

    # 2. Slide 2: Agenda 2x2 Bento Grid
    elif layout == 'agenda_grid':
        html_parts.append(f'''      <header class="slide-header">
        <div class="slide-header-left">
          <h2 class="slide-title">{title}</h2>
          <p class="slide-subtitle">{subtitle}</p>
        </div>
        <div class="slide-header-right">
          <span class="pillar-badge {badge_cls}">{badge_text}</span>
        </div>
      </header>
      <div class="slide-body">
        <div class="agenda-grid">''')
        for p in s.get('pillars', []):
            b_badge = html.escape(p['badge'])
            b_title = html.escape(p['title'])
            b_dom = html.escape(p['domain'])
            b_desc = html.escape(p['desc'])
            b_img = p['image']
            b_col = p['color']
            html_parts.append(f'''          <div class="agenda-card">
            <div>
              <div class="agenda-badge" style="color: {b_col};">{b_badge}</div>
              <div class="agenda-title">{b_title}</div>
              <div class="agenda-domain">{b_dom}</div>
              <div class="agenda-desc">{b_desc}</div>
            </div>
            <img class="agenda-preview-img" src="{b_img}" alt="{b_title}">
          </div>''')
        html_parts.append(f'''        </div>
      </div>
      <footer class="slide-footer">
        <span class="footer-course">Intelligent Systems · CNN Architecture</span>
        <span class="footer-num">{num} / 45</span>
      </footer>''')

    # 3. Split Left Image (58% Image / 42% Card)
    elif layout == 'split_left_image':
        img_src = s.get('image', '')
        caption = html.escape(s.get('caption', ''))
        bullets = s.get('bullets', [])
        
        html_parts.append(f'''      <header class="slide-header">
        <div class="slide-header-left">
          <h2 class="slide-title">{title}</h2>
          <p class="slide-subtitle">{subtitle}</p>
        </div>
        <div class="slide-header-right">
          <span class="pillar-badge {badge_cls}">{badge_text}</span>
        </div>
      </header>
      <div class="slide-body">
        <div class="split-layout split-left">
          <div class="image-panel">
            <img src="{img_src}" alt="{caption}">
            <div class="image-caption">{caption}</div>
          </div>
          <div class="bento-card">
            <div class="bento-card-title">{card_title}</div>
            <ul class="bullet-list">''')
        for lead, body in bullets:
            html_parts.append(f'''              <li class="bullet-item">
                <span class="bullet-dot"></span>
                <div><span class="bullet-lead">{html.escape(lead)}:</span> {html.escape(body)}</div>
              </li>''')
        html_parts.append(f'''            </ul>
          </div>
        </div>
      </div>
      <footer class="slide-footer">
        <span class="footer-course">Intelligent Systems · CNN Architecture</span>
        <span class="footer-num">{num} / 45</span>
      </footer>''')

    # 4. Split Right Image (42% Card / 58% Image)
    elif layout == 'split_right_image':
        img_src = s.get('image', '')
        caption = html.escape(s.get('caption', ''))
        bullets = s.get('bullets', [])
        
        html_parts.append(f'''      <header class="slide-header">
        <div class="slide-header-left">
          <h2 class="slide-title">{title}</h2>
          <p class="slide-subtitle">{subtitle}</p>
        </div>
        <div class="slide-header-right">
          <span class="pillar-badge {badge_cls}">{badge_text}</span>
        </div>
      </header>
      <div class="slide-body">
        <div class="split-layout split-right">
          <div class="bento-card">
            <div class="bento-card-title">{card_title}</div>
            <ul class="bullet-list">''')
        for lead, body in bullets:
            html_parts.append(f'''              <li class="bullet-item">
                <span class="bullet-dot"></span>
                <div><span class="bullet-lead">{html.escape(lead)}:</span> {html.escape(body)}</div>
              </li>''')
        html_parts.append(f'''            </ul>
          </div>
          <div class="image-panel">
            <img src="{img_src}" alt="{caption}">
            <div class="image-caption">{caption}</div>
          </div>
        </div>
      </div>
      <footer class="slide-footer">
        <span class="footer-course">Intelligent Systems · CNN Architecture</span>
        <span class="footer-num">{num} / 45</span>
      </footer>''')

    # 5. Split Dual Images (Left 2 Stacked Images / 42% Card)
    elif layout == 'split_dual_images':
        images = s.get('images', [])
        bullets = s.get('bullets', [])
        
        html_parts.append(f'''      <header class="slide-header">
        <div class="slide-header-left">
          <h2 class="slide-title">{title}</h2>
          <p class="slide-subtitle">{subtitle}</p>
        </div>
        <div class="slide-header-right">
          <span class="pillar-badge {badge_cls}">{badge_text}</span>
        </div>
      </header>
      <div class="slide-body">
        <div class="split-layout split-left">
          <div class="dual-images-stack">''')
        for im in images:
            im_path = im['path']
            im_cap = html.escape(im.get('caption', ''))
            html_parts.append(f'''            <div class="image-panel">
              <img src="{im_path}" alt="{im_cap}">
              <div class="image-caption">{im_cap}</div>
            </div>''')
        html_parts.append(f'''          </div>
          <div class="bento-card">
            <div class="bento-card-title">{card_title}</div>
            <ul class="bullet-list">''')
        for lead, body in bullets:
            html_parts.append(f'''              <li class="bullet-item">
                <span class="bullet-dot"></span>
                <div><span class="bullet-lead">{html.escape(lead)}:</span> {html.escape(body)}</div>
              </li>''')
        html_parts.append(f'''            </ul>
          </div>
        </div>
      </div>
      <footer class="slide-footer">
        <span class="footer-course">Intelligent Systems · CNN Architecture</span>
        <span class="footer-num">{num} / 45</span>
      </footer>''')

    # 6. Triptych Layout (Slide 34, Slide 40)
    elif layout == 'layout_triptych':
        cols = s.get('cols_data', [])
        
        html_parts.append(f'''      <header class="slide-header">
        <div class="slide-header-left">
          <h2 class="slide-title">{title}</h2>
          <p class="slide-subtitle">{subtitle}</p>
        </div>
        <div class="slide-header-right">
          <span class="pillar-badge {badge_cls}">{badge_text}</span>
        </div>
      </header>
      <div class="slide-body">
        <div class="triptych-container">''')
        for col in cols:
            c_badge = html.escape(col['badge'])
            c_title = html.escape(col['title'])
            c_col = col['color']
            c_bullets = col['bullets']
            c_img = col.get('img')
            has_img_cls = ' has-img' if c_img else ''
            
            img_html = ''
            if c_img:
                img_html = f'''            <div class="triptych-img-box">
              <img src="{c_img}" alt="{c_title} Dataset Samples" class="triptych-img">
            </div>'''
            
            html_parts.append(f'''          <div class="triptych-card{has_img_cls}">
            <div>
              <div class="triptych-badge" style="color: {c_col};">{c_badge}</div>
              <div class="triptych-title">{c_title}</div>
              <ul class="triptych-bullet-list">''')
            for lead, body in c_bullets:
                html_parts.append(f'''                <li class="triptych-bullet-item">
                  <span class="bullet-dot" style="background: {c_col};"></span>
                  <div><span class="triptych-bullet-lead">{html.escape(lead)}:</span> {html.escape(body)}</div>
                </li>''')
            html_parts.append(f'''              </ul>
            </div>
{img_html}
          </div>''')
        html_parts.append(f'''        </div>
      </div>
      <footer class="slide-footer">
        <span class="footer-course">Intelligent Systems · CNN Architecture</span>
        <span class="footer-num">{num} / 45</span>
      </footer>''')

    elif layout == 'rosetta_table':
        headers = s.get('headers', [])
        rows = s.get('rows', [])
        
        html_parts.append(f'''      <header class="slide-header">
        <div class="slide-header-left">
          <h2 class="slide-title">{title}</h2>
          <p class="slide-subtitle">{subtitle}</p>
        </div>
        <div class="slide-header-right">
          <span class="pillar-badge {badge_cls}">{badge_text}</span>
        </div>
      </header>
      <div class="slide-body">
        <div class="table-card">
          <table class="rosetta-table">
            <thead>
              <tr>''')
        for h_item in headers:
            html_parts.append(f'                <th>{html.escape(h_item)}</th>')
        html_parts.append('''              </tr>
            </thead>
            <tbody>''')
        for row in rows:
            html_parts.append('              <tr>')
            for cell in row:
                html_parts.append(f'                <td>{html.escape(cell)}</td>')
            html_parts.append('              </tr>')
        html_parts.append(f'''            </tbody>
          </table>
        </div>
      </div>
      <footer class="slide-footer">
        <span class="footer-course">Intelligent Systems · CNN Architecture</span>
        <span class="footer-num">{num} / 45</span>
      </footer>''')

    # 8. Terminal Split Layout (Slide 36, 37)
    elif layout == 'terminal_split':
        term = s.get('terminal', {})
        t_filename = html.escape(term.get('filename', 'script.py'))
        t_code = term.get('code', '')
        t_hl = highlight_python(t_code)
        
        img_src = s.get('image', '')
        caption = html.escape(s.get('caption', ''))
        
        html_parts.append(f'''      <header class="slide-header">
        <div class="slide-header-left">
          <h2 class="slide-title">{title}</h2>
          <p class="slide-subtitle">{subtitle}</p>
        </div>
        <div class="slide-header-right">
          <span class="pillar-badge {badge_cls}">{badge_text}</span>
        </div>
      </header>
      <div class="slide-body">
        <div class="split-layout split-left">
          <div class="terminal-window">
            <div class="terminal-titlebar">
              <div class="terminal-dots">
                <span class="dot dot-red"></span>
                <span class="dot dot-yellow"></span>
                <span class="dot dot-green"></span>
              </div>
              <span class="terminal-filename">{t_filename}</span>
            </div>
            <div class="terminal-code-body">
{t_hl}
            </div>
          </div>
          <div class="image-panel">
            <img src="{img_src}" alt="{caption}">
            <div class="image-caption">{caption}</div>
          </div>
        </div>
      </div>
      <footer class="slide-footer">
        <span class="footer-course">Intelligent Systems · CNN Architecture</span>
        <span class="footer-num">{num} / 45</span>
      </footer>''')

    # 9. Terminal Split Card Layout (Slide 38)
    elif layout == 'terminal_split_card':
        term = s.get('terminal', {})
        t_filename = html.escape(term.get('filename', 'modern_cnn_pytorch.py'))
        t_code = term.get('code', '')
        t_hl = highlight_python(t_code)
        bullets = s.get('bullets', [])
        
        html_parts.append(f'''      <header class="slide-header">
        <div class="slide-header-left">
          <h2 class="slide-title">{title}</h2>
          <p class="slide-subtitle">{subtitle}</p>
        </div>
        <div class="slide-header-right">
          <span class="pillar-badge {badge_cls}">{badge_text}</span>
        </div>
      </header>
      <div class="slide-body">
        <div class="split-layout" style="grid-template-columns: 62% 38%;">
          <div class="terminal-window">
            <div class="terminal-titlebar">
              <div class="terminal-dots">
                <span class="dot dot-red"></span>
                <span class="dot dot-yellow"></span>
                <span class="dot dot-green"></span>
              </div>
              <span class="terminal-filename">{t_filename}</span>
            </div>
            <div class="terminal-code-body">
{t_hl}
            </div>
          </div>
          <div class="bento-card">
            <div class="bento-card-title" style="color: var(--emerald);">{card_title}</div>
            <ul class="bullet-list">''')
        for lead, body in bullets:
            html_parts.append(f'''              <li class="bullet-item">
                <span class="bullet-dot" style="background: var(--emerald);"></span>
                <div><span class="bullet-lead">{html.escape(lead)}:</span> {html.escape(body)}</div>
              </li>''')
        html_parts.append(f'''            </ul>
          </div>
        </div>
      </div>
      <footer class="slide-footer">
        <span class="footer-course">Intelligent Systems · CNN Architecture</span>
        <span class="footer-num">{num} / 45</span>
      </footer>''')

    html_parts.append('    </section>\n')

html_parts.append('''  </main>

  <!-- Floating Navigation Controls -->
  <nav id="nav-controls">
    <button class="nav-btn" id="btn-prev" title="Previous Slide (← / PgUp)">
      <span>◂</span> Prev
    </button>
    <select id="slide-select" title="Jump directly to slide">
''')

for s in slides:
    num = s['num']
    t = html.escape(s.get('title', f'Slide {num}'))
    html_parts.append(f'      <option value="{num}">{num:02d}. {t}</option>')

html_parts.append('''    </select>
    <span class="nav-counter"><span id="curr-idx">1</span> / 45</span>
    <button class="nav-btn" id="btn-next" title="Next Slide (→ / Space / PgDn)">
      Next <span>▸</span>
    </button>
    <button class="nav-btn" id="btn-overview" title="Toggle Slide Overview (O)">
      ⊞ Overview <span class="key-hint">O</span>
    </button>
    <button class="nav-btn" id="btn-print-pdf" title="Print all slides / Save as PDF (Ctrl+P / Cmd+P)">
      <span>⎙</span> Print
    </button>
    <button class="nav-btn" id="btn-fullscreen" title="Toggle Fullscreen (F)">
      ⛶ <span class="key-hint">F</span>
    </button>
  </nav>

  <!-- Grid Overview Modal -->
  <div id="overview-modal">
    <div class="overview-header">
      <div>
        <h2>All Slides Overview (45 Slides)</h2>
        <p style="color: #94A3B8; font-size: 0.85rem; margin-top: 4px;">Click any card to jump directly to that slide. Press Esc or O to close.</p>
      </div>
      <button class="close-btn" id="btn-close-overview">✕</button>
    </div>
    <div class="overview-grid">
''')

for s in slides:
    num = s['num']
    t = html.escape(s.get('title', f'Slide {num}'))
    sec = html.escape(s.get('section', 'Section'))
    html_parts.append(f'''      <div class="overview-card" data-slide="{num}">
        <span class="overview-card-num">SLIDE {num:02d}</span>
        <div class="overview-card-title">{t}</div>
        <span class="overview-card-tag">{sec}</span>
      </div>''')

html_parts.append('''    </div>
  </div>

  <!-- Interactive Logic -->
  <script>
    (function() {
      let currentSlide = 1;
      const totalSlides = 45;
      const slides = document.querySelectorAll('.slide');
      const progressBar = document.getElementById('progress-bar');
      const currIdxSpan = document.getElementById('curr-idx');
      const slideSelect = document.getElementById('slide-select');
      const overviewModal = document.getElementById('overview-modal');
      const overviewCards = document.querySelectorAll('.overview-card');

      const navControls = document.getElementById('nav-controls');
      let hideTimeout = null;
      let isMouseOverControls = false;
      let isSelectFocused = false;

      function showControls() {
        navControls.classList.remove('autohide');
        scheduleAutoHide();
      }

      function scheduleAutoHide() {
        clearTimeout(hideTimeout);
        if (isMouseOverControls || isSelectFocused || overviewModal.classList.contains('active')) {
          return;
        }
        hideTimeout = setTimeout(() => {
          if (!isMouseOverControls && !isSelectFocused && !overviewModal.classList.contains('active')) {
            navControls.classList.add('autohide');
          }
        }, 2500);
      }

      navControls.addEventListener('mouseenter', () => {
        isMouseOverControls = true;
        showControls();
      });

      navControls.addEventListener('mouseleave', () => {
        isMouseOverControls = false;
        scheduleAutoHide();
      });

      slideSelect.addEventListener('focus', () => {
        isSelectFocused = true;
        showControls();
      });

      slideSelect.addEventListener('blur', () => {
        isSelectFocused = false;
        scheduleAutoHide();
      });

      window.addEventListener('mousemove', () => {
        showControls();
      });

      window.addEventListener('touchstart', () => {
        showControls();
      });

      function goToSlide(slideNum) {
        if (slideNum < 1) slideNum = 1;
        if (slideNum > totalSlides) slideNum = totalSlides;

        currentSlide = slideNum;

        slides.forEach((slide) => {
          const num = parseInt(slide.getAttribute('data-slide'), 10);
          if (num === currentSlide) {
            slide.classList.add('active');
          } else {
            slide.classList.remove('active');
          }
        });

        // Update progress bar
        const progress = (currentSlide / totalSlides) * 100;
        progressBar.style.width = progress + '%';

        // Update indicators
        currIdxSpan.textContent = currentSlide;
        slideSelect.value = currentSlide;

        // Update overview active card
        overviewCards.forEach((card) => {
          const cNum = parseInt(card.getAttribute('data-slide'), 10);
          if (cNum === currentSlide) {
            card.classList.add('current');
          } else {
            card.classList.remove('current');
          }
        });

        // Trigger controller display and schedule auto-hide
        showControls();
      }

      function nextSlide() {
        if (currentSlide < totalSlides) {
          goToSlide(currentSlide + 1);
        }
      }

      function prevSlide() {
        if (currentSlide > 1) {
          goToSlide(currentSlide - 1);
        }
      }

      function toggleOverview() {
        overviewModal.classList.toggle('active');
        if (overviewModal.classList.contains('active')) {
          const curCard = document.querySelector(`.overview-card[data-slide="${currentSlide}"]`);
          if (curCard) curCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
      }

      function closeOverview() {
        overviewModal.classList.remove('active');
        showControls();
      }

      function toggleFullscreen() {
        if (!document.fullscreenElement) {
          document.documentElement.requestFullscreen().catch(err => console.log(err));
        } else {
          document.exitFullscreen();
        }
      }

      // Button listeners
      document.getElementById('btn-prev').addEventListener('click', prevSlide);
      document.getElementById('btn-next').addEventListener('click', nextSlide);
      document.getElementById('btn-overview').addEventListener('click', toggleOverview);
      document.getElementById('btn-close-overview').addEventListener('click', closeOverview);
      const btnPrint = document.getElementById('btn-print-pdf');
      if (btnPrint) {
        btnPrint.addEventListener('click', () => {
          window.print();
        });
      }
      document.getElementById('btn-fullscreen').addEventListener('click', toggleFullscreen);

      slideSelect.addEventListener('change', (e) => {
        goToSlide(parseInt(e.target.value, 10));
      });

      overviewCards.forEach((card) => {
        card.addEventListener('click', () => {
          const num = parseInt(card.getAttribute('data-slide'), 10);
          goToSlide(num);
          closeOverview();
        });
      });

      // Keyboard navigation
      window.addEventListener('keydown', (e) => {
        // If select or other input has focus, ignore navigation
        if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT') return;

        if (e.key === 'ArrowRight' || e.key === ' ' || e.key === 'PageDown' || e.key === 'Enter') {
          e.preventDefault();
          nextSlide();
        } else if (e.key === 'ArrowLeft' || e.key === 'PageUp') {
          e.preventDefault();
          prevSlide();
        } else if (e.key === 'Home') {
          e.preventDefault();
          goToSlide(1);
        } else if (e.key === 'End') {
          e.preventDefault();
          goToSlide(totalSlides);
        } else if (e.key === 'o' || e.key === 'O') {
          e.preventDefault();
          toggleOverview();
        } else if (e.key === 'f' || e.key === 'F') {
          e.preventDefault();
          toggleFullscreen();
        } else if (e.key === 'Escape') {
          closeOverview();
        }
      });

      // Touch gestures for tablets / mobile
      let touchStartX = 0;
      let touchEndX = 0;
      document.addEventListener('touchstart', (e) => {
        touchStartX = e.changedTouches[0].screenX;
      }, false);
      document.addEventListener('touchend', (e) => {
        touchEndX = e.changedTouches[0].screenX;
        const diff = touchStartX - touchEndX;
        if (Math.abs(diff) > 50) {
          if (diff > 0) nextSlide();
          else prevSlide();
        }
      }, false);

      // Initial Deck Setup
      goToSlide(1);
    })();
  </script>
</body>
</html>
''')

output_html = '\n'.join(html_parts)
with open('/Users/abc/Documents/study/intelligent-systems/intelligent_system_assignments/assignment_04/slides/index.html', 'w') as f:
    f.write(output_html)

print('Generated index.html successfully! Output size:', len(output_html))
