#!/usr/bin/env python3
"""
Dedicated CNN Core Presentation Deck Generator (33 Slides - Pure Visual & Exhaustive Model Evolution)
Topic: Convolutional Neural Networks (CNN): Mathematical Foundations, Architectural Evolution, and Vectorized Implementation
Authors:
  - Bùi Nguyên Hoàng Việt (B23DCDT285)
  - Lưu Anh Dũng (B23DCDK036)
  - Nguyễn Văn Trường (B23DCCE095)
Class: E23CNPM02 | Course: Intelligent System Development
Lecturer: Assoc. Prof. Dinh Que Tran, Ph.D.
Design: 100% Light Theme, Visual-First (Minimal Text), Dedicated Slides for Every Milestone Architecture
(Neocognitron, LeNet-5, AlexNet, VGGNet, GoogLeNet, ResNet, MobileNet, ConvNeXt/ViT),
Terminal Windows, Zero Takeaway Bars, Zero Header Pills, Subtle Slide Numbers (X / 33).
"""

import os
import sys
from PIL import Image
import pptx
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# -----------------------------------------------------------------------------
# 100% Light Theme Color Palette System
# -----------------------------------------------------------------------------
CANVAS_BG      = RGBColor(248, 250, 252)  # #F8FAFC (Modern clean background canvas)
CARD_BG        = RGBColor(255, 255, 255)  # #FFFFFF (Pure card surface)
WHITE          = RGBColor(255, 255, 255)  # #FFFFFF
CARD_BORDER    = RGBColor(226, 232, 240)  # #E2E8F0 (Crisp subtle light border)
TEXT_MAIN      = RGBColor(15, 23, 42)     # #0F172A (Deep slate heading)
TEXT_BODY      = RGBColor(51, 65, 85)     # #334155 (Readability body slate)
TEXT_MUTED     = RGBColor(100, 116, 139)  # #64748B (Muted subtitle text)

# Accent Colors
COLOR_PRIMARY  = RGBColor(67, 56, 202)    # #4338CA (Royal Indigo)
COLOR_BLUE     = RGBColor(2, 132, 199)    # #0284C7 (Electric Tech Blue)
COLOR_GREEN    = RGBColor(5, 150, 105)    # #059669 (Forest Emerald)
COLOR_AMBER    = RGBColor(217, 119, 6)    # #D97706 (Amber Gold)
COLOR_PURPLE   = RGBColor(124, 58, 237)   # #7C3AED (Deep Violet)
COLOR_ORANGE   = RGBColor(234, 88, 12)    # #EA580C (Coral Orange)
COLOR_ALERT    = RGBColor(225, 29, 72)    # #E11D48 (Crimson Red)

# Terminal / Code Colors
TERM_BG        = RGBColor(15, 23, 42)     # #0F172A (Terminal dark slate)
TERM_BAR       = RGBColor(30, 41, 59)     # #1E293B (Terminal titlebar)
DOT_RED        = RGBColor(239, 68, 68)
DOT_YELLOW     = RGBColor(245, 158, 11)
DOT_GREEN      = RGBColor(16, 185, 129)

FONT_HEAD = "Segoe UI"
FONT_BODY = "Calibri"
FONT_CODE = "Consolas"


class CNNPresentationBuilder:
    def __init__(self, output_pptx="slides/CNN_Presentation.pptx"):
        self.output_pptx = output_pptx
        self.prs = Presentation()
        self.prs.slide_width = Inches(13.333)
        self.prs.slide_height = Inches(7.5)
        self.blank_layout = self.prs.slide_layouts[6]
        self.total_slides = 39
        self.current_slide_num = 0

    def add_notes(self, slide, text):
        slide.notes_slide.notes_text_frame.text = text

    def set_background(self, slide, color=CANVAS_BG):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
        bg.fill.solid()
        bg.fill.fore_color.rgb = color
        bg.line.fill.background()
        return bg

    def add_footer(self, slide, show_number=True):
        self.current_slide_num += 1
        if not show_number:
            return

        num_str = f"{self.current_slide_num} / {self.total_slides}"
        tb_num = slide.shapes.add_textbox(Inches(10.5), Inches(7.05), Inches(2.033), Inches(0.28))
        tf_num = tb_num.text_frame
        tf_num.word_wrap = False
        tf_num.margin_left = tf_num.margin_top = tf_num.margin_right = tf_num.margin_bottom = 0
        p_num = tf_num.paragraphs[0]
        p_num.text = num_str
        p_num.alignment = PP_ALIGN.RIGHT
        p_num.font.name = FONT_BODY
        p_num.font.size = Pt(9.5)
        p_num.font.bold = True
        p_num.font.color.rgb = RGBColor(148, 163, 184)

    def add_header(self, slide, title_text, subtitle_text=None):
        tb_title = slide.shapes.add_textbox(Inches(0.8), Inches(0.48), Inches(11.733), Inches(0.55))
        tf_title = tb_title.text_frame
        tf_title.word_wrap = True
        tf_title.margin_left = tf_title.margin_top = tf_title.margin_right = tf_title.margin_bottom = 0
        p_title = tf_title.paragraphs[0]
        p_title.text = title_text
        p_title.font.name = FONT_HEAD
        p_title.font.size = Pt(25)
        p_title.font.bold = True
        p_title.font.color.rgb = TEXT_MAIN

        if subtitle_text:
            tb_sub = slide.shapes.add_textbox(Inches(0.8), Inches(1.08), Inches(11.733), Inches(0.36))
            tf_sub = tb_sub.text_frame
            tf_sub.word_wrap = True
            tf_sub.margin_left = tf_sub.margin_top = tf_sub.margin_right = tf_sub.margin_bottom = 0
            p_sub = tf_sub.paragraphs[0]
            p_sub.text = subtitle_text
            p_sub.font.name = FONT_BODY
            p_sub.font.size = Pt(13.5)
            p_sub.font.color.rgb = TEXT_MUTED

    def add_image_panel(self, slide, left, top, width, height, img_path, caption=None, border=True):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = WHITE
        card.line.color.rgb = CARD_BORDER if border else WHITE
        card.line.width = Pt(1.2)

        cap_h = Inches(0.44) if caption else Inches(0)
        pad_x = Inches(0.14)
        pad_y = Inches(0.12)
        avail_w = width - pad_x * 2
        avail_h = height - pad_y * 2 - cap_h

        if os.path.exists(img_path):
            try:
                with Image.open(img_path) as im:
                    im_w, im_h = im.size
                aspect = im_w / max(1, im_h)
                box_aspect = avail_w / max(1, avail_h)

                if aspect > box_aspect:
                    final_w = avail_w
                    final_h = avail_w / aspect
                    final_left = left + pad_x
                    final_top = top + pad_y + (avail_h - final_h) / 2
                else:
                    final_h = avail_h
                    final_w = avail_h * aspect
                    final_top = top + pad_y
                    final_left = left + pad_x + (avail_w - final_w) / 2

                slide.shapes.add_picture(img_path, final_left, final_top, width=final_w, height=final_h)
            except Exception as e:
                print(f"Warning loading {img_path}: {e}")

        if caption:
            tb = slide.shapes.add_textbox(left + Inches(0.12), top + height - Inches(0.44), width - Inches(0.24), Inches(0.38))
            tf = tb.text_frame
            tf.word_wrap = True
            tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
            p = tf.paragraphs[0]
            p.text = caption
            p.font.name = FONT_HEAD
            p.font.size = Pt(11)
            p.font.bold = True
            p.font.color.rgb = COLOR_BLUE
            p.alignment = PP_ALIGN.CENTER
        return card

    def add_bento_box(self, slide, left, top, width, height, title=None, title_color=TEXT_MAIN):
        box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        box.fill.solid()
        box.fill.fore_color.rgb = CARD_BG
        box.line.color.rgb = CARD_BORDER
        box.line.width = Pt(1.2)

        tf = box.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = MSO_ANCHOR.TOP
        tf.margin_left = Inches(0.22)
        tf.margin_right = Inches(0.22)
        tf.margin_top = Inches(0.20)
        tf.margin_bottom = Inches(0.20)

        if title:
            p = tf.paragraphs[0]
            p.text = title
            p.font.name = FONT_HEAD
            p.font.size = Pt(17)
            p.font.bold = True
            p.font.color.rgb = title_color
            p.space_after = Pt(8)
        return box, tf

    def add_large_bullets(self, tf, items, font_size=Pt(13.5)):
        """Adds concise, high-impact bullet points with bold badges."""
        for idx, (badge, text) in enumerate(items):
            p = tf.add_paragraph() if (idx > 0 or len(tf.paragraphs[0].text) > 0) else tf.paragraphs[0]
            p.space_after = Pt(10)
            p.line_spacing = 1.18
            run_b = p.add_run()
            run_b.text = f"• {badge}: "
            run_b.font.name = FONT_HEAD
            run_b.font.size = font_size
            run_b.font.bold = True
            run_b.font.color.rgb = COLOR_BLUE

            run_t = p.add_run()
            run_t.text = text
            run_t.font.name = FONT_BODY
            run_t.font.size = font_size
            run_t.font.color.rgb = TEXT_BODY

    def add_terminal(self, slide, left, top, width, height, filename, code_lines, code_font_size=Pt(10)):
        win = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        win.fill.solid()
        win.fill.fore_color.rgb = TERM_BG
        win.line.color.rgb = RGBColor(51, 65, 85)
        win.line.width = Pt(1)

        bar = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, Inches(0.34))
        bar.fill.solid()
        bar.fill.fore_color.rgb = TERM_BAR
        bar.line.fill.background()

        for j, dot_col in enumerate([DOT_RED, DOT_YELLOW, DOT_GREEN]):
            dot = slide.shapes.add_shape(MSO_SHAPE.OVAL, left + Inches(0.16) + j * Inches(0.18), top + Inches(0.10), Inches(0.11), Inches(0.11))
            dot.fill.solid()
            dot.fill.fore_color.rgb = dot_col
            dot.line.fill.background()

        tb_fn = slide.shapes.add_textbox(left + Inches(0.85), top + Inches(0.06), width - Inches(1.0), Inches(0.24))
        tf_fn = tb_fn.text_frame
        p_fn = tf_fn.paragraphs[0]
        p_fn.text = filename
        p_fn.font.name = FONT_CODE
        p_fn.font.size = Pt(10.5)
        p_fn.font.color.rgb = RGBColor(148, 163, 184)

        tb_c = slide.shapes.add_textbox(left + Inches(0.22), top + Inches(0.40), width - Inches(0.44), height - Inches(0.44))
        tf_c = tb_c.text_frame
        tf_c.word_wrap = True
        tf_c.vertical_anchor = MSO_ANCHOR.TOP
        tf_c.margin_left = tf_c.margin_top = tf_c.margin_right = tf_c.margin_bottom = 0

        for idx, line in enumerate(code_lines):
            p = tf_c.paragraphs[0] if idx == 0 else tf_c.add_paragraph()
            p.text = line
            p.font.name = FONT_CODE
            p.font.size = code_font_size
            p.space_before = Pt(0)
            p.space_after = Pt(0)
            p.line_spacing = Pt(code_font_size.pt * 1.25)
            if line.strip().startswith("#"):
                p.font.color.rgb = RGBColor(100, 116, 139)
            elif "def " in line or "class " in line or "import " in line:
                p.font.color.rgb = RGBColor(244, 114, 182)
            elif "return " in line or "for " in line or "in " in line:
                p.font.color.rgb = RGBColor(56, 189, 248)
            else:
                p.font.color.rgb = RGBColor(226, 232, 240)
        return win

    # -------------------------------------------------------------------------
    # Visual-First Layout Engines (60% Visual, 40% Concise Text)
    # -------------------------------------------------------------------------

    def layout_split_left_image(self, slide, img_path, caption, card_title, bullets, title_color=COLOR_BLUE):
        """Layout: Left Image (58%) + Right Bento Card (42%)."""
        self.add_image_panel(slide, Inches(0.8), Inches(1.85), Inches(6.8), Inches(4.85), img_path, caption=caption)
        box, tf = self.add_bento_box(slide, Inches(7.85), Inches(1.85), Inches(4.683), Inches(4.85), title=card_title, title_color=title_color)
        self.add_large_bullets(tf, bullets, font_size=Pt(13.5))

    def layout_split_right_image(self, slide, card_title, bullets, img_path, caption, title_color=COLOR_BLUE):
        """Layout: Left Bento Card (42%) + Right Image (58%)."""
        box, tf = self.add_bento_box(slide, Inches(0.8), Inches(1.85), Inches(4.683), Inches(4.85), title=card_title, title_color=title_color)
        self.add_large_bullets(tf, bullets, font_size=Pt(13.5))
        self.add_image_panel(slide, Inches(5.733), Inches(1.85), Inches(6.8), Inches(4.85), img_path, caption=caption)

    def layout_split_dual_images(self, slide, img1_path, cap1, img2_path, cap2, card_title, bullets, title_color=COLOR_BLUE):
        """Layout: Left Top & Bottom Dual Images + Right Bento Card."""
        self.add_image_panel(slide, Inches(0.8), Inches(1.85), Inches(6.8), Inches(2.35), img1_path, caption=cap1)
        self.add_image_panel(slide, Inches(0.8), Inches(4.35), Inches(6.8), Inches(2.35), img2_path, caption=cap2)
        box, tf = self.add_bento_box(slide, Inches(7.85), Inches(1.85), Inches(4.683), Inches(4.85), title=card_title, title_color=title_color)
        self.add_large_bullets(tf, bullets, font_size=Pt(13.5))

    def layout_triptych(self, slide, cols_data):
        """Layout: 3-Column Bento Grid."""
        card_w = Inches(3.70)
        gap = Inches(0.31)
        for i, col in enumerate(cols_data):
            c_badge, c_title, c_bullets, c_col, c_img = col
            left_p = Inches(0.8) + i * (card_w + gap)
            c, tf = self.add_bento_box(slide, left_p, Inches(1.85), card_w, Inches(4.85))
            
            p0 = tf.paragraphs[0]
            p0.text = c_badge.upper()
            p0.font.name = FONT_HEAD
            p0.font.size = Pt(11)
            p0.font.bold = True
            p0.font.color.rgb = c_col
            p0.space_after = Pt(2)

            p1 = tf.add_paragraph()
            p1.text = c_title
            p1.font.name = FONT_HEAD
            p1.font.size = Pt(16)
            p1.font.bold = True
            p1.font.color.rgb = TEXT_MAIN
            p1.space_after = Pt(8)

            for b_title, b_desc in c_bullets:
                p_b = tf.add_paragraph()
                p_b.space_after = Pt(6)
                r1 = p_b.add_run()
                r1.text = f"• {b_title}: "
                r1.font.name = FONT_HEAD
                r1.font.size = Pt(12)
                r1.font.bold = True
                r1.font.color.rgb = c_col

                r2 = p_b.add_run()
                r2.text = b_desc
                r2.font.name = FONT_BODY
                r2.font.size = Pt(12)
                r2.font.color.rgb = TEXT_BODY

    def layout_native_rosetta_table(self, slide, headers, rows_data, col_widths):
        """Generates a native vector PowerPoint table for the Rosetta Stone with large, high-legibility fonts."""
        num_rows = len(rows_data) + 1
        num_cols = len(headers)
        left, top, width, height = Inches(0.56), Inches(1.80), Inches(12.213), Inches(5.15)
        table_shape = slide.shapes.add_table(num_rows, num_cols, left, top, width, height)
        tbl = table_shape.table

        for j, w in enumerate(col_widths):
            tbl.columns[j].width = w

        header_colors = [WHITE, RGBColor(245, 158, 11), RGBColor(56, 189, 248), RGBColor(52, 211, 153)]

        for j, h in enumerate(headers):
            cell = tbl.cell(0, j)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(15, 23, 42)
            cell.margin_left = Inches(0.08)
            cell.margin_right = Inches(0.08)
            cell.margin_top = Inches(0.12)
            cell.margin_bottom = Inches(0.12)
            p = cell.text_frame.paragraphs[0]
            p.text = h
            p.font.name = FONT_HEAD
            p.font.size = Pt(14)
            p.font.bold = True
            p.font.color.rgb = header_colors[j]

        for i, row in enumerate(rows_data):
            bg_col = RGBColor(248, 250, 252) if i % 2 == 0 else WHITE
            for j, val in enumerate(row):
                cell = tbl.cell(i + 1, j)
                cell.vertical_anchor = MSO_ANCHOR.MIDDLE
                cell.fill.solid()
                cell.fill.fore_color.rgb = bg_col
                cell.margin_left = Inches(0.08)
                cell.margin_right = Inches(0.08)
                cell.margin_top = Inches(0.08)
                cell.margin_bottom = Inches(0.08)

                lines = str(val).split("\n")
                cell.text_frame.text = ""
                for line_idx, line in enumerate(lines):
                    p = cell.text_frame.paragraphs[0] if line_idx == 0 else cell.text_frame.add_paragraph()
                    p.text = line
                    if j == 0:
                        p.font.name = FONT_HEAD
                        p.font.size = Pt(13.5)
                        p.font.bold = True
                        p.font.color.rgb = TEXT_MAIN
                    else:
                        p.font.name = FONT_CODE
                        p.font.size = Pt(12)
                        p.font.bold = True
                        if j == 1:
                            p.font.color.rgb = RGBColor(180, 83, 9)   # Amber dark
                        elif j == 2:
                            p.font.color.rgb = RGBColor(2, 132, 199)  # Electric Blue
                        elif j == 3:
                            p.font.color.rgb = RGBColor(5, 150, 105)  # Emerald

    # -------------------------------------------------------------------------
    # PART I: OVERVIEW & VISION SCOPE (Slides 1 - 3)
    # -------------------------------------------------------------------------

    def build_slide_01(self):
        """Slide 1: Title Slide (Hero Card, Light Theme, 3 Authors)"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)

        hero = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(0.8), Inches(11.733), Inches(5.9))
        hero.fill.solid()
        hero.fill.fore_color.rgb = CARD_BG
        hero.line.color.rgb = CARD_BORDER
        hero.line.width = Pt(1.5)

        tb_tag = slide.shapes.add_textbox(Inches(1.2), Inches(1.15), Inches(10.0), Inches(0.35))
        tf_tag = tb_tag.text_frame
        p_tag = tf_tag.paragraphs[0]
        p_tag.text = "INTELLIGENT SYSTEMS · DEEP LEARNING ARCHITECTURE"
        p_tag.font.name = FONT_HEAD
        p_tag.font.size = Pt(12)
        p_tag.font.bold = True
        p_tag.font.color.rgb = COLOR_PRIMARY

        tb_t = slide.shapes.add_textbox(Inches(1.2), Inches(1.55), Inches(10.8), Inches(1.1))
        tf_t = tb_t.text_frame
        p_t = tf_t.paragraphs[0]
        p_t.text = "Convolutional Neural Networks (CNN)"
        p_t.font.name = FONT_HEAD
        p_t.font.size = Pt(36)
        p_t.font.bold = True
        p_t.font.color.rgb = TEXT_MAIN

        tb_sub = slide.shapes.add_textbox(Inches(1.2), Inches(2.65), Inches(10.8), Inches(0.5))
        tf_sub = tb_sub.text_frame
        p_sub = tf_sub.paragraphs[0]
        p_sub.text = "Mathematical Foundations, Architectural Evolution, and Vectorized Implementation"
        p_sub.font.name = FONT_BODY
        p_sub.font.size = Pt(16)
        p_sub.font.color.rgb = TEXT_MUTED

        self.add_image_panel(slide, Inches(1.2), Inches(3.30), Inches(10.933), Inches(1.95),
                             "slides/assets/typical_cnn.png",
                             caption="Canonical End-to-End Convolutional Visual Perception Topology")

        tb_auth = slide.shapes.add_textbox(Inches(1.2), Inches(5.45), Inches(10.933), Inches(1.0))
        tf_auth = tb_auth.text_frame
        tf_auth.word_wrap = True

        p_a1 = tf_auth.paragraphs[0]
        p_a1.text = "Authors:  Bùi Nguyên Hoàng Việt (B23DCDT285)  |  Lưu Anh Dũng (B23DCDK036)  |  Nguyễn Văn Trường (B23DCCE095)"
        p_a1.font.name = FONT_HEAD
        p_a1.font.size = Pt(13)
        p_a1.font.bold = True
        p_a1.font.color.rgb = TEXT_MAIN

        p_a2 = tf_auth.add_paragraph()
        p_a2.text = "Class: E23CNPM02  ·  Lecturer: Assoc. Prof. Dinh Que Tran, Ph.D.  ·  Post & Telecommunications Institute of Technology (PTIT)"
        p_a2.font.name = FONT_BODY
        p_a2.font.size = Pt(11.5)
        p_a2.font.color.rgb = TEXT_MUTED

        self.add_footer(slide, show_number=False)

    def build_slide_02(self):
        """Slide 2: 4-Pillar Structural Agenda (2x2 Visual Grid, No List)"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "Structural System Agenda", "Navigating the 4 Core Pillars of Convolutional Representation Learning")

        pillars = [
            ("PILLAR 01", "Overview & Vision Scope", "Spatial Geometry & Foundations",
             "Analysis of 2D pixel topology, the failure mode of vector unrolling, and inductive locality priors.",
             COLOR_PRIMARY, "slides/assets/flattening_unrolling.png"),

            ("PILLAR 02", "Khái niệm (Concept & Mechanics)", "Mathematical Foundations",
             "Layer function composition, linear collapse proof, 2D/3D convolutions, stride/padding, and ReLU sparsity.",
             COLOR_BLUE, "slides/assets/2D_convolution_frame_mid.png"),

            ("PILLAR 03", "Phát triển CNN (Model Evolution)", "Historical Milestones & Modern Frontiers",
             "Exhaustive model deep-dives: Neocognitron, LeNet-5, AlexNet, VGG, Inception, ResNet, DenseNet, U-Net, MobileNet, EfficientNet, SENet, CBAM & ViT.",
             COLOR_PURPLE, "slides/assets/vgg_architecture_diagram.png"),

            ("PILLAR 04", "Code Demo & Verification", "Vectorized Implementation",
             "Vectorized im2col forward pass, analytical col2im backprop, PyTorch module, and learned filter maps.",
             COLOR_GREEN, "slides/assets/learned_filters_fmnist.png")
        ]

        card_w = Inches(5.72)
        card_h = Inches(2.40)
        gap_x = Inches(0.29)
        gap_y = Inches(0.20)

        coords = [
            (Inches(0.80), Inches(1.85)),
            (Inches(0.80) + card_w + gap_x, Inches(1.85)),
            (Inches(0.80), Inches(1.85) + card_h + gap_y),
            (Inches(0.80) + card_w + gap_x, Inches(1.85) + card_h + gap_y)
        ]

        for i, (badge, title, domain, desc, col, img_path) in enumerate(pillars):
            lp, tp = coords[i]
            self.add_bento_box(slide, lp, tp, card_w, card_h)

            tb = slide.shapes.add_textbox(lp + Inches(0.22), tp + Inches(0.18), Inches(2.68), Inches(2.04))
            tf_t = tb.text_frame
            tf_t.word_wrap = True
            tf_t.vertical_anchor = MSO_ANCHOR.TOP
            tf_t.margin_left = tf_t.margin_top = tf_t.margin_right = tf_t.margin_bottom = 0

            p0 = tf_t.paragraphs[0]
            p0.text = badge
            p0.font.name = FONT_HEAD
            p0.font.size = Pt(11)
            p0.font.bold = True
            p0.font.color.rgb = col
            p0.space_after = Pt(2)

            p1 = tf_t.add_paragraph()
            p1.text = title
            p1.font.name = FONT_HEAD
            p1.font.size = Pt(15)
            p1.font.bold = True
            p1.font.color.rgb = TEXT_MAIN
            p1.space_after = Pt(2)

            p2 = tf_t.add_paragraph()
            p2.text = domain
            p2.font.name = FONT_BODY
            p2.font.size = Pt(11)
            p2.font.italic = True
            p2.font.color.rgb = TEXT_MUTED
            p2.space_after = Pt(6)

            p3 = tf_t.add_paragraph()
            p3.text = desc
            p3.font.name = FONT_BODY
            p3.font.size = Pt(11)
            p3.font.color.rgb = TEXT_BODY
            p3.line_spacing = 1.15

            if os.path.exists(img_path):
                self.add_image_panel(slide, lp + Inches(3.02), tp + Inches(0.18), Inches(2.52), Inches(2.04),
                                     img_path, border=True)

        self.add_footer(slide)

    def build_slide_03(self):
        """Slide 3: Computer Vision Challenge — 2D Topology vs. 1D Flattening"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "Spatial Topology vs. Vector Unrolling",
                        "Why Traditional Dense Multi-Layer Perceptrons Fail on 2D Image Manifolds")

        bullets = [
            ("Spatial Adjacency Destruction",
             "Unrolling a 2D image matrix into a 1D vector severs vertical neighbor relationships, permanently discarding geometric 2D pixel topology."),
            ("Loss of Translation Equivariance",
             "Dense MLPs cannot recognize an object if it shifts by a single pixel without relearning independent synaptic weights across every coordinate."),
            ("Coordinate Independence",
             "Convolutions preserve the 2D grid structure, applying the same local kernel regardless of where visual patterns appear.")
        ]
        self.layout_split_right_image(slide, "The Coordinate Destruction Problem", bullets,
                                      "slides/assets/flattening_unrolling.png",
                                      "Flattening a 2D Matrix (H x W) into a 1D Array Destroys Vertical Adjacency")
        self.add_footer(slide)

    # -------------------------------------------------------------------------
    # PART II: KHÁI NIỆM - CONCEPT FUNCTION & MECHANICS (Slides 4 - 16)
    # -------------------------------------------------------------------------

    def build_slide_04(self):
        """Slide 4: Concept Function — Analytical Function Composition"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "Analytical Function Composition",
                        "Deep Networks as Layer-wise Non-Linear Coordinate Transformations")

        bullets = [
            ("Composite Mapping",
             "A deep network represents an end-to-end composite function: F(X; Theta) = (f_L o f_{L-1} o ... o f_1)(X), progressively untangling raw manifolds."),
            ("Layer Transformation",
             "Each layer f_l applies an affine transformation followed by point-wise non-linearity: f_l(h) = sigma(W_l * h + b_l)."),
            ("Separable Latent Space",
             "Intermediate representations transform non-linearly entangled input pixels into a linearly separable classification space at the final layer.")
        ]
        self.layout_split_left_image(slide, "slides/assets/cnn_hierarchy_concept.png",
                                     "Hierarchical Manifold Untangling: Low-Level Edges -> Mid Motifs -> Class Concepts",
                                     "Functional Architecture", bullets, COLOR_BLUE)
        self.add_footer(slide)

    def build_slide_05(self):
        """Slide 5: Affine Transformations & The Linear Collapse Proof"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "Mandatory Non-Linearity & Linear Collapse Proof",
                        "Mathematical Proof of Why Stacking Linear Layers Adds Zero Expressive Power")

        bullets = [
            ("The Linear Collapse Theorem",
             "If all activation functions are linear (sigma(z) = z), then F(X) = W_L (W_{L-1} ... (W_1 X)) = (Product W_l) X = W_{eff} X + b_{eff}."),
            ("Zero Depth Advantage",
             "A 100-layer pure linear network collapses mathematically to a single shallow linear regression, completely incapable of solving non-linear boundaries."),
            ("Manifold Folding via ReLU",
             "Point-wise non-linear activations (sigma) fold and partition the vector space into piecewise linear decision regions of exponential complexity.")
        ]
        self.layout_split_right_image(slide, "Analytical Proof of Linear Collapse", bullets,
                                      "slides/assets/relu_function_wiki.png",
                                      "Non-Linear Activation Function f(x) = max(0, x) Partitions the Space into Piecewise Linear Regions")
        self.add_footer(slide)

    def build_slide_06(self):
        """Slide 6: The MLP Failure Mode on High-Dimensional Grids"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "The MLP Parameter Explosion Catastrophe",
                        "Why Fully-Connected Layers Scale Catastrophically on Visual Inputs")

        bullets = [
            ("Combinatorial Weight Explosion",
             "A modest 224 x 224 x 3 image flattened into 150,528 inputs connected to 1,024 hidden neurons requires >154 Million weights in Layer 1 alone!"),
            ("Catastrophic Overfitting",
             "Massive parameter counts without spatial inductive priors cause severe overfitting and immediate GPU memory exhaustion."),
            ("Convolutional Efficiency",
             "A 3 x 3 x 3 convolutional filter requires only 28 learnable parameters, maintaining efficiency regardless of image resolution.")
        ]
        self.layout_split_left_image(slide, "slides/assets/mlp_vs_cnn_parameters.png",
                                     "Parameter Comparison: Fully-Connected (154M Params) vs. Convolutional (28 Params)",
                                     "High-Dimensional Failure Mode", bullets, COLOR_ALERT)
        self.add_footer(slide)

    def build_slide_07(self):
        """Slide 7: Biological Genesis — Hubel & Wiesel (1959)"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "Biological Blueprint: Hubel & Wiesel (1959)",
                        "Discovering Localized Receptive Fields in the Mammalian Visual Cortex (V1)")

        bullets = [
            ("Localized Simple Cells",
             "Hubel & Wiesel proved that neurons in the primary visual cortex (V1) fire only to localized oriented edges, not full-field illumination."),
            ("Orientation Tuning",
             "Individual cortical cells exhibit sharp tuning curves, responding maximally to specific bar angles (0°, 45°, 90°) within a restricted receptive field."),
            ("Hierarchical Processing",
             "Complex cells aggregate simple cell outputs, inspiring neocognitron and modern convolutional kernel hierarchies.")
        ]
        self.layout_split_dual_images(slide, "slides/assets/hubel_wiesel_receptive_field.png",
                                      "Receptive Field Biological Mechanism (Hubel & Wiesel, 1959)",
                                      "slides/assets/orientation_tuning_curve.png",
                                      "Biological Orientation Tuning Curves in V1 Cortical Cells",
                                      "Neurobiological Foundations", bullets, COLOR_PRIMARY)
        self.add_footer(slide)

    def build_slide_08(self):
        """Slide 8: 2D Convolution Mechanics — Cross-Correlation Kernel Sliding"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "2D Discrete Convolution Mechanics",
                        "Sliding Window Cross-Correlation Across Spatial Image Grids")

        bullets = [
            ("Mathematical Formulation",
             "Discrete convolution (cross-correlation): S(i, j) = (I * K)(i, j) = Sum_m Sum_n I(i+m, j+n) K(m, n) + b."),
            ("Kernel Dot Product",
             "A small K x K matrix slides across the input tensor, taking element-wise products and summing them to produce one output feature activation."),
            ("Feature Extraction",
             "Filters act as specialized feature detectors, yielding high positive activations when matching patterns appear.")
        ]
        self.layout_split_left_image(slide, "slides/assets/2D_convolution_frame_mid.png",
                                     "Sliding Window Cross-Correlation: Kernel Receptive Field Dot Product",
                                     "Sliding Window Mechanics", bullets, COLOR_BLUE)
        self.add_footer(slide)

    def build_slide_09(self):
        """Slide 9: The Two Core CNN Axioms — Local Connectivity & Weight Sharing"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "The Dual Foundations of Convolutional Efficiency",
                        "How Inductive Biases Drastically Reduce Parameter Complexity")

        bullets = [
            ("Axiom 1: Local Connectivity",
             "Each hidden neuron connects exclusively to a small spatial patch (K x K), reflecting the physical law that nearby pixels are strongly correlated."),
            ("Axiom 2: Weight Sharing",
             "The identical filter coefficients are reused across all spatial coordinates, guaranteeing translation equivariance: f(g(x)) = g(f(x))."),
            ("Statistical Efficiency",
             "Instead of learning millions of uncoordinated weights, the network learns a compact set of universal feature detectors.")
        ]
        self.layout_split_right_image(slide, "Core Inductive Biases", bullets,
                                      "slides/assets/equivariance_vs_invariance.png",
                                      "Translation Equivariance: Shifting Input Shifts Output Maps Identically")
        self.add_footer(slide)

    def build_slide_10(self):
        """Slide 10: Spatial Hyperparameters — Stride & Padding"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "Spatial Hyperparameters: Stride & Padding",
                        "Exact Boundary Control and Output Spatial Dimension Arithmetic")

        bullets = [
            ("Output Dimension Equation",
             "Universal spatial resolution formula: H_{out} = floor((H_{in} - K + 2P) / S) + 1."),
            ("Padding Modes",
             "Valid padding (P = 0) contracts boundaries; Same padding (P = (K - 1) / 2) preserves spatial resolution (H_{out} = H_{in})."),
            ("Stride Subsampling",
             "Stride S > 1 steps across multiple pixels per operation, halving spatial dimensions while expanding the receptive field.")
        ]
        self.layout_split_dual_images(slide, "slides/assets/conv_same_padding_frame.png",
                                      "Same Padding: Zero-Boundary Padding Preserves Spatial Grid Dimensions",
                                      "slides/assets/conv_strides_frame.png",
                                      "Stride S = 2: Skips Intermediate Pixels for Spatial Subsampling",
                                      "Hyperparameter Control", bullets, COLOR_BLUE)
        self.add_footer(slide)

    def build_slide_11(self):
        """Slide 11: Receptive Field Expansion Arithmetic"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "Receptive Field Growth Across Deep Stacks",
                        "How Stacking Small 3x3 Filters Yields Exponential Spatial Coverage")

        bullets = [
            ("Recursive RF Formula",
             "Receptive field expands recursively: RF_l = RF_{l-1} + (K_l - 1) * Product_{i=1}^{l-1} S_i."),
            ("The VGG-16 Architectural Law",
             "Two stacked 3 x 3 convolutions match the 5 x 5 receptive field of a single 5 x 5 layer with 18 parameters vs 25 (28% savings) plus extra non-linearity."),
            ("Hierarchical Context",
             "Deep layers integrate global visual context without requiring large, computationally prohibitive kernel windows.")
        ]
        self.layout_split_left_image(slide, "slides/assets/receptive_field_expansion.png",
                                     "Receptive Field Arithmetic: Stacking 3x3 Kernels Yields 5x5 and 7x7 Context",
                                     "Receptive Field Dynamics", bullets, COLOR_PURPLE)
        self.add_footer(slide)

    def build_slide_12(self):
        """Slide 12: Volumetric 3D Tensor Convolution"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "Volumetric 3D Tensor Convolutions",
                        "Multichannel Feature Mapping Across Spatial and Depth Dimensions")

        bullets = [
            ("Tensor Volume Shape",
             "Input tensor (C_{in} x H x W) is processed by C_{out} separate 3D filters, each having shape (C_{in} x K_h x K_w)."),
            ("Channel-Wise Summation",
             "Each 3D filter computes 2D convolutions across all C_{in} channels simultaneously, summing the results into one 2D feature map slice."),
            ("Channel Expansion",
             "Stacking C_{out} filter responses produces the output volume of shape (C_{out} x H_{out} x W_{out}).")
        ]
        self.layout_split_left_image(slide, "slides/assets/cs231n_convnet.jpeg",
                                     "Stanford CS231n Volumetric Convolution: C_in x H x W Convolved into C_out Feature Maps",
                                     "Multichannel Tensor Flow", bullets, COLOR_BLUE)
        self.add_footer(slide)

    def build_slide_13(self):
        """Slide 13: Convolutional Parameter Counting Formulation"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "Universal Parameter Counting Formulation",
                        "Exact Parameter Derivation Independent of Spatial Input Resolution")

        bullets = [
            ("Universal Parameter Equation",
             "Total learnable weights: Params = (K_h * K_w * C_{in} + 1) * C_{out}, where +1 accounts for the per-filter bias."),
            ("Resolution Independence",
             "The parameter count depends strictly on kernel dimensions and channel counts, completely independent of input image height (H) and width (W)."),
            ("Concrete Example",
             "A layer with K = 3, C_{in} = 64, C_{out} = 128 requires (3 * 3 * 64 + 1) * 128 = 73,856 parameters.")
        ]
        self.layout_split_left_image(slide, "slides/assets/parameter_formula_breakdown.png",
                                     "Analytical Parameter Breakdown: (Kernel Area x Input Channels + Bias) x Output Channels",
                                     "Exact Mathematical Derivation", bullets, COLOR_GREEN)
        self.add_footer(slide)

    def build_slide_14(self):
        """Slide 14: Non-Linear Activation Functions — ReLU Manifold Sparsity"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "Non-Linear Activations & Sparsity",
                        "Rectified Linear Unit (ReLU) Mechanics and Gradient Flow Preservation")

        bullets = [
            ("The ReLU Definition",
             "f(x) = max(0, x), with analytical derivative f'(x) = 1 for x > 0 and 0 for x < 0."),
            ("Vanishing Gradient Cure",
             "Unlike Sigmoid/Tanh which saturate at extreme values, ReLU maintains a constant gradient of 1, enabling stable training of 100+ layer networks."),
            ("Induced Sparsity",
             "Deactivating negative activations creates biological-like sparse representations (~50% active neurons), improving generalization.")
        ]
        self.layout_split_right_image(slide, "ReLU Activation Dynamics", bullets,
                                      "slides/assets/cs231n_act1.jpeg",
                                      "Visualizing Spatial Activation Maps After Non-Linear ReLU Filtering")
        self.add_footer(slide)

    def build_slide_15(self):
        """Slide 15: Spatial Downsampling — Max Pooling Mechanics"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "Spatial Downsampling & Max Pooling",
                        "Translation Invariance Induction and Dimensionality Reduction")

        bullets = [
            ("Max Pooling Operation",
             "Slides a 2 x 2 window with stride 2, retaining the maximum activation and discarding 75% of spatial coordinates."),
            ("Translation Invariance",
             "Small translations or deformations in the input feature map do not change the pooled maximum value, improving spatial robustness."),
            ("Zero Learnable Parameters",
             "Pooling is a deterministic spatial operation requiring zero parameters, reducing memory footprint and computational FLOPs.")
        ]
        self.layout_split_left_image(slide, "slides/assets/max_pooling.png",
                                     "Max Pooling 2x2 with Stride 2: Extracts Peak Activation per Local Quadrant",
                                     "Spatial Downsampling", bullets, COLOR_BLUE)
        self.add_footer(slide)

    def build_slide_16(self):
        """Slide 16: Hierarchical Feature Representation Across Depth"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "Hierarchical Feature Representation",
                        "Visual Progression from Low-Level Edges to High-Level Semantic Concepts")

        bullets = [
            ("Shallow Layers (V1)",
             "Filters learn localized oriented Gabor-like edges, color contrasts, and boundary primitives."),
            ("Intermediate Layers",
             "Convolutions combine simple edges into geometric textures, corners, junctions, and contour curves."),
            ("Deep Layers",
             "Receptive fields span the entire object, assembling semantic class detectors (wheels, eyes, faces, class templates).")
        ]
        self.layout_split_dual_images(slide, "slides/assets/cnn_hierarchy_concept.png",
                                      "Visual Feature Abstraction Progression: Edges -> Textures -> Parts -> Objects",
                                      "slides/assets/cs231n_weights.jpeg",
                                      "Learned Convolutional Filter Weights Visualized as Spatial Patterns",
                                      "Hierarchical Abstraction", bullets, COLOR_PRIMARY)
        self.add_footer(slide)

    # -------------------------------------------------------------------------
    # PART III: PHÁT TRIỂN CNN - MODEL ARCHITECTURAL EVOLUTION (Slides 17 - 29)
    # -------------------------------------------------------------------------

    def build_slide_17(self):
        """Slide 17: Milestone 1 — Neocognitron (Fukushima, 1980)"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "Architectural Milestone: Neocognitron (1980)",
                        "Kunihiko Fukushima's Bio-Inspired Ancestor of Convolutional Neural Networks")

        bullets = [
            ("S-Cells & C-Cells",
             "Directly embodied Hubel & Wiesel's neurobiology: S-cells extract localized oriented features, while C-cells pool responses to provide shift tolerance."),
            ("Self-Organizing Architecture",
             "Introduced multi-layered hierarchical visual processing without backpropagation, establishing the foundational alternating conv-pool design."),
            ("Translation Invariance",
             "First computational neural model capable of recognizing deformed and translated patterns through localized receptive field hierarchies.")
        ]
        self.layout_split_left_image(slide, "slides/assets/neocognitron_ScholarFig1.png",
                                     "Neocognitron (Fukushima, 1980): Hierarchical Alternation of S-Layers and C-Layers",
                                     "The Biological Computational Ancestor", bullets, COLOR_PRIMARY)
        self.add_footer(slide)

    def build_slide_18(self):
        """Slide 18: Milestone 2 — LeNet-5 (LeCun, 1998)"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "Architectural Milestone: LeNet-5 (1998)",
                        "Yann LeCun's End-to-End Gradient-Based Convolutional Network for Digit Recognition")

        bullets = [
            ("End-to-End Gradient Optimization",
             "Pioneered joint feature extraction and classification optimized simultaneously via backpropagation: Input -> C1 -> S2 -> C3 -> S4 -> C5 -> F6 -> Output."),
            ("Shift and Distortion Robustness",
             "Replaced fragile manual feature extractors with weight-shared convolutional filters and subsampling layers."),
            ("Commercial Validation",
             "Successfully deployed by US banks to read over 10% of all checks in North America during the late 1990s.")
        ]
        self.layout_split_left_image(slide, "slides/assets/lenet5_architecture_diagram.png",
                                     "LeNet-5 Architecture: 7 Layers of Alternating Convolutions and Subsampling",
                                     "Genesis of Modern Deep Vision", bullets, COLOR_BLUE)
        self.add_footer(slide)

    def build_slide_19(self):
        """Slide 19: Milestone 3 — AlexNet (Krizhevsky et al., 2012)"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "Architectural Milestone: AlexNet (2012)",
                        "The ImageNet Breakthrough That Ignited the Modern Deep Learning Revolution")

        bullets = [
            ("Historical Breakthrough",
             "Crushed traditional computer vision by winning ImageNet 2012 with a top-5 error of 16.4% (vs. 26.2% for the runner-up)."),
            ("Modern Technical Suite",
             "First to combine ReLU non-linearities (6x faster training), Dropout regularization (0.5), and data augmentation to prevent overfitting."),
            ("GPU Parallel Acceleration",
             "Engineered across two NVIDIA GTX 580 GPUs with cross-channel grouping, unlocking large-scale visual representation learning.")
        ]
        self.layout_split_left_image(slide, "slides/assets/alexnet_architecture_diagram.png",
                                     "AlexNet Architecture: 8 Layers Split Across Dual-GPU Parallel Processing Pipelines",
                                     "The Modern Deep Learning Revolution", bullets, COLOR_PURPLE)
        self.add_footer(slide)

    def build_slide_20(self):
        """Slide 20: Milestone 4 — VGG-16 & VGG-19 (Simonyan & Zisserman, 2014)"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "Architectural Milestone: VGGNet (2014)",
                        "The Power of Simplicity: Standardizing Homogeneous 3x3 Convolution Stacks")

        bullets = [
            ("Filter Factorization",
             "Proved that two stacked 3x3 convolutions have an effective 5x5 receptive field, and three have 7x7, while saving 28% parameters and adding non-linearities."),
            ("Homogeneous Modular Design",
             "Replaced diverse kernel sizes with clean, uniform convolutional blocks: Conv(3x3) -> BatchNorm -> ReLU -> MaxPool(2x2)."),
            ("Deep Backbone Standard",
             "VGG-16 and VGG-19 established the standard deep feature extraction backbone across computer vision transfer learning.")
        ]
        self.layout_split_left_image(slide, "slides/assets/vgg_architecture_diagram.png",
                                     "VGG-16 Architecture: 5 Homogeneous 3x3 Convolutional Blocks + Fully-Connected Head",
                                     "The 3x3 Convolution Standard", bullets, COLOR_BLUE)
        self.add_footer(slide)

    def build_slide_21(self):
        """Slide 21: Milestone 5 — GoogLeNet / Inception v1 (Szegedy et al., 2014)"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "Architectural Milestone: GoogLeNet / Inception (2014)",
                        "Multi-Scale Feature Processing and 1x1 Bottleneck Dimensionality Reduction")

        bullets = [
            ("Multi-Scale Inception Modules",
             "Processes input features simultaneously at multiple spatial scales (1x1, 3x3, 5x5) and pooling in parallel, concatenating depth responses."),
            ("1x1 Bottleneck Convolutions",
             "Introduced 1x1 convolutions for channel-dimension reduction before expensive 3x3 and 5x5 filters, curbing computational complexity."),
            ("Global Average Pooling",
             "Replaced massive fully-connected layers with Global Average Pooling (GAP), dropping total model parameters from 138M (VGG) to just 5M.")
        ]
        self.layout_split_left_image(slide, "slides/assets/inception_module_diagram.png",
                                     "GoogLeNet Inception Module: Parallel Multi-Scale Convolutions + 1x1 Channel Bottlenecks",
                                     "Multi-Scale Efficiency", bullets, COLOR_GREEN)
        self.add_footer(slide)

    def build_slide_22(self):
        """Slide 22: Milestone 6 — ResNet (He et al., 2015)"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "Architectural Milestone: ResNet (2015)",
                        "Deep Residual Learning: Conquering the Degradation Ceiling with Identity Skip Highways")

        bullets = [
            ("The Degradation Dilemma",
             "Beyond ~20 layers, plain networks suffer accuracy degradation (higher training error), not caused by overfitting but vanishing gradients."),
            ("Identity Skip Highway",
             "Formulated residual learning: H(x) = F(x) + x. The identity connection ensures gradient flow dH/dx = dF/dx + 1 directly back to early layers."),
            ("Superhuman Accuracy",
             "Scaled to 152 layers, winning ImageNet 2015 with a record 3.57% top-5 error, surpassing human-level visual performance (5.1%).")
        ]
        self.layout_split_left_image(slide, "slides/assets/resnet_skip_connection.png",
                                     "ResNet Residual Block: Identity Connection H(x) = F(x) + x Providing Direct Gradient Flow",
                                     "Deep Residual Learning", bullets, COLOR_PURPLE)
        self.add_footer(slide)

    def build_slide_23(self):
        """Slide 23: Milestone 7 — DenseNet (Huang et al., 2017)"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "Architectural Milestone: DenseNet (2017)",
                        "Dense Connectivity & Feature Reuse: Concatenation Replaces Residual Addition")

        bullets = [
            ("Full Feed-Forward Connectivity",
             "Connects each layer to all subsequent layers: X_l = H_l([X_0, X_1, ..., X_{l-1}]), passing feature maps directly via concatenation."),
            ("Compact Growth Rate (k)",
             "Each layer contributes only k new feature channels (k=12 to 32), preventing redundant filter learning and shrinking parameters."),
            ("Direct Gradient Flow",
             "Eliminates vanishing gradients by ensuring early layers receive loss gradients directly from the loss function.")
        ]
        self.layout_split_left_image(slide, "slides/assets/densenet_architecture_diagram.png",
                                     "DenseNet Architecture: Dense Block Connectivity with k Growth Rate & Transition Layers",
                                     "Dense Feature Concatenation", bullets, COLOR_BLUE)
        self.add_footer(slide)

    def build_slide_24(self):
        """Slide 24: Milestone 8 — U-Net (Ronneberger et al., 2015)"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "Architectural Milestone: U-Net (2015)",
                        "Symmetric Encoder-Decoder Network for Dense Pixel-Level Semantic Segmentation")

        bullets = [
            ("Dense Prediction Paradigm",
             "Transitions from global classification (Image -> Class) to dense pixel-level labeling (Image -> Segmentation Mask)."),
            ("Contracting & Expanding Paths",
             "Encoder contracts spatial dimensions to capture context; symmetrical decoder upsamples feature maps to restore precise spatial resolution."),
            ("Direct Skip Copy Channels",
             "High-resolution skip connections directly copy spatial feature maps from encoder to decoder, preserving crisp object boundaries.")
        ]
        self.layout_split_left_image(slide, "slides/assets/unet_architecture_diagram.png",
                                     "U-Net Architecture: Contracting Encoder + Expanding Decoder + Horizontal Skip Connections",
                                     "Dense Semantic Segmentation", bullets, COLOR_GREEN)
        self.add_footer(slide)

    def build_slide_25(self):
        """Slide 25: Milestone 9 — MobileNet (Howard et al., 2017)"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "Architectural Milestone: MobileNet (2017)",
                        "Depthwise Separable Convolutions: 8x to 9x FLOP Reduction for Edge Devices")

        bullets = [
            ("Depthwise Factorization",
             "Splits standard 3D convolutions into two steps: Depthwise Conv (spatial filtering per channel) + Pointwise Conv (1x1 channel combination)."),
            ("Massive Computational Savings",
             "Reduces multiplication FLOPs by 1/N + 1/K^2 ≈ 8x to 9x compared to standard convolutions with only ~1% top-1 accuracy loss."),
            ("Real-Time Mobile Vision",
             "Enabled real-time high-accuracy deep computer vision on battery-powered mobile phones, embedded robotics, and edge IoT devices.")
        ]
        self.layout_split_left_image(slide, "slides/assets/mobilenet_depthwise_diagram.png",
                                     "MobileNet Factorization: Depthwise (Spatial) + Pointwise (Channel) = 88% FLOPs Reduction",
                                     "Edge & Mobile Efficiency", bullets, COLOR_ORANGE)
        self.add_footer(slide)

    def build_slide_26(self):
        """Slide 26: Milestone 10 — EfficientNet (Tan & Le, 2019)"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "Architectural Milestone: EfficientNet (2019)",
                        "Principled Compound Model Scaling Across Depth, Width, and Input Resolution")

        bullets = [
            ("Compound Scaling Principle",
             "Systematically balances Depth (alpha), Width (beta), and Resolution (gamma) via: alpha * beta^2 * gamma^2 ≈ 2 under a unified compound coefficient phi."),
            ("Eliminating Scaling Bottlenecks",
             "Proved that scaling only width, depth, or resolution quickly saturates; balanced co-scaling yields superior accuracy at lower FLOPs."),
            ("State-of-the-Art Efficiency",
             "EfficientNet-B7 reached 84.3% top-1 ImageNet accuracy while being 8.4x smaller and 6.1x faster than prior state-of-the-art networks.")
        ]
        self.layout_split_left_image(slide, "slides/assets/efficientnet_scaling_diagram.png",
                                     "EfficientNet Compound Scaling: Balanced Growth of Depth, Width, and Resolution",
                                     "Principled Model Scaling", bullets, COLOR_GREEN)
        self.add_footer(slide)

    def build_slide_27(self):
        """Slide 27: Milestone 11 — SENet / Squeeze-and-Excitation (Hu et al., 2018)"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "Architectural Milestone: SENet (2018)",
                        "Adaptive Channel Attention: Dynamic Channel Recalibration (Y = X ⊙ s)")

        bullets = [
            ("Squeeze Operation",
             "Global Average Pooling (GAP) aggregates global spatial context into a 1x1xC channel descriptor vector z."),
            ("Excitation Gating",
             "A two-layer bottleneck MLP (FC -> ReLU -> FC -> Sigmoid) models non-linear channel interdependencies with reduction ratio r=16."),
            ("Dynamic Recalibration",
             "Reweights each channel by scalar activation s_c in [0, 1], dynamically amplifying task-salient features while suppressing background noise.")
        ]
        self.layout_split_left_image(slide, "slides/assets/senet_se_block_diagram.png",
                                     "SENet Squeeze-and-Excitation Block: Global Squeeze + MLP Excitation + Channel Recalibration",
                                     "Channel-Wise Attention Gating", bullets, COLOR_BLUE)
        self.add_footer(slide)

    def build_slide_28(self):
        """Slide 28: Milestone 12 — CBAM Attention Module (Woo et al., 2018)"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "Architectural Milestone: CBAM (2018)",
                        "Convolutional Block Attention Module: Sequential Channel and Spatial Refinement")

        bullets = [
            ("Sequential Dual Attention",
             "Combines Channel Attention (learning 'What' features are important) followed by Spatial Attention (learning 'Where' to look)."),
            ("Dual Pooling Aggregation",
             "Utilizes both Average-Pooling (background features) and Max-Pooling (distinct object features) inside attention generation."),
            ("Plug-and-Play Integration",
             "Lightweight attention block with negligible parameter overhead, easily integrated into standard ResNet or MobileNet architectures.")
        ]
        self.layout_split_left_image(slide, "slides/assets/cbam_attention_diagram.png",
                                     "CBAM Sequential Attention: Channel Module (What) -> Spatial Module (Where) -> Refined Tensor",
                                     "Dual Sequential Attention", bullets, COLOR_PURPLE)
        self.add_footer(slide)

    def build_slide_29(self):
        """Slide 29: Milestone 13 — Vision Transformer / ViT (Dosovitskiy et al., 2020)"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "Architectural Milestone: Vision Transformer (2020)",
                        "An Image is Worth 16x16 Words: Discarding Convolutions for Global Self-Attention")

        bullets = [
            ("Patch Tokenization",
             "Unfolds a 224x224x3 image into 196 non-overlapping 16x16 patches, linearly projecting each into a latent 1D embedding token."),
            ("Pure Self-Attention",
             "Processes tokens via Multi-Head Self-Attention: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V, capturing global context across the entire image."),
            ("Massive Scale Dominance",
             "Eliminates spatial inductive priors in favor of high model capacity, outperforming CNNs when pretrained on massive datasets (JFT-300M).")
        ]
        self.layout_split_left_image(slide, "slides/assets/vit_architecture_diagram.png",
                                     "Vision Transformer (ViT): Patch Embedding -> Transformer Encoder -> Multi-Head Self-Attention",
                                     "Tokenized Visual Transformers", bullets, COLOR_PURPLE)
        self.add_footer(slide)

    def build_slide_30(self):
        """Slide 30: Paradigm Comparison — CNN vs. Vision Transformer"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "Architectural Paradigm: CNN vs. Vision Transformer",
                        "Comparing Inductive Priors, Computational Complexity, and Modern Frontiers")

        bullets = [
            ("Inductive Bias Dichotomy",
             "CNNs encode hard spatial priors (locality & shift equivariance), while ViTs learn relationships entirely from raw data with zero geometric assumptions."),
            ("Computational Asymmetry",
             "Convolutions scale linearly with pixel resolution O(K^2 H W), whereas standard self-attention scales quadratically with tokens O(N^2)."),
            ("Modern Convergence",
             "State-of-the-art vision models unite both worlds: ConvNeXt modernizes pure CNNs with ViT design, while MaxViT blends local convs with global attention.")
        ]
        self.layout_split_left_image(slide, "slides/assets/cnn_vs_transformer_comparison.png",
                                     "Inductive Bias & Tradeoff Matrix: Local Convolutions vs. Global Self-Attention",
                                     "Architectural Tradeoff Synthesis", bullets, COLOR_PRIMARY)
        self.add_footer(slide)

    def build_slide_31(self):
        """Slide 31: Canonical CNN Architecture Pipeline"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "Canonical CNN Architecture Pipeline",
                        "The Standard Topology from Raw Input Pixels to Class Logits")

        bullets = [
            ("Feature Extraction Backbone",
             "Alternating blocks of Convolution, ReLU activation, and Spatial Pooling progressively extract invariant feature representations."),
            ("Transition to Classification",
             "Global spatial pooling or flattening compresses high-dimensional feature volumes into a compact 1D latent representation."),
            ("Decision Head",
             "Fully-connected layers followed by Softmax compute categorical probability distributions over target classes.")
        ]
        self.layout_split_left_image(slide, "slides/assets/typical_cnn.png",
                                     "End-to-End CNN Pipeline: Input -> Feature Extractor Backbone -> Classifier Head",
                                     "Canonical Visual Pipeline", bullets, COLOR_BLUE)
        self.add_footer(slide)

    def build_slide_32(self):
        """Slide 32: Tensor Shape Contraction & Channel Expansion Dynamics"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "Tensor Contraction & Channel Expansion",
                        "Geometric Downsampling Paired with Semantic Capacity Growth")

        bullets = [
            ("Spatial Resolution Contraction",
             "Spatial dimensions contract geometrically: 224x224 -> 112x112 -> 56x56 -> 28x28 -> 7x7 via pooling and strided convolutions."),
            ("Channel Capacity Growth",
             "Channel counts expand inversely: 3 -> 64 -> 128 -> 256 -> 512, allocating greater capacity to rich semantic combinations."),
            ("Conservation of Information",
             "Local pixel redundancy is traded for abstract, translation-invariant semantic feature vectors.")
        ]
        self.layout_split_left_image(slide, "slides/assets/tensor_shape_dynamics.png",
                                     "Tensor Flow: Spatial Contraction [224 -> 7] & Channel Expansion [3 -> 512]",
                                     "Tensor Flow Dynamics", bullets, COLOR_PURPLE)
        self.add_footer(slide)

    def build_slide_33(self):
        """Slide 33: Computational Budget Allocation — FLOPs vs. Parameter Bottlenecks"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "Computational Asymmetry: FLOPs vs. Parameters",
                        "Convolutions Dominate Compute; Fully-Connected Layers Dominate Memory")

        bullets = [
            ("Compute Bottleneck (FLOPs)",
             "Convolutions consume >90% of total floating-point operations due to sliding window dot products across high-resolution grids."),
            ("Memory Bottleneck (Parameters)",
             "Fully-connected layers account for >90% of model weights due to dense pairwise matrix multiplication."),
            ("Modern Solution: Global Pooling",
             "Modern architectures replace dense layers with Global Average Pooling (GAP), eliminating millions of redundant parameters.")
        ]
        self.layout_split_left_image(slide, "slides/assets/flops_params_tradeoff.png",
                                     "Workload Asymmetry: Convolutions Consume Compute; Dense Layers Consume Memory",
                                     "System Resource Tradeoffs", bullets, COLOR_ORANGE)
        self.add_footer(slide)

    def build_slide_34(self):
        """Slide 34: Three Levels of Implementation Abstraction"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "Three Architectural Abstraction Paradigms",
                        "From Low-Level Vectorized Matrix Math to Declarative & Imperative Frameworks")

        cols = [
            ("LEVEL 1", "NumPy (From Scratch)",
             [("Vectorized im2col", "Flattens sliding windows into matrices for BLAS GEMM"),
              ("Manual Backprop", "Analytical col2im gradient redistribution"),
              ("Educational Value", "100% transparency of internal tensor mechanics")],
             COLOR_AMBER, None),

            ("LEVEL 2", "Keras (Declarative)",
             [("Declarative Graph", "High-level layers.Conv2D and layers.MaxPooling2D"),
              ("Automated Graph", "Automated forward/backward graph compilation"),
              ("Rapid Prototyping", "Streamlined workflow for rapid model experimentation")],
             COLOR_BLUE, None),

            ("LEVEL 3", "PyTorch (Imperative)",
             [("Dynamic Autograd", "Imperative forward execution with automatic C++ differentiation"),
              ("Pythonic Debugging", "Direct tensor inspection and custom hook instrumentation"),
              ("Production Standard", "The dominant framework in modern deep learning research")],
             COLOR_GREEN, None)
        ]
        self.layout_triptych(slide, cols)
        self.add_footer(slide)

    def build_slide_35(self):
        """Slide 35: Cross-Framework Architectural Rosetta Stone (Native Table with Large Fonts)"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "Cross-Framework Component Rosetta Stone",
                        "Direct Structural Translation Across NumPy (Scratch), Keras (Declarative), and PyTorch (Imperative)")

        headers = ["Primitive Concept", "NumPy (From Scratch)", "Keras (Declarative Graph)", "PyTorch (Imperative Autograd)"]
        rows = [
            ["2D Convolution",
             "Y_col = np.dot(W_row, X_col)",
             "layers.Conv2D(32, (3, 3),\n              padding='same')",
             "nn.Conv2d(in_ch, out_ch,\n          kernel_size=3)"],
            ["Non-Linearity",
             "a = np.maximum(0, z)  # ReLU",
             "layers.Activation('relu') /\nlayers.ReLU()",
             "torch.relu(z) /\nnn.ReLU()"],
            ["Spatial Pooling",
             "max_pool2d_scratch(\n    X, pool_size=2)",
             "layers.MaxPooling2D(\n    pool_size=(2, 2))",
             "nn.MaxPool2d(kernel_size=2,\n             stride=2)"],
            ["Batch Normalization",
             "gamma * (x - mu) /\nnp.sqrt(var + eps) + beta",
             "layers.BatchNormalization()",
             "nn.BatchNorm2d(\n    num_features)"],
            ["Backward Pass",
             "dX_col = np.dot(\n    W_row.T, dY_col)",
             "tape.gradient(\n    loss, model.trainable_vars)",
             "loss.backward()\n# C++ Autograd DAG"],
            ["Weight Update",
             "W -= learning_rate * dW",
             "optimizer.apply_gradients(\n    zip(grads, vars))",
             "optimizer.step()"]
        ]
        widths = [Inches(1.85), Inches(3.35), Inches(3.50), Inches(3.513)]
        self.layout_native_rosetta_table(slide, headers, rows, widths)
        self.add_footer(slide)

    # -------------------------------------------------------------------------
    # PART IV: CODE DEMO & VERIFICATION (Slides 36 - 39)
    # -------------------------------------------------------------------------

    def build_slide_36(self):
        """Slide 36: Vectorized Forward Pass — im2col + GEMM Matrix Multiplication"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "Vectorized Forward Pass: im2col + BLAS GEMM",
                        "Transforming Spatial Sliding Windows into High-Performance Matrix Multiplication")

        code = [
            "# High-Performance Vectorized Convolution via im2col",
            "def conv_forward_im2col(x, w, b, stride=1, pad=1):",
            "    N, C, H, W = x.shape",
            "    F, _, HH, WW = w.shape",
            "    H_out = (H - HH + 2 * pad) // stride + 1",
            "    W_out = (W - WW + 2 * pad) // stride + 1",
            "    ",
            "    # 1. Unfold sliding receptive patches into 2D column matrix",
            "    x_col = im2col_indices(x, HH, WW, padding=pad, stride=stride)",
            "    w_row = w.reshape(F, -1)",
            "    ",
            "    # 2. Compute convolution via single optimized BLAS GEMM dot product",
            "    out = np.dot(w_row, x_col) + b.reshape(-1, 1)",
            "    ",
            "    # 3. Reshape back into output tensor volume",
            "    out = out.reshape(F, H_out, W_out, N).transpose(3, 0, 1, 2)",
            "    return out, (x, w, b, stride, pad, x_col)"
        ]
        self.add_terminal(slide, Inches(0.8), Inches(1.85), Inches(6.8), Inches(4.85), "conv_forward_im2col.py", code)
        self.add_image_panel(slide, Inches(7.85), Inches(1.85), Inches(4.683), Inches(4.85),
                             "slides/assets/im2col_diagram.png",
                             caption="im2col Mechanics: Receptive Patches Unfolded into Matrix Columns for BLAS GEMM")
        self.add_footer(slide)

    def build_slide_37(self):
        """Slide 37: Analytical Backward Pass — col2im & Gradient Flow"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "Analytical Convolution Backward Pass (col2im)",
                        "Accumulating Backpropagated Gradients Across Overlapping Receptive Fields")

        code = [
            "# Analytical Gradient Backpropagation via col2im",
            "def conv_backward_im2col(dout, cache):",
            "    x, w, b, stride, pad, x_col = cache",
            "    F, C, HH, WW = w.shape",
            "    ",
            "    # 1. Bias gradient: sum across batch and spatial positions",
            "    db = np.sum(dout, axis=(0, 2, 3))",
            "    ",
            "    # 2. Reshape upstream gradient for matrix multiplication",
            "    dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(F, -1)",
            "    ",
            "    # 3. Weight gradient via BLAS matrix product",
            "    dw = np.dot(dout_reshaped, x_col.T).reshape(w.shape)",
            "    ",
            "    # 4. Input gradient: matrix product followed by col2im folding",
            "    w_row = w.reshape(F, -1)",
            "    dx_col = np.dot(w_row.T, dout_reshaped)",
            "    dx = col2im_indices(dx_col, x.shape, HH, WW, padding=pad, stride=stride)",
            "    return dx, dw, db"
        ]
        self.add_terminal(slide, Inches(0.8), Inches(1.85), Inches(6.8), Inches(4.85), "conv_backward_im2col.py", code)
        self.add_image_panel(slide, Inches(7.85), Inches(1.85), Inches(4.683), Inches(4.85),
                             "slides/assets/backprop_convolution_math.png",
                             caption="Analytical col2im: Backpropagating and Summing Gradients Over Overlapping Patches")
        self.add_footer(slide)

    def build_slide_38(self):
        """Slide 38: Production CNN Architecture in PyTorch (nn.Module)"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "Production PyTorch Implementation (nn.Module)",
                        "Elegant Object-Oriented Deep Convolutional Network Pipeline")

        code = [
            "import torch",
            "import torch.nn as nn",
            "",
            "class ModernCNN(nn.Module):",
            "    def __init__(self, num_classes=10):",
            "        super(ModernCNN, self).__init__()",
            "        self.features = nn.Sequential(",
            "            # Stage 1: Spatial feature extraction",
            "            nn.Conv2d(3, 32, kernel_size=3, padding=1),",
            "            nn.BatchNorm2d(32),",
            "            nn.ReLU(inplace=True),",
            "            nn.MaxPool2d(kernel_size=2, stride=2),   # [B, 32, 16, 16]",
            "            # Stage 2: Deep feature abstraction",
            "            nn.Conv2d(32, 64, kernel_size=3, padding=1),",
            "            nn.BatchNorm2d(64),",
            "            nn.ReLU(inplace=True),",
            "            nn.MaxPool2d(kernel_size=2, stride=2)    # [B, 64, 8, 8]",
            "        )",
            "        self.classifier = nn.Sequential(",
            "            nn.Flatten(),",
            "            nn.Linear(64 * 8 * 8, 128),",
            "            nn.ReLU(inplace=True),",
            "            nn.Dropout(0.5),",
            "            nn.Linear(128, num_classes)",
            "        )",
            "",
            "    def forward(self, x):",
            "        x = self.features(x)       # Extract spatial representation",
            "        logits = self.classifier(x) # Compute multi-class class scores",
            "        return logits"
        ]
        self.add_terminal(slide, Inches(0.8), Inches(1.85), Inches(7.5), Inches(5.05), "modern_cnn_pytorch.py", code, code_font_size=Pt(9.5))

        b_box, tf_box = self.add_bento_box(slide, Inches(8.55), Inches(1.85), Inches(3.983), Inches(5.05),
                                           title="PyTorch Advantages", title_color=COLOR_GREEN)
        pt_bullets = [
            ("Dynamic Computation Graph",
             "Graphs are constructed on-the-fly during forward execution, enabling arbitrary control flow and simple debugging."),
            ("Production BatchNorm",
             "Tracks running mini-batch statistics during training and applies deterministic population statistics during evaluation."),
            ("Automatic Differentiation",
             "Engine automatically traces tensor dependencies, generating the exact backward DAG without manual derivative code.")
        ]
        self.add_large_bullets(tf_box, pt_bullets, font_size=Pt(12.5))
        self.add_footer(slide)

    def build_slide_39(self):
        """Slide 39: Empirical Verification & Real-Time Inference Dashboard"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "Empirical Verification & Inference Dashboard",
                        "From Learned Convolutional Weight Filters to Real-Time Predictive Consensus")

        bullets = [
            ("Learned Filter Convergence",
             "Trained 3x3 kernels autonomously converge to oriented edge detectors, color contrasts, and texture passes without manual supervision."),
            ("Real-Time Prediction Consensus",
             "Live single-sample forward pass achieves >96% classification confidence, with consensus across NumPy, Keras, and PyTorch."),
            ("Foundational Inductive Bias",
             "Convolutions enforce spatial locality and weight sharing, cementing CNNs as the foundational bedrock of modern visual intelligence.")
        ]
        self.layout_split_dual_images(slide, "slides/assets/learned_filters_fmnist.png",
                                      "32 Learned 3x3 Convolutional Filter Tensors (Autonomous Edge & Pattern Extraction)",
                                      "slides/assets/live_inference_visual.png",
                                      "Live Inference Dashboard: Multi-Model Single-Sample Classification Consensus",
                                      "Scientific Synthesis", bullets, COLOR_PRIMARY)
        self.add_footer(slide)

    # -------------------------------------------------------------------------
    # Master Build Method
    # -------------------------------------------------------------------------

    def build_all(self):
        print("Starting CNN Core Presentation build (39 slides)...")
        self.build_slide_01()
        self.build_slide_02()
        self.build_slide_03()
        self.build_slide_04()
        self.build_slide_05()
        self.build_slide_06()
        self.build_slide_07()
        self.build_slide_08()
        self.build_slide_09()
        self.build_slide_10()
        self.build_slide_11()
        self.build_slide_12()
        self.build_slide_13()
        self.build_slide_14()
        self.build_slide_15()
        self.build_slide_16()
        self.build_slide_17()
        self.build_slide_18()
        self.build_slide_19()
        self.build_slide_20()
        self.build_slide_21()
        self.build_slide_22()
        self.build_slide_23()
        self.build_slide_24()
        self.build_slide_25()
        self.build_slide_26()
        self.build_slide_27()
        self.build_slide_28()
        self.build_slide_29()
        self.build_slide_30()
        self.build_slide_31()
        self.build_slide_32()
        self.build_slide_33()
        self.build_slide_34()
        self.build_slide_35()
        self.build_slide_36()
        self.build_slide_37()
        self.build_slide_38()
        self.build_slide_39()

        os.makedirs(os.path.dirname(self.output_pptx), exist_ok=True)
        self.prs.save(self.output_pptx)
        print(f"Successfully generated {self.current_slide_num} slides at: {self.output_pptx}")


if __name__ == "__main__":
    builder = CNNPresentationBuilder()
    builder.build_all()
