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
        script_dir = os.path.dirname(os.path.abspath(__file__))
        assignment_dir = os.path.dirname(script_dir)
        os.chdir(assignment_dir)
        self.output_pptx = output_pptx
        self.prs = Presentation()
        self.prs.slide_width = Inches(13.333)
        self.prs.slide_height = Inches(7.5)
        self.blank_layout = self.prs.slide_layouts[6]
        self.total_slides = 45
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
        """Layout: 3-Column Bento Grid with high legibility."""
        card_w = Inches(3.70)
        gap = Inches(0.31)
        for i, col in enumerate(cols_data):
            c_badge, c_title, c_bullets, c_col = col[0], col[1], col[2], col[3]
            c_img = col[4] if len(col) > 4 else None
            left_p = Inches(0.8) + i * (card_w + gap)
            c, tf = self.add_bento_box(slide, left_p, Inches(1.85), card_w, Inches(4.85))
            
            p0 = tf.paragraphs[0]
            p0.text = c_badge.upper()
            p0.font.name = FONT_HEAD
            p0.font.size = Pt(12)
            p0.font.bold = True
            p0.font.color.rgb = c_col
            p0.space_after = Pt(4)

            p1 = tf.add_paragraph()
            p1.text = c_title
            p1.font.name = FONT_HEAD
            p1.font.size = Pt(17)
            p1.font.bold = True
            p1.font.color.rgb = TEXT_MAIN
            p1.space_after = Pt(10)

            bullet_font_size = Pt(11) if c_img else Pt(13)
            bullet_space_after = Pt(5) if c_img else Pt(10)

            for b_title, b_desc in c_bullets:
                p_b = tf.add_paragraph()
                p_b.space_after = bullet_space_after
                r1 = p_b.add_run()
                r1.text = f"• {b_title}: "
                r1.font.name = FONT_HEAD
                r1.font.size = bullet_font_size
                r1.font.bold = True
                r1.font.color.rgb = c_col

                r2 = p_b.add_run()
                r2.text = b_desc
                r2.font.name = FONT_BODY
                r2.font.size = bullet_font_size
                r2.font.color.rgb = TEXT_BODY

            if c_img and os.path.exists(c_img):
                img_top = Inches(4.15)
                img_h = Inches(2.35)
                img_w = card_w - Inches(0.36)
                img_left = left_p + Inches(0.18)
                self.add_image_panel(slide, img_left, img_top, img_w, img_h, c_img, border=True)

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
        p_sub.text = "Foundations, Architectural Evolution, and Implementation"
        p_sub.font.name = FONT_BODY
        p_sub.font.size = Pt(16)
        p_sub.font.color.rgb = TEXT_MUTED

        self.add_image_panel(slide, Inches(1.2), Inches(3.30), Inches(10.933), Inches(1.95),
                             "slides/assets/typical_cnn.png",
                             caption="A Typical End-to-End CNN Pipeline")

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
        """Slide 2: Structural Overview (2x2 Visual Grid, No List)"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "Overview", "An Overview of the Four Core Sections")

        pillars = [
            ("SECTION 01", "Overview & Vision Scope", "Spatial Geometry & Foundations",
             "Why 2D image structure gets lost when flattened into vectors.",
             COLOR_PRIMARY, "slides/assets/flattening_unrolling.png"),

            ("SECTION 02", "Concept & Mechanics", "Mathematical Foundations",
             "Layer composition, convolution math, stride/padding, ReLU.",
             COLOR_BLUE, "slides/assets/2D_convolution_frame_mid.png"),

            ("SECTION 03", "Model Evolution", "Historical Milestones & Modern Frontiers",
             "From Neocognitron and LeNet to ResNet, MobileNet, and ViT.",
             COLOR_PURPLE, "slides/assets/vgg_architecture_diagram.png"),

            ("SECTION 04", "Code Demo & Verification", "Vectorized Implementation",
             "im2col forward pass, col2im backprop, and a PyTorch model.",
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
        self.add_header(slide, "2D Structure vs. Vector Unrolling",
                        "Limitations of MLPs on Image Data")

        bullets = [
            ("Spatial Adjacency Destruction",
             "Flattening a 2D image into a 1D vector breaks pixel adjacency."),
            ("Loss of Translation Equivariance",
             "MLPs must relearn weights for every position if an object shifts."),
            ("Coordinate Independence",
             "Convolutions keep the 2D grid, applying the same kernel everywhere.")
        ]
        self.layout_split_right_image(slide, "Why Flattening Breaks Images", bullets,
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
        self.add_header(slide, "Networks as Function Composition",
                        "Layer-wise Non-Linear Transformations")

        bullets = [
            ("Composite Mapping",
             "A network is a composite function F(X;Θ) = f_L ∘ ... ∘ f_1(X)."),
            ("Layer Transformation",
             "Each layer applies an affine map plus non-linearity: f_l(h) = σ(W_l h + b_l)."),
            ("Separable Latent Space",
             "Layers gradually reshape pixels into a linearly separable space.")
        ]
        self.layout_split_left_image(slide, "slides/assets/cnn_hierarchy_concept.png",
                                     "Hierarchical Manifold Untangling: Low-Level Edges -> Mid Motifs -> Class Concepts",
                                     "How Layers Compose", bullets, COLOR_BLUE)
        self.add_footer(slide)

    def build_slide_05(self):
        """Slide 5: Affine Transformations & The Linear Collapse Proof"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "Why Non-Linearity Is Mandatory",
                        "Stacked Linear Layers Yield No Additional Capacity")

        bullets = [
            ("The Linear Collapse Theorem",
             "With linear activations, F(X) collapses to one effective linear map."),
            ("Zero Depth Advantage",
             "A 100-layer linear network is mathematically just linear regression."),
            ("Manifold Folding via ReLU",
             "Non-linear activations fold space into piecewise-linear decision regions.")
        ]
        self.layout_split_right_image(slide, "The Linear Collapse Proof", bullets,
                                      "slides/assets/relu_function_wiki.png",
                                      "Non-Linear Activation Function f(x) = max(0, x) Partitions the Space into Piecewise Linear Regions")
        self.add_footer(slide)

    def build_slide_06(self):
        """Slide 6: The MLP Failure Mode on High-Dimensional Grids"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "The MLP Parameter Explosion",
                        "Scalability Limits of Dense Layers on Image Data")

        bullets = [
            ("Combinatorial Weight Explosion",
             "A 224×224×3 image to 1,024 hidden units needs >154M weights."),
            ("Overfitting Risk",
             "Huge parameter counts cause overfitting and exhaust GPU memory."),
            ("Convolutional Efficiency",
             "A 3×3×3 conv filter needs only 28 parameters, at any resolution.")
        ]
        self.layout_split_left_image(slide, "slides/assets/mlp_vs_cnn_parameters.png",
                                     "Parameter Comparison: Fully-Connected (154M Params) vs. Convolutional (28 Params)",
                                     "Where Dense Layers Break Down", bullets, COLOR_ALERT)
        self.add_footer(slide)

    def build_slide_07(self):
        """Slide 7: Biological Genesis — Hubel & Wiesel (1959)"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "Hubel & Wiesel (1959)",
                        "Receptive Fields in the Visual Cortex")

        bullets = [
            ("Localized Simple Cells",
             "V1 neurons fire for localized oriented edges, not full-field light."),
            ("Orientation Tuning",
             "Cortical cells respond maximally to specific edge angles (0°, 45°, 90°)."),
            ("Hierarchical Processing",
             "Complex cells pool simple-cell outputs, inspiring CNN kernel hierarchies.")
        ]
        self.layout_split_dual_images(slide, "slides/assets/hubel_wiesel_receptive_field.png",
                                      "Receptive Field Biological Mechanism (Hubel & Wiesel, 1959)",
                                      "slides/assets/orientation_tuning_curve.png",
                                      "Biological Orientation Tuning Curves in V1 Cortical Cells",
                                      "Biological Origins", bullets, COLOR_PRIMARY)
        self.add_footer(slide)

    def build_slide_08(self):
        """Slide 8: 2D Convolution Mechanics — Cross-Correlation Kernel Sliding"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "2D Convolution Mechanics",
                        "Sliding a Kernel Across the Image")

        bullets = [
            ("Mathematical Formulation",
             "Cross-correlation: S(i,j) = ΣΣ I(i+m,j+n)K(m,n) + b."),
            ("Kernel Dot Product",
             "A K×K kernel slides across the input, computing a dot product per step."),
            ("Feature Extraction",
             "Filters act as feature detectors, activating on matching patterns.")
        ]
        self.layout_split_left_image(slide, "slides/assets/2D_convolution_frame_mid.png",
                                     "Sliding Window Cross-Correlation: Kernel Receptive Field Dot Product",
                                     "Sliding Window Mechanics", bullets, COLOR_BLUE)
        self.add_footer(slide)

    def build_slide_09(self):
        """Slide 9: The Two Core CNN Axioms — Local Connectivity & Weight Sharing"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "Two Core CNN Axioms",
                        "Inductive Biases That Cut Parameters")

        bullets = [
            ("Axiom 1: Local Connectivity",
             "Each neuron connects only to a local K×K patch, not the full image."),
            ("Axiom 2: Weight Sharing",
             "The same filter weights are reused everywhere, giving translation equivariance."),
            ("Statistical Efficiency",
             "The network learns a small set of shared feature detectors, not millions of weights.")
        ]
        self.layout_split_right_image(slide, "Core Inductive Biases", bullets,
                                      "slides/assets/equivariance_vs_invariance.png",
                                      "Translation Equivariance: Shifting Input Shifts Output Maps Identically")
        self.add_footer(slide)

    def build_slide_10(self):
        """Slide 10: Spatial Hyperparameters — Stride & Padding"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "Stride & Padding",
                        "Controlling Output Size")

        bullets = [
            ("Output Dimension Equation",
             "Output size: H_out = ⌊(H_in − K + 2P) / S⌋ + 1."),
            ("Padding Modes",
             "Valid padding (P=0) shrinks output; same padding keeps size unchanged."),
            ("Stride Subsampling",
             "Stride > 1 skips pixels, shrinking output and growing the receptive field.")
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
        self.add_header(slide, "Receptive Field Growth",
                        "Expanding Coverage by Stacking Small Filters")

        bullets = [
            ("Recursive RF Formula",
             "Receptive field grows recursively: RF_l = RF_{l-1} + (K_l−1)·Π S_i."),
            ("The VGG Insight",
             "Two stacked 3×3 convs match a 5×5 receptive field with 28% fewer params."),
            ("Hierarchical Context",
             "Deep layers capture global context without needing huge kernels.")
        ]
        self.layout_split_left_image(slide, "slides/assets/receptive_field_expansion.png",
                                     "Receptive Field Arithmetic: Stacking 3x3 Kernels Yields 5x5 and 7x7 Context",
                                     "Receptive Field Dynamics", bullets, COLOR_PURPLE)
        self.add_footer(slide)

    def build_slide_12(self):
        """Slide 12: Volumetric 3D Tensor Convolution"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "3D Tensor Convolutions",
                        "Convolving Across Channels and Space")

        bullets = [
            ("Tensor Volume Shape",
             "Input (C_in×H×W) is processed by C_out filters, each shaped C_in×K_h×K_w."),
            ("Channel-Wise Summation",
             "Each filter convolves all input channels, summing into one output map."),
            ("Channel Expansion",
             "Stacking C_out responses gives the output volume (C_out×H_out×W_out).")
        ]
        self.layout_split_left_image(slide, "slides/assets/cs231n_convnet.jpeg",
                                     "Stanford CS231n Volumetric Convolution: C_in x H x W Convolved into C_out Feature Maps",
                                     "Multichannel Tensor Flow", bullets, COLOR_BLUE)
        self.add_footer(slide)

    def build_slide_13(self):
        """Slide 13: Convolutional Parameter Counting Formulation"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "Counting Convolution Parameters",
                        "A Formula Independent of Image Size")

        bullets = [
            ("Universal Parameter Equation",
             "Params = (K_h·K_w·C_in + 1) · C_out, including one bias per filter."),
            ("Resolution Independence",
             "Parameter count depends only on kernel size and channels, not image size."),
            ("Concrete Example",
             "Example: K=3, C_in=64, C_out=128 gives 73,856 parameters.")
        ]
        self.layout_split_left_image(slide, "slides/assets/parameter_formula_breakdown.png",
                                     "Analytical Parameter Breakdown: (Kernel Area x Input Channels + Bias) x Output Channels",
                                     "The Parameter Formula", bullets, COLOR_GREEN)
        self.add_footer(slide)

    def build_slide_14(self):
        """Slide 14: Non-Linear Activation Functions — ReLU Manifold Sparsity"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "ReLU and Sparsity",
                        "How ReLU Preserves Gradient Flow")

        bullets = [
            ("The ReLU Definition",
             "f(x) = max(0, x), with derivative 1 for x>0 and 0 for x<0."),
            ("Vanishing Gradient Cure",
             "Unlike Sigmoid/Tanh, ReLU keeps a constant gradient, enabling 100+ layer training."),
            ("Induced Sparsity",
             "Zeroing negative values creates sparse activations (~50% active neurons).")
        ]
        self.layout_split_right_image(slide, "ReLU Activation Dynamics", bullets,
                                      "slides/assets/cs231n_act1.jpeg",
                                      "Visualizing Spatial Activation Maps After Non-Linear ReLU Filtering")
        self.add_footer(slide)

    def build_slide_15(self):
        """Slide 15: Spatial Downsampling — Max Pooling Mechanics"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "Max Pooling",
                        "Downsampling and Translation Invariance")

        bullets = [
            ("Max Pooling Operation",
             "A 2×2 window with stride 2 keeps the max value, dropping 75% of pixels."),
            ("Translation Invariance",
             "Small shifts in the input rarely change the pooled max, aiding robustness."),
            ("Zero Learnable Parameters",
             "Pooling has zero learnable parameters, cutting memory and compute.")
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
                        "From Edges to Semantic Concepts")

        bullets = [
            ("Shallow Layers (V1)",
             "Early filters learn oriented edges, color contrasts, and boundaries."),
            ("Intermediate Layers",
             "Mid layers combine edges into textures, corners, and contours."),
            ("Deep Layers",
             "Deep layers detect whole-object parts, like wheels, eyes, and faces.")
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
        self.add_header(slide, "Neocognitron (1980)",
                        "A Biologically Inspired Precursor to CNNs")

        bullets = [
            ("S-Cells & C-Cells",
             "S-cells extract oriented features; C-cells pool them for shift tolerance."),
            ("Self-Organizing Architecture",
             "Introduced layered visual processing and the conv-pool pattern, pre-backprop."),
            ("Translation Invariance",
             "First neural model to recognize shifted or deformed patterns via receptive fields.")
        ]
        self.layout_split_left_image(slide, "slides/assets/neocognitron_ScholarFig1.png",
                                     "Neocognitron (Fukushima, 1980): Hierarchical Alternation of S-Layers and C-Layers",
                                     "The Computational Ancestor", bullets, COLOR_PRIMARY)
        self.add_footer(slide)

    def build_slide_18(self):
        """Slide 18: Milestone 2 — LeNet-5 (LeCun, 1998)"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "LeNet-5 (1998)",
                        "Gradient-Based Digit Recognition")

        bullets = [
            ("End-to-End Gradient Optimization",
             "Trained feature extraction and classification jointly via backpropagation."),
            ("Shift and Distortion Robustness",
             "Replaced hand-crafted features with shared convolutional filters."),
            ("Commercial Validation",
             "Deployed by US banks, reading over 10% of checks in the late 1990s.")
        ]
        self.layout_split_left_image(slide, "slides/assets/lenet5_architecture_diagram.png",
                                     "LeNet-5 Architecture: 7 Layers of Alternating Convolutions and Subsampling",
                                     "The First Modern CNN", bullets, COLOR_BLUE)
        self.add_footer(slide)

    def build_slide_19(self):
        """Slide 19: Milestone 3 — AlexNet (Krizhevsky et al., 2012)"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "AlexNet (2012)",
                        "The ImageNet Milestone")

        bullets = [
            ("Historical Breakthrough",
             "Won ImageNet 2012 with 16.4% top-5 error vs. 26.2% for runner-up."),
            ("Modern Technical Suite",
             "Combined ReLU, Dropout (0.5), and data augmentation to curb overfitting."),
            ("GPU Parallel Acceleration",
             "Trained across two GTX 580 GPUs, enabling large-scale representation learning.")
        ]
        self.layout_split_left_image(slide, "slides/assets/alexnet_architecture_diagram.png",
                                     "AlexNet Architecture: 8 Layers Split Across Dual-GPU Parallel Processing Pipelines",
                                     "The Deep Learning Turning Point", bullets, COLOR_PURPLE)
        self.add_footer(slide)

    def build_slide_20(self):
        """Slide 20: Milestone 4 — VGG-16 & VGG-19 (Simonyan & Zisserman, 2014)"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "VGGNet (2014)",
                        "Standardizing 3×3 Convolution Stacks")

        bullets = [
            ("Filter Factorization",
             "Two stacked 3×3 convs match a 5×5 receptive field, saving 28% params."),
            ("Homogeneous Modular Design",
             "Used uniform blocks: Conv(3×3) → BatchNorm → ReLU → MaxPool(2×2)."),
            ("Deep Backbone Standard",
             "VGG-16/19 became the standard backbone for transfer learning.")
        ]
        self.layout_split_left_image(slide, "slides/assets/vgg_architecture_diagram.png",
                                     "VGG-16 Architecture: 5 Homogeneous 3x3 Convolutional Blocks + Fully-Connected Head",
                                     "The 3x3 Convolution Standard", bullets, COLOR_BLUE)
        self.add_footer(slide)

    def build_slide_21(self):
        """Slide 21: Milestone 5 — GoogLeNet / Inception v1 (Szegedy et al., 2014)"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "GoogLeNet / Inception (2014)",
                        "Multi-Scale Features, 1×1 Bottlenecks")

        bullets = [
            ("Multi-Scale Inception Modules",
             "Processes inputs at 1×1, 3×3, and 5×5 scales in parallel, then concatenates."),
            ("1x1 Bottleneck Convolutions",
             "1×1 convolutions reduce channels before costly 3×3/5×5 filters."),
            ("Global Average Pooling",
             "Global Average Pooling cut parameters from 138M (VGG) to 5M.")
        ]
        self.layout_split_left_image(slide, "slides/assets/inception_module_diagram.png",
                                     "GoogLeNet Inception Module: Parallel Multi-Scale Convolutions + 1x1 Channel Bottlenecks",
                                     "Multi-Scale Efficiency", bullets, COLOR_GREEN)
        self.add_footer(slide)

    def build_slide_22(self):
        """Slide 22: Milestone 6 — ResNet (He et al., 2015)"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "ResNet (2015)",
                        "Residual Learning With Skip Connections")

        bullets = [
            ("The Degradation Problem",
             "Past ~20 layers, plain networks lose accuracy from vanishing gradients."),
            ("Identity Skip Highway",
             "Residual form H(x) = F(x) + x lets gradients flow directly to early layers."),
            ("Superhuman Accuracy",
             "Scaled to 152 layers, winning ImageNet 2015 with 3.57% top-5 error.")
        ]
        self.layout_split_left_image(slide, "slides/assets/resnet_skip_connection.png",
                                     "ResNet Residual Block: Identity Connection H(x) = F(x) + x Providing Direct Gradient Flow",
                                     "Deep Residual Learning", bullets, COLOR_PURPLE)
        self.add_footer(slide)

    def build_slide_23(self):
        """Slide 23: Milestone 7 — DenseNet (Huang et al., 2017)"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "DenseNet (2017)",
                        "Dense Connectivity via Concatenation")

        bullets = [
            ("Full Feed-Forward Connectivity",
             "Each layer connects to all later layers via feature-map concatenation."),
            ("Compact Growth Rate (k)",
             "Each layer adds only k new channels (k=12–32), keeping params low."),
            ("Direct Gradient Flow",
             "Early layers get gradients directly from the loss, avoiding vanishing gradients.")
        ]
        self.layout_split_left_image(slide, "slides/assets/densenet_architecture_diagram.png",
                                     "DenseNet Architecture: Dense Block Connectivity with k Growth Rate & Transition Layers",
                                     "Dense Feature Concatenation", bullets, COLOR_BLUE)
        self.add_footer(slide)

    def build_slide_24(self):
        """Slide 24: Milestone 8 — U-Net (Ronneberger et al., 2015)"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "U-Net (2015)",
                        "Encoder-Decoder for Segmentation")

        bullets = [
            ("Dense Prediction Paradigm",
             "Shifts from image classification to pixel-level segmentation masks."),
            ("Contracting & Expanding Paths",
             "Encoder contracts for context; decoder upsamples to restore resolution."),
            ("Direct Skip Copy Channels",
             "Skip connections copy encoder features to the decoder, preserving edges.")
        ]
        self.layout_split_left_image(slide, "slides/assets/unet_architecture_diagram.png",
                                     "U-Net Architecture: Contracting Encoder + Expanding Decoder + Horizontal Skip Connections",
                                     "Dense Semantic Segmentation", bullets, COLOR_GREEN)
        self.add_footer(slide)

    def build_slide_25(self):
        """Slide 25: Milestone 9 — MobileNet (Howard et al., 2017)"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "MobileNet (2017)",
                        "Depthwise Separable Convolutions")

        bullets = [
            ("Depthwise Factorization",
             "Splits convolution into depthwise (spatial) and pointwise (1×1) steps."),
            ("Massive Computational Savings",
             "Cuts FLOPs roughly 8–9x versus standard convolutions, ~1% accuracy loss."),
            ("Real-Time Mobile Vision",
             "Enabled real-time vision on phones, embedded robotics, and edge devices.")
        ]
        self.layout_split_left_image(slide, "slides/assets/mobilenet_depthwise_diagram.png",
                                     "MobileNet Factorization: Depthwise (Spatial) + Pointwise (Channel) = 88% FLOPs Reduction",
                                     "Edge & Mobile Efficiency", bullets, COLOR_ORANGE)
        self.add_footer(slide)

    def build_slide_26(self):
        """Slide 26: Milestone 10 — EfficientNet (Tan & Le, 2019)"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "EfficientNet (2019)",
                        "Compound Scaling of Depth, Width, Resolution")

        bullets = [
            ("Compound Scaling Principle",
             "Balances depth, width, and resolution via α·β²·γ² ≈ 2."),
            ("Balanced Scaling Wins",
             "Scaling only one dimension saturates; balanced scaling wins on accuracy."),
            ("State-of-the-Art Efficiency",
             "EfficientNet-B7 hit 84.3% top-1 accuracy, 8.4x smaller and 6.1x faster.")
        ]
        self.layout_split_left_image(slide, "slides/assets/efficientnet_scaling_diagram.png",
                                     "EfficientNet Compound Scaling: Balanced Growth of Depth, Width, and Resolution",
                                     "Principled Model Scaling", bullets, COLOR_GREEN)
        self.add_footer(slide)

    def build_slide_27(self):
        """Slide 27: Milestone 11 — SENet / Squeeze-and-Excitation (Hu et al., 2018)"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "SENet (2018)",
                        "Dynamic Channel Attention (Y = X ⊙ s)")

        bullets = [
            ("Squeeze Operation",
             "Global Average Pooling compresses each channel into one descriptor."),
            ("Excitation Gating",
             "A small MLP (FC→ReLU→FC→Sigmoid) learns channel importance, r=16."),
            ("Dynamic Recalibration",
             "Reweights each channel by a learned score, boosting useful features.")
        ]
        self.layout_split_left_image(slide, "slides/assets/senet_se_block_diagram.png",
                                     "SENet Squeeze-and-Excitation Block: Global Squeeze + MLP Excitation + Channel Recalibration",
                                     "Channel-Wise Attention Gating", bullets, COLOR_BLUE)
        self.add_footer(slide)

    def build_slide_28(self):
        """Slide 28: Milestone 12 — CBAM Attention Module (Woo et al., 2018)"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "CBAM (2018)",
                        "Sequential Channel and Spatial Attention")

        bullets = [
            ("Sequential Dual Attention",
             "Channel attention (what matters) followed by spatial attention (where)."),
            ("Dual Pooling Aggregation",
             "Combines average-pooling and max-pooling to generate attention maps."),
            ("Plug-and-Play Integration",
             "A lightweight module that drops into ResNet or MobileNet with minimal overhead.")
        ]
        self.layout_split_left_image(slide, "slides/assets/cbam_attention_diagram.png",
                                     "CBAM Sequential Attention: Channel Module (What) -> Spatial Module (Where) -> Refined Tensor",
                                     "Dual Sequential Attention", bullets, COLOR_PURPLE)
        self.add_footer(slide)

    def build_slide_29(self):
        """Slide 29: Milestone 13 — Vision Transformer / ViT (Dosovitskiy et al., 2020)"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "Vision Transformer (2020)",
                        "Representing Images as Patch Sequences Without Convolutions")

        bullets = [
            ("Patch Tokenization",
             "Splits a 224×224×3 image into 196 patches, each embedded as a token."),
            ("Pure Self-Attention",
             "Tokens interact via self-attention: softmax(QK^T/√d_k)V, capturing global context."),
            ("Massive Scale Dominance",
             "Drops spatial priors, outperforming CNNs when pretrained on huge datasets.")
        ]
        self.layout_split_left_image(slide, "slides/assets/vit_architecture_diagram.png",
                                     "Vision Transformer (ViT): Patch Embedding -> Transformer Encoder -> Multi-Head Self-Attention",
                                     "Patches as Tokens", bullets, COLOR_PURPLE)
        self.add_footer(slide)

    def build_slide_30(self):
        """Slide 30: Paradigm Comparison — CNN vs. Vision Transformer"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "CNN vs. Vision Transformer",
                        "A Comparison of Inductive Priors, Cost, and Trends")

        bullets = [
            ("Inductive Bias Dichotomy",
             "CNNs bake in spatial priors; ViTs learn relationships purely from data."),
            ("Computational Asymmetry",
             "Convolutions scale linearly O(K²HW); self-attention scales quadratically O(N²)."),
            ("Modern Convergence",
             "ConvNeXt modernizes CNNs with ViT ideas; MaxViT blends convs and attention.")
        ]
        self.layout_split_left_image(slide, "slides/assets/cnn_vs_transformer_comparison.png",
                                     "Inductive Bias & Tradeoff Matrix: Local Convolutions vs. Global Self-Attention",
                                     "The Tradeoff Summary", bullets, COLOR_PRIMARY)
        self.add_footer(slide)

    def build_slide_31(self):
        """Slide 31: Canonical CNN Architecture Pipeline"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "The CNN Pipeline",
                        "From Pixels to Class Logits")

        bullets = [
            ("Feature Extraction Backbone",
             "Alternating conv, ReLU, and pooling blocks extract robust features."),
            ("Transition to Classification",
             "Pooling or flattening compresses feature maps into a 1D vector."),
            ("Decision Head",
             "Fully-connected layers plus Softmax produce class probabilities.")
        ]
        self.layout_split_left_image(slide, "slides/assets/typical_cnn.png",
                                     "End-to-End CNN Pipeline: Input -> Feature Extractor Backbone -> Classifier Head",
                                     "The Standard Pipeline", bullets, COLOR_BLUE)
        self.add_footer(slide)

    def build_slide_32(self):
        """Slide 32: Tensor Shape Contraction & Channel Expansion Dynamics"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "Tensor Shape Across Depth",
                        "Spatial Shrinkage, Channel Growth")

        bullets = [
            ("Spatial Resolution Contraction",
             "Spatial size shrinks: 224 → 112 → 56 → 28 → 7 via pooling/stride."),
            ("Channel Capacity Growth",
             "Channels grow inversely: 3 → 64 → 128 → 256 → 512."),
            ("The Tradeoff",
             "Pixel-level redundancy is traded for abstract, invariant features.")
        ]
        self.layout_split_left_image(slide, "slides/assets/tensor_shape_dynamics.png",
                                     "Tensor Flow: Spatial Contraction [224 -> 7] & Channel Expansion [3 -> 512]",
                                     "Tensor Flow Dynamics", bullets, COLOR_PURPLE)
        self.add_footer(slide)

    def build_slide_33(self):
        """Slide 33: Computational Budget Allocation — FLOPs vs. Parameter Bottlenecks"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "FLOPs vs. Parameters",
                        "Where Compute and Memory Go")

        bullets = [
            ("Compute Bottleneck (FLOPs)",
             "Convolutions account for over 90% of total FLOPs."),
            ("Memory Bottleneck (Parameters)",
             "Fully-connected layers account for over 90% of model weights."),
            ("The Fix: Global Pooling",
             "Global Average Pooling replaces dense layers, cutting millions of params.")
        ]
        self.layout_split_left_image(slide, "slides/assets/flops_params_tradeoff.png",
                                     "Workload Asymmetry: Convolutions Consume Compute; Dense Layers Consume Memory",
                                     "Compute vs. Memory", bullets, COLOR_ORANGE)
        self.add_footer(slide)

    def build_slide_34(self):
        """Slide 34: Three Levels of Implementation Abstraction"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "Three Levels of Implementation",
                        "From Raw Math to High-Level Frameworks")

        cols = [
            ("LEVEL 1", "NumPy (From Scratch)",
             [("Vectorized im2col", "Flattens sliding windows into a matrix for GEMM"),
              ("Manual Backprop", "col2im redistributes gradients by hand"),
              ("Educational Value", "Full visibility into tensor mechanics")],
             COLOR_AMBER, None),

            ("LEVEL 2", "Keras (Declarative)",
             [("Declarative Graph", "High-level layers: Conv2D, MaxPooling2D"),
              ("Automated Graph", "Forward/backward graph compiled automatically"),
              ("Rapid Prototyping", "Fast workflow for model experimentation")],
             COLOR_BLUE, None),

            ("LEVEL 3", "PyTorch (Imperative)",
             [("Dynamic Autograd", "Forward pass runs eagerly, with automatic differentiation"),
              ("Pythonic Debugging", "Direct tensor inspection and custom hooks"),
              ("Production Standard", "The dominant framework in deep learning research")],
             COLOR_GREEN, None)
        ]
        self.layout_triptych(slide, cols)
        self.add_footer(slide)

    def build_slide_35(self):
        """Slide 35: Cross-Framework Architectural Rosetta Stone (Native Table with Large Fonts)"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "Cross-Framework Rosetta Stone",
                        "NumPy, Keras, and PyTorch Side by Side")

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
        self.add_header(slide, "Forward Pass: im2col + GEMM",
                        "Turning Convolution Into Matrix Multiplication")

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
                             caption="im2col: Patches Unfolded Into Matrix Columns")
        self.add_footer(slide)

    def build_slide_37(self):
        """Slide 37: Analytical Backward Pass — col2im & Gradient Flow"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "Backward Pass: col2im",
                        "Accumulating Gradients Over Overlapping Patches")

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
                             caption="col2im: Summing Gradients Over Overlapping Patches")
        self.add_footer(slide)

    def build_slide_38(self):
        """Slide 38: Production CNN Architecture in PyTorch (nn.Module)"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "PyTorch Implementation (nn.Module)",
                        "A Compact Object-Oriented CNN")

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
             "Graphs build on-the-fly during forward execution, easing debugging."),
            ("Production BatchNorm",
             "Tracks batch statistics in training, uses fixed stats at inference."),
            ("Automatic Differentiation",
             "Autograd traces tensor operations to build the backward pass automatically.")
        ]
        self.add_large_bullets(tf_box, pt_bullets, font_size=Pt(12.5))
        self.add_footer(slide)

    def build_slide_39(self):
        """Slide 39: Empirical Verification & Real-Time Inference Dashboard"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "Verification & Inference Dashboard",
                        "Learned Filters to Live Predictions")

        bullets = [
            ("Learned Filter Convergence",
             "Trained 3×3 kernels converge to edge and texture detectors unsupervised."),
            ("Real-Time Prediction Consensus",
             "A live forward pass reaches >96% confidence, matching across all 3 frameworks."),
            ("Foundational Inductive Bias",
             "Locality and weight sharing remain the foundation of modern vision models.")
        ]
        self.layout_split_dual_images(slide, "slides/assets/learned_filters_fmnist.png",
                                      "32 Learned 3x3 Convolutional Filter Tensors (Autonomous Edge & Pattern Extraction)",
                                      "slides/assets/live_inference_visual.png",
                                      "Live Inference Dashboard: Multi-Model Single-Sample Classification Consensus",
                                      "Putting It Together", bullets, COLOR_PRIMARY)
        self.add_footer(slide)

    # -------------------------------------------------------------------------
    # PART V: FROM THEORY TO PRACTICE - THREE REAL ASSIGNMENTS (Slides 40 - 45)
    # -------------------------------------------------------------------------

    def build_slide_40(self):
        """Slide 40: Three Real Applications at a Glance (Diabetes, Fashion-MNIST, CIFAR-10)"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "From Theory to Practice",
                        "Three Applications: Tabular MLP, Grayscale CNN, RGB CNN")

        cols = [
            ("APP 01", "Diabetes (MLP)",
             [("Task", "Tabular binary classification"),
              ("Input", "50 normalized features"),
              ("Model", "MLP, 2,177 parameters")],
             COLOR_AMBER, "slides/assets/diabetes_tabular_preview.png"),

            ("APP 02", "Fashion-MNIST (CNN)",
             [("Task", "Grayscale image classification"),
              ("Input", "1×28×28, 10 classes"),
              ("Model", "2-block CNN, 421,642 parameters")],
             COLOR_BLUE, "slides/assets/fmnist_out_8_1.png"),

            ("APP 03", "CIFAR-10 (CNN)",
             [("Task", "RGB image classification"),
              ("Input", "3×32×32, 10 classes"),
              ("Model", "2-block CNN, 545,098 parameters")],
             COLOR_GREEN, "slides/assets/cifar10_out_8_1.png")
        ]
        self.layout_triptych(slide, cols)
        self.add_footer(slide)

    def build_slide_41(self):
        """Slide 41: Fashion-MNIST CNN Shape Trace (Grayscale, Single Channel)"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "Fashion-MNIST: Grayscale CNN in Practice",
                        "Tracing Tensor Shapes Through a Single-Channel CNN")

        bullets = [
            ("Input Tensor",
             "(B, 1, 28, 28): single grayscale channel, 28×28 pixels."),
            ("Conv + Pool Blocks",
             "Conv(1→32) → Pool → (B,32,14,14); Conv(32→64) → Pool → (B,64,7,7)."),
            ("Classifier Head",
             "Flatten to 3,136 → Linear(3136→128) → Linear(128→10).")
        ]
        self.layout_split_left_image(slide, "slides/assets/fmnist_out_51_7.png",
                                     "Fashion-MNIST Training Curves: Loss and Accuracy Across Epochs",
                                     "421,642 Parameters, All Frameworks", bullets, COLOR_BLUE)
        self.add_footer(slide)

    def build_slide_42(self):
        """Slide 42: CIFAR-10 CNN Shape Trace (RGB, 3 Channels)"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "CIFAR-10: RGB CNN in Practice",
                        "Tracing Tensor Shapes Through a 3-Channel CNN")

        bullets = [
            ("Input Tensor",
             "(B, 3, 32, 32): 3 color channels, 32×32 pixels."),
            ("Conv + Pool Blocks",
             "Conv(3→32) → Pool → (B,32,16,16); Conv(32→64) → Pool → (B,64,8,8)."),
            ("Classifier Head",
             "Flatten to 4,096 → Linear(4096→128) → Linear(128→10).")
        ]
        self.layout_split_left_image(slide, "slides/assets/cifar10_out_49_7.png",
                                     "CIFAR-10 Training Curves: Loss and Accuracy Across Epochs",
                                     "545,098 Parameters, All Frameworks", bullets, COLOR_GREEN)
        self.add_footer(slide)

    def build_slide_43(self):
        """Slide 43: Cross-Framework Rosetta Stone for the Three Assignments"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "Same Math, Three Frameworks",
                        "Scratch (NumPy), Keras, and PyTorch on the Same 3 Datasets")

        headers = ["Component", "Scratch (NumPy)", "Keras", "PyTorch"]
        rows = [
            ["Preprocessing",
             "/ 255.0, transpose to CHW",
             "/ 255.0, HWC (native)",
             "/ 255.0, transpose to CHW"],
            ["Convolution Layer",
             "conv2d_forward (im2col)",
             "layers.Conv2D()",
             "nn.Conv2d()"],
            ["Backpropagation",
             "Manual chain rule functions",
             "Autodiff in fit()",
             "loss.backward() (Autograd)"],
            ["Parameter Updates",
             "W -= lr * dW",
             "optimizer='sgd'",
             "torch.optim.SGD()"]
        ]
        widths = [Inches(2.6), Inches(3.4), Inches(3.1), Inches(3.113)]
        self.layout_native_rosetta_table(slide, headers, rows, widths)
        self.add_footer(slide)

    def build_slide_44(self):
        """Slide 44: Empirical Results - Accuracy and Training Time Across 3 Frameworks"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "Empirical Results Across Frameworks",
                        "Comparable Accuracy, Divergent Training Time")

        bullets = [
            ("Accuracy Parity",
             "Diabetes 73-78%, Fashion-MNIST 86-90%, CIFAR-10 57-61% across all 3 frameworks."),
            ("Identical Parameter Counts",
             "2,177 / 421,642 / 545,098 params match exactly across Scratch, Keras, PyTorch."),
            ("Training Time Gap",
             "Scratch is 15-20x slower than Keras/PyTorch on CNN workloads (pure NumPy im2col).")
        ]
        self.layout_split_dual_images(slide, "slides/assets/accuracy_comparison.png",
                                      "Test Accuracy Comparison Across Diabetes, Fashion-MNIST, and CIFAR-10",
                                      "slides/assets/training_time_scaling.png",
                                      "Training Time Scaling: Scratch vs. Keras vs. PyTorch",
                                      "Speed vs. Transparency Tradeoff", bullets, COLOR_ORANGE)
        self.add_footer(slide)

    def build_slide_45(self):
        """Slide 45: Closing Synthesis — Clean 3-Column Bento Grid (Same Style as Other Slides, No Icons, Big Font)"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.set_background(slide)
        self.add_header(slide, "Closing Synthesis",
                        "Empirical Confirmation of CNN Theory Across Three Assignments")

        cols = [
            ("KEY FINDING 01", "Mathematical Equivalence",
             [("100% Math Parity", "Identical network topologies and training splits converge to near-identical accuracy across all three frameworks."),
              ("Empirical Proof", "Diabetes 73-78%, Fashion-MNIST 86-90%, and CIFAR-10 57-61% match closely across implementations."),
              ("Universal Math", "Fundamental tensor calculus and gradient equations remain completely invariant to framework syntax.")],
             COLOR_PRIMARY, None),

            ("KEY FINDING 02", "Computational Efficiency",
             [("15x-20x Speedup", "Compiled C++/CUDA BLAS dramatically outperforms pure NumPy on convolutional workloads."),
              ("FLOPs Dominance", "Convolutions consume over 90% of total compute; memory caching and SIMD vectorization matter heavily."),
              ("Engineering Role", "NumPy teaches core mathematical intuition; compiled engines are mandatory for production scale.")],
             COLOR_ORANGE, None),

            ("KEY FINDING 03", "Abstraction Spectrum",
             [("NumPy (From Scratch)", "Provides complete visibility into tensor manipulation, im2col matrix unfolding, and col2im backprop."),
              ("Keras (High-Level)", "Maximizes developer velocity, declarative graph wiring, and rapid model prototyping."),
              ("PyTorch (Imperative)", "The modern research standard: dynamic autograd execution, pythonic debugging, and modular flexibility.")],
             COLOR_GREEN, None)
        ]
        self.layout_triptych(slide, cols)
        self.add_footer(slide)

    def build_all(self):
        print("Starting CNN Core Presentation build (45 slides)...")
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
        self.build_slide_40()
        self.build_slide_41()
        self.build_slide_42()
        self.build_slide_43()
        self.build_slide_44()
        self.build_slide_45()

        os.makedirs(os.path.dirname(self.output_pptx), exist_ok=True)
        self.prs.save(self.output_pptx)
        print(f"Successfully generated {self.current_slide_num} slides at: {self.output_pptx}")


if __name__ == "__main__":
    builder = CNNPresentationBuilder()
    builder.build_all()
