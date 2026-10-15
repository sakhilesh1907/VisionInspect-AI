"""
scripts/generate_pdf_doc.py

Generates a formatted, publication-grade PDF manual from DOCUMENTATION.md
using ReportLab.
"""

import sys
import os
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
    HRFlowable,
)
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Adds running headers and 'Page X of Y' footers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, total_pages):
        if self._pageNumber == 1:
            return  # Suppress headers/footers on cover/title page

        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#555555"))

        # Header
        self.drawString(54, letter[1] - 36, "VisionInspect AI — Technical Specification & Research Documentation")
        self.setStrokeColor(colors.HexColor("#D0D7DE"))
        self.setLineWidth(0.5)
        self.line(54, letter[1] - 42, letter[0] - 54, letter[1] - 42)

        # Footer
        page_str = f"Page {self._pageNumber} of {total_pages}"
        self.drawRightString(letter[0] - 54, 36, page_str)
        self.drawString(54, 36, "CONFIDENTIAL & PROPRIETARY — AMD PROJECT")
        self.line(54, 46, letter[0] - 54, 46)

        self.restoreState()


def build_pdf(md_file: Path, output_pdf: Path):
    doc = SimpleDocTemplate(
        str(output_pdf),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    primary_color = colors.HexColor("#1E3A8A")  # Deep Blue
    accent_color = colors.HexColor("#0D9488")   # Teal
    text_dark = colors.HexColor("#1F2937")       # Dark Charcoal
    code_bg = colors.HexColor("#F3F4F6")         # Light grey

    title_style = ParagraphStyle(
        "CoverTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=30,
        textColor=primary_color,
        alignment=0,
        spaceAfter=10,
    )

    subtitle_style = ParagraphStyle(
        "CoverSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#4B5563"),
        spaceAfter=15,
    )

    h1_style = ParagraphStyle(
        "CustomH1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=20,
        textColor=primary_color,
        spaceBefore=16,
        spaceAfter=8,
        keepWithNext=True,
    )

    h2_style = ParagraphStyle(
        "CustomH2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=accent_color,
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True,
    )

    h3_style = ParagraphStyle(
        "CustomH3",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#374151"),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "CustomBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13.5,
        textColor=text_dark,
        spaceAfter=6,
    )

    bullet_style = ParagraphStyle(
        "CustomBullet",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13.5,
        textColor=text_dark,
        leftIndent=15,
        spaceAfter=3,
    )

    code_style = ParagraphStyle(
        "CustomCode",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=7.5,
        leading=10.5,
        textColor=colors.HexColor("#111827"),
    )

    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=colors.white,
    )

    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10.5,
        textColor=text_dark,
    )

    story = []

    with open(md_file, "r", encoding="utf-8") as f:
        text = f.read()

    lines = text.split("\n")
    in_code_block = False
    code_lines = []
    in_table = False
    table_rows = []

    def clean_inline(s):
        s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
        s = re.sub(r"`(.+?)`", r'<font face="Courier" color="#1E3A8A">\1</font>', s)
        s = re.sub(r"\*(.+?)\*", r"<i>\1</i>", s)
        return s

    for line in lines:
        stripped = line.strip()

        # Handle Code Blocks
        if stripped.startswith("```"):
            if in_code_block:
                in_code_block = False
                raw_code = "\n".join(code_lines)
                escaped_code = raw_code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                code_p = Paragraph(f"<pre>{escaped_code}</pre>", code_style)
                code_table = Table([[code_p]], colWidths=[letter[0] - 108])
                code_table.setStyle(
                    TableStyle([
                        ("BACKGROUND", (0, 0), (-1, -1), code_bg),
                        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
                        ("LEFTPADDING", (0, 0), (-1, -1), 8),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                        ("TOPPADDING", (0, 0), (-1, -1), 6),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ])
                )
                story.append(Spacer(1, 4))
                story.append(code_table)
                story.append(Spacer(1, 6))
                code_lines = []
            else:
                in_code_block = True
                code_lines = []
            continue

        if in_code_block:
            code_lines.append(line)
            continue

        # Handle Tables
        if "|" in stripped and not in_code_block:
            if re.match(r"^\|?[\s\-:|]+\|?$", stripped):
                continue  # Divider line
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if cells:
                table_rows.append(cells)
                in_table = True
            continue
        elif in_table:
            # End of table
            if table_rows:
                num_cols = len(table_rows[0])
                col_width = (letter[0] - 108) / num_cols
                formatted_data = []
                for row_idx, r in enumerate(table_rows):
                    row_cells = []
                    for c in r:
                        style = table_header_style if row_idx == 0 else table_cell_style
                        row_cells.append(Paragraph(clean_inline(c), style))
                    formatted_data.append(row_cells)

                t = Table(formatted_data, colWidths=[col_width] * num_cols)
                t.setStyle(
                    TableStyle([
                        ("BACKGROUND", (0, 0), (-1, 0), primary_color),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("TOPPADDING", (0, 0), (-1, -1), 5),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                        ("LEFTPADDING", (0, 0), (-1, -1), 5),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FAFB")]),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
                    ])
                )
                story.append(Spacer(1, 4))
                story.append(t)
                story.append(Spacer(1, 6))
            in_table = False
            table_rows = []

        # Empty lines
        if not stripped:
            continue

        # Horizontal Rule
        if stripped in ("---", "***", "___"):
            story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#E5E7EB"), spaceBefore=8, spaceAfter=8))
            continue

        # Headings
        if stripped.startswith("# "):
            title_text = clean_inline(stripped[2:])
            story.append(Paragraph(title_text, title_style))
            story.append(HRFlowable(width="100%", thickness=1.5, color=primary_color, spaceBefore=4, spaceAfter=10))
            continue
        elif stripped.startswith("## "):
            story.append(Paragraph(clean_inline(stripped[3:]), h1_style))
            continue
        elif stripped.startswith("### "):
            story.append(Paragraph(clean_inline(stripped[4:]), h2_style))
            continue
        elif stripped.startswith("#### "):
            story.append(Paragraph(clean_inline(stripped[5:]), h3_style))
            continue

        # Blockquotes / Highlights
        if stripped.startswith("> "):
            quote_text = clean_inline(stripped[2:])
            quote_p = Paragraph(f"<i>{quote_text}</i>", body_style)
            quote_table = Table([[quote_p]], colWidths=[letter[0] - 108])
            quote_table.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EFF6FF")),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#BFDBFE")),
                    ("LINEBEFORE", (0, 0), (0, -1), 3, primary_color),
                    ("LEFTPADDING", (0, 0), (-1, -1), 10),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ])
            )
            story.append(Spacer(1, 4))
            story.append(quote_table)
            story.append(Spacer(1, 6))
            continue

        # Bullet lists
        if stripped.startswith("- ") or stripped.startswith("* "):
            story.append(Paragraph(f"• {clean_inline(stripped[2:])}", bullet_style))
            continue
        if re.match(r"^\d+\.\s", stripped):
            num_match = re.match(r"^(\d+\.)\s*(.*)", stripped)
            story.append(Paragraph(f"<b>{num_match.group(1)}</b> {clean_inline(num_match.group(2))}", bullet_style))
            continue

        # Normal Paragraph
        story.append(Paragraph(clean_inline(stripped), body_style))

    # Build document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"✅ PDF successfully generated at: {output_pdf.resolve()}")


if __name__ == "__main__":
    src_md = Path("DOCUMENTATION.md")
    out_pdf = Path("VisionInspect_AI_Documentation.pdf")
    build_pdf(src_md, out_pdf)
