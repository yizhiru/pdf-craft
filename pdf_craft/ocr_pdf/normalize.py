"""Normalize DeepSeek layouts into ordered plain-text lines.

Reading order for dual-column vocab pages:
  1. Full-width blocks (titles / notes), top → bottom
  2. Left column entries, top → bottom
  3. Right column entries, top → bottom

HTML ``<table>`` blocks are parsed into left/right column entries (not row-wise).
"""

from __future__ import annotations

import html
import re

from ..pdf.ngrams import trim_repetitive_tail
from ..pdf.types import Page, PageLayout
from ..sequence.vocab_split import expand_dual_column_vocab_layouts
from .types import OcrPdfPage, OcrTextLine
from .vocab_cleanup import (
    cleanup_vocab_lines,
    clean_pseudo_tags,
    is_usable_vocab_text,
    trim_repeating_table_html,
)

_MARKDOWN_HEAD = re.compile(r"^#+\s+")
_FOX_JUNK = re.compile(r"The quick brown fox", re.I)
_TABLE_BLOCK = re.compile(r"<table\b[^>]*>.*?</table>", re.IGNORECASE | re.DOTALL)
_TR_BLOCK = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
_TD_BLOCK = re.compile(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", re.IGNORECASE | re.DOTALL)
_TAG = re.compile(r"<[^>]+>")

# Blocks wider than this fraction of the page are treated as full-width preamble
_FULL_WIDTH_RATIO = 0.55


def pages_to_ocr_pdf_pages(pages: dict[int, Page]) -> list[OcrPdfPage]:
    result: list[OcrPdfPage] = []
    for page_index in sorted(pages):
        page = pages[page_index]
        lines: list[OcrTextLine] = []
        text_batch: list[PageLayout] = []

        for layout in page.body_layouts:
            if layout.ref == "image":
                continue

            raw_text = (layout.text or "").strip()
            if not raw_text:
                continue
            if not is_usable_vocab_text(raw_text) and layout.ref != "table":
                continue

            if layout.ref == "table" or _TABLE_BLOCK.search(raw_text):
                trimmed = layout
                if layout.ref == "table" or "<table" in raw_text.lower():
                    html_text = trim_repeating_table_html(
                        trim_repetitive_tail(raw_text)
                    )
                    trimmed = PageLayout(
                        ref=layout.ref,
                        det=layout.det,
                        text=html_text,
                        order=layout.order,
                        hash=layout.hash,
                    )
                text_batch.extend(table_to_vocab_layouts(trimmed))
                continue

            usable = _usable_layout(layout)
            if usable is not None:
                text_batch.append(usable)

        for layout in order_left_column_then_right(text_batch):
            text = _clean_text(layout.text)
            if text:
                lines.append(OcrTextLine(text=text, ref=layout.ref or "text"))
        texts = cleanup_vocab_lines([line.text for line in lines])
        lines = [OcrTextLine(text=text, ref="text") for text in texts]
        result.append(OcrPdfPage(page_index=page_index, lines=lines))
    return result


def order_left_column_then_right(layouts: list[PageLayout]) -> list[PageLayout]:
    """Split fused dual-column rows, then emit: full-width → left col → right col."""
    if not layouts:
        return []

    expanded = expand_dual_column_vocab_layouts(layouts)
    min_x = min(layout.det[0] for layout in expanded)
    max_x = max(layout.det[2] for layout in expanded)
    page_width = max(max_x - min_x, 1)
    mid_x = (min_x + max_x) / 2.0

    full_width: list[PageLayout] = []
    left: list[PageLayout] = []
    right: list[PageLayout] = []

    for layout in expanded:
        x1, _, x2, _ = layout.det
        width = x2 - x1
        center_x = (x1 + x2) / 2.0
        if width >= page_width * _FULL_WIDTH_RATIO:
            full_width.append(layout)
        elif center_x < mid_x:
            left.append(layout)
        else:
            right.append(layout)

    full_width.sort(key=lambda item: (item.order, item.det[1], item.det[0]))
    left.sort(key=lambda item: (item.order, item.det[1], item.det[0]))
    right.sort(key=lambda item: (item.order, item.det[1], item.det[0]))
    return full_width + left + right


def table_to_vocab_layouts(layout: PageLayout) -> list[PageLayout]:
    """Turn an OCR HTML table into per-entry layouts on left and/or right columns."""
    left_entries, right_entries = split_table_columns(layout.text)
    left_entries = cleanup_vocab_lines(left_entries)
    right_entries = cleanup_vocab_lines(right_entries)
    if not left_entries and not right_entries:
        rows = flatten_table_html(layout.text)
        if not rows:
            return []
        return [
            PageLayout(
                ref="text",
                det=layout.det,
                text=" ".join(rows),
                order=layout.order,
                hash=layout.hash,
            )
        ]

    x1, y1, x2, y2 = layout.det
    mid = (x1 + x2) // 2
    gap = max(4, (x2 - x1) // 50)
    left_box = (x1, y1, max(mid - gap, x1 + 1), y2)
    right_box = (min(mid + gap, x2 - 1), y1, x2, y2)

    result: list[PageLayout] = []
    for index, text in enumerate(left_entries):
        result.append(
            PageLayout(
                ref="text",
                det=_slice_y(left_box, index, len(left_entries)),
                text=text,
                order=layout.order,
                hash=None,
            )
        )
    for index, text in enumerate(right_entries):
        result.append(
            PageLayout(
                ref="text",
                det=_slice_y(right_box, index, len(right_entries)),
                text=text,
                order=layout.order,
                hash=None,
            )
        )
    return result


def split_table_columns(text: str) -> tuple[list[str], list[str]]:
    """Parse table rows into (left_column_entries, right_column_entries)."""
    value = html.unescape(text or "").strip()
    if not value:
        return [], []

    left: list[str] = []
    right: list[str] = []
    offset = 0
    for tr_match in _TR_BLOCK.finditer(value):
        surrounding = _clean_cell(value[offset:tr_match.start()])
        if surrounding:
            left.append(surrounding)
        raw_cells = [
            _clean_cell(cell) for cell in _TD_BLOCK.findall(tr_match.group(1))
        ]
        entry_left, entry_right = _row_to_column_entries(raw_cells)
        if entry_left:
            left.append(entry_left)
        if entry_right:
            right.append(entry_right)
        offset = tr_match.end()
    trailing = _clean_cell(value[offset:])
    if trailing:
        left.append(trailing)
    return left, right


def _row_to_column_entries(cells: list[str]) -> tuple[str | None, str | None]:
    """Map one table row to optional left/right vocab lines."""
    if len(cells) >= 4:
        left = _join_word_pages(cells[0], cells[1])
        right = " ".join(cell for cell in cells[2:] if cell)
        return left or None, right or None

    nonempty = [cell for cell in cells if cell]
    if len(nonempty) == 2:
        joined = _join_word_pages(nonempty[0], nonempty[1])
        return joined or None, None

    if len(nonempty) == 1:
        return nonempty[0], None

    if len(nonempty) == 3:
        left = _join_word_pages(nonempty[0], nonempty[1])
        return left or None, nonempty[2]

    return None, None


def _join_word_pages(word: str, pages: str) -> str:
    word = (word or "").strip()
    pages = (pages or "").strip()
    if word and pages:
        return f"{word} {pages}"
    return word or pages


def flatten_table_html(text: str) -> list[str]:
    """Convert DeepSeek HTML table markup into plain vocab lines (row-wise).

    Prefer ``split_table_columns`` for dual-column reading order.
    """
    left, right = split_table_columns(text)
    if left or right:
        rows: list[str] = []
        for index in range(max(len(left), len(right))):
            if index < len(left) and index < len(right):
                rows.append(f"{left[index]} {right[index]}")
            elif index < len(left):
                rows.append(left[index])
            else:
                rows.append(right[index])
        return rows

    value = html.unescape(text or "").strip()
    if not value:
        return []
    rows: list[str] = []
    for table_html in _TABLE_BLOCK.findall(value) or [value]:
        for tr_match in _TR_BLOCK.finditer(table_html):
            cells = [
                _clean_cell(cell) for cell in _TD_BLOCK.findall(tr_match.group(1))
            ]
            cells = [cell for cell in cells if cell]
            if not cells:
                continue
            rows.append(" ".join(cells))
    if not rows and "<table" in value.lower():
        stripped = _TAG.sub(" ", value)
        stripped = re.sub(r"\s+", " ", stripped).strip()
        if stripped:
            rows.append(stripped)
    return rows


def _slice_y(
    det: tuple[int, int, int, int],
    index: int,
    total: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = det
    total = max(total, 1)
    span = max(y2 - y1, total)
    top = y1 + (span * index) // total
    bottom = y1 + (span * (index + 1)) // total
    if bottom <= top:
        bottom = top + 1
    return (x1, top, x2, bottom)


def _clean_cell(cell_html: str) -> str:
    return clean_pseudo_tags(cell_html)


def _usable_layout(layout: PageLayout) -> PageLayout | None:
    text = _clean_text(layout.text)
    if not text:
        return None
    if "<table" in text.lower():
        return None
    return PageLayout(
        ref=layout.ref,
        det=layout.det,
        text=text,
        order=layout.order,
        hash=layout.hash,
    )


def _clean_text(text: str | None) -> str:
    value = (text or "").strip()
    if not value:
        return ""
    value = trim_repetitive_tail(value)
    value = _MARKDOWN_HEAD.sub("", value).strip()
    if value.startswith("[[") and value.endswith("]]"):
        return ""
    if _FOX_JUNK.search(value):
        return ""
    if "<table" in value.lower():
        return ""
    return clean_pseudo_tags(value)
