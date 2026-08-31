"""Typeset OCR text into a brand-new PDF (no scanned page images).

Visible Unicode glyphs are the only content — search and copy work in any
standard PDF reader because there is no image layer covering the text.
"""

from __future__ import annotations

from pathlib import Path

from .types import OcrPdfPage

_FONT_RESOURCE = "DeepSeekOCR"

# A4-like default; overridden by page_width/page_height if provided
_DEFAULT_WIDTH = 468.0
_DEFAULT_HEIGHT = 706.8


def compose_ocr_pdf(
    pages: list[OcrPdfPage],
    output_path: Path,
    *,
    fontname: str = "japan",
    fontsize: float = 12.0,
    line_gap: float = 1.35,
    margin: float = 36.0,
    page_width: float = _DEFAULT_WIDTH,
    page_height: float = _DEFAULT_HEIGHT,
    title: str | None = None,
) -> int:
    """Write a new text PDF from OCR lines. Returns number of output pages."""
    try:
        import fitz
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF is required. Install with: pip install pymupdf"
        ) from exc

    font = fitz.Font(fontname)
    font_buffer = font.buffer
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = fitz.open()
    try:
        if title:
            doc.set_metadata({"title": title})

        writer_page = None
        text_writer = None
        cursor_y = margin
        line_height = fontsize * line_gap
        max_y = page_height - margin
        out_pages = 0

        def _new_page():
            nonlocal writer_page, text_writer, cursor_y, out_pages
            if text_writer is not None and writer_page is not None:
                text_writer.write_text(writer_page, overlay=True)
            writer_page = doc.new_page(width=page_width, height=page_height)
            writer_page.insert_font(fontname=_FONT_RESOURCE, fontbuffer=font_buffer)
            text_writer = fitz.TextWriter(writer_page.rect)
            cursor_y = margin + fontsize
            out_pages += 1

        _new_page()
        assert text_writer is not None and writer_page is not None

        for src in pages:
            if src.page_index > 1 and cursor_y > margin + fontsize * 2:
                # visual break between source pages
                cursor_y += line_height * 0.5
                if cursor_y > max_y:
                    _new_page()

            for line in src.lines:
                text = line.text.strip()
                if not text:
                    continue

                # title-like refs slightly larger
                use_size = fontsize * 1.15 if line.ref in ("title", "sub_title") else fontsize
                use_height = use_size * line_gap

                # wrap long lines by character width estimate
                chunks = _wrap_line(font, text, use_size, page_width - 2 * margin)
                for chunk in chunks:
                    if cursor_y + use_size * 0.2 > max_y:
                        _new_page()
                        use_height = use_size * line_gap
                    text_writer.append(
                        fitz.Point(margin, cursor_y),
                        chunk,
                        font=font,
                        fontsize=use_size,
                    )
                    cursor_y += use_height

        if text_writer is not None and writer_page is not None:
            text_writer.write_text(writer_page, overlay=True)

        if out_pages == 0:
            # empty document guard
            page = doc.new_page(width=page_width, height=page_height)
            page.insert_font(fontname=_FONT_RESOURCE, fontbuffer=font_buffer)
            page.insert_text(
                (margin, margin + fontsize),
                "(no OCR text)",
                fontname=_FONT_RESOURCE,
                fontsize=fontsize,
            )
            out_pages = 1

        doc.save(output_path, garbage=4, deflate=True, clean=True)
    finally:
        doc.close()

    return out_pages


def _wrap_line(font, text: str, fontsize: float, max_width: float) -> list[str]:
    """Wrap only at spaces so CJK headwords are not broken mid-word."""
    if max_width <= 0:
        return [text]

    def width_of(s: str) -> float:
        total = 0.0
        for ch in s:
            try:
                total += float(font.glyph_advance(ord(ch))) * fontsize
            except Exception:
                total += fontsize
        return total

    if width_of(text) <= max_width:
        return [text]

    words = text.split(" ")
    if len(words) == 1:
        # Single token longer than the line: keep intact (no mid-glyph wrap)
        return [text]

    chunks: list[str] = []
    buf: list[str] = []
    for word in words:
        trial = (" ".join(buf + [word])).strip()
        if buf and width_of(trial) > max_width:
            chunks.append(" ".join(buf))
            buf = [word]
        else:
            buf.append(word)
    if buf:
        chunks.append(" ".join(buf))
    return chunks or [text]
