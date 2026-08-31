"""End-to-end: DeepSeek OCR → ordered text → brand-new searchable PDF."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from ..pdf.types import DeepSeekOCRSize
from ..types import DeepSeekOCRModelId, TextLayout
from .compose import compose_ocr_pdf
from .normalize import pages_to_ocr_pdf_pages
from .recognize import cached_page_indexes, load_ocr_pages, run_deepseek_ocr


def build_ocr_pdf(
    pdf_path: Path,
    output_path: Path,
    *,
    ocr_path: Path | None = None,
    run_ocr: bool = False,
    force_ocr: bool = False,
    models_cache_path: Path | None = None,
    ocr_model: DeepSeekOCRModelId = "deepseek-ocr-2",
    ocr_size: DeepSeekOCRSize = "gundam",
    dpi: int = 400,
    local_only: bool = True,
    text_layout: TextLayout = "horizontal",
    column_split: bool = True,
    slice_bands: bool = True,
    band_lines: int = 8,
    page_indexes: list[int] | None = None,
    fontname: str = "japan",
    fontsize: float = 12.0,
    title: str | None = None,
    on_ocr_event: Callable | None = None,
) -> int:
    """Build a new text PDF from DeepSeek OCR (does not keep scanned pages).

    Steps:
      1. recognize — header + sliced left/right column OCR (default)
      2. normalize — loop cut, HTML cleanup, left-then-right reading order
      3. compose   — typeset visible Unicode text into a brand-new PDF

    Returns the number of pages in the output PDF.
    """
    pdf_path = Path(pdf_path)
    output_path = Path(output_path)
    if ocr_path is None:
        ocr_path = output_path.parent / "ocr"
    ocr_path = Path(ocr_path)

    wanted = set(page_indexes) if page_indexes else None

    if force_ocr and ocr_path.exists():
        import shutil

        if wanted is None:
            shutil.rmtree(ocr_path)
        else:
            for index in wanted:
                xml_path = ocr_path / f"page_{index}.xml"
                if xml_path.exists():
                    xml_path.unlink()
            done = ocr_path / "done"
            if done.exists():
                done.unlink()

    cached = cached_page_indexes(ocr_path)
    total_pages = _pdf_page_count(pdf_path)
    needed = wanted if wanted is not None else set(range(1, total_pages + 1))
    cache_complete = bool(needed) and needed.issubset(cached)
    if run_ocr or force_ocr or not cache_complete:
        if models_cache_path is None:
            raise ValueError("models_cache_path is required when running OCR")
        pages = run_deepseek_ocr(
            pdf_path,
            ocr_path,
            models_cache_path=Path(models_cache_path),
            ocr_model=ocr_model,
            ocr_size=ocr_size,
            dpi=dpi,
            local_only=local_only,
            text_layout=text_layout,
            column_split=column_split,
            slice_bands=slice_bands,
            band_lines=band_lines,
            page_indexes=sorted(needed) if wanted is not None else None,
            on_event=on_ocr_event,
        )
    else:
        pages = load_ocr_pages(ocr_path)

    if wanted is not None:
        pages = {index: page for index, page in pages.items() if index in wanted}

    models = pages_to_ocr_pdf_pages(pages)

    page_width, page_height = _first_page_size(pdf_path)
    return compose_ocr_pdf(
        models,
        output_path,
        fontname=fontname,
        fontsize=fontsize,
        page_width=page_width,
        page_height=page_height,
        title=title or pdf_path.stem,
    )


def _pdf_page_count(pdf_path: Path) -> int:
    try:
        import fitz
    except ImportError:
        return 0
    doc = fitz.open(pdf_path)
    try:
        return int(doc.page_count)
    finally:
        doc.close()


def _first_page_size(pdf_path: Path) -> tuple[float, float]:
    try:
        import fitz
    except ImportError:
        return 468.0, 706.8

    doc = fitz.open(pdf_path)
    try:
        if doc.page_count == 0:
            return 468.0, 706.8
        rect = doc[0].rect
        return float(rect.width), float(rect.height)
    finally:
        doc.close()
