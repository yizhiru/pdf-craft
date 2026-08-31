"""DeepSeek OCR → brand-new text PDF (searchable / copyable)."""

from .pipeline import build_ocr_pdf
from .types import OcrPdfPage, OcrTextLine

__all__ = ["build_ocr_pdf", "OcrPdfPage", "OcrTextLine"]
