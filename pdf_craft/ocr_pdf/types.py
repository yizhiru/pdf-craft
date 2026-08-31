from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class OcrTextLine:
    """One logical text line after OCR normalize (no page geometry)."""

    text: str
    ref: str = "text"


@dataclass
class OcrPdfPage:
    """OCR content for one source page, ready to typeset into a new PDF."""

    page_index: int  # 1-based source page
    lines: list[OcrTextLine] = field(default_factory=list)
