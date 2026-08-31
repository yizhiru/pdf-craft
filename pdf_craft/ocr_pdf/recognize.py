"""DeepSeek OCR with header strip + left/right column crops.

Dense dual-column vocab pages truncate or loop when OCR'd as a whole page.
Grounding-markdown on a thin strip also hallucinates (math / HTML tables).

Strategy per page:
  1. Detect where the dual-column body starts (center gutter + ink on both sides)
  2. Slice each column into short overlapping bands plus a last-line tail crop
  3. Suppress verso bleed, pad the band onto a white page
  4. Transcribe with Free OCR (plain lines, keep ※ / ▼)
  5. Merge layouts with coordinate offsets
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL.Image import Image

from ..common import AssetHub, XMLReader, save_xml
from ..pdf.handler import DefaultPDFHandler
from ..pdf.ocr import OCREvent, OCREventKind
from ..pdf.page_extractor import PageExtractorNode
from ..pdf.types import DeepSeekOCRSize, Page, PageLayout, decode, encode
from ..types import DeepSeekOCRModelId, TextLayout
from .vocab_cleanup import cleanup_vocab_lines, lines_from_ocr_response
from .vocab_image import is_weak_print

# Overlap into the gutter so edge glyphs are not clipped
_GUTTER_RATIO = 0.03

# Horizontal slices keep each DeepSeek call short so it does not loop / truncate
_BAND_OVERLAP_RATIO = 0.18
_MIN_BAND_PX = 220
_TAIL_LINES = 5
_VOCAB_OCR_PROMPT = (
    "<image>\n<|grounding|>Convert the document to markdown."
)

# DeepSeek-OCR-2 SAM+Qwen encoder only has query tables for 768 (144) and
# 1024 (256). Native 512/640/1280 (tiny/small/large) crash in the vision tower.
_OCR2_UNSUPPORTED_SIZES: frozenset[str] = frozenset({"tiny", "small", "large"})


def coerce_ocr_size(
    ocr_model: DeepSeekOCRModelId,
    ocr_size: DeepSeekOCRSize,
) -> DeepSeekOCRSize:
    """Map sizes that DeepSeek-OCR-2 cannot run onto a working preset."""
    if ocr_model == "deepseek-ocr-2" and ocr_size in _OCR2_UNSUPPORTED_SIZES:
        print(
            f"Warning: DeepSeek-OCR-2 cannot use ocr-size={ocr_size!r} "
            "(vision encoder only supports 768/1024). Using gundam instead.",
            flush=True,
        )
        return "gundam"
    return ocr_size



def cached_page_indexes(ocr_path: Path) -> set[int]:
    """1-based page indexes that already have a non-empty page_*.xml."""
    indexes: set[int] = set()
    if not ocr_path.is_dir():
        return indexes
    for path in ocr_path.glob("page_*.xml"):
        try:
            index = int(path.stem.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        if index >= 1 and path.stat().st_size > 0:
            indexes.add(index)
    return indexes


def load_ocr_pages(ocr_path: Path) -> dict[int, Page]:
    if not ocr_path.is_dir():
        raise FileNotFoundError(f"OCR directory not found: {ocr_path}")
    pages = {
        page.index: page
        for page in XMLReader(prefix="page", dir_path=ocr_path, decode=decode).read()
    }
    if not pages:
        raise ValueError(f"No page_*.xml under {ocr_path}")
    return pages


def run_deepseek_ocr(
    pdf_path: Path,
    ocr_path: Path,
    *,
    models_cache_path: Path,
    ocr_model: DeepSeekOCRModelId = "deepseek-ocr-2",
    ocr_size: DeepSeekOCRSize = "gundam",
    dpi: int = 400,
    local_only: bool = True,
    text_layout: TextLayout = "horizontal",
    column_split: bool = True,
    slice_bands: bool = True,
    band_lines: int = 8,
    page_indexes: list[int] | None = None,
    on_event=None,
) -> dict[int, Page]:
    """Recognize PDF with DeepSeek OCR and write page_*.xml under ocr_path."""
    ocr_size = coerce_ocr_size(ocr_model, ocr_size)
    if column_split:
        return run_deepseek_column_ocr(
            pdf_path,
            ocr_path,
            models_cache_path=models_cache_path,
            ocr_model=ocr_model,
            ocr_size=ocr_size,
            dpi=dpi,
            local_only=local_only,
            text_layout=text_layout,
            slice_bands=slice_bands,
            band_lines=band_lines,
            page_indexes=page_indexes,
            on_event=on_event,
        )

    from ..pdf.ocr import OCR

    ocr_path.mkdir(parents=True, exist_ok=True)
    asset_path = ocr_path.parent / "assets"
    asset_path.mkdir(parents=True, exist_ok=True)

    engine = OCR(
        model_path=models_cache_path,
        pdf_handler=None,
        local_only=local_only,
        ocr_model=ocr_model,
    )
    for event in engine.recognize(
        pdf_path=pdf_path,
        asset_path=asset_path,
        ocr_path=ocr_path,
        ocr_size=ocr_size,
        dpi=dpi,
        includes_footnotes=False,
        text_layout=text_layout,
    ):
        if on_event is not None:
            on_event(event)

    return load_ocr_pages(ocr_path)


def run_deepseek_column_ocr(
    pdf_path: Path,
    ocr_path: Path,
    *,
    models_cache_path: Path,
    ocr_model: DeepSeekOCRModelId = "deepseek-ocr-2",
    ocr_size: DeepSeekOCRSize = "gundam",
    dpi: int = 400,
    local_only: bool = True,
    text_layout: TextLayout = "horizontal",
    slice_bands: bool = True,
    band_lines: int = 8,
    page_indexes: list[int] | None = None,
    on_event=None,
) -> dict[int, Page]:
    """OCR header (full width) + sliced left/right column body crops, then merge."""
    ocr_path.mkdir(parents=True, exist_ok=True)
    asset_path = ocr_path.parent / "assets"
    asset_path.mkdir(parents=True, exist_ok=True)

    extractor = PageExtractorNode(
        model_path=models_cache_path,
        local_only=local_only,
        ocr_model=ocr_model,
    )
    asset_hub = AssetHub(asset_path)
    document = DefaultPDFHandler().open(pdf_path)
    total_pages = 0
    try:
        total_pages = document.pages_count
        total = total_pages
        cached = cached_page_indexes(ocr_path)
        if cached:
            missing = [i for i in range(1, total + 1) if i not in cached]
            print(
                f"Resume: {len(cached)}/{total} page(s) cached; "
                f"OCR remaining {missing[:8]}{'...' if len(missing) > 8 else ''}",
                flush=True,
            )
        wanted = set(page_indexes) if page_indexes else set(range(1, total + 1))
        for page_index in range(1, total + 1):
            if page_index not in wanted:
                continue
            xml_path = ocr_path / f"page_{page_index}.xml"
            if xml_path.is_file() and xml_path.stat().st_size > 0:
                if on_event is not None:
                    on_event(
                        OCREvent(
                            kind=OCREventKind.SKIP,
                            page_index=page_index,
                            total_pages=total,
                        )
                    )
                continue

            if on_event is not None:
                on_event(
                    OCREvent(
                        kind=OCREventKind.START,
                        page_index=page_index,
                        total_pages=total,
                    )
                )

            image = document.render_page(page_index, dpi=dpi)
            if on_event is not None:
                on_event(
                    OCREvent(
                        kind=OCREventKind.RENDERED,
                        page_index=page_index,
                        total_pages=total,
                    )
                )

            merged = _ocr_page_by_columns(
                extractor=extractor,
                image=image,
                page_index=page_index,
                asset_hub=asset_hub,
                ocr_model=ocr_model,
                ocr_size=ocr_size,
                text_layout=text_layout,
                slice_bands=slice_bands,
                band_lines=band_lines,
                dpi=dpi,
                on_region=lambda name: print(
                    f"  region={name} page={page_index}/{total}",
                    flush=True,
                ),
            )
            save_xml(encode(merged), ocr_path / f"page_{page_index}.xml")

            if on_event is not None:
                on_event(
                    OCREvent(
                        kind=OCREventKind.COMPLETE,
                        page_index=page_index,
                        total_pages=total,
                        input_tokens=merged.input_tokens,
                        output_tokens=merged.output_tokens,
                    )
                )
    finally:
        document.close()

    if total_pages > 0 and len(cached_page_indexes(ocr_path)) >= total_pages:
        (ocr_path / "done").touch()
    return load_ocr_pages(ocr_path)


def detect_column_body_top(image: Image) -> int:
    """Find y where dual-column body starts (below full-width header/notes)."""
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    height, width = gray.shape
    mid = width // 2
    gw = max(4, width // 80)

    left = gray[:, max(0, mid // 5) : max(mid // 5 + 1, mid - gw)]
    right = gray[:, mid + gw : min(width, mid + max(gw + 1, (width - mid) * 4 // 5))]
    gutter = gray[:, mid - gw : mid + gw]

    def _row_ink(band: np.ndarray) -> np.ndarray:
        return (255.0 - band).mean(axis=1)

    left_ink = _row_ink(left)
    right_ink = _row_ink(right)
    gut_ink = _row_ink(gutter)

    win = max(5, height // 150)
    kernel = np.ones(win, dtype=np.float32) / win
    left_ink = np.convolve(left_ink, kernel, mode="same")
    right_ink = np.convolve(right_ink, kernel, mode="same")
    gut_ink = np.convolve(gut_ink, kernel, mode="same")

    text_thr = max(
        4.0,
        float(np.percentile(np.concatenate([left_ink, right_ink]), 60)) * 0.35,
    )
    gut_thr = max(3.0, float(np.percentile(gut_ink, 50)))

    min_y = int(height * 0.04)
    need = max(8, height // 100)
    run = 0
    for y in range(min_y, int(height * 0.45)):
        if (
            gut_ink[y] <= gut_thr
            and left_ink[y] >= text_thr
            and right_ink[y] >= text_thr
        ):
            run += 1
            if run >= need:
                return max(0, y - need)
        else:
            run = 0

    # Vocab index pages are dual-column from the top; keep only a thin running-header
    return int(height * 0.07)


def column_slice_boxes(
    *,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    band_px: int,
    overlap_px: int,
    image: Image,
) -> list[tuple[int, int, int, int]]:
    """Split a column's ink range into overlapping horizontal bands."""
    content_y1, content_y2 = _column_ink_range(image, x1, y1, x2, y2)
    if content_y2 - content_y1 < 8:
        return []
    height = content_y2 - content_y1
    band_px = max(_MIN_BAND_PX, min(band_px, height))
    overlap_px = max(8, min(overlap_px, band_px // 3))
    boxes: list[tuple[int, int, int, int]] = []
    y = content_y1
    while y < content_y2:
        top = y
        bottom = min(content_y2, y + band_px)
        boxes.append((x1, top, x2, bottom))
        if bottom >= content_y2:
            break
        next_y = bottom - overlap_px
        if next_y <= y:
            next_y = bottom
        y = next_y
    return boxes


def _column_ink_range(
    image: Image,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> tuple[int, int]:
    crop = image.crop((x1, y1, x2, y2))
    if crop.size[0] < 8 or crop.size[1] < 8:
        return y1, y1
    gray = np.asarray(crop.convert("L"), dtype=np.float32)
    # Sparse vocab rows (word left, number right) have a low row-mean; gray
    # kana bars do not. Use dark-pixel fraction so real lines are not clipped.
    dark = (gray < 125.0).mean(axis=1)
    win = max(3, gray.shape[0] // 400)
    dark = np.convolve(dark, np.ones(win, dtype=np.float32) / win, mode="same")
    thr = max(0.0035, float(np.percentile(dark, 70)) * 0.35)
    footer = max(8, int(gray.shape[0] * 0.08))
    limit = gray.shape[0] - footer
    hits = np.flatnonzero(dark[:limit] >= thr)
    if hits.size == 0:
        return y1, y1
    start = int(hits[0])
    end = start
    empty_run = 0
    max_empty = max(48, int(_typical_line_from_ink((255.0 - gray).mean(axis=1)) * 2.5))
    for y in range(start, limit):
        if dark[y] >= thr:
            end = y
            empty_run = 0
        else:
            empty_run += 1
            if empty_run > max_empty and (y - start) > max_empty * 3:
                break
    pad = max(8, int(_typical_line_from_ink((255.0 - gray).mean(axis=1))))
    return y1 + max(0, start - pad), y1 + min(gray.shape[0], end + pad)


def _typical_line_from_ink(ink: np.ndarray) -> int:
    thr = max(8.0, float(np.percentile(ink, 70)) * 0.5)
    in_line = ink >= thr
    heights: list[int] = []
    start = None
    for y, flag in enumerate(in_line):
        if flag and start is None:
            start = y
        elif not flag and start is not None:
            height = y - start
            if 10 <= height <= 70:
                heights.append(height)
            start = None
    if heights:
        return int(np.median(np.asarray(heights, dtype=np.float32)))
    return 24


def _typical_line_px(
    image: Image,
    box: tuple[int, int, int, int],
    dpi: int,
) -> int:
    crop = image.crop(box)
    if crop.size[0] < 8 or crop.size[1] < 8:
        return max(22, int(dpi * 0.055))
    gray = np.asarray(crop.convert("L"), dtype=np.float32)
    ink = (255.0 - gray).mean(axis=1)
    detected = _typical_line_from_ink(ink)
    fallback = max(22, int(dpi * 0.055))
    return max(fallback, detected + 8)


def _ocr_page_by_columns(
    *,
    extractor: PageExtractorNode,
    image: Image,
    page_index: int,
    asset_hub: AssetHub,
    ocr_model: DeepSeekOCRModelId,
    ocr_size: DeepSeekOCRSize,
    text_layout: TextLayout,
    slice_bands: bool = True,
    band_lines: int = 8,
    dpi: int = 400,
    on_region=None,
) -> Page:
    del asset_hub, text_layout  # Free-OCR path does not use layout extraction
    width, height = image.size
    mid = width // 2
    pad = max(8, int(width * _GUTTER_RATIO))
    detected_top = detect_column_body_top(image)
    has_preamble = detected_top >= int(height * 0.12)
    if has_preamble:
        body_top = min(detected_top, int(height * 0.18))
        column_top = max(0, body_top - max(4, height // 200))
    else:
        # Index pages: dual column from the top. Do not full-width OCR vocab rows.
        body_top = max(int(height * 0.03), min(detected_top, int(height * 0.06)))
        column_top = body_top

    line_px = _typical_line_px(image, (0, column_top, width, height), dpi)
    band_px = max(_MIN_BAND_PX, line_px * max(4, band_lines))
    overlap_px = max(line_px, int(band_px * _BAND_OVERLAP_RATIO))

    left_box = (0, column_top, min(width, mid + pad), height)
    right_box = (max(0, mid - pad), column_top, width, height)

    regions: list[tuple[str, tuple[int, int, int, int]]] = []
    if has_preamble and body_top >= max(16, height // 40):
        regions.append(("header", (0, 0, width, body_top)))

    if slice_bands:
        for name, box in (("left", left_box), ("right", right_box)):
            slices = column_slice_boxes(
                x1=box[0],
                y1=box[1],
                x2=box[2],
                y2=box[3],
                band_px=band_px,
                overlap_px=overlap_px,
                image=image,
            )
            for index, slice_box in enumerate(slices):
                regions.append((f"{name}_{index}", slice_box))
            tail = _column_tail_box(image, box, line_px)
            if tail is not None:
                regions.append((f"{name}_tail", tail))
    else:
        regions.append(("left", left_box))
        regions.append(("right", right_box))

    transcribe_size = _band_ocr_size(ocr_model, ocr_size)
    body: list[PageLayout] = []
    input_tokens = 0
    output_tokens = 0

    for region_name, box in regions:
        if on_region is not None:
            on_region(region_name)
        if is_weak_print(image, box):
            print(f"  skip blank region={region_name} page={page_index}", flush=True)
            continue

        crop = image.crop(box)
        if crop.size[0] < 8 or crop.size[1] < 8:
            continue

        try:
            raw_text, in_tok, out_tok = extractor.transcribe_image(
                crop,
                prompt=_VOCAB_OCR_PROMPT,
                ocr_size=transcribe_size,
            )
        except Exception as error:
            print(
                f"  OCR failed region={region_name} page={page_index}: {error}",
                flush=True,
            )
            continue
        input_tokens += in_tok
        output_tokens += out_tok

        lines = cleanup_vocab_lines(lines_from_ocr_response(raw_text))
        if not lines:
            continue

        side = "header"
        if region_name.startswith("left"):
            side = "left"
        elif region_name.startswith("right"):
            side = "right"

        for index, line in enumerate(lines):
            layout = PageLayout(
                ref="text",
                det=_line_box(box, index, len(lines)),
                text=line,
                order=0,
                hash=None,
            )
            center_x = (layout.det[0] + layout.det[2]) / 2.0
            center_y = (layout.det[1] + layout.det[3]) / 2.0
            if side == "header":
                if center_y >= body_top:
                    continue
            elif side == "left":
                if center_x >= mid or center_y < body_top:
                    continue
            elif side == "right":
                if center_x < mid or center_y < body_top:
                    continue
            body.append(layout)

    body = [
        PageLayout(
            ref=layout.ref,
            det=layout.det,
            text=layout.text,
            order=index,
            hash=layout.hash,
        )
        for index, layout in enumerate(body)
    ]

    return Page(
        index=page_index,
        image=None,
        body_layouts=body,
        footnotes_layouts=[],
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _band_ocr_size(
    ocr_model: DeepSeekOCRModelId,
    ocr_size: DeepSeekOCRSize,
) -> DeepSeekOCRSize:
    """Padded bands should not use gundam's dynamic crop tiles."""
    if ocr_model == "deepseek-ocr-2":
        return "base"
    return coerce_ocr_size(ocr_model, ocr_size)


def _column_tail_box(
    image: Image,
    box: tuple[int, int, int, int],
    line_px: int,
) -> tuple[int, int, int, int] | None:
    """Extra crop of the last few printed lines so the column end is not dropped."""
    x1, y1, x2, y2 = box
    content_y1, content_y2 = _column_ink_range(image, x1, y1, x2, y2)
    if content_y2 - content_y1 < 8:
        return None
    height = max(line_px * _TAIL_LINES, _MIN_BAND_PX // 2)
    top = max(content_y1, content_y2 - height)
    if content_y2 - top < 8:
        return None
    return (x1, top, x2, content_y2)


def _line_box(
    box: tuple[int, int, int, int],
    index: int,
    total: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    total = max(total, 1)
    span = max(y2 - y1, total)
    top = y1 + (span * index) // total
    bottom = y1 + (span * (index + 1)) // total
    if bottom <= top:
        bottom = top + 1
    return (x1, top, x2, bottom)
