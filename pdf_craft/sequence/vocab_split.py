"""Split OCR rows that fused left/right vocabulary entries into one line."""

from __future__ import annotations

import re
from dataclasses import replace

from ..pdf import PageLayout

# 课次标记：16※ / 10▼ / 7▼, 14※ / 5, 5▼
_PAGE_REFS = re.compile(r"\d+[※▼]?(?:\s*,\s*\d+[※▼]?)*")
# 词条：词头（含 [叹词] 等）+ 课次；词头不能以数字开头
_VOCAB_ENTRY = re.compile(
    r"^(?P<head>[^\d\s].*?)\s+(?P<pages>\d+[※▼]?(?:\s*,\s*\d+[※▼]?)*)\s*$"
)


def expand_dual_column_vocab_layouts(layouts: list[PageLayout]) -> list[PageLayout]:
    """OCR 常把双栏按行读成「左词 右词 左词 右词…」；拆开后归入左右栏 bbox。

    阅读序由调用方按「左栏整列 → 右栏整列」排列。

    注意：若 OCR 块本身已落在右半栏（如本页「う」段只在右栏），则视为单栏
    连续词条，不再按奇偶拆到左栏——否则会把「妹」和右栏首词「妹さん」拆开。
    """
    if not layouts:
        return []

    page_min_x = min(layout.det[0] for layout in layouts)
    page_max_x = max(layout.det[2] for layout in layouts)
    page_mid_x = (page_min_x + page_max_x) / 2.0

    expanded: list[PageLayout] = []
    for layout in layouts:
        if layout.ref != "text":
            expanded.append(layout)
            continue

        parts = split_fused_vocab_line(layout.text)
        if len(parts) >= 2:
            center_x = (layout.det[0] + layout.det[2]) / 2.0
            wide = _is_page_wide_row(layout.det, page_min_x, page_max_x)
            # 双栏粘连通常来自左半/整宽 OCR；右半长块多为单栏连续词条
            if _should_deinterleave_columns(layout, parts) and (
                wide or center_x < page_mid_x
            ):
                left_parts = parts[0::2]
                right_parts = parts[1::2]
                expanded.extend(
                    _layouts_for_column_entries(
                        layout,
                        left_parts,
                        column="left",
                        page_min_x=page_min_x,
                        page_mid_x=page_mid_x,
                        page_max_x=page_max_x,
                    )
                )
                expanded.extend(
                    _layouts_for_column_entries(
                        layout,
                        right_parts,
                        column="right",
                        page_min_x=page_min_x,
                        page_mid_x=page_mid_x,
                        page_max_x=page_max_x,
                    )
                )
                continue

            if len(parts) == 2 and all(is_vocab_entry(p) for p in parts) and (
                wide or abs(center_x - page_mid_x) < (page_max_x - page_min_x) * 0.1
            ):
                # 单行左右词条（整宽或近中缝）
                left_det, right_det = _column_dets(
                    layout.det, page_min_x, page_mid_x, page_max_x
                )
                expanded.append(replace(layout, text=parts[0], det=left_det))
                expanded.append(
                    replace(layout, text=parts[1], det=right_det, order=layout.order)
                )
                continue

            # 单栏多词条：整段留在所在侧
            column = "left" if center_x < page_mid_x else "right"
            expanded.extend(
                _layouts_for_column_entries(
                    layout,
                    parts,
                    column=column,
                    page_min_x=page_min_x,
                    page_mid_x=page_mid_x,
                    page_max_x=page_max_x,
                )
            )
            continue

        if is_vocab_entry(layout.text) and _is_page_wide_row(
            layout.det, page_min_x, page_max_x
        ):
            left_det, right_det = _column_dets(
                layout.det, page_min_x, page_mid_x, page_max_x
            )
            center_x = (layout.det[0] + layout.det[2]) / 2.0
            expanded.append(
                replace(
                    layout,
                    det=left_det if center_x < page_mid_x else right_det,
                )
            )
            continue

        expanded.append(layout)

    return expanded


def split_dual_vocab_line(text: str) -> tuple[str, str] | None:
    """若一行含左右两个词条，返回 (left, right)，否则 None。"""
    parts = split_fused_vocab_line(text)
    if len(parts) == 2 and all(is_vocab_entry(part) for part in parts):
        return parts[0], parts[1]
    return None


def _should_deinterleave_columns(layout: PageLayout, parts: list[str]) -> bool:
    """Tall OCR blob with many row-wise L/R entries → candidate for column split."""
    if len(parts) < 4:
        return False
    if not all(is_vocab_entry(part) for part in parts):
        entry_count = sum(1 for part in parts if is_vocab_entry(part))
        if entry_count < 4:
            return False
    x1, y1, x2, y2 = layout.det
    height = max(y2 - y1, 1)
    width = max(x2 - x1, 1)
    return height >= max(120, int(width * 0.35))

def _layouts_for_column_entries(
    layout: PageLayout,
    parts: list[str],
    *,
    column: str,
    page_min_x: float,
    page_mid_x: float,
    page_max_x: float,
) -> list[PageLayout]:
    if not parts:
        return []
    left_det, right_det = _column_dets(
        layout.det, page_min_x, page_mid_x, page_max_x
    )
    base_det = left_det if column == "left" else right_det
    result: list[PageLayout] = []
    for index, text in enumerate(parts):
        result.append(
            replace(
                layout,
                text=text,
                det=_slice_det_vertically(base_det, index, len(parts)),
                order=layout.order,
            )
        )
    return result


def _column_dets(
    det: tuple[int, int, int, int],
    page_min_x: float,
    page_mid_x: float,
    page_max_x: float,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    """Place left/right entries into page halves (not just within the OCR crop).

    Column boxes are ~42% page width so x-gap clustering (EPUB columns mode)
    can still separate them.
    """
    _, y1, _, y2 = det
    page_width = max(page_max_x - page_min_x, 1.0)
    col_width = max(page_width * 0.42, 1.0)
    left = (
        int(page_min_x),
        y1,
        int(page_min_x + col_width),
        y2,
    )
    right = (
        int(page_max_x - col_width),
        y1,
        int(page_max_x),
        y2,
    )
    if left[2] >= right[0]:
        # Degenerate page width: fall back to mid split with a small gutter
        gap = max(4, int(page_width * 0.02))
        left = (int(page_min_x), y1, max(int(page_mid_x) - gap, int(page_min_x) + 1), y2)
        right = (min(int(page_mid_x) + gap, int(page_max_x) - 1), y1, int(page_max_x), y2)
    return left, right


def _slice_det_vertically(
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

def split_fused_vocab_line(text: str) -> list[str]:
    """将同一行里粘在一起的多条词条拆开（按课次标记切分）。

    例：
      「ウーロン茶 12※ ううん [叹词] 22」
        → 「ウーロン茶 12※」, 「ううん [叹词] 22」
    """
    stripped = re.sub(r"\s+", " ", (text or "").strip())
    if not stripped:
        return []

    parts: list[str] = []
    # 行首假名分区标题：あ ア / え エ（OCR 或把 エ 认成 工）
    section = re.match(
        r"^([ぁ-んァ-ヶー]+)\s+([A-Zァ-ヶーエ工]+)\s+(.+)$",
        stripped,
    )
    if (
        section is not None
        and not re.search(r"\d", section.group(1) + section.group(2))
        and _PAGE_REFS.search(section.group(3))
    ):
        parts.append(f"{section.group(1)} {section.group(2)}")
        stripped = section.group(3).strip()

    if is_vocab_entry(stripped) and not _has_another_entry_after_first(stripped):
        parts.append(stripped)
        return parts

    remaining = stripped
    while remaining:
        cut = _first_vocab_entry_end(remaining)
        if cut is None:
            parts.append(remaining)
            break
        parts.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()

    return [part for part in parts if part]


def _has_another_entry_after_first(text: str) -> bool:
    for match in _PAGE_REFS.finditer(text):
        left = text[: match.end()].strip()
        right = text[match.end() :].strip()
        if is_vocab_entry(left) and right and (
            is_vocab_entry(right) or _first_vocab_entry_end(right) is not None
        ):
            return True
    return False


def _first_vocab_entry_end(text: str) -> int | None:
    """Return end offset of the first vocab entry in text, or None."""
    for match in _PAGE_REFS.finditer(text):
        candidate = text[: match.end()].strip()
        if not is_vocab_entry(candidate):
            continue
        rest = text[match.end() :]
        if rest and not rest[0].isspace():
            continue
        return match.end()
    return None


def is_vocab_entry(text: str) -> bool:
    return _VOCAB_ENTRY.match(text.strip()) is not None


def _is_page_wide_row(
    det: tuple[int, int, int, int],
    page_min_x: float,
    page_max_x: float,
) -> bool:
    x1, _, x2, _ = det
    page_width = max(page_max_x - page_min_x, 1)
    return (x2 - x1) >= page_width * 0.7


def _is_full_width_row(det: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = det
    width = x2 - x1
    height = max(y2 - y1, 1)
    return width >= height * 8
