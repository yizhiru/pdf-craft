"""Post-process OCR vocab lines: HTML junk, degeneration loops, orphan marks."""

from __future__ import annotations

import html
import re

from ..pdf.ngrams import trim_repetitive_tail
from ..sequence.vocab_split import split_fused_vocab_line

_TAG = re.compile(r"<[^>]+>")
_TR_BLOCK = re.compile(r"<tr\b[^>]*>.*?</tr>", re.IGNORECASE | re.DOTALL)
_TD_INNER = re.compile(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", re.IGNORECASE | re.DOTALL)
_PSEUDO_TAG = re.compile(r"<([^/>][^>]*)>")
_HTML_TAG_NAMES = frozenset(
    {
        "a",
        "b",
        "br",
        "caption",
        "col",
        "colgroup",
        "div",
        "em",
        "font",
        "h1",
        "h2",
        "h3",
        "hr",
        "i",
        "img",
        "li",
        "ol",
        "p",
        "span",
        "strong",
        "table",
        "tbody",
        "td",
        "tfoot",
        "th",
        "thead",
        "tr",
        "u",
        "ul",
    }
)
_TD_JUNK = re.compile(r"[（(]\s*t[dh][^）)]*[）)]", re.IGNORECASE)
_PAGE_TAIL = re.compile(
    r"(?:\s+\d+[※▼]?(?:\s*,\s*\d+[※▼]?)*)+\s*$"
)
_ORPHAN_MARK = re.compile(r"^[※▼]+$")
_HAS_CJK = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]")
_MATH_JUNK = re.compile(
    r"已知|\\frac|\\left|\\right|求\s*c\s*=|The quick brown fox",
    re.I,
)
_CHECKBOX_ONLY = re.compile(r"^[☐☑□■\s]+$")
_NUMBERED_KNOWN = re.compile(r"\d+\.\s*已知")
_LATIN_WORD = re.compile(r"[A-Za-z]{2,}")
_ALNUM_HEAD = re.compile(r"^\d+[A-Za-z]")
_GROUNDING_BLOCK = re.compile(
    r"<\|ref\|>.*?<\|/ref\|>\s*<\|det\|>\[\[.*?\]\]<\|/det\|>",
    re.DOTALL,
)
_LEAD_JUNK = re.compile(r"^[)\]】>Vｖ◀▶\s]+")
_LEAD_MARK = re.compile(r"^([※▼]+)\s+(.+)$")
_MARKDOWN_HEAD = re.compile(r"^#+\s+")


def is_usable_vocab_text(text: str) -> bool:
    value = (text or "").strip()
    if not value:
        return False
    if _CHECKBOX_ONLY.fullmatch(value):
        return False
    if _MATH_JUNK.search(value):
        return False
    if len(_NUMBERED_KNOWN.findall(value)) >= 2:
        return False
    if _HAS_CJK.search(value) or re.search(r"[ァ-ヶーぁ-ん]", value):
        return True
    if _ALNUM_HEAD.search(value) and re.search(r"\d", value):
        return True
    if _LATIN_WORD.search(value) and re.search(r"\d", value):
        return True
    return False


def clean_pseudo_tags(text: str) -> str:
    """Keep Japanese angle-bracket notes; drop leftover HTML such as ``<td>``."""

    def _replace(match: re.Match[str]) -> str:
        inner = (match.group(1) or "").strip()
        name = inner.split(None, 1)[0].lower().rstrip("/") if inner else ""
        if name in _HTML_TAG_NAMES:
            return " "
        if _HAS_CJK.search(inner):
            return f"（{inner}）"
        return " "

    value = html.unescape(text or "")
    value = _PSEUDO_TAG.sub(_replace, value)
    value = _TAG.sub(" ", value)
    value = _TD_JUNK.sub(" ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def clean_vocab_line(text: str) -> str:
    """Strip OCR junk around a vocab line and move leading ※/▼ onto the lesson number."""
    value = _MARKDOWN_HEAD.sub("", clean_pseudo_tags(text)).strip()
    value = _LEAD_JUNK.sub("", value).strip()
    value = re.sub(r"[Vv]+$", "", value).strip()
    lead = _LEAD_MARK.match(value)
    if lead is not None:
        marks, rest = lead.group(1), lead.group(2).strip()
        if re.search(r"[※▼]", rest):
            value = rest
        elif re.search(r"\d+\s*$", rest):
            value = rest + marks
        else:
            value = rest
    return value


def headword_key(text: str) -> str:
    """Normalize a vocab line to its headword for loop / dedupe checks."""
    value = clean_vocab_line(text)
    value = value.lstrip("※▼● ")
    value = _PAGE_TAIL.sub("", value)
    value = re.sub(r"\s+", "", value)
    return value


def cut_repeating_entries(
    entries: list[str],
    *,
    min_cycles: int = 3,
    max_period: int = 6,
) -> list[str]:
    """Keep the first cycle of a consecutive repeating block, drop the rest.

    ``了承, 涼風, 料理人, 緑化, 利率, 料理人, 緑化, 利率, …``
    → ``了承, 涼風, 料理人, 緑化, 利率``.

    Only identical complete entries count as repeats. Similar headwords and
    different lesson references can be legitimate vocabulary entries.
    """
    if len(entries) < min_cycles * 2:
        return entries

    keys = [re.sub(r"\s+", "", clean_vocab_line(item)) for item in entries]
    n = len(keys)
    i = 0
    kept: list[str] = []
    while i < n:
        cut_at: int | None = None
        period_used = 0
        for period in range(1, max_period + 1):
            if i + period * min_cycles > n:
                break
            block = keys[i : i + period]
            if not all(block):
                continue
            repeats = 1
            pos = i + period
            while pos + period <= n and keys[pos : pos + period] == block:
                repeats += 1
                pos += period
            if repeats >= min_cycles:
                cut_at = i + period
                period_used = pos
                break
        if cut_at is not None:
            kept.extend(entries[i:cut_at])
            i = period_used
            continue
        kept.append(entries[i])
        i += 1
    return kept


def dedupe_adjacent(entries: list[str], *, window: int = 16) -> list[str]:
    """Drop near-duplicate headwords from overlapping OCR bands.

    Distinct lesson numbers on the same headword (``腰痛 20`` vs ``腰痛 27▼``)
    are kept. Incomplete OCR of the same entry is merged into the richer line.
    """
    result: list[str] = []
    for item in entries:
        text = (item or "").strip()
        if not text:
            continue
        if not is_usable_vocab_text(text) and not _ORPHAN_MARK.fullmatch(text):
            continue
        key = headword_key(text)
        if not key:
            result.append(text)
            continue
        replaced = False
        start = max(0, len(result) - window)
        for index in range(len(result) - 1, start - 1, -1):
            prev = result[index]
            if headword_key(prev) != key:
                continue
            if not _same_lesson_entry(prev, text):
                continue
            if _richer_page_refs(text, prev):
                result[index] = text
            replaced = True
            break
        if not replaced:
            result.append(text)
    return result


def attach_orphan_marks(entries: list[str]) -> list[str]:
    """Join a lone ``※`` / ``▼`` onto the previous lesson number."""
    result: list[str] = []
    for item in entries:
        text = (item or "").strip()
        if not text:
            continue
        if result and _ORPHAN_MARK.fullmatch(text):
            prev = result[-1]
            if re.search(r"\d+$", prev):
                result[-1] = prev + text
                continue
            if not prev.endswith(("※", "▼")):
                result[-1] = f"{prev} {text}"
                continue
        result.append(text)
    return result


def trim_repeating_table_html(text: str, *, min_cycles: int = 3) -> str:
    """Drop degenerated repeating ``<tr>`` cycles inside an HTML table."""
    value = text or ""
    rows = list(_TR_BLOCK.finditer(value))
    if len(rows) < min_cycles * 2:
        return value

    row_html = [match.group(0) for match in rows]

    kept_html = cut_repeating_entries(row_html, min_cycles=min_cycles)
    if len(kept_html) >= len(row_html):
        return value
    # Keep all material between rows, including text between separate tables.
    kept = iter(kept_html)
    next_row = next(kept, None)
    chunks: list[str] = []
    offset = 0
    for row in rows:
        chunks.append(value[offset:row.start()])
        if row.group(0) == next_row:
            chunks.append(row.group(0))
            next_row = next(kept, None)
        offset = row.end()
    chunks.append(value[offset:])
    return "".join(chunks)


def lines_from_ocr_response(text: str) -> list[str]:
    """Turn a Free-OCR / grounding dump into candidate vocab lines."""
    value = _GROUNDING_BLOCK.sub("\n", text or "")
    value = trim_repetitive_tail(value)
    lines: list[str] = []
    # Parse table rows in place: an OCR response may mix prose and several tables.
    offset = 0
    for row in _TR_BLOCK.finditer(value):
        lines.extend(_plain_vocab_lines(value[offset:row.start()]))
        cells = [clean_pseudo_tags(cell) for cell in _TD_INNER.findall(row.group(0))]
        cells = [cell for cell in cells if cell]
        # A crop can start with the previous row's trailing lesson number.
        while cells and _looks_like_page_ref(cells[0]):
            cells.pop(0)
        lines.extend(_plain_vocab_lines(" ".join(cells)))
        offset = row.end()
    lines.extend(_plain_vocab_lines(value[offset:]))
    return lines


def _plain_vocab_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw in text.splitlines():
        line = clean_vocab_line(raw)
        if line:
            lines.extend(split_fused_vocab_line(line) or [line])
    return lines


def _looks_like_page_ref(text: str) -> bool:
    return bool(re.fullmatch(r"\d+[※▼]?(?:\s*,\s*\d+[※▼]?)*", (text or "").strip()))


def cleanup_vocab_lines(lines: list[str]) -> list[str]:
    cleaned: list[str] = []
    for line in lines:
        value = clean_vocab_line(line)
        if not value:
            continue
        cleaned.extend(split_fused_vocab_line(value) or [value])
    cleaned = attach_orphan_marks(cleaned)
    cleaned = [line for line in cleaned if is_usable_vocab_text(line)]
    cleaned = cut_repeating_entries(cleaned)
    cleaned = dedupe_adjacent(cleaned)
    return [line for line in cleaned if line]


def _richer_page_refs(candidate: str, current: str) -> bool:
    cand_refs = len(re.findall(r"\d+", candidate))
    cur_refs = len(re.findall(r"\d+", current))
    cand_marks = candidate.count("※") + candidate.count("▼")
    cur_marks = current.count("※") + current.count("▼")
    return (cand_refs, cand_marks, len(candidate)) > (cur_refs, cur_marks, len(current))


def _entry_numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+", text))


def _same_lesson_entry(left: str, right: str) -> bool:
    nums_left = _entry_numbers(left)
    nums_right = _entry_numbers(right)
    if not nums_left or not nums_right:
        return True
    return nums_left == nums_right or nums_left <= nums_right or nums_right <= nums_left
