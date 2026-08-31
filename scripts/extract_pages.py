#!/usr/bin/env python3
"""Extract a page range from a PDF and save it as a new PDF.

Page numbers are 1-based and inclusive. Examples:

    python scripts/extract_pages.py --input book.pdf --start 3 --end 10 --output part.pdf
    python scripts/extract_pages.py -i book.pdf -s 5 -e 5 -o page5.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def extract_pages(input_pdf: Path, output_pdf: Path, start: int, end: int) -> int:
    from pypdf import PdfReader, PdfWriter

    if not input_pdf.is_file():
        raise FileNotFoundError(f"Input PDF not found: {input_pdf}")
    if start < 1:
        raise ValueError(f"--start must be >= 1, got {start}")
    if end < start:
        raise ValueError(f"--end ({end}) must be >= --start ({start})")

    reader = PdfReader(str(input_pdf))
    total = len(reader.pages)
    if total == 0:
        raise ValueError(f"PDF has no pages: {input_pdf}")
    if end > total:
        raise ValueError(f"--end ({end}) exceeds total pages ({total})")

    writer = PdfWriter()
    for page_index in range(start - 1, end):
        writer.add_page(reader.pages[page_index])

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with output_pdf.open("wb") as f:
        writer.write(f)

    return end - start + 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract pages [start, end] from a PDF into a new PDF (1-based, inclusive)."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Source PDF path",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output PDF path",
    )
    parser.add_argument(
        "-s",
        "--start",
        type=int,
        required=True,
        help="Start page (1-based, inclusive)",
    )
    parser.add_argument(
        "-e",
        "--end",
        type=int,
        required=True,
        help="End page (1-based, inclusive)",
    )
    args = parser.parse_args(argv)

    try:
        count = extract_pages(args.input, args.output, args.start, args.end)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Saved pages {args.start}-{args.end} ({count} page(s)) "
        f"from {args.input} -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
