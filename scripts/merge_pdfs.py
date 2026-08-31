#!/usr/bin/env python3
"""Merge two or more PDFs into a single PDF, in the given order.

Examples:

    python scripts/merge_pdfs.py -o merged.pdf a.pdf b.pdf
    python scripts/merge_pdfs.py -o all.pdf part1.pdf part2.pdf part3.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def merge_pdfs(inputs: list[Path], output_pdf: Path) -> int:
    from pypdf import PdfReader, PdfWriter

    if len(inputs) < 2:
        raise ValueError("Need at least two input PDFs to merge")

    writer = PdfWriter()
    total_pages = 0

    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(f"Input PDF not found: {path}")
        reader = PdfReader(str(path))
        page_count = len(reader.pages)
        if page_count == 0:
            raise ValueError(f"PDF has no pages: {path}")
        for page in reader.pages:
            writer.add_page(page)
        total_pages += page_count

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with output_pdf.open("wb") as f:
        writer.write(f)

    return total_pages


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Merge two or more PDFs into one PDF (order preserved)."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Input PDF paths (at least two), merged in this order",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output PDF path",
    )
    args = parser.parse_args(argv)

    try:
        total_pages = merge_pdfs(args.inputs, args.output)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    names = ", ".join(str(p) for p in args.inputs)
    print(f"Merged {len(args.inputs)} PDFs ({total_pages} page(s)): {names} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
