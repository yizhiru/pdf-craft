#!/usr/bin/env python3
"""DeepSeek OCR → brand-new text PDF (searchable & copyable).

Does NOT keep scanned page images. OCR characters are typeset as visible
Unicode text into a new PDF, so any reader can search and copy.

Pipeline:
  1. recognize — header + left/right columns sliced into short bands
  2. normalize — drop degeneration loops, keep ※/▼, left then right
  3. compose   — typeset plain text into a new PDF

Examples:

    conda activate craft
    pip install pymupdf

    # Reuse existing DeepSeek OCR cache (no GPU)
    python scripts/ocr_searchable_pdf.py \\
      --pdf analysing/words_sample/sample_2pages.pdf \\
      --ocr-path analysing/words_sample/work/ocr \\
      --output analysing/words_sample/sample_2pages_ocr.pdf

    # Resume missing pages (keeps existing page_*.xml)
    python scripts/ocr_searchable_pdf.py \\
      --pdf analysing/words_sample/sample_2pages.pdf \\
      --ocr-path analysing/words_sample/work/ocr \\
      --output analysing/words_sample/sample_2pages_ocr.pdf \\
      --run-ocr --cuda-device 1

    # Force column OCR then compose
    python scripts/ocr_searchable_pdf.py \\
      --pdf analysing/words_sample/sample_2pages.pdf \\
      --ocr-path analysing/words_sample/work/ocr \\
      --output analysing/words_sample/sample_2pages_ocr.pdf \\
      --force-ocr --cuda-device 1

    # Re-OCR one page (e.g. last index page) without wiping the rest
    python scripts/ocr_searchable_pdf.py \\
      --pdf intermediate_words.pdf \\
      --ocr-path analysing/words_full/ocr \\
      --output analysing/words_full/page64_sliced.pdf \\
      --pages 64 --force-ocr --run-ocr --cuda-device 1
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _project_root() -> Path:
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _parse_args(project_root: Path) -> argparse.Namespace:
    default_pdf = project_root / "analysing" / "words_sample" / "sample_2pages.pdf"
    default_ocr = project_root / "analysing" / "words_sample" / "work" / "ocr"
    default_out = project_root / "analysing" / "words_sample" / "sample_2pages_ocr.pdf"
    default_models = Path(
        os.environ.get(
            "PDF_CRAFT_MODELS",
            "/mnt/zzbnew/peixunban/gl/models/DeepSeek-OCR-2",
        )
    )

    parser = argparse.ArgumentParser(
        description="DeepSeek OCR → new text PDF (no scanned pages)."
    )
    parser.add_argument("--pdf", type=Path, default=default_pdf)
    parser.add_argument("--output", "-o", type=Path, default=default_out)
    parser.add_argument("--ocr-path", type=Path, default=default_ocr)
    parser.add_argument("--run-ocr", action="store_true",
                        help="Run OCR; skip pages that already have page_*.xml (resume).")
    parser.add_argument("--force-ocr", action="store_true",
                        help="Delete OCR cache and re-recognize every page.")
    parser.add_argument("--models-cache-path", type=Path, default=default_models)
    parser.add_argument(
        "--cuda-device",
        default=os.environ.get("CUDA_VISIBLE_DEVICES", "1"),
    )
    parser.add_argument(
        "--ocr-size",
        default="gundam",
        choices=["tiny", "small", "base", "large", "gundam"],
        help=(
            "DeepSeek input scale (default: gundam). "
            "DeepSeek-OCR-2 only supports gundam/base; large/tiny/small are remapped to gundam."
        ),
    )
    parser.add_argument(
        "--ocr-model",
        default="deepseek-ocr-2",
        choices=["deepseek-ocr", "deepseek-ocr-2"],
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=400,
        help="PDF raster DPI before OCR (default: 400; higher helps small kana).",
    )
    parser.add_argument("--fontname", default="japan")
    parser.add_argument("--fontsize", type=float, default=12.0)
    parser.add_argument("--title", default=None)
    parser.add_argument(
        "--column-ocr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="OCR left/right column crops separately (default: on)",
    )
    parser.add_argument(
        "--slice-bands",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Split each column into short horizontal bands (default: on). "
        "Prevents DeepSeek looping/truncating dense vocab pages.",
    )
    parser.add_argument(
        "--band-lines",
        type=int,
        default=8,
        help="Approx. vocab lines per column band (default: 8)",
    )
    parser.add_argument(
        "--pages",
        default=None,
        help="Only these 1-based pages, e.g. 64 or 1-3,64 (default: all)",
    )
    parser.add_argument(
        "--local-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def _parse_pages(value: str | None) -> list[int] | None:
    if not value:
        return None
    pages: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            if end < start:
                raise ValueError(f"Invalid page range: {part}")
            pages.update(range(start, end + 1))
        else:
            pages.add(int(part))
    return sorted(pages)


def main() -> int:
    project_root = _project_root()
    args = _parse_args(project_root)

    if not args.pdf.is_file():
        print(f"Error: PDF not found: {args.pdf}", file=sys.stderr)
        return 1

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("CUDNN_BENCHMARK", "0")

    from pdf_craft import OCREventKind
    from pdf_craft.ocr_pdf import build_ocr_pdf

    page_indexes = _parse_pages(args.pages)
    print("Pipeline: DeepSeek sliced-column OCR → normalize → typeset NEW text PDF")
    print(f"PDF:        {args.pdf}")
    print(f"OCR path:   {args.ocr_path}")
    print(f"Output:     {args.output}")
    print(f"Run OCR:    {args.run_ocr or args.force_ocr}")
    print(f"Column OCR: {args.column_ocr}")
    print(f"Slice bands:{args.slice_bands} (band-lines={args.band_lines})")
    print(f"Pages:      {page_indexes or 'all'}")
    print(f"OCR size:   {args.ocr_size}")
    print(f"DPI:        {args.dpi}")
    print()

    def on_event(event) -> None:
        print(
            f"OCR {OCREventKind(event.kind).name} "
            f"- Page {event.page_index}/{event.total_pages}",
            flush=True,
        )

    try:
        n_pages = build_ocr_pdf(
            pdf_path=args.pdf,
            output_path=args.output,
            ocr_path=args.ocr_path,
            run_ocr=bool(args.run_ocr or args.force_ocr),
            force_ocr=bool(args.force_ocr),
            models_cache_path=args.models_cache_path,
            ocr_model=args.ocr_model,
            ocr_size=args.ocr_size,
            dpi=args.dpi,
            local_only=args.local_only,
            column_split=args.column_ocr,
            slice_bands=args.slice_bands,
            band_lines=args.band_lines,
            page_indexes=page_indexes,
            fontname=args.fontname,
            fontsize=args.fontsize,
            title=args.title,
            on_ocr_event=on_event,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        cause = exc.__cause__ or exc.__context__
        if cause is not None and cause is not exc:
            print(f"Cause: {type(cause).__name__}: {cause}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1

    print(f"Done: {args.output} ({n_pages} page(s) of typeset text)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
