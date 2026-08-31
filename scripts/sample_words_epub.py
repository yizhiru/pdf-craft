#!/usr/bin/env python3
"""Sample: extract 2 pages from 标日教材词汇表 PDF and convert to EPUB.

中文混日文、横版双栏。OCR 用 columns 阅读序：先完整读完左列（上→下），
再追加右列；EPUB 不保留两列版式，按单栏横排输出。

Examples:

    conda activate craft
    python scripts/sample_words_epub.py

    python scripts/sample_words_epub.py --start 1 --end 2 --cuda-device 0
    python scripts/sample_words_epub.py --start 3 --end 4 --force-ocr --generate-plot
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


def _project_root() -> Path:
    root = Path(__file__).resolve().parent.parent
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def _extract_pages(input_pdf: Path, output_pdf: Path, start: int, end: int) -> int:
    import importlib.util

    extract_path = Path(__file__).resolve().parent / "extract_pages.py"
    spec = importlib.util.spec_from_file_location("extract_pages", extract_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {extract_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.extract_pages(input_pdf, output_pdf, start, end)


def _format_duration(ms: int) -> str:
    if ms < 1000:
        return f"{ms}ms"
    if ms < 60000:
        return f"{ms / 1000:.2f}s"
    minutes = ms // 60000
    seconds = (ms % 60000) / 1000
    return f"{minutes}m {seconds:.2f}s"


def _parse_args(project_root: Path) -> argparse.Namespace:
    default_pdf = project_root / "elmentary_words.pdf"
    default_sample = project_root / "analysing" / "words_sample" / "sample_2pages.pdf"
    default_epub = project_root / "analysing" / "words_sample" / "sample_2pages.epub"
    default_analysing = project_root / "analysing" / "words_sample" / "work"
    default_models = Path(
        os.environ.get(
            "PDF_CRAFT_MODELS",
            "/mnt/zzbnew/peixunban/gl/models/DeepSeek-OCR-2",
        )
    )

    parser = argparse.ArgumentParser(
        description="Extract 2 pages from 标日词汇表 PDF and convert to horizontal EPUB."
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=default_pdf,
        help=f"Source vocabulary PDF (default: {default_pdf})",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="Start page, 1-based inclusive (default: 1)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=2,
        help="End page, 1-based inclusive (default: 2)",
    )
    parser.add_argument(
        "--sample-pdf",
        type=Path,
        default=default_sample,
        help="Where to write the extracted 2-page PDF",
    )
    parser.add_argument(
        "--epub",
        type=Path,
        default=default_epub,
        help="Output EPUB path",
    )
    parser.add_argument(
        "--analysing-path",
        type=Path,
        default=default_analysing,
        help="Temporary working directory for OCR / chapters",
    )
    parser.add_argument(
        "--models-cache-path",
        type=Path,
        default=default_models,
        help="DeepSeek OCR model cache directory",
    )
    parser.add_argument(
        "--cuda-device",
        default=os.environ.get("CUDA_VISIBLE_DEVICES", "1"),
        help="GPU index for CUDA_VISIBLE_DEVICES (default: 1)",
    )
    parser.add_argument(
        "--ocr-size",
        default="gundam",
        choices=["tiny", "small", "base", "large", "gundam"],
        help="DeepSeek OCR resolution preset",
    )
    parser.add_argument(
        "--ocr-model",
        default="deepseek-ocr-2",
        choices=["deepseek-ocr", "deepseek-ocr-2"],
        help="DeepSeek OCR model family",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PDF page render DPI",
    )
    parser.add_argument(
        "--title",
        default="标日教材词汇表（样例）",
        help="EPUB book title",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip PDF page extraction; convert --sample-pdf as-is",
    )
    parser.add_argument(
        "--force-ocr",
        action="store_true",
        help="Clear OCR cache under analysing-path and re-run",
    )
    parser.add_argument(
        "--generate-plot",
        action="store_true",
        help="Save OCR layout debug images under analysing-path/plots",
    )
    parser.add_argument(
        "--local-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use local models only (default: true)",
    )
    return parser.parse_args()


def main() -> int:
    project_root = _project_root()
    args = _parse_args(project_root)

    if args.cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("CUDNN_BENCHMARK", "0")

    if not args.pdf.is_file() and not args.skip_extract:
        print(f"Error: source PDF not found: {args.pdf}", file=sys.stderr)
        return 1

    if not args.skip_extract:
        count = _extract_pages(args.pdf, args.sample_pdf, args.start, args.end)
        print(
            f"Extracted pages {args.start}-{args.end} ({count} page(s)) "
            f"-> {args.sample_pdf}"
        )
    elif not args.sample_pdf.is_file():
        print(f"Error: sample PDF not found: {args.sample_pdf}", file=sys.stderr)
        return 1

    if args.local_only and not args.models_cache_path.exists():
        print(
            f"Error: model cache not found: {args.models_cache_path}",
            file=sys.stderr,
        )
        return 1

    args.epub.parent.mkdir(parents=True, exist_ok=True)
    args.analysing_path.mkdir(parents=True, exist_ok=True)

    if args.force_ocr:
        for name in ("ocr", "chapters", "assets", "plots"):
            cached = args.analysing_path / name
            if cached.exists():
                shutil.rmtree(cached)
        for name in ("toc.xml", "cover.png"):
            path = args.analysing_path / name
            if path.exists():
                path.unlink()
    else:
        # 阅读序/词条拆分只影响 chapters；保留 OCR 缓存即可重排
        chapters = args.analysing_path / "chapters"
        if chapters.exists():
            shutil.rmtree(chapters)
        toc_file = args.analysing_path / "toc.xml"
        if toc_file.exists():
            toc_file.unlink()

    import torch

    torch.backends.cudnn.benchmark = False

    from epub_generator import LaTeXRender, TableRender

    from pdf_craft import BookMeta, OCREventKind, transform_epub

    print(f"Sample PDF:     {args.sample_pdf}")
    print(f"EPUB:           {args.epub}")
    print(f"Analysing path: {args.analysing_path}")
    print(f"Models:         {args.models_cache_path}")
    print(f"GPU:            {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
    print("Text layout:    columns (left column fully, then right; append)")
    print("EPUB layout:    horizontal (single-column flow)")
    print()

    metering = transform_epub(
        pdf_path=args.sample_pdf,
        epub_path=args.epub,
        analysing_path=args.analysing_path,
        ocr_size=args.ocr_size,
        ocr_model=args.ocr_model,
        text_layout="columns",
        epub_layout="horizontal",
        models_cache_path=args.models_cache_path,
        local_only=args.local_only,
        dpi=args.dpi,
        max_page_image_file_size=None,
        includes_cover=False,
        includes_footnotes=True,
        ignore_pdf_errors=False,
        ignore_ocr_errors=False,
        generate_plot=args.generate_plot,
        toc_assumed=False,
        book_meta=BookMeta(
            title=args.title,
            authors=["标日教材"],
            publisher="sample",
        ),
        lan="zh",
        table_render=TableRender.HTML,
        latex_render=LaTeXRender.MATHML,
        inline_latex=True,
        on_ocr_event=lambda event: print(
            f"OCR {OCREventKind(event.kind).name} "
            f"- Page {event.page_index}/{event.total_pages} "
            f"- {_format_duration(event.cost_time_ms)}"
            + (f" - ERROR: {event.error}" if event.error is not None else ""),
            flush=True,
        ),
    )

    print()
    print(f"Done: {args.epub}")
    print(
        "OCR tokens: "
        f"input={metering.input_tokens}, output={metering.output_tokens}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
