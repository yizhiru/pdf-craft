#!/usr/bin/env python3
"""Convert PDF to EPUB on a GPU server (DeepSeek OCR + optional vertical layout).

Run from source without `pip install pdf-craft`:

    cd /path/to/pdf-craft
    python scripts/convert.py --pdf input.pdf --epub output.epub

Third-party deps (torch, epub-generator, doc-page-extractor, etc.) still need
to be installed in the active environment.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import re
import shutil
import sys
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import TextIO


def _setup_project_root(project_root: Path | None = None) -> Path:
    """Put the repo root on sys.path so `import pdf_craft` works without install."""
    root = project_root or Path(__file__).resolve().parent.parent
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    if not (root / "pdf_craft").is_dir():
        raise FileNotFoundError(
            f"pdf_craft source not found under {root}. "
            "Pass --project-root if the repo lives elsewhere."
        )
    return root


def _configure_gpu(cuda_device: str | None) -> None:
    if cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

    # Avoid CUDNN_STATUS_NOT_INITIALIZED on some GPU servers.
    os.environ.setdefault("CUDNN_BENCHMARK", "0")


def _configure_quiet_ocr() -> None:
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

    warnings.filterwarnings("ignore", category=UserWarning, module=r"transformers\..*")
    warnings.filterwarnings(
        "ignore", category=FutureWarning, module=r"transformers\..*"
    )

    for logger_name in (
        "transformers",
        "transformers.modeling_utils",
        "transformers.generation",
        "transformers.configuration_utils",
    ):
        logging.getLogger(logger_name).setLevel(logging.ERROR)


class _LineFilteringStream:
    def __init__(
        self,
        stream: TextIO,
        drop_patterns: tuple[re.Pattern[str], ...],
    ) -> None:
        self._stream = stream
        self._drop_patterns = drop_patterns
        self._buffer = ""

    def write(self, text: str) -> int:
        if not text:
            return 0
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if not self._should_drop(line):
                self._stream.write(line + "\n")
        return len(text)

    def flush(self) -> None:
        if self._buffer:
            if not self._should_drop(self._buffer.rstrip("\r")):
                self._stream.write(self._buffer)
            self._buffer = ""
        self._stream.flush()

    def _should_drop(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        return any(pattern.search(stripped) for pattern in self._drop_patterns)

    def __getattr__(self, name: str):
        return getattr(self._stream, name)


@contextlib.contextmanager
def _quiet_ocr_runtime() -> Iterator[None]:
    stdout_patterns = (
        re.compile(r"^=+$"),
        re.compile(r"^BASE:\s"),
        re.compile(r"^PATCHES:\s"),
    )
    stderr_patterns = (
        re.compile(r"You are using a model of type deepseek_vl_v2"),
        re.compile(r"Some weights of DeepseekOCRForCausalLM"),
        re.compile(r"You should probably TRAIN this model"),
        re.compile(r"The attention mask (and|is)"),
        re.compile(r"Setting `pad_token_id`"),
        re.compile(r"The `seen_tokens` attribute is deprecated"),
        re.compile(r"`get_max_cache\(\)` is deprecated"),
        re.compile(r"The attention layers in this model are transitioning"),
        re.compile(r"UserWarning:.*do_sample"),
    )

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = _LineFilteringStream(old_stdout, stdout_patterns)
    sys.stderr = _LineFilteringStream(old_stderr, stderr_patterns)
    try:
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = old_stdout
        sys.stderr = old_stderr


@contextlib.contextmanager
def _maybe_quiet_ocr_runtime(quiet: bool) -> Iterator[None]:
    if quiet:
        with _quiet_ocr_runtime():
            yield
    else:
        yield


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert PDF to EPUB using pdf-craft.")
    parser.add_argument(
        "--project-root",
        default=os.environ.get("PDF_CRAFT_ROOT"),
        help="Path to pdf-craft repo root (env: PDF_CRAFT_ROOT; default: parent of scripts/)",
    )
    parser.add_argument(
        "--pdf",
        default=os.environ.get(
            "PDF_CRAFT_PDF",
            "/mnt/zjl-bgi-zzb/peixunban/gl/jyzheng/data/test_20.pdf",
        ),
        help="Input PDF path (env: PDF_CRAFT_PDF)",
    )
    parser.add_argument(
        "--epub",
        default=os.environ.get(
            "PDF_CRAFT_EPUB",
            "/mnt/zjl-bgi-zzb/peixunban/gl/jyzheng/data/output.epub",
        ),
        help="Output EPUB path (env: PDF_CRAFT_EPUB)",
    )
    parser.add_argument(
        "--analysing-path",
        default=os.environ.get(
            "PDF_CRAFT_ANALYSING",
            "/mnt/zjl-bgi-zzb/peixunban/gl/jyzheng/temp",
        ),
        help="Temporary working directory (env: PDF_CRAFT_ANALYSING)",
    )
    parser.add_argument(
        "--models-cache-path",
        default=os.environ.get(
            "PDF_CRAFT_MODELS",
            "/mnt/zzbnew/peixunban/gl/models/DeepSeek-OCR-2",
        ),
        help="DeepSeek OCR model cache directory (env: PDF_CRAFT_MODELS)",
    )
    parser.add_argument(
        "--cuda-device",
        default=os.environ.get("CUDA_VISIBLE_DEVICES", "1"),
        help="GPU index passed to CUDA_VISIBLE_DEVICES (env: CUDA_VISIBLE_DEVICES)",
    )
    parser.add_argument(
        "--ocr-size",
        default=os.environ.get("PDF_CRAFT_OCR_SIZE", "gundam"),
        choices=["tiny", "small", "base", "large", "gundam"],
        help="DeepSeek OCR resolution preset (applies to v1 and v2)",
    )
    parser.add_argument(
        "--ocr-model",
        default=os.environ.get("PDF_CRAFT_OCR_MODEL", "deepseek-ocr-2"),
        choices=["deepseek-ocr", "deepseek-ocr-2"],
        help="DeepSeek OCR model family (deepseek-ocr-2 = DeepSeek-OCR-2 on Hugging Face)",
    )
    parser.add_argument(
        "--text-layout",
        default=os.environ.get("PDF_CRAFT_TEXT_LAYOUT", "vertical"),
        choices=["horizontal", "vertical", "columns"],
        help=(
            "OCR reading order: horizontal=OCR order; "
            "columns=left column fully then right (dual-column vocab); "
            "vertical=right-to-left columns for tategaki PDFs"
        ),
    )
    parser.add_argument(
        "--epub-layout",
        default=os.environ.get("PDF_CRAFT_EPUB_LAYOUT", "horizontal"),
        choices=["horizontal", "vertical"],
        help="EPUB display (horizontal = no vertical-rl CSS; vertical = tategaki EPUB)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=int(os.environ.get("PDF_CRAFT_DPI", "300")),
        help="PDF page render DPI",
    )
    parser.add_argument(
        "--title",
        default=os.environ.get("PDF_CRAFT_TITLE", "Book Title"),
        help="Book title metadata",
    )
    parser.add_argument(
        "--author",
        action="append",
        default=None,
        dest="authors",
        help="Book author metadata (repeatable)",
    )
    parser.add_argument(
        "--publisher",
        default=os.environ.get("PDF_CRAFT_PUBLISHER", "Publisher"),
        help="Publisher metadata",
    )
    parser.add_argument(
        "--local-only",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("PDF_CRAFT_LOCAL_ONLY", "1") == "1",
        help="Use local models only; do not download from Hugging Face",
    )
    parser.add_argument(
        "--offline-hub",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("HF_HUB_OFFLINE", "1") == "1",
        help="Set HF_HUB_OFFLINE=1 for air-gapped GPU servers",
    )
    parser.add_argument(
        "--includes-cover",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--includes-footnotes",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--generate-plot",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save OCR layout debug images under analysing_path/plots",
    )
    parser.add_argument(
        "--ignore-pdf-errors",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--ignore-ocr-errors",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--quiet",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("PDF_CRAFT_QUIET", "0") == "1",
        help="Suppress DeepSeek-OCR / transformers noise; keep OCR progress lines",
    )
    parser.add_argument(
        "--force-ocr",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("PDF_CRAFT_FORCE_OCR", "0") == "1",
        help="Delete cached OCR under analysing-path/ocr and re-run recognition",
    )
    return parser.parse_args()


def _format_duration(ms: int) -> str:
    if ms < 1000:
        return f"{ms}ms"
    if ms < 60000:
        return f"{ms / 1000:.2f}s"
    minutes = ms // 60000
    seconds = (ms % 60000) / 1000
    return f"{minutes}m {seconds:.2f}s"


def main() -> None:
    args = _parse_args()
    project_root = _setup_project_root(
        Path(args.project_root) if args.project_root else None
    )
    _configure_gpu(args.cuda_device)

    if args.offline_hub:
        os.environ["HF_HUB_OFFLINE"] = "1"

    if args.quiet:
        _configure_quiet_ocr()

    import torch

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

    if not torch.cuda.is_available():
        print(
            "Warning: CUDA is not available. DeepSeek OCR requires a GPU.",
            file=sys.stderr,
        )

    from epub_generator import LaTeXRender, TableRender

    from pdf_craft import BookMeta, OCREventKind, transform_epub

    pdf_path = Path(args.pdf)
    epub_path = Path(args.epub)
    analysing_path = Path(args.analysing_path)
    models_cache_path = Path(args.models_cache_path)
    model_snapshot: str | None = None

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if args.local_only and not models_cache_path.exists():
        raise FileNotFoundError(f"Model cache not found: {models_cache_path}")

    if args.local_only and args.ocr_model == "deepseek-ocr-2":
        from pdf_craft.pdf.deepseek_hf_model import (
            format_local_model_not_found_message,
            huggingface_cache_model_dir,
            resolve_local_pretrained_path,
        )

        hub_dir = huggingface_cache_model_dir("deepseek-ocr-2")
        model_snapshot = resolve_local_pretrained_path(models_cache_path, hub_dir)
        if model_snapshot is None:
            raise FileNotFoundError(
                format_local_model_not_found_message(
                    models_cache_path,
                    hub_dir,
                    "deepseek-ai/DeepSeek-OCR-2",
                )
            )

    epub_path.parent.mkdir(parents=True, exist_ok=True)
    analysing_path.mkdir(parents=True, exist_ok=True)

    if args.force_ocr:
        for subpath in ("ocr", "chapters", "assets", "plots"):
            cached = analysing_path / subpath
            if cached.exists():
                shutil.rmtree(cached)
        toc_file = analysing_path / "toc.xml"
        if toc_file.exists():
            toc_file.unlink()
        cover_file = analysing_path / "cover.png"
        if cover_file.exists():
            cover_file.unlink()

    authors = args.authors or ["Author 1"]

    print(f"Project root:   {project_root}")
    print(f"PDF:            {pdf_path}")
    print(f"EPUB:           {epub_path}")
    print(f"Analysing path: {analysing_path}")
    print(f"Models:         {models_cache_path}")
    if model_snapshot is not None:
        print(f"Model snapshot: {model_snapshot}")
    if os.environ.get("PDF_CRAFT_MODELS"):
        print(
            f"  (from env PDF_CRAFT_MODELS={os.environ.get('PDF_CRAFT_MODELS')})",
        )
    print(f"GPU:            {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
    print(f"OCR size:       {args.ocr_size}")
    print(f"OCR model:      {args.ocr_model}")
    print(f"Text layout:    {args.text_layout} (OCR / reading order)")
    print(f"EPUB layout:    {args.epub_layout}")
    print(f"Local only:     {args.local_only}")
    print(f"HF offline:     {os.environ.get('HF_HUB_OFFLINE', '0')}")
    print(f"Quiet OCR logs: {args.quiet}")
    print(f"Force OCR:      {args.force_ocr}")
    print()

    with _maybe_quiet_ocr_runtime(args.quiet):
        metering = transform_epub(
            pdf_path=pdf_path,
            epub_path=epub_path,
            analysing_path=analysing_path,
            ocr_size=args.ocr_size,
            ocr_model=args.ocr_model,
            text_layout=args.text_layout,
            epub_layout=args.epub_layout,
            models_cache_path=models_cache_path,
            local_only=args.local_only,
            dpi=args.dpi,
            max_page_image_file_size=None,
            includes_cover=args.includes_cover,
            includes_footnotes=args.includes_footnotes,
            ignore_pdf_errors=args.ignore_pdf_errors,
            ignore_ocr_errors=args.ignore_ocr_errors,
            generate_plot=args.generate_plot,
            toc_assumed=True,
            book_meta=BookMeta(
                title=args.title,
                authors=authors,
                publisher=args.publisher,
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
    print(f"Done: {epub_path}")
    print(
        "OCR tokens: " f"input={metering.input_tokens}, output={metering.output_tokens}"
    )


if __name__ == "__main__":
    main()
