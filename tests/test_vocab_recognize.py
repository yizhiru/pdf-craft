"""Vocabulary extraction must not silently cache partially recognized pages."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, create_autospec, patch

from PIL import Image, ImageDraw

from pdf_craft.error import OCRError
from pdf_craft.ocr_config import DeepSeekOCR2LocalConfig
from pdf_craft.ocr_pdf.normalize import pages_to_ocr_pdf_pages
from pdf_craft.ocr_pdf.recognize import column_slice_boxes, run_deepseek_column_ocr
from pdf_craft.pdf.page_extractor import PageExtractorNode
from pdf_craft.pdf.types import Page, PageLayout


def _page(text: str, ref: str = "text") -> Page:
    return Page(
        index=1, image=None,
        body_layouts=[PageLayout(ref=ref, det=(0, 0, 100, 100), text=text, order=0, hash=None)],
        footnotes_layouts=[], input_tokens=10, output_tokens=5,
    )


class TestVocabNormalize(unittest.TestCase):
    def test_long_ocr_block_keeps_all_entries(self):
        entries = [f"単語{chr(0x4e00 + i)}の説明です {i + 1}" for i in range(100)]
        raw = " ".join(entries)
        self.assertGreater(len(raw), 800)
        result = pages_to_ocr_pdf_pages({1: _page(raw)})
        self.assertCountEqual([line.text for line in result[0].lines], entries)

    def test_mixed_table_block_keeps_surrounding_text_and_all_cells(self):
        raw = (
            "老人 28▼\n<table><tr><td>隣国</td><td>23※</td>"
            "<td>リンス</td><td>21▼</td><td>ルーツ</td><td>21</td>"
            "</tr></table>\n若葉 18※"
        )
        for ref in ("text", "table"):
            with self.subTest(ref=ref):
                result = pages_to_ocr_pdf_pages({1: _page(raw, ref)})
                self.assertCountEqual([line.text for line in result[0].lines], [
                    "老人 28▼", "隣国 23※", "リンス 21▼", "ルーツ 21", "若葉 18※",
                ])


class TestVocabCropCoverage(unittest.TestCase):
    def test_rows_after_section_gap_and_near_page_bottom_are_covered(self):
        image = Image.new("RGB", (200, 1000), "white")
        draw = ImageDraw.Draw(image)
        rows = [*range(50, 401, 30), 700, 950]
        for y in rows:
            draw.rectangle((20, y, 120, y + 12), fill="black")
        boxes = column_slice_boxes(
            x1=0, y1=0, x2=200, y2=1000, band_px=220, overlap_px=40, image=image,
        )
        for y in rows:
            with self.subTest(y=y):
                self.assertTrue(any(top <= y and bottom > y + 12 for _, top, _, bottom in boxes))

    def test_blank_column_has_no_crops(self):
        self.assertEqual(column_slice_boxes(
            x1=0, y1=0, x2=200, y2=1000, band_px=220, overlap_px=40,
            image=Image.new("RGB", (200, 1000), "white"),
        ), [])


class TestVocabRecognition(unittest.TestCase):
    def _run(self, path: Path, responses: list[Page | Exception]):
        extractor = create_autospec(PageExtractorNode, instance=True)
        extractor.image2page.side_effect = responses
        document = Mock()
        document.pages_count = 1
        document.render_page.return_value = Image.new("RGB", (400, 1000), "white")
        with (
            patch("pdf_craft.ocr_pdf.recognize.PageExtractorNode", return_value=extractor) as factory,
            patch("pdf_craft.ocr_pdf.recognize.DefaultPDFHandler") as handler,
            patch("pdf_craft.ocr_pdf.recognize.detect_column_body_top", return_value=30),
            patch("pdf_craft.ocr_pdf.recognize.is_weak_print", return_value=False),
        ):
            handler.return_value.open.return_value = document
            try:
                result = run_deepseek_column_ocr(
                    Path("unused.pdf"), path, models_cache_path=Path("unused-models"),
                    slice_bands=False,
                )
            finally:
                document.close.assert_called_once()
            self.assertIsInstance(factory.call_args.kwargs["ocr"], DeepSeekOCR2LocalConfig)
        return result, extractor

    def test_failed_region_does_not_write_completed_page(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "ocr"
            with self.assertRaisesRegex(OCRError, "region=right page=1"):
                self._run(path, [_page("隣国 23※"), RuntimeError("recognition failed")])
            self.assertFalse((path / "page_1.xml").exists())
            self.assertFalse((path / "done").exists())
            # Retrying the same cache must recognize both regions again.
            result, extractor = self._run(path, [_page("隣国 23※"), _page("リンス 21▼")])
            self.assertEqual(extractor.image2page.call_count, 2)
            self.assertEqual([layout.text for layout in result[1].body_layouts], ["隣国 23※", "リンス 21▼"])
            self.assertEqual(result[1].input_tokens, 20)
            self.assertEqual(result[1].output_tokens, 10)
            self.assertTrue((path / "done").exists())

    def test_empty_recognition_does_not_write_completed_page(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "ocr"
            with self.assertRaisesRegex(OCRError, "no usable text"):
                self._run(path, [_page("")] * 3)
            self.assertFalse((path / "page_1.xml").exists())
            self.assertFalse((path / "done").exists())

    def test_empty_table_retries_smaller_crops_and_counts_all_tokens(self):
        with TemporaryDirectory() as tmp:
            result, extractor = self._run(Path(tmp) / "ocr", [
                _page("<table><tr><td></td></tr></table>", "table"),
                _page("隣国 23※ リンス 21▼"),
                _page("リンス 21▼"),
                _page("老人 28"),
            ])
            self.assertEqual([layout.text for layout in result[1].body_layouts], [
                "隣国 23※", "リンス 21▼", "老人 28",
            ])
            self.assertEqual(result[1].input_tokens, 40)
            self.assertEqual(result[1].output_tokens, 20)
            calls = extractor.image2page.call_args_list
            self.assertLess(calls[1].kwargs["image"].height, calls[0].kwargs["image"].height)

    def test_recognized_page_number_can_be_filtered(self):
        with TemporaryDirectory() as tmp:
            result, _ = self._run(Path(tmp) / "ocr", [_page("隣国 23※"), _page("460")])
            self.assertEqual([layout.text for layout in result[1].body_layouts], ["隣国 23※"])


if __name__ == "__main__":
    unittest.main()
