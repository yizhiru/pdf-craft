import unittest

from pdf_craft.ocr_pdf.vocab_cleanup import (
    attach_orphan_marks,
    clean_pseudo_tags,
    cleanup_vocab_lines,
    cut_repeating_entries,
    headword_key,
    trim_repeating_table_html,
)
from pdf_craft.pdf.ngrams import trim_repetitive_tail


class TestVocabCleanup(unittest.TestCase):
    def test_td_not_kept_as_visible_text(self):
        self.assertEqual(clean_pseudo_tags("利率 <td>"), "利率")
        self.assertEqual(clean_pseudo_tags("利率 （td）"), "利率")
        self.assertIn("おなかが", clean_pseudo_tags("いっぱい <おなかが>"))

    def test_cut_repeating_with_ocr_typo(self):
        kept = cut_repeating_entries(
            [
                "了承 18※",
                "涼風 18※",
                "料理人 21※",
                "緑化 19※",
                "利率 19▼",
                "料理人 21※",
                "緗化 19※",
                "利率 （td）",
                "料理人 21※",
                "緗化 19※",
                "利率 1",
            ]
        )
        self.assertEqual(
            kept,
            ["了承 18※", "涼風 18※", "料理人 21※", "緑化 19※", "利率 19▼"],
        )

    def test_cut_repeating_interest_cook_cycle(self):
        kept = cut_repeating_entries(
            [
                "了承 18※",
                "涼風 18※",
                "料理人 21※",
                "緑化 19※",
                "利率 19▼",
                "料理人 21※",
                "緑化 19※",
                "利率 19▼",
                "料理人 21※",
                "緑化 19※",
                "利率 19▼",
            ]
        )
        self.assertEqual(
            kept,
            ["了承 18※", "涼風 18※", "料理人 21※", "緑化 19※", "利率 19▼"],
        )

    def test_orphan_mark_joins_lesson_number(self):
        self.assertEqual(
            attach_orphan_marks(["料理人 21", "※", "緑化 19", "▼"]),
            ["料理人 21※", "緑化 19▼"],
        )

    def test_cleanup_drops_fox_and_td(self):
        lines = cleanup_vocab_lines(
            [
                "The quick brown fox jumps over the lazy dog.",
                "ワンワン 23▼",
                "ワンワン 23▼",
                "利率 <td>",
                r"1. 已知 \( a = \frac{1}{2} \)",
                "L",
                "23",
            ]
        )
        self.assertEqual(lines, ["ワンワン 23▼", "利率"])

    def test_keeps_distinct_lesson_numbers(self):
        lines = cleanup_vocab_lines(["腰痛 20", "腰痛 27▼", "ワンワン 23▼"])
        self.assertEqual(lines, ["腰痛 20", "腰痛 27▼", "ワンワン 23▼"])

    def test_overlap_window_dedupes_same_entry(self):
        lines = cleanup_vocab_lines(
            [
                "悪口 26※",
                "ワンワン 23▼",
                "悪口 26※",
                "ワンワン 23",
            ]
        )
        self.assertEqual(lines, ["悪口 26※", "ワンワン 23▼"])

    def test_lines_from_ocr_response_splits_fused(self):
        from pdf_craft.ocr_pdf.vocab_cleanup import lines_from_ocr_response

        lines = lines_from_ocr_response(
            "悪口 26※ ワンワン 23▼\n"
            "<|ref|>text<|/ref|><|det|>[[1, 2, 3, 4]]<|/det|>\n"
            "若い 31▼"
        )
        self.assertIn("ワンワン 23▼", lines)
        self.assertIn("悪口 26※", lines)
        self.assertIn("若い 31▼", lines)

    def test_table_html_keeps_last_entries(self):
        from pdf_craft.ocr_pdf.vocab_cleanup import lines_from_ocr_response

        html = (
            "<table><tr><td></td><td>ロジオ</td><td>ロジオ</td></tr>"
            "<tr><td>2※</td><td>悪口</td><td>26※</td></tr>"
            "<tr><td></td><td>ワンワン</td><td>23▼</td></tr></table>"
        )
        lines = cleanup_vocab_lines(lines_from_ocr_response(html))
        self.assertIn("ワンワン 23▼", lines)
        self.assertIn("悪口 26※", lines)

    def test_strips_leading_junk_and_dedupes(self):
        lines = cleanup_vocab_lines(
            [
                ") ワンワン 23▼",
                "ワンワン 23▼",
                "V 若い 31V",
                "若い 31",
                "▼ ロープ 31",
            ]
        )
        self.assertEqual(lines.count("ワンワン 23▼"), 1)
        self.assertEqual(lines.count("若い 31"), 1)
        self.assertIn("ロープ 31▼", lines)

    def test_trim_repeating_table_rows(self):
        rows = "".join(
            "<tr><td>料理人</td><td>21※</td></tr>"
            "<tr><td>緑化</td><td>19※</td></tr>"
            "<tr><td>利率</td><td>19▼</td></tr>"
            for _ in range(4)
        )
        html = f"<table>{rows}</table>"
        trimmed = trim_repeating_table_html(html)
        self.assertEqual(trimmed.count("料理人"), 1)
        self.assertEqual(trimmed.count("<tr>"), 3)

    def test_headword_ignores_page_tail(self):
        self.assertEqual(headword_key("ワンワン 23▼"), "ワンワン")
        self.assertEqual(headword_key("頭 7▼, 16※"), "頭")

    def test_trim_repetitive_tail_keeps_prefix(self):
        prefix = "了承18※涼風18※"
        loop = "料理人21※緑化19※利率19▼"
        text = prefix + loop * 6
        trimmed = trim_repetitive_tail(text, min_ngram=6, max_ngram=40, repeat_threshold=4)
        self.assertIn("了承", trimmed)
        self.assertLess(trimmed.count("料理人"), 4)


class TestVocabImage(unittest.TestCase):
    def test_suppress_bleed_keeps_dark_ink(self):
        from PIL import Image

        from pdf_craft.ocr_pdf.vocab_image import pad_vocab_crop, suppress_bleed

        img = Image.new("RGB", (40, 20), (255, 255, 255))
        for x in range(8, 18):
            for y in range(4, 16):
                img.putpixel((x, y), (20, 20, 20))
        for x in range(24, 34):
            for y in range(4, 16):
                img.putpixel((x, y), (200, 200, 200))
        out = suppress_bleed(img)
        self.assertLess(sum(out.getpixel((12, 10))) / 3, 40)
        self.assertGreater(sum(out.getpixel((28, 10))) / 3, 240)
        padded = pad_vocab_crop(out, canvas=64, margin=4)
        self.assertEqual(padded.size, (64, 64))
