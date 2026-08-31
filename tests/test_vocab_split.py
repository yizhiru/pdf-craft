import unittest

from pdf_craft.pdf import PageLayout
from pdf_craft.sequence.reading_serials import split_reading_serials
from pdf_craft.sequence.vocab_split import (
    expand_dual_column_vocab_layouts,
    is_vocab_entry,
    split_dual_vocab_line,
)


class TestVocabSplit(unittest.TestCase):
    def test_split_dual_line_example(self):
        self.assertEqual(
            split_dual_vocab_line("I T産業 16※ あなた 1※"),
            ("I T産業 16※", "あなた 1※"),
        )

    def test_split_with_comma_page_refs(self):
        self.assertEqual(
            split_dual_vocab_line("頭 7▼, 16※ あれ 2"),
            ("頭 7▼, 16※", "あれ 2"),
        )
        self.assertEqual(
            split_dual_vocab_line("アイスクリーム 8※ 暑い 9※, 10▼"),
            ("アイスクリーム 8※", "暑い 9※, 10▼"),
        )

    def test_single_entry_not_split(self):
        self.assertIsNone(split_dual_vocab_line("ああ [叹词] 9※"))
        self.assertTrue(is_vocab_entry("ああ [叹词] 9※"))

    def test_explanation_not_split(self):
        self.assertIsNone(
            split_dual_vocab_line("单词右侧的数字表示初次出现的课次。")
        )

    def test_expand_layouts_and_columns_order(self):
        layouts = [
            PageLayout(
                ref="text",
                det=(193, 100, 1931, 160),
                text="I T産業 16※ あなた 1※",
                hash=None,
                order=0,
            ),
            PageLayout(
                ref="text",
                det=(193, 200, 1931, 260),
                text="会う 8 兄 2▼, 12※",
                hash=None,
                order=1,
            ),
        ]
        expanded = expand_dual_column_vocab_layouts(layouts)
        self.assertEqual(len(expanded), 4)
        self.assertEqual(expanded[0].text, "I T産業 16※")
        self.assertEqual(expanded[1].text, "あなた 1※")

        result = list(split_reading_serials(layouts, text_layout="columns"))
        texts = [layout.text for group in result for layout in group]
        self.assertEqual(
            texts,
            ["I T産業 16※", "会う 8", "あなた 1※", "兄 2▼, 12※"],
        )

    def test_deinterleave_tall_fused_blob_left_then_right(self):
        """Tall OCR blob with row-wise L/R entries → left column then right."""
        from pdf_craft.ocr_pdf.normalize import order_left_column_then_right

        fused = (
            "ああ [叹词] 9※ ああ 14※ "
            "アイスクリーム 8※ 間 24 "
            "相手 17 I D カード 16※ "
            "会う 8 青 18"
        )
        layouts = [
            PageLayout(
                ref="text",
                det=(100, 200, 900, 2000),
                text=fused,
                hash=None,
                order=0,
            )
        ]
        texts = [layout.text for layout in order_left_column_then_right(layouts)]
        self.assertEqual(
            texts,
            [
                "ああ [叹词] 9※",
                "アイスクリーム 8※",
                "相手 17",
                "会う 8",
                "ああ 14※",
                "間 24",
                "I D カード 16※",
                "青 18",
            ],
        )


    def test_right_half_blob_stays_single_column(self):
        """右半栏长块是单栏连续词条，不能再奇偶拆到左栏。"""
        from pdf_craft.ocr_pdf.normalize import order_left_column_then_right

        layouts = [
            PageLayout(
                ref="text",
                det=(50, 100, 400, 800),
                text="妹 2▼, 4※",
                hash=None,
                order=0,
            ),
            PageLayout(
                ref="text",
                det=(500, 100, 900, 800),
                text=(
                    "妹さん 2▼ いや [叹词] 22 "
                    "インド [国名] 1▼ インド人 1▼ "
                    "ウイスキー 4※ ウーロン茶 12※"
                ),
                hash=None,
                order=1,
            ),
        ]
        texts = [layout.text for layout in order_left_column_then_right(layouts)]
        self.assertEqual(texts[0], "妹 2▼, 4※")
        self.assertEqual(texts[1], "妹さん 2▼")
        self.assertIn("ウイスキー 4※", texts)
        self.assertIn("ウーロン茶 12※", texts)
        # 二者都应在右栏，且紧跟在 妹さん 之后的连续流里
        i_imouto = texts.index("妹 2▼, 4※")
        i_san = texts.index("妹さん 2▼")
        self.assertEqual(i_san, i_imouto + 1)


if __name__ == "__main__":
    unittest.main()
