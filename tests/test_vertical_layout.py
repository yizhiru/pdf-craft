import unittest

from pdf_craft.pdf.vertical import det_rotated_to_original, rotate_for_vertical_ocr
from pdf_craft.pdf import PageLayout
from pdf_craft.sequence.reading_serials import split_reading_serials


class TestVerticalOCRGeometry(unittest.TestCase):
    def test_det_round_trip(self):
        original_width = 1000
        original_det = (800, 100, 900, 500)
        from PIL import Image

        image = Image.new("RGB", (original_width, 2000), color="white")
        rotated = rotate_for_vertical_ocr(image)
        self.assertEqual(rotated.size, (2000, 1000))

        left, top, right, bottom = (100, 100, 500, 200)
        restored = det_rotated_to_original((left, top, right, bottom), original_width)
        self.assertEqual(restored, original_det)


class TestVerticalReadingSerials(unittest.TestCase):
    def test_vertical_columns_read_right_to_left(self):
        layouts = [
            PageLayout(
                ref="1",
                det=(50, 100, 150, 200),
                text="LeftCol",
                hash=None,
                order=0,
            ),
            PageLayout(
                ref="2",
                det=(400, 100, 500, 200),
                text="RightColTop",
                hash=None,
                order=1,
            ),
            PageLayout(
                ref="3",
                det=(400, 300, 500, 400),
                text="RightColBottom",
                hash=None,
                order=2,
            ),
        ]
        result = list(split_reading_serials(layouts, text_layout="vertical"))
        texts = [layout.text for group in result for layout in group]
        self.assertEqual(texts, ["RightColTop", "RightColBottom", "LeftCol"])

    def test_vertical_sorts_top_to_bottom_within_column(self):
        layouts = [
            PageLayout(
                ref="1",
                det=(400, 300, 500, 400),
                text="Bottom",
                hash=None,
                order=0,
            ),
            PageLayout(
                ref="2",
                det=(400, 100, 500, 200),
                text="Top",
                hash=None,
                order=1,
            ),
        ]
        result = list(split_reading_serials(layouts, text_layout="vertical"))
        texts = [layout.text for group in result for layout in group]
        self.assertEqual(texts, ["Top", "Bottom"])


class TestColumnsReadingSerials(unittest.TestCase):
    def test_columns_read_left_fully_then_right(self):
        """双栏词汇表：左列全部读完后再追加右列。"""
        layouts = [
            PageLayout(
                ref="1",
                det=(50, 300, 150, 400),
                text="LeftBottom",
                hash=None,
                order=0,
            ),
            PageLayout(
                ref="2",
                det=(400, 100, 500, 200),
                text="RightTop",
                hash=None,
                order=1,
            ),
            PageLayout(
                ref="3",
                det=(50, 100, 150, 200),
                text="LeftTop",
                hash=None,
                order=2,
            ),
            PageLayout(
                ref="4",
                det=(400, 300, 500, 400),
                text="RightBottom",
                hash=None,
                order=3,
            ),
        ]
        result = list(split_reading_serials(layouts, text_layout="columns"))
        texts = [layout.text for group in result for layout in group]
        self.assertEqual(
            texts,
            ["LeftTop", "LeftBottom", "RightTop", "RightBottom"],
        )
        self.assertEqual(len(result), 2)
        self.assertEqual([layout.text for layout in result[0]], ["LeftTop", "LeftBottom"])
        self.assertEqual(
            [layout.text for layout in result[1]],
            ["RightTop", "RightBottom"],
        )


if __name__ == "__main__":
    unittest.main()
