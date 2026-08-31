import tempfile
import unittest
import zipfile
from pathlib import Path

from pdf_craft.epub.vertical_style import patch_epub_for_vertical_writing


class TestVerticalEpubPatch(unittest.TestCase):
    def test_patch_adds_vertical_css_and_body_class(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            epub_path = Path(temp_dir) / "sample.epub"
            with zipfile.ZipFile(epub_path, "w") as archive:
                archive.writestr(
                    "OEBPS/styles/style.css",
                    "body { margin: 0; }",
                )
                archive.writestr(
                    "OEBPS/Text/chapter_1.xhtml",
                    (
                        '<?xml version="1.0" encoding="utf-8"?>'
                        "<html><head></head><body><p>正文</p></body></html>"
                    ),
                )
                archive.writestr(
                    "OEBPS/content.opf",
                    "<package><metadata><dc:language>zh</dc:language></metadata></package>",
                )

            patch_epub_for_vertical_writing(epub_path)

            with zipfile.ZipFile(epub_path, "r") as archive:
                css = archive.read("OEBPS/styles/style.css").decode("utf-8")
                chapter = archive.read("OEBPS/Text/chapter_1.xhtml").decode("utf-8")
                opf = archive.read("OEBPS/content.opf").decode("utf-8")

            self.assertIn("writing-mode: vertical-rl", css)
            self.assertIn('<body class="vertical-rl">', chapter)
            self.assertIn("<dc:language>zh-Hant</dc:language>", opf)


if __name__ == "__main__":
    unittest.main()
