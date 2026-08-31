import re
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

_VERTICAL_CSS = """
body.vertical-rl {
  writing-mode: vertical-rl;
  -epub-writing-mode: vertical-rl;
  text-orientation: upright;
  -epub-text-orientation: upright;
}
"""

_BODY_TAG_PATTERN = re.compile(r"(<body)(\s|>)", re.IGNORECASE)
_HTML_TAG_PATTERN = re.compile(r"(<html)(\s|>)", re.IGNORECASE)
_LANGUAGE_PATTERN = re.compile(
    r"(<dc:language>)zh(</dc:language>)",
    re.IGNORECASE,
)


def patch_epub_for_vertical_writing(epub_path: Path, language: str = "zh-Hant") -> None:
    buffer = BytesIO()
    with ZipFile(epub_path, "r") as source, ZipFile(buffer, "w", ZIP_DEFLATED) as target:
        for item in source.infolist():
            data = source.read(item.filename)
            if item.filename == "OEBPS/styles/style.css":
                data = data + _VERTICAL_CSS.encode("utf-8")
            elif item.filename.endswith(".xhtml"):
                text = data.decode("utf-8")
                text = _HTML_TAG_PATTERN.sub(r'\1 class="vertical-rl"\2', text, count=1)
                text = _BODY_TAG_PATTERN.sub(r'\1 class="vertical-rl"\2', text, count=1)
                data = text.encode("utf-8")
            elif item.filename == "OEBPS/content.opf" and language != "zh":
                text = data.decode("utf-8")
                text = _LANGUAGE_PATTERN.sub(rf"\1{language}\2", text, count=1)
                data = text.encode("utf-8")
            target.writestr(item, data)

    epub_path.write_bytes(buffer.getvalue())
