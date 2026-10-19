"""Preprocess vocab-index crops so DeepSeek sees print, not reverse-side bleed."""

from __future__ import annotations

import numpy as np
from PIL.Image import Image
from PIL import Image as PILImage

_MIN_DARK_FRAC = 0.003


def suppress_bleed(
    image: Image,
    *,
    ink_max: float = 168.0,
    keep_below: float = 132.0,
) -> Image:
    """Push faint gray (verso bleed) to white; keep dark printed ink."""
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    gray = arr.mean(axis=2)
    span = max(ink_max - keep_below, 1.0)
    fade = np.clip((gray - keep_below) / span, 0.0, 1.0)[..., None]
    out = arr * (1.0 - fade) + 255.0 * fade
    return PILImage.fromarray(out.astype(np.uint8), mode="RGB")


def pad_vocab_crop(
    image: Image,
    *,
    canvas: int = 1024,
    margin: int = 32,
) -> Image:
    """Paste a band onto a white square so the VLM still sees a page, not a strip."""
    rgb = image.convert("RGB")
    width, height = rgb.size
    canvas_w = max(canvas, width + margin * 2)
    canvas_h = max(canvas, height + margin * 2)
    out = PILImage.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    out.paste(rgb, (margin, margin))
    return out


def is_weak_print(image: Image, box: tuple[int, int, int, int]) -> bool:
    """True if the crop is blank or only reverse-side bleed-through."""
    crop = image.crop(box)
    if crop.size[0] < 8 or crop.size[1] < 8:
        return True
    gray = np.asarray(crop.convert("L"), dtype=np.float32)
    # Sparse final rows have a low average ink value despite clear dark glyphs.
    # Decide using dark ink, not the amount of surrounding white space.
    dark = gray < 120.0
    if float(dark.mean()) >= _MIN_DARK_FRAC:
        return False
    # A short word may occupy very little of a large band. Look for several
    # ink-bearing rows before treating that band as blank.
    min_row_pixels = max(3, int(gray.shape[1] * 0.01))
    return int(np.count_nonzero(dark.sum(axis=1) >= min_row_pixels)) < 3
