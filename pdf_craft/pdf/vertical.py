from PIL import Image


def rotate_for_vertical_ocr(image: Image.Image) -> Image.Image:
    """Rotate page 90° CCW so vertical columns become horizontal lines for OCR."""
    return image.transpose(Image.ROTATE_90)


def det_rotated_to_original(
    det: tuple[int, int, int, int],
    original_width: int,
) -> tuple[int, int, int, int]:
    """Map a bounding box from the rotated OCR image back to original page coords."""
    left, top, right, bottom = det
    return (
        original_width - bottom,
        left,
        original_width - top,
        right,
    )
