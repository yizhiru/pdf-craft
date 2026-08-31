def find_repetition_span(
    text: str,
    min_ngram: int,
    max_ngram: int,
    repeat_threshold: int,
) -> tuple[int, int] | None:
    """Return ``(keep_end, loop_end)`` char indexes of a consecutive n-gram loop."""
    if not text:
        return None

    chars = list(text)
    if len(chars) < min_ngram * repeat_threshold:
        return None

    for n in range(min_ngram, min(max_ngram + 1, len(chars) // repeat_threshold + 1)):
        for i in range(len(chars) - n * repeat_threshold + 1):
            ngram = tuple(chars[i : i + n])
            consecutive_count = 1
            pos = i + n
            while pos + n <= len(chars):
                next_ngram = tuple(chars[pos : pos + n])
                if next_ngram == ngram:
                    consecutive_count += 1
                    pos += n
                else:
                    break

            if consecutive_count >= repeat_threshold:
                return i + n, pos

    return None


def trim_repetitive_tail(
    text: str,
    min_ngram: int = 6,
    max_ngram: int = 40,
    repeat_threshold: int = 4,
) -> str:
    """Keep the first copy of a degeneration loop; drop the repeating tail.

    Unlike :func:`has_repetitive_ngrams`, this preserves the real prefix
    (e.g. the first few vocab rows) instead of discarding the whole block.
    """
    value = text or ""
    guard = 0
    while value and guard < 8:
        span = find_repetition_span(
            value,
            min_ngram=min_ngram,
            max_ngram=max_ngram,
            repeat_threshold=repeat_threshold,
        )
        if span is None:
            break
        keep_end, loop_end = span
        value = value[:keep_end] + value[loop_end:]
        guard += 1
    return value.strip()


def has_repetitive_ngrams(
    text: str,
    min_ngram: int,
    max_ngram: int,
    repeat_threshold: int,
) -> bool:
    """
    检测文本中是否存在大量重复的 n-gram 模式（字符级别）。

    这个方法用于检测 OCR 模型的 "Neural Text Degeneration" 问题。
    当 Transformer 模型训练不足或解码参数配置不当时，会生成重复的文本片段，
    例如 "这是一个例子这是一个例子这是一个例子..." 或
         "this is a test this is a test this is a test..."
    这种重复模式是无意义的噪声，需要被识别和过滤。

    采用字符级别分词以支持中文等没有空格分隔的语言。
    """
    if not text:
        return False

    chars = list(text)
    if len(chars) < min_ngram * repeat_threshold:
        return False

    for n in range(min_ngram, min(max_ngram + 1, len(chars) // repeat_threshold + 1)):
        for i in range(len(chars) - n * repeat_threshold + 1):
            ngram = tuple(chars[i : i + n])
            consecutive_count = 1
            pos = i + n
            while pos + n <= len(chars):
                next_ngram = tuple(chars[pos : pos + n])
                if next_ngram == ngram:
                    consecutive_count += 1
                    pos += n
                else:
                    break

            if consecutive_count >= repeat_threshold:
                return True

    return False
