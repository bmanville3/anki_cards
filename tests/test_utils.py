import logging
import unittest

from src.common.types import RawChunk
from src.common.utils import assert_latin_extended_only, check_chunk_ordering

def _make_chunk(index: int, text: str = "テスト", start: float = 0.0, end: float = 1.0) -> RawChunk:
    return RawChunk(index=index, subtitle_text=text, start=start, end=end)

class TestDoesNotContainEnglish(unittest.TestCase):
    def test_plain_ascii_is_clean(self):
        assert_latin_extended_only("Hello, world!")

    def test_latin_extended_accepted(self):
        assert_latin_extended_only("Café résumé naïve")

    def test_allowed_punctuation(self):
        assert_latin_extended_only("It's a \"great\" day—truly…")

    def test_japanese_flagged(self):
        self.assertRaises(assert_latin_extended_only("こんにちは"))

    def test_mixed_flags_only_japanese(self):
        self.assertRaises(assert_latin_extended_only("Hello こんにちは world"))
    
    def test_chinese_flagged(self) -> None:
        self.assertRaises(assert_latin_extended_only("你好"))

    def test_empty_string_is_clean(self):
        assert_latin_extended_only("")


class TestCheckOrdering(unittest.TestCase):
    def test_ordered_no_warning(self):
        chunks = [_make_chunk(i) for i in range(3)]
        with self.assertNoLogs(level=logging.WARNING):
            check_chunk_ordering(chunks[2], [chunks[0], chunks[1]], None)

    def test_out_of_order_warns(self):
        with self.assertLogs(level=logging.WARNING):
            check_chunk_ordering(
                _make_chunk(1),
                [_make_chunk(5)],
                None,
            )

    def test_skipped_index_warns(self):
        with self.assertLogs(level=logging.WARNING):
            check_chunk_ordering(
                _make_chunk(10),
                [_make_chunk(3)],
                None,
            )
    
    def test_next_index_warns(self):
        with self.assertLogs(level=logging.WARNING):
            check_chunk_ordering(
                _make_chunk(1),
                [_make_chunk(0)],
                _make_chunk(10),
            )

    def test_no_exception_ever_raised(self):
        try:
            check_chunk_ordering(_make_chunk(0), [_make_chunk(99)], _make_chunk(1))
        except Exception as exc:
            self.fail(f"check_ordering raised unexpectedly: {exc}")
