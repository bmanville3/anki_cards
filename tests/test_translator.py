import logging
import unittest
from pathlib import Path

from src.common.types import RawChunk
from src.common.utils import assert_latin_extended_only, load_image_b64, server_available
from src.prompting.translator import translate_sentence


logger = logging.getLogger(__name__)


def _make_chunk(index: int, text: str = "テスト", start: float = 0.0, end: float = 1.0) -> RawChunk:
    return RawChunk(index=index, subtitle_text=text, start=start, end=end)


PORT = 9090
SERVER_UP = server_available(port=PORT)
SKIP_MSG = f"LLM server not reachable at localhost:{PORT} — skipping live tests"


@unittest.skipUnless(SERVER_UP, SKIP_MSG)
class TestIntegrationTranslatesToEnglishOnly(unittest.TestCase):
    def test_translation_output_is_english(self):
        chunk = _make_chunk(0, "おはようございます")
        result = translate_sentence(chunk, [], None, None)
        self.assertGreater(len(result), 0, "Got empty translation")
        assert_latin_extended_only(result)
    

@unittest.skipUnless(SERVER_UP, SKIP_MSG)
class TestIntegrationTranslationWithImageUnderstanding(unittest.TestCase):
    FIXTURE_PNG = Path(__file__).parent / "resources" / "spooky_picture.png"
    FIXTURE_JPEG = Path(__file__).parent / "resources" / "spooky_picture.jpeg"

    def _assert_translation_is_english(self, image_b64: tuple[str, str], label: str) -> None:
        chunk = _make_chunk(0, "おいしそうですね")
        result = translate_sentence(chunk, [], None, base64_encoded_image=image_b64[0], image_mime=image_b64[1])
        self.assertGreater(len(result), 0, f"Got empty translation for {label}")
        assert_latin_extended_only(result)
        print(f"Translation: 'おいしそうですね' -> {result}")

    def test_translation_with_png_context_is_english(self):
        self._assert_translation_is_english(load_image_b64(self.FIXTURE_PNG), "PNG")

    def test_translation_with_jpeg_context_is_english(self):
        self._assert_translation_is_english(load_image_b64(self.FIXTURE_JPEG), "JPEG")


if __name__ == "__main__":
    unittest.main()
