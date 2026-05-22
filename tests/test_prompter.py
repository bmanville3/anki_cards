from pathlib import Path
import unittest

from src.common.utils import assert_latin_extended_only, load_image_b64, only_japanese_and_punctuation, server_available
from src.prompting.prompter import _build_user_message, prompt_with_retries

PORT = 9090
SERVER_UP = server_available(port=PORT)
SKIP_MSG = f"LLM server not reachable at localhost:{PORT} — skipping live tests"


class TestBuildUserMessage(unittest.TestCase):
    def test_no_image_returns_plain_dict(self):
        msg = _build_user_message("translate this", None)
        self.assertEqual(msg["role"], "user")
        self.assertIsInstance(msg["content"], str)
        self.assertEqual(msg["content"], "translate this")

    def test_with_image_returns_multipart_list(self):
        msg = _build_user_message("translate this", "abc123base64==")
        self.assertEqual(msg["role"], "user")
        self.assertIsInstance(msg["content"], list)
        types = [block["type"] for block in msg["content"]]
        self.assertIn("image_url", types)
        self.assertIn("text", types)

    def test_image_url_data_uri_format(self):
        msg = _build_user_message("prompt", "BASE64DATA")
        image_block = next(b for b in msg["content"] if b["type"] == "image_url")
        self.assertTrue(image_block["image_url"]["url"].startswith("data:image/jpeg;base64,"))
        self.assertIn("BASE64DATA", image_block["image_url"]["url"])


@unittest.skipUnless(SERVER_UP, SKIP_MSG)
class TestIntegrationJapaneseOnly(unittest.TestCase):
    def test_model_responds_in_japanese_when_asked(self):
        result = prompt_with_retries(
            "You are a helpful assistant. You MUST reply only in Japanese hiragana/katakana/kanji. No English.",
            "日本語で「テストです」とだけ答えてください。",
        )
        all_japanese = only_japanese_and_punctuation(result)
        self.assertTrue(all_japanese, f"Expected Japanese in response but got: {result}")


@unittest.skipUnless(SERVER_UP, SKIP_MSG)
class TestIntegrationEnglishOnly(unittest.TestCase):
    def test_english_response_to_english_request(self):
        result = prompt_with_retries(
            "You are a helpful assistant. Reply only in English.",
            "Say 'Hello, this is a test.' and nothing else.",
        )
        assert_latin_extended_only(result)


@unittest.skipUnless(SERVER_UP, SKIP_MSG)
class TestIntegrationImageUnderstanding(unittest.TestCase):
    FIXTURE_PNG = Path(__file__).parent / "resources" / "spooky_picture.png"
    FIXTURE_JPEG = Path(__file__).parent / "resources" / "spooky_picture.jpeg"
    HOTDOG_KEYWORDS = ("hot dog", "hotdog", "sausage", "frankfurter", "bun", "frank", "wiener", "glizzy", "dawg")

    def _assert_identifies_hotdog(self, image_b64: tuple[str, str], label: str) -> None:
        result = prompt_with_retries(
            system_prompt="You are an image recognition assistant. Describe what you see in one short, simple sentence with no punctuation.",
            user_prompt="What is in this image? Reply in one sentence with no punctuation.",
             base64_encoded_image=image_b64[0],
             image_mime=image_b64[1],
        )
        self.assertGreater(len(result), 0, f"Got empty response for {label}")
        self.assertTrue(result.isalpha())
        self.assertTrue(
            any(kw in result.lower().split(" ") for kw in self.HOTDOG_KEYWORDS),
            f"[{label}] Model did not identify a hotdog. Response: {result}",
        )

    def test_identifies_hotdog_png(self):
        self._assert_identifies_hotdog(load_image_b64(self.FIXTURE_PNG), "PNG")

    def test_identifies_hotdog_jpeg(self):
        self._assert_identifies_hotdog(load_image_b64(self.FIXTURE_JPEG), "JPEG")
