from pathlib import Path
import threading
import soundfile as sf

from src.tts.tts import Lang, TTSBackend


KOKORO_MODEL_PATH  = "./models/kokoro/kokoro-v1.0.onnx"
KOKORO_VOICES_PATH = "./models/kokoro/voices-v1.0.bin"

_KOKORO_LANG_CODE: dict[Lang, str] = {
    "ja": "ja",
    "en": "en-us",
}

KOKORO_DEFAULT_VOICES: dict[Lang, str] = {
    "ja": "jf_alpha",
    "en": "af_heart",
}


class KokoroBackend(TTSBackend):
    def __init__(
        self,
        lang:        Lang = "ja",
        model_path:  str  = KOKORO_MODEL_PATH,
        voices_path: str  = KOKORO_VOICES_PATH,
        **kwargs,
    ) -> None:
        super().__init__(lang, **kwargs)
        self._model_path  = model_path
        self._voices_path = voices_path
        self._kokoro      = None
        self._lock        = threading.Lock()

    def setup(self) -> None:
        try:
            from kokoro_onnx import Kokoro
        except ImportError as e:
            raise RuntimeError("kokoro-onnx not installed — pip install kokoro-onnx") from e

        if not Path(self._model_path).exists():
            raise RuntimeError(f"Kokoro model not found: {self._model_path}")
        if not Path(self._voices_path).exists():
            raise RuntimeError(f"Kokoro voices not found: {self._voices_path}")

        self._kokoro = Kokoro(self._model_path, self._voices_path)
        print(f"  TTS ready (Kokoro v1.0 ONNX, lang={self.lang}).")

    def generate(self, text: str, out_path: str, voice: str = "") -> bool:
        if self._kokoro is None or not text.strip():
            return False

        voice = voice or KOKORO_DEFAULT_VOICES[self.lang]
        try:
            with self._lock:
                samples, sample_rate = self._kokoro.create(
                    text, voice=voice, speed=1.0, lang=_KOKORO_LANG_CODE[self.lang],
                )
            sf.write(out_path, samples, sample_rate)
            return True
        except Exception as e:
            print(f"  Kokoro TTS error ({self.lang}): {e}")
            return False
