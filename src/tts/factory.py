from src.tts.fish import FishBackend
from src.tts.kokoro import KokoroBackend
from src.tts.tts import Lang, TTSBackend


_BACKENDS: dict[str, type[TTSBackend]] = {
    "kokoro": KokoroBackend,
    "fish":   FishBackend,
}


def build_tts_engine(backend: str, lang: Lang = "ja", **kwargs) -> TTSBackend:
    cls = _BACKENDS.get(backend.lower())
    if cls is None:
        raise ValueError(f"Unknown TTS backend {backend!r}. Choose from: {list(_BACKENDS)}")
    return cls(lang=lang, **kwargs)


def build_tts_pair(
    jp_backend: str,
    en_backend: str,
    media_workers: int = 8,
) -> tuple[TTSBackend, TTSBackend]:
    if jp_backend == en_backend == "fish":
        shared = build_tts_engine("fish", lang="ja", max_connections=media_workers)
        shared.setup()
        en_shim = _LangShim(shared, lang="en")
        return shared, en_shim

    jp_tts = build_tts_engine(jp_backend, lang="ja", max_connections=media_workers)
    en_tts = build_tts_engine(en_backend, lang="en", max_connections=media_workers)
    jp_tts.setup()
    en_tts.setup()
    return jp_tts, en_tts


class _LangShim(TTSBackend):
    def __init__(self, inner: TTSBackend, lang: Lang) -> None:
        super().__init__(lang)
        self._inner = inner

    def setup(self) -> None:
        pass

    def generate(self, text: str, out_path: str, voice: str = "") -> bool:
        return self._inner.generate(text, out_path, voice=voice)
