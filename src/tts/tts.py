"""
TTS backends — Kokoro (local ONNX) and Fish Audio S2 Pro (local HTTP server).

Two language routes:
  - Japanese: Fish S2 Pro (voice cloning, highest quality) or Kokoro (offline)
  - English:  Kokoro (good English voices, offline) or Fish S2 Pro

Voice cloning (Fish JP only):
    Provide a reference audio file path to clone a speaker's voice.
    The transcript is optional but improves quality.

    jp_tts.load_voice("/path/to/reference.mp3", transcript="こんにちは")
    jp_tts.generate("日本語テキスト", "/tmp/jp.mp3")   # uses cloned voice

Batching:
    results = jp_tts.generate_batch([
        TTSRequest("テキスト1", "/tmp/1.mp3"),
        TTSRequest("テキスト2", "/tmp/2.mp3"),
    ], max_workers=4)
    # results[i].ok, results[i].error

Usage:
    jp_tts, en_tts = build_tts_pair("fish", "kokoro", media_workers=4)
    jp_tts.load_voice("reference.mp3", transcript="optional transcript")

    jp_tts.generate("日本語テキスト", "/tmp/jp.mp3")
    en_tts.generate("English text",  "/tmp/en.mp3")

Thread safety:
    Kokoro serialises synthesis under a lock (the ONNX runtime session is not
    thread-safe); Fish uses a shared httpx connection pool. Both are safe to
    call concurrently from a ThreadPoolExecutor.

Recommended pairings:
    --jp-tts fish   --en-tts fish   (default: best quality split)
    --jp-tts kokoro --en-tts kokoro   (fully offline, no cloning)
"""
import base64
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

from attrs import define
import httpx


Lang = Literal["ja", "en"]


@define
class TTSRequest:
    text:     str
    out_path: str
    voice:    str = ""


@define
class TTSResult:
    request: TTSRequest
    ok:      bool  = False
    error:   str   = ""


class TTSBackend(ABC):
    def __init__(self, lang: Lang) -> None:
        self.lang = lang

    @abstractmethod
    def setup(self) -> None:
        """Initialise the backend. Raises RuntimeError if unavailable."""

    def load_voice(self, audio_path: str, transcript: str = "") -> None:
        """
        Load a reference audio file for voice cloning.
        No-op on backends that don't support cloning (e.g. Kokoro).
        """

    @abstractmethod
    def generate(self, text: str, out_path: str, voice: str = "") -> bool:
        """Synthesise *text* and write audio to *out_path*. Returns success."""

    def generate_batch(
        self,
        requests: list[TTSRequest],
        *,
        max_workers: int = 4,
    ) -> list[TTSResult]:
        """
        Synthesise a list of TTSRequests concurrently.
        Results are returned in the same order as *requests*.
        """
        results = [TTSResult(request=req) for req in requests]

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(self.generate, req.text, req.out_path, req.voice): i
                for i, req in enumerate(requests)
            }
            for future in as_completed(futures):
                i = futures[future]
                try:
                    results[i].ok = future.result()
                    if not results[i].ok:
                        results[i].error = "generate() returned False"
                except Exception as e:
                    results[i].ok    = False
                    results[i].error = str(e)

        return results


# ── Kokoro ────────────────────────────────────────────────────────────────────

KOKORO_MODEL_PATH  = "./models/kokoro/kokoro-v1.0.onnx"
KOKORO_VOICES_PATH = "./models/kokoro/voices-v1.0.bin"

# Kokoro v1.0 voice IDs by language.
# Japanese: jf_* = female, jm_* = male
# English:  af_* = female, am_* = male
KOKORO_DEFAULT_VOICES: dict[Lang, str] = {
    "ja": "jf_alpha",
    "en": "af_heart",
}


class KokoroBackend(TTSBackend):
    """
    Runs Kokoro v1.0 via kokoro-onnx entirely in-process.
    A single ONNX session is shared across threads but serialised with a lock.
    Audio writing (sf.write) happens outside the lock so other threads stay busy.
    Voice cloning is not supported by Kokoro; load_voice() is a no-op.
    """

    def __init__(
        self,
        lang:        Lang = "ja",
        model_path:  str  = KOKORO_MODEL_PATH,
        voices_path: str  = KOKORO_VOICES_PATH,
    ) -> None:
        super().__init__(lang)
        self._model_path  = model_path
        self._voices_path = voices_path
        self._kokoro      = None
        self._lock        = threading.Lock()

    def setup(self) -> None:
        try:
            from kokoro_onnx import Kokoro  # type: ignore
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

        import soundfile as sf  # type: ignore

        voice = voice or KOKORO_DEFAULT_VOICES[self.lang]
        try:
            with self._lock:
                samples, sample_rate = self._kokoro.create(
                    text, voice=voice, speed=1.0, lang=self.lang,
                )
            sf.write(out_path, samples, sample_rate)
            return True
        except Exception as e:
            print(f"  Kokoro TTS error ({self.lang}): {e}")
            return False


# ── Fish Audio S2 Pro ─────────────────────────────────────────────────────────

FISH_TTS_URL    = "http://127.0.0.1:8080/v1/tts"
FISH_HEALTH_URL = "http://127.0.0.1:8080/v1/health"


@define
class _ClonedVoice:
    """Holds the base64-encoded reference audio and optional transcript."""
    audio_b64:  str
    transcript: str = ""


class FishBackend(TTSBackend):
    """
    Calls the Fish Audio S2 Pro local HTTP server.
    The `lang` parameter is informational only — Fish infers language from text.
    Uses a shared httpx connection pool — safe for concurrent use.

    Voice cloning:
        Call load_voice(path, transcript="...") before generate().
        The reference audio is base64-encoded once and sent with every request.
        A per-request `voice` string (reference_id) takes priority over the
        loaded clone, so you can still override individual items in a batch.
    """

    def __init__(
        self,
        lang:            Lang = "ja",
        tts_url:         str  = FISH_TTS_URL,
        health_url:      str  = FISH_HEALTH_URL,
        max_connections: int  = 8,
    ) -> None:
        super().__init__(lang)
        self._tts_url     = tts_url
        self._health_url  = health_url
        self._cloned_voice: _ClonedVoice | None = None
        self._client      = httpx.Client(
            timeout=60,
            limits=httpx.Limits(max_connections=max_connections),
        )

    def setup(self) -> None:
        try:
            r = self._client.get(self._health_url, timeout=3)
            if r.status_code == 200 and r.json().get("status") == "ok":
                print(f"  TTS ready (Fish Audio S2 Pro, lang={self.lang}).")
                return
            raise RuntimeError(f"Fish Audio server unhealthy: {r.text}")
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(
                f"Fish Audio S2 Pro not reachable at {self._tts_url}: {e}\n"
                "  Start it with: python tools/api_server.py "
                "--llama-checkpoint-path checkpoints/s2-pro "
                "--decoder-checkpoint-path checkpoints/s2-pro/codec.pth "
                "--listen 0.0.0.0:8080 --half"
            ) from e

    def load_voice(self, audio_path: str, transcript: str = "") -> None:
        """
        Load a reference audio file for voice cloning.
        The file is read and base64-encoded once here; generate() is then
        thread-safe with no further file I/O per request.

        Args:
            audio_path: path to a clear, single-speaker audio clip (mp3/wav/etc.)
            transcript: optional transcript of the reference audio — improves
                        prosody matching when provided.
        """
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Voice reference not found: {audio_path}")

        audio_b64 = base64.b64encode(path.read_bytes()).decode()
        self._cloned_voice = _ClonedVoice(audio_b64=audio_b64, transcript=transcript)
        label = f" (transcript: {transcript[:40]!r})" if transcript else ""
        print(f"  Voice cloning loaded: {path.name}{label}")

    def generate(self, text: str, out_path: str, voice: str = "") -> bool:
        if not text.strip():
            return False
        try:
            payload: dict = {"text": text, "format": "mp3", "streaming": False}

            if voice:
                # Explicit per-request reference_id takes priority
                payload["reference_id"] = voice
            elif self._cloned_voice is not None:
                # Use the loaded clone; send inline audio so the server needs
                # no separate upload step
                payload["references"] = [
                    {
                        "audio": self._cloned_voice.audio_b64,
                        "text":  self._cloned_voice.transcript,
                    }
                ]

            r = self._client.post(self._tts_url, json=payload)
            if r.status_code != 200:
                print(f"  Fish TTS error {r.status_code} ({self.lang}): {r.text[:200]}")
                return False
            Path(out_path).write_bytes(r.content)
            return True
        except Exception as e:
            print(f"  Fish TTS error ({self.lang}): {e}")
            return False


# ── Factory ───────────────────────────────────────────────────────────────────

_BACKENDS: dict[str, type[TTSBackend]] = {
    "kokoro": KokoroBackend,
    "fish":   FishBackend,
}


def build_tts_engine(backend: str, lang: Lang = "ja", **kwargs) -> TTSBackend:
    """
    Return a TTSBackend for *backend* (``"kokoro"`` or ``"fish"``).

    Args:
        backend:  ``"kokoro"`` or ``"fish"``
        lang:     ``"ja"`` (Japanese) or ``"en"`` (English)
        **kwargs: forwarded to the backend constructor (e.g. max_connections)

    Call ``.setup()`` on the result before use.
    """
    cls = _BACKENDS.get(backend.lower())
    if cls is None:
        raise ValueError(f"Unknown TTS backend {backend!r}. Choose from: {list(_BACKENDS)}")
    return cls(lang=lang, **kwargs)


def build_tts_pair(
    jp_backend: str,
    en_backend: str,
    media_workers: int = 8,
) -> tuple[TTSBackend, TTSBackend]:
    """
    Convenience: build and set up both the Japanese and English TTS engines.
    If both engines use Fish, they share the same backend instance to avoid
    two health checks and two connection pools against the same server.
    Voice cloning is only applied to the JP engine via load_voice().

    Returns:
        (jp_tts, en_tts) — both already initialised via .setup()
    """
    if jp_backend == en_backend == "fish":
        shared = build_tts_engine("fish", lang="ja", max_connections=media_workers)
        shared.setup()
        # Wrap in a shim so .lang is correct on the EN side while sharing
        # the underlying httpx client and connection pool.
        # Note: load_voice() on the shared instance affects JP requests only
        # by convention — EN text will auto-select the right language anyway.
        en_shim = _LangShim(shared, lang="en")
        return shared, en_shim

    jp_tts = build_tts_engine(jp_backend, lang="ja", max_connections=media_workers)
    en_tts = build_tts_engine(en_backend, lang="en", max_connections=media_workers)
    jp_tts.setup()
    en_tts.setup()
    return jp_tts, en_tts


class _LangShim(TTSBackend):
    """
    Thin wrapper that overrides .lang without duplicating the underlying backend.
    load_voice() is intentionally not forwarded — cloning is JP-only.
    """

    def __init__(self, inner: TTSBackend, lang: Lang) -> None:
        super().__init__(lang)
        self._inner = inner

    def setup(self) -> None:
        pass  # already set up by the inner backend

    def generate(self, text: str, out_path: str, voice: str = "") -> bool:
        return self._inner.generate(text, out_path, voice=voice)
