from pathlib import Path

import httpx

from tts.tts import Lang, TTSBackend


FISH_TTS_URL    = "http://127.0.0.1:8080/v1/tts"
FISH_HEALTH_URL = "http://127.0.0.1:8080/v1/health"


class FishBackend(TTSBackend):
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

    def generate(self, text: str, out_path: str, voice: str = "") -> bool:
        if not text.strip():
            return False
        try:
            payload: dict = {"text": text, "format": "mp3", "streaming": False}

            if voice:
                payload["reference_id"] = voice

            r = self._client.post(self._tts_url, json=payload)
            if r.status_code != 200:
                print(f"  Fish TTS error {r.status_code} ({self.lang}): {r.text[:200]}")
                return False
            Path(out_path).write_bytes(r.content)
            return True
        except Exception as e:
            print(f"  Fish TTS error ({self.lang}): {e}")
            return False
    