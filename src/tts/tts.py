from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal

from attrs import define


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

    @abstractmethod
    def generate(self, text: str, out_path: str, voice: str = "") -> bool:
        """Synthesise *text* and write audio to *out_path*. Returns success."""

    def generate_batch(
        self,
        requests: list[TTSRequest],
        *,
        max_workers: int = 4,
    ) -> list[TTSResult]:
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
