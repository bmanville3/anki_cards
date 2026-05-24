from pathlib import Path
import re

from src.common.types import RawChunk


def vtt_time_to_seconds(t: str) -> float:
    parts = t.strip().split(":")
    h, m, s = (parts if len(parts) == 3 else ["0"] + parts)
    return int(h) * 3600 + int(m) * 60 + float(s)


def parse_vtt(vtt_path: str) -> list[RawChunk]:
    text = Path(vtt_path).read_text(encoding="utf-8")
    pattern = re.compile(
        r"(\d{2}:\d{2}:\d{2}[\.,]\d{3}|\d{2}:\d{2}[\.,]\d{3})"
        r"\s*-->\s*"
        r"(\d{2}:\d{2}:\d{2}[\.,]\d{3}|\d{2}:\d{2}[\.,]\d{3})"
        r"[^\n]*\n"
        r"((?:.+\n?)+?)(?=\n|\Z)"
    )
    chunks: list[RawChunk] = []
    for m in pattern.finditer(text):
        chunk_text = m.group(3).strip()
        if chunk_text.startswith("NOTE") or not chunk_text:
            continue
        chunks.append(RawChunk(
            index=len(chunks),
            start=vtt_time_to_seconds(m.group(1).replace(",", ".")),
            end=vtt_time_to_seconds(m.group(2).replace(",", ".")),
            subtitle_text=chunk_text,
        ))
    return chunks
