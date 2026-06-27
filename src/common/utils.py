import base64
import logging
import mimetypes
from pathlib import Path
import socket
import unicodedata

import requests

from src.common.types import RawChunk

logger = logging.getLogger(__name__)

MIMES = (
    "png", "jpeg", "gif", "tiff", "bmp"
)
MIME_OVERRIDES = {
    "jpg": "image/jpeg", ".jpg": "image/jpeg",
    "pdf": "application/pdf", ".pdf": "application/pdf",
}
MIME_TO_LLM_MIME = {
    **{k: v for m in MIMES for k, v in ((m, f"image/{m}"), (f".{m}", f"image/{m}"))},
    **MIME_OVERRIDES,
}

def load_image_b64(path: Path, default_mime: str = "image/jpeg") -> tuple[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Fixture not found: {path}")
    suffix = path.suffix.lower()
    mime = MIME_TO_LLM_MIME.get(suffix) or mimetypes.guess_type(path)[0]
    if mime is None:
        mime = default_mime
        logger.warning("Unsupported image format: %r. Defaulting to %s", path, mime)
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode(), mime


def assert_latin_extended_only(string: str) -> None:
    invalid = []
    for char in string:
        cp = ord(char)
        if 0x20 <= cp <= 0x7E:
            continue
        if char in "''""…–—•·©®™°":
            continue
        if unicodedata.category(char).startswith("L") and unicodedata.name(char, "").startswith("LATIN"):
            continue
        invalid.append(char)
    if invalid:
        raise ValueError(
            f"Response contained non-English characters: {invalid}\n"
            f"Previous response:\n{string}"
        )


def only_japanese_and_punctuation(text: str) -> bool:
    for char in text:
        cp = ord(char)
        if 0x20 <= cp <= 0x40 or 0x5B <= cp <= 0x60 or 0x7B <= cp <= 0x7E:
            continue
        cat = unicodedata.category(char)
        name = unicodedata.name(char, "")
        if cat.startswith("L") and name.startswith("LATIN"):
            return False
    return True


def check_chunk_ordering(
    target_chunk: RawChunk,
    previous_chunks: list[RawChunk],
    next_chunk: RawChunk | None,
) -> None:
    all_chunks = [*previous_chunks, target_chunk]
    if next_chunk:
        all_chunks.append(next_chunk)

    last_seen: int | None = None
    for chunk in all_chunks:
        if last_seen is not None:
            if chunk.index < last_seen:
                logger.warning("Chunk ordering is out of order — index %d follows %d", chunk.index, last_seen)
            elif chunk.index > last_seen + 1:
                logger.warning("Skipped indices between %d and %d", last_seen, chunk.index)
        last_seen = chunk.index


def server_available(host: str = "localhost", port: int = 9090) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def server_available_url(url: str) -> bool:
    try:
        response = requests.head(url, allow_redirects=True, timeout=1)
        if response.status_code == 405:
            response = requests.get(url, stream=True, timeout=1)
        return response.ok
    except requests.RequestException:
        return False
