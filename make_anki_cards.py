"""
Anki deck generator — integrated pipeline.

Concurrency model:
  - Per file, translation and sense-selection run in parallel thread pools
    (both hit the vLLM server; GIL is released during HTTP I/O).
  - TTS, audio extraction, and frame extraction run concurrently in a second
    thread pool during the card-generation pass.
  - Fish Audio S2 Pro is called via its local HTTP server (tools/api_server.py).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import httpx

try:
    import genanki
except ImportError:
    print("genanki not found.  pip install genanki")
    sys.exit(1)

from src.common.types import RawChunk, FullContext, Sense, SenseResult, CompletedChunk
from src.common.utils import server_available
from src.prompting.translator import translate_sentence as _translate_sentence
from src.prompting.sense_selector import select_senses

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_ENTRIES           = 10
MAX_SENSES            = 10
MAX_GLOSSES_PER_SENSE = 2
CONTEXT_PREV_WINDOW   = 40
CONTEXT_NEXT_WINDOW   = 1

# Thread pool sizes — translation and sense-selection are I/O-bound (HTTP to
# vLLM), so more threads than CPU cores is fine. TTS + ffmpeg are also I/O /
# subprocess-bound so the same logic applies.
LLM_WORKERS = 8   # concurrent vLLM requests (translation + sense selection)
MEDIA_WORKERS = 4  # concurrent TTS + ffmpeg jobs per card

TO_PROCESS_DIR = Path("./to_process")
if not TO_PROCESS_DIR.exists() or not TO_PROCESS_DIR.is_dir():
    raise ValueError(f"Please ensure {TO_PROCESS_DIR} exists and is a directory")
PROCESSED_DIR = Path("./processed")
PROCESSED_DIR.mkdir(exist_ok=True)
if not PROCESSED_DIR.is_dir():
    raise ValueError(f"Please make sure {PROCESSED_DIR} is a dir")

FONTS_DIR     = Path("./fonts")
JLPT_DATA_DIR = Path("./yomitan-jlpt-vocab")

JAPANESE_FONTS = [
    "NotoSansJP-Regular",
    "NotoSerifJP-Regular",
    "MPLUSRounded1c-Regular",
    "MPLUS1p-Regular",
    "KosugiMaru-Regular",
    "SawarabiGothic-Regular",
    "SawarabiMincho-Regular",
    "ZenKakuGothicNew-Regular",
    "ZenAntique-Regular",
    "ShipporiMincho-Regular",
]


# ── Font CSS ──────────────────────────────────────────────────────────────────

def build_font_face_css(fonts_dir: Path) -> str:
    if not fonts_dir.exists():
        print(f"  Warning: {fonts_dir} not found — no local fonts will be embedded.")
        return ""
    css_blocks = []
    for ttf in sorted(fonts_dir.glob("*.ttf")):
        family_name = ttf.stem.replace("-", " ").replace("_", " ")
        css_blocks.append(
            f"@font-face {{ font-family: '{family_name}'; src: url('{ttf.name}'); }}"
        )
    if not css_blocks:
        print(f"  Warning: no .ttf files found in {fonts_dir}.")
    return "\n".join(css_blocks)


# ── POS → CSS class ───────────────────────────────────────────────────────────

def _pos_to_class(pos_str: str) -> str:
    p = pos_str.lower()
    if any(x in p for x in ("verb", "vt", "vi", "vs", "vk", "v1", "v5")):
        return "pos-verb"
    if any(x in p for x in ("noun", "counter", "temporal")):
        return "pos-noun"
    if any(x in p for x in ("adjective", "adj")):
        return "pos-adj"
    if any(x in p for x in ("adverb", "adv")):
        return "pos-adv"
    if any(x in p for x in ("expression", "idiomatic")):
        return "pos-expr"
    return "pos-other"


# ── JLPT lookup ───────────────────────────────────────────────────────────────

def _load_jlpt_data(directory: Path) -> dict[str, int]:
    mapping: dict[str, int] = {}
    files = sorted(directory.glob("term_meta_bank_*.json"))
    if not files:
        print(f"  Warning: no term_meta_bank_*.json files found in {directory} — JLPT badges disabled.")
        return {}
    for f in files:
        try:
            entries = json.loads(f.read_text(encoding="utf-8"))
            for entry in entries:
                if not (isinstance(entry, list) and len(entry) >= 3):
                    continue
                word    = entry[0]
                meta    = entry[2]
                display = meta.get("frequency", {}).get("displayValue", "")
                if display.startswith("N") and display[1:].isdigit():
                    level = int(display[1:])
                    if word not in mapping or level > mapping[word]:
                        mapping[word] = level
        except Exception as e:
            print(f"  Failed to load {f.name}: {e}")
    print(f"  JLPT data loaded: {len(mapping)} entries.")
    return mapping


_JLPT_MAP: dict[str, int] = _load_jlpt_data(JLPT_DATA_DIR)


def _jlpt_for_result(result) -> str:
    if not _JLPT_MAP:
        return ""
    best: int | None = None
    for entry in result.entries:
        for form in list(entry.kanji_forms) + list(entry.kana_forms):
            level = _JLPT_MAP.get(form.text)
            if level is not None and (best is None or level > best):
                best = level
    return f"N{best}" if best is not None else ""


# ── Pitch accent ──────────────────────────────────────────────────────────────

def _kata_to_moras(kana: str) -> list[str]:
    SMALL = set("ァィゥェォャュョヮヵヶぁぃぅぇぉゃゅょゎ")
    moras, i = [], 0
    while i < len(kana):
        if i + 1 < len(kana) and kana[i + 1] in SMALL:
            moras.append(kana[i : i + 2])
            i += 2
        else:
            moras.append(kana[i])
            i += 1
    return moras


def _pitch_html(reading_kata: str, accent_type: str) -> str:
    if not reading_kata or not accent_type:
        return ""
    try:
        n = int(accent_type.split(",")[0])
    except ValueError:
        return ""
    moras = _kata_to_moras(reading_kata)
    if not moras:
        return ""
    parts = []
    for i, mora in enumerate(moras):
        mora_pos = i + 1
        if n == 0:
            high = mora_pos > 1
        elif n == 1:
            high = mora_pos == 1
        else:
            high = 2 <= mora_pos <= n
        cls = "pa-high" if high else "pa-low"
        parts.append(f'<span class="{cls}">{mora}</span>')
        if n != 0 and mora_pos == n:
            parts.append('<span class="pa-drop">↘</span>')
    pattern_label = "平板" if n == 0 else f"型{n}"
    return (
        f'<span class="pa-wrap">'
        f'{"".join(parts)}'
        f'<span class="pa-label">{pattern_label}</span>'
        f'</span>'
    )


# ── VTT parsing ───────────────────────────────────────────────────────────────

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


def _font_for_index(index: int) -> str:
    return JAPANESE_FONTS[index % len(JAPANESE_FONTS)]


# ── Sliding FullContext builder ───────────────────────────────────────────────

def build_full_context(
    target_index: int,
    all_chunks: list[RawChunk],
    prev_window: int = CONTEXT_PREV_WINDOW,
    next_window: int = CONTEXT_NEXT_WINDOW,
) -> FullContext:
    target_chunk    = all_chunks[target_index]
    previous_chunks = all_chunks[max(0, target_index - prev_window) : target_index]
    next_chunks     = all_chunks[target_index + 1 : target_index + 1 + next_window]
    return FullContext(
        target_chunk=target_chunk,
        previous_chunks=list(previous_chunks),
        next_chunks=list(next_chunks),
    )


# ── Word glosser (fugashi + jamdict) ─────────────────────────────────────────

_tagger = None
_jmd    = None


def setup_word_glosser() -> bool:
    global _tagger, _jmd
    ok = True
    try:
        import fugashi
        import unidic_lite
        _tagger = fugashi.Tagger(f"-d {unidic_lite.DICDIR}")
    except ImportError:
        print("  fugashi/unidic_lite not installed.  pip install fugashi unidic-lite")
        ok = False
    except Exception as e:
        print(f"  fugashi init failed: {e}")
        ok = False
    try:
        from jamdict import Jamdict
        _jmd = Jamdict()
    except ImportError:
        print("  jamdict not installed.  pip install jamdict")
        ok = False
    except Exception as e:
        print(f"  jamdict init failed: {e}")
        ok = False
    if ok:
        print("  Word glosser ready (fugashi + jamdict + vLLM sense selector).")
    return ok


_SKIP_POS = {"助詞", "助動詞", "記号", "補助記号", "空白"}


def _has_kanji(s: str) -> bool:
    return any("\u4e00" <= c <= "\u9fff" or "\u3400" <= c <= "\u4dbf" for c in s)


def _kana_to_hiragana(s: str) -> str:
    return "".join(
        chr(ord(c) - 0x60) if "\u30a1" <= c <= "\u30f6" else c for c in s
    )


def _jamdict_senses_for(surface: str, lemma: str) -> tuple[list[Sense], any]:
    if _jmd is None:
        return [], None
    result = _jmd.lookup(lemma) or _jmd.lookup(surface)
    if not result or not result.entries:
        return [], result
    senses: list[Sense] = []
    seen: set[tuple[str, str]] = set()
    idx = 0
    for entry in result.entries[:MAX_ENTRIES]:
        for jmd_sense in entry.senses[:MAX_SENSES]:
            glosses = [g.text for g in jmd_sense.gloss[:MAX_GLOSSES_PER_SENSE]]
            if not glosses:
                continue
            meaning = ", ".join(glosses)
            pos_str = "/".join(sorted(str(p) for p in jmd_sense.pos))
            key = (meaning, pos_str)
            if key in seen:
                continue
            seen.add(key)
            senses.append(Sense(index=idx, meaning=meaning, pos=pos_str))
            idx += 1
    return senses, result


@dataclass
class _TokenisedChunk:
    """Intermediate: tokenisation done (CPU), LLM sense selection pending."""
    chunk: RawChunk
    context: FullContext
    ruby_parts: list[str]
    words_and_senses: list[tuple[str, list[Sense]]]
    word_meta: dict[str, dict]


def _tokenise_chunk(chunk: RawChunk, context: FullContext) -> _TokenisedChunk:
    """
    Pass 1 (CPU-only): tokenise with fugashi, build furigana ruby parts,
    collect (word, senses) candidates for LLM sense selection.
    fugashi is not thread-safe, so this must be called from the main thread
    or a single-threaded executor — we call it sequentially before parallelising.
    """
    ruby_parts: list[str] = []
    words_and_senses: list[tuple[str, list[Sense]]] = []
    word_meta: dict[str, dict] = {}
    seen_gloss: set[str] = set()

    if _tagger is None:
        return _TokenisedChunk(chunk, context, [chunk.subtitle_text], [], {})

    for word in _tagger(chunk.subtitle_text):
        surface = word.surface
        if not surface.strip():
            ruby_parts.append(surface)
            continue

        try:
            reading_kata = word.feature.kana
        except AttributeError:
            reading_kata = None

        try:
            accent_type = word.feature.aType
            if accent_type == "*":
                accent_type = ""
        except AttributeError:
            accent_type = ""

        if reading_kata and _has_kanji(surface):
            reading_hira = _kana_to_hiragana(reading_kata)
            ruby_parts.append(f"<ruby>{surface}<rt>{reading_hira}</rt></ruby>")
        else:
            ruby_parts.append(surface)

        pos = word.feature.pos1
        if (
            pos in _SKIP_POS
            or surface in seen_gloss
            or (not _has_kanji(surface) and len(surface) <= 1)
        ):
            continue
        seen_gloss.add(surface)

        lemma = (word.feature.lemma or surface).split("-")[0]
        senses, result = _jamdict_senses_for(surface, lemma)
        if not senses:
            continue

        words_and_senses.append((surface, senses))
        word_meta[surface] = {
            "result":       result,
            "reading_kata": reading_kata or "",
            "accent_type":  accent_type,
        }

    return _TokenisedChunk(chunk, context, ruby_parts, words_and_senses, word_meta)


def _render_gloss_html(tc: _TokenisedChunk, sense_results: list[SenseResult]) -> tuple[str, str]:
    """Pass 3 (CPU-only): render furigana + word gloss HTML from sense results."""
    sense_map = {sr.word: sr for sr in sense_results}
    gloss_pairs: list[str] = []

    for surface, _ in tc.words_and_senses:
        sr   = sense_map.get(surface)
        meta = tc.word_meta[surface]
        if sr is None:
            continue
        if sr.custom_definition:
            display_meanings = [sr.custom_definition]
            display_poses    = ["custom"]
        elif sr.selected:
            display_meanings = [s.meaning for s in sr.selected]
            display_poses    = [s.pos     for s in sr.selected]
        else:
            continue

        result       = meta["result"]
        reading_kata = meta["reading_kata"]
        accent_type  = meta["accent_type"]
        jlpt         = _jlpt_for_result(result) if result else ""
        pitch_html   = _pitch_html(_kana_to_hiragana(reading_kata) if reading_kata else "", accent_type)

        jlpt_span  = f'<span class="jlpt-badge jlpt-{jlpt}">{jlpt}</span>' if jlpt else ""
        pitch_span = f'<span class="pa-container">{pitch_html}</span>'       if pitch_html else ""
        header     = f'<span class="gloss-word">{surface}</span>{jlpt_span}{pitch_span}'
        lines      = [header]
        for i, meaning in enumerate(display_meanings, 1):
            pos_str   = display_poses[i - 1]
            pos_class = _pos_to_class(pos_str)
            lines.append(
                f'<span class="{pos_class} gloss-line">'
                f'{i}) {meaning}'
                f'<span class="pos-tag"> [{pos_str}]</span>'
                f'</span>'
            )
        gloss_pairs.append("<br>".join(lines))

    furigana_html = "".join(tc.ruby_parts)
    word_gloss    = "<br><br>".join(gloss_pairs)
    return furigana_html, word_gloss


# ── TTS (Fish Audio S2 Pro) ───────────────────────────────────────────────────

FISH_TTS_URL        = "http://127.0.0.1:8080/v1/tts"
FISH_HEALTH_URL     = "http://127.0.0.1:8080/v1/health"
_fish_tts_available = False

# A shared httpx client with a connection pool — safe to use from multiple
# threads and avoids re-establishing TCP connections on every TTS request.
_fish_client = httpx.Client(timeout=60, limits=httpx.Limits(max_connections=MEDIA_WORKERS))


def setup_tts() -> bool:
    global _fish_tts_available
    try:
        r = _fish_client.get(FISH_HEALTH_URL, timeout=3)
        if r.status_code == 200 and r.json().get("status") == "ok":
            _fish_tts_available = True
            print("  TTS ready (Fish Audio S2 Pro).")
            return True
        print(f"  Fish Audio server unhealthy: {r.text}")
        return False
    except Exception as e:
        print(f"  Fish Audio S2 Pro server not reachable at {FISH_TTS_URL}: {e}")
        print(
            "  Start it with: python tools/api_server.py "
            "--llama-checkpoint-path checkpoints/s2-pro "
            "--decoder-checkpoint-path checkpoints/s2-pro/codec.pth "
            "--listen 0.0.0.0:8080 --half"
        )
        return False


def generate_tts(text: str, out_path: str, voice: str = "") -> bool:
    if not _fish_tts_available or not text.strip():
        return False
    try:
        payload: dict = {"text": text, "format": "mp3", "streaming": False}
        if voice:
            payload["reference_id"] = voice
        r = _fish_client.post(FISH_TTS_URL, json=payload)
        if r.status_code != 200:
            print(f"  TTS error {r.status_code}: {r.text[:200]}")
            return False
        Path(out_path).write_bytes(r.content)
        return True
    except Exception as e:
        print(f"  TTS error: {e}")
        return False


# ── ffmpeg helpers ────────────────────────────────────────────────────────────

def extract_audio(mp4: str, start: float, end: float, out: str) -> bool:
    duration = max(end - start, 0.5)
    r = subprocess.run(
        ["ffmpeg", "-y", "-ss", str(start), "-i", mp4,
         "-t", str(duration), "-q:a", "4", "-vn", out],
        capture_output=True,
    )
    if r.returncode != 0:
        print(f"  ffmpeg audio error: {r.stderr.decode(errors='replace')[-200:]}")
    return r.returncode == 0


def extract_frame(mp4: str, timestamp: float, out: str) -> bool:
    r = subprocess.run(
        ["ffmpeg", "-y", "-ss", str(max(timestamp, 0)), "-i", mp4,
         "-frames:v", "1", "-q:v", "3", out],
        capture_output=True,
    )
    if r.returncode != 0:
        print(f"  ffmpeg frame error: {r.stderr.decode(errors='replace')[-200:]}")
    return r.returncode == 0


def check_ffmpeg() -> None:
    if subprocess.run(["ffmpeg", "-version"], capture_output=True).returncode != 0:
        print("ffmpeg not found.  brew install ffmpeg  /  apt install ffmpeg")
        sys.exit(1)


# ── Anki card model ───────────────────────────────────────────────────────────

MODEL_ID = 1_234_567_895
DECK_ID  = 9_876_543_210

FONT_FACE_CSS = build_font_face_css(FONTS_DIR)

CARD_CSS = FONT_FACE_CSS + """
.card {
  font-size: 24px; text-align: center; background: #1e1e2e;
  color: #cdd6f4; padding: 24px 20px; line-height: 1.7;
}
.jp-text { font-size: 30px; margin-bottom: 10px; }
.jp-furigana { font-size: 26px; margin-bottom: 10px; line-height: 2.2; }
ruby rt { font-size: 0.45em; color: #89b4fa; }
.timecode { font-size: 11px; color: #585b70; margin-top: 6px; }
hr { border: none; border-top: 1px solid #313244; margin: 16px 0; }
.frame-wrap img { max-width: 100%; border-radius: 10px; margin-bottom: 12px; }
.translation { font-size: 18px; color: #a6e3a1; font-style: italic; margin-bottom: 10px; }
.gloss-label, .furigana-label, .notes-label, .source-label {
  font-size: 11px; color: #585b70; text-transform: uppercase;
  letter-spacing: 0.08em; margin-top: 14px;
}
.word-gloss {
  font-size: 14px; color: #cdd6f4; margin-top: 4px;
  line-height: 2.0; text-align: left; display: inline-block;
}
.gloss-word { font-size: 15px; font-weight: bold; color: #cdd6f4; padding-right: 6px; }
.gloss-line { display: block; margin-left: 8px; }
.pos-tag { font-size: 11px; opacity: 0.6; }
.pos-verb  { color: #89dceb; }
.pos-noun  { color: #cdd6f4; }
.pos-adj   { color: #a6e3a1; }
.pos-adv   { color: #f9e2af; }
.pos-expr  { color: #cba6f7; }
.pos-other { color: #bac2de; }
.jlpt-badge {
  display: inline-block; font-size: 10px; font-weight: bold;
  padding: 1px 5px; border-radius: 4px; margin-left: 5px;
  vertical-align: middle; color: #1e1e2e;
}
.jlpt-N5 { background: #a6e3a1; }
.jlpt-N4 { background: #89dceb; }
.jlpt-N3 { background: #f9e2af; }
.jlpt-N2 { background: #fab387; }
.jlpt-N1 { background: #f38ba8; }
.pa-container { display: inline-block; margin-left: 8px; vertical-align: middle; }
.pa-wrap { display: inline-flex; align-items: flex-end; font-size: 12px; gap: 0; }
.pa-high { color: #89b4fa; border-top: 2px solid #89b4fa; padding: 0 1px; }
.pa-low  { color: #89b4fa; border-top: 2px solid transparent; padding: 0 1px; }
.pa-drop { color: #f38ba8; font-size: 10px; margin: 0 1px; align-self: center; }
.pa-label { font-size: 10px; color: #585b70; margin-left: 4px; align-self: center; }
.source-name { font-size: 15px; color: #bac2de; min-height: 1.4em; margin-top: 4px; padding-bottom: 4px; }
.notes {
  font-size: 15px; color: #bac2de; min-height: 1.4em;
  border-bottom: 1px dashed #45475a; margin-top: 4px; padding-bottom: 4px;
}
"""

CARD_MODEL = genanki.Model(
    MODEL_ID,
    "Japanese Video Cards",
    fields=[
        {"name": "Text"},
        {"name": "Audio"},
        {"name": "Image"},
        {"name": "Translation"},
        {"name": "TTSAudio"},
        {"name": "Furigana"},
        {"name": "WordGloss"},
        {"name": "TimeCode"},
        {"name": "FontName"},
        {"name": "Source"},
        {"name": "Notes"},
    ],
    templates=[{
        "name": "Card 1",
        "qfmt": (
            "<div class=\"jp-text\" style=\"font-family: '{{FontName}}', sans-serif;\">{{Text}}</div>\n"
            "{{Audio}}\n"
            "<div class=\"timecode\">{{TimeCode}}</div>\n"
        ),
        "afmt": (
            "{{FrontSide}}\n<hr>\n"
            "{{#Furigana}}\n"
            "<div class=\"jp-furigana\" style=\"font-family: '{{FontName}}', sans-serif;\">{{Furigana}}</div>\n"
            "{{/Furigana}}\n"
            "{{#Translation}}\n"
            "<div class=\"translation\">{{Translation}}</div>\n"
            "{{/Translation}}\n"
            "{{TTSAudio}}\n"
            "<div class=\"frame-wrap\">{{Image}}</div>\n"
            "{{#WordGloss}}\n"
            "<div class=\"gloss-label\">Word by word</div>\n"
            "<div class=\"word-gloss\">{{WordGloss}}</div>\n"
            "{{/WordGloss}}\n"
            "<div class=\"notes-label\">Notes</div>\n"
            "<div class=\"notes\">{{Notes}}</div>\n"
            "<div class=\"source-label\">Source</div>\n"
            "<div class=\"source-name\">{{Source}}</div>\n"
        ),
    }],
    css=CARD_CSS,
)


# ── File discovery ────────────────────────────────────────────────────────────

def find_pairs(directory: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for media in sorted(directory.iterdir()):
        if media.suffix.lower() not in (".mp4", ".mp3"):
            continue
        vtt = media.with_suffix(".vtt")
        if vtt.exists():
            pairs.append((media, vtt))
        else:
            print(f"  Skipping {media.name} — no matching .vtt found")
    return pairs


def already_processed(directory: Path) -> set[str]:
    return {f.name for f in directory.iterdir()}


# ── Core processing ───────────────────────────────────────────────────────────

def process_file(
    media_path: Path,
    vtt_path: Path,
    deck: genanki.Deck,
    all_media_files: list[str],
    tmpdir: str,
    frame_offset: float,
    tts_voice: str,
) -> None:
    original_name = media_path.name
    is_mp3        = media_path.suffix.lower() == ".mp3"

    safe_id    = uuid.uuid4().hex[:12]
    safe_media = Path(tmpdir) / f"media_{safe_id}{media_path.suffix.lower()}"
    safe_vtt   = Path(tmpdir) / f"vtt_{safe_id}.vtt"
    shutil.copy2(media_path, safe_media)
    shutil.copy2(vtt_path,   safe_vtt)

    print(f"\n  Parsing VTT: {vtt_path.name}")
    raw_chunks: list[RawChunk] = parse_vtt(str(safe_vtt))
    n = len(raw_chunks)
    print(f"    Found {n} chunks")

    # ── Stage 1 (main thread, sequential): tokenise all chunks ───────────────
    # fugashi's Tagger is not thread-safe, so we do this before parallelising.
    print(f"    Tokenising {n} chunks ...")
    contexts: list[FullContext] = [
        build_full_context(i, raw_chunks) for i in range(n)
    ]
    tokenised: list[_TokenisedChunk] = [
        _tokenise_chunk(raw_chunks[i], contexts[i]) for i in range(n)
    ]

    # ── Stage 2 (parallel): translation AND sense selection hit vLLM ─────────
    # Both pools run concurrently via submit-then-collect; the GIL is released
    # for the duration of each HTTP call so true parallelism is achieved.
    print(f"    Running translation + sense selection in parallel ({LLM_WORKERS} workers) ...")

    translations:  list[str]             = [""] * n
    sense_results: list[list[SenseResult]] = [[] for _ in range(n)]

    def _do_translation(i: int) -> tuple[int, str]:
        result = _translate_sentence(
            context=contexts[i],
            base64_encoded_image=None,
            image_mime="image/jpeg",
        )
        return i, result

    def _do_sense_selection(i: int) -> tuple[int, list[SenseResult]]:
        tc     = tokenised[i]
        result = select_senses(tc.context, tc.words_and_senses)
        return i, result

    with ThreadPoolExecutor(max_workers=LLM_WORKERS) as pool:
        # Submit all translation and sense-selection jobs together so vLLM can
        # continuous-batch across both task types simultaneously.
        translation_futures = {pool.submit(_do_translation, i): i for i in range(n)}
        sense_futures       = {pool.submit(_do_sense_selection, i): i for i in range(n)}
        all_futures         = {**translation_futures, **sense_futures}

        completed_count = 0
        for future in as_completed(all_futures):
            idx, payload = future.result()
            if future in translation_futures:
                translations[idx] = payload
            else:
                sense_results[idx] = payload
            completed_count += 1
            if completed_count % 10 == 0:
                print(f"      {completed_count}/{n * 2} LLM tasks done ...")

    # ── Stage 3 (main thread, sequential): render gloss HTML ─────────────────
    print(f"    Rendering gloss HTML ...")
    completed: list[CompletedChunk] = []
    for i, tc in enumerate(tokenised):
        furigana, word_gloss = _render_gloss_html(tc, sense_results[i])
        translation = translations[i]

        print(f"      [{i+1}/{n}] {tc.chunk.subtitle_text[:45]}")
        if translation:
            print(f"        → {translation[:65]}")
        if word_gloss:
            print(f"        ≈ {re.sub(r'<[^>]+>', '', word_gloss)[:65]}")

        completed.append(CompletedChunk(
            index=tc.chunk.index,
            start=tc.chunk.start,
            end=tc.chunk.end,
            subtitle_text=tc.chunk.subtitle_text,
            translation=translation,
            word_gloss=word_gloss,
            furigana=furigana,
        ))

    # ── Stage 4 (parallel): TTS + audio extraction + frame extraction ─────────
    # All three are I/O-bound (HTTP to Fish, subprocess ffmpeg) and fully
    # independent per card, so we parallelise across cards.
    print(f"    Generating media for {n} cards in parallel ({MEDIA_WORKERS} workers) ...")

    @dataclass
    class _CardMedia:
        audio_fname: str = ""
        image_fname: str = ""
        tts_fname:   str = ""
        audio_ok:    bool = False
        image_ok:    bool = False
        tts_ok:      bool = False

    def _process_card_media(chunk: CompletedChunk) -> tuple[int, _CardMedia]:
        idx       = chunk.index
        card_uuid = uuid.uuid4().hex
        media     = _CardMedia(
            audio_fname=f"clip_{idx:04d}_{card_uuid}.mp3",
            image_fname=f"frame_{idx:04d}_{card_uuid}.jpg",
            tts_fname=f"tts_{idx:04d}_{card_uuid}.mp3",
        )
        audio_path = os.path.join(tmpdir, media.audio_fname)
        image_path = os.path.join(tmpdir, media.image_fname)
        tts_path   = os.path.join(tmpdir, media.tts_fname)

        media.audio_ok = extract_audio(str(safe_media), chunk.start, chunk.end, audio_path)
        if not is_mp3:
            frame_ts = chunk.start + min(frame_offset, (chunk.end - chunk.start) * 0.5)
            media.image_ok = extract_frame(str(safe_media), frame_ts, image_path)
        if chunk.translation:
            media.tts_ok = generate_tts(chunk.translation, tts_path, voice=tts_voice)

        return idx, media

    card_media: dict[int, _CardMedia] = {}
    with ThreadPoolExecutor(max_workers=MEDIA_WORKERS) as pool:
        futures = {pool.submit(_process_card_media, chunk): chunk.index for chunk in completed}
        for future in as_completed(futures):
            idx, media = future.result()
            card_media[idx] = media
            chunk = completed[idx]
            print(f"      [{idx+1}/{n}] {chunk.start:.1f}s–{chunk.end:.1f}s  {chunk.subtitle_text[:35]}")
            if not media.audio_ok:                print("        Warning: audio extraction failed")
            if not media.image_ok and not is_mp3: print("        Warning: frame extraction failed")
            if not media.tts_ok:                  print("        Warning: TTS generation failed")

    # ── Stage 5 (main thread): assemble Anki notes in original order ──────────
    for chunk in completed:
        media     = card_media[chunk.index]
        idx       = chunk.index
        audio_path = os.path.join(tmpdir, media.audio_fname)
        image_path = os.path.join(tmpdir, media.image_fname)
        tts_path   = os.path.join(tmpdir, media.tts_fname)
        timecode   = f"{int(chunk.start // 60):02d}:{chunk.start % 60:05.2f}"

        note = genanki.Note(
            model=CARD_MODEL,
            fields=[
                chunk.subtitle_text,
                f"[sound:{media.audio_fname}]"     if media.audio_ok else "",
                f'<img src="{media.image_fname}">' if media.image_ok else "",
                chunk.translation,
                f"[sound:{media.tts_fname}]"       if media.tts_ok   else "",
                chunk.furigana,
                chunk.word_gloss,
                timecode,
                _font_for_index(idx),
                original_name,
                "",
            ],
        )
        deck.add_note(note)

        if media.audio_ok: all_media_files.append(audio_path)
        if media.image_ok: all_media_files.append(image_path)
        if media.tts_ok:   all_media_files.append(tts_path)

    safe_media.unlink(missing_ok=True)
    safe_vtt.unlink(missing_ok=True)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Batch Anki deck generator from to_process/ directory")
    parser.add_argument("--frame-offset", type=float, default=1.0,
                        help="Seconds after chunk start to grab frame (default: 1.0)")
    parser.add_argument("--deck-name", default="Japanese Video Deck")
    parser.add_argument("--tts-voice", default="",
                        help="Fish Audio reference_id for a saved speaker voice (optional)")
    parser.add_argument("--llm-workers", type=int, default=LLM_WORKERS,
                        help=f"Concurrent vLLM requests for translation+sense (default: {LLM_WORKERS})")
    parser.add_argument("--media-workers", type=int, default=MEDIA_WORKERS,
                        help=f"Concurrent TTS+ffmpeg jobs per card (default: {MEDIA_WORKERS})")
    args = parser.parse_args()

    check_ffmpeg()

    if not server_available():
        print("Warning: vLLM server not reachable at localhost:9090 — LLM calls will fail.")

    pairs = find_pairs(TO_PROCESS_DIR)
    if not pairs:
        print(f"No media+vtt pairs found in {TO_PROCESS_DIR}. Nothing to do.")
        sys.exit(0)

    print(f"Found {len(pairs)} pair(s) to process:")
    for media, vtt in pairs:
        print(f"  {media.name}  +  {vtt.name}")

    if not setup_word_glosser():
        raise ValueError("Failed to set up word glosser")
    if not setup_tts():
        raise ValueError("Failed to set up TTS")

    font_files  = list(FONTS_DIR.glob("*.ttf")) if FONTS_DIR.exists() else []
    already_done = already_processed(PROCESSED_DIR)

    # Patch worker counts from CLI if provided
    global LLM_WORKERS, MEDIA_WORKERS
    LLM_WORKERS   = args.llm_workers
    MEDIA_WORKERS = args.media_workers

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, (media_path, vtt_path) in enumerate(pairs):
            if media_path.name in already_done:
                print(f"{media_path.name} already processed. Skipping.")
                continue
            already_done.add(media_path.name)

            deck        = genanki.Deck(DECK_ID, args.deck_name)
            media_files = [str(f) for f in font_files]
            output      = f"output_{media_path.stem}.apkg"

            print(f"\n{'='*60}")
            print(f"Processing ({i+1}/{len(pairs)}): {media_path.name}")
            try:
                process_file(
                    media_path, vtt_path,
                    deck, media_files, tmpdir,
                    args.frame_offset,
                    tts_voice=args.tts_voice,
                )
                print(f"\n{'='*60}")
                print(f"Writing {output} ...")
                pkg = genanki.Package(deck)
                pkg.media_files = media_files
                pkg.write_to_file(output)

                shutil.move(str(media_path), PROCESSED_DIR / media_path.name)
                shutil.move(str(vtt_path),   PROCESSED_DIR / vtt_path.name)
                print(f"  Moved source files to {PROCESSED_DIR}/")
                print(f"  Done: {output}")
            except Exception as e:
                print(f"  ERROR processing {media_path.name}: {e}")
                import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
