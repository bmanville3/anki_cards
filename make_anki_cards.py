import argparse
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
import uuid
import shutil

try:
    import genanki
except ImportError:
    print("genanki not found.  pip install genanki")
    sys.exit(1)

MAX_ENTRIES = 10
MAX_SENSES = 10
MAX_GLOSSES_PER_SENSE = 2

TO_PROCESS_DIR = Path("./to_process")
if not TO_PROCESS_DIR.exists() or not TO_PROCESS_DIR.is_dir():
    raise ValueError(f"Please ensure {TO_PROCESS_DIR} exists and is a directory")
PROCESSED_DIR = Path("./processed")
if not PROCESSED_DIR.exists():
    PROCESSED_DIR.mkdir()
if not PROCESSED_DIR.is_dir():
    raise ValueError(f"Please make sure {PROCESSED_DIR} is a dir")

FONTS_DIR = Path("./fonts")

# Japanese fonts - rotated per card.
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

def build_font_face_css(fonts_dir: Path) -> str:
    """Build @font-face blocks for every .ttf in fonts_dir."""
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


@dataclass
class Chunk:
    index: int
    start: float
    end: float
    text: str
    translation: str = ""   # full sentence MT
    word_gloss: str = ""    # token=meaning · token=meaning
    furigana: str = ""      # HTML <ruby> markup for back of card
    font: str = ""


def vtt_time_to_seconds(t: str) -> float:
    parts = t.strip().split(":")
    h, m, s = (parts if len(parts) == 3 else [0] + parts)
    return int(h) * 3600 + int(m) * 60 + float(s)


def parse_vtt(vtt_path: str) -> list:
    text = Path(vtt_path).read_text(encoding="utf-8")
    pattern = re.compile(
        r"(\d{2}:\d{2}:\d{2}[\.,]\d{3}|\d{2}:\d{2}[\.,]\d{3})"
        r"\s*-->\s*"
        r"(\d{2}:\d{2}:\d{2}[\.,]\d{3}|\d{2}:\d{2}[\.,]\d{3})"
        r"[^\n]*\n"
        r"((?:.+\n?)+?)(?=\n|\Z)"
    )
    chunks = []
    for i, m in enumerate(pattern.finditer(text)):
        chunk_text = m.group(3).strip()
        if chunk_text.startswith("NOTE") or not chunk_text:
            continue
        chunks.append(Chunk(
            index=i,
            start=vtt_time_to_seconds(m.group(1).replace(",", ".")),
            end=vtt_time_to_seconds(m.group(2).replace(",", ".")),
            text=chunk_text,
            font=JAPANESE_FONTS[len(chunks) % len(JAPANESE_FONTS)],
        ))
    return chunks


# ── Sentence translation: Qwen2.5 via llama-cpp ──────────────────────────────

MODEL_PATH = os.path.expanduser("~/models/Qwen2.5-7B-Instruct-Q5_K_M.gguf")

_llm = None

def setup_sentence_translator() -> bool:
    global _llm
    try:
        from llama_cpp import Llama
        print(f"Loading {MODEL_PATH} ...")
        _llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=2048,
            n_gpu_layers=-1,   # offload everything to Metal GPU
            verbose=False,
        )
        print("  LLM translation model ready.")
        return True
    except ImportError:
        print("  llama-cpp-python not installed.")
        print("  CMAKE_ARGS=\"-DGGML_METAL=on\" pip install llama-cpp-python")
        return False
    except Exception as e:
        print(f"  Failed to load LLM: {e}")
        return False


_TRANSLATION_SYSTEM = (
    "You are a Japanese-to-English translator specializing in natural, idiomatic English. "
    "The input is a single subtitle line from a Japanese TV show or video. "
    "Rules:\n"
    "- Output ONLY the English translation, nothing else — no notes, no explanations.\n"
    "- Produce natural English a native speaker would say.\n"
    "- Japanese often omits the subject; infer it from context and use 'it', 'they', 'you', etc. appropriately.\n"
    "- Sound cues like （笑）、♪、or [拍手] should be passed through as-is or rendered as a brief parenthetical like (laughter).\n"
    "- Prefer the most common/literal reading unless it sounds unnatural.\n"
    "- Never add anything that isn't in the original."
)

def translate_sentence(text: str) -> str:
    if _llm is None:
        return ""
    try:
        prompt = f"Translate this Japanese subtitle line to natural English:\n{text}"
        response = _llm.create_chat_completion(
            messages=[
                {"role": "system", "content": _TRANSLATION_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=512,
            temperature=0.1,
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"  Translation error: {e}")
        return ""


# ── TTS: Kokoro (local, offline) ──────────────────────────────────────────────

KOKORO_MODEL_PATH  = os.path.expanduser("~/models/kokoro/kokoro-v1.0.onnx")
KOKORO_VOICES_PATH = os.path.expanduser("~/models/kokoro/voices-v1.0.bin")

_kokoro = None

def setup_tts() -> bool:
    global _kokoro
    try:
        from kokoro_onnx import Kokoro
        print(f"Loading Kokoro TTS...")
        _kokoro = Kokoro(KOKORO_MODEL_PATH, KOKORO_VOICES_PATH)
        print("  TTS ready (Kokoro).")
        return True
    except ImportError:
        print("  kokoro-onnx not installed.  pip install kokoro-onnx soundfile")
        return False
    except Exception as e:
        print(f"  Failed to load Kokoro: {e}")
        return False


def generate_tts(text: str, out_path: str, voice: str = "af_heart") -> bool:
    """Synthesize English text to an mp3 file. Returns True on success."""
    if _kokoro is None or not text.strip():
        return False
    try:
        import soundfile as sf
        samples, sample_rate = _kokoro.create(text, voice=voice, speed=1.0, lang="en-us")
        # soundfile writes wav; ffmpeg converts to mp3 so Anki handles it fine
        wav_path = out_path.replace(".mp3", "_tts_tmp.wav")
        sf.write(wav_path, samples, sample_rate)
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path, "-q:a", "4", out_path],
            capture_output=True,
        )
        Path(wav_path).unlink(missing_ok=True)
        if r.returncode != 0:
            print(f"  TTS ffmpeg error: {r.stderr.decode(errors='replace')[-200:]}")
            return False
        return True
    except Exception as e:
        print(f"  TTS error: {e}")
        return False


# ── Word-for-word gloss: fugashi (MeCab) + jamdict (JMdict) ──────────────────

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
        print("  Word glosser ready (fugashi + jamdict).")
    return ok


_SKIP_POS = {
    "助詞",      # particles
    "助動詞",    # auxiliary verbs
    "記号",      # punctuation / symbols
    "補助記号",  # supplementary symbols
    "空白",      # whitespace
}

def _has_kanji(s: str) -> bool:
    return any('\u4e00' <= c <= '\u9fff' or '\u3400' <= c <= '\u4dbf' for c in s)


def _kana_to_hiragana(s: str) -> str:
    return "".join(
        chr(ord(c) - 0x60) if '\u30a1' <= c <= '\u30f6' else c
        for c in s
    )

def _extract_from_cutting(result) -> tuple[list, list]:
    senses = [s for e in result.entries[:MAX_ENTRIES] for s in e.senses]
    meanings = []
    poses = []
    for sense in senses[:MAX_SENSES]:
        glosses = [g.text for g in sense.gloss[:MAX_GLOSSES_PER_SENSE]]
        if glosses:
            poses.append("/".join(sorted(str(p) for p in sense.pos)))
            meanings.append(", ".join(glosses))
    return meanings, poses

_GLOSS_SELECTION_SYSTEM = (
    "You are a Japanese dictionary assistant. "
    "You will be given a Japanese paragraph, a word, and a numbered list of its dictionary definitions. "
    "Return ONLY a comma-separated list of the numbers of the most relevant and common definitions — "
    "If the word has an extremely common definition outside of the context, return the context definition number(s) and the very common definition number(s) "
    "However, bias toward returning the context definition number(s) "
    "If multiple definitions could fit the sentence, return each definition. "
    "Make sure you respond in a comma separate list of numbers like a csv "
    "no explanations, no other text. Example output: 1,3 "
)

def _extract_from_llm_prompt(context: str, surface: str, result) -> tuple[list, list]:
    if _llm is None:
        return _extract_from_cutting(result=result)
    senses = [s for e in result.entries for s in e.senses]
    seen = set()
    meanings = []
    poses = []
    for sense in senses:
        glosses = [g.text for g in sense.gloss]
        if glosses:
            new_pos     = "/".join(sorted(str(p) for p in sense.pos))
            new_meaning = ", ".join(glosses)
            key = (new_meaning, new_pos)
            if key not in seen:
                seen.add(key)
                meanings.append(new_meaning)
                poses.append(new_pos)

    if not meanings:
        return [], []

    try:
        numbered = "\n".join(f"{i+1}) {m} [{poses[i]}]" for i, m in enumerate(meanings))
        prompt = (
            f"Context: {context}\n"
            f"Word: {surface}\n"
            f"Definitions:\n{numbered}\n\n"
            f"Which numbers are the most common/relevant to the context? "
            f"Return only the numbers, comma-separated."
        )
        response = _llm.create_chat_completion(
            messages=[
                {"role": "system", "content": _GLOSS_SELECTION_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=32,
            temperature=0.0,
        )
        raw = response["choices"][0]["message"]["content"].strip()
        indices = [int(x.strip()) - 1 for x in re.split(r"[,\s]+", raw) if x.strip().isdigit()]
        indices = [i for i in indices if 0 <= i < len(meanings)]
        if indices:
            return [meanings[i] for i in indices], [poses[i] for i in indices]
        return _extract_from_cutting(result=result)
    except Exception as e:
        print(f"  LLM gloss selection error: {e}")
        return _extract_from_cutting(result=result)


def analyze_sentence(text: str) -> tuple[str, str]:
    if _tagger is None:
        return text, ""

    ruby_parts = []
    gloss_pairs = []
    seen_gloss = set()

    for word in _tagger(text):
        surface = word.surface
        if not surface.strip():
            ruby_parts.append(surface)
            continue

        try:
            reading_kata = word.feature.kana
        except AttributeError:
            reading_kata = None

        if reading_kata and _has_kanji(surface):
            reading_hira = _kana_to_hiragana(reading_kata)
            ruby_parts.append(f"<ruby>{surface}<rt>{reading_hira}</rt></ruby>")
        else:
            ruby_parts.append(surface)

        pos = word.feature.pos1
        if pos in _SKIP_POS or surface in seen_gloss or not _has_kanji(surface) and len(surface) <= 1:
            continue
        seen_gloss.add(surface)

        lemma = (word.feature.lemma or surface).split("-")[0]
        result = None
        if _jmd is not None:
            result = _jmd.lookup(lemma) or _jmd.lookup(surface)

        if result and result.entries:
            meanings, poses = _extract_from_llm_prompt(context=text, surface=surface, result=result)
            formatted_meanings = []
            for i, sense_meaning in enumerate(meanings, 1):
                pos_for_sense = poses[i - 1]
                formatted_meanings.append(f"{i}) {sense_meaning} - {pos_for_sense}")
            if formatted_meanings:
                gloss_pairs.append(f"{surface}:<br>" + "<br>".join(formatted_meanings))

    furigana_html = "".join(ruby_parts)
    word_gloss    = "<br><br>".join(gloss_pairs)
    return furigana_html, word_gloss


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


def check_ffmpeg():
    if subprocess.run(["ffmpeg", "-version"], capture_output=True).returncode != 0:
        print("ffmpeg not found.  brew install ffmpeg  /  apt install ffmpeg")
        sys.exit(1)


# ── Card model ────────────────────────────────────────────────────────────────

MODEL_ID = 1_234_567_894   # bumped — new field added (TTSAudio)
DECK_ID  = 9_876_543_210

# Build font CSS from local files — no Google Fonts calls at card render time
FONT_FACE_CSS = build_font_face_css(FONTS_DIR)

CARD_CSS = FONT_FACE_CSS + """

.card {
  font-size: 24px;
  text-align: center;
  background: #1e1e2e;
  color: #cdd6f4;
  padding: 24px 20px;
  line-height: 1.7;
}
.jp-text {
  font-size: 30px;
  margin-bottom: 10px;
}
.jp-furigana {
  font-size: 26px;
  margin-bottom: 10px;
  line-height: 2.2;
}
ruby rt {
  font-size: 0.45em;
  color: #89b4fa;
}
.timecode {
  font-size: 11px;
  color: #585b70;
  margin-top: 6px;
}
hr {
  border: none;
  border-top: 1px solid #313244;
  margin: 16px 0;
}
.frame-wrap img {
  max-width: 100%;
  border-radius: 10px;
  margin-bottom: 12px;
}
.translation {
  font-size: 18px;
  color: #a6e3a1;
  font-style: italic;
  margin-bottom: 10px;
}
.gloss-label, .furigana-label, .notes-label, .source-label {
  font-size: 11px;
  color: #585b70;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-top: 14px;
}
.word-gloss {
  font-size: 14px;
  color: #89b4fa;
  margin-top: 4px;
  line-height: 1.9;
}
.source-name {
  font-size: 15px;
  color: #bac2de;
  min-height: 1.4em;
  margin-top: 4px;
  padding-bottom: 4px;
}
.notes {
  font-size: 15px;
  color: #bac2de;
  min-height: 1.4em;
  border-bottom: 1px dashed #45475a;
  margin-top: 4px;
  padding-bottom: 4px;
}
"""

CARD_MODEL = genanki.Model(
    MODEL_ID,
    "Japanese Video Cards",
    fields=[
        {"name": "Text"},           # raw Japanese (front)
        {"name": "Audio"},          # Japanese clip audio
        {"name": "Image"},
        {"name": "Translation"},    # English translation text
        {"name": "TTSAudio"},       # English TTS audio
        {"name": "Furigana"},       # HTML <ruby> markup
        {"name": "WordGloss"},
        {"name": "TimeCode"},
        {"name": "FontName"},
        {"name": "Source"},
        {"name": "Notes"},
    ],
    templates=[
        {
            "name": "Card 1",
            "qfmt": (
                "<div class=\"jp-text\" style=\"font-family: '{{FontName}}', sans-serif;\">{{Text}}</div>\n"
                "{{Audio}}\n"
                "<div class=\"timecode\">{{TimeCode}}</div>\n"
            ),
            "afmt": (
                "{{FrontSide}}\n"
                "<hr>\n"
                "<div class=\"frame-wrap\">{{Image}}</div>\n"
                "\n"
                "{{#Furigana}}\n"
                "<div class=\"jp-furigana\" style=\"font-family: '{{FontName}}', sans-serif;\">{{Furigana}}</div>\n"
                "{{/Furigana}}\n"
                "\n"
                "{{#Translation}}\n"
                "<div class=\"translation\">{{Translation}}</div>\n"
                "{{/Translation}}\n"
                "{{TTSAudio}}\n"   # plays automatically on card flip
                "\n"
                "{{#WordGloss}}\n"
                "<div class=\"gloss-label\">Word by word</div>\n"
                "<div class=\"word-gloss\">{{WordGloss}}</div>\n"
                "{{/WordGloss}}\n"
                "\n"
                "<div class=\"notes-label\">Notes</div>\n"
                "<div class=\"notes\">{{Notes}}</div>\n"
                "\n"
                "<div class=\"source-label\">Source</div>\n"
                "<div class=\"source-name\">{{Source}}</div>\n"
            ),
        }
    ],
    css=CARD_CSS,
)


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
    return {media.name for media in sorted(directory.iterdir())}


def process_file(
    media_path: Path,
    vtt_path: Path,
    deck: "genanki.Deck",
    all_media_files: list,
    tmpdir: str,
    frame_offset: float,
    do_sentence: bool,
    do_gloss: bool,
) -> None:
    original_name = media_path.name
    is_mp3 = media_path.suffix.lower() == ".mp3"

    safe_id    = uuid.uuid4().hex[:12]
    safe_ext   = media_path.suffix.lower()
    safe_media = Path(tmpdir) / f"media_{safe_id}{safe_ext}"
    safe_vtt   = Path(tmpdir) / f"vtt_{safe_id}.vtt"

    shutil.copy2(media_path, safe_media)
    shutil.copy2(vtt_path,   safe_vtt)

    print(f"\n  Parsing VTT: {vtt_path.name}")
    chunks = parse_vtt(str(safe_vtt))
    print(f"    Found {len(chunks)} chunks")

    if do_sentence or do_gloss:
        for i, chunk in enumerate(chunks):
            if do_sentence:
                chunk.translation = translate_sentence(chunk.text)
            if do_gloss:
                chunk.furigana, chunk.word_gloss = analyze_sentence(chunk.text)
            print(f"    [{i+1}/{len(chunks)}] {chunk.text[:45]}")
            if chunk.translation:
                print(f"      → {chunk.translation[:65]}")
            if chunk.word_gloss:
                print(f"      ≈ {chunk.word_gloss[:65]}")

    for chunk in chunks:
        idx       = chunk.index
        card_uuid = uuid.uuid4().hex

        audio_fname = f"clip_{idx:04d}_{card_uuid}.mp3"
        image_fname = f"frame_{idx:04d}_{card_uuid}.jpg"
        tts_fname   = f"tts_{idx:04d}_{card_uuid}.mp3"
        audio_path  = os.path.join(tmpdir, audio_fname)
        image_path  = os.path.join(tmpdir, image_fname)
        tts_path    = os.path.join(tmpdir, tts_fname)

        print(f"    [{idx+1}/{len(chunks)}] {chunk.start:.1f}s-{chunk.end:.1f}s  {chunk.text[:35]}")

        if is_mp3:
            audio_ok = extract_audio(str(safe_media), chunk.start, chunk.end, audio_path)
            image_ok = False
        else:
            audio_ok = extract_audio(str(safe_media), chunk.start, chunk.end, audio_path)
            frame_ts = chunk.start + min(frame_offset, (chunk.end - chunk.start) * 0.5)
            image_ok = extract_frame(str(safe_media), frame_ts, image_path)

        tts_ok = generate_tts(chunk.translation, tts_path) if chunk.translation else False

        if not audio_ok:              print(f"      Warning: audio extraction failed")
        if not image_ok and not is_mp3: print(f"      Warning: frame extraction failed")
        if not tts_ok:                print(f"      Warning: TTS generation failed (translation empty or error)")

        timecode = f"{int(chunk.start // 60):02d}:{chunk.start % 60:05.2f}"

        note = genanki.Note(
            model=CARD_MODEL,
            fields=[
                chunk.text,
                f"[sound:{audio_fname}]" if audio_ok else "",
                f'<img src="{image_fname}">'  if image_ok else "",
                chunk.translation,
                f"[sound:{tts_fname}]"   if tts_ok   else "",
                chunk.furigana,
                chunk.word_gloss,
                timecode,
                chunk.font,
                original_name,
                "",   # Notes — blank for manual entry
            ],
        )
        deck.add_note(note)

        if audio_ok: all_media_files.append(audio_path)
        if image_ok: all_media_files.append(image_path)
        if tts_ok:   all_media_files.append(tts_path)

    safe_media.unlink(missing_ok=True)
    safe_vtt.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Batch Anki deck generator from to_process/ directory")
    parser.add_argument("--frame-offset", type=float, default=1.0,
                        help="Seconds after chunk start to grab frame (default: 1.0)")
    parser.add_argument("--deck-name",   default="Japanese Video Deck")
    parser.add_argument("--tts-voice",   default="af_heart",
                        help="Kokoro voice (default: af_heart). Others: af_bella, am_adam, bf_emma, bm_george")
    args = parser.parse_args()

    check_ffmpeg()

    pairs = find_pairs(TO_PROCESS_DIR)
    if not pairs:
        print(f"No media+vtt pairs found in {TO_PROCESS_DIR}. Nothing to do.")
        sys.exit(0)

    print(f"Found {len(pairs)} pair(s) to process:")
    for media, vtt in pairs:
        print(f"  {media.name}  +  {vtt.name}")

    if not setup_sentence_translator():
        raise ValueError("Failed to set up sentence translator")
    if not setup_word_glosser():
        raise ValueError("Failed to set up word glosser")
    if not setup_tts():
        raise ValueError("Failed to set up TTS")

    # Bundle local font files into every package
    font_files = list(FONTS_DIR.glob("*.ttf")) if FONTS_DIR.exists() else []
    if not font_files:
        print("  Warning: no font files to bundle — cards will fall back to system fonts.")

    do_sentence = True
    do_gloss    = True
    deck        = genanki.Deck(DECK_ID, args.deck_name)
    media_files = [str(f) for f in font_files]   # fonts go in every package

    already_processed_names = already_processed(PROCESSED_DIR)

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, (media_path, vtt_path) in enumerate(pairs):
            if media_path.name in already_processed_names:
                print(f"{media_path.name} already processed. Skipping.")
                continue
            already_processed_names.add(media_path.name)

            output = f"output_{uuid.uuid4()}.apkg"
            print(f"\n{'='*60}")
            print(f"Processing ({i + 1}/{len(pairs)}): {media_path.name}")
            try:
                process_file(
                    media_path, vtt_path,
                    deck, media_files, tmpdir,
                    args.frame_offset, do_sentence, do_gloss,
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
