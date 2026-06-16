import html
import os
import subprocess
import sys

import requests
from pathlib import Path
import re
from kokoro_onnx import Kokoro
import soundfile as sf


# ── config ───────────────────────────────────────────────────────────────────

MEDIA = Path('/Users/bmanville3/Library/Application Support/Anki2/User 1/collection.media')
if not MEDIA.exists():
    raise ValueError(f"{MEDIA} does not exist")

OVERWRITE_EXISTS = True

KOKORO_MODEL_PATH  = os.path.expanduser("./models/kokoro/kokoro-v1.0.onnx")
KOKORO_VOICES_PATH = os.path.expanduser("./models/kokoro/voices-v1.0.bin")

# Path to the Gemma4 translations text file.
# Override via CLI: python script.py /path/to/translations.txt
TRANSLATIONS_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("gemma4_translations.txt")

ANKI_DECK = 'deck:"Core 2000"'

# ── TTS ───────────────────────────────────────────────────────────────────────

_kokoro = Kokoro(KOKORO_MODEL_PATH, KOKORO_VOICES_PATH)


def generate_tts(text: str, out_path: str | Path, voice: str = "af_heart") -> None:
    if isinstance(out_path, Path):
        out_path = str(out_path)
    if not text.strip():
        return
    samples, sample_rate = _kokoro.create(text, voice=voice, speed=1.0, lang="en-us")
    wav_path = out_path.replace(".mp3", "_tts_tmp.wav")
    sf.write(wav_path, samples, sample_rate)
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", wav_path, "-q:a", "4", out_path],
        capture_output=True,
    )
    Path(wav_path).unlink(missing_ok=True)
    if r.returncode != 0:
        raise ValueError(f"TTS ffmpeg error: {r.stderr.decode(errors='replace')[-200:]}")


# ── AnkiConnect ───────────────────────────────────────────────────────────────

def invoke(action, **params):
    return requests.post(
        "http://localhost:8765",
        json={"action": action, "version": 6, "params": params}
    ).json()


# ── translation file parser ───────────────────────────────────────────────────

def parse_translations(path: Path) -> dict[str, dict]:
    """
    Parse the Gemma4 translation file into a dict keyed by the
    normalized Sentence JP value.

    Each entry looks like:
        Vocab JP: それ
        Vocab EN: that, that one
        POS: Pronoun
        Sentence JP: <b>それ</b>はとってもいい話だ。
        Original EN: That's a really nice story.
        New Translation: As for that, it is a very good story.
        --------
    """
    text = path.read_text(encoding="utf-8")
    blocks = re.split(r"-{4,}", text)

    translations: dict[str, dict] = {}

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        entry: dict[str, str] = {}
        for line in block.splitlines():
            if ":" in line:
                key, _, value = line.partition(":")
                entry[key.strip()] = value.strip()

        sentence_jp = entry.get("Sentence JP", "").strip()
        if not sentence_jp:
            continue

        translations[sentence_jp] = entry

    return translations


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not TRANSLATIONS_FILE.exists():
        raise FileNotFoundError(f"Translations file not found: {TRANSLATIONS_FILE}")

    print(f"Loading translations from {TRANSLATIONS_FILE} …")
    translations = parse_translations(TRANSLATIONS_FILE)
    print(f"  {len(translations)} entries loaded.")

    notes = invoke("findNotes", query=ANKI_DECK)["result"]
    info  = invoke("notesInfo", notes=notes)["result"]
    print(f"Found {len(info)} notes in {ANKI_DECK!r}.\n")

    matched   = 0
    unmatched = 0

    for i, note in enumerate(info):
        print(f"Card {i + 1} / {len(info)}")
        note_id = note["noteId"]

        # Match on the full Japanese sentence (Expression field = "Sentence JP" in the file).
        sentence_jp = note["fields"]["Expression"]["value"].strip()

        translation_entry = translations.get(sentence_jp)
        if translation_entry is None:
            print("ERRORRRRRRRRRRRRRRRRRR")
            print(f"  [SKIP] No translation found for sentence_jp={sentence_jp!r}")
            unmatched += 1
            continue

        new_translation = translation_entry.get("New Translation", "").strip()
        if not new_translation:
            print("ERRORRRRRRRRRRRRRRRRRR")
            print(f"  [SKIP] Empty 'New Translation' for sentence_jp={sentence_jp!r}")
            unmatched += 1
            continue

        matched += 1

        # ── audio files ──────────────────────────────────────────────────────
        sentence_file = MEDIA / f"{note_id}_english_sentence_audio.mp3"

        # Sentence audio: generated from the New Translation text.
        if sentence_file.exists():
            if OVERWRITE_EXISTS:
                sentence_file.unlink()
                generate_tts(text=new_translation, out_path=sentence_file)
                print(f"  Regenerated sentence audio → {sentence_file.name}")
            else:
                print(f"  Skipping existing sentence audio {sentence_file.name}")
        else:
            generate_tts(text=new_translation, out_path=sentence_file)
            print(f"  Created sentence audio → {sentence_file.name}")

        # ── update note fields ────────────────────────────────────────────────
        invoke(
            "updateNoteFields",
            note={
                "id": note_id,
                "fields": {
                    "Gemma4": new_translation,
                    "English-Sentence-Audio": f"[sound:{sentence_file.name}]",
                }
            }
        )
        print(f"  Updated note {note_id}  (Gemma4 + audio)")

    print(f"\nDone. Matched: {matched}  |  Unmatched/skipped: {unmatched}")


if __name__ == "__main__":
    main()