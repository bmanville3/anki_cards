import html
import os
import subprocess

import requests
from pathlib import Path
import re
from kokoro_onnx import Kokoro
import soundfile as sf


def strip_html(text: str) -> str:
    text = re.sub(r"<.*?>", "", text)
    return html.unescape(text).strip()

MEDIA = Path('/Users/bmanville3/Library/Application Support/Anki2/User 1/collection.media')
if not MEDIA.exists():
    raise ValueError(f"{MEDIA} does not exist")

OVERWRITE_EXISTS = False
KOKORO_MODEL_PATH  = os.path.expanduser("~/models/kokoro/kokoro-v1.0.onnx")
KOKORO_VOICES_PATH = os.path.expanduser("~/models/kokoro/voices-v1.0.bin")

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


def invoke(action, **params):
    return requests.post(
        "http://localhost:8765",
        json={
            "action": action,
            "version": 6,
            "params": params
        }
    ).json()

notes = invoke(
    "findNotes",
    query='deck:"Core 2000"'
)["result"]

info = invoke(
    "notesInfo",
    notes=notes
)["result"]

for i, note in enumerate(info):
    print(f"Card {i + 1} / {len(info)}")
    note_id = note["noteId"]

    english_word = strip_html(note["fields"]["Vocabulary-English"]["value"]).strip()
    english_sentence = strip_html(note["fields"]["Sentence-English"]["value"]).strip()

    word_file = MEDIA / f"{note_id}_english_word_audio.mp3"
    sentence_file = MEDIA / f"{note_id}_english_sentence_audio.mp3"

    if word_file.exists():
        if OVERWRITE_EXISTS:
            word_file.unlink()
        else:
            print(f"Skipping existing {word_file.name}")
    else:
        generate_tts(text=english_word, out_path=word_file)
    if sentence_file.exists():
        if OVERWRITE_EXISTS:
            sentence_file.unlink()
        else:
            print(f"Skipping existing {sentence_file.name}")
    else:   
        generate_tts(text=english_sentence, out_path=sentence_file)


    invoke(
        "updateNoteFields",
        note={
            "id": note_id,
            "fields": {
                "English-Word-Audio": f"[sound:{word_file.name}]",
                "English-Sentence-Audio": f"[sound:{sentence_file.name}]",
            }
        }
    )
    print(f"Add audio for {note_id=}")
