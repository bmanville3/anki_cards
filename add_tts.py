import html
import os
import random
import subprocess

import requests
from pathlib import Path
import re
from kokoro_onnx import Kokoro
import soundfile as sf

OVERWRITE_EXISTS = False
VERBOSE = False

def strip_html(text: str) -> str:
    text = re.sub(r"<.*?>", "", text)
    return html.unescape(text).strip()

MEDIA = Path('/Users/bmanville3/Library/Application Support/Anki2/User 1/collection.media')
if not MEDIA.exists():
    raise ValueError(f"{MEDIA} does not exist")

MASTER = "Master Card"
KOKORO_MODEL_PATH  = os.path.expanduser("./models/kokoro/kokoro-v1.0.onnx")
KOKORO_VOICES_PATH = os.path.expanduser("./models/kokoro/voices-v1.0.bin")

_kokoro = Kokoro(KOKORO_MODEL_PATH, KOKORO_VOICES_PATH)

VOICES = ["af_heart"]

print(f"Using {len(VOICES)} voices: {VOICES}")


def generate_tts(text: str, out_path: str | Path, voice: str) -> None:
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
    query=f'note:"{MASTER}"'
)["result"]

print(f"Found {len(notes)} notes across all '{MASTER}' note types")

info = invoke(
    "notesInfo",
    notes=notes
)["result"]

for i, note in enumerate(info):
    if VERBOSE:
        print(f"Card {i + 1} / {len(info)} (modelName={note['modelName']})")
    note_id = note["noteId"]
    fields = note["fields"]
    tags = note['tags']
    tag_set = set(tags)

    if "English" not in fields:
        print(f"  Skipping {note_id=}: missing expected fields for modelName={note['modelName']}")
        continue
    
    english = strip_html(fields["English"]["value"]).strip()
    if not english:
        print(f"  Skipping {note_id=}: empty english field")
        continue
    

    if "English Audio" in fields \
          and strip_html(fields["English"]["value"]).strip() \
          and not OVERWRITE_EXISTS:
        if VERBOSE:
            print(f"  Skipping {note_id}: non-empty english audio field")
        continue

    word_file = MEDIA / f"{note_id}_english_audio.mp3"
    voice = random.choice(VOICES)

    if word_file.exists():
        if OVERWRITE_EXISTS:
            print(f"  Overriding existing {word_file.name}")
            word_file.unlink()
            generate_tts(text=english, out_path=word_file, voice=voice)
        else:
            print(f"  Skipping existing {word_file.name}")
            print(f"  This was unexpected as 'English Audio' for {note_id=} was empty but the file exists")
            continue
    else:
        generate_tts(text=english, out_path=word_file, voice=voice)

    fields_to_update = {
        "English Audio": f"[sound:{word_file.name}]",
        "English Audio Model": f"kokoro_{voice}",
    }

    result = invoke(
        "updateNoteFields",
        note={
            "id": note_id,
            "fields": fields_to_update
        }
    )
    if not result['error']:
        print(f"  Added audio for {note_id=}")
    else:
        print(f"  Error adding audio for {note_id=}:\n{result['error']}")
