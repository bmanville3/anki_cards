from common.utils import assert_latin_extended_only
from prompting.prompter import PromptRequest
import html
import re
import subprocess
from pathlib import Path

import requests

from src.tts.kokoro import KokoroBackend
from src.prompting.prompter import prompt_batch, PromptRequest


_JLAB_SUMMARY_SYSTEM = (
    "You are a Japanese-to-English translator specializing in concise, natural translations for Anki JLAB flashcards. "
    "The input is a Japanese sentence from the Japanese Like a Breeze (JLAB) Anki deck, along with any available "
    "explanation, context, or remarks provided by the deck author. "
    "Your job is to produce a single clean English translation suitable for the audio/summary field of the card.\n\n"
    "Here are example JLAB sentences and the style of translation to produce:\n"
    "- '今日の晩ご飯何がいいですか' -> 'As for today's dinner, what is good?'\n"
    "- '悪い話じゃないな' -> 'isn't a bad conversation'\n"
    "- '大事な話があります' -> 'important conversation is there'\n"
    "- '何しに行くの' -> 'what are you going to do'\n"
    "- 'あの男は人間じゃない' -> 'As for that guy over there, is not human.'\n"
    "- '楽しい話ね' -> 'fun conversation, isn't it!'\n"
    "Rules:\n"
    "- Output ONLY the English translation — no notes, no explanations, no punctuation beyond what belongs in a sentence.\n"
    "- Prioritize the deck author's provided translation or explanation when present; use it as the ground truth.\n"
    "- If a translation is embedded in the remarks (e.g. 'It's already evening and the people of her house might be worrying'), "
    "clean and naturalize it rather than retranslating from scratch.\n"
    "- Incorporate bracketed clarifications from the remarks (e.g. '[=her family]') naturally into the translation — "
    "do not include the brackets themselves.\n"
    "- Do not add anything not implied by the Japanese or the provided remarks.\n"
    "- Any characters not found in standard English are NOT allowed unless the word is a loanword such as 'Café'.\n"
    "- If no remarks or translation are available, produce a clean, natural translation of the Japanese directly in the style of the examples provided.\n"
    "- If a complete translation is already there, do not provide your own translation. Use the translation provided. If it is not there, then you must translate in the style of JLAB cards.\n"
    "- If a video frame is attached, it is provided for context only. Do not add visual descriptions or weight it heavily over the text."
)

def get_image_base64(note: dict) -> tuple[str | None, str]:
    """Extract image from note's Image field, return (base64_data, mime_type) or (None, '')."""
    image_html = note["fields"].get("Image", {}).get("value", "")
    if not image_html:
        print("No image html for card")
        return None, ""
    
    match = re.search(r'src="([^"]+)"', image_html)
    if not match:
        print("No image regex for card")
        return None, ""
    
    filename = match.group(1)
    # AnkiConnect serves media files directly
    result = invoke("retrieveMediaFile", filename=filename)
    b64 = result.get("result")
    if not b64:
        print("No image media for card")
        return None, ""

    ext = Path(filename).suffix.lower()
    mime = {
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png":  "image/png",
        ".gif":  "image/gif",
        ".webp": "image/webp",
    }.get(ext, "image/jpeg")

    return b64, mime

def build_jlab_summary_prompt(japanese: str, remarks: str = "", references: str = "") -> str:
    parts = [f"Japanese sentence: {japanese}"]
    if remarks.strip():
        parts.append(f"Deck remarks / translation notes: {remarks.strip()}")
    if references.strip():
        parts.append(f"References (for context only): {references.strip()}")
    parts.append("Provide the English translation:")
    return "\n\n".join(parts)

def make_jlab_summary_request(
    japanese: str,
    remarks: str = "",
    references: str = "",
    base64_encoded_image: str | None = None,
    image_mime: str = "image/jpeg",
) -> PromptRequest:
    return PromptRequest(
        system_prompt=_JLAB_SUMMARY_SYSTEM,
        user_prompt=build_jlab_summary_prompt(japanese, remarks, references),
        validator=assert_latin_extended_only,
        base64_encoded_image=base64_encoded_image,
        image_mime=image_mime,
    )

MEDIA = Path('/Users/bmanville3/Library/Application Support/Anki2/User 1/collection.media')
OVERWRITE_EXISTS = False

_tts = KokoroBackend(lang="en")
_tts.setup()


def strip_html(text: str) -> str:
    text = re.sub(r"<.*?>", "", text)
    return html.unescape(text).strip()


def invoke(action, **params):
    return requests.post(
        "http://localhost:8765",
        json={"action": action, "version": 6, "params": params},
    ).json()


def wav_to_mp3(wav_path: str, mp3_path: str) -> None:
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", wav_path, "-q:a", "4", mp3_path],
        capture_output=True,
    )
    Path(wav_path).unlink(missing_ok=True)
    if r.returncode != 0:
        raise ValueError(f"ffmpeg error: {r.stderr.decode(errors='replace')[-200:]}")


notes = invoke("findNotes", query='deck:"jlab"')["result"]
info  = invoke("notesInfo", notes=notes)["result"]

# ── 1. Build LLM requests in bulk ──────────────────────────────────────────
llm_requests: list[PromptRequest] = []
llm_requests: list[PromptRequest] = []
for note in info:
    fields = note["fields"]
    japanese   = strip_html(fields.get("Jlab-ListeningFront", {}).get("value", ""))
    remarks    = strip_html(fields.get("Jlab-Remarks",        {}).get("value", ""))
    remarks   += " " + strip_html(fields.get("RemarksBack",   {}).get("value", ""))
    references = strip_html(fields.get("References",          {}).get("value", ""))

    image_b64, image_mime = get_image_base64(note)

    llm_requests.append(make_jlab_summary_request(
        japanese,
        remarks.strip(),
        references,
        base64_encoded_image=image_b64,
        image_mime=image_mime,
    ))

print(f"Sending {len(llm_requests)} cards to LLM...")
translations = prompt_batch(llm_requests)

# ── 2. Generate TTS + update cards ─────────────────────────────────────────
for i, (note, translation) in enumerate(zip(info, translations)):
    print(f"Card {i + 1} / {len(info)}")
    if not translation:
        print("  LLM returned empty — skipping.")
        continue

    note_id   = note["noteId"]
    audio_mp3 = MEDIA / f"{note_id}_jlab_summary_audio.mp3"
    audio_wav = MEDIA / f"{note_id}_jlab_summary_audio_tmp.wav"

    if audio_mp3.exists() and not OVERWRITE_EXISTS:
        print(f"  Skipping existing {audio_mp3.name}")
    else:
        if audio_mp3.exists():
            audio_mp3.unlink()
        ok = _tts.generate(translation, str(audio_wav), voice="af_heart")
        if ok:
            wav_to_mp3(str(audio_wav), str(audio_mp3))
        else:
            print(f"  TTS failed for note {note_id}")
            continue

    invoke(
        "updateNoteFields",
        note={
            "id": note_id,
            "fields": {"Summary-Audio": f"[sound:{audio_mp3.name}]"},
        },
    )
    print(f"  Updated note {note_id} — \"{translation}\"")
