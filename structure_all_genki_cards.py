"""
anki_transform.py — three-mode Anki card transformation pipeline

Modes:
  export    Pull cards from AnkiConnect → write raw_cards.csv
  transform Read raw_cards.csv → run LLM → write transformed_cards.csv
  import    Read transformed_cards.csv → push to AnkiConnect (in-place update)

Typical remote workflow:
  [local]  python anki_transform.py export   --deck "Genki I" --out raw_cards.csv
           # upload raw_cards.csv + media folder to Google Drive
  [server] python anki_transform.py transform --in raw_cards.csv --out transformed_cards.csv --media ./media
           # download transformed_cards.csv from Google Drive
  [local]  python anki_transform.py import   --in transformed_cards.csv
"""

import argparse
import csv
import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import Any, Callable, Self, cast

import requests
from attrs import define
import attrs

from src.common.utils import load_image_b64
from src.prompting.prompter import PromptRequest, prompt_batch


# ─────────────────────────────────────────────────────────────────────────────
# Config / defaults  (override via CLI args)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_DECK       = "Genki I"
DEFAULT_MEDIA_PATH = Path('/Users/bmanville3/Library/Application Support/Anki2/User 1/collection.media')
DEFAULT_RAW_CSV    = Path("raw_cards.csv")
DEFAULT_OUT_CSV    = Path("transformed_cards.csv")
MASTER_MODEL_NAME  = "Master Genki Card"

# CSV column names for the raw export
RAW_CSV_FIELDS = ["noteId", "cardType", "fields_json"]

# CSV column names for the transformed output
TRANSFORMED_CSV_FIELDS = [
    "noteId", "cardType",
    "japanese", "japanese_audio", "furigana", "reading",
    "english", "english_audio",
    "screenshots", "screenshot_text",
    "explanations", "additional_notes",
    "tags",
    "previous_version",
]

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# AnkiConnect helper
# ─────────────────────────────────────────────────────────────────────────────

def invoke(action, **params):
    return requests.post(
        "http://localhost:8765",
        json={"action": action, "version": 6, "params": params},
    ).json()


# ─────────────────────────────────────────────────────────────────────────────
# Field / Card / Deck
# ─────────────────────────────────────────────────────────────────────────────

@define
class Field:
    name: str
    value: str
    _internal_mappings: dict[str, str] = attrs.field(factory=dict)

    def pretty_string(self) -> str:
        return f"{self.name}: {self.value}"

    def split_self(self) -> list[Self]:
        value = self.value
        sound_pattern = re.compile(r'\[sound:[^\]]+\]')
        img_pattern   = re.compile(r'<img[^>]+>')
        sounds = sound_pattern.findall(value)
        images = img_pattern.findall(value)
        stripped = re.sub(r'(\[sound:[^\]]+\]|<img[^>]+>|<br\s*/?>|\s)', '', value)
        if not stripped or (not sounds and not images):
            return [self]
        result    = []
        remaining = value
        for i, sound in enumerate(sounds):
            remaining = remaining.replace(sound, '', 1)
            result.append(Field(
                name  = f"{self.name}-audio" if len(sounds) == 1 else f"{self.name}-audio-{i+1}",
                value = sound,
            ))
        for i, img in enumerate(images):
            remaining = remaining.replace(img, '', 1)
            result.append(Field(
                name  = f"{self.name}-image" if len(images) == 1 else f"{self.name}-image-{i+1}",
                value = img,
            ))
        remaining = re.sub(r'(<br\s*/?>|\s)+', ' ', remaining).strip()
        if remaining:
            result.insert(0, Field(name=self.name, value=remaining))
        return result

    def map_ugly_ids_to_pretty(self) -> None:
        self._internal_mappings.clear()
        image_count = 1
        audio_count = 1

        def replace_image(match: re.Match[str]) -> str:
            nonlocal image_count
            placeholder = f"{{{{IMAGE:{image_count}}}}}"
            self._internal_mappings[placeholder] = match.group(0)
            image_count += 1
            return placeholder

        def replace_audio(match: re.Match[str]) -> str:
            nonlocal audio_count
            placeholder = f"{{{{AUDIO:{audio_count}}}}}"
            self._internal_mappings[placeholder] = match.group(0)
            audio_count += 1
            return placeholder

        self.value = re.sub(r"<img[^>]+>", replace_image, self.value)
        self.value = re.sub(r"\[sound:[^\]]+\]", replace_audio, self.value)

    def map_pretty_ids_to_ugly(self) -> None:
        for placeholder, original in self._internal_mappings.items():
            self.value = self.value.replace(placeholder, original)

    def produce_base_64_images(self, media_path: Path) -> list[tuple[str, str]]:
        unsubbed_value = self.value
        for placeholder, original in self._internal_mappings.items():
            unsubbed_value = unsubbed_value.replace(placeholder, original)
        img_pattern = re.compile(r'<img\s+src="([^"]+)"[^>]*>')
        results: list[tuple[str, str]] = []
        for filename in img_pattern.findall(unsubbed_value):
            path = media_path / filename
            try:
                loaded = load_image_b64(path)
                results.append(loaded)
            except Exception as e:
                logger.error("Problem loading media at '%s': %s", path, e)
        return results


def _field_names(fields: list[Field]) -> set[str]:
    return {f.name for f in fields}


def _assert_fields(card_type: str, fields: list[Field], expected: set[str]) -> None:
    actual = _field_names(fields)
    if expected != actual:
        missing = expected - actual
        extra   = actual   - expected
        parts   = []
        if missing: parts.append(f"missing={missing}")
        if extra:   parts.append(f"extra={extra}")
        raise ValueError(f"[{card_type}] Field mismatch — {', '.join(parts)}")


def _base_verify_split(card: "Card", expected_fields: set[str], starts_with_fields: set[str]) -> None:
    names = _field_names(card.fields)
    if not all(n in expected_fields or any(n.startswith(s) for s in starts_with_fields) for n in names):
        raise ValueError(f"[{card.cardType} split] Unexpected field names: {names}")
    for field in expected_fields:
        if field not in names:
            raise ValueError(f"[{card.cardType} split] Missing {field=} from {names}")


BASIC_FIELDS            = {"Front", "Back"}
BASIC_FIELDS_PLUS_DASH  = [p + "-" for p in BASIC_FIELDS]
PRACTICE_FIELDS = {
    "Prompt", "Prompt Audio", "Prompt Picture", "Prompt Additional Instructions",
    "Prompt (English Tanslation)", "Answer", "Answer Audio",
    "Answer (English Translation)", "Additional Back Explanation", "Answer Picture",
}
PRACTICE_FIELDS_PLUS_DASH = [p + "-" for p in PRACTICE_FIELDS]
VOCAB_FIELDS            = {"Japanese", "Japanese Audio", "Textbook Definition", "Picture (example)", "Additional Notes"}
VOCAB_FIELDS_PLUS_DASH  = [f + "-" for f in VOCAB_FIELDS]

def verify_basic(card: "Card") -> None:                  _assert_fields(card.cardType, card.fields, BASIC_FIELDS)
def verify_basic_split(card: "Card") -> None:            _base_verify_split(card, BASIC_FIELDS, BASIC_FIELDS_PLUS_DASH)
def verify_genki_practice_card(card: "Card") -> None:    _assert_fields(card.cardType, card.fields, PRACTICE_FIELDS)
def verify_genki_practice_card_split(card: "Card") -> None: _base_verify_split(card, PRACTICE_FIELDS, PRACTICE_FIELDS_PLUS_DASH)
def verify_genki_vocab_card(card: "Card") -> None:       _assert_fields(card.cardType, card.fields, VOCAB_FIELDS)
def verify_genki_vocab_card_split(card: "Card") -> None: _base_verify_split(card, VOCAB_FIELDS, VOCAB_FIELDS_PLUS_DASH)

card_type_to_field_verification: dict[str, Callable[["Card"], None]] = {
    "Basic":                     verify_basic,
    "Basic (split)":             verify_basic_split,
    "Genki Practice Card":       verify_genki_practice_card,
    "Genki Practice Card Split": verify_genki_practice_card_split,
    "Genki Vocab Card":          verify_genki_vocab_card,
    "Genki Vocab Card Split":    verify_genki_vocab_card_split,
}


@define
class Card:
    noteId:   str
    cardType: str
    fields:   list[Field]

    def __attrs_post_init__(self):
        verifier = card_type_to_field_verification.get(self.cardType)
        if verifier is None:
            raise ValueError(f"No verifier registered for card type '{self.cardType}'")
        verifier(self)

    @classmethod
    def from_notes_info(cls, data: Any) -> list[Self]:
        output = []
        if not isinstance(data, list):
            return output
        for note in data:
            if not isinstance(note, dict):
                continue
            noteId = note.get('noteId')
            if noteId is None:
                continue
            cardType        = note.get('modelName', '')
            fields_unloaded = note.get('fields')
            if not fields_unloaded or not isinstance(fields_unloaded, dict):
                output.append(cls(noteId, cardType, []))
                continue
            fields = [Field(k, str(v.get('value', ''))) for k, v in cast(dict[str, dict], fields_unloaded).items()]
            output.append(cls(noteId, cardType, fields))
        return output

    def to_csv_row(self) -> dict:
        """Serialise to a row for raw_cards.csv."""
        return {
            "noteId":      self.noteId,
            "cardType":    self.cardType,
            "fields_json": json.dumps({f.name: f.value for f in self.fields}, ensure_ascii=False),
        }

    @classmethod
    def from_csv_row(cls, row: dict) -> Self:
        """Deserialise from a raw_cards.csv row."""
        fields_dict = json.loads(row["fields_json"])
        fields = [Field(k, v) for k, v in fields_dict.items()]
        return cls(noteId=row["noteId"], cardType=row["cardType"], fields=fields)

    def pretty_string(self) -> str:
        joined = '\n\t'.join(f.pretty_string() for f in self.fields)
        return f"Card {self.noteId}\n- Fields:\n\t{joined}"

    def get_field(self, name: str) -> str:
        for f in self.fields:
            if f.name == name:
                return f.value
        return ""

    def get_fields_like(self, prefix: str) -> list[Field]:
        return [f for f in self.fields if f.name.startswith(prefix)]

    def split_fields(self) -> "Card":
        split_type_map = {
            "Basic":               "Basic (split)",
            "Genki Practice Card": "Genki Practice Card Split",
            "Genki Vocab Card":    "Genki Vocab Card Split",
        }
        new_type = split_type_map.get(self.cardType, self.cardType)
        new_fields: list[Field] = []
        for f in self.fields:
            new_fields.extend(f.split_self())
        return Card(noteId=self.noteId, cardType=new_type, fields=new_fields)

    def map_ugly_ids_to_pretty(self) -> None:
        for f in self.fields:
            f.map_ugly_ids_to_pretty()

    def map_pretty_ids_to_ugly(self) -> None:
        for f in self.fields:
            f.map_pretty_ids_to_ugly()

    def get_all_images(self, media_path: Path) -> list[tuple[str, str]]:
        output = []
        for field in self.fields:
            output.extend(field.produce_base_64_images(media_path))
        return output


@define
class Deck:
    name:  str
    cards: list[Card]

    @classmethod
    def from_anki(cls, name: str) -> Self:
        notes = invoke("findNotes", query=f'deck:"{name}"')["result"]
        if not notes:
            return cls(name, [])
        info = invoke("notesInfo", notes=notes)["result"]
        return cls(name, Card.from_notes_info(info))


# ─────────────────────────────────────────────────────────────────────────────
# Master card
# ─────────────────────────────────────────────────────────────────────────────

@define
class MasterGenkiCard:
    source_note:      Card
    japanese:         str       = ""
    japanese_audio:   list[str] = None
    furigana:         str       = ""
    reading:          str       = ""
    english:          str       = ""
    english_audio:    str       = ""
    screenshots:      list[str] = None
    explanations:     str       = ""
    additional_notes: str       = ""
    screenshot_text:  str       = ""
    tags:             list[str] = None

    def __attrs_post_init__(self):
        if self.screenshots    is None: self.screenshots    = []
        if self.tags           is None: self.tags           = []
        if self.japanese_audio is None: self.japanese_audio = []

    def to_anki_fields(self) -> dict[str, str]:
        return {
            "Japanese":         self.japanese,
            "Japanese Audio":   " ".join(self.japanese_audio),
            "Furigana":         self.furigana,
            "Reading":          self.reading,
            "English":          self.english,
            "English Audio":    self.english_audio,
            "Screenshots":      " ".join(self.screenshots),
            "Screenshot Text":  self.screenshot_text,
            "Explanations":     self.explanations,
            "Additional Notes": self.additional_notes,
            "Previous Version": self.source_note.pretty_string(),
        }

    def to_csv_row(self) -> dict:
        """Serialise to a row for transformed_cards.csv."""
        return {
            "noteId":          self.source_note.noteId,
            "cardType":        self.source_note.cardType,
            "japanese":        self.japanese,
            "japanese_audio":  json.dumps(self.japanese_audio,  ensure_ascii=False),
            "furigana":        self.furigana,
            "reading":         self.reading,
            "english":         self.english,
            "english_audio":   self.english_audio,
            "screenshots":     json.dumps(self.screenshots,     ensure_ascii=False),
            "screenshot_text": self.screenshot_text,
            "explanations":    self.explanations,
            "additional_notes":self.additional_notes,
            "tags":            json.dumps(self.tags,            ensure_ascii=False),
            "previous_version":self.source_note.pretty_string(),
        }

    @classmethod
    def from_csv_row(cls, row: dict, source_card: Card) -> Self:
        """Deserialise from a transformed_cards.csv row."""
        return cls(
            source_note      = source_card,
            japanese         = row.get("japanese",         ""),
            japanese_audio   = json.loads(row.get("japanese_audio",  "[]")),
            furigana         = row.get("furigana",         ""),
            reading          = row.get("reading",          ""),
            english          = row.get("english",          ""),
            english_audio    = row.get("english_audio",    ""),
            screenshots      = json.loads(row.get("screenshots",     "[]")),
            screenshot_text  = row.get("screenshot_text",  ""),
            explanations     = row.get("explanations",     ""),
            additional_notes = row.get("additional_notes", ""),
            tags             = json.loads(row.get("tags",             "[]")),
        )


# ─────────────────────────────────────────────────────────────────────────────
# CSV I/O
# ─────────────────────────────────────────────────────────────────────────────

def write_raw_csv(cards: list[Card], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=RAW_CSV_FIELDS)
        writer.writeheader()
        for card in cards:
            writer.writerow(card.to_csv_row())
    print(f"Wrote {len(cards)} raw cards → {path}")


def read_raw_csv(path: Path) -> list[Card]:
    cards = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                cards.append(Card.from_csv_row(row))
            except Exception as e:
                logger.warning("Skipping malformed row (noteId=%s): %s", row.get("noteId"), e)
    print(f"Read {len(cards)} raw cards ← {path}")
    return cards


def write_transformed_csv(master_cards: list[MasterGenkiCard], path: Path, append: bool = False) -> None:
    write_header = not append or not path.exists() or path.stat().st_size == 0
    mode = "a" if append else "w"
    with open(path, mode, newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=TRANSFORMED_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        for mc in master_cards:
            writer.writerow(mc.to_csv_row())
    action = "Appended" if append else "Wrote"
    logger.info("%s %d transformed cards → %s", action, len(master_cards), path)


def read_transformed_csv(path: Path) -> list[MasterGenkiCard]:
    """
    Reconstruct MasterGenkiCards from CSV.
    The source_card is a minimal stub — enough to populate previous_version
    and carry the noteId/cardType through to AnkiConnect.
    """
    master_cards = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            note_id   = row.get("noteId", "")
            card_type = row.get("cardType", "")
            # Stub card — fields not needed for import, only noteId + cardType
            stub_fields = [Field("_stub", row.get("previous_version", ""))]
            try:
                # Bypass verification for the stub by building directly
                stub_card = object.__new__(Card)
                object.__setattr__(stub_card, "noteId",   note_id)
                object.__setattr__(stub_card, "cardType", card_type)
                object.__setattr__(stub_card, "fields",   stub_fields)
                master_cards.append(MasterGenkiCard.from_csv_row(row, stub_card))
            except Exception as e:
                logger.warning("Skipping malformed row (noteId=%s): %s", note_id, e)
    print(f"Read {len(master_cards)} transformed cards ← {path}")
    return master_cards


# ─────────────────────────────────────────────────────────────────────────────
# LLM prompts
# ─────────────────────────────────────────────────────────────────────────────

BASE_SYSTEM_PROMPT = """\
You are an expert Japanese language tutor and Anki card formatter.

Your job is to convert a legacy Anki flashcard into a standardised "Master Genki Card".
The Master Genki Card has the following fields:

  japanese          - The Japanese text: a word, phrase, or full sentence.
  furigana          - The same text with furigana inserted above every kanji using
                      the HTML ruby format: <ruby>漢字<rt>かんじ</rt></ruby>
                      Furigana should be added for both full sentences and vocab cards.
  reading           - The full kana reading (hiragana or katakana) of the item,
                      with no kanji. Only necessary for vocabulary cards.
  english           - The English meaning or translation.
  explanations      - Grammar notes, part of speech, conjugation notes, usage context,
                      or any other information that would help a Genki learner.
                      Leave blank only if there is truly nothing to add.
  additional_notes  - Any extra details not captured above. Leave blank if not applicable.
  screenshot_text   - If a screenshot / image was provided, transcribe every piece of
                      Japanese and English text visible in it, verbatim, so it is
                      searchable. Use "---" to separate multiple screenshots.
                      Leave blank if no image was supplied or the image contains
                      no helpful text (such as a pure example picture).

Rules you must always follow:
1. Output ONLY a single JSON object with exactly these keys:
   japanese, furigana, reading, english, explanations, additional_notes, screenshot_text
2. Do not add any extra keys. Do not wrap the JSON in markdown fences.
3. All values are strings. Never use null; use "" for empty fields.
4. Do not invent meanings — base english and explanations on the source card content.
5. If the source card already contains some of this information, carry it over faithfully.
6. Use standard modern Japanese orthography for furigana and reading.
7. All audio fields are Japanese audio. You may safely ignore them.
"""

EXPECTED_OUTPUT_FIELDS = [
    "japanese", "furigana", "reading", "english",
    "explanations", "additional_notes", "screenshot_text",
]


def build_basic_prompt(card: Card) -> str:
    return f"""\
SOURCE CARD TYPE: Basic

This card was created early in my study. The Front field contains whatever was
being studied — it might be a Japanese word/phrase, a production prompt in English,
a grammar pattern, or a picture prompt. The Back field is the expected answer.
Audio and image sub-fields (e.g. Front-audio, Back-image) have been put into their
own fields. The question/answer may only be in an image so examine every image.

Your task:
- Determine which side holds the Japanese content and which holds the English answer.
- If the Front is Japanese: map it to `japanese`; map Back to `english`.
- If the Front is an English prompt and the Back is Japanese: set `japanese` from Back,
  `english` from Front (reword as a definition rather than a question).
- If neither side is clearly Japanese, put the Front in `japanese` and Back in `english`
  and leave a note in `explanations`.
- Generate furigana, reading, explanations, and additional_notes from your knowledge.

--- CARD CONTENT ---
{card.pretty_string()}
"""


def build_practice_prompt(card: Card) -> str:
    return f"""\
SOURCE CARD TYPE: Genki Practice Card

Practice cards present a stimulus on the front and an expected response on the back.
The Prompt fields form the question/stimulus; the Answer fields form the response.
Audio and image sub-fields may contain useful information so please examine them.

Field guide:
  Prompt                         - the core question or sentence to respond to
  Prompt Additional Instructions - extra instructions or hints shown on front
  Prompt (English Translation)   - English gloss of the prompt (if relevant)
  Prompt Picture                 - image to guide the prompt
  Prompt Audio                   - front prompt audio — can be ignored
  Answer                         - the correct answer (Japanese or English)
  Answer Audio                   - back answer audio — can be ignored
  Answer (English Translation)   - English meaning of the answer (if relevant)
  Additional Back Explanation    - grammar or usage note shown on the back
  Answer Picture                 - picture(s) to assist the answer; often contains the answer itself

Mapping guidance:
- `japanese`  ← Answer
- `english`   ← Answer (English Translation), informed by Prompt (English Translation)
- Synthesise `explanations` from Additional Back Explanation and the prompt context.
- Include the prompt in `explanations` if it adds useful context.
- Generate furigana, reading, and additional_notes from your knowledge.

--- CARD CONTENT ---
{card.pretty_string()}
"""


def build_vocab_prompt(card: Card) -> str:
    return f"""\
SOURCE CARD TYPE: Genki Vocab Card

Vocab cards study a single Japanese word or short phrase.
  Front: Japanese word or sentence + Japanese audio
  Back:  Textbook Definition: may be typed text, a screenshot, or both.
         Picture (example): may be a helpful picture or further textbook material.
         Additional Notes: often blank; sometimes contains the kana reading.

Mapping guidance:
- `japanese`         ← Japanese field.
- `english`          ← extracted from Textbook Definition text and/or screenshot(s).
- `additional_notes` ← Additional Notes field content.
- If Additional Notes contains a kana reading, use it for `reading` exactly.
- `screenshot_text`  ← transcribe every Japanese and English word visible in the image(s).
- `explanations`     ← grammar notes from the Textbook Definition; simple definition goes in `english`.
- `furigana` and `reading` you will likely need to generate yourself.

--- CARD CONTENT ---
{card.pretty_string()}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Transform pipeline helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_llm_json(raw: str) -> dict:
    clean  = re.sub(r'^```[a-zA-Z]*\n?', '', raw.strip())
    clean  = re.sub(r'\n?```$', '', clean)
    output = json.loads(clean)
    if not isinstance(output, dict):
        raise ValueError(f"Expected output to be a dict. Got: {output}")
    missing = [f for f in EXPECTED_OUTPUT_FIELDS if f not in output]
    extra   = [f for f in output if f not in EXPECTED_OUTPUT_FIELDS]
    if missing or extra:
        raise ValueError(
            f"Expected fields: {EXPECTED_OUTPUT_FIELDS}. "
            f"Missing: {missing}. Extra: {extra}."
        )
    return output


def _collect_audio(card: Card) -> list[str]:
    audios = []
    for f in card.fields:
        audios += re.findall(r'\[sound:[^\]]+\]', f.value)
    return audios


def _collect_images(card: Card) -> list[str]:
    imgs = []
    for f in card.fields:
        imgs += re.findall(r'<img[^>]+>', f.value)
    return imgs


def build_prompt_request_for_card(card: Card, media_path: Path) -> PromptRequest:
    split_card = card.split_fields()
    split_card.map_ugly_ids_to_pretty()
    ct = split_card.cardType
    if ct in ("Basic", "Basic (split)"):
        user_prompt = build_basic_prompt(split_card)
    elif ct in ("Genki Practice Card", "Genki Practice Card Split"):
        user_prompt = build_practice_prompt(split_card)
    elif ct in ("Genki Vocab Card", "Genki Vocab Card Split"):
        user_prompt = build_vocab_prompt(split_card)
    else:
        raise ValueError(f"Unknown card type: {ct}")

    return PromptRequest(
        system_prompt  = BASE_SYSTEM_PROMPT,
        user_prompt    = user_prompt,
        base_64_images = card.get_all_images(media_path),
        validator      = _parse_llm_json,
    )


def assemble_master_card(card: Card, llm_raw: str) -> MasterGenkiCard:
    card.map_pretty_ids_to_ugly()
    data = _parse_llm_json(llm_raw)
    return MasterGenkiCard(
        source_note      = card,
        japanese         = data.get("japanese",         ""),
        furigana         = data.get("furigana",         ""),
        reading          = data.get("reading",          ""),
        english          = data.get("english",          ""),
        english_audio    = "",
        japanese_audio   = _collect_audio(card),
        screenshots      = _collect_images(card),
        screenshot_text  = data.get("screenshot_text", ""),
        explanations     = data.get("explanations",    ""),
        additional_notes = data.get("additional_notes",""),
    )


# ─────────────────────────────────────────────────────────────────────────────
# AnkiConnect write helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_model_fields(field_names: list[str]) -> None:
    result  = invoke("modelFieldNames", modelName=MASTER_MODEL_NAME)
    current = set(result.get("result") or [])
    for field in field_names:
        if field not in current:
            r = invoke("modelFieldAdd", modelName=MASTER_MODEL_NAME, fieldName=field, index=len(current))
            if r.get("error"):
                logger.warning("Could not add field '%s' to model: %s", field, r["error"])
            else:
                current.add(field)
                print(f"  + Added field '{field}' to {MASTER_MODEL_NAME}")


def _update_note_in_place(mc: MasterGenkiCard) -> None:
    note_id       = int(mc.source_note.noteId)
    current_model = mc.source_note.cardType

    if current_model != MASTER_MODEL_NAME:
        old_fields_result = invoke("modelFieldNames", modelName=current_model)
        old_fields        = old_fields_result.get("result") or []
        result = invoke(
            "changeNotesType",
            noteIds=[note_id],
            oldModelName=None,
            newModelName=MASTER_MODEL_NAME,
            fieldMapping={old: "" for old in old_fields},
        )
        if result.get("error"):
            print(f"  ✗ changeNotesType failed for {note_id}: {result['error']}")
            return

    result = invoke("updateNoteFields", note={"id": note_id, "fields": mc.to_anki_fields()})
    if result.get("error"):
        print(f"  ✗ updateNoteFields failed for {note_id}: {result['error']}")
    else:
        print(f"  ✓ Updated note {note_id} in place")


def _copy_referenced_media(cards: list[Card], media_src: Path, media_dst: Path) -> None:
    img_re   = re.compile(r'<img\s+[^>]*src="([^"]+)"')
    sound_re = re.compile(r'\[sound:([^\]]+)\]')

    referenced: set[str] = set()
    for card in cards:
        for field in card.fields:
            referenced.update(img_re.findall(field.value))
            referenced.update(sound_re.findall(field.value))

    media_dst.mkdir(parents=True, exist_ok=True)
    copied = skipped = 0
    for filename in referenced:
        src = media_src / filename
        dst = media_dst / filename
        if dst.exists():
            skipped += 1
            continue
        if not src.exists():
            logger.warning("Referenced media file not found: %s", src)
            continue
        import shutil
        shutil.copy2(src, dst)
        copied += 1

    logger.info(
        "Media export: %d copied, %d already present, %d total referenced",
        copied, skipped, len(referenced),
    )


# ─────────────────────────────────────────────────────────────────────────────
# The three modes
# ─────────────────────────────────────────────────────────────────────────────

def mode_export(deck_name: str, out_csv: Path, media_src: Path, media_dst: Path | None, sample: int | None) -> None:
    deck = Deck.from_anki(deck_name)
    if sample:
        deck.cards = random.sample(deck.cards, min(sample, len(deck.cards)))
    logger.info("Loaded %d cards from Anki deck '%s'", len(deck.cards), deck_name)
    write_raw_csv(deck.cards, out_csv)
    if media_dst is not None:
        _copy_referenced_media(deck.cards, media_src, media_dst)
    else:
        logger.info("No --media-out specified; skipping media export")


TRANSFORM_BATCH_SIZE = 100

def mode_transform(in_csv: Path, out_csv: Path, media_path: Path, sample: int | None) -> None:
    """raw_cards.csv + media folder → transformed_cards.csv  (LLM runs here)

    Resumes automatically: any noteId already present in out_csv is skipped.
    Results are flushed to disk after every batch of TRANSFORM_BATCH_SIZE cards.
    """

    # ── Determine which noteIds are already done ──────────────────────────────
    already_done: set[str] = set()
    if out_csv.exists():
        with open(out_csv, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if nid := row.get("noteId"):
                    already_done.add(nid)
        logger.info("Resuming — %d cards already in %s, will skip them", len(already_done), out_csv)

    # ── Load and filter raw cards ─────────────────────────────────────────────
    all_cards = read_raw_csv(in_csv)
    if sample:
        all_cards = random.sample(all_cards, min(sample, len(all_cards)))

    pending: list[Card] = []
    for card in all_cards:
        if card.noteId in already_done:
            logger.info("Skipping note %s — cardType '%s' already transformed", card.noteId, card.cardType)
            continue
        if card.cardType == MASTER_MODEL_NAME:
            logger.info("Skipping note %s — already '%s', no transform needed", card.noteId, MASTER_MODEL_NAME)
            continue
        pending.append(card)

    skipped = len(all_cards) - len(pending)
    logger.info(
        "%d total cards | %d skipped (already done or already master) | %d to transform",
        len(all_cards), skipped, len(pending),
    )

    # ── Build prompt requests, skipping cards that error at prep time ─────────
    requests_list: list[PromptRequest] = []
    valid_cards:   list[Card]          = []
    for card in pending:
        try:
            requests_list.append(build_prompt_request_for_card(card, media_path))
            valid_cards.append(card)
        except Exception as e:
            logger.warning("Skipping card %s (%s) — prompt build failed: %s", card.noteId, card.cardType, e)

    if not valid_cards:
        logger.info("Nothing to transform.")
        return

    # ── Process in batches, saving after each ────────────────────────────────
    total_ok  = 0
    total_fail = 0
    num_batches = (len(valid_cards) + TRANSFORM_BATCH_SIZE - 1) // TRANSFORM_BATCH_SIZE

    for batch_idx in range(num_batches):
        lo = batch_idx * TRANSFORM_BATCH_SIZE
        hi = lo + TRANSFORM_BATCH_SIZE
        batch_cards    = valid_cards[lo:hi]
        batch_requests = requests_list[lo:hi]

        logger.info(
            "Batch %d/%d — sending %d cards to LLM (notes %s … %s)",
            batch_idx + 1, num_batches, len(batch_cards),
            batch_cards[0].noteId, batch_cards[-1].noteId,
        )

        llm_outputs = prompt_batch(batch_requests)

        batch_master: list[MasterGenkiCard] = []
        for card, raw in zip(batch_cards, llm_outputs):
            if not raw:
                logger.warning("No LLM output for card %s", card.noteId)
                total_fail += 1
                continue
            try:
                batch_master.append(assemble_master_card(card, raw))
                total_ok += 1
            except Exception as e:
                logger.warning("Assembly failed for card %s: %s | raw: %.120s", card.noteId, e, raw)
                total_fail += 1

        # Append this batch to the output CSV immediately
        write_transformed_csv(batch_master, out_csv, append=True)
        logger.info(
            "Batch %d/%d complete — %d transformed, %d failed. "
            "Running totals: %d ok / %d failed. Saved → %s",
            batch_idx + 1, num_batches, len(batch_master),
            len(batch_cards) - len(batch_master),
            total_ok, total_fail, out_csv,
        )

    logger.info(
        "Transform finished — %d succeeded, %d failed out of %d attempted",
        total_ok, total_fail, len(valid_cards),
    )


def mode_import(in_csv: Path, dry_run: bool) -> None:
    """transformed_cards.csv → AnkiConnect (in-place update, preserves review history)"""
    master_cards = read_transformed_csv(in_csv)
    if dry_run:
        print(f"DRY RUN — would import {len(master_cards)} cards (pass --no-dry-run to commit)")
        for mc in master_cards[:3]:
            print(f"  {mc.source_note.noteId}: {mc.japanese[:40]}")
        return

    _ensure_model_fields([
        "Japanese", "Japanese Audio", "Furigana", "Reading",
        "English", "English Audio", "Screenshots", "Screenshot Text",
        "Explanations", "Additional Notes", "Previous Version",
    ])
    for mc in master_cards:
        _update_note_in_place(mc)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # ── export ────────────────────────────────────────────────────────────────
    p_export = sub.add_parser("export", help="AnkiConnect → CSV")
    p_export.add_argument("--deck",      default=DEFAULT_DECK,        help="Anki deck name")
    p_export.add_argument("--out",       default=DEFAULT_RAW_CSV,     type=Path)
    p_export.add_argument("--media-src", default=DEFAULT_MEDIA_PATH,  type=Path,
                          help="Anki collection.media source folder")
    p_export.add_argument("--media-out", default=None,                type=Path,
                          help="Destination folder for referenced media files (optional)")
    p_export.add_argument("--sample",    default=None,                type=int)

    # ── transform ─────────────────────────────────────────────────────────────
    p_transform = sub.add_parser("transform", help="CSV → LLM → CSV (run on server)")
    p_transform.add_argument("--in",     dest="in_csv",  default=DEFAULT_RAW_CSV, type=Path)
    p_transform.add_argument("--out",    dest="out_csv", default=DEFAULT_OUT_CSV, type=Path)
    p_transform.add_argument("--media",  default=DEFAULT_MEDIA_PATH,              type=Path,
                             help="Path to media folder (copied during export)")
    p_transform.add_argument("--sample", default=None,                            type=int)

    # ── import ────────────────────────────────────────────────────────────────
    p_import = sub.add_parser("import", help="transformed CSV → AnkiConnect")
    p_import.add_argument("--in",         dest="in_csv", default=DEFAULT_OUT_CSV, type=Path)
    p_import.add_argument("--no-dry-run", dest="dry_run", action="store_false", default=True,
                          help="Actually write to Anki (default is dry-run)")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.mode == "export":
        mode_export(args.deck, args.out, args.media_src, args.media_out, args.sample)
    elif args.mode == "transform":
        mode_transform(args.in_csv, args.out_csv, args.media, args.sample)
    elif args.mode == "import":
        mode_import(args.in_csv, args.dry_run)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()


# real fast use case

# # 1. Local — export deck to CSV + copy only referenced media
# python anki_transform.py export \
#   --deck "Genki I" \
#   --out raw_cards.csv \
#   --media-src "/Users/bmanville3/Library/Application Support/Anki2/User 1/collection.media" \
#   --media-out ./media

# # 2. Server — transform (upload raw_cards.csv + ./media/ first)
# python anki_transform.py transform \
#   --in raw_cards.csv \
#   --out transformed_cards.csv \
#   --media ./media

# # 3. Local — import back into Anki (download transformed_cards.csv first)
# python anki_transform.py import \
#   --in transformed_cards.csv \
#   --no-dry-run
