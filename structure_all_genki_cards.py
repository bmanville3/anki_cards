import json
import logging
from pathlib import Path
import random
import re
from typing import Any, Callable, Self, cast

import requests
from attrs import define
import attrs

from src.common.utils import load_image_b64
from src.prompting.prompter import PromptRequest, prompt_batch


DRY_RUN: bool = True
SAMPLE_CARDS: int | None = None
MEDIA_PATH = Path('/Users/bmanville3/Library/Application Support/Anki2/User 1/collection.media')


def invoke(action, **params):
    return requests.post(
        "http://localhost:8765",
        json={"action": action, "version": 6, "params": params},
    ).json()


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
        if not stripped:
            return [self]

        if not sounds and not images:
            return [self]

        result    = []
        remaining = value

        for i, sound in enumerate(sounds):
            remaining = remaining.replace(sound, '', 1)
            result.append(Field(
                name  = f"{self.name}-audio"   if len(sounds) == 1 else f"{self.name}-audio-{i+1}",
                value = sound,
            ))

        for i, img in enumerate(images):
            remaining = remaining.replace(img, '', 1)
            result.append(Field(
                name  = f"{self.name}-image"   if len(images) == 1 else f"{self.name}-image-{i+1}",
                value = img,
            ))

        remaining = re.sub(r'(<br\s*/?>|\s)+', ' ', remaining).strip()
        if remaining:
            result.insert(0, Field(name=self.name, value=remaining))

        return result

    def map_ugly_ids_to_pretty(self) -> None:
        """Replace raw <img> and [sound:] tags in value with readable placeholders."""
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
        """Restore original <img> and [sound:] tags from placeholders."""
        for placeholder, original in self._internal_mappings.items():
            self.value = self.value.replace(placeholder, original)

    def produce_base_64_images(self) -> list[tuple[str, str]]:
        # Reconstruct original tags from placeholders before extracting filenames
        unsubbed_value = self.value
        for placeholder, original in self._internal_mappings.items():
            unsubbed_value = unsubbed_value.replace(placeholder, original)

        img_pattern = re.compile(r'<img\s+src="([^"]+)"[^>]*>')
        results: list[tuple[str, str]] = []
        for filename in img_pattern.findall(unsubbed_value):
            path = MEDIA_PATH / filename
            loaded = load_image_b64(path)
            if loaded is not None:
                results.append(loaded)
            else:
                print(f"ERROR: Problem loading media at '{path}'")
        return results


def _field_names(fields: list[Field]) -> set[str]:
    return {f.name for f in fields}


def _assert_fields(card_type: str, fields: list[Field], expected: set[str]) -> None:
    actual = _field_names(fields)
    if expected != actual:
        missing = expected - actual
        extra   = actual   - expected
        parts   = []
        if missing:
            parts.append(f"missing={missing}")
        if extra:
            parts.append(f"extra={extra}")
        raise ValueError(f"[{card_type}] Field mismatch — {', '.join(parts)}")


def _base_verify_split(card: "Card", expected_fields: set[str], starts_with_fields: set[str]) -> None:
    names = _field_names(card.fields)
    expected_names = all(n in expected_fields or any(n.startswith(swf) for swf in starts_with_fields) for n in names)
    if not expected_names:
        raise ValueError(
            f"[{card.cardType} split] At least one field did match expected patterns: {names}"
        )
    for field in expected_fields:
        if field not in names:
            raise ValueError(f"[{card.cardType} split] Missing {field=} from {names}")


BASIC_FIELDS = {"Front", "Back"}
BASIC_FIELDS_PLUS_DASH = [p + "-" for p in BASIC_FIELDS]

def verify_basic(card: "Card") -> None:
    _assert_fields(card.cardType, card.fields, BASIC_FIELDS)

def verify_basic_split(card: "Card") -> None:
    _base_verify_split(card, BASIC_FIELDS, BASIC_FIELDS_PLUS_DASH)


PRACTICE_FIELDS = {
    "Prompt",
    "Prompt Audio",
    "Prompt Picture",
    "Prompt Additional Instructions",
    "Prompt (English Tanslation)",
    "Answer",
    "Answer Audio",
    "Answer (English Translation)",
    "Additional Back Explanation",
    "Answer Picture",
}
PRACTICE_FIELDS_PLUS_DASH = [p + "-" for p in PRACTICE_FIELDS]

def verify_genki_practice_card(card: "Card") -> None:
    _assert_fields(card.cardType, card.fields, PRACTICE_FIELDS)

def verify_genki_practice_card_split(card: "Card") -> None:
    _base_verify_split(card, PRACTICE_FIELDS, PRACTICE_FIELDS_PLUS_DASH)


VOCAB_FIELDS = {
    "Japanese",
    "Japanese Audio",
    "Textbook Definition",
    "Picture (example)",
    "Additional Notes",
}
VOCAB_FIELDS_PLUS_DASH = [f + "-" for f in VOCAB_FIELDS]

def verify_genki_vocab_card(card: "Card") -> None:
    _assert_fields(card.cardType, card.fields, VOCAB_FIELDS)

def verify_genki_vocab_card_split(card: "Card") -> None:
    _base_verify_split(card, VOCAB_FIELDS, VOCAB_FIELDS_PLUS_DASH)


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
            for k, v in fields_unloaded.items():
                if not isinstance(k, str) and not isinstance(v, dict):
                    output.append(cls(noteId, cardType, []))
                    continue
            fields = [Field(k, str(v.get('value', ''))) for k, v in cast(dict[str, dict], fields_unloaded).items()]
            output.append(cls(noteId, cardType, fields))
        return output

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
        """Mutate all fields in place — replaces media tags with readable placeholders."""
        for f in self.fields:
            f.map_ugly_ids_to_pretty()

    def map_pretty_ids_to_ugly(self) -> None:
        """Mutate all fields in place — restores original media tags from placeholders."""
        for f in self.fields:
            f.map_pretty_ids_to_ugly()

    def get_all_images(self) -> list[tuple[str, str]]:
        output = []
        for field in self.fields:
            output.extend(field.produce_base_64_images())
        return output


@define
class Deck:
    name:  str
    cards: list[Card]

    @classmethod
    def from_deck(cls, name: str) -> Self:
        notes = invoke("findNotes", query=f'deck:"{name}"')["result"]
        if not notes:
            return cls(name, [])
        info = invoke("notesInfo", notes=notes)["result"]
        return cls(name, Card.from_notes_info(info))

    def pretty_string(self) -> str:
        return f"Deck: {self.name}:\n\n" + "\n\n".join(c.pretty_string() for c in self.cards)


@define
class MasterGenkiCard:
    source_note:      Card         # original unsplit Anki card

    # ── Front / recognition side ──────────────────────────────────────────────
    japanese:         str       = ""
    japanese_audio:   list[str] = None   # [sound:…] tags
    furigana:         str       = ""
    reading:          str       = ""

    # ── Back / production side ────────────────────────────────────────────────
    english:          str       = ""
    english_audio:    str       = ""
    screenshots:      list[str] = None   # <img> tags preserved as-is
    explanations:     str       = ""
    additional_notes: str       = ""

    # ── LLM-extracted text from screenshots ──────────────────────────────────
    screenshot_text:  str       = ""

    # ── Metadata ──────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# Prompts
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
                      with no kanji. For a sentence, reading can be ignored. It is
                      only necessary for vocabulary cards.
  english           - The English meaning or translation.
  explanations      - Grammar notes, part of speech, conjugation notes,
                      usage context, or any other information that would help a
                      Genki learner understand and use the item correctly.
                      Leave blank only if there is truly nothing to add.
  additional_notes  - Any extra details from the provided cards that were not
                      captured in the above fields. Leave blank if not applicable.
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
5. If the source card already contains some of this information (e.g. a typed definition),
   carry it over faithfully and improve/expand it where helpful.
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
The Prompt fields together form the question/stimulus; the Answer fields form the response.
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
- `japanese`  ← Answer (the core Japanese content being studied)
- `english`   ← Answer (English Translation), informed by Prompt (English Translation)
- Synthesise `explanations` from Additional Back Explanation and the prompt context.
- Include the prompt itself in `explanations` if it adds useful context
  (e.g. "Used in response to: ～ますか?").
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
- `english`          ← extracted from Textbook Definition text and/or the attached screenshot(s).
- `additional_notes` ← Additional Notes field content.
- If Additional Notes contains a kana reading, use it for `reading` exactly.
- `screenshot_text`  ← transcribe every Japanese and English word visible in the
                        attached image(s), verbatim.
- `explanations`     ← corresponds to the Textbook Definition. Extract the simple English
                        definition into `english`; put any grammar explanations here.
                        Explanations are often in pictures — read them carefully.
- `furigana` and `reading` you will likely need to generate yourself from your knowledge.

--- CARD CONTENT ---
{card.pretty_string()}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Transformation pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _parse_llm_json(raw: str) -> dict:
    clean = re.sub(r'^```[a-zA-Z]*\n?', '', raw.strip())
    clean = re.sub(r'\n?```$', '', clean)
    output = json.loads(clean)
    if not isinstance(output, dict):
        raise ValueError(f"Expected output to be a dict. Got: {output}")
    missing = [f for f in EXPECTED_OUTPUT_FIELDS if f not in output]
    extra   = [f for f in output if f not in EXPECTED_OUTPUT_FIELDS]
    if missing or extra:
        raise ValueError(
            f"Expected fields: {EXPECTED_OUTPUT_FIELDS}. "
            f"Missing: {missing}. Extra: {extra}. Output: {output}"
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


def build_prompt_request_for_card(card: Card) -> PromptRequest:
    split_card = card.split_fields()
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
        base_64_images = card.get_all_images(),
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

# Target model name — the note model must already have all these fields defined.
MASTER_MODEL_NAME = "Master Genki Card"

def _ensure_model_fields(extra_fields: list[str]) -> None:
    """
    Add any missing fields to the Master Genki Card note type in Anki.
    Safe to call repeatedly — skips fields that already exist.
    AnkiConnect action: modelFieldAdd (available since AnkiConnect v6).
    """
    result  = invoke("modelFieldNames", modelName=MASTER_MODEL_NAME)
    current = set(result.get("result") or [])
    for field in extra_fields:
        if field not in current:
            r = invoke("modelFieldAdd", modelName=MASTER_MODEL_NAME, fieldName=field, index=len(current))
            if r.get("error"):
                print(f"  ⚠ Could not add field '{field}' to model: {r['error']}")
            else:
                current.add(field)
                print(f"  + Added field '{field}' to {MASTER_MODEL_NAME}")


def _change_note_type(note_id: str, target_model: str, field_mapping: dict[str, str]) -> bool:
    """
    Move a note to a different note type using AnkiConnect's changeNotesType action.
    field_mapping maps old field names → new field names (unmapped fields become blank).
    Returns True on success.
    """
    result = invoke(
        "changeNotesType",
        noteIds=[int(note_id)],
        oldModelName=None,          # AnkiConnect infers it from the note
        newModelName=target_model,
        fieldMapping=field_mapping,
    )
    if result.get("error"):
        print(f"  ✗ changeNotesType failed for {note_id}: {result['error']}")
        return False
    return True


def _update_note_in_place(mc: MasterGenkiCard) -> None:
    """
    Write the transformed fields back onto the *existing* note, preserving review history.

    Strategy:
    1. If the note is already the Master model, call updateNoteFields directly.
    2. If it is a legacy model, use changeNotesType to switch it to Master Genki Card
       (Anki preserves scheduling when the note stays in the same deck), then
       call updateNoteFields to fill the new fields.
    """
    note_id   = int(mc.source_note.noteId)
    new_fields = mc.to_anki_fields()

    current_model = mc.source_note.cardType

    if current_model != MASTER_MODEL_NAME:
        # Build a field mapping: every old field → "" (we'll overwrite via updateNoteFields)
        old_fields_result = invoke("modelFieldNames", modelName=current_model)
        old_fields        = old_fields_result.get("result") or []
        field_mapping     = {old: "" for old in old_fields}

        ok = _change_note_type(str(note_id), MASTER_MODEL_NAME, field_mapping)
        if not ok:
            return

    result = invoke(
        "updateNoteFields",
        note={"id": note_id, "fields": new_fields},
    )
    if result.get("error"):
        print(f"  ✗ updateNoteFields failed for {note_id}: {result['error']}")
    else:
        print(f"  ✓ Updated note {note_id} in place")


# ─────────────────────────────────────────────────────────────────────────────
# High-level entry point
# ─────────────────────────────────────────────────────────────────────────────

def transform_deck(
    deck_name:    str,
    *,
    dry_run:      bool      = True,
    sample_cards: int | None = None,
) -> list[MasterGenkiCard]:
    deck = Deck.from_deck(deck_name)
    if sample_cards:
        deck.cards = random.sample(deck.cards, min(sample_cards, len(deck.cards)))
    print(f"Loaded {len(deck.cards)} cards from '{deck_name}'")

    if not dry_run:
        # Ensure the target model has all required fields before we start writing
        _ensure_model_fields(list(MasterGenkiCard.__attrs_attrs__[0].__class__.__annotations__))
        # Simpler: just pass the known field list directly
        _ensure_model_fields([
            "Japanese", "Japanese Audio", "Furigana", "Reading",
            "English", "English Audio", "Screenshots", "Screenshot Text",
            "Explanations", "Additional Notes", "Previous Version",
        ])

    requests_list: list[PromptRequest] = []
    valid_cards:   list[Card]          = []
    for card in deck.cards:
        try:
            req = build_prompt_request_for_card(card)
            requests_list.append(req)
            valid_cards.append(card)
        except Exception as e:
            print(f"  ⚠ Skipping card {card.noteId} ({card.cardType}): {e}")

    print(f"Sending {len(requests_list)} cards to LLM…")
    llm_outputs = prompt_batch(requests_list)

    master_cards: list[MasterGenkiCard] = []
    for card, raw in zip(valid_cards, llm_outputs):
        if not raw:
            print(f"  ✗ No LLM output for card {card.noteId}")
            continue
        try:
            mc = assemble_master_card(card, raw)
            master_cards.append(mc)
        except Exception as e:
            print(f"  ✗ Assembly failed for card {card.noteId}: {e}\n    Raw: {raw[:120]}")

    print(f"Successfully transformed {len(master_cards)} / {len(valid_cards)} cards")

    if not dry_run:
        for mc in master_cards:
            _update_note_in_place(mc)

    return master_cards


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = transform_deck("Genki I", dry_run=DRY_RUN, sample_cards=SAMPLE_CARDS)
    for mc in results[:3]:
        print("─" * 60)
        print(f"japanese:  {mc.japanese}")
        print(f"furigana:  {mc.furigana}")
        print(f"reading:   {mc.reading}")
        print(f"english:   {mc.english}")
        print(f"explain:   {mc.explanations[:80]}")
