"""
# 1. Local — export deck to CSV + copy only referenced media
python structure_all_cards.py export \
  --deck "Genki I" \
  --out raw_cards.csv \
  --media-src "/Users/bmanville3/Library/Application Support/Anki2/User 1/collection.media" \
  --media-out ./media

# 2. Server — transform (upload raw_cards.csv + ./media/ first)
python structure_all_cards.py transform \
  --in raw_cards.csv \
  --out transformed_cards.csv \
  --media ./media

# 3. Local — import back into Anki (download transformed_cards.csv first)
python structure_all_cards.py import \
  --in transformed_cards.csv \
  --no-dry-run
"""

import argparse
import csv
import json
import logging
import random
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Self, cast

import attrs
from bs4 import BeautifulSoup
import requests
from attrs import define

from src.common.utils import load_image_b64
from src.prompting.prompter import LLM_MODEL, LLM_WORKERS, PromptRequest, prompt_batch

TRANSFORM_BATCH_SIZE = LLM_WORKERS
DEFAULT_DECK       = "Genki I"
DEFAULT_MEDIA_PATH = Path('/Users/bmanville3/Library/Application Support/Anki2/User 1/collection.media')
DEFAULT_RAW_CSV    = Path("raw_cards.csv")
DEFAULT_OUT_CSV    = Path("transformed_cards.csv")
MASTER_MODEL_NAME  = "Master Card"
CORE_2000_DECK     = "Core 2000"

RAW_CSV_FIELDS = ["noteId", "cardType", "fields_json"]

logger = logging.getLogger(__name__)


def attr_name_to_anki_field(name: str) -> str:
    if name == "LLM Translator":
        return "llm_translator"
    return " ".join(word.capitalize() for word in name.split("_"))


def anki_field_to_attr_name(field: str) -> str:
    if field == "llm_translator":
        return "LLM Translator"
    return "_".join(word.lower() for word in field.split())


def master_card_to_anki_fields(mc: "MasterGenkiCard") -> dict[str, str]:
    result = {}
    for a in attrs.fields(MasterGenkiCard):
        if a.name in ("source_note", "tags", "is_new_note", "target_deck"):
            continue
        val = getattr(mc, a.name)
        anki_name = attr_name_to_anki_field(a.name)
        if isinstance(val, list):
            result[anki_name] = " ".join(val)
        else:
            result[anki_name] = str(val) if val else ""
    result["Previous Version"] = mc.source_note.pretty_string()
    return result


def invoke(action: str, **params) -> dict:
    return requests.post(
        "http://localhost:8765",
        json={"action": action, "version": 6, "params": params},
    ).json()


@define
class Field:
    name:  str
    value: str
    _internal_mappings: dict[str, str] = attrs.field(factory=dict)

    def pretty_string(self) -> str:
        return f"{self.name}: {self.value}"

    def split_self(self) -> list[Self]:
        value         = self.value
        sound_pattern = re.compile(r'\[sound:[^\]]+\]')
        img_pattern   = re.compile(r'<img[^>]+>')
        sounds        = sound_pattern.findall(value)
        images        = img_pattern.findall(value)
        stripped      = re.sub(r'(\[sound:[^\]]+\]|<img[^>]+>|<br\s*/?>|\s)', '', value)
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

        def replace_image(match: re.Match) -> str:
            nonlocal image_count
            placeholder = f"{{{{IMAGE:{image_count}}}}}"
            self._internal_mappings[placeholder] = match.group(0)
            image_count += 1
            return placeholder

        def replace_audio(match: re.Match) -> str:
            nonlocal audio_count
            placeholder = f"{{{{AUDIO:{audio_count}}}}}"
            self._internal_mappings[placeholder] = match.group(0)
            audio_count += 1
            return placeholder

        self.value = re.sub(r"<img[^>]+>",       replace_image, self.value)
        self.value = re.sub(r"\[sound:[^\]]+\]",  replace_audio, self.value)

    def map_pretty_ids_to_ugly(self) -> None:
        for placeholder, original in self._internal_mappings.items():
            self.value = self.value.replace(placeholder, original)

    def produce_base64_images(self, media_path: Path) -> list[tuple[str, str]]:
        unsubbed = self.value
        for ph, orig in self._internal_mappings.items():
            unsubbed = unsubbed.replace(ph, orig)
        img_re  = re.compile(r'<img\s+src="([^"]+)"[^>]*>')
        results = []
        for filename in img_re.findall(unsubbed):
            try:
                results.append(load_image_b64(media_path / filename))
            except Exception as e:
                logger.error("Problem loading media '%s': %s", filename, e)
        return results


def _field_names(fields: list[Field]) -> set[str]:
    return {f.name for f in fields}


def _assert_fields(card_type: str, fields: list[Field], expected: set[str]) -> None:
    actual  = _field_names(fields)
    missing = expected - actual
    extra   = actual   - expected
    if missing or extra:
        parts = []
        if missing: parts.append(f"missing={missing}")
        if extra:   parts.append(f"extra={extra}")
        raise ValueError(f"[{card_type}] Field mismatch — {', '.join(parts)}")


def _base_verify_split(card: "Card", expected_fields: set[str], starts_with: set[str]) -> None:
    names = _field_names(card.fields)
    for name in names:
        if name not in expected_fields and not any(name.startswith(p) for p in starts_with):
            raise ValueError(f"[{card.cardType} split] Unexpected field: {name!r}")
    for field in expected_fields:
        if field not in names:
            raise ValueError(f"[{card.cardType} split] Missing field {field!r}")


BASIC_FIELDS             = {"Front", "Back"}
PRACTICE_FIELDS          = {
    "Prompt", "Prompt Audio", "Prompt Picture", "Prompt Additional Instructions",
    "Prompt (English Tanslation)", "Answer", "Answer Audio",
    "Answer (English Translation)", "Additional Back Explanation", "Answer Picture",
}
VOCAB_FIELDS             = {"Japanese", "Japanese Audio", "Textbook Definition", "Picture (example)", "Additional Notes"}

CORE_2000_FIELDS         = {
    "Optimized-Voc-Index", "Vocabulary-Kanji", "Vocabulary-Furigana", "Vocabulary-Kana",
    "Vocabulary-English", "Vocabulary-Audio", "Vocabulary-Pos", "Caution",
    "Expression", "Reading", "Sentence-Kana", "Sentence-English", "Sentence-Clozed",
    "Sentence-Audio", "Notes", "Core-Index", "Optimized-Sent-Index", "Frequency",
    "English-Word-Audio", "English-Sentence-Audio", "Gemma4",
}

VIDEO_FIELDS             = {
    "Text", "Audio", "Image", "NaturalTranslation", "LiteralTranslation",
    "TTSAudio", "Furigana", "WordGloss", "TimeCode", "FontName", "Source", "Notes",
}

JLAB_FIELDS              = {
    "Version", "Sequence", "Source", "Audio", "Image",
    "RemarksFront", "RemarksBack", "QuestionLink", "References",
    "Other-Front", "Other-Back",
    "Jlab-Kanji", "Jlab-KanjiSpaced", "Jlab-Hiragana", "Jlab-KanjiCloze",
    "Jlab-Lemma", "Jlab-HiraganaCloze", "Jlab-Translation",
    "Jlab-DictionaryLookup", "Jlab-Metadata", "Jlab-Remarks",
    "Jlab-ListeningFront", "Jlab-ListeningBack",
    "Jlab-ClozeFront", "Jlab-ClozeBack",
}

WILD_CARD_FIELDS         = {
    "expression", "sentence", "furigana", "reading", "glossary",
    "audio", "screenshot", "pitch-accent-graphs-jj", "url",
}

def _noop(_: "Card") -> None: pass

def _verify_basic(card: "Card")          -> None: _assert_fields(card.cardType, card.fields, BASIC_FIELDS)
def _verify_basic_split(card: "Card")    -> None: _base_verify_split(card, BASIC_FIELDS, {"Front-", "Back-"})
def _verify_practice(card: "Card")       -> None: _assert_fields(card.cardType, card.fields, PRACTICE_FIELDS)
def _verify_practice_split(card: "Card") -> None: _base_verify_split(card, PRACTICE_FIELDS, {p + "-" for p in PRACTICE_FIELDS})
def _verify_vocab(card: "Card")          -> None: _assert_fields(card.cardType, card.fields, VOCAB_FIELDS)
def _verify_vocab_split(card: "Card")    -> None: _base_verify_split(card, VOCAB_FIELDS, {f + "-" for f in VOCAB_FIELDS})
def _verify_core2000(card: "Card")       -> None: _assert_fields(card.cardType, card.fields, CORE_2000_FIELDS)
def _verify_video(card: "Card")          -> None: _assert_fields(card.cardType, card.fields, VIDEO_FIELDS)
def _verify_jlab(card: "Card")           -> None: _assert_fields(card.cardType, card.fields, JLAB_FIELDS)
def _verify_wild(card: "Card")           -> None: _assert_fields(card.cardType, card.fields, WILD_CARD_FIELDS)

CARD_TYPE_VERIFIERS: dict[str, Callable[["Card"], None]] = {
    "Basic":                        _verify_basic,
    "Basic (split)":                _verify_basic_split,
    "Genki Practice Card":          _verify_practice,
    "Genki Practice Card Split":    _verify_practice_split,
    "Genki Vocab Card":             _verify_vocab,
    "Genki Vocab Card Split":       _verify_vocab_split,
    "Core 2000":                    _verify_core2000,
    "Japanese Video Sentence Cards+": _verify_video,
    "JlabNote-JlabConverted-1":     _verify_jlab,
    "Wild Cards":                   _verify_wild,
    MASTER_MODEL_NAME:              _noop,
}

SPLIT_TYPE_MAP = {
    "Basic":               "Basic (split)",
    "Genki Practice Card": "Genki Practice Card Split",
    "Genki Vocab Card":    "Genki Vocab Card Split",
}


@define
class Card:
    noteId:   str
    cardType: str
    fields:   list[Field]

    def __attrs_post_init__(self):
        verifier = CARD_TYPE_VERIFIERS.get(self.cardType)
        if verifier is None:
            raise ValueError(f"No verifier registered for card type '{self.cardType}'")
        verifier(self)

    @classmethod
    def from_notes_info(cls, data: Any) -> list[Self]:
        output = []
        for note in (data if isinstance(data, list) else []):
            if not isinstance(note, dict):
                continue
            noteId = note.get("noteId")
            if noteId is None:
                continue
            cardType  = note.get("modelName", "")
            raw_fields = note.get("fields") or {}
            fields = [Field(k, str(v.get("value", ""))) for k, v in cast(dict[str, dict], raw_fields).items()]
            try:
                output.append(cls(noteId, cardType, fields))
            except ValueError as e:
                logger.warning("Skipping note %s: %s", noteId, e)
        return output

    def to_csv_row(self) -> dict:
        return {
            "noteId":      self.noteId,
            "cardType":    self.cardType,
            "fields_json": json.dumps({f.name: f.value for f in self.fields}, ensure_ascii=False),
        }

    @classmethod
    def from_csv_row(cls, row: dict) -> Self:
        fields_dict = json.loads(row["fields_json"])
        return cls(
            noteId   = row["noteId"],
            cardType = row["cardType"],
            fields   = [Field(k, v) for k, v in fields_dict.items()],
        )

    def pretty_string(self) -> str:
        joined = "\n\t".join(f.pretty_string() for f in self.fields)
        return f"Card {self.noteId}\n- Fields:\n\t{joined}"

    def get_field(self, name: str) -> str:
        for f in self.fields:
            if f.name == name:
                return f.value
        raise ValueError(f"{self.cardType} has no field '{name}'")

    def get_fields_like(self, prefix: str) -> list[Field]:
        return [f for f in self.fields if f.name.startswith(prefix)]

    def split_fields(self) -> "Card":
        new_type   = SPLIT_TYPE_MAP.get(self.cardType, self.cardType)
        new_fields = []
        for f in self.fields:
            new_fields.extend(f.split_self())
        return Card(noteId=self.noteId, cardType=new_type, fields=new_fields)

    def map_ugly_ids_to_pretty(self) -> None:
        for f in self.fields: f.map_ugly_ids_to_pretty()

    def map_pretty_ids_to_ugly(self) -> None:
        for f in self.fields: f.map_pretty_ids_to_ugly()

    def produce_base64_images(self, media_path: Path) -> list[tuple[str, str]]:
        out = []
        for f in self.fields:
            out.extend(f.produce_base64_images(media_path))
        return out


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


def strip_html_formatting(text: str) -> str:
    """Remove inline formatting tags (bold, italic, underline) but keep ruby/structure."""
    text = re.sub(r'</?b>',  '', text)
    text = re.sub(r'</?i>',  '', text)
    text = re.sub(r'</?u>',  '', text)
    text = re.sub(r'</?strong>', '', text)
    text = re.sub(r'</?em>',     '', text)
    return text


def strip_ruby(text: str) -> str:
    text = re.sub(r'<rt>[^<]*</rt>', '', text)
    text = re.sub(r'</?ruby>', '', text)
    return text


def bracket_furigana_to_ruby(text: str) -> str:
    return re.sub(r'([^\s\[]+)\[([^\]]+)\]', r'<ruby>\1<rt>\2</rt></ruby>', text)


def strip_brackets(text: str) -> str:
    return strip_ruby(bracket_furigana_to_ruby(text))


def ruby_to_reading(text: str) -> str:
    text = re.sub(r'<ruby>.*?<rt>(.*?)</rt></ruby>', r'\1', text)
    return text


def bracket_to_reading(text: str) -> str:
    return re.sub(r'([^\s\[]+)\[([^\]]+)\]', r'\2', text)


def furigana_to_reading(text: str) -> str:
    return ruby_to_reading(bracket_furigana_to_ruby(text))

def parse_yomitan_glossary(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")

    # --- reading forms only (the row headers, not the kanji column headers) ---
    forms_table = soup.select_one("li[data-sc-content='forms'] table")
    readings = []
    if forms_table:
        for tr in forms_table.select("tr:not([data-sc-content='forms-header-row'])"):
            th = tr.find("th")
            if th:
                readings.append(th.get_text(strip=True))

    # --- senses, preserving per-sense structure ---
    senses = []
    for sense_group in soup.select("li[data-sc-content='sense-group']"):
        pos = [
            tag.get_text(strip=True)
            for tag in sense_group.find_all("span", attrs={"data-sc-content": "part-of-speech-info"}, recursive=False)
        ]

        for sense in sense_group.select("li[data-sc-content='sense']"):
            glosses = [
                li.get_text(strip=True)
                for li in sense.select("ul[data-sc-content='glossary'] li")
            ]
            if glosses:
                senses.append({"pos": pos, "glosses": glosses})

    return {"readings": readings, "senses": senses}


def glossary_to_prompt_text(html: str) -> str:
    parsed = parse_yomitan_glossary(html)
    lines = []
    if parsed["readings"]:
        lines.append(f"Word: {parsed['readings'][0]}")
    if len(parsed["readings"]) > 1:
        lines.append(f"Readings: {', '.join(parsed['readings'])}")
    for i, sense in enumerate(parsed["senses"], 1):
        pos_str = "/".join(sense["pos"]) if sense["pos"] else ""
        gloss_str = "; ".join(sense["glosses"])
        lines.append(f"  {i}. [{pos_str}] {gloss_str}" if pos_str else f"  {i}. {gloss_str}")
    return "\n".join(lines)

def build_tags(
    *,
    source_deck:  str,
    card_subtype: str | None = None,
    extra: list[str] | None  = None,
) -> list[str]:
    slug = re.sub(r'\s+', '_', source_deck.strip().lower())
    tags = [f"source::{slug}"]
    if card_subtype:
        tags.append(f"type::{card_subtype.lower()}")
    tags.extend(extra or [])
    return tags


_MASTER_CARD_CSV_FIELDS = [
    "noteId", "cardType",
    "japanese", "japanese_audio", "furigana", "reading",
    "english", "english_audio",
    "screenshots", "screenshot_text",
    "explanations", "additional_notes",
    "tags",
    "llm_translator", "japanese_audio_model", "english_audio_model",
    "source",
    "previous_version",
    # Core 2000 split: sentence cards need to be inserted rather than updated.
    # We track this in the CSV so mode_import knows what to do.
    "is_new_note",   # "1" if this row should be addNote'd, "" if updateNoteFields
    "target_deck",   # deck name for addNote (only used when is_new_note == "1")
]

_ANKI_FIELD_NAMES = [
    "Japanese", "Japanese Audio", "Furigana", "Reading",
    "English", "English Audio",
    "Screenshots", "Screenshot Text",
    "Explanations", "Additional Notes",
    "Previous Version", "LLM Translator",
    "Japanese Audio Model", "English Audio Model",
    "Source",
]


@define
class MasterGenkiCard:
    source_note:          Card
    japanese:             str       = ""
    japanese_audio:       list[str] = attrs.Factory(list)
    furigana:             str       = ""
    reading:              str       = ""
    english:              str       = ""
    english_audio:        str       = ""
    screenshots:          list[str] = attrs.Factory(list)
    explanations:         str       = ""
    additional_notes:     str       = ""
    screenshot_text:      str       = ""
    tags:                 list[str] = attrs.Factory(list)
    llm_translator:       str       = ""
    japanese_audio_model: str       = ""
    english_audio_model:  str       = ""
    source:               str       = ""
    # Import routing — not an Anki field
    is_new_note:          bool      = False
    target_deck:          str       = ""

    def __attrs_post_init__(self):
        self.japanese = strip_ruby(self.japanese)

    def to_anki_fields(self) -> dict[str, str]:
        return master_card_to_anki_fields(self)

    def to_csv_row(self) -> dict:
        return {
            "noteId":              self.source_note.noteId,
            "cardType":            self.source_note.cardType,
            "japanese":            self.japanese,
            "japanese_audio":      json.dumps(self.japanese_audio,  ensure_ascii=False),
            "furigana":            self.furigana,
            "reading":             self.reading,
            "english":             self.english,
            "english_audio":       self.english_audio,
            "screenshots":         json.dumps(self.screenshots,     ensure_ascii=False),
            "screenshot_text":     self.screenshot_text,
            "explanations":        self.explanations,
            "additional_notes":    self.additional_notes,
            "tags":                json.dumps(self.tags,            ensure_ascii=False),
            "llm_translator":      self.llm_translator,
            "japanese_audio_model":self.japanese_audio_model,
            "english_audio_model": self.english_audio_model,
            "source":              self.source,
            "previous_version":    self.source_note.pretty_string(),
            "is_new_note":         "1" if self.is_new_note else "",
            "target_deck":         self.target_deck,
        }

    @classmethod
    def from_csv_row(cls, row: dict, source_card: Card) -> Self:
        # problems with doing this with cattrs
        def parse_list(val: str) -> list[str]:
            stripped = (val or "").strip()
            if not stripped or stripped == "[]":
                return []
            return json.loads(stripped)

        return cls(
            source_note          = source_card,
            japanese             = row.get("japanese",             ""),
            japanese_audio       = parse_list(row.get("japanese_audio",    "[]")),
            furigana             = row.get("furigana",             ""),
            reading              = row.get("reading",              ""),
            english              = row.get("english",              ""),
            english_audio        = row.get("english_audio",        ""),
            screenshots          = parse_list(row.get("screenshots",       "[]")),
            explanations         = row.get("explanations",         ""),
            additional_notes     = row.get("additional_notes",     ""),
            screenshot_text      = row.get("screenshot_text",      ""),
            tags                 = parse_list(row.get("tags",              "[]")),
            llm_translator       = row.get("llm_translator",       ""),
            japanese_audio_model = row.get("japanese_audio_model", ""),
            english_audio_model  = row.get("english_audio_model",  ""),
            source               = row.get("source",               ""),
            is_new_note          = row.get("is_new_note", "") == "1",
            target_deck          = row.get("target_deck",          ""),
        )


def write_raw_csv(cards: list[Card], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=RAW_CSV_FIELDS)
        writer.writeheader()
        for card in cards:
            writer.writerow(card.to_csv_row())
    logger.info(f"Wrote {len(cards)} raw cards → {path}")


def read_raw_csv(path: Path) -> list[Card]:
    cards = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                cards.append(Card.from_csv_row(row))
            except Exception as e:
                logger.warning("Skipping malformed row (noteId=%s): %s", row.get("noteId"), e)
    logger.info(f"Read {len(cards)} raw cards ← {path}")
    return cards


def write_transformed_csv(master_cards: list["MasterGenkiCard"], path: Path, append: bool = False) -> None:
    write_header = not append or not path.exists() or path.stat().st_size == 0
    mode = "a" if append else "w"
    with open(path, mode, newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_MASTER_CARD_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        for mc in master_cards:
            writer.writerow(mc.to_csv_row())
    action = "Appended" if append else "Wrote"
    logger.info("%s %d transformed cards → %s", action, len(master_cards), path)


def read_transformed_csv(path: Path) -> list["MasterGenkiCard"]:
    master_cards = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            note_id   = row.get("noteId", "")
            card_type = row.get("cardType", "")
            stub_card = object.__new__(Card)
            object.__setattr__(stub_card, "noteId",   note_id)
            object.__setattr__(stub_card, "cardType", card_type)
            object.__setattr__(stub_card, "fields",   [Field("_stub", row.get("previous_version", ""))])
            try:
                master_cards.append(MasterGenkiCard.from_csv_row(row, stub_card))
            except Exception as e:
                logger.warning("Skipping malformed row (noteId=%s): %s", note_id, e)
    logger.info(f"Read {len(master_cards)} transformed cards ← {path}")
    return master_cards


BASE_SYSTEM_PROMPT = f"""\
You are an expert Japanese language tutor and Anki card formatter.

Your job is to convert a legacy Anki flashcard into a standardised "{MASTER_MODEL_NAME}".
The {MASTER_MODEL_NAME} has the following fields:

  japanese          - The Japanese text: a word, phrase, or full sentence.
  furigana          - The japanese text with furigana inserted above every kanji using
                      the HTML ruby format. Add furigana ONLY above kanji characters,
                      never above hiragana or katakana (they are already readable).

                      Examples:
                        Input:  日本語
                        Output: <ruby>日本語<rt>にほんご</rt></ruby>

                        Input:  食べる
                        Output: <ruby>食<rt>た</rt></ruby>べる

                        Input:  私は学生です
                        Output: <ruby>私<rt>わたし</rt></ruby>は<ruby>学生<rt>がくせい</rt></ruby>です

                        Input:  アメリカに行きました
                        Output: アメリカに<ruby>行<rt>い</rt></ruby>きました

                        Input:  飲み物
                        Output: <ruby>飲<rt>の</rt></ruby>み<ruby>物<rt>もの</rt></ruby>

                      Key rules:
                      - Each kanji or kanji compound gets its own ruby tag
                      - Hiragana and katakana pass through unchanged, no ruby tag
                      - Okurigana (the hiragana attached to a kanji verb/adjective)
                        stays outside the ruby tag: <ruby>食<rt>た</rt></ruby>べる
                      Furigana should be added for both full sentences and vocab cards.
                      If a vocab card is pure hiragana or katakana, just place the
                      hiragana or katakana in this field with no ruby tags.
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
                      no helpful text.

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

def _base_sentence_prompt(card_name: str, front_content: str, additional_rules: list[str] | None = None) -> str:
    rules = [
        "No markdown fences. No extra explanations.",
        "The card will likely contain everything you need to do this "
        "or may already be a one-line English summary. In this case, "
        "pull only from the card - do not add anything. If the card "
        "does NOT contain enough information to produce a one-line "
        "summary, you may produce your own one-line summary. Make "
        "it concise and literal.",
        'The "one-line" summary should always be the answer to the front.',
    ]
    rules.extend(additional_rules or [])
    str_rules = ""
    for i, rule in enumerate(rules):
        str_rules += f"\n{i+1}. {rule}"
    return f"""\
You are an expert Japanese language tutor and Anki card formatter.

Your job is to convert a {card_name} flashcard into a standardised "{MASTER_MODEL_NAME}".

The card shows {front_content} on the front. Your primary task is to
produce a concise one-line English summary for the `english` field (the answer field).
All other fields are mapped directly from the source — do NOT invent or
reinterpret them.

Output ONLY a concise one-line English summary for the `english` field.

Rules:
{str_rules}
"""

JLAB_SYSTEM_PROMPT = _base_sentence_prompt(
    "Japanese Like a Breeze (JLab)",
    "a Japanese sentence",
    [
        "Jlab often has words in '[]'. "
        "This means that the words are implied "
        "but not explicitly said. Do NOT add these "
        "words into your summary. If jlab considers "
        "them implied by adding '[]', ignore them."
    ],
)
WILD_SYSTEM_PROMPT = _base_sentence_prompt("Yomitan", "a Japanese vocab word/expression and a context sentence")


EXPECTED_OUTPUT_FIELDS = [
    "japanese", "furigana", "reading", "english",
    "explanations", "additional_notes", "screenshot_text",
]


def _parse_llm_json(raw: str) -> dict:
    clean   = re.sub(r'^```[a-zA-Z]*\n?', '', raw.strip())
    clean   = re.sub(r'\n?```$', '', clean)
    output  = json.loads(clean)
    if not isinstance(output, dict):
        raise ValueError(f"Expected dict, got: {type(output)}")
    missing = [f for f in EXPECTED_OUTPUT_FIELDS if f not in output]
    extra   = [f for f in output if f not in EXPECTED_OUTPUT_FIELDS]
    if missing or extra:
        raise ValueError(f"Field mismatch — missing={missing}, extra={extra}")
    return output


def _parse_llm_sentence(raw: str) -> str:
    clean = re.sub(r'^```[a-zA-Z]*\n?', '', raw.strip())
    clean = re.sub(r'\n?```$', '', clean)
    return clean


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


def build_jlab_prompt(card: Card) -> str:
    kanji   = card.get_field("Jlab-Kanji")
    remarks = card.get_field("RemarksBack")
    return f"""\
SOURCE CARD TYPE: JLab (Japanese Like a Breeze)

Japanese sentence: {kanji}
RemarksBack (existing back translation/explanation): {remarks or "(none)"}

Produce a one-line English translation/answer for the front of the card.
It should draw on RemarksBack if present or any other information within the card.

--- FULL CARD ---
{card.pretty_string()}
"""


def build_wild_prompt(card: Card) -> str:
    expression = card.get_field("expression")
    sentence   = card.get_field("sentence")
    glossary   = glossary_to_prompt_text(card.get_field("glossary"))
    return f"""\
SOURCE CARD TYPE: Yomitan

Japanese vocab/expression: {expression}
Context sentence: {sentence}
Yomitan dictionary: {glossary}

Produce a one-line English translation/answer for the front of the card.
It should draw from the Yomitan dictionary and answer the Japanese vocab/expression.

--- FULL CARD ---
{card.pretty_string()}
"""


def transform_core2000(card: Card) -> tuple["MasterGenkiCard", "MasterGenkiCard"]:
    # vocab
    vocab_kanji      = card.get_field("Vocabulary-Kanji")
    vocab_furigana   = card.get_field("Vocabulary-Furigana")
    vocab_kana       = card.get_field("Vocabulary-Kana")
    vocab_english    = card.get_field("Vocabulary-English")
    vocab_audio      = card.get_field("Vocabulary-Audio")
    vocab_pos        = card.get_field("Vocabulary-Pos")
    vocab_freq       = card.get_field("Frequency")
    vocab_eng_audio  = card.get_field("English-Word-Audio")
    # sentence
    sent_eng_audio   = card.get_field("English-Sentence-Audio")
    sent_eng_gemma4  = card.get_field("Gemma4")
    sent_eng_orig    = card.get_field("Sentence-English")
    sent_expression  = card.get_field("Expression")
    sent_furigana    = card.get_field("Reading")
    sent_kana        = card.get_field("Sentence-Kana")
    sent_audio       = card.get_field("Sentence-Audio")
    # shared
    caution          = card.get_field("Caution")
    notes_field      = card.get_field("Notes")

    sent_expression = strip_html_formatting(sent_expression)
    sent_furigana   = strip_html_formatting(sent_furigana)
    sent_kana       = strip_html_formatting(sent_kana)
    vocab_kanji     = strip_html_formatting(vocab_kanji)
    vocab_furigana  = strip_html_formatting(vocab_furigana)
    vocab_kana      = strip_html_formatting(vocab_kana)

    vocab_furigana = bracket_furigana_to_ruby(vocab_furigana) if vocab_furigana else vocab_kanji
    sent_furigana  = bracket_furigana_to_ruby(sent_furigana)  if sent_furigana  else sent_expression

    vocab_explanations = []
    if vocab_pos:    vocab_explanations.append(f"Part of speech: {vocab_pos}")
    if caution:      vocab_explanations.append(f"Caution: {caution}")
    if vocab_freq:   vocab_explanations.append(f"Frequency rank: {vocab_freq}")
    if notes_field:  vocab_explanations.append(notes_field)

    sentence_additional = []
    if caution:       sentence_additional.append(f"Caution: {caution}")
    if notes_field:   sentence_additional.append(notes_field)
    if sent_eng_orig: sentence_additional.append(f"Original translation: {sent_eng_orig}")

    base_tags = build_tags(source_deck="Core 2000", extra=["llm::gemma4_31b"])

    vocab_card = MasterGenkiCard(
        source_note          = card,
        japanese             = strip_brackets(vocab_kanji),
        japanese_audio       = [vocab_audio] if vocab_audio else [],
        furigana             = vocab_furigana,
        reading              = vocab_kana,
        english              = vocab_english,
        english_audio        = vocab_eng_audio,
        screenshots          = [],
        explanations         = "\n".join(vocab_explanations),
        additional_notes     = "",
        screenshot_text      = "",
        tags                 = base_tags + ["type::vocab"],
        llm_translator       = "core2000_bundled",
        japanese_audio_model = "core2000_bundled",
        english_audio_model  = "kokoro_af_heart",
        source               = "Core 2000",
        is_new_note          = False,
        target_deck          = "",
    )

    sent_card = MasterGenkiCard(
        source_note          = card,
        japanese             = strip_brackets(sent_expression),
        japanese_audio       = [sent_audio] if sent_audio else [],
        furigana             = sent_furigana,
        reading              = sent_kana,
        english              = sent_eng_gemma4,
        english_audio        = sent_eng_audio,
        screenshots          = [],
        explanations         = f"Vocabulary in context: {vocab_kanji} ({vocab_english})",
        additional_notes     = "\n".join(sentence_additional),
        screenshot_text      = "",
        tags                 = base_tags + ["type::sentence"],
        llm_translator       = "gemma4-31b",
        japanese_audio_model = "core2000_bundled",
        english_audio_model  = "kokoro_af_heart",
        source               = "Core 2000",
        is_new_note          = True,
        target_deck          = CORE_2000_DECK,
    )

    return (vocab_card, sent_card)


def transform_video(card: Card) -> "MasterGenkiCard":
    japanese        = card.get_field("Text")
    japanese_audio  = card.get_field("Audio")
    image           = card.get_field("Image")
    natural_english = card.get_field("NaturalTranslation")
    literal_english = card.get_field("LiteralTranslation")
    english_audio   = card.get_field("TTSAudio")
    furigana_raw    = card.get_field("Furigana")
    word_gloss      = card.get_field("WordGloss")
    source_time     = card.get_field("TimeCode")
    source_file     = card.get_field("Source")
    notes           = card.get_field("Notes")

    source = f"Japanese Video Deck: {source_file} {source_time}"

    additional = []
    if literal_english: additional.append(f"Literal: {literal_english}")
    if source_file:     additional.append(f"Source: {source_file}")
    if notes:           additional.append(notes)

    return MasterGenkiCard(
        source_note          = card,
        japanese             = japanese,
        japanese_audio       = [japanese_audio] if japanese_audio else [],
        furigana             = furigana_raw or japanese,
        reading              = furigana_to_reading(furigana_raw),
        english              = natural_english,
        english_audio        = english_audio,
        screenshots          = [image] if image else [],
        explanations         = word_gloss,
        additional_notes     = "\n".join(additional),
        screenshot_text      = "",
        tags                 = build_tags(
            source_deck  = "Japanese Video Deck",
            card_subtype = "sentence",
            extra        = ["source_media::cij"],
        ),
        llm_translator       = "gemma4-31b",  # english provided by this beforehand
        japanese_audio_model = "video_bundled",
        english_audio_model  = "kokoro_af_heart",
        source               = source,
    )


def transform_wild(card: Card, english: str) -> "MasterGenkiCard":
    expression   = card.get_field("expression")
    sentence     = card.get_field("sentence")
    furigana_raw = card.get_field("furigana")
    reading      = card.get_field("reading")
    glossary     = card.get_field("glossary")
    audio        = card.get_field("audio")
    screenshot   = card.get_field("screenshot")
    pitch_accent = card.get_field("pitch-accent-graphs-jj")
    url          = card.get_field("url")

    furigana_html = bracket_furigana_to_ruby(furigana_raw)

    additional_parts = []
    if pitch_accent: additional_parts.append(f"{pitch_accent}")
    if sentence:     additional_parts.append(f"Example sentence: {sentence}")

    return MasterGenkiCard(
        source_note          = card,
        japanese             = expression,
        japanese_audio       = [audio] if audio else [],
        furigana             = furigana_html or expression,
        reading              = reading,
        english              = english,
        english_audio        = "",
        screenshots          = [screenshot] if screenshot else [],
        explanations         = glossary,
        additional_notes     = "\n".join(additional_parts),
        screenshot_text      = "",
        tags                 = build_tags(
            source_deck  = "In the Wild",
            card_subtype = "vocab",
            extra        = ["source::yomitan"],
        ),
        llm_translator       = LLM_MODEL,
        japanese_audio_model = "yomitan_bundled",
        english_audio_model  = "",
        source               = url or "Yomitan",
    )


def transform_jlab(card: Card, english: str) -> "MasterGenkiCard":
    source         = card.get_field("Source")
    japanese_audio = card.get_field("Audio")
    image          = card.get_field("Image")
    remarks_front  = card.get_field("RemarksFront")
    explanation    = card.get_field("RemarksBack")
    question_link  = card.get_field("QuestionLink")
    references     = card.get_field("References")
    furigana       = card.get_field("Other-Front")
    other_back     = card.get_field("Other-Back")
    japanese       = card.get_field("Jlab-Kanji")
    spaced_jp      = card.get_field("Jlab-KanjiSpaced")
    reading        = card.get_field("Jlab-Hiragana")
    romaji         = card.get_field("Jlab-ListeningFront")

    furigana = bracket_furigana_to_ruby(furigana)

    additional_parts = []
    if romaji:        additional_parts.append(f"Romaji: {romaji}")
    if spaced_jp:     additional_parts.append(f"Spaced: {spaced_jp}")
    if remarks_front: additional_parts.append(f"Remarks front: {remarks_front}")
    if question_link: additional_parts.append(f"Question link: {question_link}")
    if references:    additional_parts.append(f"References: {references}")
    if other_back:    additional_parts.append(f"Other back: {other_back}")

    return MasterGenkiCard(
        source_note          = card,
        japanese             = japanese,
        japanese_audio       = [japanese_audio] if japanese_audio else [],
        furigana             = furigana,
        reading              = reading,
        english              = english,
        english_audio        = "",
        screenshots          = [image] if image else [],
        explanations         = explanation,
        additional_notes     = "\n".join(additional_parts),
        screenshot_text      = "",
        tags                 = build_tags(
            source_deck  = "Jlab's beginner course",
            card_subtype = "sentence",
            extra        = ["source::jlab"],
        ),
        llm_translator       = LLM_MODEL,
        japanese_audio_model = "jlab_bundled",
        english_audio_model  = "",
        source               = source or "Jlab",
    )


def build_prompt_request_for_card(card: Card, media_path: Path) -> PromptRequest:
    ct = card.cardType
    if ct not in LLM_CARD_TYPES:
        raise ValueError(f"{ct!r} not in {LLM_CARD_TYPES}")

    split_card = card.split_fields()
    split_card.map_ugly_ids_to_pretty()
    sct = split_card.cardType
    if sct not in LLM_CARD_TYPES:
        raise ValueError(f"{sct} not in {LLM_CARD_TYPES}")

    if ct == "JlabNote-JlabConverted-1":
        return PromptRequest(
            system_prompt  = JLAB_SYSTEM_PROMPT,
            user_prompt    = build_jlab_prompt(split_card),
            base_64_images = None,  # not helpful for jlab
            validator      = _parse_llm_sentence,
        )
    
    if ct == "Wild Cards":
        return PromptRequest(
            system_prompt  = WILD_SYSTEM_PROMPT,
            user_prompt    = build_wild_prompt(split_card),
            base_64_images = None,  # not helpful for wild cards
            validator      = _parse_llm_sentence,
        )

    if sct in ("Basic", "Basic (split)"):
        user_prompt = build_basic_prompt(split_card)
    elif sct in ("Genki Practice Card", "Genki Practice Card Split"):
        user_prompt = build_practice_prompt(split_card)
    elif sct in ("Genki Vocab Card", "Genki Vocab Card Split"):
        user_prompt = build_vocab_prompt(split_card)
    else:
        raise ValueError(f"No prompt builder for card type: {ct!r}")

    return PromptRequest(
        system_prompt  = BASE_SYSTEM_PROMPT,
        user_prompt    = user_prompt,
        base_64_images = card.produce_base64_images(media_path),
        validator      = _parse_llm_json,
    )


def _llm_card_tags(card: Card) -> list[str]:
    ct = card.cardType
    if "Genki Practice Card" in ct:
        return build_tags(source_deck="Genki I", card_subtype="sentence")
    if "Genki Vocab Card" in ct:
        return build_tags(source_deck="Genki I", card_subtype="vocab-sentence")
    if "Basic" in ct:
        return build_tags(source_deck="Genki I", card_subtype="vocab-sentence")
    if "Jlab" in ct:
        return build_tags(source_deck="Jlab's beginner course", card_subtype="sentence")
    if "Core 2000" in ct:
        return build_tags(source_deck="Core 2000", card_subtype="vocab-sentence")
    if "Wild" in ct:
        return build_tags(source_deck="In the Wild", card_subtype="vocab")
    if "Japanese Video Sentence Cards" in ct:
        return build_tags(source_deck="Japanese Video Deck", card_subtype="sentence")
    raise ValueError(f"Unsupported llm card type: '{ct}'")


NO_LLM_CARD_TYPES = {"Core 2000", "Japanese Video Sentence Cards+"}
LLM_CARD_TYPES    = {"Basic", "Basic (split)", "Genki Practice Card", "Genki Vocab Card",
                     "JlabNote-JlabConverted-1", "Wild Cards"}


def assemble_master_card_from_llm(card: Card, llm_raw: str) -> "MasterGenkiCard":
    card.map_pretty_ids_to_ugly()
    if "Jlab" in card.cardType:
        return transform_jlab(card, _parse_llm_sentence(llm_raw))
    if "Wild" in card.cardType:
        return transform_wild(card, _parse_llm_sentence(llm_raw))

    data = _parse_llm_json(llm_raw)
    return MasterGenkiCard(
        source_note          = card,
        japanese             = data.get("japanese",         ""),
        furigana             = data.get("furigana",         ""),
        reading              = data.get("reading",          ""),
        english              = data.get("english",          ""),
        english_audio        = "",
        japanese_audio       = _collect_audio(card),
        screenshots          = _collect_images(card),
        screenshot_text      = data.get("screenshot_text", ""),
        explanations         = data.get("explanations",    ""),
        additional_notes     = data.get("additional_notes",""),
        tags                 = _llm_card_tags(card),
        llm_translator       = LLM_MODEL,
        japanese_audio_model = "",
        english_audio_model  = "",
        source               = "",
    )


def transform_no_llm(card: Card) -> Iterable["MasterGenkiCard"]:
    ct = card.cardType
    if ct not in NO_LLM_CARD_TYPES:
        raise ValueError(f"{ct!r} not in NO_LLM_CARD_TYPES")
    if ct == "Core 2000":
        return transform_core2000(card)
    if ct == "Japanese Video Sentence Cards+":
        return [transform_video(card)]
    raise ValueError(f"Unhandled no-LLM card type: {ct!r}")


def _ensure_model_fields(field_names: list[str]) -> None:
    result  = invoke("modelFieldNames", modelName=MASTER_MODEL_NAME)
    current = set(result.get("result") or [])
    for field in field_names:
        if field not in current:
            r = invoke("modelFieldAdd", modelName=MASTER_MODEL_NAME, fieldName=field, index=len(current))
            if r.get("error"):
                raise ValueError(f"Could not add field '{field}': {r['error']}")
            current.add(field)
            logger.info(f"  + Added field '{field}' to {MASTER_MODEL_NAME}")


def _update_note_in_place(mc: "MasterGenkiCard") -> None:
    note_id = int(mc.source_note.noteId)
    result  = invoke("updateNoteFields", note={"id": note_id, "fields": mc.to_anki_fields()})
    if result.get("error"):
        logger.error(f"  ✗ updateNoteFields failed for {note_id}: {result['error']}")
        return

    info = invoke("notesInfo", notes=[note_id])
    if info.get("error"):
        logger.error(f"  ✗ notesInfo failed for {note_id}: {result['error']}")
        return
    existing_tags = set(info.get("result", [{}])[0].get("tags", []))
    merged_tags   = existing_tags | set(mc.tags)
    result = invoke("updateNoteTags", note=note_id, tags=" ".join(sorted(merged_tags)))
    if result.get("error"):
        logger.error(f"  ✗ updateNoteTags failed for {note_id}: {result['error']}")
        return

    logger.info(f"  ✓ Updated {note_id}")


def _add_new_note(mc: "MasterGenkiCard") -> None:
    """Insert a brand-new note (used for Core 2000 sentence cards)."""
    result = invoke(
        "addNote",
        note={
            "deckName":  mc.target_deck,
            "modelName": MASTER_MODEL_NAME,
            "fields":    mc.to_anki_fields(),
            "tags":      mc.tags,
            # "options":   {"allowDuplicate": True},
        },
    )
    if result.get("error"):
        error_msg = result["error"]
        if "duplicate" in error_msg.lower():
            logger.info(
                f"  ~ addNote duplicate detected for parent {mc.source_note.noteId}, "
                f"searching for existing note to update"
            )
            escaped = mc.japanese.replace('"', '\\"')
            find    = invoke("findNotes", query=f'note:"{MASTER_MODEL_NAME}" Japanese:"{escaped}"')
            ids     = find.get("result") or []
            if not ids:
                logger.error(
                    f"  ✗ Could not locate duplicate note for '{mc.japanese}' "
                    f"(parent {mc.source_note.noteId})"
                )
                return
            dup_id = ids[0]
            if len(ids) > 1:
                logger.warning(f"  ~ {len(ids)} duplicates found for '{mc.japanese}', using {dup_id}")
            mc.source_note.noteId = str(dup_id)
            _update_note_in_place(mc)
        else:
            logger.error(
                f"  ✗ addNote failed for sentence of {mc.source_note.noteId}: {error_msg}"
            )
    else:
        logger.info(f"  ✓ Added new sentence note (parent {mc.source_note.noteId}) → new id {result.get('result')}")


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
            logger.warning("Missing media: %s", src)
            continue
        shutil.copy2(src, dst)
        copied += 1
    logger.info("Media: %d copied, %d already present, %d total referenced", copied, skipped, len(referenced))


def mode_export(deck_name: str, out_csv: Path, media_src: Path, media_dst: Path | None, sample: int | None) -> None:
    deck = Deck.from_anki(deck_name)
    if sample:
        deck.cards = random.sample(deck.cards, min(sample, len(deck.cards)))
    logger.info("Loaded %d cards from '%s'", len(deck.cards), deck_name)
    write_raw_csv(deck.cards, out_csv)
    if media_dst is not None:
        _copy_referenced_media(deck.cards, media_src, media_dst)


def mode_transform(in_csv: Path, out_csv: Path, media_path: Path, sample: int | None) -> None:
    already_done: set[str] = set()
    if out_csv.exists():
        with open(out_csv, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if nid := row.get("noteId"):
                    already_done.add(nid)
        logger.info("Resuming — %d noteIds already done", len(already_done))

    all_cards = read_raw_csv(in_csv)
    if sample:
        all_cards = random.sample(all_cards, min(sample, len(all_cards)))

    pending_no_llm: list[Card] = []
    pending_llm:    list[Card] = []

    for card in all_cards:
        if card.noteId in already_done:
            continue
        if card.cardType == MASTER_MODEL_NAME:
            logger.debug("Skipping %s — already master", card.noteId)
            continue
        if card.cardType in NO_LLM_CARD_TYPES:
            pending_no_llm.append(card)
        elif card.cardType in LLM_CARD_TYPES:
            pending_llm.append(card)
        else:
            logger.warning("Unknown card type '%s' for note %s — skipping", card.cardType, card.noteId)

    logger.info("%d no-LLM cards | %d LLM cards", len(pending_no_llm), len(pending_llm))

    no_llm_results: list[MasterGenkiCard] = []
    for card in pending_no_llm:
        try:
            no_llm_results.extend(transform_no_llm(card))
        except Exception as e:
            logger.warning("No-LLM transform failed for %s: %s", card.noteId, e)

    if no_llm_results:
        write_transformed_csv(no_llm_results, out_csv, append=True)
        logger.info("No-LLM: wrote %d cards", len(no_llm_results))

    requests_list: list[PromptRequest] = []
    valid_cards:   list[Card]          = []
    for card in pending_llm:
        try:
            requests_list.append(build_prompt_request_for_card(card, media_path))
            valid_cards.append(card)
        except Exception as e:
            logger.warning("Prompt build failed for %s: %s", card.noteId, e)

    total_ok = total_fail = total_partial = 0
    num_batches = (len(valid_cards) + TRANSFORM_BATCH_SIZE - 1) // TRANSFORM_BATCH_SIZE

    for batch_idx in range(num_batches):
        lo = batch_idx * TRANSFORM_BATCH_SIZE
        hi = lo + TRANSFORM_BATCH_SIZE
        batch_cards    = valid_cards[lo:hi]
        batch_requests = requests_list[lo:hi]

        logger.info("LLM batch %d/%d — %d cards", batch_idx + 1, num_batches, len(batch_cards))
        llm_outputs = prompt_batch(batch_requests)

        batch_master: list[MasterGenkiCard] = []
        for card, raw in zip(batch_cards, llm_outputs):
            trying_partial = False
            if not raw:
                logger.warning("No LLM output for %s", card.noteId)
                trying_partial = True
            try:
                batch_master.append(assemble_master_card_from_llm(card, raw))
                if trying_partial:
                    total_partial += 1
                else:
                    total_ok += 1
            except Exception as e:
                logger.warning("Assembly failed for %s: %s | raw: %.120s", card.noteId, e, raw)
                total_fail += 1

        write_transformed_csv(batch_master, out_csv, append=True)
        logger.info("Batch %d/%d done — %d ok or partial / %d failed (running: %d/%d/%d)",
                    batch_idx + 1, num_batches,
                    len(batch_master), len(batch_cards) - len(batch_master),
                    total_ok, total_partial, total_fail)

    logger.info("Transform complete — %d ok, %d failed", total_ok, total_fail)


def mode_import(in_csv: Path, dry_run: bool) -> None:
    master_cards = read_transformed_csv(in_csv)
    if dry_run:
        updates = [mc for mc in master_cards if not mc.is_new_note]
        inserts = [mc for mc in master_cards if mc.is_new_note]
        logger.info(
            f"DRY RUN — {len(master_cards)} total: "
            f"{len(updates)} updates, {len(inserts)} new inserts "
            f"(pass --no-dry-run to commit)"
        )
        for mc in master_cards[:5]:
            action = "ADD" if mc.is_new_note else "UPD"
            logger.info(f"  [{action}] {mc.source_note.noteId}: {mc.japanese[:50]}")
        return

    _ensure_model_fields(_ANKI_FIELD_NAMES)
    for mc in master_cards:
        if mc.is_new_note:
            _add_new_note(mc)
        else:
            _update_note_in_place(mc)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    p_export = sub.add_parser("export", help="AnkiConnect → CSV")
    p_export.add_argument("--deck",      default=DEFAULT_DECK)
    p_export.add_argument("--out",       default=DEFAULT_RAW_CSV, type=Path)
    p_export.add_argument("--media-src", default=DEFAULT_MEDIA_PATH, type=Path)
    p_export.add_argument("--media-out", default=None, type=Path)
    p_export.add_argument("--sample",    default=None, type=int)

    p_transform = sub.add_parser("transform", help="CSV → (LLM) → CSV")
    p_transform.add_argument("--in",     dest="in_csv",  default=DEFAULT_RAW_CSV, type=Path)
    p_transform.add_argument("--out",    dest="out_csv", default=DEFAULT_OUT_CSV, type=Path)
    p_transform.add_argument("--media",  default=DEFAULT_MEDIA_PATH, type=Path)
    p_transform.add_argument("--sample", default=None, type=int)

    p_import = sub.add_parser("import", help="transformed CSV → AnkiConnect")
    p_import.add_argument("--in",         dest="in_csv", default=DEFAULT_OUT_CSV, type=Path)
    p_import.add_argument("--no-dry-run", dest="dry_run", action="store_false", default=True)

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
