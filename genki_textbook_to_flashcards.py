"""
genki_to_anki.py
────────────────
Converts Genki textbook images or PDFs into Anki-ready CSV flashcards.

Fully automated pipeline per page:
  1. EXTRACT  — LLM sees page image + rolling 1-2 page context summaries,
                returns raw candidate cards + a summary for the next page.
  2. CRITIQUE — second LLM call re-examines the same image alongside the
                candidates, checking for hallucinations, wrong definitions,
                irrelevant text, missed items, and duplicates.
                Returns each card tagged keep/fix/add/remove WITH the
                corrected field values already written in.
  3. MERGE    — fully automatic: removes bad cards, applies corrected
                definitions/notes from fix cards, inserts add cards,
                deduplicates. Zero human input required.

Anki note fields produced:
  Japanese | Furigana | Japanese Audio | Textbook Definition | Picture | Additional Notes | Tags

Furigana format uses Anki's ruby notation: 漢字[かんじ]
Pure-kana items get an empty Furigana field (no need to echo kana).

Usage:
    python genki_to_anki.py --input genki_ch5.pdf --output ch5.csv --tag "Genki::Chapter5"
    python genki_to_anki.py --input p40.png p41.png --output cards.csv
    python genki_to_anki.py --input genki.pdf --output cards.csv --dpi 200 --context-pages 2

Requirements:
    pip install openai pymupdf attrs
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import logging
import re
import sys
import unicodedata
from pathlib import Path

from attr import define

from src.common.utils import load_image_b64, server_available
from src.prompting.prompter import PromptRequest, prompt_with_retries

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

@define
class AnkiCard:
    japanese: str
    furigana: str           # Anki ruby notation e.g. 食べ物[たべもの] — empty if pure kana
    textbook_definition: str
    picture: str
    additional_notes: str = ""
    tags: str = ""


@define
class PageResult:
    label: str
    cards: list[AnkiCard]
    context_summary: str


# ─────────────────────────────────────────────────────────────────────────────
# Kanji detection helper
# ─────────────────────────────────────────────────────────────────────────────

def _has_kanji(text: str) -> bool:
    """Return True if the string contains at least one CJK unified ideograph."""
    return any(unicodedata.category(ch) == "Lo" and "\u4e00" <= ch <= "\u9fff" for ch in text)


# ─────────────────────────────────────────────────────────────────────────────
# System prompts
# ─────────────────────────────────────────────────────────────────────────────

_EXTRACT_SYSTEM = """\
You are an expert Japanese-language flashcard creator working from scanned pages \
of the Genki textbook series.

You will receive:
  • A textbook page image (primary input).
  • Optionally, a "Rolling context" block describing what the preceding 1-2 pages \
covered. Use it to understand the chapter topic and avoid re-extracting items \
already captured from those pages.

Your task:
  Extract EVERY Japanese vocabulary item and example sentence visible on this page.

Return ONLY a valid JSON object with exactly these two keys — no prose, no fences:

{
  "cards": [
    {
      "japanese":   "<word, phrase, or full sentence exactly as printed>",
      "furigana":   "<Anki ruby notation string — see rules below>",
      "definition": "<English meaning/translation exactly as printed; translate yourself only if no gloss is shown>",
      "card_type":  "vocab" | "sentence",
      "notes":      "<grammar notes, conjugation hints, usage context visible near this item — empty string if none>"
    }
  ],
  "page_summary": "<2-4 sentence plain-English description of what this page covers, for use as context on the next page>"
}

Furigana rules:
  - Use Anki ruby notation: write each kanji run immediately followed by its \
reading in square brackets.  Example: 食べ物[たべもの] is WRONG — kanji and \
kana must be separated correctly: 食[た]べ物[もの] is WRONG too since べ is \
okurigana.  The correct form is  食べ[たべ]物[もの]  — group each kanji+okurigana \
block together before the bracket.
  - Simpler rule: wrap the SMALLEST natural reading unit.  \
食べる → 食[た]べる  (only the kanji 食 needs a bracket, okurigana べる stays outside). \
日本語 → 日本[にほん]語[ご]  (two separate kanji words). \
東京 → 東京[とうきょう]  (one compound, one bracket).
  - If the item contains NO kanji (pure hiragana or katakana), set furigana to "".
  - If you are unsure of the correct reading, set furigana to "" rather than guessing.
  - The textbook often prints furigana directly above kanji — use those readings \
exactly when visible; do not substitute your own.

Extraction rules:
  - Capture vocabulary lists, conjugation tables, grammar pattern boxes, \
dialogue lines, and example sentences.
  - Skip pure English headings, page numbers, section titles, and decorative text.
  - If the page has no Japanese text, return {"cards": [], "page_summary": "No Japanese text on this page."}.
  - Return valid JSON ONLY.
"""

_CRITIQUE_SYSTEM = """\
You are a strict quality-control reviewer for Japanese Anki flashcards built \
from Genki textbook pages.

You will receive:
  • The original page image.
  • A JSON array of candidate flashcards extracted from that page.
  • A rolling context summary of the surrounding pages.

Audit every candidate card against the page image, then return a CORRECTED \
JSON array. Your output is consumed directly by an automated pipeline — \
no human will review it — so every fix must be complete and self-contained.

Apply these checks in order:

  HALLUCINATION  — Verify the Japanese text is actually visible on the page.
                   If it is not found, set action "remove".

  CORRECTNESS    — Verify the English definition is accurate.
                   If it is wrong or imprecise, set action "fix" AND write the
                   corrected definition directly into the "definition" field.

  FURIGANA       — Verify the furigana field for every card that contains kanji.
                   Check the readings visible on the page (printed furigana, \
                   vocabulary lists, etc.).
                   Fix errors using Anki ruby notation: 食[た]べる, 東京[とうきょう].
                   Set furigana to "" for pure-kana items.
                   Set action "fix" if you change furigana (even if definition was fine).

  NOTES FIX      — If the notes field is missing useful grammar/context visible
                   on the page, add it and set action "fix".

  RELEVANCE      — Remove pure meta-text: exercise labels, "Answer Key",
                   "Listen and repeat", section headers, etc.

  COMPLETENESS   — Scan the image for Japanese text not in the candidate list.
                   Add missed items with action "add", filling ALL fields.

  DUPLICATES     — Keep the most complete version; remove the rest.

Return ONLY a valid JSON array — no fences, no prose:

[
  {
    "japanese":   "<exactly as on the page>",
    "furigana":   "<Anki ruby notation, or empty string for pure kana>",
    "definition": "<correct English — YOUR corrected value if action is fix>",
    "card_type":  "vocab" | "sentence",
    "notes":      "<complete, corrected notes>",
    "action":     "keep" | "fix" | "add" | "remove",
    "reason":     "<one sentence explaining non-keep decisions; empty for keep>"
  }
]
"""


# ─────────────────────────────────────────────────────────────────────────────
# JSON helpers
# ─────────────────────────────────────────────────────────────────────────────

def _strip_fences(raw: str) -> str:
    return re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()


def _parse_extract_response(raw: str) -> tuple[list[dict], str]:
    data = json.loads(_strip_fences(raw))
    if not isinstance(data, dict):
        raise ValueError("Extractor must return a JSON object, got list/other")
    cards = data.get("cards", [])
    summary = str(data.get("page_summary", ""))
    if not isinstance(cards, list):
        raise ValueError("'cards' key must be a list")
    return cards, summary


def _parse_critique_response(raw: str) -> list[dict]:
    data = json.loads(_strip_fences(raw))
    if not isinstance(data, list):
        raise ValueError("Critiquer must return a JSON array")
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Image / PDF loading
# ─────────────────────────────────────────────────────────────────────────────

def load_pages(
    paths: list[Path],
    dpi: int = 150,
    png_out_dir: Path | None = None,
) -> list[tuple[str, str, str]]:
    """
    Returns [(base64_data, mime_type, page_label), ...].

    For PDF inputs, each page is rendered to a PNG and saved to `png_out_dir`
    (defaults to the same directory as the PDF).  The saved PNG filename matches
    the page_label used in the Anki <img> tags, so Anki can find the file once
    you copy it into your collection.media folder.
    """
    pages: list[tuple[str, str, str]] = []
    for src in paths:
        if src.suffix.lower() == ".pdf":
            try:
                import fitz
            except ImportError:
                sys.exit("PyMuPDF required for PDF input: pip install pymupdf")

            out_dir = png_out_dir or src.parent
            out_dir.mkdir(parents=True, exist_ok=True)

            doc = fitz.open(str(src))
            for page_num, page in enumerate(doc, start=1):
                label    = f"{src.stem}_p{page_num:03d}"
                png_path = out_dir / f"{label}.png"

                if png_path.exists():
                    logger.info("  Skipping existing page image → %s", png_path)
                else:
                    mat = fitz.Matrix(dpi / 72, dpi / 72)
                    pix = page.get_pixmap(matrix=mat)
                    pix.save(str(png_path))
                    logger.info("  Saved page image → %s", png_path)

                b64 = base64.b64encode(png_path.read_bytes()).decode()
                pages.append((b64, "image/png", label))
            doc.close()
        else:
            b64, mime = load_image_b64(src)
            pages.append((b64, mime, src.stem))
    return pages


# ─────────────────────────────────────────────────────────────────────────────
# Rolling context window
# ─────────────────────────────────────────────────────────────────────────────

def _build_context_block(summaries: list[str]) -> str:
    if not summaries:
        return ""
    lines = ["Rolling context (preceding pages):"]
    for i, s in enumerate(summaries):
        ago = len(summaries) - i
        lines.append(f"  {ago} page(s) ago: {s}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# LLM calls
# ─────────────────────────────────────────────────────────────────────────────

def _run_extraction(b64: str, mime: str, context_block: str) -> tuple[list[dict], str]:
    user_text = "Extract all Japanese items from this textbook page."
    if context_block:
        user_text = f"{context_block}\n\n{user_text}"
    req = PromptRequest(
        system_prompt=_EXTRACT_SYSTEM,
        user_prompt=user_text,
        base64_encoded_image=b64,
        image_mime=mime,
        max_retries=4,
        validator=_parse_extract_response,
    )
    raw = prompt_with_retries(req)
    return _parse_extract_response(raw)


def _run_critique(
    b64: str,
    mime: str,
    label: str,
    candidate_cards: list[dict],
    context_block: str,
) -> list[dict]:
    cards_json = json.dumps(candidate_cards, ensure_ascii=False, indent=2)
    user_text = (
        f"Page label: {label}\n\n"
        + (f"{context_block}\n\n" if context_block else "")
        + f"Candidate cards:\n{cards_json}\n\n"
        "Audit every card against the page image. "
        "For any 'fix' card, write the corrected values directly into 'definition', 'furigana', and 'notes'. "
        "Return the complete corrected JSON array."
    )
    req = PromptRequest(
        system_prompt=_CRITIQUE_SYSTEM,
        user_prompt=user_text,
        base64_encoded_image=b64,
        image_mime=mime,
        max_retries=4,
        validator=_parse_critique_response,
    )
    raw = prompt_with_retries(req)
    return _parse_critique_response(raw)


# ─────────────────────────────────────────────────────────────────────────────
# Merge — fully automatic, no human input
# ─────────────────────────────────────────────────────────────────────────────

def _merge_cards(audited: list[dict], label: str) -> list[dict]:
    """
    Apply the critique's decisions automatically:
      remove  → dropped entirely
      fix     → kept with the critique's corrected definition/furigana/notes/card_type
      add     → inserted with all fields from the critique
      keep    → passed through unchanged
    Duplicates (same japanese string) → only the first surviving occurrence kept.
    """
    kept: list[dict] = []
    seen: set[str] = set()
    counts = {"keep": 0, "fix": 0, "add": 0, "remove": 0, "dedup": 0}

    for card in audited:
        action   = str(card.get("action", "keep")).strip().lower()
        japanese = str(card.get("japanese", "")).strip()
        reason   = card.get("reason", "")

        if action == "remove":
            logger.info("  ✗ REMOVE  %-40s  %s", japanese, reason)
            counts["remove"] += 1
            continue

        if not japanese:
            logger.warning("  ? SKIP empty japanese field: %s", card)
            continue

        if japanese in seen:
            logger.info("  ~ DEDUP   %s", japanese)
            counts["dedup"] += 1
            continue
        seen.add(japanese)

        if action == "fix":
            logger.info("  ✎ FIX     %-40s  %s", japanese, reason)
            counts["fix"] += 1
        elif action == "add":
            logger.info("  + ADD     %-40s  %s", japanese, reason)
            counts["add"] += 1
        else:
            counts["keep"] += 1

        # Sanitise furigana: if the item has no kanji, blank it out regardless
        # of what the LLM returned (prevents spurious kana echoes).
        raw_furigana = str(card.get("furigana", "")).strip()
        furigana = raw_furigana if _has_kanji(japanese) else ""

        clean = {
            "japanese":   japanese,
            "furigana":   furigana,
            "definition": str(card.get("definition", "")).strip(),
            "card_type":  str(card.get("card_type", "vocab")).strip(),
            "notes":      str(card.get("notes", "")).strip(),
        }
        kept.append(clean)

    logger.info(
        "  [%s] keep=%d  fix=%d  add=%d  remove=%d  dedup=%d  → %d final",
        label, counts["keep"], counts["fix"], counts["add"],
        counts["remove"], counts["dedup"], len(kept),
    )
    return kept


# ─────────────────────────────────────────────────────────────────────────────
# Per-page orchestration
# ─────────────────────────────────────────────────────────────────────────────

def process_page(
    b64: str,
    mime: str,
    label: str,
    context_summaries: list[str],
    base_tag: str,
    context_pages: int = 2,
) -> PageResult:
    recent = context_summaries[-context_pages:] if context_pages > 0 else []
    context_block = _build_context_block(recent)
    pic_ref  = f'<img src="{label}.png">'
    page_tag = label.replace(" ", "_")

    # ── Step 1: Extract ──────────────────────────────────────────────────────
    logger.info("[%s] 1/2 extracting...", label)
    try:
        candidate_cards, page_summary = _run_extraction(b64, mime, context_block)
        logger.info("[%s]   %d candidates", label, len(candidate_cards))
    except Exception as exc:
        logger.error("[%s] Extraction failed: %s", label, exc)
        return PageResult(label=label, cards=[], context_summary="Extraction failed.")

    if not candidate_cards:
        return PageResult(label=label, cards=[], context_summary=page_summary or "Empty page.")

    # ── Step 2: Critique ─────────────────────────────────────────────────────
    logger.info("[%s] 2/2 critiquing %d cards...", label, len(candidate_cards))
    try:
        audited_cards = _run_critique(b64, mime, label, candidate_cards, context_block)
    except Exception as exc:
        logger.warning("[%s] Critique failed (%s) — using raw extraction.", label, exc)
        audited_cards = [{**c, "action": "keep", "reason": ""} for c in candidate_cards]

    # ── Step 3: Merge (fully automatic) ──────────────────────────────────────
    final_dicts = _merge_cards(audited_cards, label)

    # ── Assemble AnkiCard objects ─────────────────────────────────────────────
    anki_cards: list[AnkiCard] = []
    for item in final_dicts:
        japanese   = item["japanese"]
        definition = item["definition"]
        if not japanese or not definition:
            logger.warning("[%s]   Skipping card with empty japanese or definition: %s", label, item)
            continue
        tags = " ".join([base_tag, f"page::{page_tag}", f"type::{item['card_type']}"])
        anki_cards.append(AnkiCard(
            japanese=japanese,
            furigana=item["furigana"],
            textbook_definition=definition,
            picture=pic_ref,
            additional_notes=item["notes"],
            tags=tags,
        ))

    return PageResult(label=label, cards=anki_cards, context_summary=page_summary)


# ─────────────────────────────────────────────────────────────────────────────
# CSV export
# ─────────────────────────────────────────────────────────────────────────────

CSV_FIELDS = [
    "Japanese",
    "Furigana",
    "Japanese Audio",
    "Textbook Definition",
    "Picture",
    "Additional Notes",
    "Tags",
]


def write_csv(cards: list[AnkiCard], out_path: Path) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_FIELDS)
        for c in cards:
            writer.writerow([
                c.japanese,
                c.furigana,
                "",                       # Japanese Audio — filled later by TTS pipeline
                c.textbook_definition,
                c.picture,
                c.additional_notes,
                c.tags,
            ])
    print(f"\nWrote {len(cards)} cards → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Convert Genki pages to Anki CSV")
    parser.add_argument("--input",  nargs="+", required=True,
                        help="Image files (.png/.jpg) or PDF(s) to process")
    parser.add_argument("--output", default="genki_cards.csv",
                        help="Output CSV path (default: genki_cards.csv)")
    parser.add_argument("--tag",    default="Genki",
                        help="Base Anki tag, e.g. 'Genki::Chapter5' (default: Genki)")
    parser.add_argument("--dpi",    type=int, default=150,
                        help="DPI for PDF→image rendering (default: 150)")
    parser.add_argument("--png-dir", default=None,
                        help="Where to save extracted page PNGs (default: same folder as the PDF). "
                             "Copy these into your Anki collection.media folder so <img> tags resolve.")
    parser.add_argument("--context-pages", type=int, default=2,
                        help="How many prior page summaries to pass as context (default: 2)")
    args = parser.parse_args()

    input_paths = [Path(p) for p in args.input]
    for p in input_paths:
        if not p.exists():
            sys.exit(f"File not found: {p}")
    
    if not server_available():
        print("Server not found. Check port")
        sys.exit(1)
    
    png_out_dir = Path(args.png_dir) if args.png_dir else None

    print("Loading input file(s)...")
    pages = load_pages(input_paths, dpi=args.dpi, png_out_dir=png_out_dir)
    print(f"  → {len(pages)} page(s) to process")
    if png_out_dir:
        print(f"  Page PNGs saved to: {png_out_dir}")
    print()

    all_cards: list[AnkiCard] = []
    context_summaries: list[str] = []
    out_path = Path(args.output)
    FLUSH_EVERY = 100   # write a checkpoint CSV after every N cards

    for i, (b64, mime, label) in enumerate(pages, start=1):
        print(f"── Page {i}/{len(pages)}: {label}")
        result = process_page(
            b64, mime, label,
            context_summaries=context_summaries,
            base_tag=args.tag,
            context_pages=args.context_pages,
        )
        all_cards.extend(result.cards)
        if result.context_summary:
            context_summaries.append(result.context_summary)
        print(f"   → {len(result.cards)} cards  |  running total: {len(all_cards)}\n")

        # Periodic checkpoint flush
        if len(all_cards) // FLUSH_EVERY > (len(all_cards) - len(result.cards)) // FLUSH_EVERY:
            write_csv(all_cards, out_path)
            print(f"  [checkpoint] flushed {len(all_cards)} cards → {out_path}\n")

    write_csv(all_cards, out_path)

    print("\nFirst 5 cards preview:")
    for c in all_cards[:5]:
        print(f"  {c.japanese!r:<35}  →  {c.textbook_definition!r}")
        if c.furigana:
            print(f"    振  {c.furigana}")
        if c.additional_notes:
            print(f"    ℹ  {c.additional_notes}")
        print(f"    🏷  {c.tags}")


if __name__ == "__main__":
    main()
