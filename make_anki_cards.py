"""
Anki deck generator — integrated pipeline.

Concurrency model:
  - Per file, translation (natural + literal) and sense-selection run in
    parallel thread pools (both hit the vLLM server; GIL is released during
    HTTP I/O).
  - TTS, audio extraction, and frame extraction run concurrently in a second
    thread pool during the card-generation pass.
  - Fish Audio S2 Pro is called via its local HTTP server (tools/api_server.py).

Deck structure:
  - A parent deck "Japanese Video Deck" is created (or reused across runs).
  - Each video gets its own subdeck: "Japanese Video Deck::<video stem>".
  - Within each subdeck cards are ordered chronologically:
      sentence card  →  vocab card(s) for new words introduced in that sentence
  - Vocab cards: JP word on front (cloned-voice TTS), EN definition on back
    (neutral Kokoro TTS).  Words are de-duplicated across the whole video so
    each word appears only once, anchored to the earliest sentence that
    introduced it.
"""
import argparse
import os
import subprocess
import sys
import tempfile
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import fugashi
import genanki
import unidic_lite

from anki.fonts_helper import FONTS_DIR, font_for_index, pos_to_class
from anki.models import SENTENCE_MODEL, VOCAB_MODEL
from parser.jamdict import jamdict_senses_for
from parser.jlpt import jlpt_for_result
from parser.vtt import parse_vtt
from src.common.types import RawChunk, FullContext, Sense, SenseResult, CompletedChunk
from src.common.utils import server_available
from src.prompting.translator import translate_sentence as _translate_sentence
from src.prompting.sense_selector import select_senses
from src.tts.tts import TTSBackend
from src.tts.factory import build_tts_pair
from src.parser.jlpt import pitch_html as _pitch_html


_SKIP_POS = {"助詞", "助動詞", "記号", "補助記号", "空白"}
TAGGER = fugashi.Tagger(f"-d {unidic_lite.DICDIR}")


def _has_kanji(s: str) -> bool:
    return any("\u4e00" <= c <= "\u9fff" or "\u3400" <= c <= "\u4dbf" for c in s)


def _kana_to_hiragana(s: str) -> str:
    return "".join(
        chr(ord(c) - 0x60) if "\u30a1" <= c <= "\u30f6" else c for c in s
    )

MAX_ENTRIES           = 10
MAX_SENSES            = 10
MAX_GLOSSES_PER_SENSE = 2
CONTEXT_PREV_WINDOW   = 100
CONTEXT_NEXT_WINDOW   = 1

LLM_WORKERS   = 8
MEDIA_WORKERS = 4

TO_PROCESS_DIR = Path("./to_process")
if not TO_PROCESS_DIR.exists() or not TO_PROCESS_DIR.is_dir():
    raise ValueError(f"Please ensure {TO_PROCESS_DIR} exists and is a directory")
PROCESSED_DIR = Path("./processed")
PROCESSED_DIR.mkdir(exist_ok=True)
if not PROCESSED_DIR.is_dir():
    raise ValueError(f"Please make sure {PROCESSED_DIR} is a dir")


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
    fugashi is not thread-safe, so this must be called sequentially.
    """
    ruby_parts: list[str] = []
    words_and_senses: list[tuple[str, list[Sense]]] = []
    word_meta: dict[str, dict] = {}
    seen_gloss: set[str] = set()

    if TAGGER is None:
        return _TokenisedChunk(chunk, context, [chunk.subtitle_text], [], {})

    for word in TAGGER(chunk.subtitle_text):
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
        senses, result = jamdict_senses_for(surface, lemma)
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
        jlpt         = jlpt_for_result(result) if result else ""
        pitch_html   = _pitch_html(_kana_to_hiragana(reading_kata) if reading_kata else "", accent_type)

        jlpt_span  = f'<span class="jlpt-badge jlpt-{jlpt}">{jlpt}</span>' if jlpt else ""
        pitch_span = f'<span class="pa-container">{pitch_html}</span>'       if pitch_html else ""
        header     = f'<span class="gloss-word">{surface}</span>{jlpt_span}{pitch_span}'
        lines      = [header]
        for i, meaning in enumerate(display_meanings, 1):
            pos_str   = display_poses[i - 1]
            pos_class = pos_to_class(pos_str)
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


# ── TTS helpers ───────────────────────────────────────────────────────────────

# Built once per run in main(); None until then.
_jp_tts:  TTSBackend | None = None
_en_tts:  TTSBackend | None = None


def setup_tts(jp_backend: str, en_backend: str) -> bool:
    global _jp_tts, _en_tts
    try:
        _jp_tts, _en_tts = build_tts_pair(jp_backend, en_backend, media_workers=MEDIA_WORKERS)
        return True
    except RuntimeError as e:
        print(f"  TTS setup failed: {e}")
        return False


def generate_jp_tts(text: str, out_path: str) -> bool:
    if _jp_tts is None or not text.strip():
        return False
    return _jp_tts.generate(text, out_path)


def generate_en_tts(text: str, out_path: str) -> bool:
    if _en_tts is None or not text.strip():
        return False
    return _en_tts.generate(text, out_path)


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


# ── Vocab card helpers ────────────────────────────────────────────────────────

# @dataclass
# class VocabCard:
#     """All data needed to build one vocab Anki note."""
#     word: str
#     reading: str          # hiragana
#     meaning_plain: str    # clean English string — used for EN TTS + display
#     gloss_html: str       # full HTML gloss block (same style as sentence back)
#     jlpt: str
#     font_name: str
#     source: str           # video filename


# @dataclass
# class _WordAccumulator:
#     """
#     Accumulates every distinct sense selected for one word across the whole
#     video, so the final vocab card reflects all usages, not just the first.
#     """
#     word: str
#     first_chunk_index: int          # card is anchored here for ordering
#     meta: dict                      # reading_kata, accent_type, result
#     # Each entry is (meaning, pos) — kept ordered, deduped by this set:
#     seen_sense_keys: set[tuple[str, str]] = field(default_factory=set)
#     meanings: list[str]             = field(default_factory=list)
#     poses:    list[str]             = field(default_factory=list)


# def _accumulate_senses(
#     tc: _TokenisedChunk,
#     sense_results: list[SenseResult],
#     accumulator: dict[str, _WordAccumulator],
# ) -> None:
#     """
#     For every word in this chunk's sense results, add any new (meaning, pos)
#     pairs to the accumulator.  Records the first chunk index where the word
#     appeared.  Call this once per chunk in forward order.
#     """
#     sense_map = {sr.word: sr for sr in sense_results}

#     for surface, _ in tc.words_and_senses:
#         sr   = sense_map.get(surface)
#         meta = tc.word_meta.get(surface, {})
#         if sr is None:
#             continue

#         if sr.custom_definition:
#             new_meanings = [sr.custom_definition]
#             new_poses    = ["custom"]
#         elif sr.selected:
#             new_meanings = [s.meaning for s in sr.selected]
#             new_poses    = [s.pos     for s in sr.selected]
#         else:
#             continue

#         if surface not in accumulator:
#             accumulator[surface] = _WordAccumulator(
#                 word=surface,
#                 first_chunk_index=tc.chunk.index,
#                 meta=meta,
#             )

#         acc = accumulator[surface]
#         for meaning, pos in zip(new_meanings, new_poses):
#             key = (meaning, pos)
#             if key not in acc.seen_sense_keys:
#                 acc.seen_sense_keys.add(key)
#                 acc.meanings.append(meaning)
#                 acc.poses.append(pos)


# def _build_vocab_card(acc: _WordAccumulator, source_name: str) -> VocabCard:
#     """Render a VocabCard from a fully-populated _WordAccumulator."""
#     reading_kata = acc.meta.get("reading_kata", "")
#     reading_hira = _kana_to_hiragana(reading_kata) if reading_kata else ""
#     accent_type  = acc.meta.get("accent_type", "")
#     result       = acc.meta.get("result")
#     jlpt         = jlpt_for_result(result) if result else ""
#     pitch_html   = _pitch_html(reading_hira, accent_type)

#     # Plain meaning string for TTS — no HTML, numbers prefix each sense
#     meaning_plain = "; ".join(
#         f"{acc.meanings[i]} ({acc.poses[i]})" for i in range(len(acc.meanings))
#     )

#     # HTML gloss — same structure as sentence card back
#     jlpt_span  = f'<span class="jlpt-badge jlpt-{jlpt}">{jlpt}</span>' if jlpt else ""
#     pitch_span = f'<span class="pa-container">{pitch_html}</span>'       if pitch_html else ""
#     header     = f'<span class="gloss-word">{acc.word}</span>{jlpt_span}{pitch_span}'
#     lines      = [header]
#     for i, meaning in enumerate(acc.meanings, 1):
#         pos_class = pos_to_class(acc.poses[i - 1])
#         lines.append(
#             f'<span class="{pos_class} gloss-line">'
#             f'{i}) {meaning}'
#             f'<span class="pos-tag"> [{acc.poses[i-1]}]</span>'
#             f'</span>'
#         )
#     gloss_html = "<br>".join(lines)

#     return VocabCard(
#         word=acc.word,
#         reading=reading_hira,
#         meaning_plain=meaning_plain,
#         gloss_html=gloss_html,
#         jlpt=jlpt,
#         font_name=font_for_index(acc.first_chunk_index),
#         source=source_name,
#     )


# ── Core processing ───────────────────────────────────────────────────────────

def process_file(
    media_path: Path,
    vtt_path: Path,
    subdeck: genanki.Deck,
    all_media_files: list[str],
    tmpdir: str,
    frame_offset: float,
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
    print(f"    Tokenising {n} chunks ...")
    contexts: list[FullContext] = [
        build_full_context(i, raw_chunks) for i in range(n)
    ]
    tokenised: list[_TokenisedChunk] = [
        _tokenise_chunk(raw_chunks[i], contexts[i]) for i in range(n)
    ]

    # ── Stage 2 (parallel): natural + literal translation AND sense selection ─
    print(f"    Running translation + sense selection in parallel ({LLM_WORKERS} workers) ...")

    natural_translations: list[str]              = [""] * n
    literal_translations: list[str]              = [""] * n
    sense_results:        list[list[SenseResult]] = [[] for _ in range(n)]

    def _do_natural_translation(i: int) -> tuple[int, str, str]:
        result = _translate_sentence(
            context=contexts[i],
            base64_encoded_image=None,
            image_mime="image/jpeg",
            translation_type="natural",
        )
        return i, "natural", result

    def _do_literal_translation(i: int) -> tuple[int, str, str]:
        result = _translate_sentence(
            context=contexts[i],
            base64_encoded_image=None,
            image_mime="image/jpeg",
            translation_type="literal",
        )
        return i, "literal", result

    def _do_sense_selection(i: int) -> tuple[int, str, list[SenseResult]]:
        tc     = tokenised[i]
        result = select_senses(tc.context, tc.words_and_senses)
        return i, "sense", result

    with ThreadPoolExecutor(max_workers=LLM_WORKERS) as pool:
        futures: dict = {}
        for i in range(n):
            futures[pool.submit(_do_natural_translation, i)] = i
            futures[pool.submit(_do_literal_translation, i)] = i
            futures[pool.submit(_do_sense_selection, i)]     = i

        completed_count = 0
        total_tasks     = n * 3
        for future in as_completed(futures):
            result_tuple = future.result()
            idx, kind    = result_tuple[0], result_tuple[1]
            payload      = result_tuple[2]
            if kind == "natural":
                natural_translations[idx] = payload
            elif kind == "literal":
                literal_translations[idx] = payload
            else:
                sense_results[idx] = payload
            completed_count += 1
            if completed_count % 10 == 0:
                print(f"      {completed_count}/{total_tasks} LLM tasks done ...")

    # ── Stage 3 (main thread, sequential): render gloss HTML + accumulate vocab ─
    # Full forward pass: accumulate every sense selected for each word across
    # the entire video before building vocab cards.  This ensures that if 行く
    # appears in chunk 2 as "to go" and again in chunk 47 as "to leave", the
    # single vocab card (anchored at chunk 2) shows both senses.
    print(f"    Rendering gloss HTML and collecting vocab ...")

    @dataclass
    class _ChunkData:
        completed: CompletedChunk

    # word_accumulator: dict[str, _WordAccumulator] = {}
    chunk_data: list[_ChunkData] = []

    for i, tc in enumerate(tokenised):
        furigana, word_gloss = _render_gloss_html(tc, sense_results[i])
        natural = natural_translations[i]
        literal = literal_translations[i]

        print(f"      [{i+1}/{n}] {tc.chunk.subtitle_text[:45]}")
        if natural:
            print(f"        → {natural[:65]}")

        completed_chunk = CompletedChunk(
            index=tc.chunk.index,
            start=tc.chunk.start,
            end=tc.chunk.end,
            subtitle_text=tc.chunk.subtitle_text,
            natural_translation=natural,
            literal_translation=literal,
            word_gloss=sense_results[i],
            furigana=furigana,
        )

        # Accumulate senses for every word in this chunk (no early exit on
        # already-seen words — a word may carry a new sense later in the video)
        # _accumulate_senses(tc, sense_results[i], word_accumulator)

        chunk_data.append(_ChunkData(completed=completed_chunk))

    # Build one VocabCard per word with all senses seen across the video.
    # vocab_cards_by_chunk maps first_chunk_index → cards so stage 5 can insert
    # each vocab card immediately after the sentence that introduced the word.

    # vocab_cards_by_chunk: dict[int, list[VocabCard]] = {}
    # for acc in word_accumulator.values():
    #     vocab_cards_by_chunk.setdefault(acc.first_chunk_index, []).append(
    #         _build_vocab_card(acc, original_name)
    #     )
    # total_vocab = sum(len(v) for v in vocab_cards_by_chunk.values())
    # print(f"    {total_vocab} unique vocab cards across {n} chunks.")

    # ── Stage 4 (parallel): all media generation ──────────────────────────────
    # Per sentence: source audio clip, video frame, EN TTS of natural translation.
    # Per vocab card: JP TTS of word (front), EN TTS of meaning (back).
    # All are I/O-bound and fully independent.
    print(f"    Generating media in parallel ({MEDIA_WORKERS} workers) ...")

    @dataclass
    class _SentenceMedia:
        audio_fname: str = ""
        image_fname: str = ""
        tts_fname:   str = ""
        audio_ok:    bool = False
        image_ok:    bool = False
        tts_ok:      bool = False

    @dataclass
    class _VocabMedia:
        word_tts_fname:    str  = ""
        meaning_tts_fname: str  = ""
        word_tts_ok:       bool = False
        meaning_tts_ok:    bool = False

    sentence_media: dict[int, _SentenceMedia] = {}
    # keyed by word string (one entry per unique word across whole video)
    vocab_media: dict[str, _VocabMedia] = {}

    def _process_sentence_media(cd: _ChunkData) -> tuple[int, _SentenceMedia]:
        chunk     = cd.completed
        idx       = chunk.index
        card_uuid = uuid.uuid4().hex
        media     = _SentenceMedia(
            audio_fname=f"clip_{idx:04d}_{card_uuid}.mp3",
            image_fname=f"frame_{idx:04d}_{card_uuid}.jpg",
            tts_fname=f"tts_sent_{idx:04d}_{card_uuid}.mp3",
        )
        audio_path = os.path.join(tmpdir, media.audio_fname)
        image_path = os.path.join(tmpdir, media.image_fname)
        tts_path   = os.path.join(tmpdir, media.tts_fname)

        media.audio_ok = extract_audio(str(safe_media), chunk.start, chunk.end, audio_path)
        if not is_mp3:
            frame_ts = chunk.start + min(frame_offset, (chunk.end - chunk.start) * 0.5)
            media.image_ok = extract_frame(str(safe_media), frame_ts, image_path)
        if chunk.natural_translation:
            # EN TTS for the natural translation on the sentence card back
            media.tts_ok = generate_en_tts(chunk.natural_translation, tts_path)

        return idx, media

    # def _process_vocab_media(vc: VocabCard) -> tuple[str, _VocabMedia]:
    #     card_uuid = uuid.uuid4().hex
    #     media     = _VocabMedia(
    #         word_tts_fname=f"tts_vocab_jp_{card_uuid}.mp3",
    #         meaning_tts_fname=f"tts_vocab_en_{card_uuid}.mp3",
    #     )
    #     word_tts_path    = os.path.join(tmpdir, media.word_tts_fname)
    #     meaning_tts_path = os.path.join(tmpdir, media.meaning_tts_fname)

    #     # JP TTS: just the word itself
    #     media.word_tts_ok    = generate_jp_tts(vc.word, word_tts_path)
    #     # EN TTS: the plain meaning string (all senses, no HTML)
    #     media.meaning_tts_ok = generate_en_tts(vc.meaning_plain, meaning_tts_path)

    #     return vc.word, media

    # Collect all unique vocab cards (one per word, already built above)
    # all_vocab_cards_flat: list[VocabCard] = [
    #     vc for vcs in vocab_cards_by_chunk.values() for vc in vcs
    # ]

    with ThreadPoolExecutor(max_workers=MEDIA_WORKERS) as pool:
        sent_futures  = {pool.submit(_process_sentence_media, cd): cd.completed.index for cd in chunk_data}
        # One TTS job per unique word — keyed by word string
        # vocab_futures = {
        #     pool.submit(_process_vocab_media, vc): vc.word
        #     for vc in all_vocab_cards_flat
        # }
        # all_futures = {**sent_futures, **vocab_futures}
        all_futures = sent_futures

        for future in as_completed(all_futures):
            if future in sent_futures:
                idx, media = future.result()
                sentence_media[idx] = media
                chunk = chunk_data[idx].completed
                print(f"      [sent {idx+1}/{n}] {chunk.start:.1f}s–{chunk.end:.1f}s  {chunk.subtitle_text[:35]}")
                if not media.audio_ok:                print("        Warning: audio extraction failed")
                if not media.image_ok and not is_mp3: print("        Warning: frame extraction failed")
                if not media.tts_ok:                  print("        Warning: sentence TTS failed")
            else:
                raise ValueError("Not possible")
            # else:
            #     word, media = future.result()
            #     vocab_media[word] = media
            #     if not media.word_tts_ok:    print(f"        Warning: JP TTS failed for vocab word '{word}'")
            #     if not media.meaning_tts_ok: print(f"        Warning: EN TTS failed for vocab word '{word}'")

    # ── Stage 5: assemble Anki notes in chronological order ──────────────────
    # For each chunk: sentence note first, then vocab notes for new words.
    print(f"    Assembling notes in order ...")
    for cd in chunk_data:
        chunk      = cd.completed
        idx        = chunk.index
        s_media    = sentence_media.get(idx, _SentenceMedia())
        timecode   = f"{int(chunk.start // 60):02d}:{chunk.start % 60:05.2f}"

        # Sentence note
        sentence_note = genanki.Note(
            model=SENTENCE_MODEL,
            fields=[
                chunk.subtitle_text,
                f"[sound:{s_media.audio_fname}]"     if s_media.audio_ok else "",
                f'<img src="{s_media.image_fname}">' if s_media.image_ok else "",
                chunk.natural_translation,
                chunk.literal_translation,
                f"[sound:{s_media.tts_fname}]"       if s_media.tts_ok   else "",
                chunk.furigana,
                # word_gloss field is HTML rendered in stage 3
                _render_gloss_html(tokenised[idx], chunk.word_gloss)[1],
                timecode,
                font_for_index(idx),
                original_name,
                "",
            ],
        )
        subdeck.add_note(sentence_note)

        if s_media.audio_ok: all_media_files.append(os.path.join(tmpdir, s_media.audio_fname))
        if s_media.image_ok: all_media_files.append(os.path.join(tmpdir, s_media.image_fname))
        if s_media.tts_ok:   all_media_files.append(os.path.join(tmpdir, s_media.tts_fname))

        # Vocab notes immediately after — words introduced for the first time in this chunk
        # for vc in vocab_cards_by_chunk.get(idx, []):
        #     v_media = vocab_media.get(vc.word, _VocabMedia())

        #     vocab_note = genanki.Note(
        #         model=VOCAB_MODEL,
        #         fields=[
        #             vc.word,
        #             f"[sound:{v_media.word_tts_fname}]"    if v_media.word_tts_ok    else "",
        #             vc.reading,
        #             vc.meaning_plain,
        #             f"[sound:{v_media.meaning_tts_fname}]" if v_media.meaning_tts_ok else "",
        #             vc.gloss_html,
        #             vc.jlpt,
        #             vc.font_name,
        #             vc.source,
        #         ],
        #     )
        #     subdeck.add_note(vocab_note)

        #     if v_media.word_tts_ok:    all_media_files.append(os.path.join(tmpdir, v_media.word_tts_fname))
        #     if v_media.meaning_tts_ok: all_media_files.append(os.path.join(tmpdir, v_media.meaning_tts_fname))

    safe_media.unlink(missing_ok=True)
    safe_vtt.unlink(missing_ok=True)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Batch Anki deck generator from to_process/ directory")
    parser.add_argument("--frame-offset", type=float, default=1.0,
                        help="Seconds after chunk start to grab frame (default: 1.0)")
    parser.add_argument("--deck-name", default="Japanese Video Deck",
                        help="Name of the parent Anki deck (default: 'Japanese Video Deck')")
    parser.add_argument("--jp-tts", default="kokoro",
                        help="JP TTS backend: 'fish' or 'kokoro' (default: kokoro)")
    parser.add_argument("--en-tts", default="kokoro",
                        help="EN TTS backend: 'fish' or 'kokoro' (default: kokoro)")
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

    if not setup_tts(args.jp_tts, args.en_tts):
        raise ValueError("Failed to set up TTS")

    font_files   = list(FONTS_DIR.glob("*.ttf")) if FONTS_DIR.exists() else []
    already_done = already_processed(PROCESSED_DIR)

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, (media_path, vtt_path) in enumerate(pairs):
            if media_path.name in already_done:
                print(f"{media_path.name} already processed. Skipping.")
                continue
            already_done.add(media_path.name)

            # Each video gets its own subdeck under the parent.
            video_title  = media_path.stem
            subdeck_name = f"{args.deck_name}::{video_title}"
            # Use a deterministic deck ID derived from the subdeck name so
            # re-running the same video doesn't create duplicate decks.
            subdeck_id   = abs(hash(subdeck_name)) % (10 ** 10)
            subdeck      = genanki.Deck(subdeck_id, subdeck_name)

            media_files = [str(f) for f in font_files]
            output      = f"output_{media_path.stem}.apkg"

            print(f"\n{'='*60}")
            print(f"Processing ({i+1}/{len(pairs)}): {media_path.name}")
            print(f"  Subdeck: {subdeck_name}")
            try:
                process_file(
                    media_path, vtt_path,
                    subdeck, media_files, tmpdir,
                    args.frame_offset,
                )
                print(f"\n{'='*60}")
                print(f"Writing {output} ...")
                pkg = genanki.Package(subdeck)
                pkg.media_files = media_files
                pkg.write_to_file(output)

                shutil.move(str(media_path), PROCESSED_DIR / media_path.name)
                shutil.move(str(vtt_path),   PROCESSED_DIR / vtt_path.name)
                print(f"  Moved source files to {PROCESSED_DIR}/")
                print(f"  Done: {output}")
            except Exception as e:
                print(f"  ERROR processing {media_path.name}: {e}")
                import traceback; traceback.print_exc()


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


if __name__ == "__main__":
    main()
