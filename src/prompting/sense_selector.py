import logging
import re

from src.common.types import FullContext, Sense, SenseResult
from src.prompting.prompter import prompt_with_retries

logger = logging.getLogger(__name__)


_SELECT_SYSTEM = (
    "You are a Japanese dictionary assistant helping choose the most relevant definitions "
    "for words appearing in a specific context. "
    "Rules:\n"
    "- You will receive a Japanese sentence and one or more words with numbered definitions.\n"
    "- For each word, return the numbers of the best definitions given the context.\n"
    "- Return a max of three definitions - no more. Ideally, only return the most relevant one or two, but "
    "three may be returned if necessary."
    "- If NONE of the provided definitions fit the word as used in this sentence, respond with "
    "'NONE' for that word instead of a number.\n"
    "- Output ONLY a structured response, no explanations.\n"
    "- For a single word: respond with comma-separated numbers, or 'NONE'. Example: 1,3\n"
    "- For multiple words: respond with one line per word in the format 'word: 1,3' or 'word: NONE'. "
    "Example:\n"
    "  食べる: 1\n"
    "  行く: NONE\n"
)

_VERIFY_SYSTEM = (
    "You are a Japanese dictionary assistant reviewing sense selections made for words in a sentence. "
    "Rules:\n"
    "- You will receive a sentence, and for each word: its chosen definitions and the full definition list.\n"
    "- Respond with either 'AGREE' if all selections look correct, or list the words you disagree with, "
    "one per line, in the format 'DISAGREE: word'. No other text.\n"
    "- Only flag genuine mistakes, not stylistic preferences.\n"
)

_CUSTOM_SYSTEM = (
    "You are a Japanese dictionary assistant. "
    "The provided dictionary definitions do not adequately capture how a word is used in this sentence. "
    "Rules:\n"
    "- Write a short, precise English definition (max 10 words) for the word as used in this sentence.\n"
    "- Output ONLY the definition, nothing else.\n"
)


def _format_senses(word: str, senses: list[Sense]) -> str:
    lines = [f"Word: {word}"]
    for s in senses:
        lines.append(f"  {s.pretty_string()}")
    return "\n".join(lines)


def _build_select_prompt(context: FullContext, words_and_senses: list[tuple[str, list[Sense]]]) -> str:
    prompt = f"Full Context: {context.get_chunking_context_prompt()}\n\n"
    for word, senses in words_and_senses:
        prompt += _format_senses(word, senses) + "\n\n"
    if len(words_and_senses) == 1:
        prompt += "Which definition numbers best fit the sentence? Respond with comma-separated numbers, or 'NONE'."
    else:
        prompt += (
            "For each word, which definition numbers best fit the sentence?\n"
            "Respond with one line per word: 'word: numbers' or 'word: NONE'."
        )
    return prompt


def _build_verify_prompt(context: FullContext, words_and_senses: list[tuple[str, list[Sense]]], selections: dict[str, list[int]]) -> str:
    prompt = f"Full Context: {context.get_chunking_context_prompt()}\n\n"
    for word, senses in words_and_senses:
        chosen_indices = selections.get(word, [])
        chosen = [s for s in senses if s.index in chosen_indices]
        prompt += f"Word: {word}\n"
        prompt += "  Chosen:\n"
        for s in chosen:
            prompt += f"    - {s.meaning} [{s.pos}]\n"
        prompt += "  All definitions:\n"
        for s in senses:
            prompt += f"    {s.index + 1}) {s.meaning} [{s.pos}]\n"
        prompt += "\n"
    prompt += "Do you agree with all selections? Respond 'AGREE' or list disagreements as 'DISAGREE: word'."
    return prompt


def _build_custom_prompt(context: FullContext, word: str, senses: list[Sense]) -> str:
    lines = ["Full Context:", context.get_chunking_context_prompt(), f"Word: {word}", "Provided definitions (all considered inadequate):"]
    for s in senses:
        lines.append(f"  {s.pretty_string()}")
    lines.append("\nWrite a short definition for this word as used in the sentence.")
    return "\n".join(lines)


def _parse_single(raw: str, senses: list[Sense]) -> list[int] | None:
    raw = raw.strip()
    if raw.upper() == "NONE":
        return None
    indices = []
    for tok in re.split(r"[,\s]+", raw):
        if tok.isdigit():
            i = int(tok) - 1
            if 0 <= i < len(senses):
                indices.append(i)
    return indices if indices else None


def _parse_batch(raw: str, words_and_senses: list[tuple[str, list[Sense]]]) -> dict[str, list[int] | None]:
    results: dict[str, list[int] | None] = {}
    word_lookup = {w: senses for w, senses in words_and_senses}

    for line in raw.strip().splitlines():
        line = line.strip()
        if ":" not in line:
            continue
        word, _, rest = line.partition(":")
        word = word.strip()
        rest = rest.strip()
        if word not in word_lookup:
            continue
        results[word] = _parse_single(rest, word_lookup[word])

    for word, _ in words_and_senses:
        if word not in results:
            logger.warning("LLM did not return a selection for word %r — treating as NONE", word)
            results[word] = None

    return results


def _parse_verify(raw: str) -> list[str]:
    raw = raw.strip()
    if raw.upper() == "AGREE":
        return []
    disagreed = []
    for line in raw.splitlines():
        line = line.strip()
        if line.upper().startswith("DISAGREE:"):
            word = line.split(":", 1)[1].strip()
            if word:
                disagreed.append(word)
    return disagreed


def _select_single(context: FullContext, word: str, senses: list[Sense]) -> SenseResult:
    prompt = _build_select_prompt(context, [(word, senses)])
    raw = prompt_with_retries(
        system_prompt=_SELECT_SYSTEM,
        user_prompt=prompt,
    )
    indices = _parse_single(raw, senses)

    if indices is not None:
        return SenseResult(word=word, selected=[senses[i] for i in indices])

    logger.debug("No suitable sense found for %r — requesting custom definition", word)
    custom_prompt = _build_custom_prompt(context, word, senses)
    custom_def = prompt_with_retries(
        system_prompt=_CUSTOM_SYSTEM,
        user_prompt=custom_prompt,
    )
    return SenseResult(word=word, selected=[], custom_definition=custom_def or None)


def _select_batch(context: FullContext, words_and_senses: list[tuple[str, list[Sense]]]) -> list[SenseResult]:
    select_prompt = _build_select_prompt(context, words_and_senses)
    raw_select = prompt_with_retries(
        system_prompt=_SELECT_SYSTEM,
        user_prompt=select_prompt,
    )
    selections = _parse_batch(raw_select, words_and_senses)

    verify_prompt = _build_verify_prompt(context, words_and_senses, {
        w: idxs for w, idxs in selections.items() if idxs is not None
    })
    raw_verify = prompt_with_retries(
        system_prompt=_VERIFY_SYSTEM,
        user_prompt=verify_prompt,
    )
    disagreed = _parse_verify(raw_verify)

    if disagreed:
        logger.debug("LLM disagreed with batch selections for: %s — re-prompting individually", disagreed)
        word_lookup = dict(words_and_senses)
        for word in disagreed:
            if word in word_lookup:
                selections[word] = None

    results = []
    word_lookup = dict(words_and_senses)
    for word, senses in words_and_senses:
        indices = selections.get(word)
        if indices is not None:
            results.append(SenseResult(word=word, selected=[senses[i] for i in indices]))
        else:
            results.append(_select_single(context, word, senses))

    return results


def select_senses(context: FullContext, words_and_senses: list[tuple[str, list[Sense]]]) -> list[SenseResult]:
    if not words_and_senses:
        return []

    if len(words_and_senses) <= 2:
        return [_select_single(context, word, senses) for word, senses in words_and_senses]

    return _select_batch(context, words_and_senses)