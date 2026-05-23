import logging
from typing import Literal
from src.common.types import FullContext
from src.common.utils import assert_latin_extended_only
from src.prompting.prompter import prompt_with_retries


_NATURAL_TRANSLATION_SYSTEM = (
    "You are a Japanese-to-English translator specializing in natural, idiomatic English. "
    "The input is the surrounding context and a single subtitle line from a Japanese TV show or video. "
    "The output is the english translation of the subtitle line. "
    "The end goal is to make an Anki Flashcard course focusing on JP->EN cards. "
    "Rules:\n"
    "- Output ONLY the English translation, nothing else — no notes, no explanations.\n"
    "- Any characters not found in standard English are NOT allowed unless the word is a loanword such as 'Café'.\n"
    "- Produce natural English a native speaker would say.\n"
    "- Japanese often omits the subject; infer it from context and use 'it', 'they', 'you', etc. appropriately.\n"
    "- Sound cues like （笑）、♪、or [拍手] should be rendered as a brief parenthetical like (laughter) or (music).\n"
    "- Prefer the most common/literal reading unless it sounds unnatural in which case you may style it some.\n"
    "- Never add anything that isn't in the original.\n"
    "- If a video frame is attached, it is provided for context only. Do not add visual descriptions or weight it heavily over the text."
)

_LITERAL_TRANSLATION_SYSTEM = (
    "You are a Japanese-to-English translator specializing in literal Japanese-to-English translations. "
    "The input is the surrounding context and a single subtitle line from a Japanese TV show or video. "
    "The output is the english translation of the subtitle line. "
    "Your main goal is to translate cards like the Japanese like a breeze (JLAB) Anki deck does. Here are some example translation from that Anki deck:\n"
    "- '今日の晩ご飯何がいいですか' -> 'As for today's dinner, what is good?'\n"
    "- '悪い話じゃないな' -> 'isn't a bad conversation'\n"
    "- '大事な話があります' -> 'important conversation is there'\n"
    "- '何しに行くの' -> 'what are you going to do'\n"
    "- 'あの男は人間じゃない' -> 'As for that guy over there, is not human.'\n"
    "- '楽しい話ね' -> 'fun conversation, isn't it!'\n"
    "The end goal is to make an Anki Flashcard course focusing on JP->EN cards. "
    "Rules:\n"
    "- Output ONLY the English translation, nothing else — no notes, no explanations.\n"
    "- Any characters not found in standard English are NOT allowed unless the word is a loanword such as 'Café'.\n"
    "- Produce literal Japanese-to-English translations. However, if the literal translation is too obscure, it may be converted to a more comprehensible form.\n"
    "- Sound cues like （笑）、♪、or [拍手] should be rendered as a brief parenthetical like (laughter) or (music).\n"
    "- Prefer the most literal reading unless it sounds very unnatural in which case you may style it some.\n"
    "- Never add anything that isn't in the original.\n"
    "- If a video frame is attached, it is provided for context only. Do not add visual descriptions or weight it heavily over the text."
)

logger = logging.getLogger(__name__)


def translate_sentence(
    context: FullContext,
    base64_encoded_image: str | None,
    image_mime: str,
    translation_type: Literal["natural"] | Literal["literal"]
) -> str:
    prompt = f"Translate the Japanese subtitle chunk to {translation_type} English."
    prompt += context.get_chunking_context_prompt()
    if base64_encoded_image:
        prompt += "\n\nA frame captured sometime into this subtitle is attached for context."
    prompt += f"\nNow translate the Japanese subtitle target chunk to {translation_type} English.\nThe chunk:\n{context.target_chunk.pretty_string()}"
    system_prompt = ""
    if translation_type == "natural":
        system_prompt = _NATURAL_TRANSLATION_SYSTEM
    elif translation_type == "literal":
        system_prompt = _LITERAL_TRANSLATION_SYSTEM
    else:
        raise ValueError(f"Unknown translation type: {translation_type}")
    return prompt_with_retries(
        system_prompt=system_prompt,
        user_prompt=prompt,
        base64_encoded_image=base64_encoded_image,
        image_mime=image_mime,
        validator=assert_latin_extended_only
    )
