import logging
from src.common.types import FullContext
from src.common.utils import assert_latin_extended_only
from src.prompting.prompter import prompt_with_retries


_TRANSLATION_SYSTEM = (
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

logger = logging.getLogger(__name__)


def translate_sentence(
    context: FullContext,
    base64_encoded_image: str | None,
    image_mime: str,
) -> str:
    prompt = "Translate the Japanese subtitle chunk to natural English."
    prompt += context.get_chunking_context_prompt()
    if base64_encoded_image:
        prompt += "\n\nA frame captured approximately 1 second into this subtitle is attached for context."
    prompt += f"\nNow translate the Japanese subtitle target chunk to natural English.\nThe chunk:\n{context.target_chunk.pretty_string()}"
    return prompt_with_retries(
        system_prompt=_TRANSLATION_SYSTEM,
        user_prompt=prompt,
        base64_encoded_image=base64_encoded_image,
        image_mime=image_mime,
        validator=assert_latin_extended_only
    )
