from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import logging
import sqlite3
from typing import Callable
from openai import OpenAI

_LITERAL_TRANSLATION_SYSTEM = (
    "You are a Japanese-to-English translator specializing in Japanese-to-English translations for learners. "
    "The input a card from a Japanese-to-English Anki deck and the output it the best translation possible for that card for learners. "
    "The output should favor more literal translations over paraphrased translations. "
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
)

LLM_API_KEY = "EMPTY"
LLM_MODEL = "google/gemma-4-31B-it"
LLM_BASE_URL = "http://localhost:9090/v1"
LLM_MAX_TOKENS = 2048
LLM_TEMPERATURE = 0.1
LLM_WORKERS = 32
CLIENT = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@dataclass
class PromptRequest:
    system_prompt: str
    user_prompt: str
    base64_encoded_image: str | None = None
    image_mime: str = "image/jpeg"
    max_retries: int = 3
    validator: Callable[[str], None] | None = None

def _build_user_message(user_prompt: str, base64_encoded_image: str | None, image_mime: str = "image/jpeg") -> dict:
    if base64_encoded_image is None:
        return {"role": "user", "content": user_prompt}
    return {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": f",{base64_encoded_image}"}
            },
            {"type": "text", "text": user_prompt}
        ]
    }

def prompt_with_retries(prompt: PromptRequest) -> str:
    for _ in range(prompt.max_retries):
        try:
            response = CLIENT.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": prompt.system_prompt},
                    _build_user_message(prompt.user_prompt, prompt.base64_encoded_image, image_mime=prompt.image_mime),
                ],
                max_tokens=LLM_MAX_TOKENS,
                temperature=LLM_TEMPERATURE,
            )
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("Response content was None")
            if prompt.validator:
                prompt.validator(content)
            logger.info("New response: %s", content)
            return content 
        except Exception as e:
            logger.error("Error when prompting LLM: %s", e)
            prompt.user_prompt = (
                f"{prompt.user_prompt}\n\n--------\n"
                f"Please try again. Your last response contained the following error(s):\n{e}"
            )
    logger.error("LLM failed Final prompt:\n%s", prompt.user_prompt)
    return ""

def prompt_batch(
    requests: list[PromptRequest],
    *,
    max_workers: int = LLM_WORKERS,
) -> list[str]:
    """Send a batch of prompts concurrently. Returns results in input order."""
    results = [""] * len(requests)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(prompt_with_retries, req): i
            for i, req in enumerate(requests)
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return results

path = "c/collection.anki2"
conn = sqlite3.connect(path)
cursor = conn.cursor()
cursor.execute("SELECT models FROM col")
models_json = cursor.fetchone()[0]
models = json.loads(models_json)

model_fields = {}
for model_id, model in models.items():
    model_fields[int(model_id)] = [
        field["name"] for field in model["flds"]
    ]

cursor.execute("SELECT mid, flds FROM notes")

@dataclass
class Card:
    vocab_jp: str
    vocab_en: str
    pos: str
    sentence_jp: str
    sentence_en: str

cards = []
for mid, flds in cursor.fetchall():
    field_names = model_fields[mid]
    field_values = flds.split("\x1f")
    data = dict(zip(field_names, field_values))
    vocab_jp = data.get("Vocabulary-Kanji", "")
    vocab_en = data.get("Vocabulary-English", "")
    pos = data.get("Vocabulary-Pos", "")
    sentence_jp = data.get("Expression", "")
    sentence_en = data.get("Sentence-English", "")
    cards.append(Card(vocab_jp=vocab_jp, vocab_en=vocab_en, pos=pos, sentence_jp=sentence_jp, sentence_en=sentence_en))

# Build prompts
prompts = []
for card in cards:
    user_prompt = f"""Translate this Japanese sentence to English:

Vocabulary: {card.vocab_jp} ({card.vocab_en}) - {card.pos}
Japanese Sentence: {card.sentence_jp}

NOTE: An existing translation is provided below, but it may be INCORRECT or not literal enough. Do NOT trust it blindly. Translate from scratch based on the Japanese text:
Existing translation (possibly wrong): {card.sentence_en}

Provide your literal translation:"""
    
    prompts.append(PromptRequest(
        system_prompt=_LITERAL_TRANSLATION_SYSTEM,
        user_prompt=user_prompt
    ))

# Get translations
print(f"Translating {len(prompts)} cards...")
translations = prompt_batch(prompts)

# Write results to file
with open("translations.txt", "w", encoding="utf-8") as f:
    for card, translation in zip(cards, translations):
        f.write(f"Vocab JP: {card.vocab_jp}\n")
        f.write(f"Vocab EN: {card.vocab_en}\n")
        f.write(f"POS: {card.pos}\n")
        f.write(f"Sentence JP: {card.sentence_jp}\n")
        f.write(f"Original EN: {card.sentence_en}\n")
        f.write(f"New Translation: {translation}\n")
        f.write("--------\n\n")

print(f"Done! Results written to translations.txt")
conn.close()