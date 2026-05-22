import logging
from typing import Callable

from openai import OpenAI


API_KEY = "EMPTY"
MODEL = "google/gemma-4-31B-it"
BASE_URL = "http://localhost:9090/v1"
MAX_TOKENS = 2048
TEMPERATURE = 0.1

CLIENT = OpenAI(base_url=BASE_URL, api_key=API_KEY)

logger = logging.getLogger(__name__)


def _build_user_message(user_prompt: str, base64_encoded_image: str | None, image_mime: str = "image/jpeg") -> dict:
    if base64_encoded_image is None:
        return {"role": "user", "content": user_prompt}
    return {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": f"data:{image_mime};base64,{base64_encoded_image}"}
            },
            {"type": "text", "text": user_prompt}
        ]
    }


def prompt_with_retries(
    system_prompt: str,
    user_prompt: str,
    *,
    base64_encoded_image: str | None = None,
    image_mime: str = "image/jpeg",
    max_retries: int = 3,
    validator: Callable[[str], None] | None = None,
) -> str:
    for _ in range(max_retries):
        try:
            response = CLIENT.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    _build_user_message(user_prompt, base64_encoded_image, image_mime=image_mime),
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("Response content was None")
            if validator:
                validator(content)
            return content 
        except Exception as e:
            logger.error("Error when prompting LLM: %s", e)
            user_prompt = (
                f"{user_prompt}\n\n--------\n"
                f"Please try again. Your last response contained the following error(s):\n{e}"
            )
    return ""
