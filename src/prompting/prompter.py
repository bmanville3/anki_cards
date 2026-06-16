from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from typing import Callable

from attr import define
from openai import OpenAI

LLM_API_KEY = "EMPTY"
LLM_MODEL = "google/gemma-4-31B-it"
LLM_BASE_URL = "http://localhost:9090/v1"
LLM_MAX_TOKENS = 8192
LLM_TEMPERATURE = 0.1
LLM_WORKERS = 8

CLIENT = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

logger = logging.getLogger(__name__)


@define
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
                "image_url": {"url": f"data:{image_mime};base64,{base64_encoded_image}"}
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
