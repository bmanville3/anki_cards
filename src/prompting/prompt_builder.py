from src.common.types import RawChunk
from src.common.utils import check_chunk_ordering


def build_chunk_prompt_with_surrounding_context(
    target_chunk: RawChunk,
    previous_chunks: list[RawChunk],
    next_chunk: RawChunk | None,
    should_check_ordering: bool,
) -> str:
    if should_check_ordering:
        check_chunk_ordering(target_chunk=target_chunk, previous_chunks=previous_chunks, next_chunk=next_chunk)
    prompt = ""
    if previous_chunks:
        prompt += "\nHere is a list of subtitle chunks that occured before the target chunk"
        for chunk in previous_chunks:
            prompt += f"\n\n{chunk.pretty_string()}"
    prompt += f"\n\n<Start of Target Chunk>\n{target_chunk.pretty_string()}\n<End of Target Chunk>"
    if next_chunk:
        prompt += f"\n\nHere is the subtitle chunk directly following the target chunk: {next_chunk.pretty_string()}"
    return prompt
