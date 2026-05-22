from typing import Iterable, Self

from attr import define


@define
class RawChunk:
    index: int
    start: float
    end: float
    subtitle_text: str

    def pretty_string(self) -> str:
        return f"\t- Start: {self.start} -> End: {self.end}\n\t- {self.index}): {self.subtitle_text}"
    
@define
class FullContext:
    target_chunk: RawChunk
    previous_chunks: list[RawChunk]
    next_chunks: list[RawChunk]

    def sort_inplace_from_start_to_end(self) -> None:
        self.previous_chunks.sort(reverse=False, key=lambda x: x.index)
        self.next_chunks.sort(reverse=False, key=lambda x: x.index)
        if self.previous_chunks and self.previous_chunks[-1].index > self.target_chunk.index:
            raise ValueError(f"Cannot sort as {self.previous_chunks[-1].index=} but {self.target_chunk.index=}")
        if self.next_chunks and self.next_chunks[0].index < self.target_chunk.index:  # Fix: was self.previous_chunks
            raise ValueError(f"Cannot sort as {self.next_chunks[0].index=} but {self.target_chunk.index=}")

    def get_chunking_context_prompt(self, sort_in_place_beforehand: bool = True) -> str:
        if sort_in_place_beforehand:
            self.sort_inplace_from_start_to_end()
        prompt = ""
        if self.previous_chunks:
            prompt += "\nHere is a list of subtitle chunks that occured before the target chunk"
            for chunk in self.previous_chunks:
                prompt += f"\n\n{chunk.pretty_string()}"
        prompt += f"\n\n<Start of Target Chunk>\n{self.target_chunk.pretty_string()}\n<End of Target Chunk>"
        if self.next_chunks:
            prompt += "\nHere is a list of subtitle chunks that occur after the target chunk"
            for chunk in self.next_chunks:
                prompt += f"\n\n{chunk.pretty_string()}"
        return prompt

    @classmethod
    def from_chunks(cls, target_index: int, raw_chunks: Iterable[RawChunk]) -> Self:
        previous_chunks: list[RawChunk] = []
        target_chunk: RawChunk | None = None
        next_chunks: list[RawChunk] = []

        for chunk in raw_chunks:
            if chunk.index < target_index:
                previous_chunks.append(chunk)
            elif chunk.index == target_index:
                target_chunk = chunk
            else:
                next_chunks.append(chunk)

        if target_chunk is None:
            raise ValueError(f"No chunk found with {target_index=} in raw_chunks")

        return cls(
            target_chunk=target_chunk,
            previous_chunks=previous_chunks,
            next_chunks=next_chunks,
        )

@define
class Sense:
    index: int
    meaning: str
    pos: str

    def pretty_string(self) -> str:
        return f"{self.index + 1}) {self.meaning} [{self.pos}]"


@define
class SenseResult:
    word: str
    selected: list[Sense]
    custom_definition: str | None = None


@define
class CompletedChunk(RawChunk):
    translation: str
    word_gloss: list[SenseResult]
    furigana: str
