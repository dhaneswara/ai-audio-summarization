"""Ollama-backed summarizer with map-reduce chunking for long transcripts."""

from __future__ import annotations

from typing import Literal

import tiktoken

Style = Literal["bullets", "paragraphs"]
Length = Literal["short", "medium", "long"]


class SummarizerError(Exception):
    """Raised when summarization fails for a reason worth showing the user."""


_FINAL_TEMPLATE = """You are a careful summarizer. Read the transcript below and produce a structured summary.

Format the summary as Markdown with these sections:

## TL;DR
{tldr_hint}

## Key Points
{keypoints_hint}

## Notable Details
{details_hint}

Transcript:
\"\"\"
{transcript}
\"\"\"
"""

_CHUNK_TEMPLATE = """You are summarizing one part of a longer transcript. Produce a concise, factual partial summary of just this section. Do not write a final overview — another pass will combine all partial summaries.

Section transcript:
\"\"\"
{transcript}
\"\"\"

Partial summary:"""


_LENGTH_HINTS = {
    "short": {
        "tldr_hint": "Write 1–2 concise sentences.",
        "keypoints_hint": "List 3–5 short bullet points.",
        "details_hint": "Keep this section brief, only the most important specifics.",
    },
    "medium": {
        "tldr_hint": "Write 2–3 sentences.",
        "keypoints_hint": "List 5–8 bullet points covering the main topics.",
        "details_hint": "Cover notable specifics, decisions, or quotes.",
    },
    "long": {
        "tldr_hint": "Write 3–4 sentences.",
        "keypoints_hint": "List 8–12 bullet points with thorough coverage.",
        "details_hint": "Cover specifics, decisions, quotes, names, and timing.",
    },
}


class Summarizer:
    CHUNK_SIZE = 6000
    OVERLAP = 200
    TIMEOUT_SECONDS = 300

    def __init__(
        self,
        model: str = "gemma4:e4b",
        base_url: str = "http://localhost:11434",
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    @property
    def _encoder(self):
        if not hasattr(self, "_enc"):
            self._enc = tiktoken.get_encoding("cl100k_base")
        return self._enc

    def chunk_text(self, text: str) -> list[str]:
        tokens = self._encoder.encode(text)
        if len(tokens) <= self.CHUNK_SIZE:
            return [text]

        chunks: list[str] = []
        step = self.CHUNK_SIZE - self.OVERLAP
        start = 0
        while start < len(tokens):
            end = min(start + self.CHUNK_SIZE, len(tokens))
            chunks.append(self._encoder.decode(tokens[start:end]))
            if end == len(tokens):
                break
            start += step
        return chunks

    def _build_prompt(
        self,
        transcript: str,
        style: Style,
        length: Length,
        is_chunk: bool = False,
    ) -> str:
        if is_chunk:
            return _CHUNK_TEMPLATE.format(transcript=transcript)

        hints = dict(_LENGTH_HINTS[length])
        if style == "paragraphs":
            hints["details_hint"] = hints["details_hint"] + " Write this section as paragraphs, not bullets."
        else:
            hints["details_hint"] = hints["details_hint"] + " Use bullet points where natural."
        return _FINAL_TEMPLATE.format(transcript=transcript, **hints)
