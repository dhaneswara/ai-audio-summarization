"""Audio Summarization app — Gradio UI and pipeline handler."""

from __future__ import annotations

from typing import Optional

from summarizer import Summarizer, SummarizerError
from transcriber import Transcriber


def run_pipeline(
    audio_path: Optional[str],
    style: str,
    length: str,
    transcriber: Transcriber,
    summarizer: Summarizer,
) -> tuple[str, str]:
    """Transcribe an audio file, unload the model, then summarize.

    Returns (transcript, summary_or_message). Exceptions from the summarizer are
    caught and rendered as a user-facing message so the UI never crashes.
    """
    if not audio_path:
        return "", "Please upload an audio file."

    result = transcriber.transcribe(audio_path)
    transcript = result.text.strip()
    transcriber.unload()

    if not transcript:
        return "", "No speech detected in audio."

    try:
        summary = summarizer.summarize(transcript, style=style, length=length)
    except SummarizerError as e:
        return transcript, f"Summarization failed: {e}"

    return transcript, summary
