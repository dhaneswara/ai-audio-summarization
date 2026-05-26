"""Tests for the pipeline handler in app.py."""

from unittest.mock import MagicMock

import pytest


def test_run_pipeline_returns_transcript_and_summary():
    """run_pipeline runs transcribe → unload → summarize and returns both outputs."""
    from app import run_pipeline
    from transcriber import TranscriptionResult

    transcriber = MagicMock()
    transcriber.transcribe.return_value = TranscriptionResult(
        text="hello world", language="en", segments=[]
    )
    summarizer = MagicMock()
    summarizer.summarize.return_value = "## TL;DR\nA greeting."

    transcript, summary = run_pipeline(
        "fake.wav",
        style="bullets",
        length="medium",
        transcriber=transcriber,
        summarizer=summarizer,
    )

    assert transcript == "hello world"
    assert summary == "## TL;DR\nA greeting."
    transcriber.transcribe.assert_called_once_with("fake.wav")
    transcriber.unload.assert_called_once()
    summarizer.summarize.assert_called_once_with(
        "hello world", style="bullets", length="medium"
    )


def test_run_pipeline_unloads_before_summarizing():
    """unload() must be called before summarize() so they don't share VRAM."""
    from app import run_pipeline
    from transcriber import TranscriptionResult

    call_order: list[str] = []
    transcriber = MagicMock()
    transcriber.transcribe.return_value = TranscriptionResult(
        text="text", language="en", segments=[]
    )
    transcriber.unload.side_effect = lambda: call_order.append("unload")

    summarizer = MagicMock()
    summarizer.summarize.side_effect = lambda *a, **kw: (
        call_order.append("summarize") or "summary"
    )

    run_pipeline(
        "fake.wav",
        style="bullets",
        length="medium",
        transcriber=transcriber,
        summarizer=summarizer,
    )

    assert call_order == ["unload", "summarize"]


def test_run_pipeline_skips_summary_when_transcript_is_empty():
    """Empty transcript means no speech detected — return a message instead of summarizing."""
    from app import run_pipeline
    from transcriber import TranscriptionResult

    transcriber = MagicMock()
    transcriber.transcribe.return_value = TranscriptionResult(
        text="   ", language="en", segments=[]
    )
    summarizer = MagicMock()

    transcript, summary = run_pipeline(
        "fake.wav",
        style="bullets",
        length="medium",
        transcriber=transcriber,
        summarizer=summarizer,
    )

    assert "no speech" in summary.lower()
    summarizer.summarize.assert_not_called()


def test_run_pipeline_requires_audio_path():
    """A missing audio path returns a friendly message, no work done."""
    from app import run_pipeline

    transcriber = MagicMock()
    summarizer = MagicMock()

    transcript, summary = run_pipeline(
        None,
        style="bullets",
        length="medium",
        transcriber=transcriber,
        summarizer=summarizer,
    )
    assert "upload" in summary.lower()
    transcriber.transcribe.assert_not_called()
    summarizer.summarize.assert_not_called()


def test_run_pipeline_surfaces_summarizer_errors():
    """If the summarizer raises SummarizerError, return the transcript and the error text."""
    from app import run_pipeline
    from summarizer import SummarizerError
    from transcriber import TranscriptionResult

    transcriber = MagicMock()
    transcriber.transcribe.return_value = TranscriptionResult(
        text="ok", language="en", segments=[]
    )
    summarizer = MagicMock()
    summarizer.summarize.side_effect = SummarizerError("Ollama is not running.")

    transcript, summary = run_pipeline(
        "fake.wav",
        style="bullets",
        length="medium",
        transcriber=transcriber,
        summarizer=summarizer,
    )

    assert transcript == "ok"
    assert "ollama is not running" in summary.lower()
