"""Tests for the Summarizer class."""


def test_build_prompt_includes_transcript():
    from summarizer import Summarizer

    s = Summarizer()
    prompt = s._build_prompt("Hello world.", style="bullets", length="medium")
    assert "Hello world." in prompt


def test_build_prompt_short_length_mentions_brevity():
    from summarizer import Summarizer

    s = Summarizer()
    prompt = s._build_prompt("text", style="bullets", length="short")
    assert "short" in prompt.lower() or "brief" in prompt.lower() or "concise" in prompt.lower()


def test_build_prompt_bullets_style_asks_for_bullets():
    from summarizer import Summarizer

    s = Summarizer()
    prompt = s._build_prompt("text", style="bullets", length="medium")
    assert "bullet" in prompt.lower()


def test_build_prompt_chunk_mode_asks_for_partial_summary():
    """Chunk prompts should ask for a partial summary, not the final structured output."""
    from summarizer import Summarizer

    s = Summarizer()
    prompt = s._build_prompt("text", style="bullets", length="medium", is_chunk=True)
    assert "part" in prompt.lower() or "chunk" in prompt.lower() or "section" in prompt.lower()
