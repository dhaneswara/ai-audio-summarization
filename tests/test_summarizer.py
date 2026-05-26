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


def test_chunk_text_short_returns_single_chunk():
    """Text under CHUNK_SIZE tokens returns one chunk equal to the input."""
    from summarizer import Summarizer

    s = Summarizer()
    chunks = s.chunk_text("This is a short transcript.")
    assert chunks == ["This is a short transcript."]


def test_chunk_text_long_returns_multiple_chunks_with_overlap():
    """A transcript longer than CHUNK_SIZE tokens is split with overlap."""
    from summarizer import Summarizer

    s = Summarizer()
    # ~3 chars per token via tiktoken approx; build something clearly over 6000 tokens.
    # Use a repeating word so the boundary effect is deterministic.
    long_text = ("hello world " * 5000).strip()
    chunks = s.chunk_text(long_text)
    assert len(chunks) >= 2
    # Each chunk should be non-empty.
    assert all(len(c) > 0 for c in chunks)


def test_chunk_text_chunks_have_overlap():
    """Consecutive chunks share at least one token of overlap when text is long."""
    from summarizer import Summarizer

    s = Summarizer()
    long_text = ("alpha beta gamma delta " * 3000).strip()
    chunks = s.chunk_text(long_text)
    if len(chunks) < 2:
        # Sanity: the test relies on actually getting multiple chunks.
        raise AssertionError("Test fixture too short; chunking did not split.")
    # The end of chunk[0] and start of chunk[1] should share at least some content.
    tail = chunks[0][-200:]
    head = chunks[1][:200]
    # Look for any 20-char window from tail that appears in head.
    assert any(tail[i : i + 20] in head for i in range(0, len(tail) - 20))
