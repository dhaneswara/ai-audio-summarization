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


from unittest.mock import MagicMock, patch

import pytest


def _ok_response(text="ok response"):
    r = MagicMock()
    r.status_code = 200
    r.json.return_value = {"response": text}
    return r


def test_call_ollama_posts_to_generate_endpoint():
    from summarizer import Summarizer

    s = Summarizer(base_url="http://example.com")
    with patch("summarizer.requests.post", return_value=_ok_response("hi")) as post:
        result = s._call_ollama("prompt")
    assert result == "hi"
    post.assert_called_once()
    url = post.call_args.args[0]
    payload = post.call_args.kwargs["json"]
    assert url == "http://example.com/api/generate"
    assert payload["model"] == "gemma4:e4b"
    assert payload["prompt"] == "prompt"
    assert payload["stream"] is False


def test_call_ollama_raises_friendly_error_when_not_running():
    import requests as real_requests
    from summarizer import Summarizer, SummarizerError

    s = Summarizer()
    with patch("summarizer.requests.post", side_effect=real_requests.ConnectionError()):
        with pytest.raises(SummarizerError) as exc:
            s._call_ollama("prompt")
    assert "ollama" in str(exc.value).lower()


def test_call_ollama_raises_friendly_error_when_model_missing():
    from summarizer import Summarizer, SummarizerError

    r = MagicMock()
    r.status_code = 404
    r.json.return_value = {"error": "model 'gemma4:e4b' not found"}
    with patch("summarizer.requests.post", return_value=r):
        s = Summarizer()
        with pytest.raises(SummarizerError) as exc:
            s._call_ollama("prompt")
    assert "gemma4:e4b" in str(exc.value)


def test_call_ollama_raises_friendly_error_on_timeout():
    import requests as real_requests
    from summarizer import Summarizer, SummarizerError

    with patch("summarizer.requests.post", side_effect=real_requests.Timeout()):
        s = Summarizer()
        with pytest.raises(SummarizerError) as exc:
            s._call_ollama("prompt")
    assert "timed out" in str(exc.value).lower()


def test_summarize_short_text_makes_one_ollama_call():
    """A transcript fitting in one chunk uses a single Ollama call with the final prompt."""
    from summarizer import Summarizer

    s = Summarizer()
    with patch.object(s, "_call_ollama", return_value="FINAL SUMMARY") as call:
        out = s.summarize("Short transcript.", style="bullets", length="medium")
    assert out == "FINAL SUMMARY"
    assert call.call_count == 1
    # The single call should use the final (not chunk) template.
    sent_prompt = call.call_args.args[0]
    assert "TL;DR" in sent_prompt


def test_summarize_long_text_does_map_reduce():
    """A transcript spanning multiple chunks calls Ollama once per chunk plus once to combine."""
    from summarizer import Summarizer

    s = Summarizer()
    long_text = ("alpha beta gamma delta " * 5000).strip()

    # First, confirm the fixture actually chunks.
    chunks = s.chunk_text(long_text)
    assert len(chunks) >= 2

    responses = [f"partial {i}" for i in range(len(chunks))] + ["FINAL"]
    with patch.object(s, "_call_ollama", side_effect=responses) as call:
        out = s.summarize(long_text, style="bullets", length="medium")

    assert out == "FINAL"
    assert call.call_count == len(chunks) + 1
    # The last call combines the partials and uses the final template.
    final_prompt = call.call_args_list[-1].args[0]
    assert "TL;DR" in final_prompt
    for i in range(len(chunks)):
        assert f"partial {i}" in final_prompt
