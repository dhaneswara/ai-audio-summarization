# Audio Summarization App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local Gradio web app that transcribes uploaded audio files via `faster-whisper` and summarizes them via Ollama (`gemma4:e4b`), running the two models sequentially to fit a 10 GB GPU.

**Architecture:** Single-process Python app. Three modules: `transcriber.py` (faster-whisper wrapper with explicit unload), `summarizer.py` (Ollama HTTP client with map-reduce chunking), `app.py` (Gradio UI + pipeline handler that orchestrates transcribe → unload → summarize). Tests use mocks for the heavy dependencies (WhisperModel, requests) so they run without GPU or Ollama.

**Tech Stack:** Python 3.10+, faster-whisper, gradio, requests, tiktoken, pytest, pytest-mock.

---

## File Structure

Files created by this plan:

| Path | Responsibility |
|---|---|
| `requirements.txt` | All Python dependencies (runtime + dev) |
| `.gitignore` | Standard Python + IDE exclusions |
| `transcriber.py` | `Transcriber` class: lazy-load Whisper, transcribe, unload |
| `summarizer.py` | `Summarizer` class: prompt building, chunking, Ollama HTTP, map-reduce |
| `app.py` | Gradio UI + pipeline handler |
| `tests/__init__.py` | Marks tests as a package |
| `tests/conftest.py` | Shared pytest fixtures |
| `tests/test_transcriber.py` | Unit tests for `Transcriber` (mocked WhisperModel) |
| `tests/test_summarizer.py` | Unit tests for `Summarizer` (mocked requests) |
| `tests/test_app.py` | Unit tests for pipeline handler (mocked Transcriber + Summarizer) |
| `README.md` | Install, run, troubleshoot |

---

## Task 1: Project scaffold and git init

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `README.md`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create `requirements.txt`**

```
faster-whisper>=1.0.0
gradio>=4.0.0
requests>=2.31.0
tiktoken>=0.5.0
pytest>=7.0.0
pytest-mock>=3.10.0
```

- [ ] **Step 2: Create `.gitignore`**

```
__pycache__/
*.py[cod]
*$py.class
.pytest_cache/
.venv/
venv/
env/
.env
.idea/
.vscode/
*.egg-info/
build/
dist/
.coverage
htmlcov/
# Audio test artifacts
*.mp3
*.wav
*.m4a
*.mp4
*.webm
!tests/fixtures/**
```

- [ ] **Step 3: Create `README.md` skeleton**

```markdown
# Audio Summarization

Local web app: upload audio → transcript + summary. Uses `faster-whisper` for speech-to-text and Ollama (`gemma4:e4b`) for summarization.

## Requirements

- Python 3.10+
- Ollama running locally (`ollama serve`) with `gemma4:e4b` pulled
- NVIDIA GPU recommended (10 GB VRAM minimum for default model)

## Install

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
python app.py
```

Open the URL Gradio prints (typically http://127.0.0.1:7860).

## Tests

```powershell
pytest
```
```

- [ ] **Step 4: Create `tests/__init__.py`**

Empty file.

- [ ] **Step 5: Create `tests/conftest.py`**

```python
"""Shared pytest fixtures."""
```

- [ ] **Step 6: Initialize git and make first commit**

```powershell
git init
git add .gitignore README.md requirements.txt tests/__init__.py tests/conftest.py docs/
git commit -m "chore: project scaffold"
```

Expected: a new repo with one commit.

---

## Task 2: Transcriber class with lazy load and CPU fallback

**Files:**
- Create: `transcriber.py`
- Create: `tests/test_transcriber.py`

- [ ] **Step 1: Write the failing tests for init and lazy load**

Create `tests/test_transcriber.py`:

```python
"""Tests for the Transcriber class."""

from unittest.mock import MagicMock, patch

import pytest


def test_transcriber_init_does_not_load_model():
    """Constructing a Transcriber must not load the heavy model."""
    from transcriber import Transcriber

    with patch("transcriber.WhisperModel") as MockModel:
        t = Transcriber()
        MockModel.assert_not_called()
        assert t._model is None


def test_transcriber_load_uses_gpu_by_default():
    """First transcribe call loads the model on cuda with the configured compute type."""
    from transcriber import Transcriber

    with patch("transcriber.WhisperModel") as MockModel:
        MockModel.return_value = MagicMock()
        t = Transcriber()
        t._load()
        MockModel.assert_called_once_with(
            "large-v3", device="cuda", compute_type="int8_float16"
        )


def test_transcriber_falls_back_to_cpu_on_gpu_failure():
    """If cuda load fails, fall back to CPU and remember it."""
    from transcriber import Transcriber

    cpu_model = MagicMock()
    with patch("transcriber.WhisperModel") as MockModel:
        MockModel.side_effect = [RuntimeError("CUDA out of memory"), cpu_model]
        t = Transcriber()
        t._load()
        assert MockModel.call_count == 2
        assert MockModel.call_args_list[1].kwargs["device"] == "cpu"
        assert t.device == "cpu"
        assert t._model is cpu_model
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
pytest tests/test_transcriber.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'transcriber'`.

- [ ] **Step 3: Implement `transcriber.py` (init + load only)**

```python
"""Wraps faster-whisper for sequential GPU use with an explicit unload step."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from faster_whisper import WhisperModel


@dataclass
class TranscriptionResult:
    text: str
    language: str
    segments: list


class Transcriber:
    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "int8_float16",
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model: Optional[WhisperModel] = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            self._model = WhisperModel(
                self.model_size, device=self.device, compute_type=self.compute_type
            )
        except Exception:
            self._model = WhisperModel(
                self.model_size, device="cpu", compute_type="int8"
            )
            self.device = "cpu"
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
pytest tests/test_transcriber.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```powershell
git add transcriber.py tests/test_transcriber.py
git commit -m "feat(transcriber): add Transcriber class with lazy load and CPU fallback"
```

---

## Task 3: Transcriber `transcribe` and `unload` methods

**Files:**
- Modify: `transcriber.py`
- Modify: `tests/test_transcriber.py`

- [ ] **Step 1: Add failing tests for transcribe and unload**

Append to `tests/test_transcriber.py`:

```python
def test_transcribe_returns_joined_text_and_language():
    """transcribe() returns concatenated segment text, the detected language, and segments."""
    from transcriber import Transcriber

    seg1 = MagicMock()
    seg1.text = " Hello "
    seg2 = MagicMock()
    seg2.text = " world. "
    info = MagicMock()
    info.language = "en"

    fake_model = MagicMock()
    fake_model.transcribe.return_value = (iter([seg1, seg2]), info)

    with patch("transcriber.WhisperModel", return_value=fake_model):
        t = Transcriber()
        result = t.transcribe("ignored.wav")

    assert result.text == "Hello world."
    assert result.language == "en"
    assert len(result.segments) == 2


def test_transcribe_calls_load_once():
    """Repeated transcribe() calls must not reload the model."""
    from transcriber import Transcriber

    fake_model = MagicMock()
    fake_model.transcribe.return_value = (iter([]), MagicMock(language="en"))

    with patch("transcriber.WhisperModel", return_value=fake_model) as MockModel:
        t = Transcriber()
        t.transcribe("a.wav")
        # Reset the iterator for the second call
        fake_model.transcribe.return_value = (iter([]), MagicMock(language="en"))
        t.transcribe("b.wav")
        MockModel.assert_called_once()


def test_unload_clears_model_reference():
    """unload() must drop the model reference so it can be garbage collected."""
    from transcriber import Transcriber

    with patch("transcriber.WhisperModel", return_value=MagicMock()):
        t = Transcriber()
        t._load()
        assert t._model is not None
        t.unload()
        assert t._model is None


def test_unload_when_not_loaded_is_safe():
    """Calling unload() before any load must not error."""
    from transcriber import Transcriber

    t = Transcriber()
    t.unload()  # should not raise
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
pytest tests/test_transcriber.py -v
```

Expected: 4 new tests FAIL with `AttributeError: 'Transcriber' object has no attribute 'transcribe'` and similar.

- [ ] **Step 3: Add `transcribe` and `unload` methods to `transcriber.py`**

Append to the `Transcriber` class in `transcriber.py`:

```python
    def transcribe(self, audio_path: str) -> TranscriptionResult:
        self._load()
        assert self._model is not None
        segments_iter, info = self._model.transcribe(audio_path)
        segments = list(segments_iter)
        text = " ".join(s.text.strip() for s in segments).strip()
        return TranscriptionResult(text=text, language=info.language, segments=segments)

    def unload(self) -> None:
        if self._model is None:
            return
        del self._model
        self._model = None
        try:
            import torch

            torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
pytest tests/test_transcriber.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```powershell
git add transcriber.py tests/test_transcriber.py
git commit -m "feat(transcriber): add transcribe and unload methods"
```

---

## Task 4: Summarizer prompt builder

**Files:**
- Create: `summarizer.py`
- Create: `tests/test_summarizer.py`

- [ ] **Step 1: Write failing tests for the prompt builder**

Create `tests/test_summarizer.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
pytest tests/test_summarizer.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'summarizer'`.

- [ ] **Step 3: Create `summarizer.py` with prompt builder only**

```python
"""Ollama-backed summarizer with map-reduce chunking for long transcripts."""

from __future__ import annotations

from typing import Literal

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
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
pytest tests/test_summarizer.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```powershell
git add summarizer.py tests/test_summarizer.py
git commit -m "feat(summarizer): add prompt builder for final and chunk prompts"
```

---

## Task 5: Summarizer chunking

**Files:**
- Modify: `summarizer.py`
- Modify: `tests/test_summarizer.py`

- [ ] **Step 1: Write failing tests for chunking**

Append to `tests/test_summarizer.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
pytest tests/test_summarizer.py -v
```

Expected: 3 new tests FAIL with `AttributeError: 'Summarizer' object has no attribute 'chunk_text'`.

- [ ] **Step 3: Add chunking to `summarizer.py`**

At the top of `summarizer.py`, add the import:

```python
import tiktoken
```

Add this method to the `Summarizer` class (place it after `__init__`):

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
pytest tests/test_summarizer.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```powershell
git add summarizer.py tests/test_summarizer.py
git commit -m "feat(summarizer): add token-based chunking with overlap"
```

---

## Task 6: Summarizer Ollama HTTP call with error handling

**Files:**
- Modify: `summarizer.py`
- Modify: `tests/test_summarizer.py`

- [ ] **Step 1: Write failing tests for the Ollama call**

Append to `tests/test_summarizer.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
pytest tests/test_summarizer.py -v
```

Expected: 4 new tests FAIL with `AttributeError: 'Summarizer' object has no attribute '_call_ollama'`.

- [ ] **Step 3: Add `_call_ollama` to `summarizer.py`**

At the top of `summarizer.py`, add the import:

```python
import requests
```

Add this method to the `Summarizer` class:

```python
    def _call_ollama(self, prompt: str) -> str:
        url = f"{self.base_url}/api/generate"
        try:
            response = requests.post(
                url,
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=self.TIMEOUT_SECONDS,
            )
        except requests.ConnectionError as e:
            raise SummarizerError(
                "Ollama is not running. Start it with `ollama serve` and try again."
            ) from e
        except requests.Timeout as e:
            raise SummarizerError(
                "Ollama request timed out. Try a shorter clip or a smaller model."
            ) from e

        if response.status_code == 404:
            raise SummarizerError(
                f"Model `{self.model}` not found. Run `ollama pull {self.model}`."
            )
        if response.status_code >= 400:
            raise SummarizerError(
                f"Ollama returned HTTP {response.status_code}: {response.text[:200]}"
            )

        data = response.json()
        return data.get("response", "")
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
pytest tests/test_summarizer.py -v
```

Expected: 11 passed.

- [ ] **Step 5: Commit**

```powershell
git add summarizer.py tests/test_summarizer.py
git commit -m "feat(summarizer): add Ollama HTTP call with friendly error handling"
```

---

## Task 7: Summarizer map-reduce orchestration

**Files:**
- Modify: `summarizer.py`
- Modify: `tests/test_summarizer.py`

- [ ] **Step 1: Write failing tests for `summarize`**

Append to `tests/test_summarizer.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
pytest tests/test_summarizer.py -v
```

Expected: 2 new tests FAIL with `AttributeError: 'Summarizer' object has no attribute 'summarize'`.

- [ ] **Step 3: Add `summarize` to `summarizer.py`**

Add this method to the `Summarizer` class:

```python
    def summarize(
        self,
        transcript: str,
        style: Style = "bullets",
        length: Length = "medium",
    ) -> str:
        chunks = self.chunk_text(transcript)
        if len(chunks) == 1:
            prompt = self._build_prompt(chunks[0], style=style, length=length, is_chunk=False)
            return self._call_ollama(prompt)

        partials: list[str] = []
        for chunk in chunks:
            prompt = self._build_prompt(chunk, style=style, length=length, is_chunk=True)
            partials.append(self._call_ollama(prompt))

        combined = "\n\n---\n\n".join(partials)
        final_prompt = self._build_prompt(combined, style=style, length=length, is_chunk=False)
        return self._call_ollama(final_prompt)
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
pytest tests/test_summarizer.py -v
```

Expected: 13 passed.

- [ ] **Step 5: Commit**

```powershell
git add summarizer.py tests/test_summarizer.py
git commit -m "feat(summarizer): add map-reduce summarize method"
```

---

## Task 8: Pipeline handler (testable orchestration function)

**Files:**
- Create: `app.py`
- Create: `tests/test_app.py`

- [ ] **Step 1: Write failing tests for the pipeline handler**

Create `tests/test_app.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
pytest tests/test_app.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'app'`.

- [ ] **Step 3: Create `app.py` with `run_pipeline` only**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
pytest tests/test_app.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```powershell
git add app.py tests/test_app.py
git commit -m "feat(app): add run_pipeline orchestrating transcribe + unload + summarize"
```

---

## Task 9: Gradio UI assembly

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Append Gradio UI to `app.py`**

Append the following to the end of `app.py`:

```python
def _ui_handler(audio_path: Optional[str], style: str, length: str):
    """Gradio adapter: wraps run_pipeline with module-level singletons and progress text."""
    yield "Transcribing…", ""

    transcript, summary = run_pipeline(
        audio_path,
        style=style,
        length=length,
        transcriber=_TRANSCRIBER,
        summarizer=_SUMMARIZER,
    )

    yield transcript, summary


def build_ui():
    """Construct the Gradio Blocks app. Kept as a function so tests can import it."""
    import gradio as gr

    with gr.Blocks(title="Audio Summarizer") as ui:
        gr.Markdown(
            "# Audio Summarizer\n"
            "Upload an audio file. Transcription runs first on GPU, then the model "
            "unloads and Ollama (`gemma4:e4b`) summarizes the transcript."
        )
        audio = gr.File(
            label="Audio file",
            file_types=[".mp3", ".wav", ".m4a", ".mp4", ".webm", ".ogg", ".flac"],
            type="filepath",
        )
        with gr.Row():
            style = gr.Dropdown(
                ["bullets", "paragraphs"], value="bullets", label="Summary style"
            )
            length = gr.Dropdown(
                ["short", "medium", "long"], value="medium", label="Summary length"
            )
        submit = gr.Button("Transcribe & Summarize", variant="primary")
        with gr.Row():
            transcript_box = gr.Textbox(
                label="Transcript",
                lines=20,
                show_copy_button=True,
                interactive=True,
            )
            summary_box = gr.Markdown(label="Summary")
        submit.click(
            _ui_handler,
            inputs=[audio, style, length],
            outputs=[transcript_box, summary_box],
        )
    return ui


# Module-level singletons so the model is loaded at most once per process.
_TRANSCRIBER = Transcriber()
_SUMMARIZER = Summarizer()


if __name__ == "__main__":
    build_ui().launch()
```

- [ ] **Step 2: Verify `app.py` imports cleanly**

```powershell
python -c "import app; print('ok')"
```

Expected: `ok` (no exceptions). If `faster-whisper` or `gradio` is missing, install dependencies first: `pip install -r requirements.txt`.

- [ ] **Step 3: Confirm existing tests still pass**

```powershell
pytest -v
```

Expected: all 15 tests pass.

- [ ] **Step 4: Manual smoke test — UI launches**

```powershell
python app.py
```

Open the URL Gradio prints (typically http://127.0.0.1:7860). Confirm the page renders with the upload control, two dropdowns, the submit button, and the two output panes. Stop the server with Ctrl+C.

Expected: UI loads with no errors in the console.

- [ ] **Step 5: Commit**

```powershell
git add app.py
git commit -m "feat(app): add Gradio UI wired to the pipeline"
```

---

## Task 10: README finalization and end-to-end smoke test

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace `README.md` with the full version**

```markdown
# Audio Summarization

Local web app that transcribes uploaded audio with `faster-whisper` and summarizes it with Ollama (`gemma4:e4b`). The two models run sequentially so they fit in a single 10 GB GPU.

## Requirements

- Python 3.10 or newer
- [Ollama](https://ollama.com) running locally, with the model pulled:
  ```powershell
  ollama pull gemma4:e4b
  ollama serve
  ```
- NVIDIA GPU with at least 10 GB VRAM (CPU fallback works but is much slower)

## Install

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If you have a GPU but `faster-whisper` falls back to CPU, install a CUDA-enabled build of `ctranslate2` per the [faster-whisper README](https://github.com/SYSTRAN/faster-whisper#requirements).

## Run

```powershell
python app.py
```

Open the URL Gradio prints (typically http://127.0.0.1:7860). Upload an audio file, pick a summary style and length, and click **Transcribe & Summarize**.

Supported formats: `.mp3`, `.wav`, `.m4a`, `.mp4`, `.webm`, `.ogg`, `.flac`.

## How it works

1. **Transcribe** — `faster-whisper` `large-v3` (int8_float16) runs on the GPU (~3 GB VRAM).
2. **Unload** — the Whisper model is released and CUDA cache is emptied.
3. **Summarize** — the transcript is sent to Ollama (`gemma4:e4b`, ~9.6 GB VRAM). For long transcripts a map-reduce pass chunks the text at ~6000 tokens with 200-token overlap, summarizes each chunk, and then combines them into a structured final summary.

## Tests

```powershell
pytest
```

The test suite mocks `faster-whisper` and Ollama, so it runs without a GPU or a running Ollama server.

## Troubleshooting

| Symptom | Fix |
|---|---|
| "Ollama is not running" | Start the server: `ollama serve` |
| "Model `gemma4:e4b` not found" | Pull it: `ollama pull gemma4:e4b` |
| GPU OOM at startup | The app falls back to CPU automatically; expect slower transcription |
| Transcript is empty | Audio likely has no speech, or the file is corrupt |

## Project layout

```
.
├── app.py              # Gradio UI + pipeline handler
├── transcriber.py      # faster-whisper wrapper
├── summarizer.py       # Ollama wrapper + chunking
├── requirements.txt
├── tests/
│   ├── test_app.py
│   ├── test_summarizer.py
│   └── test_transcriber.py
└── docs/superpowers/
    ├── specs/
    └── plans/
```
```

- [ ] **Step 2: Run the full test suite once more**

```powershell
pytest -v
```

Expected: all 15 tests pass.

- [ ] **Step 3: End-to-end manual smoke test**

Pre-conditions: Ollama is running (`ollama serve` in another terminal), and `gemma4:e4b` is pulled.

1. Start the app: `python app.py`
2. Open the printed URL.
3. Upload a short (≤ 1 minute) audio file containing speech.
4. Click **Transcribe & Summarize**.
5. Confirm the transcript pane fills with text and the summary pane fills with a markdown summary having `## TL;DR`, `## Key Points`, and `## Notable Details` sections.

If the transcript appears but the summary shows an Ollama error, follow the troubleshooting table.

- [ ] **Step 4: Commit**

```powershell
git add README.md
git commit -m "docs: complete README with install, run, and troubleshooting"
```

---

## Self-Review

**Spec coverage:**
- Architecture (sequential pipeline): Tasks 2, 3, 7, 8 — covered.
- Transcriber component (lazy load, CPU fallback, transcribe, unload): Tasks 2, 3 — covered.
- Summarizer component (prompt, chunking, Ollama call, map-reduce): Tasks 4, 5, 6, 7 — covered.
- Gradio UI (upload, style/length, transcript + summary panes): Tasks 8, 9 — covered.
- Error handling matrix (Ollama not running, model missing, OOM fallback, empty transcript, timeout): Task 2 (OOM), Task 6 (Ollama errors), Task 8 (empty transcript), Task 9 (UI surface) — covered.
- Dependencies (faster-whisper, gradio, requests, tiktoken): Task 1 — covered.
- Project layout: Task 1 (skeleton) + each module task — covered.
- Testing approach: pytest with mocks throughout, manual smoke in Tasks 9 and 10 — covered.
- Out-of-scope items (diarization, microphone, YouTube, batch, persistence): explicitly excluded; no tasks for them. ✓

**Placeholder scan:** No TBDs, no "implement later", no "similar to Task N". Every code block is complete and runnable.

**Type / name consistency:**
- `Transcriber(model_size, device, compute_type)` — matches between Task 2 and Task 3.
- `TranscriptionResult(text, language, segments)` — used identically in Tasks 3, 8.
- `Summarizer.chunk_text`, `_build_prompt(transcript, style, length, is_chunk)`, `_call_ollama`, `summarize(transcript, style, length)` — consistent across Tasks 4–7 and referenced correctly in Task 8.
- `SummarizerError` — defined in Task 4, imported in Task 8.
- `run_pipeline(audio_path, style, length, transcriber, summarizer)` — defined in Task 8, called in Task 9.

No issues found.
