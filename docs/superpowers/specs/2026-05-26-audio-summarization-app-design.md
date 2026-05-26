# Audio Summarization App — Design

**Date:** 2026-05-26
**Status:** Approved (awaiting implementation plan)

## Goal

A local web app that accepts an uploaded audio file, transcribes it to text, and produces a structured summary. Runs entirely on the user's machine using a local GPU (10GB VRAM) plus an existing Ollama install.

## Target user

A single user on a Windows machine with:
- Ollama installed, with `gemma4:e4b` (9.6 GB) available locally.
- 10 GB GPU VRAM.
- Audio files (meetings, lectures, podcasts) in common formats — multilingual content.

## Constraints

- **VRAM budget:** 10 GB. Whisper and Ollama cannot be GPU-resident simultaneously without OOM, so the pipeline must run them sequentially with an explicit unload between stages.
- **No cloud APIs.** Everything runs locally.
- **Multilingual** transcription required.

## Architecture

Single-process Python application with a Gradio web UI. The pipeline is strictly sequential to respect the VRAM budget:

```
Upload audio file
      │
      ▼
[Stage 1] faster-whisper transcribes on GPU (~3 GB VRAM)
      │
      ▼
[Unload]  Whisper model released, CUDA cache emptied
      │
      ▼
[Stage 2] Ollama (gemma4:e4b) summarizes via HTTP (~9.6 GB VRAM)
      │
      ▼
Display transcript + summary in the browser
```

Ollama runs as a separate process; its GPU footprint is independent of the Python app's. Unloading Whisper before calling Ollama avoids the two models contending for the same GPU.

## Components

### `transcriber.py` — faster-whisper wrapper

Responsibilities:
- Lazily load the Whisper model on first use.
- Transcribe an audio file path to text (returns the full transcript as a single string; optionally segments with timestamps for future use).
- Explicitly unload from GPU and free CUDA cache.

Defaults:
- Model: `large-v3`
- `compute_type="int8_float16"` (≈3 GB VRAM, best quality at this VRAM tier)
- `device="cuda"` with CPU fallback if CUDA init fails

Interface:
```python
class Transcriber:
    def __init__(self, model_size: str = "large-v3", device: str = "cuda"): ...
    def transcribe(self, audio_path: str) -> TranscriptionResult: ...
    def unload(self) -> None: ...
```

`TranscriptionResult` carries `text: str`, `language: str`, and `segments: list` (timestamped chunks, kept for possible future features).

### `summarizer.py` — Ollama wrapper + chunking

Responsibilities:
- Send prompts to Ollama's HTTP API at `http://localhost:11434/api/generate`.
- Handle long transcripts via map-reduce summarization:
  1. **Map:** split transcript into chunks of ~6000 tokens with ~200-token overlap; summarize each chunk independently.
  2. **Reduce:** concatenate the chunk summaries and produce a final structured summary.
- Short transcripts (single chunk) skip the reduce step.

Token counting uses `tiktoken` with the `cl100k_base` encoder as a rough proxy — gemma's tokenizer differs, but for chunk-size planning the approximation is sufficient.

Output structure (markdown):
```
## TL;DR
<2–3 sentence overview>

## Key Points
- bullet
- bullet

## Notable Details
<paragraph or further bullets, depending on style setting>
```

Interface:
```python
class Summarizer:
    def __init__(self, model: str = "gemma4:e4b", base_url: str = "http://localhost:11434"): ...
    def summarize(self, transcript: str, style: str, length: str) -> str: ...
```

`style ∈ {"bullets", "paragraphs"}`, `length ∈ {"short", "medium", "long"}`.

### `app.py` — Gradio UI

Layout (top to bottom):
1. Title and short description.
2. File upload component (accepts `.mp3`, `.wav`, `.m4a`, `.mp4`, `.webm`, `.ogg`, `.flac`).
3. Settings row: summary style dropdown, summary length dropdown.
4. "Transcribe & Summarize" button.
5. Progress text ("Transcribing…", "Summarizing…").
6. Two side-by-side panes: **Transcript** (editable textarea) and **Summary** (markdown render).
7. Copy and download buttons under each pane.

The handler function is a single generator that yields progress updates so the UI feels responsive: yield "Transcribing…", do transcription, yield transcript + "Summarizing…", do summarization, yield final.

## Data flow

1. User drops `meeting.mp3` into the upload widget.
2. User clicks **Transcribe & Summarize**.
3. Handler calls `Transcriber.transcribe(path)` → transcript string.
4. Handler calls `Transcriber.unload()`.
5. Handler calls `Summarizer.summarize(transcript, style, length)` → summary markdown.
6. UI updates both panes.

No persistence — transcript and summary live only in the UI session. Users can download as `.txt` / `.md` if they want to keep them.

## Error handling

| Failure mode | Detection | User-visible behavior |
|---|---|---|
| Ollama not running | `requests.ConnectionError` on first call | Show error: "Ollama is not running. Start it with `ollama serve` and try again." |
| Model not pulled | Ollama API returns 404 / "model not found" | Show error: "Model `gemma4:e4b` not found. Run `ollama pull gemma4:e4b`." |
| CUDA OOM on Whisper load | `RuntimeError` during model init | Retry with `medium` model; if that also fails, retry on CPU; show a banner noting the fallback. |
| Unsupported audio format | Gradio's accept list rejects at upload | Native Gradio behavior; supported list shown in the upload component. |
| Empty / silent audio | Transcript is empty or whitespace-only | Skip summarization, show: "No speech detected in audio." |
| Ollama timeout | `requests.Timeout` (5 min limit) | Show error with suggestion to try a shorter clip or smaller model. |

## Performance expectations

Rough numbers for planning (real numbers will vary):
- Whisper `large-v3` int8_float16 on a mid-range 10 GB GPU: ~0.1–0.3× realtime (10 min audio → 1–3 min transcribe).
- Gemma `gemma4:e4b` summarization: a few seconds for short transcripts, up to a minute for long map-reduce passes.

## Out of scope (v1)

These are deliberately excluded; they can be added in future iterations:
- Speaker diarization (pyannote-audio + HuggingFace token, extra VRAM).
- Live microphone recording.
- YouTube / URL input.
- Batch / multi-file processing.
- Persistent history or database.
- Authentication or multi-user features.
- Cloud deployment.

## Dependencies

```
faster-whisper>=1.0
gradio>=4.0
requests>=2.31
tiktoken>=0.5
```

CUDA / cuDNN / CTranslate2 GPU support are required for GPU transcription. CPU fallback works without them.

## Project layout

```
AudioSummarization/
├── app.py
├── transcriber.py
├── summarizer.py
├── requirements.txt
├── README.md
└── docs/
    └── superpowers/
        └── specs/
            └── 2026-05-26-audio-summarization-app-design.md
```

## Testing approach

- **`transcriber.py`**: unit test against a short fixture WAV file; assert language detection and non-empty transcript. Skip if no GPU available.
- **`summarizer.py`**: unit test the chunking logic with mocked Ollama responses. Integration test against a running Ollama if available; otherwise skip.
- **`app.py`**: smoke test by launching Gradio in test mode and invoking the handler with a fixture file. Manual UI verification before declaring done.
