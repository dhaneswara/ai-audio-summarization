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

### GPU acceleration on Windows

`faster-whisper` needs cuBLAS and cuDNN at runtime. They're listed in `requirements.txt` as Windows-only dependencies, so `pip install -r requirements.txt` pulls them in automatically — no admin rights, no CUDA Toolkit needed.

`transcriber.py` registers the DLL directories at import time so CTranslate2 can find them. If you ever see `Library cublas64_12.dll is not found`, the libs likely aren't installed; run:

```powershell
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

If you'd rather not install ~1.3 GB of CUDA libs, the app falls back to CPU at transcription time automatically. CPU is roughly 3–5× slower than a 10 GB GPU on `large-v3`, but produces identical output.

## Run

```powershell
python app.py
```

Open the URL Gradio prints (typically http://127.0.0.1:7860). Upload an audio file, pick a summary style and length, and click **Transcribe & Summarize**. A progress bar tracks each Whisper segment as it lands.

Supported formats: `.mp3`, `.wav`, `.m4a`, `.mp4`, `.webm`, `.ogg`, `.flac`.

### UI options

- **Summary style** — `bullets` or `paragraphs`. Affects the *Notable Details* section of the summary.
- **Summary length** — `short` / `medium` / `long`. Tunes the TL;DR length and the number of key-point bullets.
- **Transcribe only — skip summarization** — runs Whisper but leaves Ollama untouched. Useful for sanity-checking the transcript before trusting any summary built from it. The submit button label switches to **Transcribe Only** when this is on.
- **Higher accuracy · slower** — bundles three Whisper accuracy tweaks:
  - `vad_filter=True` (Silero VAD strips silence segments to prevent hallucinations on quiet stretches)
  - `beam_size=10` (wider decoding search, vs. the default 5)
  - `compute_type=float16` (full half-precision instead of int8_float16; uses ~5 GB VRAM instead of ~3 GB)

  Roughly 2× slower in exchange for cleaner output. Flipping the checkbox triggers a one-time model reload (~10–30 s on GPU) to switch compute types; subsequent runs in the same mode reuse the loaded model.

## How it works

1. **Transcribe** — `faster-whisper` `large-v3` (`int8_float16` by default, or `float16` in higher-accuracy mode) runs on the GPU (~3 GB / ~5 GB VRAM respectively). Progress is reported per Whisper segment via `segment.end / total_duration`.
2. **Unload** — the Whisper model is released and the CUDA cache is emptied so it doesn't compete with Ollama for VRAM.
3. **Summarize** — the transcript is sent to Ollama (`gemma4:e4b`, ~9.6 GB VRAM). For long transcripts a map-reduce pass chunks the text at ~6 000 tokens with 200-token overlap, summarizes each chunk, then combines them into a structured final summary (`## TL;DR` / `## Key Points` / `## Notable Details`).

If GPU transcription fails at runtime (typically a missing cuBLAS/cuDNN library), the Transcriber unloads, re-loads on CPU with `compute_type=int8`, and retries automatically — the user sees the same flow, just slower.

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
| "Library cublas64_12.dll is not found" | Install cuBLAS/cuDNN: `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12`. The app falls back to CPU if you skip this. |
| Transcript is empty | Audio likely has no speech, or the file is corrupt |
| Slow Whisper model download | Optional: set `HF_TOKEN` to your Hugging Face read token for higher rate limits. Only matters the first time — the model is cached at `~/.cache/huggingface/hub` after that. |

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
