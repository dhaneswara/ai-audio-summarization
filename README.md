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
