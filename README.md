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
