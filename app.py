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
                buttons=["copy"],
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
