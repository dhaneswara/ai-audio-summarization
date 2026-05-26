"""Audio Summarization app — Gradio UI and pipeline handler."""

from __future__ import annotations

from typing import Generator, Optional

from summarizer import Summarizer, SummarizerError
from transcriber import Transcriber


def run_pipeline(
    audio_path: Optional[str],
    style: str,
    length: str,
    transcriber: Transcriber,
    summarizer: Summarizer,
    transcribe_only: bool = False,
) -> Generator[tuple[str, str], None, None]:
    """Transcribe an audio file, unload the model, then optionally summarize.

    Yields intermediate (transcript, status) tuples while progressing, and a
    final (transcript, summary) when complete. When transcribe_only is True
    the summarization step is skipped — useful for verifying the transcript
    against an audio source. Transcriber and summarizer errors are caught and
    surfaced as user-facing text so the UI never crashes.
    """
    if not audio_path:
        yield "", "Please upload an audio file."
        return

    yield "Transcribing…", ""

    try:
        result = transcriber.transcribe(audio_path)
    except Exception as e:
        yield "", f"Transcription failed: {e}"
        return

    transcript = result.text.strip()
    transcriber.unload()

    if not transcript:
        yield "", "No speech detected in audio."
        return

    if transcribe_only:
        yield transcript, "_Transcription complete. Summarization skipped._"
        return

    yield transcript, "Summarizing…"

    try:
        summary = summarizer.summarize(transcript, style=style, length=length)
    except SummarizerError as e:
        yield transcript, f"Summarization failed: {e}"
        return

    yield transcript, summary


def _ui_handler(
    audio_path: Optional[str],
    style: str,
    length: str,
    transcribe_only: bool,
):
    """Gradio adapter: forwards run_pipeline's progress yields to the UI."""
    yield from run_pipeline(
        audio_path,
        style=style,
        length=length,
        transcriber=_TRANSCRIBER,
        summarizer=_SUMMARIZER,
        transcribe_only=transcribe_only,
    )


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
        transcribe_only = gr.Checkbox(
            value=False,
            label="Transcribe only (skip summary)",
            info="Useful for verifying the transcript before generating a summary.",
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
            inputs=[audio, style, length, transcribe_only],
            outputs=[transcript_box, summary_box],
        )
    return ui


# Module-level singletons so the model is loaded at most once per process.
_TRANSCRIBER = Transcriber()
_SUMMARIZER = Summarizer()


if __name__ == "__main__":
    build_ui().launch()
