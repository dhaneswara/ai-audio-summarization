"""Audio Summarization app — Gradio UI and pipeline handler."""

from __future__ import annotations

from typing import Callable, Generator, Optional

import gradio as gr

from summarizer import Summarizer, SummarizerError
from transcriber import Transcriber


def run_pipeline(
    audio_path: Optional[str],
    style: str,
    length: str,
    transcriber: Transcriber,
    summarizer: Summarizer,
    transcribe_only: bool = False,
    high_quality: bool = False,
    progress_cb: Optional[Callable[[float], None]] = None,
) -> Generator[tuple[str, str], None, None]:
    """Transcribe an audio file, unload the model, then optionally summarize.

    Yields intermediate (transcript, status) tuples while progressing, and a
    final (transcript, summary) when complete. When transcribe_only is True
    the summarization step is skipped — useful for verifying the transcript
    against an audio source. Transcriber and summarizer errors are caught and
    surfaced as user-facing text so the UI never crashes.

    `progress_cb`, if given, is forwarded to the transcriber and called with a
    0–1 fraction as each Whisper segment completes.
    """
    if not audio_path:
        yield "", "Please upload an audio file."
        return

    yield "Transcribing…", ""

    try:
        result = transcriber.transcribe(
            audio_path, progress_cb=progress_cb, high_quality=high_quality
        )
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
    high_quality: bool,
    progress=gr.Progress(),
):
    """Gradio adapter: drives gr.Progress and forwards run_pipeline yields.

    Progress bar layout:
      0.00 → 0.90 : transcription (driven per Whisper segment via progress_cb)
      0.95        : summarization in progress
      1.00        : done
    """
    progress(0, desc="Preparing")

    def cb(frac: float) -> None:
        # Map Whisper's 0–1 segment progress into the 0–0.90 slice.
        progress(frac * 0.90, desc=f"Transcribing · {int(frac * 100)}%")

    yield_count = 0
    for chunk in run_pipeline(
        audio_path,
        style=style,
        length=length,
        transcriber=_TRANSCRIBER,
        summarizer=_SUMMARIZER,
        transcribe_only=transcribe_only,
        high_quality=high_quality,
        progress_cb=cb,
    ):
        yield_count += 1
        transcript, _status = chunk
        # The 2nd yield from run_pipeline is the transition point: transcription
        # is complete, summarization (if any) is about to begin. We only switch
        # the label when there's a real transcript AND we're going to summarize.
        if yield_count == 2 and transcript and not transcribe_only:
            progress(0.95, desc="Summarizing")
        yield chunk

    progress(1.0, desc="Done")


_BUTTON_LABEL_DEFAULT = "Transcribe & Summarize"
_BUTTON_LABEL_TRANSCRIBE_ONLY = "Transcribe Only"


_HEADER_HTML = """
<div id="masthead">
  <div class="kicker">Local · Faster-Whisper × Ollama</div>
  <h1 class="title">Audio <em>Summarizer</em></h1>
  <p class="lede">
    Drop an audio file below. Transcription runs first on your GPU; the
    Whisper model unloads, and Gemma then summarizes the transcript.
    Two models, one card, ten gigabytes.
  </p>
</div>
"""


_CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght,SOFT@0,9..144,400..600,100;1,9..144,400..500,100&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body {
    background: #0f0d0a !important;
    color: #f0e9d8 !important;
    font-family: 'JetBrains Mono', ui-monospace, monospace !important;
    min-height: 100vh;
}

body {
    background-image:
        radial-gradient(ellipse 900px 500px at 15% -10%, rgba(232, 151, 69, 0.07), transparent 60%),
        radial-gradient(ellipse 700px 400px at 100% 110%, rgba(232, 151, 69, 0.04), transparent 60%) !important;
    background-attachment: fixed !important;
    background-color: #0f0d0a !important;
}

/* Gradio 6 wraps content in .main.fillable.app which caps width — force-expand. */
.main, .main.fillable, .main.app, .fillable.app {
    width: 100% !important;
    max-width: 1080px !important;
    margin: 0 auto !important;
    background: transparent !important;
    padding: 0 !important;
}

main.contain {
    max-width: none !important;
    width: 100% !important;
    padding: 0 !important;
}

.gradio-container {
    max-width: 1080px !important;
    width: 100% !important;
    margin: 0 auto !important;
    padding: 3.5rem 2rem 5rem !important;
    background: transparent !important;
}

/* ---------- Masthead ---------- */
#masthead { margin-bottom: 3rem; }

#masthead .kicker {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.22em;
    color: #c9925a;
    margin-bottom: 1.5rem;
    opacity: 0.85;
}

#masthead .title {
    font-family: 'Fraunces', Georgia, serif;
    font-weight: 400;
    font-variation-settings: "SOFT" 100, "opsz" 144;
    font-size: clamp(3rem, 7vw, 5rem);
    line-height: 0.95;
    letter-spacing: -0.035em;
    margin: 0 0 1.5rem 0;
    color: #f0e9d8;
}

#masthead .title em {
    font-style: italic;
    color: #e89745;
    font-weight: 400;
}

#masthead .lede {
    font-family: 'Fraunces', Georgia, serif;
    font-weight: 400;
    font-size: 1.125rem;
    line-height: 1.55;
    max-width: 54ch;
    color: #b8ad96;
    margin: 0;
}

/* ---------- Section rules ---------- */
.gradio-container > .block { margin-bottom: 1rem !important; }

/* All labels go uppercase mono */
.block .label-wrap span,
.block > label > span,
label.svelte-1b6s6s span,
.checkbox-wrap label span,
.gr-form > label > span {
    font-family: 'JetBrains Mono', ui-monospace, monospace !important;
    font-size: 0.68rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.18em !important;
    color: #8a8170 !important;
    font-weight: 500 !important;
}

/* ---------- Upload ---------- */
#audio-upload { border-radius: 6px !important; }

#audio-upload .file-preview,
#audio-upload .upload-container,
#audio-upload .wrap {
    background: #1a1611 !important;
    border: 1px dashed #3a3128 !important;
    border-radius: 6px !important;
    min-height: 110px !important;
    transition: border-color 0.18s ease, background 0.18s ease;
}

#audio-upload:hover .wrap,
#audio-upload:hover .upload-container {
    border-color: #e89745 !important;
    background: #221c15 !important;
}

/* ---------- Controls row ---------- */
#controls-row { gap: 1rem !important; }

#controls-row .form,
#controls-row .wrap,
.gradio-container select,
.gradio-container input[type="text"],
.gradio-container textarea {
    background: #1a1611 !important;
    border: 1px solid #2a241d !important;
    color: #f0e9d8 !important;
    border-radius: 4px !important;
    font-family: 'JetBrains Mono', ui-monospace, monospace !important;
}

/* Radio group: pointer cursor everywhere on the option pills and dots. */
#controls-row label,
#controls-row input[type="radio"] {
    cursor: pointer !important;
}

/* Amber selected accent on the radio dot. */
#controls-row input[type="radio"] {
    accent-color: #e89745;
}

.gradio-container select:focus,
.gradio-container input:focus,
.gradio-container textarea:focus {
    border-color: #e89745 !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(232, 151, 69, 0.12) !important;
}

/* ---------- Checkbox rows ---------- */
#transcribe-only-row { margin: 0.5rem 0 0.25rem 0 !important; }
#high-quality-row { margin: 0 0 1.5rem 0 !important; }

#transcribe-only-row .checkbox-wrap,
#transcribe-only-row label,
#high-quality-row .checkbox-wrap,
#high-quality-row label {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

#transcribe-only-row input[type="checkbox"],
#high-quality-row input[type="checkbox"] {
    accent-color: #e89745 !important;
    width: 16px !important;
    height: 16px !important;
}

#transcribe-only-row .label-wrap > span,
#transcribe-only-row label > span,
#high-quality-row .label-wrap > span,
#high-quality-row label > span {
    color: #d6cdb4 !important;
    text-transform: none !important;
    letter-spacing: 0.01em !important;
    font-family: 'JetBrains Mono', ui-monospace, monospace !important;
    font-size: 0.82rem !important;
}

#transcribe-only-row .info-text,
#transcribe-only-row .gr-info,
#high-quality-row .info-text,
#high-quality-row .gr-info {
    font-family: 'Fraunces', serif !important;
    font-size: 0.95rem !important;
    color: #8a8170 !important;
    font-style: italic !important;
}

/* ---------- Submit button ---------- */
#submit-row { margin-top: 0.5rem !important; }

button#submit-btn,
#submit-btn {
    background: #e89745 !important;
    color: #0f0d0a !important;
    border: none !important;
    border-radius: 4px !important;
    font-family: 'JetBrains Mono', ui-monospace, monospace !important;
    font-weight: 600 !important;
    font-size: 0.78rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.18em !important;
    padding: 1.25rem 2rem !important;
    min-height: 56px !important;
    height: auto !important;
    width: 100% !important;
    transition: all 0.2s cubic-bezier(0.2, 0.8, 0.2, 1) !important;
    cursor: pointer !important;
    line-height: 1 !important;
}

button#submit-btn:hover,
#submit-btn:hover {
    background: #f4a854 !important;
    box-shadow: 0 0 32px rgba(232, 151, 69, 0.25) !important;
    transform: translateY(-1px);
}

button#submit-btn:active { transform: translateY(0); }

/* ---------- Output panes ---------- */
#output-grid {
    margin-top: 2.5rem !important;
    gap: 1rem !important;
}

#output-grid > div { min-width: 0 !important; }

#transcript-pane, #summary-pane {
    background: #1a1611 !important;
    border: 1px solid #2a241d !important;
    border-radius: 6px !important;
    padding: 1.25rem !important;
}

/* Pad the markdown content so it visually fills the same space the
   transcript's textarea occupies, with internal scroll if needed. */
#summary-pane .prose,
#summary-pane [class*="prose"] {
    max-height: 470px !important;
    overflow-y: auto !important;
}

/* Inner textareas — same treatment in both panes so they read as a pair. */
#transcript-pane textarea,
#summary-pane textarea {
    background: #0f0d0a !important;
    border: 1px solid #2a241d !important;
    border-radius: 4px !important;
    color: #f0e9d8 !important;
    font-family: 'JetBrains Mono', ui-monospace, monospace !important;
    font-size: 0.88rem !important;
    line-height: 1.65 !important;
    padding: 0.75rem !important;
}

#transcript-pane textarea:focus,
#summary-pane textarea:focus {
    border-color: #e89745 !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(232, 151, 69, 0.12) !important;
}

#summary-pane {
    color: #e8e2d4 !important;
}

#summary-pane .prose,
#summary-pane .markdown {
    font-family: 'Fraunces', Georgia, serif !important;
    font-size: 1.05rem !important;
    line-height: 1.65 !important;
    color: #e8e2d4 !important;
}

#summary-pane h2,
#summary-pane .prose h2,
#summary-pane .markdown h2 {
    font-family: 'JetBrains Mono', ui-monospace, monospace !important;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.18em !important;
    color: #e89745 !important;
    margin: 1.6rem 0 0.6rem 0 !important;
    padding-bottom: 0.4rem !important;
    border-bottom: 1px solid #2a241d !important;
}

#summary-pane h2:first-child,
#summary-pane .prose h2:first-child { margin-top: 0 !important; }

#summary-pane ul, #summary-pane ol {
    padding-left: 1.2rem !important;
}

#summary-pane li {
    margin-bottom: 0.4rem !important;
}

#summary-pane em { color: #8a8170 !important; }
#summary-pane code,
#summary-pane .prose code {
    font-family: 'JetBrains Mono', ui-monospace, monospace !important;
    background: #241f1a !important;
    color: #e89745 !important;
    padding: 0.1rem 0.4rem !important;
    border-radius: 3px !important;
    font-size: 0.85em !important;
}

/* ---------- Hide Gradio footer chrome ---------- */
footer { display: none !important; }
.gradio-container .built-with,
.gradio-container > .footer { display: none !important; }

/* ---------- Page-load motion ---------- */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

#masthead { animation: fadeUp 0.7s cubic-bezier(0.2, 0.8, 0.2, 1) backwards; animation-delay: 0.05s; }
#audio-upload { animation: fadeUp 0.7s cubic-bezier(0.2, 0.8, 0.2, 1) backwards; animation-delay: 0.18s; }
#controls-row { animation: fadeUp 0.7s cubic-bezier(0.2, 0.8, 0.2, 1) backwards; animation-delay: 0.26s; }
#transcribe-only-row { animation: fadeUp 0.7s cubic-bezier(0.2, 0.8, 0.2, 1) backwards; animation-delay: 0.32s; }
#high-quality-row { animation: fadeUp 0.7s cubic-bezier(0.2, 0.8, 0.2, 1) backwards; animation-delay: 0.36s; }
#submit-row { animation: fadeUp 0.7s cubic-bezier(0.2, 0.8, 0.2, 1) backwards; animation-delay: 0.42s; }
#output-grid { animation: fadeUp 0.7s cubic-bezier(0.2, 0.8, 0.2, 1) backwards; animation-delay: 0.46s; }
"""


def _make_theme():
    """Build the Gradio theme."""
    return gr.themes.Base(
        primary_hue=gr.themes.Color(
            c50="#fdf6ec", c100="#fae6c5", c200="#f5cd8a",
            c300="#efb155", c400="#e89745", c500="#dc7d2a",
            c600="#b3621f", c700="#8a4a18", c800="#5e2f0e",
            c900="#3a1c08", c950="#1f0e04",
        ),
        neutral_hue=gr.themes.Color(
            c50="#f5f1e8", c100="#e8e2d4", c200="#cdc5b5",
            c300="#a89e88", c400="#8a8170", c500="#6b6354",
            c600="#4d4639", c700="#3a3128", c800="#241f1a",
            c900="#1a1611", c950="#0f0d0a",
        ),
        font=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
    ).set(
        body_background_fill="#0f0d0a",
        body_text_color="#f0e9d8",
        body_text_color_subdued="#8a8170",
        background_fill_primary="#1a1611",
        background_fill_secondary="#241f1a",
        border_color_primary="#2a241d",
        border_color_accent="#3a3128",
        block_background_fill="#1a1611",
        block_border_color="#2a241d",
        block_border_width="1px",
        block_radius="6px",
        block_label_text_color="#8a8170",
        block_label_background_fill="#1a1611",
        block_title_text_color="#8a8170",
        button_primary_background_fill="#e89745",
        button_primary_background_fill_hover="#f4a854",
        button_primary_text_color="#0f0d0a",
        button_primary_border_color="#e89745",
        button_secondary_background_fill="#241f1a",
        button_secondary_text_color="#f0e9d8",
        button_secondary_border_color="#3a3128",
        input_background_fill="#0f0d0a",
        input_border_color="#2a241d",
        input_border_color_focus="#e89745",
        checkbox_background_color="#0f0d0a",
        checkbox_background_color_selected="#e89745",
        checkbox_border_color="#3a3128",
        checkbox_border_color_focus="#e89745",
    )


def build_ui():
    """Construct the Gradio Blocks app. Kept as a function so tests can import it."""
    with gr.Blocks(title="Audio Summarizer") as ui:
        gr.HTML(_HEADER_HTML)

        audio = gr.File(
            label="Audio file",
            file_types=[".mp3", ".wav", ".m4a", ".mp4", ".webm", ".ogg", ".flac"],
            type="filepath",
            elem_id="audio-upload",
        )

        with gr.Row(elem_id="controls-row"):
            style = gr.Radio(
                ["bullets", "paragraphs"],
                value="bullets",
                label="Summary style",
            )
            length = gr.Radio(
                ["short", "medium", "long"],
                value="medium",
                label="Summary length",
            )

        with gr.Row(elem_id="transcribe-only-row"):
            transcribe_only = gr.Checkbox(
                value=False,
                label="Transcribe only — skip summarization",
                info="Use this to verify the transcript before generating a summary.",
            )

        with gr.Row(elem_id="high-quality-row"):
            high_quality = gr.Checkbox(
                value=False,
                label="Higher accuracy · slower",
                info="Enables VAD silence filtering, beam_size=10, and float16 compute. Cleaner transcripts but roughly 2× slower.",
            )

        with gr.Row(elem_id="submit-row"):
            submit = gr.Button(
                _BUTTON_LABEL_DEFAULT,
                variant="primary",
                elem_id="submit-btn",
            )

        with gr.Row(elem_id="output-grid", equal_height=True):
            transcript_box = gr.Textbox(
                label="Transcript",
                lines=18,
                max_lines=18,
                buttons=["copy"],
                interactive=True,
                placeholder="Your transcript will appear here…",
                elem_id="transcript-pane",
            )
            summary_box = gr.Textbox(
                label="Summary",
                lines=18,
                max_lines=18,
                buttons=["copy"],
                interactive=False,
                placeholder="Your summary will appear here…",
                elem_id="summary-pane",
            )

        def _sync_button_label(only: bool):
            return gr.update(
                value=_BUTTON_LABEL_TRANSCRIBE_ONLY if only else _BUTTON_LABEL_DEFAULT
            )

        transcribe_only.change(
            _sync_button_label,
            inputs=transcribe_only,
            outputs=submit,
        )

        submit.click(
            _ui_handler,
            inputs=[audio, style, length, transcribe_only, high_quality],
            outputs=[transcript_box, summary_box],
        )
    return ui


# Module-level singletons so the model is loaded at most once per process.
_TRANSCRIBER = Transcriber()
_SUMMARIZER = Summarizer()


if __name__ == "__main__":
    build_ui().launch(theme=_make_theme(), css=_CUSTOM_CSS)
