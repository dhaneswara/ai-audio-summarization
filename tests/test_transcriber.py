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


def test_transcribe_reports_progress_per_segment():
    """progress_cb is invoked with segment.end / duration after each segment."""
    from transcriber import Transcriber

    seg1 = MagicMock(); seg1.text = "a"; seg1.end = 5.0
    seg2 = MagicMock(); seg2.text = "b"; seg2.end = 10.0
    info = MagicMock(); info.language = "en"; info.duration = 10.0

    fake_model = MagicMock()
    fake_model.transcribe.return_value = (iter([seg1, seg2]), info)

    calls: list[float] = []
    with patch("transcriber.WhisperModel", return_value=fake_model):
        t = Transcriber()
        t.transcribe("ignored.wav", progress_cb=calls.append)

    assert calls == [0.5, 1.0]


def test_transcribe_falls_back_to_cpu_when_runtime_cuda_error():
    """A cuBLAS/cuDNN runtime error during transcription should retry on CPU."""
    from transcriber import Transcriber

    seg = MagicMock(); seg.text = "ok"; seg.end = 1.0
    info = MagicMock(); info.language = "en"; info.duration = 1.0

    gpu_model = MagicMock()
    gpu_model.transcribe.side_effect = RuntimeError(
        "Library cublas64_12.dll is not found or cannot be loaded"
    )
    cpu_model = MagicMock()
    cpu_model.transcribe.return_value = (iter([seg]), info)

    with patch("transcriber.WhisperModel", side_effect=[gpu_model, cpu_model]) as MockModel:
        t = Transcriber()
        result = t.transcribe("ignored.wav")

    assert result.text == "ok"
    assert t.device == "cpu"
    # Two constructions: the original GPU one, then the CPU retry.
    assert MockModel.call_count == 2
    assert MockModel.call_args_list[1].kwargs["device"] == "cpu"


def test_transcribe_does_not_swallow_non_cuda_errors():
    """Non-CUDA runtime errors should propagate, not trigger a CPU retry."""
    from transcriber import Transcriber

    gpu_model = MagicMock()
    gpu_model.transcribe.side_effect = RuntimeError("audio decode failed")

    with patch("transcriber.WhisperModel", return_value=gpu_model):
        t = Transcriber()
        with pytest.raises(RuntimeError, match="audio decode failed"):
            t.transcribe("ignored.wav")


def test_transcribe_without_progress_cb_still_works():
    """progress_cb is optional; transcribe must work without it."""
    from transcriber import Transcriber

    seg = MagicMock(); seg.text = "hi"; seg.end = 1.0
    info = MagicMock(); info.language = "en"; info.duration = 1.0
    fake_model = MagicMock()
    fake_model.transcribe.return_value = (iter([seg]), info)

    with patch("transcriber.WhisperModel", return_value=fake_model):
        t = Transcriber()
        result = t.transcribe("ignored.wav")

    assert result.text == "hi"
