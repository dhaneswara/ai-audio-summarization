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
