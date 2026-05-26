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
        except (RuntimeError, OSError, ValueError):
            self._model = WhisperModel(
                self.model_size, device="cpu", compute_type="int8"
            )
            self.device = "cpu"

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
