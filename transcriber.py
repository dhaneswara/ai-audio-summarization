"""Wraps faster-whisper for sequential GPU use with an explicit unload step."""

from __future__ import annotations

# Silence huggingface_hub noise *before* faster_whisper imports it. The
# symlink warning is Windows-specific and harmless; the HF_TOKEN suggestion
# only matters at high request volume, not for a one-time model download.
import os
import warnings

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings(
    "ignore",
    message=".*unauthenticated requests to the HF Hub.*",
)

from dataclasses import dataclass
from typing import Callable, Optional

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

    _CUDA_ERROR_KEYWORDS = ("cublas", "cudnn", "cuda", "library", "dll")

    def _is_cuda_runtime_error(self, exc: BaseException) -> bool:
        """Detect runtime errors that mean 'GPU is unusable, retry on CPU'."""
        msg = str(exc).lower()
        return any(kw in msg for kw in self._CUDA_ERROR_KEYWORDS)

    def transcribe(
        self,
        audio_path: str,
        progress_cb: Optional[Callable[[float], None]] = None,
    ) -> TranscriptionResult:
        """Transcribe audio, optionally reporting progress as a 0–1 fraction.

        Faster-whisper exposes segments via a generator and the total audio
        duration on the `info` object. Each segment carries its end timestamp,
        so we can report progress as `segment.end / total_duration` as we go.

        If a GPU transcription fails at runtime with a CUDA / cuBLAS / cuDNN
        error (typical when the right CUDA libraries are not installed), the
        model is unloaded and the transcription retried on CPU.
        """
        self._load()
        try:
            return self._transcribe_now(audio_path, progress_cb)
        except Exception as e:
            if self.device == "cpu" or not self._is_cuda_runtime_error(e):
                raise
            # GPU path failed mid-transcription; degrade to CPU and retry once.
            self.unload()
            self.device = "cpu"
            self.compute_type = "int8"
            self._load()
            return self._transcribe_now(audio_path, progress_cb)

    def _transcribe_now(
        self,
        audio_path: str,
        progress_cb: Optional[Callable[[float], None]],
    ) -> TranscriptionResult:
        assert self._model is not None
        segments_iter, info = self._model.transcribe(audio_path)
        duration = getattr(info, "duration", 0) or 0
        segments = []
        for seg in segments_iter:
            segments.append(seg)
            if progress_cb and duration > 0:
                progress_cb(min(seg.end / duration, 1.0))
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
