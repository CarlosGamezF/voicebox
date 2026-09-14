"""
STT (Speech-to-Text) module - delegates to backend abstraction layer.
"""

import logging

from sqlalchemy.orm import Session

from ..backends import get_stt_backend, STTBackend
from .settings import get_capture_settings

logger = logging.getLogger(__name__)


def get_whisper_model() -> STTBackend:
    """
    Get STT backend instance (MLX or PyTorch based on platform).
    
    Returns:
        STT backend instance
    """
    return get_stt_backend()


def unload_whisper_model():
    """Unload Whisper model to free memory."""
    backend = get_stt_backend()
    backend.unload_model()


def resolve_transcription_model(
    requested: str | None,
    db: Session,
    whisper: STTBackend | None = None,
) -> str:
    """Pick the Whisper size for a transcription request.

    An explicit ``requested`` size wins. Otherwise the persisted STT setting
    (``stt_model`` in the capture settings) applies whenever that model is
    loaded or already downloaded, so the sample "Transcribe" button, REST
    callers and the MCP tool use the model the user configured. If the
    configured model is not on disk yet, keep the backend's current size and
    say so: a transcription must not turn into a surprise download (the
    0.5.0 desktop UI treats the 202 "downloading" reply as a success) or,
    on MCP, into an error.
    """
    if requested:
        return requested
    configured = get_capture_settings(db).stt_model
    if whisper is None or transcription_model_available(whisper, configured):
        return configured
    logger.warning(
        "Configured STT model %r is not downloaded; transcribing with %r instead. "
        "Download it in Settings > Models to use it here.",
        configured,
        whisper.model_size,
    )
    return whisper.model_size


def transcription_model_available(whisper: STTBackend, size: str) -> bool:
    """True when ``size`` is loaded or already on disk, so using it downloads nothing."""
    if whisper.is_loaded() and whisper.model_size == size:
        return True
    is_cached = getattr(whisper, "_is_model_cached", None)
    return bool(is_cached is not None and is_cached(size))
