"""
STT (Speech-to-Text) module - delegates to backend abstraction layer.
"""

from typing import Optional
from ..backends import get_stt_backend, STTBackend
from .settings import get_capture_settings


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


def resolve_transcription_model(requested: str | None, db) -> str:
    """Pick the Whisper size for a transcription request.

    An explicit ``requested`` size wins. Otherwise the persisted STT setting
    (``stt_model`` in the capture settings) applies, so the sample
    "Transcribe" button, REST callers and the MCP tool all use the model the
    user configured instead of whichever size happened to be loaded.
    """
    if requested:
        return requested
    return get_capture_settings(db).stt_model
