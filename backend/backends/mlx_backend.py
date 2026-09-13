"""
MLX backend implementation for TTS and STT using mlx-audio.
"""

from typing import Optional, List, Tuple
import inspect
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

# PATCH: Import and apply offline patch BEFORE any huggingface_hub usage
# This prevents mlx_audio from making network requests when models are cached
from ..utils.hf_offline_patch import patch_huggingface_hub_offline, ensure_original_qwen_config_cached

patch_huggingface_hub_offline()
ensure_original_qwen_config_cached()

from . import TTSBackend, STTBackend, LANGUAGE_CODE_TO_NAME, WHISPER_HF_REPOS
from .base import is_model_cached, combine_voice_prompts as _combine_voice_prompts, model_load_progress
from ..utils.cache import get_cache_key, get_cached_voice_prompt, cache_voice_prompt
from ..utils.mlx_executor import run_in_mlx_executor


class MLXTTSBackend:
    """MLX-based TTS backend using mlx-audio."""

    def __init__(self, model_size: str = "1.7B"):
        self.model = None
        self.model_size = model_size
        self._current_model_size = None

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None

    def _get_model_path(self, model_size: str) -> str:
        """
        Get the MLX model path.

        Args:
            model_size: Model size (1.7B or 0.6B)

        Returns:
            HuggingFace Hub model ID for MLX
        """
        mlx_model_map = {
            "1.7B": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            "0.6B": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
        }

        if model_size not in mlx_model_map:
            raise ValueError(f"Unknown model size: {model_size}")

        hf_model_id = mlx_model_map[model_size]
        logger.info("Will download MLX model from HuggingFace Hub: %s", hf_model_id)

        return hf_model_id

    def _is_model_cached(self, model_size: str) -> bool:
        return is_model_cached(
            self._get_model_path(model_size),
            weight_extensions=(".safetensors", ".bin", ".npz"),
        )

    async def load_model_async(self, model_size: Optional[str] = None):
        """
        Lazy load the MLX TTS model.

        Args:
            model_size: Model size to load (1.7B or 0.6B)
        """
        if model_size is None:
            model_size = self.model_size

        # If already loaded with correct size, return
        if self.model is not None and self._current_model_size == model_size:
            return

        # Unload existing model if different size requested
        if self.model is not None and self._current_model_size != model_size:
            self.unload_model()

        # Run blocking load on the dedicated MLX thread
        await run_in_mlx_executor(self._load_model_sync, model_size)

    # Alias for compatibility
    load_model = load_model_async

    def _load_model_sync(self, model_size: str):
        """Synchronous model loading."""
        model_path = self._get_model_path(model_size)
        model_name = f"qwen-tts-{model_size}"
        is_cached = self._is_model_cached(model_size)

        with model_load_progress(model_name, is_cached):
            from mlx_audio.tts import load

            logger.info("Loading MLX TTS model %s...", model_size)

            self.model = load(model_path)

        self._current_model_size = model_size
        self.model_size = model_size
        logger.info("MLX TTS model %s loaded successfully", model_size)

    def unload_model(self):
        """Unload the model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
            self._current_model_size = None
            logger.info("MLX TTS model unloaded")

    async def create_voice_prompt(
        self,
        audio_path: str,
        reference_text: str,
        use_cache: bool = True,
    ) -> Tuple[dict, bool]:
        """
        Create voice prompt from reference audio.

        MLX backend stores voice prompt as a dict with audio path and text.
        The actual voice prompt processing happens during generation.

        Args:
            audio_path: Path to reference audio file
            reference_text: Transcript of reference audio
            use_cache: Whether to use cached prompt if available

        Returns:
            Tuple of (voice_prompt_dict, was_cached)
        """
        await self.load_model_async(None)

        # Check cache if enabled
        if use_cache:
            cache_key = get_cache_key(audio_path, reference_text)
            cached_prompt = get_cached_voice_prompt(cache_key)
            if cached_prompt is not None:
                # Return cached prompt (should be dict format)
                if isinstance(cached_prompt, dict):
                    # Validate that the cached audio file still exists
                    cached_audio_path = cached_prompt.get("ref_audio") or cached_prompt.get("ref_audio_path")
                    if cached_audio_path and Path(cached_audio_path).exists():
                        return cached_prompt, True
                    else:
                        # Cached file no longer exists, invalidate cache
                        logger.warning("Cached audio file not found: %s, regenerating prompt", cached_audio_path)

        # MLX voice prompt format - store audio path and text
        # The model will process this during generation
        voice_prompt_items = {
            "ref_audio": str(audio_path),
            "ref_text": reference_text,
        }

        # Cache if enabled
        if use_cache:
            cache_key = get_cache_key(audio_path, reference_text)
            cache_voice_prompt(cache_key, voice_prompt_items)

        return voice_prompt_items, False

    async def combine_voice_prompts(self, audio_paths, reference_texts):
        return await _combine_voice_prompts(audio_paths, reference_texts)

    async def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: Optional[int] = None,
        instruct: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate audio from text using voice prompt.

        Args:
            text: Text to synthesize
            voice_prompt: Voice prompt dictionary with ref_audio and ref_text
            language: ISO code (en, es, zh, ...); mapped to the Qwen3-TTS
                language name, unsupported codes fall back to "auto"
            seed: Random seed for reproducibility
            instruct: Ignored by the Base checkpoint on MLX (protocol compatibility)

        Returns:
            Tuple of (audio_array, sample_rate)

        Raises:
            FileNotFoundError: the voice prompt points at a missing reference file
            RuntimeError: the model cannot clone, fails while cloning, or yields no audio
        """
        await self.load_model_async(None)

        logger.info("Generating audio for text: %s", text)
        if instruct:
            logger.debug("instruct is ignored by the Qwen3-TTS Base checkpoint on MLX")

        def _generate_sync():
            """Run synchronous generation on the dedicated MLX thread."""
            if seed is not None:
                import mlx.core as mx

                np.random.seed(seed)
                mx.random.seed(seed)

            lang = self._resolve_language(language)
            ref_audio, ref_text = self._reference_from_prompt(voice_prompt)
            results = self._run_model(text, ref_audio, ref_text, lang)
            if not results:
                raise RuntimeError("MLX TTS model produced no audio")

            self._warn_if_token_capped(text, results)
            audio = np.concatenate([np.asarray(np.array(r.audio), dtype=np.float32) for r in results])
            return audio, getattr(results[-1], "sample_rate", 24000)

        # Run blocking inference on the dedicated MLX thread
        return await run_in_mlx_executor(_generate_sync)

    @staticmethod
    def _resolve_language(language: str) -> str:
        """Map an ISO code to the language name Qwen3-TTS expects.

        Unknown codes used to become "auto" silently, which conditions the
        text on no language at all; make the fallback visible.
        """
        lang = LANGUAGE_CODE_TO_NAME.get(language)
        if lang is None:
            logger.warning("Language %r is not supported by Qwen3-TTS; falling back to auto", language)
            return "auto"
        return lang

    @staticmethod
    def _reference_from_prompt(voice_prompt: dict) -> tuple[str | None, str]:
        """Return (ref_audio_path, ref_text) from a voice prompt, or fail loudly.

        Substituting the model's default voice for a missing reference used to
        hide a broken clone behind a warning; the take sounded like a bad
        accent instead of an error.
        """
        ref_audio = voice_prompt.get("ref_audio") or voice_prompt.get("ref_audio_path")
        ref_text = voice_prompt.get("ref_text") or ""
        if ref_audio and not Path(ref_audio).exists():
            raise FileNotFoundError(
                f"Reference audio for this voice profile is missing: {ref_audio}. "
                "Re-add the voice sample or clear the voice prompt cache."
            )
        if not ref_audio:
            logger.warning("No reference audio in voice prompt; generating with the model's default voice")
        elif not ref_text:
            logger.warning("Reference transcript is empty; in-context cloning will run without text")
        return ref_audio, ref_text

    def _run_model(self, text: str, ref_audio: str | None, ref_text: str, lang: str) -> list:
        """Call mlx-audio once and collect its GenerationResult objects."""
        if ref_audio is None:
            return list(self.model.generate(text, lang_code=lang))

        if "ref_audio" not in inspect.signature(self.model.generate).parameters:
            raise RuntimeError("Loaded MLX model does not support voice cloning (no ref_audio parameter)")

        # mlx-audio takes the in-context (ICL) cloning path when it has a
        # reference, a transcript and a speech-tokenizer encoder.
        has_encoder = getattr(getattr(self.model, "speech_tokenizer", None), "has_encoder", None)
        logger.info(
            "MLX TTS clone: lang_code=%s icl=%s ref_text_chars=%d",
            lang,
            has_encoder is not False,
            len(ref_text or ""),
        )
        return list(self.model.generate(text, ref_audio=ref_audio, ref_text=ref_text, lang_code=lang))

    def _warn_if_token_capped(self, text: str, results: list) -> None:
        """Flag chunks that hit mlx-audio's token cap; diagnostics only.

        mlx-audio stops in-context generation at max(75, 6 * text tokens)
        codec tokens and yields the truncated audio without any signal, so a
        slow or pause-heavy delivery is cut mid-sentence. Voicebox cannot
        recover the lost speech, but it can say so. Nothing here may fail a
        take that already produced audio.
        """
        try:
            self._check_token_cap(text, results)
        except Exception as exc:
            logger.debug("Token-cap check skipped: %s", exc)

    def _check_token_cap(self, text: str, results: list) -> None:
        tokenizer = getattr(self.model, "tokenizer", None)
        if tokenizer is None or not hasattr(tokenizer, "encode"):
            return
        counts = [int(getattr(r, "token_count", 0) or 0) for r in results]
        # Without a reference mlx-audio splits on newlines and caps each
        # segment; the clone path yields one result for the whole text.
        segments = [s for s in text.split("\n") if s.strip()]
        if len(segments) != len(counts):
            segments, counts = [text], [sum(counts)]
        for segment, used in zip(segments, counts, strict=True):
            cap = max(75, len(tokenizer.encode(segment)) * 6)
            if used >= cap:
                logger.warning(
                    "MLX TTS chunk hit mlx-audio's token cap (%d codec tokens for %d chars); "
                    "the audio may be truncated. Use shorter chunks for slow or pause-heavy delivery.",
                    used,
                    len(segment),
                )
            else:
                logger.debug("MLX TTS chunk used %d/%d codec tokens", used, cap)


class MLXSTTBackend:
    """MLX-based STT backend using mlx-audio Whisper."""

    def __init__(self, model_size: str = "base"):
        self.model = None
        self.model_size = model_size

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None

    def _is_model_cached(self, model_size: str) -> bool:
        hf_repo = WHISPER_HF_REPOS.get(model_size, f"openai/whisper-{model_size}")
        return is_model_cached(hf_repo, weight_extensions=(".safetensors", ".bin", ".npz"))

    async def load_model_async(self, model_size: Optional[str] = None):
        """
        Lazy load the MLX Whisper model.

        Args:
            model_size: Model size (tiny, base, small, medium, large)
        """
        if model_size is None:
            model_size = self.model_size

        if self.model is not None and self.model_size == model_size:
            return

        # Run blocking load on the dedicated MLX thread
        await run_in_mlx_executor(self._load_model_sync, model_size)

    # Alias for compatibility
    load_model = load_model_async

    def _load_model_sync(self, model_size: str):
        """Synchronous model loading."""
        progress_model_name = f"whisper-{model_size}"
        is_cached = self._is_model_cached(model_size)

        with model_load_progress(progress_model_name, is_cached):
            from mlx_audio.stt import load

            model_name = WHISPER_HF_REPOS.get(model_size, f"openai/whisper-{model_size}")
            logger.info("Loading MLX Whisper model %s...", model_size)

            self.model = load(model_name)

        self.model_size = model_size
        logger.info("MLX Whisper model %s loaded successfully", model_size)

    def unload_model(self):
        """Unload the model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
            logger.info("MLX Whisper model unloaded")

    async def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        model_size: Optional[str] = None,
    ) -> str:
        """
        Transcribe audio to text.

        Args:
            audio_path: Path to audio file
            language: Optional language hint
            model_size: Optional model size override

        Returns:
            Transcribed text
        """
        await self.load_model_async(model_size)

        def _transcribe_sync():
            """Run synchronous transcription in thread pool."""
            # MLX Whisper transcription using generate method
            # The generate method accepts audio path directly
            decode_options = {}
            if language:
                decode_options["language"] = language

            # Inference runs with the process's default HF_HUB_OFFLINE
            # state — see the comment in MLXTTSBackend.generate for the
            # regression this revert fixes (issue #462).
            result = self.model.generate(str(audio_path), **decode_options)

            # Extract text from result
            if isinstance(result, str):
                return result.strip()
            elif isinstance(result, dict):
                return result.get("text", "").strip()
            elif hasattr(result, "text"):
                return result.text.strip()
            else:
                return str(result).strip()

        # Run blocking transcription on the dedicated MLX thread
        return await run_in_mlx_executor(_transcribe_sync)
