"""
Single-thread executor for MLX inference.

MLX (>= 0.31.2) keeps the default GPU stream in *thread-local* state. When
blocking inference is dispatched via ``asyncio.to_thread`` -- which uses the
default ``ThreadPoolExecutor`` and picks an arbitrary worker each call -- the
model may be loaded on one worker thread and later generate/transcribe on a
different one. The second thread has no default GPU stream, which surfaces as:

    RuntimeError: There is no Stream(gpu, N) in current thread.

(reported for TTS generation, Whisper transcription and LLM refinement over
REST, MCP and the desktop app: voicebox issues #675 / #699 / #606, and
upstream mlx-lm #1181 / #1256).

Routing every MLX call through a single dedicated thread whose default device
is initialized once guarantees that model load and all subsequent inference
run on the same thread, so the stream always exists. Generation is already
serialized elsewhere, so a single worker adds no throughput loss and also
removes GPU contention.
"""

import asyncio
import concurrent.futures
import functools
import logging

logger = logging.getLogger(__name__)

_executor: concurrent.futures.ThreadPoolExecutor | None = None


def _init_mlx_thread() -> None:
    """Pin the executor's thread to the GPU device so its default stream exists.

    Best-effort: on machines without MLX (non-Apple-Silicon builds) or without
    a GPU, this is a no-op and inference falls back to whatever device MLX picks.
    """
    try:
        import mlx.core as mx

        mx.set_default_device(mx.gpu)
        logger.debug("MLX executor thread initialized with default device: gpu")
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.debug("MLX executor thread GPU init skipped: %s", exc)


def get_mlx_executor() -> concurrent.futures.ThreadPoolExecutor:
    """Return the shared single-thread executor for MLX work, creating it lazily."""
    global _executor
    if _executor is None:
        _executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="mlx",
            initializer=_init_mlx_thread,
        )
    return _executor


async def run_in_mlx_executor(func, /, *args, **kwargs):
    """Run a blocking MLX callable on the dedicated MLX thread.

    Drop-in replacement for ``asyncio.to_thread`` that guarantees all MLX
    inference shares one thread (and therefore one default GPU stream).
    """
    loop = asyncio.get_running_loop()
    call = functools.partial(func, *args, **kwargs)
    return await loop.run_in_executor(get_mlx_executor(), call)


def shutdown_mlx_executor() -> None:
    """Shut down the MLX executor (used on app shutdown / test teardown)."""
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=False)
        _executor = None
