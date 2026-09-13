"""Regression tests for the dedicated MLX executor.

MLX (>= 0.31.2) keeps its default GPU stream in thread-local state, so a
model loaded on one ``asyncio.to_thread`` worker and used on another raises
"There is no Stream(gpu, N) in current thread" (issues #675, #699, #606).
Every MLX call must therefore land on one dedicated worker thread.

These tests fake the heavy mlx-audio / mlx-lm calls and only check thread
affinity, so they run anywhere (no MLX, no Apple Silicon required).

Usage:
    python -m pytest backend/tests/test_mlx_executor.py -v
"""

import asyncio
import threading

import pytest

from backend.utils import mlx_executor
from backend.utils.mlx_executor import (
    get_mlx_executor,
    run_in_mlx_executor,
    shutdown_mlx_executor,
)


@pytest.fixture(autouse=True)
def _fresh_executor():
    """Each test starts and ends without a live executor."""
    shutdown_mlx_executor()
    yield
    shutdown_mlx_executor()


def _current_thread_info() -> tuple[int, str]:
    return threading.get_ident(), threading.current_thread().name


async def test_runs_off_the_event_loop_thread():
    ident, _ = await run_in_mlx_executor(_current_thread_info)

    assert ident != threading.get_ident()


async def test_all_calls_share_one_worker_thread():
    results = await asyncio.gather(*(run_in_mlx_executor(_current_thread_info) for _ in range(16)))

    idents = {ident for ident, _ in results}
    names = {name for _, name in results}
    assert len(idents) == 1, "MLX work must stay pinned to one worker thread"
    assert all(name.startswith("mlx") for name in names)


async def test_sequential_calls_reuse_the_same_thread():
    first, _ = await run_in_mlx_executor(_current_thread_info)
    second, _ = await run_in_mlx_executor(_current_thread_info)

    assert first == second


async def test_passes_positional_and_keyword_arguments():
    def add(a, b, *, scale=1):
        return (a + b) * scale

    assert await run_in_mlx_executor(add, 2, 3, scale=10) == 50


async def test_propagates_exceptions_from_the_worker():
    def boom():
        raise RuntimeError("There is no Stream(gpu, 1) in current thread.")

    with pytest.raises(RuntimeError, match="Stream"):
        await run_in_mlx_executor(boom)


def test_get_mlx_executor_is_a_lazy_singleton():
    assert mlx_executor._executor is None

    first = get_mlx_executor()
    second = get_mlx_executor()

    assert first is second
    assert first._max_workers == 1


async def test_shutdown_discards_the_executor_and_a_new_one_is_created():
    before = get_mlx_executor()
    await run_in_mlx_executor(_current_thread_info)

    shutdown_mlx_executor()

    assert mlx_executor._executor is None
    after = get_mlx_executor()
    assert after is not before
    # The replacement executor is usable.
    ident, _ = await run_in_mlx_executor(_current_thread_info)
    assert ident != threading.get_ident()


def test_shutdown_is_idempotent():
    shutdown_mlx_executor()
    shutdown_mlx_executor()

    assert mlx_executor._executor is None


class _ThreadRecorder:
    """Fake model whose calls record the thread they ran on."""

    def __init__(self, seen: set[int]):
        self.seen = seen

    def generate(self, *args, **kwargs):
        self.seen.add(threading.get_ident())
        return "transcribed text"


async def test_mlx_tts_backend_loads_and_generates_on_one_thread():
    from backend.backends.mlx_backend import MLXTTSBackend

    backend = MLXTTSBackend()
    seen: set[int] = set()

    class FakeResult:
        audio = (0.0, 0.1)
        sample_rate = 24000

    class FakeTTSModel:
        def generate(self, text, lang_code="auto"):
            seen.add(threading.get_ident())
            yield FakeResult()

    def fake_load(model_size):
        seen.add(threading.get_ident())
        backend.model = FakeTTSModel()
        backend._current_model_size = model_size

    backend._load_model_sync = fake_load

    audio, sample_rate = await backend.generate("hola", voice_prompt={}, language="es")

    assert sample_rate == 24000
    assert len(audio) == 2
    assert len(seen) == 1, "TTS load and generate must run on the same MLX thread"
    assert threading.get_ident() not in seen


async def test_mlx_stt_backend_loads_and_transcribes_on_one_thread():
    from backend.backends.mlx_backend import MLXSTTBackend

    backend = MLXSTTBackend()
    seen: set[int] = set()

    def fake_load(model_size):
        seen.add(threading.get_ident())
        backend.model = _ThreadRecorder(seen)
        backend.model_size = model_size

    backend._load_model_sync = fake_load

    text = await backend.transcribe("/tmp/does-not-matter.wav")

    assert text == "transcribed text"
    assert len(seen) == 1, "STT load and transcribe must run on the same MLX thread"
    assert threading.get_ident() not in seen


async def test_mlx_llm_backend_loads_and_generates_on_one_thread():
    from backend.backends.qwen_llm_backend import MLXQwenLLMBackend

    backend = MLXQwenLLMBackend()
    seen: set[int] = set()

    def fake_load(model_size):
        seen.add(threading.get_ident())
        backend.model = object()
        backend.tokenizer = object()
        backend._current_model_size = model_size

    def fake_generate(prompt, system, max_tokens, temperature, examples=None):
        seen.add(threading.get_ident())
        return "refined"

    backend._load_model_sync = fake_load
    backend._generate_sync = fake_generate

    assert await backend.generate("raw dictation") == "refined"
    assert len(seen) == 1, "LLM load and generate must run on the same MLX thread"
    assert threading.get_ident() not in seen
