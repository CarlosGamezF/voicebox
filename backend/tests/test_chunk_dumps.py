"""Optional per-chunk WAV dumps for listening tests and seam analysis.

When ``VOICEBOX_DUMP_CHUNKS`` names a directory, ``generate_chunked`` writes
every chunk's audio (before crossfading) plus its text next to it, so seam
pauses, truncation and speed drift can be measured per chunk instead of on
the concatenated file. Unset, nothing is written.
"""

import numpy as np
import pytest

from backend.utils.chunked_tts import generate_chunked

SAMPLE_RATE = 1000


class FakeBackend:
    async def generate(self, text, *_args):
        return np.full(SAMPLE_RATE, 0.2, dtype=np.float32), SAMPLE_RATE


TWO_SENTENCES = f"{'A' * 119}. {'B' * 119}."


@pytest.mark.asyncio
async def test_dumps_one_wav_and_txt_per_chunk(tmp_path, monkeypatch):
    monkeypatch.setenv("VOICEBOX_DUMP_CHUNKS", str(tmp_path))

    await generate_chunked(FakeBackend(), TWO_SENTENCES, {}, seed=7, max_chunk_chars=150, crossfade_ms=50)

    wavs = sorted(tmp_path.rglob("chunk_*.wav"))
    txts = sorted(tmp_path.rglob("chunk_*.txt"))
    assert len(wavs) == 2
    assert [t.read_text() for t in txts] == [f"{'A' * 119}.", f"{'B' * 119}."]


@pytest.mark.asyncio
async def test_single_shot_text_is_dumped_too(tmp_path, monkeypatch):
    monkeypatch.setenv("VOICEBOX_DUMP_CHUNKS", str(tmp_path))

    await generate_chunked(FakeBackend(), "Hola.", {}, max_chunk_chars=800)

    assert len(list(tmp_path.rglob("chunk_*.wav"))) == 1


@pytest.mark.asyncio
async def test_nothing_is_written_without_the_env_var(tmp_path, monkeypatch):
    monkeypatch.delenv("VOICEBOX_DUMP_CHUNKS", raising=False)

    await generate_chunked(FakeBackend(), TWO_SENTENCES, {}, max_chunk_chars=150)

    assert list(tmp_path.rglob("*")) == []


@pytest.mark.asyncio
async def test_dump_failure_does_not_break_generation(tmp_path, monkeypatch):
    # A file where a directory is expected: dumping must log and carry on.
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x")
    monkeypatch.setenv("VOICEBOX_DUMP_CHUNKS", str(blocker))

    audio, sample_rate = await generate_chunked(FakeBackend(), "Hola.", {}, max_chunk_chars=800)

    assert sample_rate == SAMPLE_RATE
    assert len(audio) == SAMPLE_RATE
