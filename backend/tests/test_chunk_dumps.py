"""Optional per-chunk WAV dumps for listening tests and seam analysis.

When ``VOICEBOX_DUMP_CHUNKS`` names a directory, ``generate_chunked`` writes
every chunk's audio (before crossfading) plus its text next to it, so seam
pauses, truncation and speed drift can be measured per chunk instead of on
the concatenated file. Unset, nothing is written.
"""

import logging

import numpy as np
import pytest

from backend.utils import chunked_tts
from backend.utils.chunked_tts import _chunk_dump_dir, generate_chunked

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
async def test_nothing_is_written_without_the_env_var(monkeypatch):
    monkeypatch.delenv("VOICEBOX_DUMP_CHUNKS", raising=False)
    writes = []
    monkeypatch.setattr(chunked_tts, "save_audio", lambda *a, **k: writes.append(a))

    assert _chunk_dump_dir(TWO_SENTENCES, 7) is None
    await generate_chunked(FakeBackend(), TWO_SENTENCES, {}, max_chunk_chars=150)

    assert writes == []


def test_seeded_runs_share_a_directory_and_unseeded_runs_do_not(tmp_path, monkeypatch):
    monkeypatch.setenv("VOICEBOX_DUMP_CHUNKS", str(tmp_path))

    assert _chunk_dump_dir("hola", 3) == _chunk_dump_dir("hola", 3)
    assert _chunk_dump_dir("hola", 3) != _chunk_dump_dir("hola", 4)
    assert _chunk_dump_dir("hola", None) != _chunk_dump_dir("hola", None)


@pytest.mark.asyncio
async def test_dump_failure_does_not_break_generation(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("VOICEBOX_DUMP_CHUNKS", str(tmp_path))

    def boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(chunked_tts, "save_audio", boom)

    with caplog.at_level(logging.WARNING):
        audio, sample_rate = await generate_chunked(FakeBackend(), "Hola.", {}, max_chunk_chars=800)

    assert sample_rate == SAMPLE_RATE
    assert len(audio) == SAMPLE_RATE
    assert any("Could not dump chunk" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_unwritable_root_does_not_break_generation(tmp_path, monkeypatch):
    # A file where a directory is expected: mkdir fails, generation goes on.
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x")
    monkeypatch.setenv("VOICEBOX_DUMP_CHUNKS", str(blocker))

    audio, sample_rate = await generate_chunked(FakeBackend(), "Hola.", {}, max_chunk_chars=800)

    assert sample_rate == SAMPLE_RATE
    assert len(audio) == SAMPLE_RATE
