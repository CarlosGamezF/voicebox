"""Reference-sample quality analysis, warnings and the optional trim window.

Voice-clone quality follows the reference clip: Qwen3-TTS plateaus around
10-15 s, a quiet or clipped recording degrades the codec conditioning and a
clip cut mid-word leaks into every generated take. The app only validated
duration and a silence floor, so a 29 s clip ending at 0.00 s of silence was
accepted without a word. These tests pin the analysis, the warnings and the
server-side window trim.
"""

import numpy as np
import pytest
import soundfile as sf
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend import config
from backend.database import Base, VoiceProfile
from backend.services import profiles
from backend.utils.audio import (
    analyze_reference_audio,
    reference_audio_warnings,
    trim_reference_window,
)

SR = 24000


def _speech_like(duration_s: float, amp: float = 0.3, lead_s: float = 0.1, tail_s: float = 0.5) -> np.ndarray:
    """Tone burst with silence around it, standing in for a spoken clip."""
    n = int(duration_s * SR)
    t = np.arange(n, dtype=np.float32) / SR
    tone = (amp * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    return np.concatenate([np.zeros(int(lead_s * SR), np.float32), tone, np.zeros(int(tail_s * SR), np.float32)])


def test_analysis_reports_duration_level_and_edge_silence():
    audio = _speech_like(10.0, amp=0.3, lead_s=0.2, tail_s=0.6)

    a = analyze_reference_audio(audio, SR)

    assert a["duration_s"] == pytest.approx(10.8, abs=0.05)
    assert a["leading_silence_s"] == pytest.approx(0.2, abs=0.05)
    assert a["trailing_silence_s"] == pytest.approx(0.6, abs=0.05)
    assert -16 < a["rms_dbfs"] < -12  # 0.3 amplitude sine ~ -13.5 dBFS over the whole clip
    assert a["clipping_ratio"] == 0.0


def test_no_warnings_for_a_good_clip():
    audio = _speech_like(12.0, amp=0.3, tail_s=0.5)

    assert reference_audio_warnings(analyze_reference_audio(audio, SR)) == []


def test_long_quiet_abrupt_clip_gets_the_three_warnings():
    audio = _speech_like(27.0, amp=0.02, tail_s=0.0)

    warnings = reference_audio_warnings(analyze_reference_audio(audio, SR))

    joined = " ".join(warnings)
    assert "10-15" in joined  # too long
    assert "quiet" in joined.lower()
    assert "abrupt" in joined.lower() or "silence after" in joined.lower()


def test_clipping_is_reported():
    audio = _speech_like(10.0, amp=0.3)
    audio[SR : SR + 2400] = 1.0  # 100 ms slammed against full scale

    warnings = reference_audio_warnings(analyze_reference_audio(audio, SR))

    assert any("clipp" in w.lower() for w in warnings)


def test_short_clip_is_flagged():
    warnings = reference_audio_warnings(analyze_reference_audio(_speech_like(3.0), SR))

    assert any("short" in w.lower() for w in warnings)


def test_trim_window_cuts_and_keeps_bounds_sane():
    audio = _speech_like(20.0)

    out = trim_reference_window(audio, SR, start_s=2.0, end_s=14.0)

    assert len(out) == 12 * SR
    with pytest.raises(ValueError, match="Invalid reference window"):
        trim_reference_window(audio, SR, start_s=14.0, end_s=2.0)
    with pytest.raises(ValueError, match="Invalid reference window"):
        trim_reference_window(audio, SR, start_s=-1.0, end_s=5.0)


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "_data_dir", tmp_path)
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(autocommit=False, autoflush=False, bind=engine)()
    session.add(VoiceProfile(id="p1", name="Carlos", language="es"))
    session.commit()
    yield session
    session.close()


async def test_add_sample_returns_warnings_and_applies_the_window(db, tmp_path):
    src = tmp_path / "upload.wav"
    sf.write(src, _speech_like(26.0, amp=0.02, tail_s=0.0), SR)

    sample = await profiles.add_profile_sample("p1", str(src), "texto", db, start_s=1.0, end_s=13.0)

    stored, sr = sf.read(config.resolve_storage_path(sample.audio_path))
    assert 11.5 <= len(stored) / sr <= 12.3  # 12 s window, edges trimmed by preprocessing
    joined = " ".join(sample.warnings).lower()
    assert "quiet" in joined
    assert "10-15" not in joined  # the window fixed the length


async def test_add_sample_without_window_keeps_the_whole_clip(db, tmp_path):
    src = tmp_path / "upload.wav"
    sf.write(src, _speech_like(12.0, amp=0.3, tail_s=0.5), SR)

    sample = await profiles.add_profile_sample("p1", str(src), "texto", db)

    stored, sr = sf.read(config.resolve_storage_path(sample.audio_path))
    assert 11.9 <= len(stored) / sr <= 12.7  # 12 s of speech plus the 100 ms edge pads
    assert sample.warnings == []
