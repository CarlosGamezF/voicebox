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
    transcript_tail_mismatch,
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


def test_trim_window_tolerates_centisecond_rounding_at_the_clip_ends():
    # The UI rounds the end thumb to centiseconds, so a 20.346 s clip arrives as end_s=20.35.
    audio = _speech_like(20.0, lead_s=0.146, tail_s=0.2)

    out = trim_reference_window(audio, SR, start_s=1.0, end_s=20.35)

    assert len(out) == len(audio) - SR
    assert len(trim_reference_window(audio, SR, start_s=-0.01, end_s=10.0)) == 10 * SR
    with pytest.raises(ValueError, match="Invalid reference window"):
        trim_reference_window(audio, SR, start_s=1.0, end_s=21.0)


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


async def test_add_sample_rejects_a_corrupt_upload_as_invalid_audio(db, tmp_path):
    src = tmp_path / "upload.wav"
    src.write_bytes(b"RIFF" + bytes(range(256)) * 8)

    with pytest.raises(ValueError, match="Invalid reference audio"):
        await profiles.add_profile_sample("p1", str(src), "texto", db)


async def test_add_sample_without_window_keeps_the_whole_clip(db, tmp_path):
    src = tmp_path / "upload.wav"
    sf.write(src, _speech_like(12.0, amp=0.3, tail_s=0.5), SR)

    sample = await profiles.add_profile_sample("p1", str(src), "texto", db)

    stored, sr = sf.read(config.resolve_storage_path(sample.audio_path))
    assert 11.9 <= len(stored) / sr <= 12.7  # 12 s of speech plus the 100 ms edge pads
    assert sample.warnings == []


# Transcript that continues past the audio (upstream issue #604's mechanism)


def test_tail_mismatch_detects_words_the_audio_never_says():
    stored = "Espero que esta prueba sea suficiente para capturar cada matiz, tono y pausa con total"
    spoken = "espero que esta prueba sea suficiente para capturar cada matiz, tono..."

    extra = transcript_tail_mismatch(stored, spoken)

    assert extra == "y pausa con total"


def test_tail_mismatch_is_none_when_the_transcript_matches():
    stored = "Espero que esta prueba sea suficiente para capturar cada matiz y tono."
    spoken = "Espero que esta prueba sea suficiente para capturar cada matiz y tono"

    assert transcript_tail_mismatch(stored, spoken) is None


def test_tail_mismatch_tolerates_one_missing_word_and_whisper_variants():
    # A single trailing word can be a Whisper miss rather than a cut recording.
    assert transcript_tail_mismatch("uno dos tres cuatro cinco", "uno dos tres cuatro") is None
    assert transcript_tail_mismatch("La Sra. García llegó tarde", "la señora garcia llego tarde") is None


class _FakeWhisper:
    model_size = "large"

    def __init__(self, downloaded: bool = True):
        self.downloaded = downloaded

    def is_loaded(self):
        return self.downloaded

    def _is_model_cached(self, size):
        return self.downloaded

    async def transcribe(self, path, language, model_size):
        assert self.downloaded, "an undownloaded model must not be loaded by a quality check"
        return "texto de la muestra hasta aquí"


async def test_analysis_with_transcript_check_warns_about_the_extra_words(db, tmp_path, monkeypatch):
    src = tmp_path / "upload.wav"
    sf.write(src, _speech_like(12.0, amp=0.3, tail_s=0.5), SR)
    sample = await profiles.add_profile_sample(
        "p1", str(src), "texto de la muestra hasta aquí y algo que nunca se dijo", db
    )
    monkeypatch.setattr(profiles, "get_stt_backend", lambda: _FakeWhisper())

    analysis = await profiles.analyze_profile_sample(sample.id, db, verify_transcript=True, language="es")

    assert analysis["transcript_extra_words"] == "y algo que nunca se dijo"
    assert any("continues past the audio" in w for w in analysis["warnings"])


async def test_analysis_without_transcript_check_does_not_touch_whisper(db, tmp_path, monkeypatch):
    src = tmp_path / "upload.wav"
    sf.write(src, _speech_like(12.0, amp=0.3, tail_s=0.5), SR)
    sample = await profiles.add_profile_sample("p1", str(src), "texto", db)

    def boom():
        raise AssertionError("Whisper must not be loaded")

    monkeypatch.setattr(profiles, "get_stt_backend", boom)

    analysis = await profiles.analyze_profile_sample(sample.id, db)

    assert analysis["transcript_extra_words"] is None


async def test_transcript_check_is_skipped_when_whisper_is_not_downloaded(db, tmp_path, monkeypatch):
    src = tmp_path / "upload.wav"
    sf.write(src, _speech_like(12.0, amp=0.3, tail_s=0.5), SR)
    sample = await profiles.add_profile_sample("p1", str(src), "texto que sigue", db)
    monkeypatch.setattr(profiles, "get_stt_backend", lambda: _FakeWhisper(downloaded=False))

    analysis = await profiles.analyze_profile_sample(sample.id, db, verify_transcript=True, language="es")

    assert analysis["transcript_extra_words"] is None
    assert any("not downloaded" in w for w in analysis["warnings"])


async def test_analysis_of_a_sample_whose_file_is_gone_raises_file_not_found(db, tmp_path):
    src = tmp_path / "upload.wav"
    sf.write(src, _speech_like(12.0, amp=0.3, tail_s=0.5), SR)
    sample = await profiles.add_profile_sample("p1", str(src), "texto", db)
    config.resolve_storage_path(sample.audio_path).unlink()

    with pytest.raises(FileNotFoundError):
        await profiles.analyze_profile_sample(sample.id, db)
