"""``POST /transcribe`` accepts the same start_s/end_s window as the sample upload.

The sample "Transcribe" button used to send the whole clip even when the user
had narrowed the reference window, so the saved transcript covered audio the
backend then discarded: exactly the transcript/audio mismatch the analysis
endpoint warns about.
"""

import numpy as np
import pytest
import soundfile as sf
from fastapi import FastAPI
from starlette.testclient import TestClient

from backend.database import get_db
from backend.routes import transcription
from backend.services import transcribe

SR = 16000


class _FakeWhisper:
    model_size = "base"

    def __init__(self):
        self.paths = []

    def is_loaded(self):
        return True

    def _is_model_cached(self, size):
        return True

    async def transcribe(self, path, language, model_size):
        self.paths.append(path)
        return "texto"


@pytest.fixture
def client(monkeypatch):
    whisper = _FakeWhisper()
    monkeypatch.setattr(transcribe, "get_whisper_model", lambda: whisper)
    monkeypatch.setattr(transcribe, "resolve_transcription_model", lambda requested, db, whisper=None: "base")
    app = FastAPI()
    app.include_router(transcription.router)
    app.dependency_overrides[get_db] = lambda: iter([object()])
    return TestClient(app), whisper


def _clip(tmp_path, seconds=20.0):
    path = tmp_path / "clip.wav"
    sf.write(path, np.zeros(int(seconds * SR), dtype=np.float32), SR)
    return path


def test_window_is_applied_before_transcribing(client, tmp_path):
    api, whisper = client
    heard = {}

    async def transcribe_and_measure(path, language, model_size):
        audio, sr = sf.read(path)
        heard["seconds"] = len(audio) / sr
        return "texto"

    whisper.transcribe = transcribe_and_measure

    with _clip(tmp_path).open("rb") as fh:
        resp = api.post(
            "/transcribe", files={"file": ("clip.wav", fh, "audio/wav")}, data={"start_s": "2", "end_s": "14"}
        )

    assert resp.status_code == 200, resp.text
    assert resp.json()["duration"] == pytest.approx(12.0)
    assert heard["seconds"] == pytest.approx(12.0)


def test_without_a_window_the_whole_clip_is_transcribed(client, tmp_path):
    api, whisper = client

    with _clip(tmp_path).open("rb") as fh:
        resp = api.post("/transcribe", files={"file": ("clip.wav", fh, "audio/wav")})

    assert resp.status_code == 200, resp.text
    assert resp.json()["duration"] == pytest.approx(20.0)
    assert len(whisper.paths) == 1


def test_invalid_window_is_a_bad_request(client, tmp_path):
    api, _ = client

    with _clip(tmp_path).open("rb") as fh:
        resp = api.post(
            "/transcribe", files={"file": ("clip.wav", fh, "audio/wav")}, data={"start_s": "14", "end_s": "2"}
        )

    assert resp.status_code == 400
    assert "Invalid reference window" in resp.json()["detail"]
