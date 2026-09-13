"""Retry and regenerate reuse the chunking and normalisation of the original take.

The generation row stored text, language, seed, engine and instruct, but not
max_chunk_chars / crossfade_ms / normalize. Retrying a failed take or
regenerating a completed one therefore ran with the 800-char / 50 ms schema
defaults and always normalised, so a "regenerate to compare seeds" produced
different chunking than the original. The three values are now stored on
the row; rows written before the columns existed fall back to the persisted
generation settings.
"""

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

import backend.routes.generations as generations
from backend.database import Base, Generation, VoiceProfile
from backend.database.migrations import _migrate_generations
from backend.services import history, settings as settings_service


def test_migration_adds_the_replay_columns(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'old.db'}")
    with engine.connect() as conn:
        conn.execute(
            text(
                "CREATE TABLE generations (id VARCHAR PRIMARY KEY, profile_id VARCHAR NOT NULL, "
                "text TEXT NOT NULL, language VARCHAR, audio_path VARCHAR, duration FLOAT, seed INTEGER, "
                "instruct TEXT, engine VARCHAR, model_size VARCHAR, status VARCHAR, error TEXT, "
                "is_favorited BOOLEAN, source VARCHAR NOT NULL DEFAULT 'manual', created_at DATETIME)"
            )
        )
        conn.commit()

    _migrate_generations(engine, inspect(engine), {"generations"})

    columns = {c["name"] for c in inspect(engine).get_columns("generations")}
    assert {"max_chunk_chars", "crossfade_ms", "normalize"} <= columns


@pytest.fixture
def db(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(autocommit=False, autoflush=False, bind=engine)()
    session.add(VoiceProfile(id="p1", name="Carlos", language="es"))
    session.commit()
    yield session
    session.close()


async def test_create_generation_stores_the_settings(db):
    created = await history.create_generation(
        profile_id="p1",
        text="hola",
        language="es",
        audio_path="",
        duration=0.0,
        seed=3,
        db=db,
        status="generating",
        max_chunk_chars=550,
        crossfade_ms=80,
        normalize=False,
    )

    row = db.query(Generation).filter_by(id=created.id).one()
    assert (row.max_chunk_chars, row.crossfade_ms, row.normalize) == (550, 80, False)


@pytest.fixture
def captured_run(monkeypatch):
    captured = {}

    async def _noop():
        return None

    def fake_run_generation(**kwargs):
        # run_generation is called (not awaited) by the route; capture the
        # kwargs eagerly and hand back a coroutine for the queue to close.
        captured.update(kwargs)
        return _noop()

    def fake_enqueue(generation_id, coro):
        coro.close()

    class _TaskManager:
        def start_generation(self, **kwargs):
            pass

    monkeypatch.setattr(generations, "run_generation", fake_run_generation)
    monkeypatch.setattr(generations, "enqueue_generation", fake_enqueue)
    monkeypatch.setattr(generations, "get_task_manager", lambda: _TaskManager())
    return captured


def _row(db, status, **settings):
    gen = Generation(id="g1", profile_id="p1", text="hola", language="es", status=status, seed=7, **settings)
    db.add(gen)
    db.commit()
    return gen


async def test_retry_replays_the_stored_settings(db, captured_run):
    _row(db, "failed", max_chunk_chars=300, crossfade_ms=120, normalize=False)

    await generations.retry_generation("g1", db)

    assert captured_run["mode"] == "retry"
    assert captured_run["max_chunk_chars"] == 300
    assert captured_run["crossfade_ms"] == 120
    assert captured_run["normalize"] is False


async def test_regenerate_replays_the_stored_settings(db, captured_run):
    _row(db, "completed", max_chunk_chars=300, crossfade_ms=120, normalize=True)

    await generations.regenerate_generation("g1", db)

    assert captured_run["mode"] == "regenerate"
    assert (captured_run["max_chunk_chars"], captured_run["crossfade_ms"], captured_run["normalize"]) == (300, 120, True)


async def test_legacy_rows_fall_back_to_the_persisted_settings(db, captured_run):
    # A row written before the columns existed: every value is NULL. The
    # persisted settings differ from the schema defaults so the test can tell
    # the two apart.
    settings_service.update_generation_settings(db, {"max_chunk_chars": 550, "crossfade_ms": 80, "normalize_audio": False})
    _row(db, "failed")

    await generations.retry_generation("g1", db)

    assert captured_run["max_chunk_chars"] == 550
    assert captured_run["crossfade_ms"] == 80
    assert captured_run["normalize"] is False


async def test_partially_legacy_row_fills_only_the_missing_value(db, captured_run):
    settings_service.update_generation_settings(db, {"normalize_audio": False})
    _row(db, "failed", max_chunk_chars=300, crossfade_ms=120)

    await generations.retry_generation("g1", db)

    assert (captured_run["max_chunk_chars"], captured_run["crossfade_ms"]) == (300, 120)
    assert captured_run["normalize"] is False


async def test_regenerate_of_a_legacy_row_keeps_normalising(db, captured_run):
    # Before the columns existed regenerate always normalised; a NULL flag on
    # regenerate keeps that behaviour instead of following the current setting.
    settings_service.update_generation_settings(db, {"normalize_audio": False})
    _row(db, "completed")

    await generations.regenerate_generation("g1", db)

    assert captured_run["normalize"] is True
