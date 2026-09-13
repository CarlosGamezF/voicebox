"""Two first requests on a fresh database must not 500 on the settings row.

The settings getters run on hot read paths (/speak, /transcribe, MCP). When
the singleton row does not exist yet, both requests try to insert id=1; the
loser used to surface an IntegrityError. It now rolls back and reads the
row the winner created.
"""

from sqlalchemy.exc import IntegrityError

from backend.services import settings


class _Query:
    def __init__(self, rows):
        self._rows = rows

    def filter(self, *_a, **_k):
        return self

    def first(self):
        return self._rows.pop(0) if self._rows else None


class _RacySession:
    """First query finds nothing, commit collides, second query finds the winner's row."""

    def __init__(self, winner):
        self._rows = [None, winner]
        self.rolled_back = False
        self.refreshed = []

    def query(self, *_a, **_k):
        return _Query(self._rows)

    def add(self, row):
        pass

    def commit(self):
        raise IntegrityError("INSERT", {}, Exception("UNIQUE constraint failed"))

    def rollback(self):
        self.rolled_back = True

    def refresh(self, row):
        self.refreshed.append(row)


def test_generation_settings_survive_the_insert_race():
    winner = object()
    db = _RacySession(winner)

    row = settings.get_generation_settings(db)

    assert row is winner
    assert db.rolled_back is True
    assert db.refreshed == []


def test_capture_settings_survive_the_insert_race():
    winner = object()
    db = _RacySession(winner)

    assert settings.get_capture_settings(db) is winner
    assert db.rolled_back is True
