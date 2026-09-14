"""Console streams that survive a vanished reader (backend/utils/safe_streams.py).

With "keep the server running" enabled the Tauri app exits while the sidecar
stays alive, and the next app instance adopts it. The sidecar's stdout and
stderr still point at the dead parent's pipes, so the first print()/tqdm
write raised BrokenPipeError inside whatever request triggered it: every
transcription failed with "[Errno 32] Broken pipe" until the server was
restarted. These tests pin the wrappers that swallow that.
"""

import io
import logging
import sys

import pytest
from tqdm import tqdm

from backend.utils.safe_streams import PipeSafeStream, install_pipe_safe_streams


class _DeadPipe(io.StringIO):
    """A text stream whose reader has gone away."""

    def write(self, s):
        raise BrokenPipeError(32, "Broken pipe")

    def flush(self):
        raise BrokenPipeError(32, "Broken pipe")


def test_writes_to_a_dead_pipe_are_swallowed():
    stream = PipeSafeStream(_DeadPipe())

    print("hola", file=stream)
    stream.flush()

    assert stream.write("x") == 1


def test_a_healthy_target_receives_everything():
    target = io.StringIO()
    stream = PipeSafeStream(target)

    print("hola", file=stream)
    stream.flush()

    assert target.getvalue() == "hola\n"


def test_logging_and_tqdm_keep_working_on_a_dead_pipe(monkeypatch):
    stream = PipeSafeStream(_DeadPipe())
    logger = logging.getLogger("test_safe_streams")
    handler = logging.StreamHandler(stream)
    # logging would swallow the error itself; the wrapper must stop it from ever reaching handleError.
    monkeypatch.setattr(handler, "handleError", lambda record: pytest.fail("the write reached the dead pipe"))
    logger.addHandler(handler)
    try:
        logger.warning("still alive")
        assert list(tqdm(range(3), file=stream)) == [0, 1, 2]
    finally:
        logger.removeHandler(handler)


def test_a_pipe_that_dies_on_flush_is_handled_too():
    # Real pipes buffer write() and only raise EPIPE on flush().
    class _DiesOnFlush(io.StringIO):
        def flush(self):
            raise BrokenPipeError(32, "Broken pipe")

    target = _DiesOnFlush()
    stream = PipeSafeStream(target)
    stream.write("buffered")

    stream.flush()
    stream.write("after")

    assert target.getvalue() == "buffered"  # later writes go to devnull, not the dead target


def test_an_unencodable_character_does_not_kill_a_healthy_stream(tmp_path):
    with open(tmp_path / "cp1252.txt", "w", encoding="cp1252", errors="strict") as raw:
        stream = PipeSafeStream(raw)
        print("uno → dos", file=stream)
        print("tres", file=stream)

    assert (tmp_path / "cp1252.txt").read_text(encoding="cp1252") == "uno \\u2192 dos\ntres\n"


def test_isatty_follows_the_target_until_it_dies():
    class _Tty(io.StringIO):
        def isatty(self):
            return True

    assert PipeSafeStream(_Tty()).isatty() is True
    dead = PipeSafeStream(_DeadPipe())
    dead.write("x")
    assert dead.isatty() is False


def test_other_os_errors_on_write_are_swallowed_too():
    class _Full(io.StringIO):
        def write(self, s):
            raise OSError(28, "No space left on device")

    stream = PipeSafeStream(_Full())

    print("hola", file=stream)


def test_no_raw_file_descriptor_is_exposed():
    # Libraries that get a fileno write to the fd directly and would hit EPIPE again.
    with pytest.raises(io.UnsupportedOperation):
        PipeSafeStream(io.StringIO()).fileno()
    assert PipeSafeStream(io.StringIO()).isatty() is False


def test_install_wraps_the_console_streams_and_tolerates_none(monkeypatch):
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", _DeadPipe())

    install_pipe_safe_streams()

    assert isinstance(sys.stdout, PipeSafeStream)
    assert isinstance(sys.stderr, PipeSafeStream)
    print("to nobody")
    print("to a dead pipe", file=sys.stderr)


def test_install_is_idempotent(monkeypatch):
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    monkeypatch.setattr(sys, "stderr", io.StringIO())

    install_pipe_safe_streams()
    first = sys.stdout
    install_pipe_safe_streams()

    assert sys.stdout is first
    assert not isinstance(sys.stdout._target, PipeSafeStream)
