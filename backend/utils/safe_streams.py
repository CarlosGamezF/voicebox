"""Console streams that survive a vanished reader.

The desktop app launches the server with stdout/stderr piped into the Tauri
process. With "keep the server running" enabled that process exits while the
server stays alive, and the next app instance adopts the running server. Its
pipes now lead nowhere, so the first ``print()``/tqdm write, typically the
Hugging Face progress bar shown when Whisper loads, raised
``BrokenPipeError`` inside the request that triggered it. These wrappers
swallow such failures and fall back to ``os.devnull`` for the rest of the
process: losing log lines is fine, failing a transcription is not.

The wrappers expose no file descriptor, so ``subprocess(stderr=sys.stderr)``
and ``faulthandler.enable()`` are not supported on them; use
``sys.__stderr__`` for those.
"""

import io
import os
import sys


class PipeSafeStream(io.TextIOBase):
    """Text stream that forwards to *target* until it fails, then discards output."""

    def __init__(self, target: io.TextIOBase) -> None:
        super().__init__()
        self._target = target
        self._fallen_back = False

    def write(self, s: str) -> int:
        try:
            return self._target.write(s)
        except UnicodeEncodeError:
            return self._write_replaced(s)
        except (OSError, ValueError) as e:
            self._handle_failure(e)
            return len(s)

    def flush(self) -> None:
        try:
            self._target.flush()
        except (OSError, ValueError) as e:
            self._handle_failure(e)

    def _write_replaced(self, s: str) -> int:
        """The target's encoding cannot represent *s*: keep the line, escape the characters."""
        encoding = self.encoding
        try:
            self._target.write(s.encode(encoding, "backslashreplace").decode(encoding))
        except (OSError, ValueError) as e:
            self._handle_failure(e)
        return len(s)

    def _handle_failure(self, error: Exception) -> None:
        # A broken pipe or a closed file never recovers; any other ValueError is a caller bug.
        if isinstance(error, ValueError) and not getattr(self._target, "closed", False):
            raise error
        if not self._fallen_back:
            self._fallen_back = True
            self._target = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115 -- lives as long as the process

    @property
    def encoding(self) -> str:
        return getattr(self._target, "encoding", None) or "utf-8"

    @property
    def errors(self) -> str:
        return getattr(self._target, "errors", None) or "strict"

    def writable(self) -> bool:
        return True

    def isatty(self) -> bool:
        if self._fallen_back:
            return False
        return bool(getattr(self._target, "isatty", lambda: False)())

    def fileno(self) -> int:
        # No raw descriptor: a library writing to it directly would hit EPIPE again.
        raise io.UnsupportedOperation("fileno")


def install_pipe_safe_streams() -> None:
    """Wrap ``sys.stdout`` and ``sys.stderr`` once; ``None`` or unusable streams become devnull."""
    if not isinstance(sys.stdout, PipeSafeStream):
        sys.stdout = PipeSafeStream(_usable(sys.stdout))
    if not isinstance(sys.stderr, PipeSafeStream):
        sys.stderr = PipeSafeStream(_usable(sys.stderr))


def _usable(stream: io.TextIOBase | None) -> io.TextIOBase:
    """*stream* if it accepts writes (PyInstaller ``--noconsole`` leaves ``None``), else devnull."""
    if stream is None:
        return open(os.devnull, "w", encoding="utf-8")
    try:
        stream.write("")
    except Exception:
        return open(os.devnull, "w", encoding="utf-8")
    return stream
