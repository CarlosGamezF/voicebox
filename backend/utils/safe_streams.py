"""Console streams that survive a vanished reader.

The desktop app launches the server with stdout/stderr piped into the Tauri
process. With "keep the server running" enabled that process exits while the
server stays alive, and the next app instance adopts the running server. Its
pipes now lead nowhere, so the first ``print()``/tqdm write, typically the
Hugging Face progress bar shown when Whisper loads, raised
``BrokenPipeError`` inside the request that triggered it. These wrappers
swallow such failures and fall back to ``os.devnull`` for the rest of the
process: losing log lines is fine, failing a transcription is not.
"""

import io
import os
import sys


class PipeSafeStream(io.TextIOBase):
    """Text stream that forwards to *target* until it fails, then discards output."""

    def __init__(self, target):
        super().__init__()
        self._target = target
        self._fallen_back = False

    def write(self, s: str) -> int:
        try:
            return self._target.write(s)
        except (OSError, ValueError):  # ValueError: write on a closed file
            self._fall_back()
            return len(s)

    def flush(self) -> None:
        try:
            self._target.flush()
        except (OSError, ValueError):
            self._fall_back()

    def _fall_back(self) -> None:
        if not self._fallen_back:
            self._fallen_back = True
            self._target = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115 - lives as long as the process

    @property
    def encoding(self) -> str:
        return getattr(self._target, "encoding", None) or "utf-8"

    @property
    def errors(self) -> str:
        return getattr(self._target, "errors", None) or "strict"

    def writable(self) -> bool:
        return True

    def isatty(self) -> bool:
        return False

    def fileno(self) -> int:
        # No raw descriptor: a library writing to it directly would hit EPIPE again.
        raise io.UnsupportedOperation("fileno")


def install_pipe_safe_streams() -> None:
    """Wrap ``sys.stdout`` and ``sys.stderr``; ``None`` or unusable streams become devnull."""
    sys.stdout = PipeSafeStream(_usable(sys.stdout))
    sys.stderr = PipeSafeStream(_usable(sys.stderr))


def _usable(stream):
    if stream is None:
        return open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115
    try:
        stream.write("")
    except Exception:
        return open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115
    return stream
