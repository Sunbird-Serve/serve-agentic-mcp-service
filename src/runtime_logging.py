import io
import os
import sys
from pathlib import Path
from threading import Lock
from typing import TextIO

from config import settings


_IS_CONFIGURED = False


class RotatingLogFileStream(io.TextIOBase):
    """Simple size-based rotating file stream."""

    def __init__(self, path: Path, max_bytes: int, backup_count: int):
        self._path = path
        self._max_bytes = max(1, int(max_bytes))
        self._backup_count = max(1, int(backup_count))
        self._lock = Lock()
        self._stream = self._open_stream()

    def _open_stream(self) -> TextIO:
        return open(self._path, "a", encoding="utf-8", buffering=1)

    def _rotate_if_needed(self, incoming_bytes: int) -> None:
        try:
            current_size = self._path.stat().st_size
        except FileNotFoundError:
            current_size = 0

        if current_size + incoming_bytes <= self._max_bytes:
            return

        self._stream.close()

        oldest = self._path.with_name(f"{self._path.name}.{self._backup_count}")
        if oldest.exists():
            oldest.unlink()

        for index in range(self._backup_count - 1, 0, -1):
            src = self._path.with_name(f"{self._path.name}.{index}")
            dst = self._path.with_name(f"{self._path.name}.{index + 1}")
            if src.exists():
                src.replace(dst)

        if self._path.exists():
            self._path.replace(self._path.with_name(f"{self._path.name}.1"))

        self._stream = self._open_stream()

    def write(self, data: str) -> int:
        if not data:
            return 0
        encoded_size = len(data.encode("utf-8", errors="replace"))
        with self._lock:
            self._rotate_if_needed(encoded_size)
            self._stream.write(data)
        return len(data)

    def flush(self) -> None:
        with self._lock:
            self._stream.flush()

    def close(self) -> None:
        with self._lock:
            self._stream.close()

    @property
    def encoding(self) -> str:
        return "utf-8"


class TeeStream(io.TextIOBase):
    """Mirror writes to both the original stream and a log file stream."""

    def __init__(self, primary: TextIO, mirror: TextIO):
        self._primary = primary
        self._mirror = mirror

    def write(self, data: str) -> int:
        written = self._primary.write(data)
        self._mirror.write(data)
        return written

    def flush(self) -> None:
        self._primary.flush()
        self._mirror.flush()

    @property
    def encoding(self) -> str:
        return getattr(self._primary, "encoding", "utf-8")

    def fileno(self) -> int:
        return self._primary.fileno()

    def isatty(self) -> bool:
        return self._primary.isatty()


def _resolve_log_path() -> Path:
    env_path = os.getenv("LOG_FILE_PATH")
    configured_path = getattr(settings, "LOG_FILE_PATH", "")
    candidate = env_path or configured_path
    if candidate:
        return Path(candidate).expanduser().resolve()

    repo_root = Path(__file__).resolve().parent.parent
    return repo_root / "logs" / "serve-agentic-mcp-service.log"


def _resolve_log_rotation_settings() -> tuple[int, int]:
    max_bytes = int(os.getenv("LOG_MAX_BYTES", str(settings.LOG_MAX_BYTES)))
    backup_count = int(os.getenv("LOG_BACKUP_COUNT", str(settings.LOG_BACKUP_COUNT)))
    return max_bytes, backup_count


def _configure_utf8_stream(stream_name: str) -> None:
    stream = getattr(sys, stream_name)
    if getattr(stream, "encoding", "").lower() == "utf-8":
        return

    try:
        stream.reconfigure(encoding="utf-8")
        return
    except (AttributeError, ValueError):
        pass

    if hasattr(stream, "buffer"):
        setattr(sys, stream_name, io.TextIOWrapper(stream.buffer, encoding="utf-8"))


def configure_runtime_logging() -> Path:
    """Ensure console logs are UTF-8 and mirrored into a persistent file."""
    global _IS_CONFIGURED
    if _IS_CONFIGURED:
        return _resolve_log_path()

    _configure_utf8_stream("stdout")
    _configure_utf8_stream("stderr")

    log_path = _resolve_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    max_bytes, backup_count = _resolve_log_rotation_settings()
    log_file = RotatingLogFileStream(log_path, max_bytes=max_bytes, backup_count=backup_count)

    if not isinstance(sys.stdout, TeeStream):
        sys.stdout = TeeStream(sys.stdout, log_file)
    if not isinstance(sys.stderr, TeeStream):
        sys.stderr = TeeStream(sys.stderr, log_file)

    _IS_CONFIGURED = True
    return log_path

