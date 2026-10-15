"""Session directory layout.

A *session* is the unit of work shared by the SDK and the CLI::

    <session>/
      raw.mp4           high-quality intermediate capture
      telemetry.jsonl   one JSON object per logged agent action
      meta.json         how the session was recorded (display, fps, geometry...)
      display.log       Xvfb stderr, when the display was managed by Dynamo

Everything downstream — :class:`~agent_visual_harness.processor.Processor` and
:class:`~agent_visual_harness.token_reducer.TokenReducer` — reads a session, so
``dynamo record`` on one machine and ``dynamo export`` on another produce the same
artifacts.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from collections.abc import Iterator

from .errors import SessionError
from .recorder import read_actions
from .types import Action, MediaInfo, parse_resolution

__all__ = ["Session", "RAW_FILENAME", "TELEMETRY_FILENAME", "META_FILENAME"]

RAW_FILENAME = "raw.mp4"
TELEMETRY_FILENAME = "telemetry.jsonl"
META_FILENAME = "meta.json"
DISPLAY_LOG_FILENAME = "display.log"


@dataclass(slots=True)
class Session:
    """A directory holding one capture plus its telemetry."""

    root: Path
    meta: dict[str, Any] = field(default_factory=dict)

    # ----------------------------------------------------------- construction

    @classmethod
    def create(
        cls,
        root: Path | str,
        *,
        display: str = ":99",
        resolution: tuple[int, int] = (1920, 1080),
        depth: int = 24,
        fps: int = 24,
        crf: int = 16,
        preset: str = "veryfast",
        command: list[str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Session:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        session = cls(root=root)
        session.meta = {
            "dynamo_version": _version(),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "display": str(display),
            "resolution": list(resolution),
            "resolution_x": f"{resolution[0]}x{resolution[1]}",
            "depth": depth,
            "fps": fps,
            "capture_crf": crf,
            "capture_preset": preset,
            "command": list(command or []),
            "python": _python_version(),
            "platform": _platform(),
            **(extra or {}),
        }
        session.write_meta()
        (root / DISPLAY_LOG_FILENAME).touch()
        return session

    @classmethod
    def load(cls, root: Path | str) -> Session:
        """Load an existing session directory, validating that it is usable."""
        root = Path(root).expanduser()
        if not root.exists():
            raise SessionError(f"session directory does not exist: {root}")
        if not root.is_dir():
            raise SessionError(f"session path is not a directory: {root}")
        meta_path = root / META_FILENAME
        meta: dict[str, Any] = {}
        if meta_path.exists():
            try:
                loaded = json.loads(meta_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    meta = loaded
            except json.JSONDecodeError as exc:
                raise SessionError(f"{meta_path} is not valid JSON: {exc}") from exc
        return cls(root=root, meta=meta)

    # ------------------------------------------------------------------ paths

    @property
    def raw_path(self) -> Path:
        return self.root / RAW_FILENAME

    @property
    def telemetry_path(self) -> Path:
        return self.root / TELEMETRY_FILENAME

    @property
    def meta_path(self) -> Path:
        return self.root / META_FILENAME

    @property
    def display_log(self) -> Path:
        return self.root / DISPLAY_LOG_FILENAME

    def has_recording(self) -> bool:
        return self.raw_path.exists() and self.raw_path.stat().st_size > 0

    def require_recording(self) -> Path:
        if not self.has_recording():
            raise SessionError(
                f"no recording found in {self.root} (expected {self.raw_path.name}). "
                f"Run `dynamo record` first."
            )
        return self.raw_path

    # ------------------------------------------------------------------ meta

    def write_meta(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.meta_path.write_text(json.dumps(self.meta, indent=2) + "\n", encoding="utf-8")

    def update_meta(self, **values: Any) -> None:
        self.meta.update(values)
        self.write_meta()

    def mark_finished(self, duration: float | None = None, **extra: Any) -> None:
        payload: dict[str, Any] = {
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "wall_seconds": round(duration if duration is not None else 0.0, 3),
        }
        payload.update(extra)
        self.update_meta(**payload)

    # -------------------------------------------------------------- contents

    @property
    def display(self) -> str:
        return str(self.meta.get("display", ":99"))

    @property
    def fps(self) -> int:
        return int(self.meta.get("fps", 24) or 24)

    @property
    def resolution(self) -> tuple[int, int]:
        raw = self.meta.get("resolution")
        if isinstance(raw, (list, tuple)) and len(raw) >= 2:
            return (int(raw[0]), int(raw[1]))
        text = self.meta.get("resolution_x") or "1920x1080"
        return parse_resolution(text)

    @property
    def actions(self) -> list[Action]:
        """Every logged action, sorted by timestamp."""
        return read_actions(self.telemetry_path)

    def media_info(self) -> MediaInfo:
        from ._ffmpeg import probe  # local import: keeps `Session` cheap to import

        return probe(self.require_recording())

    def iter_files(self) -> Iterator[Path]:
        yield from sorted(p for p in self.root.iterdir() if p.is_file())

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "meta": self.meta,
            "has_recording": self.has_recording(),
            "actions": len(self.actions),
        }

    def __fspath__(self) -> str:  # allows os.fspath(session)
        return str(self.root)

    def __str__(self) -> str:  # pragma: no cover - display only
        return str(self.root)


def _version() -> str:
    from . import __version__

    return __version__


def _python_version() -> str:
    import platform

    return platform.python_version()


def _platform() -> str:
    import platform
    import sys

    return f"{platform.system()} {platform.release()} ({sys.platform})"


def now() -> float:  # pragma: no cover - trivial seam for tests
    return time.monotonic()
