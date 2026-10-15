"""Typed value objects shared by every module.

These are deliberately plain dataclasses with ``to_dict``/``from_dict`` helpers so
that telemetry written by the recorder can be read back by the reducer, the
processor and the CLI without any serialisation dependency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from collections.abc import Iterable, Iterator, Mapping, Sequence

__all__ = [
    "Action",
    "Keyframe",
    "Artifacts",
    "MediaInfo",
    "Resolution",
    "parse_resolution",
    "read_jsonl",
]

Resolution = tuple[int, int]


def parse_resolution(value: str | Resolution, depth: int = 24) -> Resolution:
    """Parse ``"1920x1080"`` / ``"1920x1080x24"`` / ``(1920, 1080)`` into a tuple.

    The third component of the X-style geometry string is the colour depth; it is
    validated here and then discarded, because the harness only ever needs the
    pixel dimensions.
    """
    if isinstance(value, (tuple, list)):
        if len(value) not in (2, 3):
            raise ValueError(f"resolution tuple must have 2 or 3 items, got {len(value)}")
        width, height = int(value[0]), int(value[1])
        bits = int(value[2]) if len(value) == 3 else depth
    else:
        parts = str(value).lower().replace("*", "x").split("x")
        if len(parts) not in (2, 3):
            raise ValueError(f"cannot parse resolution {value!r}; expected e.g. 1920x1080x24")
        width, height = int(parts[0]), int(parts[1])
        bits = int(parts[2]) if len(parts) == 3 else depth

    if width <= 0 or height <= 0:
        raise ValueError(f"resolution must be positive, got {value!r}")
    if bits not in (0, 8, 15, 16, 24, 32):
        raise ValueError(f"unsupported colour depth {bits} in {value!r}")
    return width, height


@dataclass(slots=True)
class Action:
    """A single agent action logged against the display.

    The JSON field name is ``timestamp`` (seconds since recording start) to match
    the ``telemetry.jsonl`` format documented on the project site.
    """

    action: str
    timestamp: float = 0.0
    x: int | None = None
    y: int | None = None
    target: str | None = None
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def has_position(self) -> bool:
        return self.x is not None and self.y is not None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "timestamp": round(self.timestamp, 4),
            "action": self.action,
        }
        if self.x is not None:
            payload["x"] = int(self.x)
        if self.y is not None:
            payload["y"] = int(self.y)
        if self.target is not None:
            payload["target"] = self.target
        if self.data:
            payload["data"] = self.data
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Action:
        return cls(
            action=str(payload.get("action", "unknown")),
            timestamp=float(payload.get("timestamp", 0.0)),
            x=int(payload["x"]) if payload.get("x") is not None else None,
            y=int(payload["y"]) if payload.get("y") is not None else None,
            target=str(payload["target"]) if payload.get("target") is not None else None,
            data=dict(payload.get("data") or {}),
        )


@dataclass(slots=True)
class Keyframe:
    """One extracted frame in a VLM keyframe set."""

    path: Path
    index: int
    timestamp: float
    reason: str  # "scene" | "action"
    action: str | None = None
    width: int = 0
    height: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "path": self.path.name,
            "timestamp": round(self.timestamp, 4),
            "reason": self.reason,
            "action": self.action,
            "width": self.width,
            "height": self.height,
        }


@dataclass(slots=True)
class MediaInfo:
    """The subset of ``ffprobe`` output that Dynamo cares about."""

    path: Path
    duration: float
    width: int
    height: int
    fps: float
    nb_frames: int
    codec: str

    @property
    def size(self) -> Resolution:
        return (self.width, self.height)

    def to_dict(self) -> dict[str, Any]:
        return {
            "duration": round(self.duration, 3),
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 3),
            "nb_frames": self.nb_frames,
            "codec": self.codec,
        }


@dataclass(slots=True)
class Artifacts:
    """Result of :meth:`agent_visual_harness.harness.VisualHarness.export`."""

    session_dir: Path
    telemetry_path: Path
    raw_path: Path | None = None
    video_path: Path | None = None
    webm_path: Path | None = None
    gif_path: Path | None = None
    keyframes_dir: Path | None = None
    keyframes: list[Keyframe] = field(default_factory=list)
    manifest_path: Path | None = None
    duration: float = 0.0
    raw_frame_count: int = 0

    def __len__(self) -> int:
        return len(self.keyframes)

    def __iter__(self) -> Iterator[Keyframe]:
        return iter(self.keyframes)

    @property
    def paths(self) -> list[Path]:
        return [
            p
            for p in (self.video_path, self.webm_path, self.gif_path, self.manifest_path)
            if p is not None
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_dir": str(self.session_dir),
            "raw_path": str(self.raw_path) if self.raw_path else None,
            "video_path": str(self.video_path) if self.video_path else None,
            "webm_path": str(self.webm_path) if self.webm_path else None,
            "gif_path": str(self.gif_path) if self.gif_path else None,
            "keyframes_dir": str(self.keyframes_dir) if self.keyframes_dir else None,
            "manifest_path": str(self.manifest_path) if self.manifest_path else None,
            "telemetry_path": str(self.telemetry_path),
            "duration": round(self.duration, 3),
            "raw_frame_count": self.raw_frame_count,
            "keyframes": [k.to_dict() for k in self.keyframes],
        }

    def summary(self) -> str:
        lines = [
            f"session   {self.session_dir}",
            f"telemetry {self.telemetry_path}",
        ]
        if self.raw_path:
            lines.append(f"raw       {self.raw_path}")
        for label, path in (
            ("mp4", self.video_path),
            ("webm", self.webm_path),
            ("gif", self.gif_path),
        ):
            if path:
                lines.append(f"{label:<10} {path}  ({_human_size(path)})")
        if self.manifest_path or self.keyframes or self.keyframes_dir:
            size = f"  ({_human_size(self.manifest_path)} manifest)" if self.manifest_path else ""
            where = self.keyframes_dir or self.session_dir
            lines.append(f"keyframes {len(self.keyframes)} frames -> {where}{size}")
        return "\n".join(lines)


def _human_size(path: Path) -> str:
    try:
        size = path.stat().st_size
    except OSError:
        return "?"
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file, skipping blank and malformed lines."""
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
    return records


def records_to_actions(records: Iterable[Mapping[str, Any]]) -> list[Action]:
    return [Action.from_dict(r) for r in records]


def sort_actions(actions: Sequence[Action]) -> list[Action]:
    return sorted(actions, key=lambda a: a.timestamp)
