"""Thin, well-behaved wrappers around the ``ffmpeg`` / ``ffprobe`` binaries.

Everything in Dynamo shells out to ffmpeg; this module is the single place where
that happens so that error messages, timeouts and version quirks are handled
consistently. Nothing here is public API.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from collections.abc import Sequence

from .errors import DependencyError, ProcessingError
from .types import MediaInfo

__all__ = [
    "FFMPEG",
    "FFPROBE",
    "have_binary",
    "require_binary",
    "run",
    "run_bytes",
    "probe",
    "has_encoder",
    "repair_media",
    "ffmpeg_version",
    "vfr_flag",
]

FFMPEG = os.environ.get("DYNAMO_FFMPEG", "ffmpeg")
FFPROBE = os.environ.get("DYNAMO_FFPROBE", "ffprobe")

_CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


def have_binary(name: str) -> bool:
    return shutil.which(name) is not None


def require_binary(name: str, hint: str = "") -> str:
    """Return the resolved path to ``name`` or raise a helpful error."""
    found = shutil.which(name)
    if found is None:
        detail = f" {hint}" if hint else ""
        raise DependencyError(
            f"required binary {name!r} was not found on PATH.{detail}\n"
            f"Install it with: apt-get update && apt-get install -y {name}"
        )
    return found


def _popen_kwargs() -> dict:
    return {"creationflags": _CREATE_NO_WINDOW} if os.name == "nt" else {}


def run(
    cmd: Sequence[str],
    *,
    timeout: float | None = None,
    check: bool = True,
    input_text: str | None = None,
) -> subprocess.CompletedProcess:
    """Run a command with ``-nostdin`` semantics and capture text output."""
    try:
        proc = subprocess.run(  # noqa: S603 - argv list, never a shell string
            list(cmd),
            stdin=subprocess.DEVNULL if input_text is None else None,
            input=input_text,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            **_popen_kwargs(),  # type: ignore[arg-type]
        )
    except subprocess.TimeoutExpired as exc:
        raise ProcessingError(f"command timed out after {timeout}s: {_pretty(cmd)}") from exc
    if check and proc.returncode != 0:
        raise ProcessingError(
            f"command failed (exit {proc.returncode}): {_pretty(cmd)}\n{_tail(proc.stderr)}"
        )
    return proc


def run_bytes(
    cmd: Sequence[str],
    *,
    timeout: float | None = None,
    input_bytes: bytes | None = None,
) -> bytes:
    """Run a command and return raw stdout bytes.

    Pass ``input_bytes`` when the command reads an image (or any other payload)
    from stdin — ``-f image2pipe -i -`` style filters need the data to arrive on
    the pipe, not on devnull.
    """
    # subprocess.run refuses `stdin` and `input` together, so pick one: pipe the
    # payload when there is one, otherwise make sure ffmpeg cannot block on a tty.
    stdin = None if input_bytes is not None else subprocess.DEVNULL
    try:
        proc = subprocess.run(  # noqa: S603
            list(cmd),
            stdin=stdin,
            input=input_bytes,
            capture_output=True,
            timeout=timeout,
            check=False,
            **_popen_kwargs(),  # type: ignore[arg-type]
        )
    except subprocess.TimeoutExpired as exc:
        raise ProcessingError(f"command timed out after {timeout}s: {_pretty(cmd)}") from exc
    if proc.returncode != 0:
        raise ProcessingError(
            f"command failed (exit {proc.returncode}): {_pretty(cmd)}\n"
            f"{_tail(proc.stderr.decode('utf-8', 'replace'))}"
        )
    if not proc.stdout:
        raise ProcessingError(f"command produced no output: {_pretty(cmd)}")
    return proc.stdout


def ffmpeg_version() -> tuple[int, int]:
    """Return ffmpeg's ``(major, minor)`` version."""
    out = run([FFMPEG, "-version"], check=True).stdout
    match = re.search(r"ffmpeg version n?(\d+)\.(\d+)", out)
    if not match:
        return (0, 0)
    return (int(match.group(1)), int(match.group(2)))


def vfr_flag() -> str:
    """Pick the right "variable frame rate" switch for this ffmpeg build.

    ``-vsync`` was replaced by ``-fps_mode`` in ffmpeg 5.1 and removed later; old
    Docker images still carry ffmpeg 4.x, so we probe rather than assume.
    """
    major, _ = ffmpeg_version()
    return "-vsync" if major and major < 5 else "-fps_mode"


_ENCODER_CACHE: dict[str, bool] | None = None


def has_encoder(name: str) -> bool:
    """Whether this ffmpeg build has ``name`` (e.g. ``libwebp``) compiled in.

    Returns ``False`` rather than raising when ffmpeg is missing entirely, so
    that importing the package and constructing a reducer never depends on a
    working toolchain.
    """
    global _ENCODER_CACHE
    if _ENCODER_CACHE is None:
        _ENCODER_CACHE = {}
    if name not in _ENCODER_CACHE:
        try:
            proc = run([FFMPEG, "-hide_banner", "-encoders"], check=False)
        except (OSError, Exception):  # noqa: B014 - ffmpeg missing or failed
            _ENCODER_CACHE[name] = False
        else:
            _ENCODER_CACHE[name] = bool(
                re.search(rf"^\s*\S+\s+{re.escape(name)}\b", proc.stdout, re.M)
            )
    return _ENCODER_CACHE[name]


def probe(path: Path | str) -> MediaInfo:
    """Read stream metadata for ``path``.

    Raises :class:`DependencyError` when ffprobe is missing and
    :class:`ProcessingError` when the file cannot be decoded (typically a missing
    ``moov`` atom after a hard kill).
    """
    path = Path(path)
    require_binary(FFPROBE)
    if not path.exists():
        raise ProcessingError(f"cannot probe missing file: {path}")
    proc = run(
        [
            FFPROBE,
            "-v", "error",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        check=False,
    )
    if proc.returncode != 0:
        raise ProcessingError(
            f"ffprobe could not read {path.name} (corrupt or unfinished?): "
            f"{_tail(proc.stderr)}"
        )
    data = json.loads(proc.stdout or "{}")
    streams = data.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise ProcessingError(f"{path} contains no video stream")

    duration = _first_float(video.get("duration"), data.get("format", {}).get("duration"), 0.0)
    return MediaInfo(
        path=path,
        duration=duration,
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
        fps=_parse_rate(video.get("avg_frame_rate") or video.get("r_frame_rate") or "0/1"),
        nb_frames=int(video.get("nb_frames") or 0),
        codec=str(video.get("codec_name") or "unknown"),
    )


def repair_media(path: Path | str) -> bool:
    """Best-effort rebuild of a damaged MP4, with one hard guarantee.

    ffmpeg writes the ``moov`` atom — the index that makes a file seekable — last,
    so a hard kill can leave a file whose payload is intact but whose index is
    gone. This attempts a stream copy to rebuild it and returns ``True`` only when
    the result actually decodes.

    Be clear about what that means, because it is a weaker claim than it sounds:
    **a missing or corrupt ``moov`` is not recoverable for MP4.** Without an index
    there is nothing to demux from, so the stream copy fails and this returns
    ``False``. What it does fix is damage around a surviving index — an
    interrupted finalisation, a truncated ``mdat``, trailing garbage.

    The real protection is upstream: :class:`~agent_visual_harness.recorder.Recorder`
    stops ffmpeg with ``SIGINT`` on every exit path, including when the caller's
    code raises, so the trailer is written in the first place.

    Either way the original is never lost: on failure the input is moved back into
    place byte-for-byte and the caller can inspect or salvage it itself.
    """
    path = Path(path)
    backup = path.with_suffix(path.suffix + ".damaged")
    path.replace(backup)
    proc = run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", str(backup), "-c", "copy", str(path)],
        check=False,
        timeout=300,
    )
    if proc.returncode == 0 and path.exists() and path.stat().st_size > 0:
        try:
            probe(path)
        except (ProcessingError, DependencyError):
            pass
        else:
            backup.unlink(missing_ok=True)
            return True
    if backup.exists() and not path.exists():
        backup.replace(path)
    elif path.exists():
        path.unlink(missing_ok=True)
        backup.replace(path)
    return False


def _parse_rate(rate: str) -> float:
    try:
        num, _, den = rate.partition("/")
        den_f = float(den or 1)
        return float(num) / den_f if den_f else 0.0
    except (TypeError, ValueError):
        return 0.0


def _first_float(*values: object) -> float:
    for value in values:
        try:
            result = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if result > 0:
            return result
    return 0.0


def _pretty(cmd: Sequence[str]) -> str:
    parts = []
    for item in cmd:
        text = str(item)
        parts.append(f'"{text}"' if " " in text else text)
    return " ".join(parts)


def _tail(text: str | None, lines: int = 6) -> str:
    if not text:
        return "(no stderr)"
    kept = [ln for ln in text.strip().splitlines() if ln.strip()][-lines:]
    return "\n".join(kept)
