"""Stream recorder — a non-blocking ``ffmpeg -f x11grab`` daemon.

Two properties matter more than anything else here:

1. **It never blocks the agent.** Capture runs in a detached child process while
   :meth:`Recorder.log_action` and :meth:`Recorder.take_screenshot` return
   immediately.
2. **It always leaves a playable file.** ffmpeg writes the ``moov`` atom last, so
   a ``SIGTERM``/``SIGKILL``/power-cut produces an unplayable MP4. Dynamo stops
   the recorder with ``SIGINT`` — including when the caller's code raises — which
   makes ffmpeg run its trailer path and finalise the index. If a file still ends
   up broken, :func:`_ffmpeg.repair_media` gets a shot at it before we give up.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from types import FrameType
from typing import Any
from collections.abc import Sequence

from ._ffmpeg import FFMPEG, probe, repair_media, require_binary, run_bytes
from .errors import RecorderError, RecorderStartTimeout
from .types import Action, MediaInfo, Resolution, parse_resolution

__all__ = ["Recorder"]

_SCREENSHOT_TIMEOUT = 30.0


class Recorder:
    """Capture an X display to a raw MP4 while logging agent actions.

    Parameters
    ----------
    display:
        Display to capture, e.g. ``":99"``.
    size:
        Capture geometry. Must match the display or ffmpeg will grab garbage.
    output:
        Destination for the raw capture. Defaults to ``<session_dir>/raw.mp4``.
    fps:
        Capture framerate. 15 is usually enough for UI review; 30 when you need
        to judge animation smoothness.
    crf / preset:
        Raw-capture quality. Deliberately generous (low CRF) — this file is an
        intermediate that gets re-encoded during :meth:`export`, and re-encoding
        from a high-quality source keeps code text crisp.
    """

    def __init__(
        self,
        display: str | int = ":99",
        size: str | Resolution = (1920, 1080),
        output: Path | str | None = None,
        fps: int = 24,
        crf: int = 16,
        preset: str = "veryfast",
        pix_fmt: str = "yuv420p",
        telemetry_path: Path | str | None = None,
        draw_mouse: int = 1,
        startup_timeout: float = 30.0,
        shutdown_timeout: float = 60.0,
    ) -> None:
        self.display = str(display)
        self.size: Resolution = parse_resolution(size)
        self.output = Path(output) if output else Path("raw.mp4")
        self.fps = int(fps)
        self.crf = int(crf)
        self.preset = preset
        self.pix_fmt = pix_fmt
        self.telemetry_path = Path(telemetry_path) if telemetry_path else None
        self.draw_mouse = int(draw_mouse)
        self.startup_timeout = float(startup_timeout)
        self.shutdown_timeout = float(shutdown_timeout)

        self._proc: subprocess.Popen[bytes] | None = None
        self._stderr: deque[str] = deque(maxlen=60)
        self._pump: threading.Thread | None = None
        self._start_time: float | None = None
        self._stop_time: float | None = None
        self._repaired = False
        self._lock = threading.Lock()
        self._telemetry_handle = None

    # ------------------------------------------------------------- properties

    @property
    def width(self) -> int:
        return self.size[0]

    @property
    def height(self) -> int:
        return self.size[1]

    @property
    def is_recording(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def elapsed(self) -> float:
        """Seconds since :meth:`start` (frozen once stopped)."""
        if self._start_time is None:
            return 0.0
        end = self._stop_time if self._stop_time is not None else time.monotonic()
        return max(0.0, end - self._start_time)

    @property
    def raw_frame_count(self) -> int:
        """Approximate captured frames, for reporting before the file is closed."""
        return int(self.elapsed * self.fps)

    @property
    def stderr_tail(self) -> list[str]:
        return list(self._stderr)

    def _error_tail(self, lines: int = 6) -> str:
        """Last few ffmpeg diagnostics. ``deque`` does not support slicing."""
        return "\n".join(list(self._stderr)[-lines:]) or "(no stderr captured)"

    @property
    def was_repaired(self) -> bool:
        """``True`` if the raw file needed a rebuild pass after shutdown."""
        return self._repaired

    # -------------------------------------------------------------- lifecycle

    def build_command(self) -> list[str]:
        return [
            FFMPEG,
            "-hide_banner",
            "-loglevel", "warning",
            "-y",
            "-f", "x11grab",
            "-draw_mouse", str(self.draw_mouse),
            "-framerate", str(self.fps),
            "-video_size", f"{self.width}x{self.height}",
            "-i", self.display,
            "-an",
            "-c:v", "libx264",
            "-preset", self.preset,
            "-crf", str(self.crf),
            "-pix_fmt", self.pix_fmt,
            # Without this, a static screen produces so little data that ffmpeg
            # holds it all in its internal buffer and the file stays at the
            # 48-byte container header for the whole session. Flushing per
            # packet means progress is visible on disk *and* a hard kill loses
            # only the current buffer.
            "-flush_packets", "1",
            str(self.output),
        ]

    def start(self) -> Recorder:
        if self.is_recording:
            raise RecorderError("recorder is already running")

        require_binary(
            FFMPEG,
            "Dynamo needs ffmpeg to capture and encode. On Debian/Ubuntu: "
            "apt-get install -y ffmpeg",
        )
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self._stderr.clear()
        self._repaired = False
        self._stop_time = None

        try:
            self._proc = subprocess.Popen(  # noqa: S603 - argv list
                self.build_command(),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                start_new_session=True,  # isolate from the agent's Ctrl-C
            )
        except OSError as exc:
            raise RecorderError(f"could not spawn ffmpeg: {exc}") from exc

        self._pump = threading.Thread(target=self._drain_stderr, daemon=True, name="dynamo-recorder-stderr")
        self._pump.start()
        self._open_telemetry()

        try:
            self._wait_until_capturing()
        except Exception:
            self._kill()
            self._close_telemetry()
            raise

        self._start_time = time.monotonic()
        return self

    def _drain_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        for raw in iter(proc.stderr.readline, b""):
            line = raw.decode("utf-8", "replace").rstrip()
            if line:
                self._stderr.append(line)

    def _wait_until_capturing(self) -> None:
        """Block until ffmpeg is demonstrably writing frames.

        The raw capture is an ordinary MP4, so it only becomes *decodable* once
        the trailer is written — probing it here would never succeed. Readiness
        is therefore "the process is alive and the file is growing", which is
        both cheap and exactly the property we need: frames are landing.

        ``-flush_packets 1`` is what makes the second half of that true on a
        static screen; without it the file legitimately sits at the container
        header until the muxer decides to flush.
        """
        proc = self._proc
        assert proc is not None
        deadline = time.monotonic() + self.startup_timeout
        observed: set[int] = set()

        while time.monotonic() < deadline:
            code = proc.poll()
            if code is not None:
                tail = self._error_tail()
                raise RecorderError(f"ffmpeg exited immediately (exit {code}):\n{tail}")
            try:
                size = self.output.stat().st_size
            except OSError:
                size = 0
            if size > 0:
                observed.add(size)
                # Two distinct sizes means the encoder advanced between polls.
                if len(observed) >= 2:
                    return
            time.sleep(0.1)

        tail = self._error_tail()
        raise RecorderStartTimeout(
            f"ffmpeg produced no frames from {self.display} within "
            f"{self.startup_timeout:g}s.\n{tail}\n"
            f"Check that the display is up and matches {self.width}x{self.height}."
        )

    def stop(self) -> Path:
        """Finalise the recording and return the path to the playable file.

        Safe to call more than once.
        """
        proc, self._proc = self._proc, None
        if proc is not None:
            if proc.poll() is None:
                self._send_sigint(proc)
                try:
                    proc.wait(timeout=self.shutdown_timeout)
                except subprocess.TimeoutExpired:
                    self._stderr.append("SIGINT ignored; escalating to SIGTERM")
                    try:
                        proc.terminate()
                        proc.wait(timeout=15.0)
                    except subprocess.TimeoutExpired:  # pragma: no cover
                        proc.kill()
                        proc.wait(timeout=5.0)
            elif proc.returncode not in (0, 255):
                # ffmpeg died on its own: SIGINT exit code is 255 on some builds
                # when it is interrupted, anything else is a genuine failure.
                tail = "\n".join(self._stderr)
                if "Immediate exit requested" not in tail and tail:
                    self._stderr.append(f"ffmpeg exited with {proc.returncode}")
        if self._pump is not None:
            self._pump.join(timeout=5.0)
            self._pump = None

        self._stop_time = time.monotonic()
        self._close_telemetry()
        self._finalize()
        return self.output

    def _send_sigint(self, proc: subprocess.Popen[bytes]) -> None:
        """SIGINT the ffmpeg process group so it flushes the ``moov`` atom."""
        try:
            if hasattr(os, "killpg") and hasattr(signal, "SIGINT"):
                os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            else:  # pragma: no cover - non-POSIX
                proc.send_signal(signal.SIGINT)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.send_signal(signal.SIGINT)
            except (ProcessLookupError, OSError):
                pass

    def _kill(self) -> None:
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.kill()
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:  # pragma: no cover
                pass
        self._proc = None

    def _finalize(self) -> None:
        """Make sure the file we hand back is actually playable."""
        if not self.output.exists():
            raise RecorderError(
                f"recording produced no file at {self.output}\n" + self._error_tail()
            )
        if self.output.stat().st_size == 0:
            raise RecorderError(f"recording at {self.output} is empty")
        try:
            probe(self.output)
        except Exception as exc:
            self._stderr.append(f"unplayable output: {exc}")
            if repair_media(self.output):
                self._repaired = True
            else:
                raise RecorderError(
                    f"recording at {self.output} could not be finalised or repaired\n"
                    + self._error_tail()
                ) from exc

    # -------------------------------------------------------------- telemetry

    def _open_telemetry(self) -> None:
        if self.telemetry_path is None:
            return
        self.telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        self._telemetry_handle = self.telemetry_path.open("a", encoding="utf-8")

    def _close_telemetry(self) -> None:
        handle, self._telemetry_handle = self._telemetry_handle, None
        if handle is not None:
            try:
                handle.flush()
                handle.close()
            except OSError:  # pragma: no cover
                pass

    def log_action(
        self,
        action: str,
        x: int | None = None,
        y: int | None = None,
        target: str | None = None,
        **data: Any,
    ) -> Action:
        """Timestamp an agent action into ``telemetry.jsonl``.

        Non-blocking: one line written and flushed. Never raises because of an
        agent-supplied value — telemetry is observability, it must not be able to
        take down the run it is observing.
        """
        record = Action(
            action=str(action),
            timestamp=self.elapsed,
            x=None if x is None else int(x),
            y=None if y is None else int(y),
            target=target,
            data=dict(data),
        )
        if self._telemetry_handle is not None:
            with self._lock:
                try:
                    self._telemetry_handle.write(json.dumps(record.to_dict()) + "\n")
                    self._telemetry_handle.flush()
                except (OSError, ValueError, TypeError) as exc:  # pragma: no cover
                    self._stderr.append(f"telemetry write failed: {exc}")
        return record

    @property
    def actions(self) -> list[Action]:
        """Every action logged so far, read back from disk."""
        if self.telemetry_path is None or not self.telemetry_path.exists():
            return []
        out: list[Action] = []
        with self.telemetry_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(Action.from_dict(json.loads(line)))
                except (json.JSONDecodeError, ValueError):
                    continue
        return out

    # ------------------------------------------------------------- screenshot

    def take_screenshot(self, fmt: str = "png") -> bytes:
        """Grab the current framebuffer as encoded image bytes.

        Uses a short-lived ffmpeg invocation rather than ``xwd`` + a converter so
        the harness has no extra system dependency, and so it works while the
        recorder is running.
        """
        codec = {"png": "png", "jpeg": "mjpeg", "jpg": "mjpeg", "webp": "libwebp"}.get(fmt.lower())
        if codec is None:
            raise ValueError(f"unsupported screenshot format {fmt!r}; use png, jpeg or webp")
        if codec == "libwebp":
            from ._ffmpeg import has_encoder  # local import keeps module import cheap

            if not has_encoder("libwebp"):
                codec = "png"

        data = run_bytes(
            [
                FFMPEG, "-hide_banner", "-loglevel", "error",
                "-f", "x11grab",
                "-draw_mouse", str(self.draw_mouse),
                "-video_size", f"{self.width}x{self.height}",
                "-i", self.display,
                "-frames:v", "1",
                "-c:v", codec,
                "-f", "image2pipe",
                "-",
            ],
            timeout=_SCREENSHOT_TIMEOUT,
        )
        if len(data) < 128:
            raise RecorderError(
                f"screenshot from {self.display} returned only {len(data)} bytes — "
                "is the display up and the geometry correct?"
            )
        return data

    # ------------------------------------------------------------- inspection

    def media_info(self) -> MediaInfo:
        """Probe the finished capture (raises if it was never finalised)."""
        return probe(self.output)

    def __enter__(self) -> Recorder:
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: FrameType | None,
    ) -> bool:
        try:
            self.stop()
        except RecorderError:
            if exc is None:
                raise
        return False

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        state = "recording" if self.is_recording else "stopped"
        return f"<Recorder {self.display} {self.width}x{self.height}@{self.fps} {state}>"


def read_actions(path: Path | str) -> list[Action]:
    """Read a ``telemetry.jsonl`` file into :class:`Action` objects."""
    file = Path(path)
    if not file.exists():
        return []
    out: list[Action] = []
    with file.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(Action.from_dict(json.loads(line)))
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
    return sorted(out, key=lambda a: a.timestamp)


def x11_clients_ok(display: str) -> bool:  # pragma: no cover - helper for callers
    """Cheap sanity check that something can render to ``display``."""
    require_binary(FFMPEG)
    try:
        data = run_bytes(
            [FFMPEG, "-hide_banner", "-loglevel", "error", "-f", "x11grab",
             "-video_size", "16x16", "-i", display, "-frames:v", "1",
             "-f", "rawvideo", "-pix_fmt", "gray", "-"],
            timeout=20.0,
        )
    except Exception:
        return False
    return len(data) > 0


def build_env(display: str, extra: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for a child process that should render into ``display``."""
    env = dict(os.environ)
    env["DISPLAY"] = display
    if extra:
        env.update(extra)
    return env


def run_on_display(cmd: Sequence[str], display: str, **kwargs: Any) -> subprocess.CompletedProcess:
    """Run ``cmd`` with ``DISPLAY`` set. Used by the CLI and by tests."""
    return subprocess.run(  # noqa: S603
        list(cmd),
        env=build_env(display),
        **kwargs,
    )
