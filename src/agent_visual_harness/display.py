"""Virtual display manager — owns the full Xvfb lifecycle.

Dynamo starts its own X server instead of asking the caller to export
``DISPLAY`` and hope. That removes the single most common source of flaky CI:
recording before X is actually accepting connections.

The readiness probe connects to the X11 unix socket. It is cheap (no subprocess),
portable, and fails fast; when ``xdpyinfo`` happens to be installed it is used as
a second, stronger confirmation.
"""

from __future__ import annotations

import errno
import os
import shutil
import socket
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from types import FrameType
from collections.abc import Sequence

from ._ffmpeg import require_binary
from .errors import DisplayError, DisplayInUseError, DisplayStartTimeout
from .types import Resolution, parse_resolution

__all__ = [
    "VirtualDisplay",
    "display_socket_path",
    "is_display_up",
    "ensure_x11_socket_dir",
]

_X11_SOCKET_DIR = Path("/tmp/.X11-unix")
_SOCKET_DIR_MODE = 0o1777


def ensure_x11_socket_dir() -> bool:
    """Make sure ``/tmp/.X11-unix`` exists with mode 1777.

    X refuses to bind its unix listener unless the directory is world-writable
    *with the sticky bit set*, and it only says so once, on stderr:

        _XSERVTransmkdir: Mode of /tmp/.X11-unix should be set to 1777
        _XSERVTransSocketCreateListener: failed to bind listener

    …then it keeps running, accepts nothing, and every connection attempt times
    out. Minimal container images ship the directory without the sticky bit, so
    this is worth fixing rather than waiting out the timeout for.

    Returns ``True`` when the directory had to be changed.
    """
    try:
        if not _X11_SOCKET_DIR.exists():
            _X11_SOCKET_DIR.mkdir(mode=_SOCKET_DIR_MODE, parents=True, exist_ok=True)
        # mkdir()'s mode argument is masked by umask, so it can never be trusted
        # to produce 1777 on its own; chmod is the only reliable half.
        if (_X11_SOCKET_DIR.stat().st_mode & 0o7777) != _SOCKET_DIR_MODE:
            _X11_SOCKET_DIR.chmod(_SOCKET_DIR_MODE)
            return True
    except OSError:
        return False
    return False


def display_socket_path(display: str | int) -> Path:
    """Filesystem path of the unix socket backing ``:99``."""
    number = _display_number(display)
    return _X11_SOCKET_DIR / f"X{number}"


def _display_number(display: str | int) -> int:
    text = str(display).strip()
    if not text:
        raise DisplayError("display must look like ':99', got an empty string")
    if not text.startswith(":"):
        # Tolerate "99" as a convenience.
        if text.isdigit():
            return int(text)
        raise DisplayError(f"invalid display {display!r}; expected ':99'")
    tail = text[1:].split(".", 1)[0]
    if not tail.isdigit():
        raise DisplayError(f"invalid display {display!r}; expected ':99'")
    return int(tail)


def is_display_up(display: str | int, timeout: float = 0.5) -> bool:
    """Return ``True`` when an X server is listening on ``display``."""
    sock_path = display_socket_path(display)
    if not sock_path.exists():
        return False
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect(str(sock_path))
    except OSError:
        return False
    finally:
        sock.close()
    return True


class VirtualDisplay:
    """Start, probe and stop a private ``Xvfb`` server.

    Parameters
    ----------
    display:
        Display number, e.g. ``":99"``.
    resolution:
        ``(1920, 1080)`` or the X geometry string ``"1920x1080x24"``.
    depth:
        Colour depth used when ``resolution`` does not carry one.
    dpi:
        Reported DPI. Some toolkits size widgets differently at 96 vs 144, so this
        is configurable for reproducible layout.
    timeout:
        Seconds to wait for X to start accepting connections.
    adopt_existing:
        When ``False`` (default) starting on a display that is already serving
        raises :class:`DisplayInUseError` — Dynamo must never kill a display it
        does not own. Set ``True`` to attach to an existing server and leave it
        running on exit.
    """

    def __init__(
        self,
        display: str | int = ":99",
        resolution: str | Resolution = (1920, 1080),
        depth: int = 24,
        dpi: int = 96,
        timeout: float = 20.0,
        adopt_existing: bool = False,
        log_path: Path | str | None = None,
    ) -> None:
        self.display = self._normalise(display)
        self.resolution: Resolution = parse_resolution(resolution, depth=depth)
        self.depth = depth
        self.dpi = dpi
        self.timeout = float(timeout)
        self.adopt_existing = adopt_existing
        self.log_path = Path(log_path) if log_path else None

        self._proc: subprocess.Popen[bytes] | None = None
        self._drain: threading.Thread | None = None
        self._stderr: deque[str] = deque(maxlen=80)
        self._owns_process = False
        self._started = False
        self._startup_time: float | None = None

    # ------------------------------------------------------------------ setup

    @staticmethod
    def _normalise(display: str | int) -> str:
        number = _display_number(display)
        if number < 0:
            raise DisplayError(f"invalid display {display!r}")
        return f":{number}"

    @property
    def width(self) -> int:
        return self.resolution[0]

    @property
    def height(self) -> int:
        return self.resolution[1]

    @property
    def size(self) -> str:
        return f"{self.width}x{self.height}"

    @property
    def geometry(self) -> str:
        return f"{self.width}x{self.height}x{self.depth}"

    @property
    def env(self) -> dict[str, str]:
        """Environment overlay that points child processes at this display."""
        return {"DISPLAY": self.display}

    @property
    def is_running(self) -> bool:
        if self.adopt_existing and not self._owns_process:
            return is_display_up(self.display)
        return self._proc is not None and self._proc.poll() is None

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc is not None else None

    # ------------------------------------------------------------- lifecycle

    def start(self) -> VirtualDisplay:
        # Already up, and either adopted or started by us: nothing to do.
        # Without the `_owns_process` half, a second start() on a display this
        # object launched would look like somebody else's server and raise.
        if self.is_running and (self.adopt_existing or self._owns_process):
            self._started = True
            return self

        if is_display_up(self.display):
            if not self.adopt_existing:
                raise DisplayInUseError(
                    f"display {self.display} is already serving connections. "
                    f"Pick a free display number, or pass adopt_existing=True to "
                    f"record onto the existing X server."
                )
            self._started = True
            return self

        xvfb = require_binary(
            "Xvfb",
            "Dynamo needs Xvfb to provide a virtual display. On Debian/Ubuntu: "
            "apt-get install -y xvfb",
        )
        repaired_dir = ensure_x11_socket_dir()
        cmd: list[str] = [
            xvfb,
            self.display,
            "-screen", "0", self.geometry,
            "-dpi", str(self.dpi),
            "-nolisten", "tcp",
            "-noreset",
        ]
        log_handle = None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = self.log_path.open("ab")

        try:
            self._proc = subprocess.Popen(  # noqa: S603 - argv list
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                start_new_session=True,  # own process group: signals don't leak
            )
        except OSError as exc:
            raise DisplayError(f"could not spawn Xvfb: {exc}") from exc
        finally:
            if log_handle is not None:
                log_handle.close()

        self._owns_process = True
        self._started = True
        self._drain = threading.Thread(target=self._drain_stderr, daemon=True, name="dynamo-xvfb-stderr")
        self._drain.start()
        try:
            self._wait_until_ready()
        except Exception:
            self.stop()
            raise
        self._startup_time = time.monotonic()
        if repaired_dir:
            self._stderr.append(
                f"repaired {_X11_SOCKET_DIR} permissions to 1777 "
                "(X refuses to bind its unix socket otherwise)"
            )
        return self

    def _drain_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        for raw in iter(proc.stderr.readline, b""):
            line = raw.decode("utf-8", "replace").rstrip()
            if line:
                self._stderr.append(line)

    @property
    def stderr_tail(self) -> list[str]:
        """Xvfb's own diagnostics — the only place some failures explain themselves."""
        return list(self._stderr)

    def _wait_until_ready(self) -> None:
        """Block until X accepts connections, or give up and explain why."""
        deadline = time.monotonic() + self.timeout
        socket_path = display_socket_path(self.display)
        last_error: Exception | None = None

        while time.monotonic() < deadline:
            proc = self._proc
            if proc is not None and proc.poll() is not None:
                raise DisplayError(
                    f"Xvfb exited immediately with code {proc.returncode} on {self.display}."
                    + (f" See {self.log_path} for details." if self.log_path else "")
                )
            if socket_path.exists():
                try:
                    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
                        probe.settimeout(0.5)
                        probe.connect(str(socket_path))
                    break
                except OSError as exc:
                    last_error = exc
            time.sleep(0.05)
        else:
            raise DisplayStartTimeout(self._timeout_message(last_error))

        # Second, stronger check when xdpyinfo is available.
        if shutil.which("xdpyinfo"):
            proc = subprocess.run(  # noqa: S603
                ["xdpyinfo", "-display", self.display],
                capture_output=True,
                timeout=min(10.0, self.timeout),
                check=False,
            )
            if proc.returncode != 0:
                raise DisplayStartTimeout(
                    f"X socket for {self.display} exists but xdpyinfo could not query it"
                )

    def _timeout_message(self, last_error: Exception | None) -> str:
        """Explain a readiness timeout instead of just reporting one.

        Xvfb's own stderr is the only source of truth for most startup failures,
        and the sticky-bit-on-/tmp/.X11-unix one is both common and silent from
        the outside.
        """
        socket_path = display_socket_path(self.display)
        mode_line = ""
        try:
            actual = _X11_SOCKET_DIR.stat().st_mode & 0o7777
            if actual != _SOCKET_DIR_MODE:
                mode_line = (
                    f"\nNote: {_X11_SOCKET_DIR} is mode {actual:04o}; X requires "
                    f"{_SOCKET_DIR_MODE:04o} and silently binds nothing otherwise."
                )
        except OSError:
            mode_line = f"\nNote: {_X11_SOCKET_DIR} does not exist."

        tail = [line for line in self._stderr if "xkbcomp" not in line.lower()
                and "keysym" not in line.lower()]
        detail = "\n".join(tail[-6:])
        return (
            f"Xvfb on {self.display} did not accept connections within "
            f"{self.timeout:g}s (socket {socket_path}: "
            f"{errno.errorcode.get(last_error.errno, 'absent') if last_error else 'absent'})"
            f"{mode_line}"
            + (f"\nXvfb said:\n{detail}" if detail else "")
        )

    def stop(self, timeout: float = 10.0) -> None:
        """Terminate Xvfb and wait for the socket to disappear."""
        proc, self._proc = self._proc, None
        if proc is not None:
            if proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    try:
                        proc.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:  # pragma: no cover - unkillable
                        pass
                except OSError:  # pragma: no cover - already reaped
                    pass
        drain, self._drain = self._drain, None
        if drain is not None:
            drain.join(timeout=5.0)
        if self._owns_process:
            self._wait_until_down(timeout=5.0)
        self._owns_process = False
        self._started = False

    def _wait_until_down(self, timeout: float = 5.0) -> None:
        deadline = time.monotonic() + timeout
        socket_path = display_socket_path(self.display)
        while time.monotonic() < deadline:
            if not is_display_up(self.display, timeout=0.2):
                return
            time.sleep(0.05)
        # A leftover socket with no server is harmless; remove it so the next
        # start() does not think the display is busy.
        try:
            socket_path.unlink()
        except OSError:
            pass

    # -------------------------------------------------------- context manager

    def __enter__(self) -> VirtualDisplay:
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: FrameType | None,
    ) -> bool:
        self.stop()
        return False

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        state = "running" if self.is_running else "stopped"
        return f"<VirtualDisplay {self.display} {self.size}x{self.depth} {state}>"


def probe_display(cmd: Sequence[str], display: str, timeout: float = 10.0) -> subprocess.CompletedProcess:
    """Run ``cmd`` against ``display``; convenience for tests and the CLI."""
    env = {**os.environ, "DISPLAY": display}
    return subprocess.run(  # noqa: S603
        list(cmd),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
