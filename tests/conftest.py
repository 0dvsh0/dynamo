"""Shared fixtures.

The suite is layered so that most of it runs anywhere and only the genuinely
X11-dependent tests need a Linux host with Xvfb. On Windows, or on CI without an
X server, those tests skip with a clear reason instead of failing.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from collections.abc import Iterator

import pytest

# Allow `pytest tests/` from a checkout without installing the package first.
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from agent_visual_harness import (  # noqa: E402
    Recorder,
    Session,
    VirtualDisplay,
    is_display_up,
)
from agent_visual_harness._ffmpeg import FFMPEG  # noqa: E402

XFVB = "Xvfb"
XSETROOT = "xsetroot"

#: High-contrast, luma-separated colours. Every cut between adjacent entries is
#: reliably scored above ffmpeg's default 0.05 scene threshold at any size.
CUT_PALETTE = (
    "0x101010",
    "0xf0f0f0",
    "0x1070d0",
    "0xf08010",
    "0x20b060",
    "0xa030e0",
    "0x101010",
)
#: Colours for repainting a live X root window in recorder/harness tests.
SCREEN_PAINT = (
    "#101010",
    "#f0f0f0",
    "#1070d0",
    "#f08010",
    "#20b060",
    "#a030e0",
)


def _have(binary: str) -> bool:
    return shutil.which(binary) is not None


requires_ffmpeg = pytest.mark.skipif(
    not _have(FFMPEG), reason=f"{FFMPEG} is not on PATH"
)
requires_xvfb = pytest.mark.skipif(
    not _have(XFVB), reason=f"{XFVB} is not on PATH (apt-get install xvfb)"
)
requires_xsetroot = pytest.mark.skipif(
    not _have(XSETROOT),
    reason=f"{XSETROOT} is not on PATH (apt-get install x11-xserver-utils)",
)
requires_linux = pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="X11 virtual display is Linux-only",
)

def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: end-to-end tests that encode real video")
    config.addinivalue_line("markers", "xfvb: needs a working Xvfb / X11 display")


# ------------------------------------------------------------------ utilities


def ffmpeg(*args: str) -> str:
    """Run ffmpeg, raising with its stderr on failure; returns stderr."""
    proc = subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr}")
    return proc.stderr


def make_clip(
    path: Path,
    colours: tuple[str, ...] = CUT_PALETTE[:3],
    size: str = "640x480",
    fps: int = 15,
    seconds_per_colour: float = 1.0,
) -> Path:
    """Build a deterministic clip with one hard cut per colour.

    ``make_clip(p, CUT_PALETTE[:3])`` yields exactly two scene changes, at t=1.0s
    and t=2.0s, which is what the reducer tests assert against.

    The palette is high-contrast on purpose. ffmpeg's scene metric is a
    luma-weighted mean absolute difference, so low-contrast cuts can score 0 and
    vanish — ff *named* colours like ``red`` -> ``green`` do exactly that. Test
    fixtures must not depend on a detector's blind spots.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    inputs: list[str] = []
    for colour in colours:
        inputs += [
            "-f", "lavfi",
            "-i", f"color=c={colour}:s={size}:r={fps}:d={seconds_per_colour}",
        ]
    streams = "".join(f"[{i}:v]" for i in range(len(colours)))
    graph = f"{streams}concat=n={len(colours)}:v=1:a=0,format=yuv420p[v]"
    ffmpeg(
        *inputs,
        "-filter_complex", graph,
        "-map", "[v]",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-g", "1",  # all-intra keeps scene cuts exactly where we put them
        "-crf", "18",
        str(path),
    )
    return path


def make_noisy_clip(
    path: Path, size: str = "1280x720", fps: int = 15, seconds: float = 2.0
) -> Path:
    """A clip with high-frequency detail, so bitrate tests mean something."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg(
        "-f", "lavfi",
        "-i", f"testsrc2=s={size}:r={fps}:d={seconds}",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "14",
        str(path),
    )
    return path


def paint(display: str, colour: str) -> bool:
    """Repaint the X root window so the capture contains a visible change."""
    if not _have(XSETROOT):
        return False
    proc = subprocess.run(
        [XSETROOT, "-display", display, "-solid", colour],
        capture_output=True,
        check=False,
    )
    return proc.returncode == 0


# ------------------------------------------------------------------- fixtures


@pytest.fixture(scope="session")
def display_number() -> int:
    """Pick a display number that is currently free."""
    for number in range(90, 120):
        if not is_display_up(f":{number}"):
            return number
    pytest.skip("no free X display number in :90-:119")


@pytest.fixture(scope="session")
def display(display_number: int):
    """A live Xvfb display shared by the whole test session.

    Skips (rather than errors) when Xvfb is unavailable, so the X11-dependent
    modules can be collected and reported cleanly on a non-Linux host.
    """
    if not _have(XFVB):
        pytest.skip(f"{XFVB} is not on PATH (apt-get install xvfb)")
    vd = VirtualDisplay(display=f":{display_number}", resolution=(640, 480), dpi=96)
    try:
        vd.start()
    except Exception as exc:  # no X server, no /tmp/.X11-unix, sandboxed CI...
        pytest.skip(f"cannot start Xvfb here: {exc}")
    yield vd
    vd.stop()


@pytest.fixture
def free_display() -> Iterator[str]:
    """A display number that is free right now, *not* the shared one.

    Tests that manage a display's whole lifecycle need a number nobody else in
    the session is holding, otherwise they collide with the session fixture.
    """
    for number in range(150, 175):
        if not is_display_up(f":{number}"):
            yield f":{number}"
            return
    pytest.skip("no free display number in :150-:174")


@pytest.fixture
def session(tmp_path: Path) -> Session:
    return Session.create(tmp_path / "session", display=":99", resolution=(640, 480), fps=15)


@pytest.fixture
def synthetic_clip(tmp_path: Path) -> Path:
    """A 3-second clip with exactly two scene changes, at t=1.0s and t=2.0s."""
    return make_clip(tmp_path / "clip.mp4", CUT_PALETTE[:3])


@pytest.fixture
def clip_recorder(tmp_path: Path, display) -> Recorder:
    """A recorder already capturing the shared display."""
    rec = Recorder(
        display=display.display,
        size=(640, 480),
        output=tmp_path / "raw.mp4",
        fps=15,
        crf=20,
        telemetry_path=tmp_path / "telemetry.jsonl",
        startup_timeout=30.0,
    )
    rec.start()
    yield rec
    if rec.is_recording:
        rec.stop()
