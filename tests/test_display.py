"""VirtualDisplay tests — all require a real Xvfb."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from agent_visual_harness import (
    DisplayInUseError,
    VirtualDisplay,
    display_socket_path,
    is_display_up,
)
from agent_visual_harness.errors import DisplayError

from .conftest import requires_ffmpeg, requires_linux, requires_xvfb, requires_xsetroot

pytestmark = [requires_linux, requires_xvfb, requires_ffmpeg, pytest.mark.xfvb]


class TestDisplayHelpers:
    def test_socket_path_is_derived_from_the_number(self):
        assert display_socket_path(":99") == Path("/tmp/.X11-unix/X99")
        assert display_socket_path(99) == Path("/tmp/.X11-unix/X99")
        assert display_socket_path(":12.0") == Path("/tmp/.X11-unix/X12")

    @pytest.mark.parametrize("bad", ["", "abc", ":x", ":", "unix:0"])
    def test_rejects_malformed_displays(self, bad):
        with pytest.raises(DisplayError):
            display_socket_path(bad)

    def test_normalises_bare_numbers(self):
        assert VirtualDisplay(display="99").display == ":99"
        assert VirtualDisplay(display=":99.0").display == ":99"


class TestSocketDirectory:
    """Minimal container images ship /tmp/.X11-unix without the sticky bit.

    X then refuses to bind its listener, keeps running, and accepts nothing —
    the symptom is a 20-second readiness timeout with no explanation.
    """

    def test_repairs_a_wrongly_moded_socket_directory(self, tmp_path: Path, monkeypatch):
        from agent_visual_harness import display as display_mod

        fake_dir = tmp_path / ".X11-unix"
        fake_dir.mkdir(mode=0o777)
        assert fake_dir.stat().st_mode & 0o7777 != 0o1777

        monkeypatch.setattr(display_mod, "_X11_SOCKET_DIR", fake_dir)
        assert display_mod.ensure_x11_socket_dir() is True
        assert fake_dir.stat().st_mode & 0o7777 == 0o1777

    def test_creates_the_directory_when_absent(self, tmp_path: Path, monkeypatch):
        """mkdir() masks its mode with umask, so the chmod half must run too."""
        from agent_visual_harness import display as display_mod

        fake_dir = tmp_path / "nested" / ".X11-unix"
        monkeypatch.setattr(display_mod, "_X11_SOCKET_DIR", fake_dir)
        assert display_mod.ensure_x11_socket_dir() is True
        assert fake_dir.is_dir()
        assert fake_dir.stat().st_mode & 0o7777 == 0o1777

    def test_is_a_no_op_when_already_correct(self, tmp_path: Path, monkeypatch):
        from agent_visual_harness import display as display_mod

        fake_dir = tmp_path / ".X11-unix"
        fake_dir.mkdir()
        # mkdir()'s mode argument is masked by umask, so set it explicitly.
        fake_dir.chmod(0o1777)
        monkeypatch.setattr(display_mod, "_X11_SOCKET_DIR", fake_dir)
        assert display_mod.ensure_x11_socket_dir() is False


class TestLifecycle:
    def test_start_creates_a_reachable_socket(self, display: VirtualDisplay):
        assert display.is_running
        assert display_socket_path(display.display).exists()
        assert is_display_up(display.display)

    def test_geometry_is_reported(self, display: VirtualDisplay):
        assert display.resolution == (640, 480)
        assert display.size == "640x480"
        assert display.geometry == "640x480x24"

    def test_env_points_children_at_the_display(self, display: VirtualDisplay):
        assert display.env == {"DISPLAY": display.display}

    def test_stop_releases_the_socket(self, free_display: str):
        vd = VirtualDisplay(display=free_display, resolution=(320, 240))
        vd.start()
        try:
            assert is_display_up(vd.display)
        finally:
            vd.stop()
        assert not vd.is_running
        assert not is_display_up(vd.display)

    def test_context_manager_starts_and_stops(self, free_display: str):
        with VirtualDisplay(display=free_display, resolution=(320, 240)) as vd:
            assert is_display_up(vd.display)
        assert not is_display_up(vd.display)

    def test_start_is_idempotent(self, free_display: str):
        vd = VirtualDisplay(display=free_display, resolution=(320, 240))
        vd.start()
        pid = vd.pid
        try:
            vd.start()
            assert vd.pid == pid
        finally:
            vd.stop()

    def test_stop_is_idempotent(self, free_display: str):
        vd = VirtualDisplay(display=free_display, resolution=(320, 240))
        vd.start()
        vd.stop()
        vd.stop()  # must not raise

    def test_restart_after_stop(self, free_display: str):
        vd = VirtualDisplay(display=free_display, resolution=(320, 240))
        for _ in range(2):
            vd.start()
            assert is_display_up(vd.display)
            vd.stop()
            assert not is_display_up(vd.display)

    def test_start_before_ready_raises_when_xvfb_is_missing(self, free_display, monkeypatch):
        """A missing binary must produce a DependencyError, not a hang."""
        from agent_visual_harness import display as display_mod
        from agent_visual_harness.errors import DependencyError

        monkeypatch.setattr(display_mod, "require_binary", _raise_dependency)
        vd = VirtualDisplay(display=free_display, resolution=(320, 240), timeout=3.0)
        with pytest.raises(DependencyError, match="Xvfb"):
            vd.start()


class TestOwnership:
    def test_refuses_a_display_it_does_not_own(self, display: VirtualDisplay):
        """Dynamo must never kill an X server somebody else started."""
        intruder = VirtualDisplay(display=display.display, resolution=(320, 240))
        with pytest.raises(DisplayInUseError, match="already serving"):
            intruder.start()
        # The original display must be untouched.
        assert is_display_up(display.display)

    def test_adopt_existing_attaches_without_killing(self, display: VirtualDisplay):
        adopter = VirtualDisplay(
            display=display.display, resolution=(320, 240), adopt_existing=True
        )
        adopter.start()
        assert adopter.is_running
        assert adopter.pid is None
        adopter.stop()
        assert is_display_up(display.display), "adopting must not shut the display down"

    def test_xvfb_writes_a_log_when_requested(self, tmp_path: Path, free_display: str):
        log = tmp_path / "xvfb.log"
        with VirtualDisplay(display=free_display, resolution=(320, 240), log_path=log):
            assert log.exists()

    def test_timeout_message_names_the_socket_directory(self, tmp_path: Path, monkeypatch):
        """A readiness timeout must explain the sticky-bit cause, not just time out."""
        import errno as _errno

        from agent_visual_harness import display as display_mod

        # Point the diagnostic at a directory that really is mis-moded, so the
        # assertion does not depend on the host's state.
        fake_dir = tmp_path / ".X11-unix"
        fake_dir.mkdir(mode=0o777)
        monkeypatch.setattr(display_mod, "_X11_SOCKET_DIR", fake_dir)

        vd = VirtualDisplay(display=":150", resolution=(320, 240))
        message = vd._timeout_message(OSError(_errno.ENOENT, "no such file"))
        assert "did not accept connections" in message
        assert str(display_socket_path(":150")) in message
        assert ".X11-unix" in message
        assert "1777" in message


class TestRenderability:
    """The display is only useful if something can actually draw on it."""

    def test_ffmpeg_can_grab_a_frame(self, display: VirtualDisplay):
        from agent_visual_harness._ffmpeg import FFMPEG, run_bytes

        png = run_bytes(
            [
                FFMPEG, "-hide_banner", "-loglevel", "error",
                "-f", "x11grab", "-video_size", "64x64",
                "-i", display.display, "-frames:v", "1",
                "-f", "image2pipe", "-c:v", "png", "-",
            ],
            timeout=30,
        )
        assert png.startswith(b"\x89PNG")

    @requires_xsetroot
    def test_changing_the_root_changes_captured_pixels(self, display: VirtualDisplay):
        from agent_visual_harness._ffmpeg import FFMPEG, run_bytes

        def grab() -> bytes:
            return run_bytes(
                [
                    FFMPEG, "-hide_banner", "-loglevel", "error",
                    "-f", "x11grab", "-video_size", "64x64",
                    "-i", display.display, "-frames:v", "1",
                    "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
                ],
                timeout=30,
            )

        assert _xsetroot(display.display, "#000000")
        black = grab()
        assert _xsetroot(display.display, "#ffffff")
        white = grab()
        assert black != white, "root repaint did not reach the framebuffer"


def _xsetroot(display: str, colour: str) -> bool:
    if not shutil.which("xsetroot"):
        return False
    import subprocess

    return (
        subprocess.run(
            ["xsetroot", "-display", display, "-solid", colour],
            capture_output=True,
            check=False,
        ).returncode
        == 0
    )


def _raise_dependency(name: str, hint: str = "") -> str:
    from agent_visual_harness.errors import DependencyError

    raise DependencyError(f"required binary {name!r} was not found on PATH.")
