"""End-to-end SDK tests: the whole pipeline against a real Xvfb display."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from agent_visual_harness import Artifacts, HarnessError, VisualHarness
from agent_visual_harness._ffmpeg import probe

from .conftest import paint, requires_ffmpeg, requires_linux, requires_xsetroot, requires_xvfb

pytestmark = [requires_linux, requires_xvfb, requires_ffmpeg, pytest.mark.xfvb, pytest.mark.slow]


#: The module-scoped display the SDK tests record onto.
SHARED_DISPLAY = ":121"
#: Numbers reserved for tests that must own their display for the whole run.
OWN_DISPLAY = ":131"
AUTO_EXPORT_DISPLAY = ":132"
REPR_DISPLAY = ":133"


@pytest.fixture(scope="module")
def harness_display():
    """An 800x600 display dedicated to the SDK tests."""
    from agent_visual_harness import VirtualDisplay, is_display_up

    if is_display_up(SHARED_DISPLAY):
        pytest.skip(f"{SHARED_DISPLAY} is already in use")
    vd = VirtualDisplay(display=SHARED_DISPLAY, resolution=(800, 600))
    vd.start()
    yield vd
    vd.stop()


class TestDocumentedExample:
    """The snippet from the landing page, executed verbatim in spirit."""

    def test_four_line_context_manager_produces_artifacts(self, tmp_path: Path, harness_display):
        with VisualHarness(
            display=harness_display.display,
            manage_display=False,
            resolution=(800, 600),
            record=True,
            fps=15,
            session_dir=tmp_path / "session",
        ) as harness:
            harness.log_action("click", x=500, y=300)
            screenshot_bytes = harness.take_screenshot()

        artifacts = harness.export(
            video_path=tmp_path / "repro.mp4",
            gif_path=tmp_path / "demo.gif",
            vlm_keyframes_dir=tmp_path / "vlm_frames",
        )

        assert screenshot_bytes.startswith(b"\x89PNG")
        assert isinstance(artifacts, Artifacts)
        assert Path(artifacts.video_path).exists()
        assert Path(artifacts.gif_path).exists()
        assert artifacts.manifest_path.exists()
        assert len(artifacts.keyframes) >= 1
        assert f"Generated {len(artifacts.keyframes)} frames" in f"Generated {len(artifacts.keyframes)} frames for VLM context."


class TestLifecycle:
    def test_export_works_after_the_block_exits(self, tmp_path: Path, harness_display):
        harness = VisualHarness(
            display=harness_display.display,
            manage_display=False, resolution=(800, 600), fps=15,
            session_dir=tmp_path / "s",
        )
        with harness:
            harness.log_action("click", x=10, y=10)
        assert not harness.recorder.is_recording
        artifacts = harness.export(video_path=tmp_path / "v.mp4")
        assert Path(artifacts.video_path).exists()

    def test_export_works_inside_the_block(self, tmp_path: Path, harness_display):
        with VisualHarness(
            display=harness_display.display,
            manage_display=False, resolution=(800, 600), fps=15,
            session_dir=tmp_path / "s",
        ) as harness:
            harness.log_action("click", x=10, y=10)
            artifacts = harness.export(video_path=tmp_path / "v.mp4")
        assert Path(artifacts.video_path).exists()

    def test_manages_its_own_display_by_default(self, tmp_path: Path):
        harness = VisualHarness(display=OWN_DISPLAY, resolution=(320, 240), fps=10, session_dir=tmp_path / "s")
        try:
            harness.start()
            assert harness.display_manager.is_running
        finally:
            harness.stop()
        assert not harness.display_manager.is_running

    def test_exception_still_finalises_the_recording(self, tmp_path: Path, harness_display):
        output = tmp_path / "session" / "raw.mp4"
        with pytest.raises(RuntimeError, match="agent exploded"):
            with VisualHarness(
                display=harness_display.display,
                manage_display=False, resolution=(800, 600), fps=15,
                session_dir=output.parent,
            ) as harness:
                harness.log_action("click", x=1, y=1)
                time.sleep(0.8)
                raise RuntimeError("agent exploded")
        assert output.exists()
        assert probe(output).duration > 0, "a crashed agent must still leave a playable file"

    def test_telemetry_is_written_before_the_crash(self, tmp_path: Path, harness_display):
        session = tmp_path / "s"
        with pytest.raises(ValueError):
            with VisualHarness(
                display=harness_display.display,
                manage_display=False, resolution=(800, 600), fps=15, session_dir=session
            ) as harness:
                harness.log_action("click", x=1, y=1)
                harness.log_action("type", target="q")
                raise ValueError("boom")
        lines = (session / "telemetry.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2

    def test_record_disabled_produces_no_video(self, tmp_path: Path, harness_display):
        with VisualHarness(
            display=harness_display.display,
            manage_display=False, resolution=(800, 600),
            record=False, session_dir=tmp_path / "s",
        ) as harness:
            assert harness.take_screenshot().startswith(b"\x89PNG")
        assert not (tmp_path / "s" / "raw.mp4").exists()

    def test_double_start_and_stop_are_safe(self, tmp_path: Path, harness_display):
        harness = VisualHarness(
            display=harness_display.display,
            manage_display=False, resolution=(800, 600), fps=15,
            session_dir=tmp_path / "s",
        )
        harness.start()
        harness.start()
        harness.stop()
        harness.stop()


class TestExport:
    @pytest.fixture
    def recorded(self, tmp_path: Path, harness_display):
        """A short session with three logged actions."""
        session = tmp_path / "s"
        with VisualHarness(
            display=harness_display.display,
            manage_display=False, resolution=(800, 600), fps=15,
            session_dir=session,
        ) as harness:
            for i in range(3):
                harness.log_action("click", x=100 + i, y=100)
                time.sleep(0.15)
        return harness

    def test_video_is_playable(self, recorded: VisualHarness, tmp_path: Path):
        artifacts = recorded.export(video_path=tmp_path / "out.mp4")
        info = probe(Path(artifacts.video_path))
        assert info.duration > 0
        assert (info.width, info.height) == (800, 600)

    def test_gif_is_produced(self, recorded: VisualHarness, tmp_path: Path):
        artifacts = recorded.export(gif_path=tmp_path / "out.gif")
        assert Path(artifacts.gif_path).read_bytes()[:6] in (b"GIF87a", b"GIF89a")

    def test_webm_is_produced(self, recorded: VisualHarness, tmp_path: Path):
        artifacts = recorded.export(webm_path=tmp_path / "out.webm")
        assert Path(artifacts.webm_path).exists()

    def test_keyframes_include_every_action(self, recorded: VisualHarness, tmp_path: Path):
        artifacts = recorded.export(vlm_keyframes_dir=tmp_path / "kf")
        assert len(artifacts.keyframes) >= 3
        anchored = [f for f in artifacts.keyframes if f.reason == "action"]
        assert len(anchored) == 3
        for frame in artifacts.keyframes:
            assert frame.path.exists()
            assert frame.width <= 1280

    def test_manifest_is_written_next_to_the_frames(self, recorded: VisualHarness, tmp_path: Path):
        artifacts = recorded.export(vlm_keyframes_dir=tmp_path / "kf")
        manifest = json.loads(Path(artifacts.manifest_path).read_text(encoding="utf-8"))
        assert manifest["keyframes_kept"] == len(artifacts.keyframes)
        assert manifest["action_anchors"] == 3

    def test_nothing_requested_produces_nothing(self, recorded: VisualHarness, tmp_path: Path):
        artifacts = recorded.export()
        assert artifacts.video_path is None
        assert artifacts.gif_path is None
        assert artifacts.keyframes == []
        assert artifacts.raw_path is not None

    def test_all_outputs_at_once(self, recorded: VisualHarness, tmp_path: Path):
        artifacts = recorded.export(
            video_path=tmp_path / "a.mp4",
            gif_path=tmp_path / "a.gif",
            webm_path=tmp_path / "a.webm",
            vlm_keyframes_dir=tmp_path / "kf",
            crf=28,
        )
        assert len(artifacts.paths) == 4
        for path in artifacts.paths:
            assert path.exists() and path.stat().st_size > 0

    def test_overlay_clicks_option(self, recorded: VisualHarness, tmp_path: Path):
        artifacts = recorded.export(
            video_path=tmp_path / "ov.mp4", overlay_clicks=True
        )
        assert probe(Path(artifacts.video_path)).duration > 0

    def test_export_is_idempotent(self, recorded: VisualHarness, tmp_path: Path):
        first = recorded.export(video_path=tmp_path / "1.mp4")
        second = recorded.export(video_path=tmp_path / "2.mp4")
        assert first.session_dir == second.session_dir
        assert Path(second.video_path).exists()

    def test_summary_is_human_readable(self, recorded: VisualHarness, tmp_path: Path):
        summary = recorded.export(
            video_path=tmp_path / "s.mp4", vlm_keyframes_dir=tmp_path / "kf"
        ).summary()
        assert "mp4" in summary and "keyframes" in summary and "MB" in summary or "KB" in summary


class TestAgentHooks:
    def test_screenshot_is_png(self, harness_display, tmp_path: Path):
        with VisualHarness(
            display=harness_display.display,
            manage_display=False, resolution=(800, 600), record=False,
            session_dir=tmp_path / "s",
        ) as harness:
            assert harness.take_screenshot().startswith(b"\x89PNG")

    def test_save_screenshot_picks_the_format_from_the_suffix(self, harness_display, tmp_path: Path):
        with VisualHarness(
            display=harness_display.display,
            manage_display=False, resolution=(800, 600), record=False,
            session_dir=tmp_path / "s",
        ) as harness:
            path = harness.save_screenshot(tmp_path / "shot.jpg")
        assert path.read_bytes()[:2] == b"\xff\xd8"

    def test_log_action_returns_the_record(self, harness_display, tmp_path: Path):
        with VisualHarness(
            display=harness_display.display,
            manage_display=False, resolution=(800, 600), fps=15,
            session_dir=tmp_path / "s",
        ) as harness:
            record = harness.log_action("click", x=42, y=43)
            assert record.action == "click"
            assert (record.x, record.y) == (42, 43)

    def test_actions_are_readable_from_the_harness(self, harness_display, tmp_path: Path):
        with VisualHarness(
            display=harness_display.display,
            manage_display=False, resolution=(800, 600), fps=15,
            session_dir=tmp_path / "s",
        ) as harness:
            harness.log_action("click", x=1, y=1)
            harness.log_action("scroll")
            assert len(harness.actions) == 2
            assert harness.telemetry_path.exists()

    def test_run_on_display_sets_the_env(self, harness_display, tmp_path: Path):
        with VisualHarness(
            display=harness_display.display,
            manage_display=False, resolution=(800, 600), fps=15,
            session_dir=tmp_path / "s",
        ) as harness:
            proc = harness.run_on_display(
                ["python3", "-c", "import os; print(os.environ['DISPLAY'])"]
            )
            assert proc.returncode == 0, proc.stderr
            assert proc.stdout.strip() == harness_display.display

    def test_env_property(self, harness_display, tmp_path: Path):
        harness = VisualHarness(
            display=harness_display.display,
            manage_display=False, resolution=(800, 600),
            session_dir=tmp_path / "s",
        )
        assert harness.env["DISPLAY"] == harness_display.display

    @requires_xsetroot
    def test_scene_changes_become_keyframes(self, harness_display, tmp_path: Path):
        """The end-to-end promise: static frames collapse, real ones survive."""
        with VisualHarness(
            display=harness_display.display,
            manage_display=False, resolution=(800, 600), fps=15,
            session_dir=tmp_path / "s",
        ) as harness:
            for colour in ("#000000", "#ffffff", "#ff0000", "#00ff00", "#0000ff", "#ffff00"):
                paint(harness_display.display, colour)
                time.sleep(0.3)
        artifacts = harness.export(vlm_keyframes_dir=tmp_path / "kf")
        assert len(artifacts.keyframes) >= 3
        assert artifacts.raw_frame_count > artifacts.raw_frame_count - len(artifacts.keyframes) + 1


class TestConfiguration:
    def test_manages_its_own_display_end_to_end(self, tmp_path: Path):
        """Full stack including Xvfb ownership, start to finish."""
        with VisualHarness(
            display=OWN_DISPLAY, resolution=(640, 480), fps=10,
            session_dir=tmp_path / "own",
        ) as harness:
            harness.log_action("click", x=100, y=100)
            assert harness.take_screenshot().startswith(b"\x89PNG")
            time.sleep(0.5)
        artifacts = harness.export(
            video_path=tmp_path / "own.mp4", vlm_keyframes_dir=tmp_path / "kf"
        )
        assert probe(Path(artifacts.video_path)).duration > 0

    def test_adopting_an_existing_display(self, harness_display, tmp_path: Path):
        with VisualHarness(
            display=harness_display.display,
            manage_display=False, resolution=(800, 600), fps=15,
            session_dir=tmp_path / "s",
        ) as harness:
            harness.log_action("click", x=1, y=1)
        assert harness_display.is_running, "adopting must not shut the shared display down"

    def test_busy_display_is_rejected(self, harness_display, tmp_path: Path):
        """With manage_display=True on someone else's X server, refuse loudly."""
        with pytest.raises(HarnessError, match="already serving"):
            with VisualHarness(
                display=harness_display.display,
                resolution=(800, 600),
                session_dir=tmp_path / "s",
            ):
                pass
        assert harness_display.is_running, "the refused display must be untouched"

    def test_meta_records_the_run(self, harness_display, tmp_path: Path):
        session = tmp_path / "s"
        with VisualHarness(
            display=harness_display.display,
            manage_display=False, resolution=(800, 600), fps=15, session_dir=session
        ):
            pass
        meta = json.loads((session / "meta.json").read_text(encoding="utf-8"))
        assert meta["display"] == harness_display.display
        assert meta["resolution"] == [800, 600]
        assert meta["fps"] == 15
        assert "finished_at" in meta

    def test_export_on_exit(self, tmp_path: Path):
        with VisualHarness(
            display=AUTO_EXPORT_DISPLAY, resolution=(320, 240), fps=10,
            session_dir=tmp_path / "s",
            export_on_exit=True,
            auto_export_paths={"video_path": tmp_path / "auto.mp4"},
        ) as harness:
            harness.log_action("click", x=1, y=1)
        assert (tmp_path / "auto.mp4").exists()
        assert harness.artifacts is not None

    def test_repr_is_informative(self, tmp_path: Path):
        harness = VisualHarness(display=REPR_DISPLAY, resolution=(640, 480), session_dir=tmp_path / "s")
        text = repr(harness)
        assert isinstance(text, str)
        assert text.startswith("<"), text
        assert REPR_DISPLAY in text, text
        assert "640x480" in text, text
        assert "idle" in text, text  # never started
        assert str(tmp_path) in text, text
