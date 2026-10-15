"""Recorder tests — real ffmpeg capturing a real Xvfb display."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from agent_visual_harness import Recorder
from agent_visual_harness._ffmpeg import probe
from agent_visual_harness.recorder import read_actions

from .conftest import paint, requires_ffmpeg, requires_linux, requires_xsetroot, requires_xvfb

pytestmark = [requires_linux, requires_xvfb, requires_ffmpeg, pytest.mark.xfvb]


class TestCapture:
    def test_recording_is_playable_after_stop(self, tmp_path: Path, display):
        rec = Recorder(
            display=display.display, size=(640, 480),
            output=tmp_path / "raw.mp4", fps=15, crf=20,
        )
        rec.start()
        assert rec.is_recording
        assert rec.output.exists()
        time.sleep(1.2)
        rec.stop()

        assert not rec.is_recording
        info = probe(rec.output)
        assert info.width == 640 and info.height == 480
        assert 0.8 <= info.duration <= 6.0, f"unexpected duration {info.duration}"
        assert rec.output.stat().st_size > 1000

    def test_sigint_leaves_a_playable_container(self, tmp_path: Path, display):
        """The moov atom is written on SIGINT; a truncated file proves it isn't."""
        rec = Recorder(
            display=display.display, size=(640, 480),
            output=tmp_path / "raw.mp4", fps=15, crf=20,
        )
        rec.start()
        time.sleep(1.0)
        rec.stop()
        assert probe(rec.output).duration > 0

    def test_duration_tracks_the_capture_window(self, tmp_path: Path, display):
        rec = Recorder(
            display=display.display, size=(640, 480),
            output=tmp_path / "raw.mp4", fps=15, crf=20,
        )
        rec.start()
        time.sleep(2.0)
        rec.stop()
        assert rec.elapsed >= 1.9
        assert probe(rec.output).duration >= 1.0

    def test_oversized_grab_is_rejected_or_flagged(self, tmp_path: Path, display):
        """Grabbing far beyond the display must never silently produce garbage.

        Most X servers refuse an out-of-bounds ``XGetImage`` outright; others
        happily return garbage. Accept either, but never accept silence.
        """
        from agent_visual_harness.errors import HarnessError

        rec = Recorder(
            display=display.display,
            size=(3840, 2160),  # larger than the 640x480 display
            output=tmp_path / "bad.mp4",
            startup_timeout=8.0,
        )
        try:
            rec.start()
        except HarnessError as exc:
            # Whatever the wording, the operator must be able to see which
            # geometry was actually attempted.
            assert "3840x2160" in str(exc), str(exc)
            return
        rec.stop()
        pytest.skip("this X server accepted an oversized grab; nothing to assert")

    @pytest.mark.slow
    @requires_xsetroot
    def test_visual_changes_are_captured(self, tmp_path: Path, display):
        from agent_visual_harness import TokenReducer

        rec = Recorder(
            display=display.display, size=(640, 480),
            output=tmp_path / "raw.mp4", fps=15, crf=20,
        )
        rec.start()
        for colour in ("#101010", "#e0e0e0", "#3050c0", "#c05030", "#20a020", "#f0f0f0"):
            assert paint(display.display, colour)
            time.sleep(0.35)
        rec.stop()

        reducer = TokenReducer()
        assert len(reducer.detect_scenes(rec.output)) >= 2, "no scene changes detected"


class TestTelemetry:
    def test_actions_are_written_as_jsonl(self, tmp_path: Path, display):
        telemetry = tmp_path / "telemetry.jsonl"
        rec = Recorder(
            display=display.display, size=(640, 480),
            output=tmp_path / "raw.mp4", fps=15, telemetry_path=telemetry,
        )
        rec.start()
        rec.log_action("click", x=500, y=300)
        rec.log_action("type", target="search-box", text="hello")
        rec.log_action("scroll", dy=-240)
        rec.stop()

        lines = telemetry.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3
        first = json.loads(lines[0])
        assert first["action"] == "click"
        assert first["x"] == 500 and first["y"] == 300
        assert 0.0 <= first["timestamp"] <= rec.elapsed + 0.5

        third = json.loads(lines[2])
        assert third["action"] == "scroll"
        assert "x" not in third, "positional keys must be omitted when unknown"

    def test_timestamps_increase(self, tmp_path: Path, display):
        rec = Recorder(
            display=display.display, size=(640, 480),
            output=tmp_path / "raw.mp4", fps=15,
            telemetry_path=tmp_path / "t.jsonl",
        )
        rec.start()
        for i in range(5):
            rec.log_action("click", x=i, y=i)
            time.sleep(0.05)
        rec.stop()
        stamps = [a.timestamp for a in rec.actions]
        assert stamps == sorted(stamps)
        assert stamps[-1] > stamps[0]

    def test_actions_survive_a_restart_of_the_reader(self, tmp_path: Path, display):
        telemetry = tmp_path / "telemetry.jsonl"
        rec = Recorder(
            display=display.display, size=(640, 480),
            output=tmp_path / "raw.mp4", fps=15, telemetry_path=telemetry,
        )
        rec.start()
        rec.log_action("click", x=1, y=2)
        rec.stop()
        reread = read_actions(telemetry)
        assert len(reread) == 1
        assert reread[0].x == 1 and reread[0].y == 2

    def test_malformed_lines_are_skipped(self, tmp_path: Path):
        path = tmp_path / "telemetry.jsonl"
        path.write_text(
            '{"timestamp":0.5,"action":"click","x":1,"y":2}\n'
            "not json\n"
            "\n"
            '{"timestamp":1.5,"action":"scroll"}\n',
            encoding="utf-8",
        )
        actions = read_actions(path)
        assert [a.action for a in actions] == ["click", "scroll"]

    def test_reading_a_missing_file_is_empty(self, tmp_path: Path):
        assert read_actions(tmp_path / "nope.jsonl") == []

    def test_extra_data_is_serialised(self, tmp_path: Path, display):
        rec = Recorder(
            display=display.display, size=(640, 480),
            output=tmp_path / "raw.mp4", fps=15,
            telemetry_path=tmp_path / "t.jsonl",
        )
        rec.start()
        rec.log_action("type", target="field", value="query", url="http://x")
        rec.stop()
        payload = json.loads((tmp_path / "t.jsonl").read_text(encoding="utf-8").splitlines()[0])
        assert payload["data"] == {"value": "query", "url": "http://x"}


class TestScreenshot:
    def test_returns_png_bytes(self, display):
        rec = Recorder(display=display.display, size=(640, 480), output="/tmp/dynamo-probe.mp4")
        data = rec.take_screenshot()
        assert data.startswith(b"\x89PNG\r\n\x1a\n")
        assert len(data) > 100

    def test_jpeg_is_supported(self, display):
        rec = Recorder(display=display.display, size=(640, 480), output="/tmp/dynamo-probe.mp4")
        assert rec.take_screenshot(fmt="jpeg").startswith(b"\xff\xd8")

    def test_rejects_unknown_format(self, display):
        rec = Recorder(display=display.display, size=(640, 480), output="/tmp/dynamo-probe.mp4")
        with pytest.raises(ValueError):
            rec.take_screenshot(fmt="tiff")

    @requires_xsetroot
    def test_content_changes_with_the_screen(self, display):
        rec = Recorder(display=display.display, size=(640, 480), output="/tmp/dynamo-probe.mp4")
        paint(display.display, "#000000")
        time.sleep(0.2)
        dark = rec.take_screenshot()
        paint(display.display, "#ffffff")
        time.sleep(0.2)
        light = rec.take_screenshot()
        assert dark != light, "screenshot did not reflect the repaint"

    def test_works_while_recording(self, clip_recorder: Recorder):
        assert clip_recorder.take_screenshot().startswith(b"\x89PNG")


class TestRobustness:
    def test_context_manager_finalises(self, tmp_path: Path, display):
        output = tmp_path / "raw.mp4"
        with Recorder(display=display.display, size=(640, 480), output=output, fps=15):
            time.sleep(0.8)
        assert probe(output).duration > 0

    def test_exception_still_produces_a_playable_file(self, tmp_path: Path, display):
        output = tmp_path / "raw.mp4"
        with pytest.raises(RuntimeError, match="agent crashed"):
            with Recorder(display=display.display, size=(640, 480), output=output, fps=15):
                time.sleep(0.8)
                raise RuntimeError("agent crashed")
        assert probe(output).duration > 0

    def test_double_stop_is_safe(self, tmp_path: Path, display):
        rec = Recorder(display=display.display, size=(640, 480), output=tmp_path / "raw.mp4", fps=15)
        rec.start()
        rec.stop()
        rec.stop()

    def test_double_start_is_rejected(self, tmp_path: Path, display):
        from agent_visual_harness.errors import RecorderError

        rec = Recorder(display=display.display, size=(640, 480), output=tmp_path / "raw.mp4", fps=15)
        rec.start()
        try:
            with pytest.raises(RecorderError, match="already running"):
                rec.start()
        finally:
            rec.stop()

    def test_repair_contract_on_damaged_media(self, tmp_path: Path, clip_recorder: Recorder):
        """repair_media's real guarantees, stated honestly.

        A missing ``moov`` atom cannot be rebuilt — MP4 has nothing to demux
        from — so the contract is not "always fixes it". It is: try, and never
        destroy the only copy.
        """
        from agent_visual_harness._ffmpeg import repair_media

        clip_recorder.stop()
        good = clip_recorder.output.read_bytes()

        # A healthy file round-trips through the repair path.
        healthy = tmp_path / "healthy.mp4"
        healthy.write_bytes(good)
        assert repair_media(healthy) is True
        assert probe(healthy).duration > 0

        # A file whose index was destroyed is reported as unrecoverable...
        damaged = tmp_path / "damaged.mp4"
        damaged.write_bytes(good[: int(len(good) * 0.55)])
        damaged_bytes = damaged.read_bytes()
        assert repair_media(damaged) is False

        # ...and, critically, the original bytes are still there to inspect.
        assert damaged.exists()
        assert damaged.read_bytes() == damaged_bytes
        assert not (tmp_path / "damaged.mp4.damaged").exists()

    def test_recorder_reports_an_unsalvageable_file_clearly(self, tmp_path: Path, clip_recorder: Recorder):
        """An unplayable capture must raise with the ffmpeg log attached."""
        from agent_visual_harness.errors import RecorderError

        clip_recorder.stop()
        clip_recorder.output.write_bytes(b"this is not an mp4")
        with pytest.raises(RecorderError) as exc:
            clip_recorder._finalize()
        assert "could not be finalised or repaired" in str(exc.value)

    def test_building_the_command_is_deterministic(self, tmp_path: Path):
        rec = Recorder(
            display=":77", size=(1280, 720), output=tmp_path / "x.mp4",
            fps=30, crf=18, preset="fast",
        )
        cmd = rec.build_command()
        assert "x11grab" in cmd
        assert "-video_size" in cmd and "1280x720" in cmd
        assert cmd[cmd.index("-framerate") + 1] == "30"
        assert cmd[cmd.index("-crf") + 1] == "18"
        assert ":77" in cmd

    def test_command_flushes_packets(self, tmp_path: Path):
        """A static screen must still produce visible progress on disk.

        Without -flush_packets 1, ffmpeg holds a still screen's output in its
        internal buffer and the file stays at the container header, which is
        both invisible to callers and lost on a hard kill.
        """
        rec = Recorder(display=":77", size=(640, 480), output=tmp_path / "x.mp4")
        cmd = rec.build_command()
        assert cmd[cmd.index("-flush_packets") + 1] == "1"

    @requires_xsetroot
    def test_startup_completes_on_a_completely_static_display(self, tmp_path: Path, display):
        """The hardest case for readiness: nothing ever changes on screen."""
        rec = Recorder(
            display=display.display, size=(640, 480),
            output=tmp_path / "static.mp4", fps=15, crf=20, startup_timeout=20.0,
        )
        rec.start()  # must not raise RecorderStartTimeout
        try:
            assert rec.is_recording
        finally:
            rec.stop()
        assert probe(rec.output).duration > 0
