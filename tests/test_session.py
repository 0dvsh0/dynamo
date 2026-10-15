"""Session directory tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_visual_harness import Session, SessionError

from .conftest import requires_ffmpeg


class TestCreate:
    def test_creates_the_expected_layout(self, tmp_path: Path):
        session = Session.create(tmp_path / "s", display=":42", resolution=(1280, 720), fps=30)
        assert session.root.is_dir()
        assert session.meta_path.exists()
        assert session.display_log.exists()
        assert session.telemetry_path.parent == session.root

    def test_meta_records_the_configuration(self, tmp_path: Path):
        session = Session.create(
            tmp_path / "s", display=":42", resolution=(1280, 720), depth=32, fps=30, crf=18
        )
        meta = json.loads(session.meta_path.read_text(encoding="utf-8"))
        assert meta["display"] == ":42"
        assert meta["resolution"] == [1280, 720]
        assert meta["resolution_x"] == "1280x720"
        assert meta["depth"] == 32
        assert meta["fps"] == 30
        assert meta["capture_crf"] == 18
        assert meta["dynamo_version"]
        assert meta["created_at"].endswith("+00:00")

    def test_typed_accessors(self, tmp_path: Path):
        session = Session.create(tmp_path / "s", display=":7", resolution=(800, 600), fps=12)
        assert session.display == ":7"
        assert session.resolution == (800, 600)
        assert session.fps == 12


class TestLoad:
    def test_missing_directory_is_reported(self, tmp_path: Path):
        with pytest.raises(SessionError, match="does not exist"):
            Session.load(tmp_path / "absent")

    def test_file_instead_of_directory_is_reported(self, tmp_path: Path):
        path = tmp_path / "file"
        path.write_text("x", encoding="utf-8")
        with pytest.raises(SessionError, match="not a directory"):
            Session.load(path)

    def test_corrupt_meta_is_reported(self, tmp_path: Path):
        root = tmp_path / "s"
        root.mkdir()
        (root / "meta.json").write_text("{not json", encoding="utf-8")
        with pytest.raises(SessionError, match="not valid JSON"):
            Session.load(root)

    def test_load_without_meta_still_works(self, tmp_path: Path):
        root = tmp_path / "s"
        root.mkdir()
        session = Session.load(root)
        assert session.meta == {}
        assert session.fps == 24  # documented default

    def test_round_trip(self, tmp_path: Path):
        original = Session.create(tmp_path / "s", display=":11", resolution=(640, 480), fps=15)
        original.update_meta(crashed=False)
        loaded = Session.load(tmp_path / "s")
        assert loaded.meta["display"] == ":11"
        assert loaded.meta["crashed"] is False


class TestRecording:
    def test_missing_recording_is_reported(self, tmp_path: Path):
        session = Session.create(tmp_path / "s")
        assert not session.has_recording()
        with pytest.raises(SessionError, match="no recording found"):
            session.require_recording()

    def test_empty_recording_counts_as_missing(self, tmp_path: Path):
        session = Session.create(tmp_path / "s")
        session.raw_path.write_bytes(b"")
        assert not session.has_recording()

    @requires_ffmpeg
    def test_present_recording_is_found(self, tmp_path: Path, synthetic_clip: Path):
        session = Session.create(tmp_path / "s")
        session.raw_path.write_bytes(synthetic_clip.read_bytes())
        assert session.has_recording()
        assert session.require_recording() == session.raw_path
        assert session.media_info().width == 640


class TestTelemetry:
    def test_actions_are_read_back(self, tmp_path: Path):
        session = Session.create(tmp_path / "s")
        session.telemetry_path.write_text(
            '{"timestamp": 1.5, "action": "click", "x": 1, "y": 2}\n'
            '{"timestamp": 0.5, "action": "scroll"}\n',
            encoding="utf-8",
        )
        actions = session.actions
        assert [a.action for a in actions] == ["scroll", "click"]
        assert actions[1].x == 1

    def test_no_telemetry_file_yields_no_actions(self, tmp_path: Path):
        assert Session.create(tmp_path / "s").actions == []


class TestFinishing:
    def test_mark_finished_is_timestamped(self, tmp_path: Path):
        session = Session.create(tmp_path / "s")
        session.mark_finished(duration=4.25, recording_ok=True)
        meta = session.meta
        assert meta["wall_seconds"] == 4.25
        assert meta["recording_ok"] is True
        assert meta["finished_at"].endswith("+00:00")

    def test_fspath_protocol(self, tmp_path: Path):
        session = Session.create(tmp_path / "s")
        assert Path(session) == session.root
