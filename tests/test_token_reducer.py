"""TokenReducer tests — the "don't pay for the same frame twice" module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_visual_harness import Action, ProcessingError, ReducerConfig, TokenReducer
from agent_visual_harness.token_reducer import _jpeg_dimensions, _webp_dimensions

from .conftest import CUT_PALETTE, make_clip, requires_ffmpeg

pytestmark = [requires_ffmpeg, pytest.mark.slow]


def _reduce(clip: Path, out: Path, actions=(), **config) -> tuple[list, Path]:
    reducer = TokenReducer(config=ReducerConfig(**config))
    frames = reducer.reduce(clip, out, actions=list(actions), source_size=(640, 480))
    return frames, out / "manifest.json"


class TestConfigValidation:
    def test_rejects_unknown_format(self):
        with pytest.raises(ProcessingError, match="unknown keyframe format"):
            ReducerConfig(fmt="tiff").normalised()

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.5, 2.0])
    def test_rejects_out_of_range_threshold(self, bad):
        with pytest.raises(ProcessingError, match="scene threshold"):
            ReducerConfig(scene_threshold=bad).normalised()

    def test_rejects_absurd_size(self):
        with pytest.raises(ProcessingError, match="max_size"):
            ReducerConfig(max_size=8).normalised()

    def test_webp_falls_back_to_jpeg_without_the_encoder(self, monkeypatch):
        from agent_visual_harness import token_reducer as tr

        monkeypatch.setattr(tr, "has_encoder", lambda name: False)
        assert ReducerConfig(fmt="webp").normalised().fmt == "jpeg"


class TestSceneDetection:
    def test_two_cuts_in_a_three_colour_clip(self, synthetic_clip: Path):
        times = TokenReducer().detect_scenes(synthetic_clip)
        assert len(times) == 2
        assert all(t1 < t2 for t1, t2 in zip(times, times[1:], strict=False))

    def test_threshold_controls_sensitivity(self, tmp_path: Path):
        clip = make_clip(tmp_path / "c.mp4", CUT_PALETTE[:5])
        assert len(TokenReducer().detect_scenes(clip)) == 4
        # A near-maximal threshold can only catch the strongest cuts.
        strict = TokenReducer(config=ReducerConfig(scene_threshold=0.99)).detect_scenes(clip)
        assert 0 < len(strict) < 4
        assert set(strict) <= set(TokenReducer().detect_scenes(clip))

    def test_scene_detection_can_be_switched_off(self, tmp_path: Path):
        clip = make_clip(tmp_path / "c.mp4", CUT_PALETTE[:4])
        frames, _ = _reduce(
            clip, tmp_path / "kf", fmt="jpeg", extract_scene_frames=False
        )
        assert frames == []

    def test_missing_video_is_reported(self, tmp_path: Path):
        with pytest.raises(ProcessingError, match="not found"):
            TokenReducer().detect_scenes(tmp_path / "nope.mp4")


class TestKeyframeExtraction:
    def test_scene_only_run_keeps_the_cuts(self, synthetic_clip: Path, tmp_path: Path):
        frames, manifest = _reduce(
            synthetic_clip, tmp_path / "kf", include_actions=False, fmt="jpeg"
        )
        assert len(frames) == 2
        assert all(f.reason == "scene" for f in frames)
        assert manifest.exists()

    def test_action_anchors_are_added_and_labelled(self, synthetic_clip: Path, tmp_path: Path):
        actions = [
            Action(action="click", timestamp=0.5, x=10, y=10),
            Action(action="type", timestamp=1.5, target="search"),
            Action(action="scroll", timestamp=2.5),
        ]
        frames, _ = _reduce(synthetic_clip, tmp_path / "kf", actions, fmt="jpeg")
        anchored = [f for f in frames if f.reason == "action"]
        assert len(anchored) == 3
        assert {f.action for f in anchored} == {"click", "type", "scroll"}
        for frame in anchored:
            assert frame.path.exists() and frame.path.stat().st_size > 0

    def test_frames_are_chronological_and_indexed_from_one(self, synthetic_clip: Path, tmp_path: Path):
        actions = [Action(action="click", timestamp=t) for t in (0.2, 1.2, 2.2)]
        frames, _ = _reduce(synthetic_clip, tmp_path / "kf", actions, fmt="jpeg")
        stamps = [f.timestamp for f in frames]
        assert stamps == sorted(stamps)
        assert [f.index for f in frames] == list(range(1, len(frames) + 1))

    def test_filenames_are_sequential(self, synthetic_clip: Path, tmp_path: Path):
        frames, _ = _reduce(synthetic_clip, tmp_path / "kf", fmt="jpeg")
        names = sorted(p.name for p in (tmp_path / "kf").glob("frame_*.jpg"))
        assert names == [f"frame_{i:04d}.jpg" for i in range(1, len(frames) + 1)]

    def test_scratch_files_are_removed(self, synthetic_clip: Path, tmp_path: Path):
        out = tmp_path / "kf"
        _reduce(synthetic_clip, out, [Action(action="click", timestamp=0.5)], fmt="jpeg")
        assert list(out.glob("_*")) == [], "temporary frames leaked into the output dir"

    def test_rerun_cleans_stale_frames(self, synthetic_clip: Path, tmp_path: Path):
        out = tmp_path / "kf"
        _reduce(synthetic_clip, out, fmt="jpeg")  # 2 frames
        frames, _ = _reduce(
            synthetic_clip, out, [Action(action="click", timestamp=0.4)], fmt="jpeg"
        )
        on_disk = sorted((out).glob("frame_*.jpg"))
        assert len(on_disk) == len(frames)
        assert [p.name for p in on_disk] == [f"frame_{i:04d}.jpg" for i in range(1, len(frames) + 1)]

    @pytest.mark.parametrize("fmt", ["jpeg", "webp", "png"])
    def test_each_output_format_works(self, synthetic_clip: Path, tmp_path: Path, fmt):
        frames, manifest = _reduce(synthetic_clip, tmp_path / fmt, fmt=fmt)
        assert len(frames) == 2
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        assert payload["config"]["format"] == fmt
        for frame in frames:
            assert frame.path.exists() and frame.path.stat().st_size > 0

    def test_max_size_caps_the_width(self, synthetic_clip: Path, tmp_path: Path):
        frames, _ = _reduce(synthetic_clip, tmp_path / "kf", max_size=320, fmt="jpeg")
        assert frames
        assert all(f.width == 320 for f in frames), [f.width for f in frames]
        assert all(f.height == 240 for f in frames)

    def test_never_upscales_a_small_capture(self, tmp_path: Path):
        small = make_clip(tmp_path / "small.mp4", size="320x240")
        frames, _ = _reduce(small, tmp_path / "kf", max_size=1280, fmt="jpeg")
        assert all(f.width == 320 for f in frames)

    def test_max_frames_budget_is_respected(self, tmp_path: Path):
        clip = make_clip(tmp_path / "c.mp4", CUT_PALETTE[:5])
        frames, _ = _reduce(clip, tmp_path / "kf", max_frames=2, fmt="jpeg")
        assert len(frames) == 2

    def test_min_gap_collapses_nearby_frames(self, synthetic_clip: Path, tmp_path: Path):
        tight, _ = _reduce(synthetic_clip, tmp_path / "tight", min_gap=0.01, fmt="jpeg")
        loose, _ = _reduce(synthetic_clip, tmp_path / "loose", min_gap=5.0, fmt="jpeg")
        assert len(tight) >= len(loose)
        assert len(loose) == 1

    def test_anchor_at_the_scene_boundary_wins(self, synthetic_clip: Path, tmp_path: Path):
        actions = [Action(action="click", timestamp=1.0, x=5, y=5)]
        frames, _ = _reduce(synthetic_clip, tmp_path / "kf", actions, min_gap=0.5, fmt="jpeg")
        at_one_second = [f for f in frames if abs(f.timestamp - 1.0) < 0.3]
        assert len(at_one_second) == 1
        assert at_one_second[0].reason == "action"


class TestManifest:
    def test_manifest_describes_the_run(self, synthetic_clip: Path, tmp_path: Path):
        actions = [Action(action="click", timestamp=0.5, x=1, y=1)]
        _, manifest = _reduce(synthetic_clip, tmp_path / "kf", actions, fmt="jpeg")
        payload = json.loads(manifest.read_text(encoding="utf-8"))

        assert payload["version"] == 1
        assert payload["generator"] == "agent_visual_harness"
        assert payload["source_size"] == {"width": 640, "height": 480}
        assert payload["duration"] > 2.0
        assert payload["scene_frames_detected"] == 2
        assert payload["action_anchors"] == 1
        assert payload["keyframes_kept"] == len(payload["keyframes"])
        assert payload["config"]["scene_threshold"] == 0.05
        assert 90 < payload["reduction_pct"] < 100
        assert len(payload["keyframes"]) == payload["keyframes_kept"]

    def test_every_manifest_entry_matches_a_file_on_disk(self, synthetic_clip: Path, tmp_path: Path):
        out = tmp_path / "kf"
        _, manifest = _reduce(synthetic_clip, out, fmt="jpeg")
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        for entry in payload["keyframes"]:
            path = out / entry["path"]
            assert path.exists(), entry["path"]
            assert path.stat().st_size > 0
            assert set(entry) >= {"index", "path", "timestamp", "reason", "action", "width", "height"}

    def test_manifest_is_valid_json_with_a_trailing_newline(self, synthetic_clip: Path, tmp_path: Path):
        _, manifest = _reduce(synthetic_clip, tmp_path / "kf", fmt="jpeg")
        text = manifest.read_text(encoding="utf-8")
        assert text.endswith("\n")
        json.loads(text)


class TestDimensionParsers:
    def test_jpeg_header(self, synthetic_clip: Path, tmp_path: Path):
        from agent_visual_harness import Processor

        out = Processor().frame_at(synthetic_clip, 0.5, tmp_path / "f.jpg")
        data = out.read_bytes()[:4096]
        assert _jpeg_dimensions(data) == (640, 480)

    def test_png_header(self, synthetic_clip: Path, tmp_path: Path):
        from agent_visual_harness import Processor

        out = Processor().frame_at(synthetic_clip, 0.5, tmp_path / "f.png")
        data = out.read_bytes()[:64]
        assert _webp_dimensions(data) is None  # sanity: wrong parser
        assert int.from_bytes(data[16:20], "big") == 640

    def test_webp_header_when_available(self, synthetic_clip: Path, tmp_path: Path):
        from agent_visual_harness._ffmpeg import has_encoder

        if not has_encoder("libwebp"):
            pytest.skip("libwebp not compiled in")
        frames, _ = _reduce(synthetic_clip, tmp_path / "kf", fmt="webp")
        for frame in frames:
            assert _webp_dimensions(frame.path.read_bytes()[:64]) == (frame.width, frame.height)
