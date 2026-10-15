"""Pure-logic tests: no ffmpeg, no X server. These must pass everywhere."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_visual_harness import Action, Keyframe, parse_resolution
from agent_visual_harness.token_reducer import TokenReducer, _pair
from agent_visual_harness.processor import estimate_token_saving, parse_showinfo_times
from agent_visual_harness.types import Artifacts


class TestParseResolution:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("1920x1080", (1920, 1080)),
            ("1920x1080x24", (1920, 1080)),
            ("1280X720", (1280, 720)),
            ("800*600", (800, 600)),
            ((1024, 768), (1024, 768)),
            ((640, 480, 32), (640, 480)),
        ],
    )
    def test_accepts_valid_geometry(self, value, expected):
        assert parse_resolution(value) == expected

    @pytest.mark.parametrize(
        "value", ["", "abc", "1920", "1920x1080x99", "0x100", (-1, 10), (10,), "1920x1080x24x1"]
    )
    def test_rejects_invalid_geometry(self, value):
        with pytest.raises(ValueError):
            parse_resolution(value)


class TestAction:
    def test_round_trip(self):
        original = Action(action="click", timestamp=1.45, x=420, y=680, target="Deploy")
        restored = Action.from_dict(json.loads(json.dumps(original.to_dict())))
        assert restored == original

    def test_omits_empty_coordinates(self):
        payload = Action(action="scroll", timestamp=2.0).to_dict()
        assert payload == {"timestamp": 2.0, "action": "scroll"}
        assert "x" not in payload and "y" not in payload

    def test_telemetry_format_matches_documented_shape(self):
        payload = Action(action="click", timestamp=1.45, x=420, y=680).to_dict()
        assert payload == {"timestamp": 1.45, "action": "click", "x": 420, "y": 680}
        assert json.loads(json.dumps(payload)) == payload

    def test_has_position(self):
        assert Action(action="click", x=1, y=2).has_position
        assert not Action(action="scroll").has_position
        assert not Action(action="click", x=1).has_position

    def test_extra_data_is_carried(self):
        payload = Action(action="type", data={"text": "hello"}).to_dict()
        assert payload["data"] == {"text": "hello"}


class TestArtifacts:
    def test_keyframes_are_iterable_and_sized(self, tmp_path: Path):
        frames = [Keyframe(path=tmp_path / f"f{i}.webp", index=i, timestamp=float(i), reason="scene") for i in range(3)]
        artifacts = Artifacts(session_dir=tmp_path, telemetry_path=tmp_path / "t.jsonl", keyframes=frames)
        assert len(artifacts) == 3
        assert [k.index for k in artifacts] == [0, 1, 2]
        assert "3 frames" in artifacts.summary()

    def test_serialises_to_json(self, tmp_path: Path):
        artifacts = Artifacts(
            session_dir=tmp_path,
            telemetry_path=tmp_path / "t.jsonl",
            keyframes=[Keyframe(path=tmp_path / "f.webp", index=1, timestamp=0.5, reason="action", action="click")],
        )
        payload = json.loads(json.dumps(artifacts.to_dict()))
        assert payload["keyframes"][0]["action"] == "click"


class TestShowinfoParsing:
    def test_extracts_pts_times_in_order(self):
        stderr = "\n".join(
            "[Parsed_showinfo_1 @ 0x1] n:  0 pts:  1024 pts_time:1 pos: 2048"
            for _ in range(1)
        ) + "\n" + "\n".join(
            f"[Parsed_showinfo_1 @ 0x1] n: {i} pts: {i*512} pts_time:{1.5 + i} pos: {2048*(i+1)}"
            for i in range(3)
        )
        assert parse_showinfo_times(stderr) == [1.0, 1.5, 2.5, 3.5]

    def test_ignores_na_and_empty(self):
        assert parse_showinfo_times("pts_time:N/A\nnothing here\n") == []
        assert parse_showinfo_times(None) == []
        assert parse_showinfo_times("") == []


class TestPairing:
    def test_pairs_when_counts_match(self):
        files = [Path("a.png"), Path("b.png")]
        assert _pair(files, [0.0, 2.0]) == [(files[0], 0.0), (files[1], 2.0)]

    def test_falls_back_to_uniform_spacing_on_mismatch(self):
        files = [Path("a.png"), Path("b.png"), Path("c.png")]
        paired = _pair(files, [0.0, 4.0])
        assert [t for _, t in paired] == [0.0, 2.0, 4.0]

    def test_empty(self):
        assert _pair([], []) == []


class TestDedupe:
    @staticmethod
    def _rows(spec: list[tuple[float, str]]):
        from pathlib import Path

        return [(ts, reason, None, Path(f"{reason}-{i}.png")) for i, (ts, reason) in enumerate(spec)]

    def test_keeps_scene_frames_beyond_the_gap(self):
        rows = self._rows([(0.0, "scene"), (1.0, "scene"), (2.0, "scene")])
        assert len(TokenReducer._dedupe(rows, 0.3, None)) == 3

    def test_collapses_scene_frames_inside_the_gap(self):
        rows = self._rows([(0.0, "scene"), (0.1, "scene"), (0.2, "scene"), (2.0, "scene")])
        assert len(TokenReducer._dedupe(rows, 0.3, None)) == 2

    def test_a_scene_frame_near_an_anchor_is_dropped(self):
        rows = self._rows([(0.0, "scene"), (0.1, "action"), (0.2, "scene"), (5.0, "scene")])
        kept = TokenReducer._dedupe(rows, 0.3, None)
        assert len(kept) == 2
        assert {row[1] for row in kept} == {"action", "scene"}

    def test_action_anchors_are_never_dropped(self):
        """The documented contract: log_action() instants always survive."""
        rows = self._rows([(0.0, "action"), (0.05, "action"), (0.10, "action")])
        assert len(TokenReducer._dedupe(rows, 5.0, None)) == 3

    def test_anchors_survive_alongside_scene_frames(self):
        rows = self._rows([(0.0, "scene"), (5.0, "action"), (10.0, "scene")])
        kept = TokenReducer._dedupe(rows, 0.3, None)
        assert [row[1] for row in kept] == ["scene", "action", "scene"]
        assert [row[0] for row in kept] == [0.0, 5.0, 10.0]

    def test_scene_frames_surrounding_an_anchor_are_dropped(self):
        rows = self._rows([(0.0, "scene"), (0.1, "action"), (0.2, "scene"), (5.0, "scene")])
        kept = TokenReducer._dedupe(rows, 0.3, None)
        assert [row[1] for row in kept] == ["action", "scene"]
        assert [row[0] for row in kept] == [0.1, 5.0]

    def test_respects_max_frames(self):
        rows = self._rows([(float(i), "scene") for i in range(20)])
        assert len(TokenReducer._dedupe(rows, 0.0, 5)) == 5

    def test_dedup_is_chronological(self):
        rows = self._rows([(2.0, "scene"), (0.0, "scene"), (1.0, "scene")])
        kept = TokenReducer._dedupe(rows, 0.1, None)
        assert [row[0] for row in kept] == [0.0, 1.0, 2.0]


class TestScaleFilter:
    def test_downscales_only_when_too_wide(self):
        reducer = TokenReducer(config=None)
        assert reducer.scale_filter((1920, 1080)) == "scale=1280:-2:flags=lanczos"
        assert reducer.scale_filter((640, 480)) == "null"

    def test_uses_a_concrete_width_not_an_expression(self):
        """Parentheses in a filtergraph expression are an escaping trap."""
        reducer = TokenReducer(config=None)
        for size in ((1920, 1080), (3840, 2160), (640, 480), (0, 0)):
            out = reducer.scale_filter(size)
            assert "(" not in out and "min" not in out, out


class TestTokenSavingsEstimate:
    def test_matches_the_published_calculator_defaults(self):
        # Mirrors the JavaScript calculator exactly:
        #   raw   = 120s x 30fps                = 3600 frames
        #   scene = 14 keyframes/min x 2 min    = 28
        #   kept  = round((18 actions + 28) x 0.8 + 1) = 38
        result = estimate_token_saving(120, 30, 18, 14, tokens_per_image=1100)
        assert result["raw_frames"] == 3600
        assert result["raw_tokens"] == pytest.approx(3_960_000)
        assert result["kept_frames"] == 38
        assert result["kept_tokens"] == pytest.approx(41_800)
        assert 98 < result["reduction_pct"] < 99.5

    def test_kept_never_exceeds_raw(self):
        result = estimate_token_saving(1, 60, 500, 200, tokens_per_image=1000)
        assert result["kept_frames"] <= result["raw_frames"]
        assert result["reduction_pct"] >= 0

    def test_zero_duration_is_safe(self):
        assert estimate_token_saving(0, 30, 0, 0)["reduction_pct"] == 0.0


class TestPackageSurface:
    """The public surface, checked against itself.

    ``__all__`` is the promise the README makes to anyone typing an import. A
    name listed there that does not resolve is only ever found by the user who
    tried it, which is the worst possible time to discover it — so it is checked
    here instead.
    """

    def test_every_exported_name_resolves(self):
        import agent_visual_harness as package

        assert [n for n in package.__all__ if not hasattr(package, n)] == []

    def test_the_submodule_contract_holds_too(self):
        from agent_visual_harness import vision

        assert [n for n in vision.__all__ if not hasattr(vision, n)] == []

    def test_no_name_is_exported_twice(self):
        import agent_visual_harness as package

        assert len(package.__all__) == len(set(package.__all__))

    def test_the_model_layer_is_reachable_from_the_top_level(self):
        # The triage names are the ones most likely to be missed, because they
        # were added to a module that already had an export list.
        import agent_visual_harness as package

        for name in ("Triage", "OllamaBackend", "LayaBackend", "RemoteBackend",
                     "Question", "Decision", "Frame", "frames_from_manifest",
                     "estimate_tokens", "DEFAULT_VLM_MODEL"):
            assert name in package.__all__, f"{name} is not exported"
            assert hasattr(package, name), f"{name} is exported but does not resolve"
