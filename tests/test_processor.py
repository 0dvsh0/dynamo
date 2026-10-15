"""Processor tests — all driven by a deterministic synthetic clip."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_visual_harness import Action, ProcessingError, Processor
from agent_visual_harness._ffmpeg import has_encoder, probe

from .conftest import CUT_PALETTE, make_clip, make_noisy_clip, requires_ffmpeg

pytestmark = [requires_ffmpeg, pytest.mark.slow]


@pytest.fixture
def processor() -> Processor:
    return Processor(crf=23, preset="ultrafast")


class TestVideo:
    def test_mp4_is_playable_and_right_sized(self, synthetic_clip: Path, tmp_path: Path, processor):
        out = processor.to_video(synthetic_clip, tmp_path / "out.mp4")
        info = probe(out)
        assert out.exists() and out.stat().st_size > 0
        assert (info.width, info.height) == (640, 480)
        assert info.codec == "h264"
        assert info.duration > 2.0

    def test_mp4_carries_the_faststart_flag(self, synthetic_clip: Path, tmp_path: Path, processor):
        out = processor.to_video(synthetic_clip, tmp_path / "out.mp4")
        head = out.read_bytes()[:4096]
        assert b"moov" in head, "moov atom should precede mdat for faststart"

    def test_webm_uses_vp9(self, synthetic_clip: Path, tmp_path: Path, processor):
        if not has_encoder("libvpx-vp9"):
            pytest.skip("libvpx-vp9 not compiled into this ffmpeg")
        out = processor.to_video(synthetic_clip, tmp_path / "out.webm")
        assert probe(out).codec in ("vp9", "vp8")

    def test_unknown_container_is_rejected(self, synthetic_clip: Path, tmp_path: Path, processor):
        with pytest.raises(ProcessingError, match="unsupported output container"):
            processor.to_video(synthetic_clip, tmp_path / "out.avi")

    def test_missing_source_is_reported(self, tmp_path: Path, processor):
        with pytest.raises(ProcessingError, match="not found"):
            processor.to_video(tmp_path / "nope.mp4", tmp_path / "out.mp4")

    def test_lower_crf_produces_a_bigger_file(self, tmp_path: Path, processor):
        # Only meaningful on detailed content; a flat colour block compresses to
        # the same handful of bytes at any CRF.
        noisy = make_noisy_clip(tmp_path / "noisy.mp4")
        high = processor.to_video(noisy, tmp_path / "crf18.mp4", crf=18, preset="ultrafast")
        low = processor.to_video(noisy, tmp_path / "crf34.mp4", crf=34, preset="ultrafast")
        assert high.stat().st_size > low.stat().st_size

    def test_scaling_caps_the_width(self, synthetic_clip: Path, tmp_path: Path, processor):
        out = processor.to_video(synthetic_clip, tmp_path / "small.mp4", scale=320)
        info = probe(out)
        assert info.width == 320
        assert info.height == 240  # aspect ratio preserved, even height


class TestGif:
    def test_two_pass_gif_is_produced(self, synthetic_clip: Path, tmp_path: Path, processor):
        out = processor.to_gif(synthetic_clip, tmp_path / "out.gif")
        assert out.exists() and out.stat().st_size > 0
        assert out.read_bytes()[:6] in (b"GIF87a", b"GIF89a")

    def test_palette_scratch_file_is_cleaned_up(self, synthetic_clip: Path, tmp_path: Path, processor):
        processor.to_gif(synthetic_clip, tmp_path / "out.gif")
        leftovers = list(tmp_path.glob("*.palette.png"))
        assert leftovers == [], f"palette file leaked: {leftovers}"
    def test_max_colors_is_validated(self, synthetic_clip: Path, tmp_path: Path, processor):
        with pytest.raises(ProcessingError, match="2-256"):
            processor.to_gif(synthetic_clip, tmp_path / "out.gif", max_colors=1000)

    def test_dither_option_is_validated(self, synthetic_clip: Path, tmp_path: Path, processor):
        with pytest.raises(ValueError, match="unknown dither"):
            processor.to_gif(synthetic_clip, tmp_path / "out.gif", dither="sneaky")

    @pytest.mark.parametrize("dither", ["bayer", "floyd_steinberg", "none", "sierra2"])
    def test_each_dither_produces_a_gif(self, synthetic_clip: Path, tmp_path: Path, processor, dither):
        out = processor.to_gif(synthetic_clip, tmp_path / f"{dither}.gif", dither=dither, fps=10)
        assert out.stat().st_size > 0

    def test_width_cap_is_honoured(self, synthetic_clip: Path, tmp_path: Path, processor):
        out = processor.to_gif(synthetic_clip, tmp_path / "narrow.gif", width=320, fps=10)
        info = probe(out)
        assert info.width == 320


class TestOverlays:
    def test_filter_is_built_from_positioned_actions(self, processor):
        actions = [
            Action(action="click", timestamp=1.0, x=100, y=200),
            Action(action="scroll", timestamp=2.0),  # no position: skipped
        ]
        vf = processor.build_overlay_filter(actions, (1000, 1000), (1000, 1000))
        assert "drawbox" in vf
        assert vf.count("enable='between(t,1.000,1.450)'") == 3  # ring + 2 crosshair arms
        assert "2.000" not in vf, "unpositioned actions must not be drawn"

    def test_coordinates_scale_with_the_target(self, processor):
        actions = [Action(action="click", timestamp=0.0, x=500, y=500)]
        vf = processor.build_overlay_filter(actions, (1000, 1000), (500, 500))
        # 500 source px at half scale lands on 250; the ring is radius 18.
        assert "x=232.0:y=232.0" in vf, vf

    def test_negative_coordinates_are_clamped(self, processor):
        actions = [Action(action="click", timestamp=0.0, x=5, y=5)]
        vf = processor.build_overlay_filter(actions, (1000, 1000), (1000, 1000))
        assert "x=0.0" in vf and "y=0.0" in vf

    def test_commas_inside_enable_are_quoted_not_escaped(self, processor):
        """Unescaped commas would split the filtergraph and break the encode."""
        actions = [Action(action="click", timestamp=1.5, x=10, y=10)]
        vf = processor.build_overlay_filter(actions, (100, 100), (100, 100))
        assert "\\," not in vf
        assert "enable='between(t,1.500,1.950)'" in vf

    def test_empty_actions_produce_no_filter(self, processor):
        assert processor.build_overlay_filter([], (100, 100), (100, 100)) == ""

    def test_overlay_render_produces_a_playable_file(self, synthetic_clip: Path, tmp_path: Path, processor):
        actions = [
            Action(action="click", timestamp=0.4, x=100, y=100),
            Action(action="click", timestamp=1.4, x=300, y=200),
            Action(action="click", timestamp=2.4, x=500, y=300),
        ]
        out = processor.apply_overlays(synthetic_clip, tmp_path / "overlay.mp4", actions)
        assert probe(out).duration > 0

    def test_export_video_routes_through_overlays_when_asked(self, synthetic_clip: Path, tmp_path: Path, processor):
        actions = [Action(action="click", timestamp=1.0, x=100, y=100)]
        out = processor.export_video(
            synthetic_clip, tmp_path / "out.mp4", actions=actions, with_overlays=True
        )
        assert probe(out).duration > 0

    def test_export_video_skips_overlays_when_not_asked(self, synthetic_clip: Path, tmp_path: Path):
        plain = Processor(overlay_clicks=False)
        out = plain.export_video(
            synthetic_clip, tmp_path / "plain.mp4",
            actions=[Action(action="click", timestamp=1.0, x=1, y=1)],
            with_overlays=False,
        )
        assert probe(out).duration > 0


class TestFrameExtraction:
    def test_frame_at_returns_an_image(self, synthetic_clip: Path, tmp_path: Path, processor):
        out = processor.frame_at(synthetic_clip, 1.5, tmp_path / "frame.png")
        assert out.read_bytes().startswith(b"\x89PNG")

    def test_frame_at_can_scale(self, synthetic_clip: Path, tmp_path: Path, processor):
        out = processor.frame_at(synthetic_clip, 0.5, tmp_path / "frame.png", scale=160)
        info = probe(out)
        assert (info.width, info.height) == (160, 120)

    def test_timestamp_past_the_end_is_reported(self, synthetic_clip: Path, tmp_path: Path, processor):
        with pytest.raises(ProcessingError, match="no frame decoded"):
            processor.frame_at(synthetic_clip, 9_999.0, tmp_path / "frame.png")


class TestSceneDetection:
    def test_finds_the_expected_number_of_cuts(self, synthetic_clip: Path, processor):
        times = processor.scene_timestamps(synthetic_clip, threshold=0.05)
        assert len(times) == 2, times
        assert 0.8 < times[0] < 1.2, times
        assert 1.8 < times[1] < 2.2, times

    def test_high_threshold_reports_a_subset(self, tmp_path: Path, processor):
        clip = make_clip(tmp_path / "c.mp4", CUT_PALETTE[:5])
        strict = processor.scene_timestamps(clip, threshold=0.99)
        assert 0 < len(strict) < 4
        assert set(strict) <= set(processor.scene_timestamps(clip, threshold=0.05))

    def test_low_threshold_finds_at_least_the_cuts(self, tmp_path: Path, processor):
        noisy = make_clip(tmp_path / "noisy.mp4", CUT_PALETTE[:5])
        assert len(processor.scene_timestamps(noisy, threshold=0.05)) == 4

    def test_high_threshold_finds_fewer(self, tmp_path: Path, processor):
        clip = make_clip(tmp_path / "c.mp4", CUT_PALETTE[:5])
        assert len(processor.scene_timestamps(clip, threshold=0.99)) < 4
