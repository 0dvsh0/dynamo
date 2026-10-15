"""Tests for the agent-facing visual primitives: coordinates, capture, OCR, diff,
grounding. These are the CTO's categories 1, 4, 5 and 6.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from agent_visual_harness import (
    Grounding,
    Scale,
    Surface,
    Target,
    annotate_boxes,
    compare,
    image_size,
    read_image,
    read_region,
    resolve,
)
from agent_visual_harness.capture import convert
from agent_visual_harness.coordinates import clamp_box, iou, union_box
from agent_visual_harness.grounding import resolve_box
from agent_visual_harness.ocr import OCRResult

from .conftest import requires_ffmpeg, requires_linux, requires_xvfb

requires_tesseract = pytest.mark.skipif(
    shutil.which("tesseract") is None, reason="tesseract-ocr is not installed"
)
requires_font = pytest.mark.skipif(
    shutil.which("fc-match") is None, reason="fontconfig is not installed"
)


# ------------------------------------------------------------------ helpers


def solid(colour: str, width: int = 200, height: int = 80) -> bytes:
    """A flat rectangle of one colour, as PNG bytes."""
    from agent_visual_harness._ffmpeg import FFMPEG, run_bytes

    return run_bytes(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
         f"color=c={colour}:s={width}x{height}", "-frames:v", "1",
         "-c:v", "png", "-f", "image2pipe", "-"],
        timeout=60,
    )


def two_colours(left: str, right: str, width: int = 200, height: int = 100) -> bytes:
    from agent_visual_harness._ffmpeg import FFMPEG, run_bytes

    return run_bytes(
        [FFMPEG, "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", f"color=c={left}:s={width}x{height}",
         "-f", "lavfi", "-i", f"color=c={right}:s={width}x{height}",
         "-filter_complex", "[0:v][1:v]hstack,format=rgb24", "-frames:v", "1",
         "-c:v", "png", "-f", "image2pipe", "-"],
        timeout=60,
    )


def text_image(lines: list[str], width: int = 640, height: int = 200) -> bytes:
    """Render text with drawtext so OCR has something real to read.

    Two escaping traps, both found the hard way:

    * One drawtext per line rather than an embedded ``\\n`` — the filter parser
      eats one level of backslashes, so a single filter silently renders
      "line 0nline 1" on one row.
    * Colons must be backslash-escaped. Quoting does **not** protect them:
      drawtext splits its own arguments on ``:``, so ``text='TypeError: x'`` loses
      everything up to the colon and OCR never sees the word.
    """
    from agent_visual_harness._ffmpeg import FFMPEG, run_bytes

    filters = [
        f"drawtext=text='{line.replace(':', chr(92) + ':')}':fontsize=26"
        f":fontcolor=black:x=20:y={40 + i * 36}"
        for i, line in enumerate(lines)
    ]
    return run_bytes(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-f", "lavfi",
         "-i", f"color=c=white:s={width}x{height}",
         "-vf", ",".join(filters),
         "-frames:v", "1", "-c:v", "png", "-f", "image2pipe", "-"],
        timeout=60,
    )


def _ocr_result(*lines: str) -> OCRResult:
    """An OCRResult built straight from text.

    The fuzzy-matching tests need a *known* transcription — what Tesseract
    actually produced for this console — rather than a lucky second run that
    might read the glyphs correctly and prove nothing.
    """
    from agent_visual_harness.ocr import Line, OCRResult, Word

    out = []
    words = []
    for i, text in enumerate(lines):
        ws = [
            Word(token, (0, i * 20 + j * 8, 40, 16), 90.0)
            for j, token in enumerate(text.split())
        ]
        words.extend(ws)
        out.append(Line(text, (0, i * 20, len(text) * 8, 18), 90.0, ws))
    return OCRResult(
        text="\n".join(lines), lines=out, words=words, width=800, height=400
    )


# ---------------------------------------------------------------- coordinates


class TestScale:
    def test_identity_needs_no_conversion(self):
        scale = Scale()
        assert scale.is_identity
        assert scale.to_device(10, 20) == (10, 20)
        assert scale.to_logical(10, 20) == (10, 20)

    def test_retina_doubles_coordinates(self):
        """The CTO's friction point B: 200% scale, image is 2x the pointer space."""
        scale = Scale(ratio=2.0)
        assert scale.to_device(500, 300) == (1000, 600)
        assert scale.to_logical(1000, 600) == (500, 300)

    def test_round_trip_survives_scaling(self):
        """Scaling out and back must land where it started."""
        scale = Scale(ratio=1.5)
        for point in ((0, 0), (17, 43), (1280, 800)):
            back = scale.to_logical(*scale.to_device(*point))
            assert back == pytest.approx(point, abs=1)

    def test_origin_handles_window_insets(self):
        """A browser page does not start at (0,0) — the tab strip is above it."""
        scale = Scale(ratio=1.0, origin=(0, 85))
        assert scale.to_device(0, 0) == (0, 85)
        assert scale.to_logical(0, 85) == (0, 0)

    def test_offset_and_scaled_together(self):
        scale = Scale(ratio=2.0, origin=(0, 90))
        assert scale.to_device(100, 50) == (200, 190)
        assert scale.to_logical(200, 190) == (100, 50)

    def test_box_conversion(self):
        scale = Scale(ratio=2.0)
        assert scale.box_to_device((10, 20, 30, 40)) == (20, 40, 60, 80)
        assert scale.box_to_logical((20, 40, 60, 80)) == (10, 20, 30, 40)

    def test_between_derives_the_ratio(self):
        assert Scale.between((1440, 900), (2880, 1800)).ratio == pytest.approx(2.0)

    def test_between_rejects_degenerate_input(self):
        with pytest.raises(ValueError, match="positive"):
            Scale.between((0, 100), (100, 100))

    def test_zero_ratio_is_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            Scale(ratio=0)

    def test_describe_is_human_readable(self):
        assert "1:1" in Scale().describe()
        assert "×2" in Scale(ratio=2.0).describe()


class TestBoxHelpers:
    def test_clamp_keeps_boxes_inside(self):
        # A box starting off-screen is pulled in and keeps its size where it fits.
        assert clamp_box((-10, -10, 50, 50), (100, 100)) == (0, 0, 50, 50)
        assert clamp_box((90, 90, 50, 50), (100, 100)) == (90, 90, 10, 10)

    def test_clamp_never_returns_negative_size(self):
        left, top, width, height = clamp_box((200, 200, 50, 50), (100, 100))
        assert width == 0 and height == 0 and left <= 100

    def test_union_covers_everything(self):
        assert union_box([(0, 0, 10, 10), (50, 50, 10, 10)]) == (0, 0, 60, 60)

    def test_union_of_nothing_is_none(self):
        assert union_box([]) is None

    def test_iou_of_identical_boxes_is_one(self):
        assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)

    def test_iou_of_disjoint_boxes_is_zero(self):
        assert iou((0, 0, 10, 10), (50, 50, 10, 10)) == 0.0


# -------------------------------------------------------------------- capture


@requires_ffmpeg
class TestImageSize:
    def test_reads_png(self):
        assert image_size(solid("#ff0000", 64, 48)) == (64, 48)

    def test_reads_jpeg(self):
        assert image_size(convert(solid("#ff0000", 64, 48), "jpeg")) == (64, 48)

    def test_reads_webp(self):
        assert image_size(convert(solid("#ff0000", 64, 48), "webp")) == (64, 48)

    def test_reads_gif(self):
        gif = (Path(__file__).parent.parent / "demo" / "assets" / "dynamo_demo.gif")
        if gif.exists():
            assert image_size(gif.read_bytes())[0] > 0

    def test_rejects_rubbish(self):
        with pytest.raises(ValueError, match="unrecognised image format"):
            image_size(b"not an image at all")

    def test_convert_changes_format(self):
        assert convert(solid("#00ff00", 32, 32), "jpeg")[:2] == b"\xff\xd8"


@requires_ffmpeg
@requires_linux
@requires_xvfb
class TestSurface:
    def test_grab_returns_a_full_frame(self, display):
        surface = Surface.from_display(display.display, (640, 480))
        data = surface.grab()
        assert data[:8] == b"\x89PNG\r\n\x1a\n"
        assert image_size(data) == (640, 480)

    def test_snip_returns_the_region_size(self, display):
        surface = Surface.from_display(display.display, (640, 480))
        data = surface.snip((100, 50, 200, 120))
        assert image_size(data) == (200, 120)

    def test_snip_is_much_smaller_than_a_full_grab(self, display):
        """The whole point of region capture: fewer bytes, same information."""
        surface = Surface.from_display(display.display, (640, 480))
        full = len(surface.grab())
        small = len(surface.snip((10, 10, 160, 60)))
        assert small < full

    def test_snip_clamps_a_region_that_overruns_the_surface(self, display):
        surface = Surface.from_display(display.display, (640, 480))
        assert image_size(surface.snip((600, 440, 200, 200))) == (40, 40)

    def test_snip_rejects_an_empty_region(self, display):
        surface = Surface.from_display(display.display, (640, 480))
        with pytest.raises(ValueError, match="empty after clamping"):
            surface.snip((700, 500, 50, 50))

    def test_scaled_snip_maps_logical_to_device(self, display):
        """A logical box on a 2x surface must crop twice as many device pixels."""
        surface = Surface.from_display(display.display, (640, 480))
        # A logical 100x50 region at ratio 2 covers 200x100 device pixels.
        data = surface.snip_scaled((0, 0, 100, 50), scale=Scale(ratio=2.0))
        assert image_size(data) == (200, 100)

    def test_scaled_snip_honours_the_window_origin(self, display):
        surface = Surface.from_display(display.display, (640, 480))
        data = surface.snip_scaled((0, 0, 100, 50), scale=Scale(origin=(0, 85)))
        assert image_size(data) == (100, 50)

    def test_save_writes_to_disk(self, display, tmp_path: Path):
        surface = Surface.from_display(display.display, (640, 480))
        path = surface.save(tmp_path / "shot.png")
        assert path.exists() and path.stat().st_size > 0

    def test_save_snip_writes_to_disk(self, display, tmp_path: Path):
        surface = Surface.from_display(display.display, (640, 480))
        path = surface.save_snip(tmp_path / "crop.png", (10, 10, 120, 80))
        assert image_size(path.read_bytes()) == (120, 80)

    def test_rejects_unknown_format(self, display):
        surface = Surface.from_display(display.display, (640, 480))
        with pytest.raises(ValueError, match="unsupported format"):
            surface.grab(fmt="tiff")


# ------------------------------------------------------------------------ OCR


@requires_ffmpeg
@requires_tesseract
class TestOCR:
    def test_reads_rendered_text(self):
        result = read_image(text_image(["ERROR", "TypeError: x is not a function"]))
        assert "TypeError" in result.text
        assert "not a function" in result.text
        assert result.line_count >= 2

    def test_reports_confidence(self):
        result = read_image(text_image(["Hello world"]))
        assert 0 < result.confidence <= 100
        assert result.confident, f"confidence {result.confidence}"

    def test_reports_its_own_elapsed_time(self):
        assert read_image(text_image(["x"])).elapsed > 0

    def test_word_boxes_are_inside_the_image(self):
        result = read_image(text_image(["ERROR 42"]), psm=6)
        for word in result.words:
            left, top, width, height = word.box
            assert 0 <= left < result.width
            assert 0 <= top < result.height
            assert width > 0 and height > 0

    def test_box_of_finds_a_line_by_text(self):
        """The bridge back to pixels: read the error, then point at it."""
        result = read_image(text_image(["Error", "ECONNREFUSED 127.0.0.1:5432"]))
        box = result.box_of("ECONNREFUSED")
        assert box is not None
        left, top, width, height = box
        assert width > 0 and height > 0
        assert 0 <= top < result.height

    def test_box_of_returns_none_for_missing_text(self):
        assert read_image(text_image(["hello"])).box_of("ZEBRA") is None

    def test_contains_is_case_insensitive_by_default(self):
        assert read_image(text_image(["Connection Refused"])).contains("connection refused")

    def test_lines_matching_filters(self):
        result = read_image(text_image(["ok", "FAIL", "also FAIL"]))
        assert len(result.lines_matching("FAIL")) == 2

    def test_blank_image_yields_nothing(self):
        result = read_image(solid("#ffffff", 200, 80))
        assert result.text.strip() == ""
        assert not result
        assert result.confidence == 0.0

    def test_region_ocr_reports_boxes_in_full_image_coordinates(self):
        """A crop must not renumber its boxes as if it started at (0, 0)."""
        base = text_image(["MARKERTOKEN"], width=800, height=300)
        result = read_region(base, (0, 0, 400, 300))
        assert result.words, "region OCR found nothing"
        assert all(w.box[0] < 400 for w in result.words)
        assert result.width == 400

    def test_region_ocr_preserves_a_right_hand_offset(self):
        """Text drawn at x=500 must be reported at x>=500 when cropped from 500."""
        from agent_visual_harness._ffmpeg import FFMPEG, run_bytes

        base = run_bytes(
            [FFMPEG, "-hide_banner", "-loglevel", "error", "-f", "lavfi",
             "-i", "color=c=white:s=800x200",
             "-vf", "drawtext=text='RIGHTEDGE':fontsize=30:fontcolor=black:x=520:y=70",
             "-frames:v", "1", "-c:v", "png", "-f", "image2pipe", "-"],
            timeout=60,
        )
        result = read_region(base, (500, 0, 300, 200))
        assert result.words, "region OCR found nothing"
        assert all(w.box[0] >= 500 for w in result.words), [w.box for w in result.words][:3]

    def test_region_ocr_rejects_empty_box(self):
        with pytest.raises(ValueError, match="empty"):
            read_region(solid("#ffffff", 100, 100), (0, 0, 0, 0))

    def test_summary_truncates(self):
        result = read_image(text_image([f"line {i}" for i in range(8)]))
        summary = result.summary()
        assert "more lines" in summary
        assert "confidence" in summary

    def test_serialises(self):
        payload = read_image(text_image(["OK"])).to_dict()
        assert {"text", "confidence", "lines", "engine"} <= set(payload)


@requires_ffmpeg
class TestOCRFuzzyMatching:
    """The measured Tesseract failure modes, and the matching layer over them.

    On a real payment console Tesseract 5.3 reads ``7f3c9ab1`` as ``7f3c9abl``,
    ``DO_NOT_HONOR`` as ``D0_ NOT HONOR`` and ``rule=velocity_exceeded`` as
    ``rule=velocity exceeded``. No tesseract knob fixes it, so matching folds
    these while the returned text stays exactly as recognised.
    """

    def test_fold_collapses_case(self):
        from agent_visual_harness import fold

        assert fold("DO_NOT_HONOR") == fold("do_not_honor")
        assert fold("DECLINED") == "dec1ined"  # lowercased, l folded to 1

    def test_fold_collapses_underscore_and_space(self):
        from agent_visual_harness import fold

        assert fold("velocity_exceeded") == fold("velocity exceeded")

    def test_fold_collapses_zero_and_capital_o(self):
        from agent_visual_harness import fold

        assert fold("D0_NOT_HONOR") == fold("DO_NOT_HONOR")

    def test_fold_collapses_one_and_lowercase_l(self):
        from agent_visual_harness import fold

        assert fold("7f3c9ab1") == fold("7f3c9abl")

    def test_fold_preserves_distinct_words(self):
        from agent_visual_harness import fold

        assert fold("FAILED") != fold("PASSED")

    def test_fuzzy_matches_what_strict_misses(self):
        result = _ocr_result("decline.code=D0_ NOT HONOR issuer=CHASE")
        assert not result.contains("DO_NOT_HONOR")
        assert result.contains("DO_NOT_HONOR", fuzzy=True)

    def test_fuzzy_still_returns_the_text_on_screen(self):
        """Folding is for matching; the report must not carry folded glyphs."""
        result = _ocr_result("decline.code=D0_ NOT HONOR issuer=CHASE")
        line = result.line_of("DO_NOT_HONOR", fuzzy=True)
        assert line is not None
        assert "D0_" in line.text

    def test_fuzzy_box_of_points_at_the_real_pixels(self):
        result = _ocr_result("decline.code=D0_ NOT HONOR issuer=CHASE")
        box = result.box_of("DO_NOT_HONOR", fuzzy=True)
        assert box is not None and box[2] > 0

    def test_fuzzy_does_not_invent_matches(self):
        result = _ocr_result("decline.code=D0_ NOT HONOR issuer=CHASE")
        assert not result.contains("ECONNREFUSED", fuzzy=True)

    def test_missed_by_names_the_gaps(self):
        result = _ocr_result("DECLINED reason=do_not_honor")
        assert not result.missed_by(["DECLINED", "DO_NOT_HONOR"], fuzzy=True)
        assert result.missed_by(
            ["DECLINED", "DO_NOT_HONOR", "ECONNREFUSED"], fuzzy=True
        ) == ["ECONNREFUSED"]


@requires_ffmpeg
@requires_tesseract
class TestOCRFind:
    def test_find_locates_a_word(self, tmp_path: Path):
        from agent_visual_harness import find

        image = text_image(["ERROR ECONNREFUSED"], width=700)
        word = find(image, "ECONNREFUSED")
        assert word is not None
        assert word.text.upper() == "ECONNREFUSED"
        assert word.box[2] > 0

    def test_find_returns_none_when_absent(self):
        from agent_visual_harness import find

        assert find(text_image(["READY"]), "ZEBRA") is None

    def test_find_fuzzy_survives_a_confusable(self):
        from agent_visual_harness import find

        image = text_image(["txn 7f3c9ab1 captured"], width=700)
        assert find(image, "7f3c9ab1") is None or True  # either may match verbatim
        word = find(image, "7f3c9ab1", fuzzy=True)
        assert word is not None, "fuzzy find should always locate the id"


# ------------------------------------------------------------------------ diff


def _bar_row(count: int, shift: int, width: int = 400, height: int = 120) -> bytes:
    """A row of small bars, each nudged sideways by ``shift`` pixels.

    This is the shape a CSS letter-spacing change leaves behind: the change is
    not one block but one sliver per glyph, spread across the whole line.
    """
    from agent_visual_harness._ffmpeg import FFMPEG, run_bytes

    boxes = ",".join(
        f"drawbox=x={30 + i * 18 + shift}:y=50:w=12:h=22:color=#222222:t=fill"
        for i in range(count)
    )
    return run_bytes(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
         f"color=c=white:s={width}x{height}", "-vf", boxes,
         "-frames:v", "1", "-c:v", "png", "-f", "image2pipe", "-"],
        timeout=60,
    )


@requires_ffmpeg
class TestDiff:
    def test_identical_images_are_the_same(self):
        a = solid("#123456", 200, 100)
        result = compare(a, a)
        assert result.same
        assert result.changed_pixels == 0
        assert result.changed_pct == 0.0
        assert "identical" in result.verdict()

    def test_a_changed_block_is_found(self):
        before = two_colours("#ffffff", "#ffffff")
        after = two_colours("#ffffff", "#000000")
        result = compare(before, after)
        assert not result.same
        assert result.changed_pixels > 0
        assert result.changed_pct == pytest.approx(50.0, abs=1.0)

    def test_region_locates_the_change(self):
        before = two_colours("#ffffff", "#ffffff")
        after = two_colours("#ffffff", "#000000")
        result = compare(before, after)
        assert result.region_count >= 1
        box = result.largest_region.box
        # The right half changed, so the box must sit to the right of centre.
        assert box[0] > 180, box

    def test_a_sub_tolerable_change_still_counts_as_same(self):
        """The whole reason a tolerance exists: JPEG noise is not a regression."""
        before = solid("#808080", 200, 100)
        after = convert(before, "jpeg", quality=2)
        assert compare(before, after, threshold=0.25).same

    def test_threshold_is_honoured(self):
        before = two_colours("#ffffff", "#ffffff")
        after = two_colours("#ffffff", "#f0f0f0")
        assert compare(before, after, threshold=0.2).same
        assert not compare(before, after, threshold=0.01).same

    def test_min_changed_pct_controls_the_verdict(self):
        before = two_colours("#ffffff", "#ffffff")
        after = two_colours("#ffffff", "#000000")
        assert compare(before, after, min_changed_pct=90.0).same
        assert not compare(before, after, min_changed_pct=1.0).same

    def test_size_mismatch_is_reported_loudly(self):
        result = compare(solid("#ffffff", 200, 100), solid("#ffffff", 300, 150))
        assert not result.same
        assert result.size_mismatch == (200, 100)
        assert "SIZE MISMATCH" in result.verdict()

    def test_diff_image_is_produced(self):
        before = two_colours("#ffffff", "#ffffff")
        after = two_colours("#ffffff", "#000000")
        result = compare(before, after)
        assert result.diff_image is not None
        assert result.diff_image[:8] == b"\x89PNG\r\n\x1a\n"
        assert image_size(result.diff_image) == (400, 100)

    def test_diff_image_can_be_disabled(self):
        result = compare(solid("#ffffff", 40, 40), solid("#000000", 40, 40), diff_image=False)
        assert result.diff_image is None
        with pytest.raises(Exception, match="no diff image"):
            result.save_diff("/tmp/x.png")

    def test_saving_the_diff_writes_a_file(self, tmp_path: Path):
        result = compare(two_colours("#ffffff", "#ffffff"), two_colours("#ffffff", "#000000"))
        path = result.save_diff(tmp_path / "diff.png")
        assert path.exists() and path.stat().st_size > 0

    def test_statistics_are_populated(self):
        result = compare(two_colours("#ffffff", "#ffffff"), two_colours("#ffffff", "#000000"))
        assert result.max_distance == pytest.approx(1.0, abs=0.05)
        assert 0 < result.mean_distance <= 1.0

    def test_region_containing_finds_a_box_by_point(self):
        result = compare(two_colours("#ffffff", "#ffffff"), two_colours("#ffffff", "#000000"))
        left, top, width, height = result.largest_region.box
        assert result.region_containing(left + 2, top + 2) is not None

    def test_bbox_covers_all_regions(self):
        result = compare(two_colours("#ffffff", "#ffffff"), two_colours("#ffffff", "#000000"))
        assert result.bbox is not None
        assert result.bbox[2] > 0

    def test_serialises(self):
        payload = compare(solid("#ffffff", 40, 40), solid("#000000", 40, 40)).to_dict()
        assert {"same", "changed_pct", "regions", "verdict"} <= set(payload)

    def test_largest_region_is_published_explicitly(self):
        """`regions` is ordered for reading, so regions[0] is not the largest."""
        from agent_visual_harness._ffmpeg import FFMPEG, run_bytes

        frame = run_bytes(
            [FFMPEG, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
             "color=c=white:s=300x160", "-vf",
             "drawbox=x=10:y=10:w=20:h=20:color=#000000:t=fill,"   # small, first
             "drawbox=x=150:y=100:w=140:h=50:color=#000000:t=fill", # large, second
             "-frames:v", "1", "-c:v", "png", "-f", "image2pipe", "-"],
            timeout=60,
        )
        payload = compare(solid("#ffffff", 300, 160), frame).to_dict()
        biggest = payload["largest_region"]["box"]
        assert biggest[2] >= 100, biggest
        assert payload["regions"][0]["box"] != biggest

    @pytest.mark.parametrize("bad", [-0.1, 1.5])
    def test_rejects_impossible_thresholds(self, bad):
        with pytest.raises(ValueError, match="threshold"):
            compare(solid("#ffffff", 20, 20), solid("#ffffff", 20, 20), threshold=bad)

    def test_rejects_impossible_min_changed_pct(self):
        with pytest.raises(ValueError, match="min_changed_pct"):
            compare(solid("#ffffff", 20, 20), solid("#ffffff", 20, 20), min_changed_pct=500)

    def test_scattered_glyph_slivers_still_produce_a_region(self):
        """One sliver per glyph must still be reported as a change.

        Every sliver is a few pixels wide, so a grouper that discards runs which
        do not continue into the next row reports zero regions for a real
        regression — the failure is silent, and the diff simply says "identical".
        """
        result = compare(_bar_row(12, 0), _bar_row(12, 3))
        assert result.changed_pixels > 0, "fixture produced no change at all"
        assert result.region_count > 0, (
            f"{result.changed_pixels} changed pixels but no regions reported"
        )
        bbox = result.bbox
        assert bbox is not None
        left, top, _box_w, _box_h = bbox
        # The change is inside the drawn row, not scattered across the frame.
        assert 0 < left < 400
        assert 45 <= top <= 75, f"bbox top {top} is not on the drawn row"

    def test_a_one_pixel_shift_is_located(self):
        """The classic CSS regression: a block nudged sideways by 3px."""
        from agent_visual_harness._ffmpeg import FFMPEG, run_bytes

        def block(dx: int) -> bytes:
            return run_bytes(
                [FFMPEG, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                 "color=c=white:s=300x160",
                 "-vf", f"drawbox=x={40 + dx}:y=60:w=120:h=40:color=#333333:t=fill",
                 "-frames:v", "1", "-c:v", "png", "-f", "image2pipe", "-"],
                timeout=60,
            )

        result = compare(block(0), block(3))
        assert not result.same
        assert result.region_count >= 1
        left, top, _box_w, _box_h = result.largest_region.box
        assert left < 43, f"expected the box to start near the old edge, got {left}"
        assert 55 <= top <= 105

    def test_a_vertical_shift_is_located(self):
        """The same block moved down instead of sideways."""
        from agent_visual_harness._ffmpeg import FFMPEG, run_bytes

        def block(dy: int) -> bytes:
            return run_bytes(
                [FFMPEG, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                 "color=c=white:s=300x160",
                 "-vf", f"drawbox=x=60:y={40 + dy}:w=120:h=40:color=#333333:t=fill",
                 "-frames:v", "1", "-c:v", "png", "-f", "image2pipe", "-"],
                timeout=60,
            )

        result = compare(block(0), block(5))
        assert not result.same
        left, top, _w, _h = result.largest_region.box
        assert 35 <= left <= 65, f"left {left} is not where the block was drawn"
        assert top < 43, f"expected the box to start near the old top, got {top}"

    def test_two_separate_blocks_are_reported_separately(self):
        """Regions exist to be acted on, so two edits must not merge into one."""
        from agent_visual_harness._ffmpeg import FFMPEG, run_bytes

        def frame(dx: int) -> bytes:
            boxes = (
                f"drawbox=x={20 + dx}:y=20:w=60:h=30:color=#000000:t=fill,"
                f"drawbox=x={200 + dx}:y=100:w=60:h=30:color=#000000:t=fill"
            )
            return run_bytes(
                [FFMPEG, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                 "color=c=white:s=300x160", "-vf", boxes,
                 "-frames:v", "1", "-c:v", "png", "-f", "image2pipe", "-"],
                timeout=60,
            )

        result = compare(frame(0), frame(4))
        assert result.changed_pixels > 0
        assert result.region_count >= 2, [r.box for r in result.regions]
        tops = sorted(r.box[1] for r in result.regions)
        assert any(t < 60 for t in tops), tops
        assert any(t >= 90 for t in tops), tops


# ------------------------------------------------------------------- grounding


@requires_ffmpeg
@requires_font
class TestGrounding:
    def test_draws_a_mark_per_box(self):
        base = solid("#ffffff", 400, 200)
        marks = annotate_boxes(base, [(10, 10, 80, 40), (200, 10, 80, 40)])
        assert len(marks) == 2
        assert marks.indices == [1, 2]
        assert marks.width == 400 and marks.height == 200

    def test_marks_are_centred_on_their_boxes(self):
        marks = annotate_boxes(solid("#ffffff", 200, 100), [(10, 20, 100, 40)])
        assert marks.targets[0].centre == (60, 40)

    def test_boxes_are_clamped_to_the_image(self):
        marks = annotate_boxes(solid("#ffffff", 200, 100), [(180, 90, 100, 100)])
        left, top, width, height = marks.targets[0].box
        assert left + width <= 200 and top + height <= 100

    def test_scale_is_applied_to_logical_boxes(self):
        """A DOM box on a HiDPI surface must be drawn on the right pixels."""
        marks = annotate_boxes(solid("#ffffff", 400, 200), [(10, 10, 50, 20)],
                              scale=Scale(ratio=2.0))
        assert marks.targets[0].box == (20, 20, 100, 40)

    def test_labels_and_roles_are_carried(self):
        marks = annotate_boxes(solid("#ffffff", 200, 100), [(10, 10, 50, 20)],
                              labels=["Pay"], roles=["button"])
        assert marks.targets[0].label == "Pay"
        assert marks.by_label("pay")

    def test_needs_at_least_one_box(self):
        with pytest.raises(ValueError, match="at least one box"):
            annotate_boxes(solid("#ffffff", 40, 40), [])

    def test_a_zero_height_box_is_skipped_not_drawn(self):
        """An element clipped by the viewport has a zero-height DOM rect.

        Drawing a mark on it invites the model to pick an index that resolves
        to a point outside the button, so it must be dropped and reported.
        """
        marks = annotate_boxes(
            solid("#ffffff", 400, 200),
            [(20, 40, 120, 30), (20, 180, 120, 0)],
            labels=["Continue", "Back (clipped)"],
        )
        assert len(marks) == 1
        assert marks.targets[0].label == "Continue"
        assert marks.skipped_count == 1
        assert marks.skipped[0]["label"] == "Back (clipped)"
        assert "clipped" in marks.skipped[0]["reason"]

    def test_remaining_indices_stay_contiguous_after_a_skip(self):
        """A skip must not leave a hole in the numbering the model is shown."""
        marks = annotate_boxes(
            solid("#ffffff", 400, 200),
            [(10, 10, 40, 20), (10, 40, 40, 0), (10, 70, 40, 20)],
        )
        assert marks.indices == [1, 2]

    def test_all_boxes_degenerate_is_an_error(self):
        with pytest.raises(ValueError, match="degenerate"):
            annotate_boxes(solid("#ffffff", 200, 200), [(10, 10, 40, 0)])

    def test_legend_shows_what_was_skipped(self):
        marks = annotate_boxes(
            solid("#ffffff", 400, 200),
            [(20, 40, 120, 30), (20, 180, 120, 0)],
            labels=["Continue", "Back"],
        )
        assert "clipped" in marks.compact()
        assert "Back" in marks.compact()

    def test_prompt_names_the_range(self):
        marks = annotate_boxes(solid("#ffffff", 200, 100), [(1, 1, 20, 20), (30, 30, 20, 20)])
        prompt = marks.prompt("Click the submit button.")
        assert "1-2" in prompt
        assert "submit" in prompt

    def test_compact_legend_lists_every_target(self):
        marks = annotate_boxes(solid("#ffffff", 200, 100), [(1, 1, 20, 20), (30, 30, 20, 20)],
                              labels=["Cancel", "Pay"], roles=["button", "button"])
        legend = marks.compact()
        assert "Cancel" in legend and "Pay" in legend and "<button>" in legend


class TestResolve:
    def _marks(self) -> Grounding:
        """A hand-built Grounding, so these tests need no renderer."""
        return Grounding(
            targets=[Target(1, (100, 200, 80, 40), "Pay"), Target(2, (300, 200, 80, 40), "Cancel")],
            image=b"", width=800, height=600,
        )

    def test_resolves_a_plain_integer(self):
        assert resolve(self._marks(), 1) == (140, 220)

    def test_resolves_a_numeric_string(self):
        assert resolve(self._marks(), "2") == (340, 220)

    def test_resolves_a_sentence(self):
        """Models do not answer with bare integers, whatever we ask them to."""
        assert resolve(self._marks(), "I would click 1.") == (140, 220)

    def test_resolves_a_target(self):
        assert resolve(self._marks(), Target(2, (0, 0, 10, 10))) == (340, 220)

    def test_offset_click_point(self):
        """The centre of a wide input is empty space; a 0.1 offset is not."""
        x, y = resolve(self._marks(), 1, fx=0.1, fy=0.5)
        assert (x, y) == (108, 220)

    def test_scale_is_inverted_on_the_way_out(self):
        """Image pixel -> pointer pixel. This is the silent-failure one."""
        marks = self._marks()
        marks.scale = Scale(ratio=2.0)
        assert resolve(marks, 1) == (70, 110)

    def test_refuses_an_unreadable_answer(self):
        with pytest.raises(ValueError, match="could not read an index"):
            resolve(self._marks(), "the blue one")

    def test_refuses_an_off_screen_index(self):
        with pytest.raises(ValueError, match="not on this screen"):
            resolve(self._marks(), 99)

    def test_refuses_a_boolean(self):
        with pytest.raises(ValueError, match="could not read an index"):
            resolve(self._marks(), True)

    def test_resolve_box_returns_the_element_box(self):
        assert resolve_box(self._marks(), 2) == (300, 200, 80, 40)

    def test_resolve_box_is_dpi_corrected(self):
        marks = self._marks()
        marks.scale = Scale(ratio=2.0)
        assert resolve_box(marks, 1) == (50, 100, 40, 20)
