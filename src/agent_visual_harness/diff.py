"""Visual regression diffing — the thing no operating system does.

``diff`` compares text. It has never once told you that a CSS tweak moved a
submit button by six pixels. That is what pixel diffing is for, and the reason it
is in every serious visual-regression suite is also why it is worth getting right:
a diff that flags a one-pixel antialiasing shift on every run gets switched off
within a week, and then it catches nothing.

Three decisions make this usable rather than noisy:

* **Perceptual distance, not byte equality.** Pixels are compared on luma
  difference, so a JPEG artefact does not read as a redesign.
* **A tolerance you have to choose, and that gets reported.**
  ``threshold`` is how far apart two pixels must be before one counts as
  changed; ``min_changed_pct`` is how much of the image must change before the
  run fails. :meth:`DiffResult.verdict` names which one tripped.
* **A diff image an agent can read.** Changed pixels in red over a dimmed
  original. That is the artefact a reviewer looks at, and it is also something a
  vision model can describe in one sentence.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Sequence

from ._ffmpeg import FFMPEG, run, run_bytes
from .capture import image_size, save_image
from .coordinates import Box, iou, union_box
from .errors import ProcessingError

__all__ = [
    "ChangedRegion",
    "DiffResult",
    "DEFAULT_THRESHOLD",
    "DEFAULT_MIN_CHANGED_PCT",
    "bounding_box",
    "compare",
    "compare_files",
]

#: Luma distance at which a pixel counts as changed. 0.10 ignores JPEG noise
#: while still catching a one-pixel shift in a border colour.
DEFAULT_THRESHOLD = 0.10

#: How much of the image must change before the comparison is a failure.
DEFAULT_MIN_CHANGED_PCT = 0.05

#: Regions smaller than this are folded away; antialiasing around one glyph
#: otherwise produces a cloud of boxes that tells the reader nothing.
_MIN_REGION_AREA = 64

#: Two runs are considered the same horizontal run when the gap is no wider than
#: this, in pixels. Chosen to bridge a character's internal spacing.
_RUN_GAP = 8


@dataclass(slots=True)
class ChangedRegion:
    """A rectangle of meaningful change, and how much of it actually changed."""

    box: Box
    changed_pixels: int
    kind: str = "changed"

    @property
    def area(self) -> int:
        return self.box[2] * self.box[3]

    def to_dict(self) -> dict:
        return {"box": list(self.box), "changed_pixels": self.changed_pixels, "kind": self.kind}


@dataclass(slots=True)
class DiffResult:
    """The verdict, the numbers behind it, and an image a human can read."""

    same: bool
    width: int
    height: int
    changed_pixels: int
    total_pixels: int
    changed_pct: float
    max_distance: float
    mean_distance: float
    threshold: float
    min_changed_pct: float
    regions: list[ChangedRegion] = field(default_factory=list)
    diff_image: bytes | None = None
    size_mismatch: tuple[int, int] | None = None
    elapsed: float = 0.0

    @property
    def region_count(self) -> int:
        return len(self.regions)

    @property
    def largest_region(self) -> ChangedRegion | None:
        return max(self.regions, key=lambda r: r.area, default=None)

    @property
    def bbox(self) -> Box | None:
        return bounding_box(self.regions)

    def region_containing(self, x: int, y: int) -> ChangedRegion | None:
        for region in self.regions:
            left, top, width, height = region.box
            if left <= x < left + width and top <= y < top + height:
                return region
        return None

    def verdict(self) -> str:
        """One line a CI log — or an agent — can act on without parsing the rest."""
        if self.size_mismatch:
            return (f"SIZE MISMATCH: expected {self.size_mismatch[0]}x{self.size_mismatch[1]}, "
                    f"got {self.width}x{self.height}")
        if self.same:
            return f"identical across {self.total_pixels:,} pixels (threshold {self.threshold})"
        where = ""
        biggest = self.largest_region
        if biggest is not None:
            left, top, width, height = biggest.box
            where = f", largest region {width}x{height} at ({left},{top})"
        return (f"{self.changed_pct:.3f}% of pixels changed "
                f"({self.changed_pixels:,} of {self.total_pixels:,}, "
                f"max Δ {self.max_distance:.3f}){where}")

    def save_diff(self, path: Path | str) -> Path:
        if self.diff_image is None:
            raise ProcessingError("this result carries no diff image")
        return save_image(path, self.diff_image)

    def to_dict(self) -> dict:
        biggest = self.largest_region
        return {
            "same": self.same,
            "width": self.width,
            "height": self.height,
            "changed_pixels": self.changed_pixels,
            "total_pixels": self.total_pixels,
            "changed_pct": round(self.changed_pct, 4),
            "max_distance": round(self.max_distance, 4),
            "mean_distance": round(self.mean_distance, 5),
            "threshold": self.threshold,
            "min_changed_pct": self.min_changed_pct,
            "region_count": self.region_count,
            # `regions` is ordered for reading (top-left first), so the largest is
            # not regions[0]. Publishing it explicitly saves every caller from
            # re-deriving it — and from the mistake of assuming.
            "largest_region": biggest.to_dict() if biggest else None,
            "bbox": list(self.bbox) if self.bbox else None,
            "regions": [r.to_dict() for r in self.regions[:20]],
            "size_mismatch": list(self.size_mismatch) if self.size_mismatch else None,
            "verdict": self.verdict(),
        }


# ------------------------------------------------------------------- plumbing


def _ffmpeg(args: Sequence[str]) -> bytes:
    """Run ffmpeg and return stdout as bytes."""
    return run_bytes(list(args), timeout=300)


def _ffmpeg_to_file(args: Sequence[str]) -> None:
    """Run ffmpeg for its side effect: writing a file, not filling stdout."""
    run(list(args), timeout=300)


class _Pair:
    """Two normalised images on disk, ready for ffmpeg's two-input filters.

    ffmpeg reads each ``-i`` from its own file descriptor, so two images cannot
    share one pipe. Materialising them once per comparison costs a couple of
    small writes and makes every subsequent pass a plain, inspectable command.
    """

    def __init__(self, before: bytes, after: bytes, width: int, height: int):
        import tempfile

        self._tmp = tempfile.TemporaryDirectory(prefix="dynamo-diff-")
        root = Path(self._tmp.name)
        self.left = root / "before.png"
        self.right = root / "after.png"
        self.left.write_bytes(_normalise(before, width, height))
        self.right.write_bytes(_normalise(after, width, height))

    def __enter__(self) -> _Pair:
        return self

    def __exit__(self, *_exc) -> bool:
        self._tmp.cleanup()
        return False

    # ---------------------------------------------------------- pass helpers

    def distances(self) -> bytes:
        return _ffmpeg([
            FFMPEG, "-hide_banner", "-loglevel", "error",
            "-i", str(self.left), "-i", str(self.right),
            "-filter_complex", "[0:v][1:v]blend=all_mode=difference,format=gray",
            "-f", "rawvideo", "-pix_fmt", "gray", "-",
        ])

    def write_mask(self, cut: int) -> Path:
        """Binary mask on disk: white where the pair exceeded ``cut``."""
        path = self.left.parent / "mask.png"
        _ffmpeg_to_file([
            FFMPEG, "-hide_banner", "-loglevel", "error",
            "-i", str(self.left), "-i", str(self.right),
            "-filter_complex",
            "[0:v][1:v]blend=all_mode=difference,format=gray,"
            f"lut=y='if(gt(val,{cut}),255,0)'",
            "-frames:v", "1", "-c:v", "png", str(path),
        ])
        return path

    def overlay(self, mask_path: Path) -> bytes:
        """Changed pixels in red over a dimmed copy of the original.

        The mask is strictly black-or-white, so mapping it through
        colorchannelmixer produces pure red, and colorkey then makes the black
        half transparent. No alpha maths in an expression parser.
        """
        return _ffmpeg([
            FFMPEG, "-hide_banner", "-loglevel", "error",
            "-i", str(self.left), "-i", str(mask_path),
            "-filter_complex",
            "[0:v]format=gray,geq=lum='lum(X,Y)*0.30':cb=128:cr=128,format=rgb24[dim];"
            "[1:v]format=gray,colorchannelmixer="
            "rr=1:rg=0:rb=0:gr=0.235:gg=0:gb=0:br=0.27:bg=0:bb=0,"
            "colorkey=0x000000:0.05:0.0[red];"
            # `rgb` is the overlay filter's name for the packed-RGB format;
            # `rgb24` is the encoder's name for the same thing and is rejected here.
            "[dim][red]overlay=format=rgb",
            "-frames:v", "1", "-c:v", "png", "-f", "image2pipe", "-",
        ])


def _normalise(data: bytes, width: int, height: int) -> bytes:
    """Decode to a known-size RGB PNG so both sides are byte-comparable."""
    source_w, source_h = image_size(data)
    args = [FFMPEG, "-hide_banner", "-loglevel", "error", "-f", "image2pipe", "-i", "-"]
    if (source_w, source_h) != (width, height):
        args += ["-vf", f"scale={width}:{height}:flags=neighbor"]
    args += ["-pix_fmt", "rgb24", "-c:v", "png", "-f", "image2pipe", "-"]
    return run_bytes(args, timeout=120, input_bytes=data)


# ----------------------------------------------------------------------- core


def compare(
    before: bytes,
    after: bytes,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    min_changed_pct: float = DEFAULT_MIN_CHANGED_PCT,
    diff_image: bool = True,
    max_regions: int = 20,
) -> DiffResult:
    """Compare two images and say whether the difference matters.

    Parameters
    ----------
    before, after:
        Encoded image bytes (PNG, JPEG, WebP, GIF).
    threshold:
        Per-pixel luma distance, 0-1, above which a pixel counts as changed.
    min_changed_pct:
        Percentage of the image that must change before ``same`` is ``False``.
    diff_image:
        Build the red-on-dimmed overlay.
    max_regions:
        Cap on how many changed boxes to report.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be between 0 and 1, got {threshold}")
    if not 0.0 <= min_changed_pct <= 100.0:
        raise ValueError(f"min_changed_pct must be between 0 and 100, got {min_changed_pct}")

    started = time.monotonic()
    before_size = image_size(before)
    after_size = image_size(after)

    if before_size != after_size:
        return _resized(before, after, before_size, after_size, threshold,
                        min_changed_pct, diff_image, started)

    width, height = after_size
    with _Pair(before, after, width, height) as pair:
        distances = [value / 255.0 for value in pair.distances()]
        cut = int(round(threshold * 255))
        changed = sum(1 for d in distances if d > threshold)
        total = width * height
        overlay = None
        if diff_image:
            overlay = pair.overlay(pair.write_mask(cut))

    return DiffResult(
        same=(100.0 * changed / total if total else 0.0) < min_changed_pct,
        width=width, height=height,
        changed_pixels=changed, total_pixels=total,
        changed_pct=100.0 * changed / total if total else 0.0,
        max_distance=max(distances) if distances else 0.0,
        mean_distance=(sum(distances) / len(distances)) if distances else 0.0,
        threshold=threshold, min_changed_pct=min_changed_pct,
        regions=_regions(distances, width, height, threshold, max_regions) if changed else [],
        diff_image=overlay,
        elapsed=time.monotonic() - started,
    )


def _resized(before: bytes, after: bytes, before_size: tuple[int, int],
             after_size: tuple[int, int], threshold: float, min_changed_pct: float,
             diff_image: bool, started: float) -> DiffResult:
    """A resize is never a subtle regression. Report it loudly and stop."""
    with _Pair(before, after, after_size[0], after_size[1]) as pair:
        distances = [value / 255.0 for value in pair.distances()]
        changed = sum(1 for d in distances if d > threshold)
        total = after_size[0] * after_size[1]
        overlay = None
        if diff_image:
            cut = int(round(threshold * 255))
            overlay = pair.overlay(pair.write_mask(cut))

    return DiffResult(
        same=False,
        width=after_size[0], height=after_size[1],
        changed_pixels=changed, total_pixels=total,
        changed_pct=100.0 * changed / total if total else 0.0,
        max_distance=max(distances) if distances else 0.0,
        mean_distance=(sum(distances) / len(distances)) if distances else 0.0,
        threshold=threshold, min_changed_pct=min_changed_pct,
        regions=_regions(distances, after_size[0], after_size[1], threshold, 5),
        diff_image=overlay,
        size_mismatch=before_size,
        elapsed=time.monotonic() - started,
    )


# --------------------------------------------------------------------- regions


def _row_runs(row: list[float], threshold: float) -> list[tuple[int, int]]:
    """Contiguous changed spans in one row, bridging gaps no wider than _RUN_GAP."""
    runs: list[tuple[int, int]] = []
    for x, value in enumerate(row):
        if value <= threshold:
            continue
        if runs and x <= runs[-1][1] + _RUN_GAP:
            runs[-1] = (runs[-1][0], x)
        else:
            runs.append((x, x))
    return runs


def _regions(distances: list[float], width: int, height: int, threshold: float,
             max_regions: int) -> list[ChangedRegion]:
    """Group changed pixels into boxes worth showing a human.

    Horizontal runs are merged, then tracked across rows, rather than
    flood-filling the mask: a real diff is a handful of contiguous blocks, and
    connected-component labelling on antialiased text produces thousands of
    single pixels that tell the reader nothing.

    The tracker keeps one row of look-behind. A run from the previous row that
    nothing continues is *closed* there — dropping it instead is the difference
    between reporting where a stylesheet moved the form and reporting nothing
    at all, because text reflow leaves every glyph a separate few-pixel sliver.
    """
    boxes: list[Box] = []
    open_runs: list[list[int]] = []  # [x0, x1, start_y, last_y]

    for y in range(height):
        runs = _row_runs(distances[y * width : (y + 1) * width], threshold)
        still_open: list[list[int]] = []

        for x0, x1 in runs:
            match = next(
                (r for r in open_runs
                 if not (x1 < r[0] - _RUN_GAP or x0 > r[1] + _RUN_GAP)),
                None,
            )
            if match is not None:
                open_runs.remove(match)
                still_open.append([min(match[0], x0), max(match[1], x1), match[2], y])
            else:
                still_open.append([x0, x1, y, y])

        # Whatever the previous row was carrying and this row did not continue.
        for x0, x1, start_y, _last_y in open_runs:
            boxes.append((x0, start_y, x1 - x0 + 1, max(y - start_y, 1)))
        open_runs = still_open

    for x0, x1, start_y, last_y in open_runs:
        boxes.append((x0, start_y, x1 - x0 + 1, last_y - start_y + 1))

    regions: list[ChangedRegion] = []
    for left, top, box_w, box_h in boxes:
        if box_w * box_h < _MIN_REGION_AREA:
            continue
        count = sum(
            1
            for y in range(top, min(top + box_h, height))
            for x in range(left, min(left + box_w, width))
            if distances[y * width + x] > threshold
        )
        regions.append(ChangedRegion(box=(left, top, box_w, box_h), changed_pixels=count))

    regions.sort(key=lambda r: r.area, reverse=True)
    collapsed: list[ChangedRegion] = []
    for region in regions:
        if any(iou(region.box, kept.box) > 0.5 for kept in collapsed):
            continue
        collapsed.append(region)
        if len(collapsed) >= max_regions:
            break
    collapsed.sort(key=lambda r: (r.box[1], r.box[0]))
    return collapsed


def bounding_box(regions: Sequence[ChangedRegion]) -> Box | None:
    """Smallest box covering every changed region."""
    return union_box(r.box for r in regions)


def compare_files(
    before: Path | str,
    after: Path | str,
    *,
    diff_path: Path | str | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    min_changed_pct: float = DEFAULT_MIN_CHANGED_PCT,
) -> DiffResult:
    """Compare two image files on disk."""
    result = compare(Path(before).read_bytes(), Path(after).read_bytes(),
                     threshold=threshold, min_changed_pct=min_changed_pct)
    if diff_path is not None:
        result.save_diff(diff_path)
    return result


def to_json(result: DiffResult) -> str:  # pragma: no cover - debugging aid
    return json.dumps(result.to_dict(), indent=2)
