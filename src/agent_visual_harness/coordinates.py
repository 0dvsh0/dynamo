"""Coordinate spaces, and the DPI drift that breaks them.

An agent looking at a screenshot reasons in *device pixels*: the thing it saw at
(500, 300) is 500, 300 in the image. A tool that wants to click reasons in
*logical pixels*: the X11/Wayland pointer, or a browser's ``clientX``. On a
200%-scaled display those differ by exactly the scale factor, so an agent that
clicks the coordinate it saw lands somewhere else entirely — and, worse, lands
somewhere plausible, so the failure is silent.

This module is the single place that conversion happens, so nothing else in the
harness has to guess which space it is holding.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable, Sequence

__all__ = [
    "Box",
    "Scale",
    "Point",
    "clamp_box",
    "offset_box",
    "union_box",
]

Point = tuple[int, int]
Box = tuple[int, int, int, int]  # left, top, width, height


@dataclass(frozen=True, slots=True)
class Scale:
    """Conversion between logical (CSS/pointer) and device (image) pixels.

    Parameters
    ----------
    ratio:
        Device pixels per logical pixel. Chromium's ``devicePixelRatio``, X11's
        physical scale, or a Retina factor of 2.0.
    origin:
        Device-pixel offset of the logical origin. Non-zero when the captured
        surface is a window inset within a larger screen — the browser's tab
        strip, for instance, puts the page viewport at (0, 85).
    """

    ratio: float = 1.0
    origin: Point = (0, 0)

    def __post_init__(self) -> None:
        if self.ratio <= 0:
            raise ValueError(f"scale ratio must be positive, got {self.ratio}")

    @property
    def is_identity(self) -> bool:
        return self.ratio == 1.0 and self.origin == (0, 0)

    # --------------------------------------------------------- conversions

    def to_device(self, x: float, y: float) -> Point:
        """Logical → device, i.e. where a click should land in the image."""
        return (
            int(round(self.origin[0] + x * self.ratio)),
            int(round(self.origin[1] + y * self.ratio)),
        )

    def to_logical(self, x: float, y: float) -> Point:
        """Device → logical, i.e. where a coordinate read off an image must be sent."""
        return (
            int(round((x - self.origin[0]) / self.ratio)),
            int(round((y - self.origin[1]) / self.ratio)),
        )

    def box_to_device(self, box: Box) -> Box:
        left, top, right, bottom = _unpack(box)
        x1, y1 = self.to_device(left, top)
        x2, y2 = self.to_device(right, bottom)
        return (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))

    def box_to_logical(self, box: Box) -> Box:
        left, top, right, bottom = _unpack(box)
        x1, y1 = self.to_logical(left, top)
        x2, y2 = self.to_logical(right, bottom)
        return (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))

    def points_to_device(self, points: Iterable[Point]) -> list[Point]:
        return [self.to_device(x, y) for x, y in points]

    def points_to_logical(self, points: Iterable[Point]) -> list[Point]:
        return [self.to_logical(x, y) for x, y in points]

    # ----------------------------------------------------------- measuring

    @classmethod
    def between(cls, logical_size: Sequence[int], device_size: Sequence[int],
                origin: Point = (0, 0)) -> Scale:
        """Derive a scale by comparing two measured sizes.

        Useful when nothing reports a ratio: capture the surface and the window
        it represents, and let the arithmetic find the factor.
        """
        if logical_size[0] <= 0 or logical_size[1] <= 0:
            raise ValueError(f"logical size must be positive, got {tuple(logical_size)}")
        ratio = device_size[0] / float(logical_size[0])
        # Guard against a non-integer ratio (a cropped capture, say) by trusting
        # whichever axis is more trustworthy.
        other = device_size[1] / float(logical_size[1])
        if abs(other - ratio) / max(ratio, other) > 0.02:
            ratio = max(ratio, other)
        return cls(ratio=ratio, origin=origin)

    def describe(self) -> str:
        if self.is_identity:
            return "1:1, no offset"
        parts = []
        if self.ratio != 1.0:
            parts.append(f"×{self.ratio:g}")
        if self.origin != (0, 0):
            parts.append(f"origin {self.origin[0]},{self.origin[1]}")
        return " ".join(parts) or "1:1"

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"Scale({self.describe()})"


def _unpack(box: Box) -> tuple[float, float, float, float]:
    """Accept ``(l, t, w, h)`` or ``(l, t, r, b)`` and return corners."""
    left, top, a, b = box
    # A width/height box whose third and fourth values cannot be right/bottom
    # when they are smaller than the origin is almost certainly w/h; otherwise
    # treat them as corners. Callers inside the harness always pass w/h.
    return (left, top, left + a, top + b)


def clamp_box(box: Box, size: Sequence[int]) -> Box:
    """Clip a box to a ``(width, height)`` surface, never returning negatives."""
    width, height = int(size[0]), int(size[1])
    left, top, w, h = box
    left = max(0, min(left, width))
    top = max(0, min(top, height))
    w = max(0, min(w, width - left))
    h = max(0, min(h, height - top))
    return (left, top, w, h)


def offset_box(box: Box, dx: int, dy: int) -> Box:
    left, top, w, h = box
    return (left + int(dx), top + int(dy), w, h)


def union_box(boxes: Iterable[Box]) -> Box | None:
    """Smallest box containing all inputs, or ``None`` for an empty sequence."""
    boxes = list(boxes)
    if not boxes:
        return None
    lefts = [b[0] for b in boxes]
    tops = [b[1] for b in boxes]
    rights = [b[0] + b[2] for b in boxes]
    bottoms = [b[1] + b[3] for b in boxes]
    return (min(lefts), min(tops),
            max(rights) - min(lefts), max(bottoms) - min(tops))


def iou(a: Box, b: Box) -> float:
    """Intersection over union — used to collapse overlapping changed regions."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0.0


__all__ += ["iou"]
