"""Post-processing engine — turn a raw capture into review-ready artifacts.

Three jobs, all CPU-only:

* **Video** — CRF-tuned H.264 (MP4) and VP9 (WebM). Code and monospace text
  survive CRF 23 in a way they do not at CRF 30, so the default is deliberately
  conservative; it is still an order of magnitude smaller than a raw capture.
* **GIF** — two-pass ``palettegen`` → ``paletteuse``. The single-pass GIF
  encoder produces banding on dark terminal themes; ``stats_mode=diff`` with a
  generated palette is what makes a 40 MB recording become a 2 MB PR asset.
* **Overlays** — replay the telemetry timeline and burn click rings and
  crosshairs onto the video, so a reviewer can see *which* click caused a
  regression instead of guessing from a PNG.
"""

from __future__ import annotations

import re
from pathlib import Path
from collections.abc import Iterable, Sequence

from ._ffmpeg import FFMPEG, has_encoder, probe, run
from .errors import ProcessingError
from .types import Action, MediaInfo, Resolution

__all__ = [
    "Processor",
    "VALID_DITHERS",
    "VIDEO_CONTAINERS",
    "parse_showinfo_times",
    "estimate_token_saving",
]

VALID_DITHERS = (
    "none",
    "sierra2",
    "sierra2_4a",
    "bayer",
    "bayer:bayer_scale=2",
    "bayer:bayer_scale=3",
    "bayer:bayer_scale=4",
    "bayer:bayer_scale=5",
    "floyd_steinberg",
)

VIDEO_CONTAINERS = {
    ".mp4": "h264",
    ".m4v": "h264",
    ".mov": "h264",
    ".webm": "vp9",
    ".mkv": "h264",
}


class Processor:
    """Stateless post-processing helper bound to one ffmpeg pair."""

    def __init__(
        self,
        ffmpeg: str = FFMPEG,
        crf: int = 23,
        preset: str = "medium",
        gif_max_colors: int = 128,
        gif_dither: str = "bayer",
        gif_fps: int = 15,
        gif_width: int | None = None,
        pixel_format: str = "yuv420p",
        ring_color: str = "0xC8FF3D",
        ring_radius: int = 18,
        ring_hold: float = 0.45,
        overlay_clicks: bool = False,
    ) -> None:
        self.ffmpeg = ffmpeg
        self.crf = int(crf)
        self.preset = preset
        self.gif_max_colors = int(gif_max_colors)
        self.gif_dither = self._check_dither(gif_dither)
        self.gif_fps = int(gif_fps)
        self.gif_width = gif_width
        self.pixel_format = pixel_format
        self.ring_color = ring_color
        self.ring_radius = int(ring_radius)
        self.ring_hold = float(ring_hold)
        self.overlay_clicks = overlay_clicks

    @staticmethod
    def _check_dither(dither: str) -> str:
        if dither not in VALID_DITHERS:
            raise ValueError(f"unknown dither {dither!r}; choose one of {', '.join(VALID_DITHERS)}")
        return dither

    # ------------------------------------------------------------------ video

    def to_video(
        self,
        source: Path | str,
        output: Path | str,
        crf: int | None = None,
        preset: str | None = None,
        faststart: bool | None = None,
        scale: int | None = None,
    ) -> Path:
        """Encode ``source`` to H.264/MP4 (or VP9/WebM based on the suffix)."""
        source, output = Path(source), Path(output)
        if not source.exists():
            raise ProcessingError(f"source video not found: {source}")
        output.parent.mkdir(parents=True, exist_ok=True)
        suffix = output.suffix.lower()
        codec = VIDEO_CONTAINERS.get(suffix)
        if codec is None:
            raise ProcessingError(
                f"unsupported output container {suffix!r}; expected one of "
                f"{', '.join(sorted(VIDEO_CONTAINERS))}"
            )
        effective_crf = self.crf if crf is None else int(crf)
        effective_preset = self.preset if preset is None else preset
        use_faststart = (suffix in (".mp4", ".m4v", ".mov")) if faststart is None else faststart

        if codec == "vp9" and not has_encoder("libvpx-vp9"):
            raise ProcessingError(
                "this ffmpeg build has no libvpx-vp9 encoder; use .mp4 for H.264 output"
            )

        cmd = [self.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source)]
        filters: list[str] = []
        if scale:
            filters.append(f"scale={int(scale)}:-2:flags=lanczos")
        if filters:
            cmd += ["-vf", ",".join(filters)]
        cmd += ["-c:v", "libx264" if codec == "h264" else "libvpx-vp9"]
        cmd += ["-crf", str(effective_crf)]
        if codec == "h264":
            cmd += ["-preset", effective_preset, "-pix_fmt", self.pixel_format]
            if use_faststart:
                cmd += ["-movflags", "+faststart"]
        else:
            cmd += ["-b:v", "0", "-row-mt", "1", "-deadline", "good", "-cpu-used", "2"]
        cmd += ["-an", str(output)]

        run(cmd, timeout=1800)
        self._assert_written(output)
        return output

    # -------------------------------------------------------------------- gif

    def to_gif(
        self,
        source: Path | str,
        output: Path | str,
        max_colors: int | None = None,
        dither: str | None = None,
        fps: int | None = None,
        width: int | None = None,
    ) -> Path:
        """Two-pass palette GIF, sized to survive a GitHub PR description."""
        source, output = Path(source), Path(output)
        if not source.exists():
            raise ProcessingError(f"source video not found: {source}")
        output.parent.mkdir(parents=True, exist_ok=True)

        colors = int(max_colors if max_colors is not None else self.gif_max_colors)
        if not 2 <= colors <= 256:
            raise ProcessingError(f"gif max_colors must be 2-256, got {colors}")
        dither = self._check_dither(dither if dither is not None else self.gif_dither)
        gif_fps = int(fps if fps is not None else self.gif_fps)
        target_width = width if width is not None else self.gif_width

        scale = f"scale={target_width}:-2:flags=lanczos" if target_width else "scale=trunc(iw/2)*2:-1"
        pre = f"fps={gif_fps},{scale}"

        palette = output.with_name(f"{output.stem}.palette.png")
        try:
            # Pass 1 — analyse the whole clip and build an optimised palette.
            run(
                [
                    self.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(source),
                    "-vf", f"{pre},palettegen=max_colors={colors}:stats_mode=diff",
                    "-frames:v", "1",
                    "-f", "image2", str(palette),
                ],
                timeout=1800,
            )
            if not palette.exists() or palette.stat().st_size == 0:
                raise ProcessingError("palettegen produced no palette")

            # Pass 2 — re-read the source and apply it with dithering.
            run(
                [
                    self.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(source),
                    "-i", str(palette),
                    "-lavfi", f"{pre}[x];[x][1:v]paletteuse=dither={dither}",
                    "-loop", "0",
                    str(output),
                ],
                timeout=1800,
            )
        finally:
            palette.unlink(missing_ok=True)

        self._assert_written(output)
        return output

    # --------------------------------------------------------------- overlays

    def build_overlay_filter(
        self,
        actions: Sequence[Action],
        source_size: Resolution,
        target_size: Resolution,
        ring_color: str | None = None,
        ring_radius: int | None = None,
        hold: float | None = None,
    ) -> str:
        """Build a ``-vf`` chain that draws a ring at every positioned action."""
        color = ring_color or self.ring_color
        radius = int(ring_radius if ring_radius is not None else self.ring_radius)
        span = float(hold if hold is not None else self.ring_hold)
        sw, sh = source_size
        tw, th = target_size
        if sw <= 0 or sh <= 0:
            return ""
        scale_x, scale_y = tw / sw, th / sh

        parts: list[str] = []
        for act in actions:
            if not act.has_position:
                continue  # scroll / keyboard events have no point to mark
            cx = act.x * scale_x  # type: ignore[operator]
            cy = act.y * scale_y  # type: ignore[operator]
            # Quoting the expression keeps the commas from splitting the filtergraph.
            enable = (
                f"between(t,{max(0.0, act.timestamp):.3f},"
                f"{max(0.0, act.timestamp) + span:.3f})"
            )
            x, y = cx - radius, cy - radius
            parts.append(
                f"drawbox=x={max(0.0, x):.1f}:y={max(0.0, y):.1f}"
                f":w={radius * 2}:h={radius * 2}:color={color}:t=3:enable='{enable}'"
            )
            arm = radius + 10
            parts.append(
                f"drawbox=x={cx - arm:.1f}:y={cy - 1:.1f}:w={arm * 2}:h=2"
                f":color={color}@0.55:enable='{enable}'"
            )
            parts.append(
                f"drawbox=x={cx - 1:.1f}:y={cy - arm:.1f}:w=2:h={arm * 2}"
                f":color={color}@0.55:enable='{enable}'"
            )
        return ",".join(parts)

    def apply_overlays(
        self,
        source: Path | str,
        output: Path | str,
        actions: Sequence[Action],
        source_size: Resolution | None = None,
        crf: int | None = None,
    ) -> Path:
        """Re-encode ``source`` to ``output`` with click rings burned in."""
        source, output = Path(source), Path(output)
        if not source.exists():
            raise ProcessingError(f"source video not found: {source}")
        output.parent.mkdir(parents=True, exist_ok=True)
        info = probe(source)
        src_size: Resolution = tuple(source_size or info.size)  # type: ignore[assignment]

        cmd = [self.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source)]
        filters: list[str] = []
        target: Resolution = src_size
        if info.size != src_size:
            filters.append(f"scale={info.width}x{info.height}:flags=lanczos")
            target = info.size
        overlay = self.build_overlay_filter(actions, src_size, target)
        if overlay:
            filters.append(overlay)
        if filters:
            cmd += ["-vf", ",".join(filters)]
        cmd += [
            "-c:v", "libx264",
            "-crf", str(self.crf if crf is None else crf),
            "-preset", self.preset,
            "-pix_fmt", self.pixel_format,
            "-movflags", "+faststart",
            "-an",
            str(output),
        ]
        run(cmd, timeout=1800)
        self._assert_written(output)
        return output

    def export_video(
        self,
        source: Path | str,
        output: Path | str,
        actions: Sequence[Action] = (),
        source_size: Resolution | None = None,
        crf: int | None = None,
        with_overlays: bool | None = None,
    ) -> Path:
        """One call for the common path: overlays (optional) then final encode."""
        want_overlays = self.overlay_clicks if with_overlays is None else bool(with_overlays)
        positioned = [a for a in actions if a.has_position]
        if want_overlays and positioned:
            return self.apply_overlays(source, output, positioned, source_size=source_size, crf=crf)
        return self.to_video(source, output, crf=crf)

    # ------------------------------------------------------------- inspection

    def media_info(self, path: Path | str) -> MediaInfo:
        return probe(path)

    def frame_at(
        self,
        source: Path | str,
        timestamp: float,
        output: Path | str,
        scale: int | None = None,
        quality: int | None = None,
    ) -> Path:
        """Extract the single frame at ``timestamp`` (fast seek, before ``-i``)."""
        source, output = Path(source), Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            self.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{max(0.0, timestamp):.3f}",
            "-i", str(source),
            "-frames:v", "1",
        ]
        if scale:
            cmd += ["-vf", f"scale={int(scale)}:-2:flags=lanczos"]
        if quality is not None:
            cmd += ["-q:v", str(int(quality))]
        cmd += [str(output)]
        run(cmd, timeout=300)
        if not output.exists() or output.stat().st_size == 0:
            raise ProcessingError(f"no frame decoded at t={timestamp:.3f}s from {source.name}")
        return output

    def scene_timestamps(
        self,
        source: Path | str,
        threshold: float = 0.05,
        max_frames: int | None = None,
    ) -> list[float]:
        """Return the PTS (seconds) of every scene change above ``threshold``.

        Uses ffmpeg's ``select='gt(scene,T)'`` plus ``showinfo`` so timestamps
        come straight from the decoder instead of being re-derived from indices.
        """
        source = Path(source)
        # Commas inside the quoted expression must NOT be backslash-escaped:
        # quoting already shields them from the filtergraph splitter, and an
        # escape would reach the expression parser as a literal backslash.
        vf = f"select='gt(scene,{float(threshold):.4f})',showinfo"
        cmd = [
            self.ffmpeg, "-hide_banner", "-nostats", "-i", str(source),
            "-vf", vf,
        ]
        if max_frames:
            cmd += ["-frames:v", str(int(max_frames))]
        cmd += ["-an", "-f", "null", "-"]
        proc = run(cmd, timeout=3600, check=False)
        if proc.returncode != 0 and not parse_showinfo_times(proc.stderr):
            tail = "\n".join((proc.stderr or "").strip().splitlines()[-4:])
            raise ProcessingError(f"scene detection failed for {Path(source).name}:\n{tail}")
        return parse_showinfo_times(proc.stderr)

    # --------------------------------------------------------------- helpers

    @staticmethod
    def _assert_written(path: Path) -> None:
        if not path.exists():
            raise ProcessingError(f"ffmpeg reported success but {path} was not created")
        if path.stat().st_size == 0:
            raise ProcessingError(f"ffmpeg produced an empty file at {path}")


def parse_showinfo_times(stderr: str | None) -> list[float]:
    """Pull ``pts_time:`` values out of ffmpeg's ``showinfo`` filter log."""
    if not stderr:
        return []
    times: list[float] = []
    for match in re.finditer(r"pts_time:(-?[\d.]+|N/?A)", stderr):
        token = match.group(1)
        if token in ("N/A", "NA"):
            continue
        try:
            times.append(float(token))
        except ValueError:
            continue
    return times


def estimate_token_saving(
    duration: float,
    fps: float,
    action_count: int,
    scene_per_minute: float,
    tokens_per_image: float = 1100,
    dedup_factor: float = 0.8,
) -> dict[str, float]:
    """Mirror of the numbers on the website's calculator.

    Included so the README, the CLI and the marketing page cannot drift apart.
    These are estimates, not measurements.
    """
    raw_frames = max(0.0, duration * fps)
    scene_frames = scene_per_minute * duration / 60.0
    kept = min(raw_frames, max(1.0, round((action_count + scene_frames) * dedup_factor + 1)))
    return {
        "raw_frames": raw_frames,
        "raw_tokens": raw_frames * tokens_per_image,
        "kept_frames": kept,
        "kept_tokens": kept * tokens_per_image,
        "reduction_pct": (1 - kept / raw_frames) * 100 if raw_frames else 0.0,
    }


def iter_sequenced(items: Iterable[str]) -> Iterable[str]:  # pragma: no cover - tiny helper
    return iter(items)
