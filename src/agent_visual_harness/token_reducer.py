"""Token reducer — the VLM vision pre-processor.

A 60-second capture at 30 fps is 1,800 frames. Handing all of them to a
multimodal model costs real money to teach it nothing: the screen is static for
most of a session, and the frames that *do* matter are already known, because the
agent told us about them.

This module keeps exactly two kinds of frame:

* **Scene changes** — ffmpeg's ``select='gt(scene,T)'`` finds visual discontinuities.
* **Action anchors** — the exact frame at every timestamp passed to
  ``harness.log_action()``.

Everything is deduplicated, downsampled to a bounding box, renumbered, and
written next to a ``manifest.json`` that carries the timing and the action label
so the frames can be read as a story rather than as noise.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Sequence

from ._ffmpeg import FFMPEG, has_encoder, probe, run, vfr_flag
from .capture import image_size
from .errors import ProcessingError
from .processor import Processor, parse_showinfo_times
from .types import Action, Keyframe, Resolution

__all__ = ["TokenReducer", "ReducerConfig", "FRAME_FORMATS"]

FRAME_FORMATS = {
    "jpeg": {"suffix": ".jpg", "encoder_arg": ["-c:v", "mjpeg", "-q:v", "4"]},
    "jpg": {"suffix": ".jpg", "encoder_arg": ["-c:v", "mjpeg", "-q:v", "4"]},
    "webp": {"suffix": ".webp", "encoder_arg": ["-c:v", "libwebp", "-quality", "82"]},
    "png": {"suffix": ".png", "encoder_arg": ["-c:v", "png"]},
}


@dataclass(slots=True)
class ReducerConfig:
    """Knobs for :class:`TokenReducer`.

    ``scene_threshold`` is ffmpeg's scene score, 0-1. ``0.05`` is the documented
    default.

    That metric is a luminance-weighted mean absolute difference, so it is not a
    change detector in the general case: two saturated colours of similar
    luma (ffmpeg's ``red`` and ``green``, for instance) can score 0 on a cut as
    hard as a page turn. Drop the threshold towards ``0.01`` for UIs built from
    a narrow palette, and lean on ``log_action()`` anchors for anything that must
    never be missed — which is exactly what they are there for.
    """

    scene_threshold: float = 0.05
    max_size: int = 1280
    fmt: str = "webp"
    quality: int = 82
    max_frames: int | None = None
    min_gap: float = 0.30
    include_actions: bool = True
    extract_scene_frames: bool = True
    extras: dict[str, object] = field(default_factory=dict)

    def normalised(self) -> ReducerConfig:
        fmt = self.fmt.lower()
        if fmt not in FRAME_FORMATS:
            raise ProcessingError(
                f"unknown keyframe format {self.fmt!r}; choose one of "
                f"{', '.join(sorted(set(FRAME_FORMATS)))}"
            )
        if not 0.0 < self.scene_threshold < 1.0:
            raise ProcessingError(
                f"scene threshold must be between 0 and 1, got {self.scene_threshold}"
            )
        if self.max_size < 64:
            raise ProcessingError(f"max_size must be at least 64px, got {self.max_size}")
        object.__setattr__(self, "fmt", fmt)
        if fmt == "webp" and not has_encoder("libwebp"):
            object.__setattr__(self, "fmt", "jpeg")
        return self

class TokenReducer:
    """Extract a token-budgeted keyframe set from a recording."""

    def __init__(self, ffmpeg: str = FFMPEG, config: ReducerConfig | None = None) -> None:
        self.ffmpeg = ffmpeg
        self.config = (config or ReducerConfig()).normalised()
        self._info: dict[str, object] = {}

    def info_of(self, video: Path | str):
        """Probed stream metadata for ``video``, read once and remembered.

        Cached because the reducer extracts every frame from the same clip, and
        one ffprobe per frame would cost more than the extractions themselves.
        """
        key = str(video)
        if key not in self._info:
            try:
                self._info[key] = probe(key)
            except Exception:  # noqa: BLE001 - no ffprobe means no clamp, as before
                self._info[key] = None
        return self._info[key]

    def duration_of(self, video: Path | str) -> float:
        """Seconds in ``video``, or 0.0 when it cannot be read."""
        info = self.info_of(video)
        return float(getattr(info, "duration", 0.0) or 0.0) if info else 0.0

    # ------------------------------------------------------------------ scale

    def scale_filter(self, source_size: Resolution) -> str:
        """Bounding-box scale filter; never upscales a small capture.

        The target width is computed here rather than expressed as an ffmpeg
        expression like ``scale='min(1280,iw)'`` — parenthesis escaping in a
        filtergraph is a well-known trap, and we already know the geometry.
        """
        width = int(self.config.max_size)
        source_width = int(source_size[0] or 0)
        if source_width and source_width <= width:
            return "null"
        target = min(width, source_width) if source_width else width
        return f"scale={target}:-2:flags=lanczos"

    # ------------------------------------------------------------- detection

    def detect_scenes(self, video: Path | str) -> list[float]:
        """Timestamps (seconds) of every scene change above the threshold."""
        video = Path(video)
        if not video.exists():
            raise ProcessingError(f"video not found: {video}")
        times = Processor(self.ffmpeg).scene_timestamps(
            video, threshold=float(self.config.scene_threshold)
        )
        return sorted(set(round(t, 3) for t in times if t >= 0))

    # ------------------------------------------------------------- extraction

    def extract_scenes(
        self,
        video: Path | str,
        out_dir: Path | str,
        source_size: Resolution,
        limit: int | None = None,
    ) -> list[float]:
        """Write one frame per scene change; returns the timestamps in order."""
        video, out_dir = Path(video), Path(out_dir)
        suffix = FRAME_FORMATS[self.config.fmt]["suffix"]
        encoder_args = list(FRAME_FORMATS[self.config.fmt]["encoder_arg"])
        if self.config.fmt == "webp" and int(self.config.quality) != 82:
            encoder_args[-1] = str(int(self.config.quality))

        out_dir.mkdir(parents=True, exist_ok=True)
        pattern = out_dir / f"_scene_%05d{suffix}"
        for stale in out_dir.glob(f"_scene_*{suffix}"):
            stale.unlink()

        scale = self.scale_filter(source_size)
        vf = f"select='gt(scene,{float(self.config.scene_threshold):.4f})',showinfo"
        vf = f"{vf},{scale}" if scale != "null" else vf

        cmd = [
            self.ffmpeg, "-hide_banner", "-nostdin", "-y",
            # showinfo logs at INFO, so the loglevel must not be raised above it
            # or the timestamps we parse back would be missing.
            "-loglevel", "info",
            "-i", str(video),
            "-vf", vf,
            vfr_flag(), "vfr",
        ]
        if limit:
            cmd += ["-frames:v", str(int(limit))]
        cmd += encoder_args + [str(pattern)]

        proc = run(cmd, timeout=3600, check=False)
        files = sorted(out_dir.glob(f"_scene_*{suffix}"))
        if not files:
            if proc.returncode != 0:
                tail = "\n".join((proc.stderr or "").strip().splitlines()[-4:])
                raise ProcessingError(f"keyframe extraction failed:\n{tail}")
            return []

        times = parse_showinfo_times(proc.stderr)
        return _pair(files, times)

    def extract_at(
        self,
        video: Path | str,
        timestamp: float,
        out_path: Path | str,
        source_size: Resolution,
    ) -> Path | None:
        """Write the frame at exactly ``timestamp`` (returns ``None`` if absent)."""
        video, out_path = Path(video), Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        encoder_args = list(FRAME_FORMATS[self.config.fmt]["encoder_arg"])
        scale = self.scale_filter(source_size)

        # An action can be logged a moment after the last frame exists: the
        # telemetry runs on the wall clock, while the video stops the instant
        # the recorder finalises. Seeking past the end makes ffmpeg write an
        # 8-byte stub and still exit successfully, and a stub that exists is
        # worse than a missing frame — everything downstream treats it as one
        # until something finally tries to decode it.
        #
        # The clamp backs off by one frame interval rather than by an arbitrary
        # epsilon: the last frame of a 10fps clip is stamped at duration minus
        # 0.1s, so clamping to duration minus 0.05s still finds nothing.
        info = self.info_of(video)
        duration = float(getattr(info, "duration", 0.0) or 0.0) if info else 0.0
        if duration > 0:
            fps = float(getattr(info, "fps", 0.0) or 0.0) if info else 0.0
            step = 1.0 / fps if fps > 0 else 0.05
            timestamp = min(timestamp, max(0.0, duration - step))

        cmd = [
            self.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-ss", f"{max(0.0, timestamp):.3f}",
            "-i", str(video),
            "-frames:v", "1",
        ]
        if scale != "null":
            cmd += ["-vf", scale]
        cmd += encoder_args + [str(out_path)]
        run(cmd, timeout=300, check=False)
        if not out_path.exists() or out_path.stat().st_size == 0:
            # Seeking past the end of a clip yields no frame; that is not an error.
            out_path.unlink(missing_ok=True)
            return None
        try:
            image_size(out_path.read_bytes())
        except (ValueError, OSError):
            # Present but not an image. Same reasoning as an empty file: the
            # evidence does not exist, so the keyframe set should not claim it.
            out_path.unlink(missing_ok=True)
            return None
        return out_path

    # ---------------------------------------------------------------- reduce

    def reduce(
        self,
        video: Path | str,
        out_dir: Path | str,
        actions: Sequence[Action] = (),
        source_size: Resolution | None = None,
        session_dir: Path | str | None = None,
    ) -> list[Keyframe]:
        """Build the full keyframe set and write ``manifest.json``.

        Returns the keyframes in chronological order.
        """
        video, out_dir = Path(video), Path(out_dir)
        if not video.exists():
            raise ProcessingError(f"video not found: {video}")
        config = self.config
        info = probe(video)
        size: Resolution = tuple(source_size or info.size)  # type: ignore[assignment]

        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = FRAME_FORMATS[config.fmt]["suffix"]
        for stale in out_dir.glob(f"_stale_*{suffix}"):
            stale.unlink()

        staged: list[tuple[float, str, str | None, Path]] = []
        detected = 0

        # 1. Scene changes — one pass, straight into the final format and size.
        if config.extract_scene_frames:
            times = self.extract_scenes(video, out_dir, size, limit=config.max_frames)
            detected = len(times)
            for path, ts in times:
                staged.append((ts, "scene", None, path))

        # 2. Action anchors — exact frames at the timestamps the agent logged.
        if config.include_actions:
            for idx, act in enumerate(actions):
                if act.timestamp < 0:
                    continue
                target = out_dir / f"_action_{idx:05d}{suffix}"
                written = self.extract_at(video, act.timestamp, target, size)
                if written is not None:
                    staged.append((round(act.timestamp, 3), "action", act.action, written))

        # 3. Deduplicate: keep the earliest frame in any min_gap window, but an
        #    action anchor always wins over a scene frame at the same instant.
        kept = self._dedupe(staged, config.min_gap, config.max_frames)

        # 4. Renumber chronologically so the VLM reads the set as a timeline.
        keyframes: list[Keyframe] = []
        for index, (ts, reason, action, path) in enumerate(sorted(kept, key=lambda r: r[0]), start=1):
            final = out_dir / f"frame_{index:04d}{suffix}"
            if path.resolve() != final.resolve():
                shutil.move(str(path), str(final))
            frame_w, frame_h = _dimensions(final, size)
            keyframes.append(
                Keyframe(
                    path=final,
                    index=index,
                    timestamp=ts,
                    reason=reason,
                    action=action,
                    width=frame_w,
                    height=frame_h,
                )
            )

        # 5. Drop scratch files and stale frames from a previous run.
        for scratch in out_dir.glob(f"_*{suffix}"):
            scratch.unlink()
        for scratch in out_dir.glob(f"frame_*{suffix}"):
            if scratch not in {k.path for k in keyframes}:
                scratch.unlink()

        manifest = out_dir / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "version": 1,
                    "generator": "agent_visual_harness",
                    "source_video": str(video),
                    "session_dir": str(session_dir) if session_dir else None,
                    "source_size": {"width": info.width, "height": info.height},
                    "duration": round(info.duration, 3),
                    "raw_frame_estimate": int(info.duration * (info.fps or config_scene_fps(video))),
                    "config": {
                        "scene_threshold": config.scene_threshold,
                        "max_size": config.max_size,
                        "format": config.fmt,
                        "quality": config.quality,
                        "min_gap": config.min_gap,
                        "max_frames": config.max_frames,
                    },
                    "scene_frames_detected": detected,
                    "action_anchors": len(actions) if config.include_actions else 0,
                    "keyframes_kept": len(keyframes),
                    "reduction_pct": (
                        round(100.0 * (1 - len(keyframes) / max(1, int(info.duration * max(1.0, info.fps)))), 2)
                    ),
                    "keyframes": [k.to_dict() for k in keyframes],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return keyframes

    # --------------------------------------------------------------- helpers

    @staticmethod
    def _dedupe(
        staged: Sequence[tuple[float, str, str | None, Path]],
        min_gap: float,
        max_frames: int | None,
    ) -> list[tuple[float, str, str | None, Path]]:
        """Collapse scene frames that sit inside ``min_gap`` of a kept frame.

        Action anchors are **never** dropped. That is the contract the module
        exists to keep: the agent told us those instants mattered, so they are
        kept even when three clicks land inside the same 300 ms window and the
        frames are near-identical. ``min_gap`` filters what the scene detector
        guessed, not what the agent reported.
        """
        anchors = [row for row in staged if row[1] == "action"]
        scenes = sorted(
            (row for row in staged if row[1] != "action"), key=lambda r: r[0]
        )

        kept: list[tuple[float, str, str | None, Path]] = []
        taken = [row[0] for row in sorted(anchors, key=lambda r: r[0])]

        for scene in scenes:
            if any(abs(scene[0] - t) < min_gap for t in taken):
                continue  # too close to an anchor we already kept
            if kept and (scene[0] - kept[-1][0]) < min_gap:
                continue  # too close to another scene frame
            kept.append(scene)
            taken.append(scene[0])
            if max_frames and len(kept) + len(anchors) >= max_frames:
                break

        return sorted(anchors + kept, key=lambda r: r[0])


def _pair(files: Sequence[Path], times: Sequence[float]) -> list[tuple[Path, float]]:
    """Zip extracted frames with their decoder timestamps.

    ``showinfo`` counts and image2 filenames are produced by the same filtered
    frame stream, so index N of one always matches index N of the other. If a
    build ever disagrees we respace the known timestamps evenly across the frames
    we actually got, rather than silently emitting wrong ones.
    """
    if not files:
        return []
    if len(times) == len(files):
        return list(zip(files, times, strict=True))
    if not times:
        return [(f, float(i)) for i, f in enumerate(files)]
    first, last = times[0], times[-1]
    if len(files) == 1:
        return [(files[0], first)]
    step = (last - first) / (len(files) - 1)
    return [(f, round(first + i * step, 3)) for i, f in enumerate(files)]


def _dimensions(path: Path, fallback: Resolution) -> tuple[int, int]:
    """Read a keyframe's pixel size straight from its header.

    The manifest has to describe the real frame size, and spawning ffprobe once
    per frame would dominate the runtime of a 40-frame extract. All three
    formats we emit have their dimensions in the first few bytes.
    """
    try:
        with path.open("rb") as handle:
            data = handle.read(65536)
    except OSError:  # pragma: no cover
        return fallback

    suffix = path.suffix.lower()
    if suffix == ".png" and data[:8] == b"\x89PNG\r\n\x1a\n":
        width = int.from_bytes(data[16:20], "big")
        height = int.from_bytes(data[20:24], "big")
        return (width, height) if width and height else fallback
    if suffix in (".jpg", ".jpeg"):
        size = _jpeg_dimensions(data)
        return size if size else fallback
    if suffix == ".webp":
        size = _webp_dimensions(data)
        return size if size else fallback
    return fallback


def _jpeg_dimensions(data: bytes) -> tuple[int, int] | None:
    """Walk JPEG segments to the first SOFn marker, which carries the size."""
    offset = 2
    end = len(data)
    while offset + 9 < end:
        if data[offset] != 0xFF:
            offset += 1
            continue
        marker = data[offset + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            offset += 2
            continue
        if marker in (0xD9, 0xDA):  # end of image / start of scan: no SOFn found
            return None
        length = int.from_bytes(data[offset + 2 : offset + 4], "big")
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            height = int.from_bytes(data[offset + 5 : offset + 7], "big")
            width = int.from_bytes(data[offset + 7 : offset + 9], "big")
            return (width, height)
        offset += 2 + length
    return None


def _webp_dimensions(data: bytes) -> tuple[int, int] | None:
    if data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        return None
    chunk = data[12:16]
    if chunk == b"VP8X" and len(data) >= 30:
        width = int.from_bytes(data[24:27], "little") + 1
        height = int.from_bytes(data[27:30], "little") + 1
        return (width, height)
    if chunk == b"VP8 " and len(data) >= 30:
        width = int.from_bytes(data[26:28], "little") & 0x3FFF
        height = int.from_bytes(data[28:30], "little") & 0x3FFF
        return (width, height)
    if chunk == b"VP8L" and len(data) >= 25 and data[20] == 0x2F:
        bits = int.from_bytes(data[21:25], "little")
        return ((bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1)
    return None


def config_scene_fps(video: Path | str) -> float:
    """Frames-per-second to use when estimating raw frame count."""
    try:
        return probe(video).fps or 24.0
    except Exception:
        return 24.0
