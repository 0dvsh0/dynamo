"""Static capture: full-surface grabs and region snips.

The CTO's first category — static UI verification — splits into two very different
asks. "Screenshot the whole thing" is easy. "Screenshot just the element that
regressed" is what actually saves tokens, and it is the one nobody has a good
primitive for: the tools that do support it (`screencapture -R`, Playwright's
clip) are single-platform or already inside a browser session.

Everything here returns encoded image bytes, so results can be handed straight to
a multimodal model, to :mod:`agent_visual_harness.ocr`, or to
:mod:`agent_visual_harness.diff`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable, Sequence

from ._ffmpeg import FFMPEG, run_bytes
from .coordinates import Box, Scale, clamp_box
from .errors import DependencyError, HarnessError, ProcessingError

__all__ = ["Surface", "SurfaceInfo", "image_size", "have_image_size_tools"]

_FORMATS = {"png": "png", "jpeg": "mjpeg", "jpg": "mjpeg", "webp": "libwebp"}


@dataclass(slots=True)
class SurfaceInfo:
    """What a capture surface actually is, as opposed to what you asked for."""

    width: int
    height: int
    format: str = "png"

    @property
    def size(self) -> tuple[int, int]:
        return (self.width, self.height)

    def to_dict(self) -> dict:
        return {"width": self.width, "height": self.height, "format": self.format}


@dataclass(slots=True)
class Surface:
    """A grabbable X display, with optional region and scaling.

    Use :meth:`from_display` rather than constructing directly; the constructor
    is public for tests and for surfaces that are not X11.
    """

    display: str
    size: tuple[int, int]
    scale: Scale | None = None
    _format: str = "png"

    # ------------------------------------------------------------- builders

    @classmethod
    def from_display(cls, display: str, size: Sequence[int], **kwargs) -> Surface:
        return cls(display=str(display), size=(int(size[0]), int(size[1])), **kwargs)

    # ------------------------------------------------------------- captures

    def grab(self, fmt: str = "png", quality: int | None = None) -> bytes:
        """Capture the whole surface."""
        codec = _FORMATS.get(fmt.lower())
        if codec is None:
            raise ValueError(f"unsupported format {fmt!r}; use png, jpeg or webp")
        args = [
            FFMPEG, "-hide_banner", "-loglevel", "error",
            "-f", "x11grab", "-video_size", f"{self.size[0]}x{self.size[1]}",
            "-i", self.display, "-frames:v", "1",
        ]
        if codec == "libwebp":
            if quality is not None:
                args += ["-quality", str(int(quality))]
        elif codec == "mjpeg":
            args += ["-q:v", str(int(quality) if quality is not None else 2)]
        args += ["-c:v", codec, "-f", "image2pipe", "-"]
        data = run_bytes(args, timeout=60)
        if len(data) < 128:
            raise HarnessError(
                f"capture from {self.display} returned only {len(data)} bytes; "
                f"is the display up and sized {self.size[0]}x{self.size[1]}?"
            )
        return data

    def snip(self, box: Sequence[int], fmt: str = "png", quality: int | None = None) -> bytes:
        """Capture one region of the surface.

        The crop happens in ffmpeg, before encoding, so a 200×60 snip costs the
        same as a 40 KB image rather than a 4 MB one downscaled afterwards.
        """
        region = clamp_box(tuple(int(v) for v in box), self.size)  # type: ignore[arg-type]
        left, top, width, height = region
        if width == 0 or height == 0:
            raise ValueError(f"region {tuple(box)} is empty after clamping to {self.size}")
        codec = _FORMATS.get(fmt.lower())
        if codec is None:
            raise ValueError(f"unsupported format {fmt!r}; use png, jpeg or webp")
        args = [
            FFMPEG, "-hide_banner", "-loglevel", "error",
            "-f", "x11grab", "-video_size", f"{self.size[0]}x{self.size[1]}",
            "-i", self.display,
            "-vf", f"crop={width}:{height}:{left}:{top}",
            "-frames:v", "1",
        ]
        if codec == "libwebp":
            if quality is not None:
                args += ["-quality", str(int(quality))]
        elif codec == "mjpeg":
            args += ["-q:v", str(int(quality) if quality is not None else 2)]
        args += ["-c:v", codec, "-f", "image2pipe", "-"]
        data = run_bytes(args, timeout=60)
        if len(data) < 64:
            raise HarnessError(f"snip {tuple(box)} produced no data")
        return data

    def snip_scaled(self, box: Sequence[int], scale: Scale | None = None,
                    fmt: str = "png", quality: int | None = None) -> bytes:
        """Capture a region given in *logical* coordinates.

        This is the call an agent should make: it reads a box off a screenshot,
        and the surface knows whether that screenshot was HiDPI.
        """
        effective = scale if scale is not None else self.scale
        region = _unpack(int(b) for b in box)
        if effective is not None:
            region = effective.box_to_device(region)  # type: ignore[assignment]
        return self.snip(region, fmt=fmt, quality=quality)  # type: ignore[arg-type]

    def save(self, path: Path | str, fmt: str | None = None, quality: int | None = None) -> Path:
        """Capture and write to disk, choosing the codec from the suffix."""
        path = Path(path)
        fmt = fmt or {".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg",
                      ".webp": "webp"}.get(path.suffix.lower(), "png")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.grab(fmt=fmt, quality=quality))
        return path

    def save_snip(self, path: Path | str, box: Sequence[int], fmt: str | None = None,
                  quality: int | None = None) -> Path:
        path = Path(path)
        fmt = fmt or {".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg",
                      ".webp": "webp"}.get(path.suffix.lower(), "png")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.snip(box, fmt=fmt, quality=quality))
        return path

    def info(self) -> SurfaceInfo:
        return SurfaceInfo(self.size[0], self.size[1], self._format)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Surface {self.display} {self.size[0]}x{self.size[1]} scale={self.scale}>"


def _unpack(box: Sequence[int] | Iterable[int]) -> Box:
    """Normalise any iterable of numbers to ``(x, y, w, h)``.

    The form an agent reads off a screenshot is always x/y/w/h, so there is no
    ambiguity to resolve here — but callers hand us lists, tuples and generators
    interchangeably, and a generator cannot be sliced.
    """
    values = [int(v) for v in box]
    left, top, width, height = values[:4]
    return (left, top, width, height)


def have_image_size_tools() -> bool:
    """Whether an image-inspection backend is available (ImageMagick or ffprobe)."""
    import shutil

    return bool(shutil.which("identify") or shutil.which("ffprobe"))


def image_size(data: bytes) -> tuple[int, int]:
    """Read an image's dimensions without decoding the pixels.

    Header parsing for PNG, JPEG and WebP: a full decode to answer "how big is
    this" costs orders of magnitude more, and the reducer already depends on
    being able to do this cheaply.
    """
    if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 24:
        return (int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big"))
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        chunk = data[12:16]
        if chunk == b"VP8X" and len(data) >= 30:
            return (int.from_bytes(data[24:27], "little") + 1,
                    int.from_bytes(data[27:30], "little") + 1)
        if chunk == b"VP8 " and len(data) >= 30:
            return (int.from_bytes(data[26:28], "little") & 0x3FFF,
                    int.from_bytes(data[28:30], "little") & 0x3FFF)
        if chunk == b"VP8L" and len(data) >= 25 and data[20] == 0x2F:
            bits = int.from_bytes(data[21:25], "little")
            return ((bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1)
    if data[:2] == b"\xff\xd8":
        offset, end = 2, len(data)
        while offset + 9 < end:
            if data[offset] != 0xFF:
                offset += 1
                continue
            marker = data[offset + 1]
            if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
                offset += 2
                continue
            if marker in (0xD9, 0xDA):
                break
            length = int.from_bytes(data[offset + 2 : offset + 4], "big")
            if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                          0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                return (int.from_bytes(data[offset + 7 : offset + 9], "big"),
                        int.from_bytes(data[offset + 5 : offset + 7], "big"))
            offset += 2 + length
    if data[:6] in (b"GIF87a", b"GIF89a") and len(data) >= 10:
        return (int.from_bytes(data[6:8], "little"), int.from_bytes(data[8:10], "little"))
    raise ValueError("unrecognised image format (expected PNG, JPEG, WebP or GIF)")


def save_image(path: Path | str, data: bytes) -> Path:
    """Write encoded image bytes, creating parents."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def convert(data: bytes, fmt: str = "png", quality: int | None = None) -> bytes:
    """Re-encode image bytes to another format via ffmpeg."""
    codec = _FORMATS.get(fmt.lower())
    if codec is None:
        raise ValueError(f"unsupported format {fmt!r}; use png, jpeg or webp")
    args = [FFMPEG, "-hide_banner", "-loglevel", "error", "-f", "image2pipe", "-i", "-"]
    if fmt.lower() == "webp" and quality is not None:
        args += ["-quality", str(int(quality))]
    elif fmt.lower() in ("jpeg", "jpg") and quality is not None:
        args += ["-q:v", str(max(1, min(31, int(quality))))]
    args += ["-c:v", codec, "-f", "image2pipe", "-"]
    return run_bytes(args, timeout=60, input_bytes=data)


def compose(parts: Sequence[bytes]) -> bytes:
    """Stack several images vertically into one PNG.

    Handy for building a "contact sheet" of a run without a display attached.
    """
    if not parts:
        raise ValueError("compose needs at least one image")
    if len(parts) == 1:
        return parts[0]
    import subprocess

    args: list[str] = []
    for _ in parts:
        args += ["-i", "pipe:0"]
    chain = "".join(f"[{i}:v]" for i in range(len(parts)))
    args += [
        "-filter_complex", f"{chain}vstack=inputs={len(parts)}:format=rgb24[v]",
        "-map", "[v]", "-frames:v", "1", "-c:v", "png", "-f", "image2pipe", "-",
    ]
    proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    out, err = proc.communicate(b"".join(parts), timeout=120)
    if proc.returncode != 0:
        tail = (err or b"").decode("utf-8", "replace").strip().splitlines()[-3:]
        raise ProcessingError("could not compose images:\n" + "\n".join(tail))
    return out


def require_image_tool(name: str, hint: str = "") -> str:
    """Resolve an external image tool with a useful error when it is missing."""
    import shutil

    found = shutil.which(name)
    if found is None:
        raise DependencyError(
            f"required tool {name!r} was not found on PATH.{' ' + hint if hint else ''}\n"
            f"Install it with: apt-get install -y {name}"
        )
    return found


def _unused(*_: object) -> None:  # pragma: no cover
    return None
