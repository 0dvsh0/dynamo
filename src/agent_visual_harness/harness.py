"""The ``VisualHarness`` SDK — one context manager around all five modules.

::

    from agent_visual_harness import VisualHarness

    with VisualHarness(display=":99", resolution=(1920, 1080), record=True) as h:
        h.log_action("click", x=500, y=300)
        frame = h.take_screenshot()

    artifacts = h.export(
        video_path="output/repro.mp4",
        gif_path="output/demo.gif",
        vlm_keyframes_dir="output/vlm_frames/",
    )
    print(len(artifacts.keyframes))

Everything is optional and independently usable: the display, the recorder, the
processor and the reducer are each a plain class you can drive on its own. This
module only wires them together and gets the failure semantics right — above all,
that a crashing agent still leaves a playable recording behind.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from types import FrameType
from typing import Any
from collections.abc import Mapping, Sequence

from ._ffmpeg import probe
from .capture import Surface
from .coordinates import Scale
from .diff import DiffResult, compare
from .display import VirtualDisplay, is_display_up
from .errors import HarnessError, RecorderError
from .grounding import Grounding, annotate_boxes
from .ocr import OCRResult, read_image, read_region
from .processor import Processor
from .recorder import Recorder
from .session import Session
from .token_reducer import ReducerConfig, TokenReducer
from .types import Action, Artifacts, Keyframe, MediaInfo, Resolution, parse_resolution

__all__ = ["VisualHarness"]

DEFAULT_SESSION_DIR = "dynamo-session"


class VisualHarness:
    """Record an agent session and export review artifacts.

    Parameters
    ----------
    display:
        Display number for the managed X server, e.g. ``":99"``.
    resolution:
        Capture size. The display is created at this geometry so the recorder
        never has to guess.
    record:
        Master switch for video capture. With ``record=False`` the harness is a
        display manager plus a screenshot API — useful when you only want
        ``take_screenshot()``.
    session_dir:
        Where the raw capture and telemetry live. Everything ``export()`` needs
        comes from here, which is what makes ``dynamo record`` + ``dynamo export``
        a valid split.
    manage_display:
        Set ``False`` to attach to a display somebody else already owns.
    export_on_exit:
        Run the standard export automatically when the context exits.
    """

    def __init__(
        self,
        display: str | int = ":99",
        resolution: str | Resolution = (1920, 1080),
        depth: int = 24,
        dpi: int = 96,
        record: bool = True,
        fps: int = 24,
        session_dir: Path | str = DEFAULT_SESSION_DIR,
        manage_display: bool = True,
        adopt_existing: bool = False,
        capture_crf: int = 16,
        capture_preset: str = "veryfast",
        telemetry: bool = True,
        startup_timeout: float = 30.0,
        crf: int = 23,
        preset: str = "medium",
        gif_max_colors: int = 128,
        gif_dither: str = "bayer",
        gif_fps: int = 15,
        overlay_clicks: bool = False,
        export_on_exit: bool = False,
        auto_export_paths: dict[str, str | Path] | None = None,
        keyframes: bool = True,
        scene_threshold: float = 0.05,
        keyframe_max_size: int = 1280,
        keyframe_format: str = "webp",
    ) -> None:
        self.resolution: Resolution = parse_resolution(resolution, depth=depth)
        self.record_enabled = bool(record)
        self.manage_display = bool(manage_display)
        self.adopt_existing = bool(adopt_existing)
        self.depth = depth
        self.fps = int(fps)
        self.telemetry_enabled = bool(telemetry)
        self.startup_timeout = float(startup_timeout)
        self.export_on_exit = bool(export_on_exit)
        self.auto_export_paths = dict(auto_export_paths or {})
        self.keyframes_enabled = bool(keyframes)
        self._auto_export_kwargs = self.auto_export_paths

        self.session = Session.create(
            session_dir,
            display=str(display),
            resolution=self.resolution,
            depth=depth,
            fps=self.fps,
            crf=capture_crf,
            preset=capture_preset,
        )

        self.display_manager = VirtualDisplay(
            display=display,
            resolution=self.resolution,
            depth=depth,
            dpi=dpi,
            timeout=self.startup_timeout,
            adopt_existing=self.adopt_existing,
            log_path=self.session.display_log,
        )
        self.recorder = Recorder(
            display=display,
            size=self.resolution,
            output=self.session.raw_path,
            fps=self.fps,
            crf=capture_crf,
            preset=capture_preset,
            telemetry_path=self.session.telemetry_path if self.telemetry_enabled else None,
            startup_timeout=self.startup_timeout,
        )
        self.processor = Processor(
            crf=crf,
            preset=preset,
            gif_max_colors=gif_max_colors,
            gif_dither=gif_dither,
            gif_fps=gif_fps,
            overlay_clicks=overlay_clicks,
        )
        self.reducer = TokenReducer(
            config=ReducerConfig(
                scene_threshold=scene_threshold,
                max_size=keyframe_max_size,
                fmt=keyframe_format,
            )
        )

        self._started = False
        self._finished = False
        self._last_artifacts: Artifacts | None = None
        self._start_wall = time.monotonic()

    # ------------------------------------------------------------- properties

    @property
    def display(self) -> str:
        return self.display_manager.display

    @property
    def size(self) -> tuple[int, int]:
        return self.resolution

    @property
    def elapsed(self) -> float:
        """Seconds since the harness started recording."""
        if not self._started:
            return 0.0
        return self.recorder.elapsed

    @property
    def env(self) -> dict[str, str]:
        """Environment for child processes that should render into this harness."""
        env = dict(os.environ)
        env["DISPLAY"] = self.display
        return env

    @property
    def telemetry_path(self) -> Path:
        return self.session.telemetry_path

    @property
    def raw_path(self) -> Path:
        return self.session.raw_path

    @property
    def actions(self) -> list[Action]:
        return self.recorder.actions

    @property
    def artifacts(self) -> Artifacts | None:
        """Result of the most recent :meth:`export` call, if any."""
        return self._last_artifacts

    # -------------------------------------------------------------- lifecycle

    def start(self) -> VisualHarness:
        """Bring up the display and start capturing. Idempotent."""
        if self._started:
            return self

        if self.manage_display:
            self.display_manager.start()
        else:
            if not is_display_up(self.display_manager.display):
                raise HarnessError(
                    f"display {self.display_manager.display} is not running and "
                    f"manage_display=False. Start it yourself, or drop manage_display."
                )
            self.display_manager.adopt_existing = True
            self.display_manager.start()

        if self.record_enabled:
            try:
                self.recorder.start()
            except Exception:
                # Never leave a stray Xvfb behind because capture failed.
                if self.manage_display:
                    self.display_manager.stop()
                raise

        self._started = True
        self._start_wall = time.monotonic()
        self.session.update_meta(
            display=self.display,
            resolution=list(self.resolution),
            fps=self.fps,
            recording=self.record_enabled,
            pid=os.getpid(),
        )
        return self

    def stop(self, raise_on_error: bool = True) -> None:
        """Finalise capture and tear the display down. Idempotent.

        ``raise_on_error`` is set to ``False`` by :meth:`__exit__` when an
        exception is already propagating: a teardown failure must never mask the
        error that caused the unwind.
        """
        if not self._started or self._finished:
            return
        self._finished = True
        errors: list[BaseException] = []

        if self.recorder.is_recording:
            try:
                self.recorder.stop()
            except RecorderError as exc:
                errors.append(exc)
        if self.manage_display:
            try:
                self.display_manager.stop()
            except HarnessError as exc:
                errors.append(exc)

        self.session.mark_finished(
            duration=time.monotonic() - self._start_wall,
            recording_ok=self.session.has_recording(),
        )
        if errors and raise_on_error:
            raise errors[0]

    def __enter__(self) -> VisualHarness:
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: FrameType | None,
    ) -> bool:
        """Always finalise, then let the original exception propagate.

        This ordering is the whole point: an agent that crashes at step 3 still
        gets a playable MP4 and a complete telemetry timeline.
        """
        failing = exc is not None
        # A teardown failure must never mask the error that caused the unwind.
        self.stop(raise_on_error=not failing)

        if self.export_on_exit and not failing:
            self.export(**self._auto_export_kwargs)
        return False

    # ------------------------------------------------------------ agent hooks

    def log_action(
        self,
        action: str,
        x: int | None = None,
        y: int | None = None,
        target: str | None = None,
        **data: Any,
    ) -> Action:
        """Timestamp an agent action into ``telemetry.jsonl``.

        Returns the recorded :class:`Action`. Safe to call before
        :meth:`start` (timestamp will be ``0.0``) and after the session ends.
        """
        return self.recorder.log_action(action, x=x, y=y, target=target, **data)

    def take_screenshot(self, fmt: str = "png") -> bytes:
        """Current framebuffer as encoded image bytes."""
        return self.recorder.take_screenshot(fmt=fmt)

    def save_screenshot(self, path: Path | str, fmt: str | None = None) -> Path:
        """Write a screenshot to disk; returns the path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if fmt is None:
            fmt = {".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg", ".webp": "webp"}.get(
                path.suffix.lower(), "png"
            )
        path.write_bytes(self.take_screenshot(fmt=fmt))
        return path

    # ------------------------------------------------- inspection primitives

    @property
    def surface(self) -> Surface:
        """A grabbable view of this harness's display."""
        return Surface.from_display(self.display, self.resolution)

    def snip(self, box: Sequence[int], fmt: str = "png", quality: int | None = None) -> bytes:
        """Capture one region of the display.

        Cropping before encoding is the difference between shipping a 4 MB
        screenshot and shipping a 30 KB one, and it is the cheapest way to cut
        visual tokens: the model does not need the browser chrome to read an
        error banner.
        """
        return self.surface.snip(box, fmt=fmt, quality=quality)

    def snip_scaled(self, box: Sequence[int], scale: Scale | None = None,
                    fmt: str = "png", quality: int | None = None) -> bytes:
        """Capture a region given in logical coordinates, DPI-corrected."""
        return self.surface.snip_scaled(box, scale=scale, fmt=fmt, quality=quality)

    def save_snip(self, path: Path | str, box: Sequence[int], fmt: str | None = None) -> Path:
        path = Path(path)
        fmt = fmt or {".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg",
                      ".webp": "webp"}.get(path.suffix.lower(), "png")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.snip(box, fmt=fmt))
        return path

    def read_screen(self, **kwargs) -> OCRResult:
        """OCR the whole screen.

        The escape hatch for surfaces that will not give up their text: a
        canvas-rendered stack trace, a VNC window, an installer screen.
        """
        return read_image(self.take_screenshot(), **kwargs)

    def read_region(self, box: Sequence[int], **kwargs) -> OCRResult:
        """OCR one region, reporting boxes in full-screen coordinates."""
        return read_region(self.take_screenshot(), box, **kwargs)

    def ground(self, boxes: Sequence[Sequence[int]], **kwargs) -> Grounding:
        """Draw numbered Set-of-Marks over element boxes on a live screenshot."""
        return annotate_boxes(self.take_screenshot(), boxes, **kwargs)

    def compare(self, before: bytes, after: bytes | None = None, **kwargs) -> DiffResult:
        """Diff a stored screenshot against the screen right now.

        The common visual-regression call: capture yesterday, compare today.
        """
        return compare(before, after if after is not None else self.take_screenshot(), **kwargs)

    def run_on_display(
        self,
        cmd: Sequence[str],
        timeout: float = 600.0,
        check: bool = False,
        extra_env: Mapping[str, str] | None = None,
    ) -> subprocess.CompletedProcess:
        """Run a child process with ``DISPLAY`` pointed at this harness.

        This is how you drive Playwright, a GUI toolkit or any other renderer
        from inside the recorded display::

            h.run_on_display(["python", "my_gui_agent.py"])
            h.log_action("click", x=500, y=300)
        """
        env = {**self.env, **dict(extra_env or {})}
        return subprocess.run(  # noqa: S603 - argv list, never a shell string
            list(cmd),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=check,
        )

    # ----------------------------------------------------------------- export

    def export(
        self,
        video_path: Path | str | None = None,
        gif_path: Path | str | None = None,
        vlm_keyframes_dir: Path | str | None = None,
        webm_path: Path | str | None = None,
        crf: int | None = None,
        overlay_clicks: bool | None = None,
        scene_threshold: float | None = None,
        max_keyframe_size: int | None = None,
        keyframe_format: str | None = None,
        manifest_path: Path | str | None = None,
    ) -> Artifacts:
        """Compile the finished session into review artifacts.

        Works both inside and after the ``with`` block; if capture is still
        running it is finalised first. Every argument is optional — ask only for
        what you need and nothing is encoded.
        """
        if self.recorder.is_recording:
            self.recorder.stop()

        raw = self.session.require_recording()
        actions = self.session.actions
        info: MediaInfo = probe(raw)

        artifacts = Artifacts(
            session_dir=self.session.root,
            telemetry_path=self.session.telemetry_path,
            raw_path=raw,
            duration=info.duration,
            raw_frame_count=info.nb_frames or int(info.duration * (info.fps or self.fps)),
        )

        if video_path is not None:
            artifacts.video_path = self.processor.export_video(
                raw,
                video_path,
                actions=actions,
                source_size=self.resolution,
                crf=crf,
                with_overlays=overlay_clicks,
            )
        if webm_path is not None:
            artifacts.webm_path = self.processor.to_video(raw, webm_path, crf=crf)
        if gif_path is not None:
            artifacts.gif_path = self.processor.to_gif(raw, gif_path)

        if vlm_keyframes_dir is not None:
            config = ReducerConfig(
                scene_threshold=self.reducer.config.scene_threshold
                if scene_threshold is None
                else scene_threshold,
                max_size=max_keyframe_size or self.reducer.config.max_size,
                fmt=keyframe_format or self.reducer.config.fmt,
            )
            reducer = TokenReducer(config=config)
            keyframes = reducer.reduce(
                raw,
                vlm_keyframes_dir,
                actions=actions,
                source_size=self.resolution,
                session_dir=self.session.root,
            )
            artifacts.keyframes = keyframes
            artifacts.keyframes_dir = Path(vlm_keyframes_dir)
            artifacts.manifest_path = Path(
                manifest_path or Path(vlm_keyframes_dir) / "manifest.json"
            )

        self._last_artifacts = artifacts
        return artifacts

    # -------------------------------------------------------------- inspection

    def keyframes(self, out_dir: Path | str, **kwargs: Any) -> list[Keyframe]:
        """Run only the token reducer against the current session."""
        raw = self.session.require_recording()
        return TokenReducer(config=ReducerConfig(**kwargs)).reduce(
            raw, out_dir, actions=self.session.actions, source_size=self.resolution
        )

    def media_info(self) -> MediaInfo:
        return probe(self.session.require_recording())

    def command_for(self, *cmd: str) -> list[str]:
        """Convenience: prefix a shell-ish command with the ``dynamo`` binary."""
        return [_module_python(), "-m", "agent_visual_harness", *cmd]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        state = "running" if self._started and not self._finished else "idle"
        return (
            f"<VisualHarness {self.display} {self.resolution[0]}x{self.resolution[1]} "
            f"@{self.fps}fps {state} session={self.session.root}>"
        )


def _module_python() -> str:
    import sys

    return sys.executable or "python3"
