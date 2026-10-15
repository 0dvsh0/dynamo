"""Dynamo — the headless AI Visual & Video Harness for coding agents.

Typical use::

    from agent_visual_harness import VisualHarness

    with VisualHarness(display=":99", resolution=(1920, 1080), record=True) as h:
        h.run_on_display(["python", "agent.py"])
        h.log_action("click", x=500, y=300)

    artifacts = h.export(
        video_path="output/repro.mp4",
        gif_path="output/demo.gif",
        vlm_keyframes_dir="output/vlm_frames/",
    )
    print(f"{len(artifacts.keyframes)} frames for VLM context")

Every layer is independently usable:

* :class:`~agent_visual_harness.display.VirtualDisplay` — Xvfb lifecycle
* :class:`~agent_visual_harness.recorder.Recorder` — ffmpeg x11grab daemon
* :class:`~agent_visual_harness.processor.Processor` — MP4 / WebM / GIF / overlays
* :class:`~agent_visual_harness.token_reducer.TokenReducer` — VLM keyframes
* :class:`~agent_visual_harness.harness.VisualHarness` — the SDK that wires them
* ``dynamo`` — CLI parity for shell-driven agents and CI
"""

from __future__ import annotations

__version__ = "1.1.0"

from .capture import Surface, SurfaceInfo, image_size
from .coordinates import Box, Point, Scale
from .diff import DiffResult, ChangedRegion, compare, compare_files
from .display import VirtualDisplay, display_socket_path, ensure_x11_socket_dir, is_display_up
from .errors import (
    DependencyError,
    DisplayError,
    DisplayInUseError,
    DisplayStartTimeout,
    HarnessError,
    ProcessingError,
    RecorderError,
    RecorderStartTimeout,
    SessionError,
    VisionError,
)
from .grounding import Grounding, Target, annotate, annotate_boxes, resolve, resolve_box
from .harness import VisualHarness
from .ocr import (
    CONFUSABLE_GROUPS,
    OCRResult,
    find,
    fold,
    have_tesseract,
    read_image,
    read_region,
)
from .processor import Processor
from .recorder import Recorder
from .session import Session
from .token_reducer import ReducerConfig, TokenReducer
from .types import Action, Artifacts, Keyframe, MediaInfo, parse_resolution
from .vision import (
    DEFAULT_LAYA_ENDPOINT,
    DEFAULT_OLLAMA_ENDPOINT,
    DEFAULT_VLM_MODEL,
    MAX_STATE_CHARS,
    TOKENS_PER_MEGAPIXEL,
    Answer,
    Backend,
    Decision,
    Frame,
    LayaBackend,
    OllamaBackend,
    Question,
    RemoteBackend,
    Report,
    StubBackend,
    Triage,
    crop_image,
    estimate_tokens,
    frames_from_manifest,
    regions_between,
    union_box,
)

__all__ = [
    "__version__",
    # SDK
    "VisualHarness",
    # layers
    "VirtualDisplay",
    "Recorder",
    "Processor",
    "TokenReducer",
    "ReducerConfig",
    "Session",
    # static capture, OCR, diffing, grounding
    "Surface",
    "SurfaceInfo",
    "image_size",
    "read_image",
    "read_region",
    "find",
    "fold",
    "have_tesseract",
    "CONFUSABLE_GROUPS",
    "OCRResult",
    "compare",
    "compare_files",
    "DiffResult",
    "ChangedRegion",
    "annotate",
    "annotate_boxes",
    "resolve",
    "resolve_box",
    "Grounding",
    "Target",
    # coordinates
    "Scale",
    "Box",
    "Point",
    # model-facing triage
    "Triage",
    "Backend",
    "OllamaBackend",
    "LayaBackend",
    "RemoteBackend",
    "StubBackend",
    "Frame",
    "Answer",
    "Report",
    "Question",
    "Decision",
    "crop_image",
    "estimate_tokens",
    "union_box",
    "frames_from_manifest",
    "regions_between",
    "TOKENS_PER_MEGAPIXEL",
    "DEFAULT_VLM_MODEL",
    "DEFAULT_OLLAMA_ENDPOINT",
    "DEFAULT_LAYA_ENDPOINT",
    "MAX_STATE_CHARS",
    # value objects
    "Action",
    "Artifacts",
    "Keyframe",
    "MediaInfo",
    "parse_resolution",
    # display helpers
    "display_socket_path",
    "is_display_up",
    "ensure_x11_socket_dir",
    # errors
    "HarnessError",
    "DependencyError",
    "DisplayError",
    "DisplayInUseError",
    "DisplayStartTimeout",
    "RecorderError",
    "RecorderStartTimeout",
    "ProcessingError",
    "SessionError",
    "VisionError",
]
