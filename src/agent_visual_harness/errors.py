"""Exception hierarchy for Dynamo.

Every failure raised by the public API derives from :class:`HarnessError`, so an
agent can wrap a whole recording session in a single ``except`` clause without
accidentally swallowing unrelated bugs.
"""

from __future__ import annotations

__all__ = [
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


class HarnessError(Exception):
    """Base class for every error raised by ``agent_visual_harness``."""


class DependencyError(HarnessError):
    """A required external binary (ffmpeg, ffprobe, Xvfb) is missing or unusable."""


class DisplayError(HarnessError):
    """Base class for virtual-display failures."""


class DisplayInUseError(DisplayError):
    """Another X server already owns the requested display number."""


class DisplayStartTimeout(DisplayError):
    """Xvfb started but never accepted X connections within the timeout."""


class RecorderError(HarnessError):
    """Base class for recorder failures."""


class RecorderStartTimeout(RecorderError):
    """ffmpeg never produced a readable first frame within the timeout."""


class ProcessingError(HarnessError):
    """An ffmpeg post-processing step failed."""


class SessionError(HarnessError):
    """A session directory is missing, malformed, or lacks a recording."""


class VisionError(HarnessError):
    """A model backend was unreachable, or asked a question it cannot answer."""
