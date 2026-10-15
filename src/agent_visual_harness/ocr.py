"""OCR — recovering text from surfaces that refuse to give it up.

This is the capability with the shortest list of alternatives. macOS Live Text
and the Windows Snipping Tool both exist, both are interactive, and neither is
reachable from a shell. An agent that finds itself staring at a canvas-rendered
stack trace, a VNC window, a Remote Desktop session or a BIOS screen has no way
to read it except by spending a vision model on it.

Tesseract does it locally, for free, at ~100 ms a screen — and, unlike a VLM, it
returns *word-level boxes*, which is what lets you point back at the pixels you
just read.

The interesting failure mode is not "OCR is wrong", it is "OCR is confidently
wrong": a stack trace read as ``Erro: T0keenExcepion`` will send an agent down
the wrong path. :attr:`OCRResult.confidence` exists so callers can notice.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from collections.abc import Iterable, Sequence

from ._ffmpeg import FFMPEG, run_bytes
from .capture import image_size
from .coordinates import Box
from .errors import DependencyError, HarnessError

__all__ = [
    "CONFUSABLE_GROUPS",
    "OCRResult",
    "Word",
    "Line",
    "fold",
    "read_image",
    "read_region",
    "find",
    "have_tesseract",
    "TESSERACT_HINT",
]

TESSERACT_HINT = "Install it with: apt-get install -y tesseract-ocr"

#: PSM 6 = "assume a single uniform block of text", which is what a screenshot of
#: a UI panel actually is. PSM 3 (the default) assumes a scanned page and wastes
#: effort hunting for columns that are not there.
DEFAULT_PSM = 6

#: Glyphs Tesseract reliably confuses on monospace UI text, folded to one
#: representative each.
#:
#: This is measured, not theoretical. On a real payment console, Tesseract 5.3
#: returns ``7f3c9abl`` for the transaction id ``7f3c9ab1``, ``D0_ NOT HONOR``
#: for ``DO_NOT_HONOR``, and ``rule=velocity exceeded`` for
#: ``rule=velocity_exceeded`` — and neither a character whitelist nor any page
#: segmentation mode changes it. There is no knob that fixes this.
#:
#: So the recognised text is left exactly as it is, and *matching* is what
#: becomes tolerant. An agent asking "is DO_NOT_HONOR on this screen" wants a
#: yes or no; an agent pasting an error code into a report wants the characters
#: that were actually on screen. Those are different questions, and folding one
#: into the other would corrupt whichever it got wrong.
CONFUSABLE_GROUPS = (
    # Leading space: an underscore and a dash fold *to a space*, because on
    # monospace text Tesseract renders them interchangeably and a log search
    # that treats them as different characters fails on text that is right there.
    " -‐‑‒–—―_",
    "0Oo",
    "1lI|",
    "5S",
    "8B",
    "2Z",
    "6G",
    "9g",
)

_FOLD = {char: group[0] for group in CONFUSABLE_GROUPS for char in group}


def fold(text: str) -> str:
    """Normalise text for *matching only* — confusables collapsed, case gone.

    Underscores become spaces because Tesseract reads a monospace underscore as
    a space about half the time, and a log search that treats those as different
    strings fails on text that is plainly on screen.
    """
    out = "".join(_FOLD.get(char, char) for char in str(text).lower())
    return " ".join(out.split())


def have_tesseract() -> bool:
    return shutil.which("tesseract") is not None


@dataclass(slots=True)
class Word:
    """One recognised word and where it sat."""

    text: str
    box: Box  # left, top, width, height in pixels of the input image
    confidence: float

    def to_dict(self) -> dict:
        return {"text": self.text, "box": list(self.box), "confidence": self.confidence}


@dataclass(slots=True)
class Line:
    """A line of recognised text with its own box."""

    text: str
    box: Box
    confidence: float
    words: list[Word] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"text": self.text, "box": list(self.box),
                "confidence": self.confidence,
                "words": [w.to_dict() for w in self.words]}


@dataclass(slots=True)
class OCRResult:
    """Everything OCR found, plus the judgements a caller needs to trust it."""

    text: str
    lines: list[Line]
    words: list[Word]
    width: int
    height: int
    engine: str = "tesseract"
    elapsed: float = 0.0

    @property
    def confidence(self) -> float:
        """Mean word confidence, 0-100.

        Below roughly 70 the text is usually legible but the characters are a
        guess. Check it before letting an agent act on a stack trace.
        """
        if not self.words:
            return 0.0
        return sum(w.confidence for w in self.words) / len(self.words)

    @property
    def confident(self) -> bool:
        return self.confidence >= 70.0

    @property
    def line_count(self) -> int:
        return len(self.lines)

    def box_of(self, text: str, case_sensitive: bool = False,
               fuzzy: bool = False) -> Box | None:
        """Bounding box of the first line containing ``text``.

        This is the bridge back to pixels: read the error, then point at exactly
        where it was on screen. Pass ``fuzzy`` to match through the confusable
        set — see :data:`CONFUSABLE_GROUPS`.
        """
        found = self._lines_matching(text, case_sensitive, fuzzy)
        return found[0].box if found else None

    def line_of(self, text: str, case_sensitive: bool = False,
                fuzzy: bool = False) -> Line | None:
        """The matching line itself, for callers that want its words and boxes."""
        found = self._lines_matching(text, case_sensitive, fuzzy)
        return found[0] if found else None

    def _lines_matching(self, text: str, case_sensitive: bool,
                        fuzzy: bool) -> list[Line]:
        if fuzzy:
            needle = fold(text)
            return [line for line in self.lines if needle in fold(line.text)]
        needle = text if case_sensitive else text.lower()
        return [
            line for line in self.lines
            if needle in (line.text if case_sensitive else line.text.lower())
        ]

    def contains(self, text: str, case_sensitive: bool = False,
                 fuzzy: bool = False) -> bool:
        return bool(self._lines_matching(text, case_sensitive, fuzzy))

    def missed_by(self, expected: Iterable[str], case_sensitive: bool = False,
                  fuzzy: bool = False) -> list[str]:
        """Which of the strings you expected are *not* on screen.

        More useful than a recall percentage when an agent is deciding whether
        it can trust what it read: this names the tokens that went missing.
        """
        return [item for item in expected
                if not self.contains(item, case_sensitive, fuzzy)]

    def lines_matching(self, pattern: str | Iterable[str],
                       case_sensitive: bool = False,
                       fuzzy: bool = False) -> list[Line]:
        """Lines containing any of the given substrings."""
        needles = [pattern] if isinstance(pattern, str) else list(pattern)
        out: list[Line] = []
        for needle in needles:
            out.extend(self._lines_matching(needle, case_sensitive, fuzzy))
        return out

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "confidence": round(self.confidence, 1),
            "confident": self.confident,
            "width": self.width,
            "height": self.height,
            "line_count": self.line_count,
            "engine": self.engine,
            "elapsed": round(self.elapsed, 3),
            "lines": [line.to_dict() for line in self.lines],
        }

    def summary(self) -> str:
        head = self.text.strip().splitlines()[:3]
        tail = "" if self.line_count <= 3 else f"\n… {self.line_count - 3} more lines"
        return f"{chr(10).join(head)}{tail}\n(confidence {self.confidence:.0f}%)"

    def __len__(self) -> int:
        return len(self.lines)

    def __bool__(self) -> bool:
        return bool(self.text.strip())


# --------------------------------------------------------------------------- core


def _tesseract() -> str:
    found = shutil.which("tesseract")
    if found is None:
        raise DependencyError(
            f"tesseract is not installed, so on-screen text cannot be recovered.\n{TESSERACT_HINT}"
        )
    return found


def read_image(
    data: bytes,
    *,
    psm: int = DEFAULT_PSM,
    lang: str = "eng",
    whitelist: str | None = None,
    timeout: float = 60.0,
) -> OCRResult:
    """Read text out of encoded image bytes.

    Uses Tesseract's ``tsv`` output because it is the only format that carries
    word boxes *and* confidence in one pass — running twice for text and then
    boxes would double the cost for no reason.
    """
    binary = _tesseract()
    width, height = image_size(data)
    started = time.monotonic()

    with tempfile.TemporaryDirectory(prefix="dynamo-ocr-") as tmp:
        source = Path(tmp) / "in.png"
        source.write_bytes(data)
        proc = subprocess.run(
            [binary, str(source), "stdout", "-l", lang, "--psm", str(int(psm)), "tsv"],
            capture_output=True, text=True, timeout=timeout, check=False,
        )
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-4:]
        raise HarnessError("tesseract failed:\n" + "\n".join(tail))

    return _parse_tsv(proc.stdout, width, height, time.monotonic() - started, whitelist)


def _parse_tsv(tsv: str, width: int, height: int, elapsed: float,
                whitelist: str | None = None) -> OCRResult:
    lines = tsv.splitlines()
    if not lines:
        return OCRResult("", [], [], width, height, elapsed=elapsed)

    header = lines[0].split("\t")
    try:
        cols = {name: header.index(name) for name in
                ("level", "page_num", "block_num", "par_num", "line_num", "left", "top",
                 "width", "height", "conf", "text")}
    except ValueError:  # pragma: no cover - tesseract changed its output
        return OCRResult("", [], [], width, height, elapsed=elapsed)

    words: list[Word] = []
    rows_by_line: dict[tuple[int, int, int, int], list[Word]] = {}

    for row in lines[1:]:
        cells = row.split("\t")
        if len(cells) <= max(cols.values()):
            continue
        if cells[cols["level"]] != "5":  # 5 == word
            continue
        text = cells[cols["text"]].strip()
        try:
            conf = float(cells[cols["conf"]])
        except ValueError:
            continue
        if not text or conf < 0:
            continue
        word = Word(
            text=text,
            box=(int(cells[cols["left"]]), int(cells[cols["top"]]),
                 int(cells[cols["width"]]), int(cells[cols["height"]])),
            confidence=conf,
        )
        if whitelist and not all(ch in whitelist for ch in text):
            continue
        words.append(word)
        key = (int(cells[cols["page_num"]]), int(cells[cols["block_num"]]),
               int(cells[cols["par_num"]]), int(cells[cols["line_num"]]))
        rows_by_line.setdefault(key, []).append(word)

    out_lines: list[Line] = []
    for group in rows_by_line.values():
        text = " ".join(w.text for w in group)
        lefts = [w.box[0] for w in group]
        tops = [w.box[1] for w in group]
        rights = [w.box[0] + w.box[2] for w in group]
        bottoms = [w.box[1] + w.box[3] for w in group]
        out_lines.append(Line(
            text=text,
            box=(min(lefts), min(tops),
                 max(rights) - min(lefts), max(bottoms) - min(tops)),
            confidence=sum(w.confidence for w in group) / len(group),
            words=group,
        ))
    out_lines.sort(key=lambda line: (line.box[1], line.box[0]))

    return OCRResult(
        text="\n".join(line.text for line in out_lines),
        lines=out_lines, words=words, width=width, height=height, elapsed=elapsed,
    )


def read_region(
    data: bytes,
    box: Sequence[int],
    *,
    psm: int = DEFAULT_PSM,
    lang: str = "eng",
    timeout: float = 60.0,
) -> OCRResult:
    """OCR one region, and report boxes in the *original* image's coordinates.

    Cropping before recognition is the difference between reading a whole
    screenshot and reading a 200×40 error banner: it is both faster and far more
    accurate, because Tesseract's page analysis does not have to work out which
    part of the noise is text.
    """

    left, top, width, height = (int(v) for v in box[:4])
    if width <= 0 or height <= 0:
        raise ValueError(f"region {tuple(box)} is empty")
    cropped = _crop_with_ffmpeg(data, left, top, width, height)
    result = read_image(cropped, psm=psm, lang=lang, timeout=timeout)
    return _shift(result, left, top, width, height)


def _crop_with_ffmpeg(data: bytes, left: int, top: int, width: int, height: int) -> bytes:
    return run_bytes(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-f", "image2pipe", "-i", "-",
         "-vf", f"crop={width}:{height}:{left}:{top}", "-c:v", "png", "-f", "image2pipe", "-"],
        timeout=60,
        input_bytes=data,
    )


def _shift(result: OCRResult, dx: int, dy: int, crop_w: int, crop_h: int) -> OCRResult:
    """Move boxes from crop space back into full-image space."""
    def move(box: Box) -> Box:
        return (box[0] + dx, box[1] + dy, box[2], box[3])

    words = [Word(w.text, move(w.box), w.confidence) for w in result.words]
    lines = [Line(line.text, move(line.box), line.confidence,
                  [Word(w.text, move(w.box), w.confidence) for w in line.words])
             for line in result.lines]
    return OCRResult(
        text=result.text, lines=lines, words=words,
        width=crop_w + dx, height=crop_h + dy, engine=result.engine, elapsed=result.elapsed,
    )


def find(
    data: bytes,
    needle: str,
    *,
    psm: int = DEFAULT_PSM,
    lang: str = "eng",
    case_sensitive: bool = False,
    fuzzy: bool = False,
    timeout: float = 60.0,
) -> Word | None:
    """Locate a word on screen without returning the whole page of text.

    The cheapest useful question an agent can ask a screen: *is this error code
    here, and where?* With ``fuzzy=True`` the match survives the glyph
    confusions Tesseract makes on monospace text, while the returned
    :class:`Word` still carries the characters that were actually on screen.
    """
    result = read_image(data, psm=psm, lang=lang, timeout=timeout)
    target = fold(needle) if fuzzy else (
        needle if case_sensitive else needle.lower()
    )
    for word in result.words:
        candidate = fold(word.text) if fuzzy else (
            word.text if case_sensitive else word.text.lower()
        )
        if target in candidate:
            return word
    return None


def to_tsv(result: OCRResult) -> str:  # pragma: no cover - debugging aid
    return json.dumps(result.to_dict(), indent=2)


def iter_words(result: OCRResult) -> Iterable[Word]:
    return iter(result.words)


def _unused(*_: Any) -> None:  # pragma: no cover
    return None
