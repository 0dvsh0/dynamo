"""Grounding — turning "the red button" into (740, 318).

This is the CTO's hardest category, and the one that most often gets hand-waved.
Ask a vision model "click the Pay button" and it will give you a plausible
coordinate, because it is pattern-matching a description onto an image. It is
wrong often enough to be useless, and wrong *confidently*, which is worse.

Set-of-Marks fixes this without making the model better at vision. You draw a
numbered box on every interactive element, hand the model an image full of those
numbers, and it answers "12". The mapping from number to pixel is computed by the
tool, not guessed — so a wrong answer is a wrong *index*, which is checkable,
rather than a wrong coordinate, which is not.

This module owns three things:

* :class:`Target` — one grounded element and where it is.
* :func:`annotate` — render numbered marks over a screenshot.
* :func:`resolve` — map a model's answer back to coordinates, DPI-correctly.

Getting element boxes is the caller's job: CDP, Playwright, an accessibility
tree, or AT-SPI. That separation is deliberate — grounding is the part that must
not be guessed, and the source of boxes is the part that legitimately differs per
platform.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from collections.abc import Iterable, Sequence

from ._ffmpeg import FFMPEG, run_bytes
from .capture import image_size, save_image
from .coordinates import Box, Scale, clamp_box
from .errors import ProcessingError

__all__ = ["Target", "Grounding", "annotate", "annotate_boxes", "resolve", "PROMPT_TEMPLATE"]

#: Volt, matching the harness palette. Chosen for contrast against both light
#: and dark UI: a marker that vanishes against the page is worse than none.
MARK_COLOUR = "0xC8FF3D"


PROMPT_TEMPLATE = """This screenshot has interactive elements outlined and numbered 1-{count}.

Identify which numbered element you want by replying with only the number.

{question}"""


@dataclass(slots=True)
class Target:
    """One grounded element.

    ``box`` is in *image* pixels. ``index`` is the number drawn on screen and the
    only thing a model should ever have to answer with.
    """

    index: int
    box: Box
    label: str = ""
    role: str = ""
    disabled: bool = False

    @property
    def centre(self) -> tuple[int, int]:
        left, top, width, height = self.box
        return (left + width // 2, top + height // 2)

    @property
    def area(self) -> int:
        return self.box[2] * self.box[3]

    def point_at(self, fx: float = 0.5, fy: float = 0.5) -> tuple[int, int]:
        """A click point inside the element, optionally offset from its centre.

        Offsets matter for real UI: the centre of a wide button is often empty
        space, and the centre of a text input is exactly where you do not want to
        click when you are trying to focus it.
        """
        left, top, width, height = self.box
        return (int(left + width * fx), int(top + height * fy))

    def to_dict(self) -> dict:
        return {"index": self.index, "box": list(self.box), "label": self.label,
                "role": self.role, "disabled": self.disabled, "centre": list(self.centre)}

    @classmethod
    def from_dict(cls, payload: dict) -> Target:
        return cls(
            index=int(payload["index"]),
            box=tuple(int(v) for v in payload["box"]),  # type: ignore[arg-type]
            label=str(payload.get("label", "")),
            role=str(payload.get("role", "")),
            disabled=bool(payload.get("disabled", False)),
        )


@dataclass(slots=True)
class Grounding:
    """A numbered map of a screen, plus the image that shows it.

    ``image`` is the annotated PNG. ``targets`` is the ground truth the model is
    guessing at, and the only thing a caller needs to turn its answer into a
    click.

    ``skipped`` holds boxes that were dropped and why — see
    :func:`annotate_boxes`. It is not cosmetic: a silently dropped target means
    an index the model can never pick, and one that is kept but degenerate means
    a click that goes nowhere.
    """

    targets: list[Target]
    image: bytes
    width: int
    height: int
    scale: Scale | None = None
    skipped: list[dict] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.skipped is None:
            self.skipped = []

    def __len__(self) -> int:
        return len(self.targets)

    def __bool__(self) -> bool:
        return bool(self.targets)

    @property
    def indices(self) -> list[int]:
        return [t.index for t in self.targets]

    @property
    def skipped_count(self) -> int:
        return len(self.skipped)

    def by_index(self, index: int) -> Target | None:
        return next((t for t in self.targets if t.index == index), None)

    def by_label(self, label: str, exact: bool = False) -> list[Target]:
        needle = label if exact else label.lower()
        return [
            t for t in self.targets
            if (t.label if exact else t.label.lower()) == needle
        ]

    def prompt(self, question: str) -> str:
        """The question to put to a vision model alongside the annotated image."""
        return PROMPT_TEMPLATE.format(count=len(self.targets), question=question)

    def save(self, path) -> Grounding:
        save_image(path, self.image)
        return self

    def to_dict(self) -> dict:
        return {
            "count": len(self.targets),
            "width": self.width,
            "height": self.height,
            "scale": self.scale.describe() if self.scale else "1:1",
            "skipped_count": self.skipped_count,
            "skipped": self.skipped,
            "targets": [t.to_dict() for t in self.targets],
        }

    def compact(self) -> str:
        """A text legend, for when the model gets the list as well as the image.

        Numbers alone force a visual lookup; a legend lets a text-only model
        reason too, at the cost of a few hundred tokens.
        """
        lines = []
        for target in self.targets:
            bits = [f"{target.index:>3}"]
            if target.label:
                bits.append(target.label)
            if target.role:
                bits.append(f"<{target.role}>")
            if target.disabled:
                bits.append("(disabled)")
            lines.append("  ".join(bits))
        for item in self.skipped:
            lines.append(f"  -  {item['label'] or '(unlabelled)'} — {item['reason']}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- core


def annotate_boxes(
    image: bytes,
    boxes: Sequence[Box],
    *,
    labels: Sequence[str] | None = None,
    roles: Sequence[str] | None = None,
    start_index: int = 1,
    colour: str = MARK_COLOUR,
    scale: Scale | None = None,
    thickness: int = 3,
) -> Grounding:
    """Draw numbered marks on a screenshot over known element boxes.

    Boxes are expected in *logical* coordinates when ``scale`` is given, so a
    box scraped from a DOM tree lands on the right pixels even on a HiDPI
    surface.

    Boxes with no area are dropped and recorded in
    :attr:`Grounding.skipped` rather than drawn. The usual cause is an element
    clipped by the viewport, which the DOM reports as a zero-height rect: it
    cannot be clicked, and a mark drawn on it would invite a model to pick an
    index that resolves to a point outside the button.
    """
    if not boxes:
        raise ValueError("annotate_boxes needs at least one box")

    width, height = image_size(image)
    targets: list[Target] = []
    skipped: list[dict] = []

    for offset, raw in enumerate(boxes):
        box = clamp_box(tuple(int(v) for v in raw[:4]), (width, height))  # type: ignore[arg-type]
        if scale is not None:
            box = clamp_box(scale.box_to_device(box), (width, height))
        label = (labels[offset] if labels and offset < len(labels) else "")
        if box[2] == 0 or box[3] == 0:
            skipped.append({
                "label": label,
                "box": list(box),
                "reason": "no area — element is clipped or hidden",
            })
            continue
        targets.append(Target(
            index=start_index + len(targets),
            box=box,
            label=label,
            role=(roles[offset] if roles and offset < len(roles) else ""),
        ))

    if not targets:
        raise ValueError(
            f"all {len(skipped)} boxes were degenerate (zero width or height); "
            "the elements are clipped or hidden"
        )

    filters = []
    for target in targets:
        left, top, box_w, box_h = target.box
        label_x = min(left, max(0, width - 44))
        label_y = max(14, top - 6)
        filters.append(
            f"drawbox=x={left}:y={top}:w={box_w}:h={box_h}:"
            f"color={colour}:t={thickness}"
        )
        # The number itself. A filled plate behind the digit keeps it readable
        # over any background, which matters because the page is arbitrary.
        filters.append(
            f"drawbox=x={label_x - 4}:y={label_y - 16}:w=44:h=22:"
            f"color={colour}@0.92:t=fill"
        )
        filters.append(
            f"drawtext=x={label_x}:y={label_y - 15}:text='{target.index}':"
            "fontsize=18:fontcolor=black"
        )

    return Grounding(
        targets=targets,
        image=_render(image, filters, width, height),
        width=width, height=height, scale=scale, skipped=skipped,
    )


def _render(image: bytes, filters: Sequence[str], width: int, height: int) -> bytes:
    """Run a drawbox/drawtext filter chain over an encoded image."""
    try:
        return run_bytes(
            [FFMPEG, "-hide_banner", "-loglevel", "error",
             "-f", "image2pipe", "-i", "-",
             "-vf", ",".join(filters),
             "-frames:v", "1", "-c:v", "png", "-f", "image2pipe", "-"],
            timeout=120,
            input_bytes=image,
        )
    except ProcessingError as exc:
        raise ProcessingError(
            "could not render set-of-marks:\n"
            f"{exc}\n"
            "(drawtext needs a font; install one with: apt-get install -y fonts-dejavu-core)"
        ) from exc


def annotate(
    image: bytes,
    targets: Sequence[Target],
    *,
    scale: Scale | None = None,
    colour: str = MARK_COLOUR,
) -> Grounding:
    """Draw marks for targets that already carry labels and roles."""
    return annotate_boxes(
        image,
        [t.box for t in targets],
        labels=[t.label for t in targets],
        roles=[t.role for t in targets],
        start_index=targets[0].index if targets else 1,
        colour=colour,
        scale=scale,
    )


def resolve(
    grounding: Grounding,
    answer: object,
    *,
    scale: Scale | None = None,
    fx: float = 0.5,
    fy: float = 0.5,
) -> tuple[int, int]:
    """Turn a model's answer into a click coordinate.

    Accepts an int, a numeric string, or a string with the number buried in it
    ("I think it is 12."). Anything unresolvable raises rather than guessing,
    because a silent default here is how an agent ends up clicking the wrong
    thing and reporting success.

    **The result is in pointer space**, which is what ``page.mouse.click``,
    ``xdotool click`` and every other input API expects — *not* image pixels.
    ``Target.box`` is in image pixels. On a 200%-scaled display or under
    browser chrome those two differ, so compare them with
    :meth:`Scale.to_device`, not with ``==``.
    """
    effective = scale if scale is not None else grounding.scale
    index = _coerce_index(answer)
    if index is None:
        raise ValueError(f"could not read an index out of {answer!r}")
    target = grounding.by_index(index)
    if target is None:
        raise ValueError(
            f"index {index} is not on this screen; valid marks are "
            f"{min(grounding.indices, default=0)}-{max(grounding.indices, default=0)}"
        )
    left, top, width, height = target.box
    logical = (left + width * fx, top + height * fy)
    if effective is not None:
        return effective.to_logical(logical[0], logical[1])
    return (int(logical[0]), int(logical[1]))


def resolve_box(grounding: Grounding, answer: object) -> Box:
    """The element's box rather than a click point, DPI-corrected."""
    index = _coerce_index(answer)
    if index is None:
        raise ValueError(f"could not read an index out of {answer!r}")
    target = grounding.by_index(index)
    if target is None:
        raise ValueError(f"index {index} is not on this screen")
    if grounding.scale is None:
        return target.box
    return grounding.scale.box_to_logical(target.box)


def _coerce_index(answer: object) -> int | None:
    """Pull a mark index out of whatever shape the model replied in."""
    if isinstance(answer, bool):
        return None
    if isinstance(answer, int):
        return answer
    if isinstance(answer, Target):
        return answer.index
    if isinstance(answer, dict):
        if "index" in answer:
            return _coerce_index(answer["index"])
        return None
    text = str(answer).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        pass
    digits = "".join(ch for ch in text if ch.isdigit())
    return int(digits) if digits else None


def from_boxes(payload: Iterable[dict]) -> list[Target]:
    """Build targets from whatever a DOM/CDP/AT-SPI dump produced."""
    return [Target.from_dict(item) for item in payload]


def to_dict(targets: Sequence[Target]) -> list[dict]:  # pragma: no cover - debugging aid
    return [t.to_dict() for t in targets]


def to_json(grounding: Grounding) -> str:  # pragma: no cover - debugging aid
    return json.dumps(grounding.to_dict(), indent=2)
