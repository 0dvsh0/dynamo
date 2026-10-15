"""Route frames to the cheapest model that can actually answer.

Recording gets an agent a *replay*. :mod:`ocr`, :mod:`diff` and :mod:`grounding`
answer a great deal about a screen with no model at all. This module is for the
residue — the frames where the answer is a judgement rather than a measurement,
such as "is this a payment decline or a slow spinner", or "what did the user
actually see go wrong".

Three things decide what that costs, and none of them is the model's quality:

* **How many frames are sent.**  A 52-second session reduces to ~29 keyframes,
  and most of those changed nothing an agent cares about.  Those are settled by
  the diff regions alone, at zero model calls.
* **How much of each frame is sent.**  :func:`crop_image` cuts the changed
  region in ffmpeg *before* encoding, so the model looks at 480×90 of a
  1440×900 screen.  That is the difference between a 4B model being usable on a
  CPU and being unusable.
* **Which model sees it.**  A calibrated decision model (Laya) scores frames for
  almost nothing and hands the survivors to a generative VLM (Qwen3-VL).
  Generation is the expensive half, and only the survivors pay it.

The layer is deliberately model-agnostic: the backend is a constructor argument,
so the same session can be answered locally, remotely, or by a stub in CI.  This
module is never constructed implicitly — importing the package still works with
no model, no server and no network, which is what keeps ``dependencies = []``
true.

Typical use::

    from agent_visual_harness import Frame, OllamaBackend, Triage

    triage = Triage(OllamaBackend(model="qwen3-vl:4b"))
    report = triage.run(frames, "What error did the user hit, and on which step?")
    print(report.summary())

Everything a caller needs to argue about cost is on the report: how many frames
reached a generative model, how many were cropped first, and what the frames
would have cost uncropped.
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from collections.abc import Callable, Mapping, Sequence

from ._ffmpeg import FFMPEG, run_bytes
from .capture import image_size
from .diff import compare
from .errors import HarnessError, VisionError
from .ocr import read_image

__all__ = [
    "TOKENS_PER_MEGAPIXEL",
    "DEFAULT_VLM_MODEL",
    "DEFAULT_OLLAMA_ENDPOINT",
    "DEFAULT_LAYA_ENDPOINT",
    "MAX_STATE_CHARS",
    "Backend",
    "OllamaBackend",
    "LayaBackend",
    "RemoteBackend",
    "StubBackend",
    "Question",
    "Decision",
    "Frame",
    "Answer",
    "Report",
    "Triage",
    "crop_image",
    "estimate_tokens",
    "union_box",
    "frames_from_manifest",
    "regions_between",
]


#: Roughly 765 tokens per megapixel, from published vision pricing.  This is a
#: model of a price list, not a measurement from a tokenizer — it exists so two
#: crops can be compared on one consistent scale, and it is labelled as such
#: wherever it reaches a page.
TOKENS_PER_MEGAPIXEL = 765

#: The local default.  4B because it has to run on a CPU when nothing else is
#: available; the model id is a constructor argument, not a constant, so an 8B
#: or a 72B is the same one-line change.
DEFAULT_VLM_MODEL = "qwen3-vl:4b"

DEFAULT_OLLAMA_ENDPOINT = os.environ.get("DYNAMO_OLLAMA", "http://127.0.0.1:11434")
DEFAULT_LAYA_ENDPOINT = os.environ.get("DYNAMO_LAYA", "http://127.0.0.1:8000")

#: A decision model reads the first ~512 tokens of a state, so there is no point
#: handing it a whole screenful. This is roughly where the decisive text at the
#: top of a page lives, and it stops a long page from burying the error message.
MAX_STATE_CHARS = 1200

Box = tuple[int, int, int, int]


# --------------------------------------------------------------------- geometry


def estimate_tokens(image: bytes) -> int:
    """Price an image the way a size-aware vision API would.

    Reads the dimensions out of the image header — no decoder, no ffprobe —
    which is why this is cheap enough to call on every frame of a session.
    """
    width, height = image_size(image)
    return int(round((width * height) / 1e6 * TOKENS_PER_MEGAPIXEL))


def union_box(boxes: Sequence[Sequence[int]]) -> Box | None:
    """The smallest box containing every given box, or ``None`` for none.

    Zero-area boxes are dropped first: a clipped element reports a 0-height
    rect, and letting one of those into the union would stretch the crop across
    the page for no reason.
    """
    rects = [b for b in boxes if len(b) >= 4 and int(b[2]) > 0 and int(b[3]) > 0]
    if not rects:
        return None
    x0 = min(int(b[0]) for b in rects)
    y0 = min(int(b[1]) for b in rects)
    x1 = max(int(b[0]) + int(b[2]) for b in rects)
    y1 = max(int(b[1]) + int(b[3]) for b in rects)
    return (x0, y0, x1 - x0, y1 - y0)


def pad_box(box: Sequence[int], pad: int) -> Box:
    """Grow a box by ``pad`` on every side, so the crop keeps its context."""
    x, y, w, h = (int(v) for v in box[:4])
    return (x - pad, y - pad, w + 2 * pad, h + 2 * pad)


def clamp_box(box: Sequence[int], size: Sequence[int]) -> Box:
    """Fit a box inside ``size``. ffmpeg fails on a crop that leaves the frame."""
    x, y, w, h = (int(v) for v in box[:4])
    width, height = int(size[0]), int(size[1])
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    return (x, y, w, h)


def crop_image(image: bytes, region: Sequence[int], *, timeout: float = 60.0) -> bytes:
    """Cut ``region`` out of an image, in ffmpeg, before it reaches a model.

    Cropping here rather than inside the model's own preprocessing is the whole
    trick.  A 480×90 crop is 0.043 MP and a 1440×900 frame is 1.296 MP, and a
    vision model is billed — and slowed — per pixel it has to attend to.  The
    model is not asked to ignore the rest of the screen; it is never shown it.
    """
    x, y, w, h = (int(v) for v in region[:4])
    if w <= 0 or h <= 0:
        raise VisionError(f"crop region has no area: {region!r}")
    return run_bytes(
        [FFMPEG, "-hide_banner", "-loglevel", "error",
         "-f", "image2pipe", "-i", "-",
         "-vf", f"crop={w}:{h}:{x}:{y}",
         "-c:v", "png", "-f", "image2pipe", "-"],
        timeout=timeout,
        input_bytes=image,
    )


def _tokens(image: bytes) -> int:
    """Price a frame, tolerating one whose header cannot be read.

    Token accounting exists to argue about cost, not to produce the answer, so a
    frame with an unreadable header is priced at zero rather than allowed to end
    the run.  That is the same rule the recorder applies to telemetry: nothing
    that only observes may bring down the thing it observes.
    """
    try:
        return estimate_tokens(image)
    except (ValueError, IndexError, OSError):
        return 0


# ------------------------------------------------------------------------- http


def _b64(image: bytes) -> str:
    return base64.b64encode(image).decode("ascii")


def _as_float(value: Any) -> float | None:
    """A number, or ``None``. Laya omits fields rather than sending nulls."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _post_json(
    url: str,
    payload: dict[str, Any],
    *,
    timeout: float,
    api_key: str | None = None,
) -> dict[str, Any]:
    """POST JSON with ``urllib`` and hand back the decoded object.

    The standard library is the point: adding a model to a zero-dependency
    package must not add a dependency, or the harness can be broken by a
    resolver conflict in an agent's own virtualenv.
    """
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", "replace")[:400]
        except (OSError, ValueError):  # pragma: no cover - body already gone
            pass
        raise VisionError(f"POST {url} failed: HTTP {exc.code} {exc.reason}. {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise VisionError(
            f"POST {url} failed: {exc}. Is the model server running and reachable?"
        ) from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise VisionError(f"POST {url} returned non-JSON: {raw[:200]!r}") from exc
    if not isinstance(data, dict):
        raise VisionError(f"POST {url} returned {type(data).__name__}, expected an object")
    return data


# --------------------------------------------------------------------- backends


class Backend:
    """One model, behind one method.

    ``ask`` is the generative half and is the only thing :class:`Triage`
    requires.  A backend that cannot generate (a decision model) overrides
    ``decide`` instead and is used as a filter.
    """

    name = "backend"
    local = True

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "local": self.local}

    def ask(self, image: bytes, question: str, *, timeout: float | None = None) -> str:
        raise NotImplementedError

    def decide(
        self,
        state: str | Mapping[str, Any] | Sequence[Any],
        questions: Sequence["Question"],
        *,
        timeout: float | None = None,
    ) -> list["Decision"]:
        raise VisionError(
            f"{self.name} does not answer typed questions; it is not usable as a filter"
        )


class OllamaBackend(Backend):
    """An open-weight VLM served by Ollama, over its local HTTP API.

    Ollama rather than a Python binding, for three reasons that all matter here:
    the package keeps zero runtime dependencies (``urllib`` is standard library),
    a server holds the model warm between frames — so a latency number measures
    the model rather than the load — and ``ollama pull`` is the entire install
    step, which is what a reader will actually follow.
    """

    def __init__(
        self,
        model: str = DEFAULT_VLM_MODEL,
        endpoint: str = DEFAULT_OLLAMA_ENDPOINT,
        *,
        timeout: float = 600.0,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> None:
        self.model = model
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.name = f"ollama:{model}"
        self.calls = 0
        self.seconds = 0.0

    def describe(self) -> dict[str, Any]:
        return {
            "name": "ollama",
            "model": self.model,
            "endpoint": self.endpoint,
            "local": True,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    def ask(self, image: bytes, question: str, *, timeout: float | None = None) -> str:
        payload = {
            "model": self.model,
            "prompt": question,
            "images": [_b64(image)],
            "stream": False,
            # Zero temperature so the same frame gives the same answer twice.
            # A regression gate that drifts between runs is not a gate.
            #
            # ``num_predict`` bounds the worst case: left unset, a model that
            # fails to emit EOS generates until the context runs out, which on a
            # CPU is the difference between a slow run and one that never ends.
            # 256 tokens is far more than "answer in one sentence" needs.
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        started = time.monotonic()
        data = _post_json(
            f"{self.endpoint}/api/generate", payload, timeout=timeout or self.timeout
        )
        self.calls += 1
        self.seconds += time.monotonic() - started
        return str(data.get("response") or "").strip()

    def available(self) -> bool:
        """Whether the server answers at all.  Never raises — probing is not a failure."""
        try:
            with urllib.request.urlopen(f"{self.endpoint}/api/tags", timeout=2.0) as response:
                return int(getattr(response, "status", 200)) == 200
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            return False

    def models(self) -> list[str]:
        """Tags the server currently holds, so a caller can name a model it does not have."""
        try:
            with urllib.request.urlopen(f"{self.endpoint}/api/tags", timeout=5.0) as response:
                data = json.loads(response.read().decode("utf-8", "replace"))
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            return []
        return [str(m.get("name", "")) for m in (data.get("models") or [])]


@dataclass(slots=True)
class Question:
    """One typed question for a decision model.

    ``kind`` uses Laya's vocabulary, because its three types are genuinely three
    different heads rather than three wordings of one question: a ``choice``
    picks a label out of ``criteria`` (a mapping of label to description), a
    ``score`` returns an expected level on an ordered ``criteria`` list, and a
    ``noul`` returns **P(true) as a float** — not a boolean, deliberately, so a
    caller cannot mistake a probability for a decision.

    ``criteria`` is part of the prompt, not metadata: the wording of an option is
    the question. It is included here for that reason.
    """

    id: str
    kind: str  # "choice" | "score" | "noul"
    instructions: str
    criteria: Mapping[str, str] | Sequence[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"type": self.kind, "instructions": self.instructions}
        if self.criteria is not None:
            payload["criteria"] = (
                dict(self.criteria) if isinstance(self.criteria, Mapping) else list(self.criteria)
            )
        return payload


@dataclass(slots=True)
class Decision:
    """A decision model's answer to one :class:`Question`.

    Two different numbers come back and they are not interchangeable, which is
    the trap Laya's own documentation warns about, so both are kept:

    ``probability``
        The quantity to threshold, on a scale that means the same thing for
        every question type: P(true) for a ``noul``, and ``answer_confidence``
        (``max(p)``, the mass on the answer being reported) for a ``choice`` or a
        ``score``.  Still a policy, not a property — Laya's shipped checkpoints
        are over-confident out of the box, and a cutoff has to be fitted per
        option count on held-out data.
    ``confidence``
        Laya's other number: normalised entropy on a choice or score, so its
        scale moves with the option count. Kept for diagnosis, never gated on.
    """

    id: str
    kind: str
    label: str | None = None
    probability: float = 0.0
    value: float | None = None
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "type": self.kind,
            "label": self.label,
            "p": round(self.probability, 4),
        }
        if self.value is not None:
            payload["value"] = round(self.value, 4)
        if self.confidence is not None:
            payload["confidence"] = round(self.confidence, 4)
        return payload


class LayaBackend(Backend):
    """Calibrated typed decisions, in one forward pass, with no generation.

    Laya answers ``choice`` / ``score`` / ``noul`` questions and returns typed
    answers with calibrated probabilities. That shape is exactly what makes it a
    *filter* rather than a replacement: it can say whether a frame is worth a
    generative model's attention, cheaply and with a number a caller can
    threshold — but it cannot describe what it saw, because it generates nothing.

    This speaks Laya's own server protocol, ``POST /v1/systemone`` (the same wire
    format the hosted Jev API uses), so there is no invented shim in the path:

        pip install "laya[serve]"
        laya-serve                            # exposes POST /v1/systemone
        python3 demo/run_agent_bench.py --laya http://127.0.0.1:8000

    Laya reads text, not pixels.  What a frame *means* therefore has to reach it
    as text — :meth:`Triage.state_for` is where that happens — which is worth
    knowing before designing a filter around it.

    Licence, checked rather than assumed: the official checkpoints
    (``convaiinnovations/laya``, ``laya-multilingual``,
    ``laya-typed-decisions``) are all Apache-2.0, so this path is clean for a
    commercial build.  Community *vision* forks are not uniformly so licensed —
    one on the Hub is CC BY-NC-SA — which is a second reason to hand Laya text
    rather than pixels: the text checkpoints are the officially licensed ones.
    """

    def __init__(self, endpoint: str = DEFAULT_LAYA_ENDPOINT, *, timeout: float = 30.0) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self.name = "laya"
        self.calls = 0
        self.seconds = 0.0

    def describe(self) -> dict[str, Any]:
        return {"name": "laya", "endpoint": self.endpoint, "local": True, "generates": False}

    def available(self) -> bool:
        """Whether anything is listening. Never raises — probing is not a failure.

        ``laya-serve`` does ship ``GET /health``, but the probe is deliberately
        satisfied by *any* HTTP response, a 404 included: the question is whether
        a server is there, not whether it has the route you expected, and
        requiring one would make this report a working server as missing.
        """
        try:
            urllib.request.urlopen(f"{self.endpoint}/health", timeout=2.0).close()
            return True
        except urllib.error.HTTPError:
            return True
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            return False

    def decide(
        self, state: str | Mapping[str, Any] | Sequence[Any],
        questions: Sequence[Question], *, timeout: float | None = None,
    ) -> list[Decision]:
        payload = {
            "state": state,
            "questions": {q.id: q.to_dict() for q in questions},
        }
        started = time.monotonic()
        data = _post_json(
            f"{self.endpoint}/v1/systemone", payload, timeout=timeout or self.timeout
        )
        self.calls += 1
        self.seconds += time.monotonic() - started

        answers = data.get("answers")
        if not isinstance(answers, Mapping):
            raise VisionError(
                f"laya returned no 'answers' object: {str(data)[:200]}"
            )
        return [
            self._decision(q, answers.get(q.id))
            for q in questions
        ]

    @staticmethod
    def _decision(question: Question, answer: Any) -> Decision:
        """One answer dict, in Laya's shape, to one :class:`Decision`."""
        if not isinstance(answer, Mapping):
            return Decision(id=question.id, kind=question.kind)

        confidence = _as_float(answer.get("confidence"))
        if question.kind == "noul":
            # noul is P(true). answer_confidence here is max(p_true, 1 - p_true)
            # — the probability of the side being reported, not of "yes" — so
            # gating on it would flip meaning with the answer.
            p_true = _as_float(answer.get("noul"))
            return Decision(
                id=question.id,
                kind="noul",
                label="true" if (p_true or 0.0) >= 0.5 else "false",
                probability=p_true if p_true is not None else 0.0,
                value=p_true,
                confidence=confidence,
            )

        answer_confidence = _as_float(answer.get("answer_confidence"))
        return Decision(
            id=question.id,
            kind=question.kind,
            label=answer.get("choice"),
            probability=(
                answer_confidence
                if answer_confidence is not None
                else (confidence if confidence is not None else 0.0)
            ),
            value=_as_float(answer.get("score")),
            confidence=confidence,
        )

    def ask(self, image: bytes, question: str, *, timeout: float | None = None) -> str:
        raise VisionError(
            "Laya makes decisions; it does not generate text. "
            "Use it as Triage(filter=LayaBackend(...)) and keep a generative "
            "backend as the primary."
        )


class RemoteBackend(Backend):
    """Any OpenAI-compatible chat endpoint, for the frames that earn it.

    Off by default, and never constructed implicitly.  The point of the local
    path is that nothing leaves the machine unless a caller decides it should —
    so escalation is something a caller opts into per run, and every escalated
    frame is counted in the report.
    """

    local = False

    def __init__(
        self,
        model: str,
        *,
        endpoint: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        name: str = "remote",
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key or os.environ.get("DYNAMO_REMOTE_KEY")
        self.timeout = timeout
        self.name = f"{name}:{model}"
        self.calls = 0
        self.seconds = 0.0

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model,
            "endpoint": self.endpoint,
            "local": False,
            "authenticated": bool(self.api_key),
        }

    def ask(self, image: bytes, question: str, *, timeout: float | None = None) -> str:
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{_b64(image)}"},
                        },
                    ],
                }
            ],
        }
        started = time.monotonic()
        data = _post_json(
            f"{self.endpoint}/chat/completions",
            payload,
            timeout=timeout or self.timeout,
            api_key=self.api_key,
        )
        self.calls += 1
        self.seconds += time.monotonic() - started
        choices = data.get("choices") or []
        if not choices:
            raise VisionError(f"{self.name} returned no choices: {str(data)[:200]}")
        message = choices[0].get("message") or {}
        return str(message.get("content") or "").strip()


class StubBackend(Backend):
    """A deterministic backend that answers without a model.

    It exists so the routing can be tested for real.  The part of this module
    that can be wrong is not the model call — it is *how many* calls happen, what
    gets cropped, and which frames never reach a model at all.  A stub lets the
    tests assert exactly those, on any machine, with no GPU and no network.
    """

    def __init__(
        self,
        *,
        name: str = "stub",
        answers: dict[str, str] | None = None,
        local: bool = True,
        latency: float = 0.0,
        decision_probability: float = 0.5,
        decision_label: str = "interesting",
        empty: bool = False,
    ) -> None:
        self.name = name
        self.local = local
        self.answers = dict(answers or {})
        self.latency = latency
        self.decision_probability = decision_probability
        self.decision_label = decision_label
        self.empty = empty
        self.calls = 0
        self.seen_images: list[int] = []

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "local": self.local, "stub": True}

    def ask(self, image: bytes, question: str, *, timeout: float | None = None) -> str:
        self.calls += 1
        self.seen_images.append(len(image))
        if self.latency:
            time.sleep(self.latency)
        if self.empty:
            return ""
        if question in self.answers:
            return self.answers[question]
        return f"[{self.name}] {question}"

    def decide(
        self,
        state: str | Mapping[str, Any] | Sequence[Any],
        questions: Sequence[Question],
        *,
        timeout: float | None = None,
    ) -> list[Decision]:
        self.calls += 1
        return [
            Decision(
                id=q.id,
                kind=q.kind,
                label=self.decision_label,
                probability=self.decision_probability,
            )
            for q in questions
        ]


# ------------------------------------------------------------------- value types


@dataclass(slots=True)
class Frame:
    """One keyframe, plus everything the harness already knows about it.

    ``regions`` is the output of :func:`agent_visual_harness.diff.compare` — the
    boxes that actually moved since the previous frame.  Carrying them here is
    what lets a crop target the part of the screen worth looking at, and what
    lets an unchanged frame be answered without a model.

    ``text`` is what :mod:`ocr` read off the frame, when a caller has run it.
    That matters because the filter is a *text* decision model: it cannot look at
    the image, so a frame arrives at it as the description above plus whatever
    text was read off the pixels.  A frame with neither is a frame the filter is
    guessing about, and the report says so.
    """

    name: str
    image: bytes
    regions: tuple[Box, ...] = ()
    action: str | None = None
    timestamp: float = 0.0
    reason: str = "scene"  # "scene" | "action"
    text: str = ""

    @property
    def is_anchor(self) -> bool:
        """Anchors are the frames at a logged action.  They are never dropped."""
        return self.reason == "action"

    @property
    def has_change(self) -> bool:
        return bool(self.regions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "bytes": len(self.image),
            "regions": [list(r) for r in self.regions],
            "action": self.action,
            "timestamp": round(self.timestamp, 3),
            "reason": self.reason,
            "text": self.text,
            "tokens_if_sent_whole": estimate_tokens(self.image),
        }


@dataclass(slots=True)
class Answer:
    """What happened to one frame, including the decision not to send it."""

    frame: str
    question: str
    text: str = ""
    backend: str = ""
    region: Box | None = None
    cropped: bool = False
    escalated: bool = False
    elapsed: float = 0.0
    tokens_before: int = 0
    tokens_after: int = 0
    filtered_p: float | None = None
    skipped_reason: str | None = None

    @property
    def skipped(self) -> bool:
        return self.skipped_reason is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "question": self.question,
            "text": self.text,
            "backend": self.backend,
            "region": list(self.region) if self.region else None,
            "cropped": self.cropped,
            "escalated": self.escalated,
            "elapsed": round(self.elapsed, 3),
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "filtered_p": None if self.filtered_p is None else round(self.filtered_p, 4),
            "skipped_reason": self.skipped_reason,
        }


@dataclass(slots=True)
class Report:
    """The full account of one triage run, including what it declined to do."""

    question: str
    frames: int = 0
    answered: int = 0
    skipped: int = 0
    model_calls: int = 0
    filter_calls: int = 0
    escalated: int = 0
    cropped: int = 0
    tokens_before: int = 0
    tokens_after: int = 0
    elapsed: float = 0.0
    answers: list[Answer] = field(default_factory=list)
    backends: dict[str, Any] = field(default_factory=dict)

    @property
    def calls_avoided(self) -> int:
        """Frames that never reached a generative model."""
        return max(0, self.frames - self.model_calls)

    @property
    def token_ratio(self) -> float:
        """Tokens actually sent, as a fraction of sending every frame whole."""
        if not self.tokens_before:
            return 1.0
        return self.tokens_after / self.tokens_before

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "frames": self.frames,
            "answered": self.answered,
            "skipped": self.skipped,
            "model_calls": self.model_calls,
            "filter_calls": self.filter_calls,
            "escalated": self.escalated,
            "cropped": self.cropped,
            "calls_avoided": self.calls_avoided,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "token_ratio": round(self.token_ratio, 4),
            "elapsed": round(self.elapsed, 3),
            "backends": self.backends,
            "answers": [a.to_dict() for a in self.answers],
        }

    def summary(self) -> str:
        lines = [
            f"question  {self.question}",
            f"frames    {self.frames}",
            f"sent      {self.model_calls} to a generative model "
            f"({self.answered} answered, {self.calls_avoided} never sent)",
        ]
        if self.filter_calls:
            lines.append(f"filter    {self.filter_calls} frames scored")
        if self.cropped:
            lines.append(f"cropped   {self.cropped} frames before sending")
        if self.escalated:
            lines.append(f"escalated {self.escalated} frames off-machine")
        if self.tokens_before:
            lines.append(
                f"tokens    {self.tokens_before:,} -> {self.tokens_after:,} "
                f"({self.token_ratio * 100:.1f}% of sending every frame whole)"
            )
        lines.append(f"elapsed   {self.elapsed:.2f}s")
        return "\n".join(lines)


def frames_from_manifest(
    directory: Path | str,
    *,
    limit: int | None = None,
    ocr: bool = False,
) -> list[Frame]:
    """Rebuild frames from a keyframes directory the token reducer wrote.

    Reads the ``manifest.json`` sitting next to the images and re-derives what
    moved by diffing consecutive frames with :mod:`diff` — the same comparison the
    rest of the harness uses, so the regions are *measured* rather than assumed.
    Those regions are what let a crop target the part of the screen worth looking
    at, and what let an unchanged frame be answered by nothing at all.

    With ``ocr=True`` each frame is also read with tesseract — locally, no model,
    no network — and the text is attached to the frame. That matters because a
    filter like Laya is a text model: it cannot see the pixels, so without this it
    is judging the action label alone.
    """
    root = Path(directory)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise VisionError(f"no manifest.json in {root} — run `dynamo keyframes` first")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = list(manifest.get("keyframes") or [])
    if limit is not None:
        entries = entries[:limit]

    images = [(root / str(entry["path"])).read_bytes() for entry in entries]

    frames: list[Frame] = []
    for index, (entry, image) in enumerate(zip(entries, images)):
        text = ""
        if ocr:
            try:
                text = " ".join((read_image(image).text or "").split())[:MAX_STATE_CHARS]
            except HarnessError:
                # A frame nobody can read is not a reason to abandon the set.
                text = ""
        frames.append(
            Frame(
                name=str(entry.get("path", f"frame_{index + 1:04d}")),
                image=image,
                regions=regions_between(images[index - 1], image) if index else (),
                action=entry.get("action"),
                timestamp=float(entry.get("timestamp", 0.0)),
                reason=str(entry.get("reason", "scene")),
                text=text,
            )
        )
    return frames


def regions_between(before: bytes, after: bytes) -> tuple[Box, ...]:
    """What moved between two frames, as boxes. Empty when nothing did."""
    result = compare(before, after, diff_image=False)
    if result.same:
        return ()
    return tuple(region.box for region in result.regions)


# ------------------------------------------------------------------------ triage


class Triage:
    """Decide, per frame, how little work can still answer the question.

    The order below is the design, and each step exists because the one after it
    is more expensive than the one before:

    1. **No model.**  A frame with no changed regions and no logged action
       carries no new information, so it is answered by omission.  On a real
       session this is where most frames go.
    2. **Cheap model.**  Optional.  A decision backend scores the survivors; the
       ones below the threshold are dropped, with the probability that dropped
       them recorded on the answer rather than discarded.  The filter is a text
       model, so it sees the frame as text — see :meth:`_state_for`.
    3. **Generative model.**  Only now is a VLM asked, and it is asked about a
       crop rather than a whole screen whenever the diff can point at one.
    4. **Optional escalation.**  A remote backend, if one was configured and the
       local answer came back empty.
    """

    def __init__(
        self,
        primary: Backend,
        *,
        filter: Backend | None = None,
        escalation: Backend | None = None,
        crop: bool = True,
        skip_unchanged: bool = True,
        filter_threshold: float = 0.5,
        min_region_px: int = 400,
        pad: int = 8,
        probe: Question | None = None,
        state_for: Callable[[Frame], Any] | None = None,
    ) -> None:
        self.primary = primary
        self.filter = filter
        self.escalation = escalation
        self.crop = crop
        self.skip_unchanged = skip_unchanged
        self.filter_threshold = filter_threshold
        self.min_region_px = min_region_px
        self.pad = pad
        self.probe = probe or Question(
            id="worth_a_model",
            kind="noul",
            instructions=(
                "Does this screen show a state or message that an automated agent "
                "would have to act on, or that a person would need to be told about? "
                "A steady screen, a spinner, or a page that merely changed styling "
                "is false."
            ),
        )
        self._state_for_override = state_for

    # ------------------------------------------------------------------ public

    def run(
        self,
        frames: Sequence[Frame],
        question: str,
        *,
        timeout: float | None = None,
        on_answer: Callable[[Answer, int, int], None] | None = None,
    ) -> Report:
        """Answer the question about every frame, as cheaply as each allows.

        ``on_answer`` is called with ``(answer, index, total)`` after each frame.
        It exists because the honest version of this is slow: a 4B model on a CPU
        takes tens of seconds a frame, and a caller watching a terminal for an
        hour needs to see which frame it is on. Nothing passed here can change the
        result.
        """
        report = Report(
            question=question,
            frames=len(frames),
            backends={
                "primary": self.primary.describe(),
                "filter": self.filter.describe() if self.filter else None,
                "escalation": self.escalation.describe() if self.escalation else None,
            },
        )
        started = time.monotonic()
        total = len(frames)

        for index, frame in enumerate(frames, start=1):
            answer = self._one(frame, question, timeout=timeout)
            report.answers.append(answer)
            report.tokens_before += answer.tokens_before
            report.tokens_after += answer.tokens_after

            if answer.filtered_p is not None:
                report.filter_calls += 1
            if answer.skipped:
                report.skipped += 1
            else:
                report.answered += 1
                report.model_calls += 1
                if answer.cropped:
                    report.cropped += 1
                if answer.escalated:
                    report.escalated += 1

            if on_answer is not None:
                on_answer(answer, index, total)

        report.elapsed = time.monotonic() - started
        return report

    # ----------------------------------------------------------------- private

    def _one(self, frame: Frame, question: str, *, timeout: float | None) -> Answer:
        tokens_before = _tokens(frame.image)

        # 1. Nothing moved and nothing happened.  No model can add to that, and
        #    asking one would be paying to be told the frame is unchanged.
        if self.skip_unchanged and not frame.has_change and not frame.is_anchor:
            return Answer(
                frame=frame.name,
                question=question,
                skipped_reason="unchanged (no diff regions, not an action anchor)",
                tokens_before=tokens_before,
            )

        answer = Answer(frame=frame.name, question=question, tokens_before=tokens_before)

        # 2. A decision model drops a frame far more cheaply than a generative
        #    one answers it.  Anchors bypass the filter: a frame at a logged
        #    action is evidence, and evidence is not filtered.
        if self.filter is not None:
            try:
                decisions = self.filter.decide(
                    self.state_for(frame), [self.probe], timeout=timeout
                )
            except VisionError:
                decisions = []
            answer.filtered_p = decisions[0].probability if decisions else None
            if (
                answer.filtered_p is not None
                and answer.filtered_p < self.filter_threshold
                and not frame.is_anchor
            ):
                answer.skipped_reason = (
                    f"filtered (p={answer.filtered_p:.2f} < {self.filter_threshold:.2f})"
                )
                return answer

        # 3. Crop to what moved, in ffmpeg, before the model sees anything.
        payload = frame.image
        region = self._region_for(frame) if self.crop else None
        if region is not None:
            try:
                payload = crop_image(frame.image, region, timeout=timeout or 60.0)
            except HarnessError:
                # A crop that cannot be made is not a reason to lose the frame.
                # Both failures land here — VisionError for a region with no
                # area, ProcessingError when ffmpeg refuses the crop — and both
                # mean the same thing: fall back to the whole frame and let the
                # report show that no crop happened.
                region = None
            else:
                answer.region = region
                answer.cropped = True

        started = time.monotonic()
        text = self.primary.ask(payload, question, timeout=timeout)
        answer.elapsed = time.monotonic() - started
        answer.text = text
        answer.backend = self.primary.name
        answer.tokens_after = _tokens(payload)

        # 4. Escalate only when a caller opted in and the local answer came back
        #    with nothing to say.  Silence is the honest trigger; a caller who
        #    wants "always escalate the hard ones" can write that rule on top.
        if self.escalation is not None and not text:
            answer.text = self.escalation.ask(payload, question, timeout=timeout)
            answer.backend = self.escalation.name
            answer.escalated = True

        return answer

    def state_for(self, frame: Frame) -> Any:
        """The frame, as the *text* a decision model can read.

        Public because it is the honest answer to "what does the filter actually
        see?" — a caller diagnosing a filter that drops too much needs to be able
        to print this for one frame and read it.

        A filter like Laya never sees the pixels — it makes typed decisions over
        text.  Handing it the harness's own structured observation is not a
        limitation being worked around: it is the observation the recorder
        already made, and it is why the filter costs a single forward pass rather
        than a vision encode.

        ``frame.text`` is whatever :mod:`ocr` read off the frame, if a caller ran
        it.  With it, the filter judges the screen's actual words; without it, it
        judges the action that produced the frame — useful, but weaker, and
        nothing here should pretend otherwise.
        """
        if self._state_for_override is not None:
            return self._state_for_override(frame)

        state: dict[str, Any] = {
            "on_screen": frame.text or "",
            "last_action": frame.action or "",
            "frame_kind": frame.reason,
            "seconds_into_session": round(frame.timestamp, 2),
            "changed_regions": len(frame.regions),
        }
        if frame.regions:
            # Summarised, not enumerated. A text model handed twenty 10×10 boxes
            # learns that a lot happened and nothing about what; the largest
            # change and the share of screen are what it can actually reason over.
            areas = [int(r[2]) * int(r[3]) for r in frame.regions]
            largest = max(range(len(areas)), key=areas.__getitem__)
            state["largest_change"] = list(frame.regions[largest])
            try:
                width, height = image_size(frame.image)
            except (ValueError, IndexError, OSError):
                pass
            else:
                if width and height:
                    state["changed_share_of_screen"] = round(sum(areas) / (width * height), 4)
        return state

    def _region_for(self, frame: Frame) -> Box | None:
        """The smallest padded box covering what moved, or ``None``.

        Regions below ``min_region_px`` are treated as glyph-level noise rather
        than as a reason to look — the same tolerance :mod:`diff` applies before
        it reports a region at all.
        """
        if not frame.regions:
            return None
        significant = [
            r for r in frame.regions if int(r[2]) * int(r[3]) >= self.min_region_px
        ]
        box = union_box(significant or frame.regions)
        if box is None:
            return None
        try:
            size = image_size(frame.image)
        except (ValueError, IndexError, OSError):
            # An unreadable header means the crop cannot be bounded, and an
            # unbounded crop is how ffmpeg gets a region that leaves the frame.
            return None
        return clamp_box(pad_box(box, self.pad), size)
