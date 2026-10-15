"""Tests for the model-facing triage layer.

Almost none of this needs a model, a GPU or a network. The part of
:mod:`agent_visual_harness.vision` that can be silently wrong is the *routing* —
how many frames reach a generative model, which ones get cropped, and which are
answered by nothing at all — so that is tested against a deterministic stub where
the answer is known and the call count is assertable.

The crop tests decode and re-encode a real PNG, so they need a real ffmpeg and
skip without one, exactly as the X11 tests do.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from agent_visual_harness import (
    Frame,
    LayaBackend,
    OllamaBackend,
    Question,
    RemoteBackend,
    StubBackend,
    Triage,
    crop_image,
    estimate_tokens,
    union_box,
)
from agent_visual_harness._ffmpeg import FFMPEG
from agent_visual_harness.errors import VisionError
from agent_visual_harness.vision import clamp_box, pad_box

from .conftest import requires_ffmpeg


def png(width: int = 1440, height: int = 900, colour: str = "0x203040") -> bytes:
    """A real PNG of a given size, made by ffmpeg.

    Synthetic rather than a fixture on disk so a test can ask for exactly the
    geometry it is reasoning about, and so `image_size` is exercised on bytes
    that actually came out of an encoder rather than on a hand-built header.
    """
    proc = subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", f"color=c={colour}:s={width}x{height}:d=1",
         "-frames:v", "1", "-c:v", "png", "-f", "image2pipe", "-"],
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:  # pragma: no cover - only on a broken ffmpeg
        raise RuntimeError(proc.stderr.decode("utf-8", "replace"))
    return proc.stdout


def frame(name: str = "f1", *, regions=(), reason: str = "scene", image: bytes | None = None) -> Frame:
    return Frame(name=name, image=image if image is not None else b"", regions=tuple(regions), reason=reason)


# ------------------------------------------------------------------- geometry


def test_union_box_of_nothing_is_none():
    assert union_box([]) is None


def test_union_box_ignores_zero_area_regions():
    # A clipped element reports a 0-height rect. Letting one in would stretch the
    # crop across the page for a region that has no pixels.
    assert union_box([(10, 10, 0, 40), (5, 5, 30, 0)]) is None


def test_union_box_contains_every_input():
    box = union_box([(100, 50, 40, 20), (200, 80, 10, 10)])
    assert box == (100, 50, 110, 40)
    x, y, w, h = box
    for bx, by, bw, bh in ((100, 50, 40, 20), (200, 80, 10, 10)):
        assert x <= bx and y <= by and bx + bw <= x + w and by + bh <= y + h


def test_pad_then_clamp_never_leaves_the_frame():
    # A region hugging the corner: padding pushes it negative, clamping pulls it
    # back. ffmpeg fails outright on a crop that leaves the frame.
    padded = pad_box((2, 2, 10, 10), 8)
    assert padded[0] < 0 and padded[1] < 0
    # The padding is kept where it fits; only the out-of-frame part is dropped.
    assert clamp_box(padded, (1440, 900)) == (0, 0, 26, 26)

    # A region hugging the far edge is trimmed rather than rejected.
    assert clamp_box(pad_box((1430, 890, 40, 40), 8), (1440, 900)) == (1422, 882, 18, 18)


def test_clamp_never_returns_a_zero_area_box():
    # Off-frame boxes come back as a 1x1 sliver, not as (0, 0, 0, 0): a zero-area
    # crop raises, and raising here would lose a frame that might matter.
    _, _, w, h = clamp_box((9999, 9999, 10, 10), (1440, 900))
    assert w >= 1 and h >= 1


# ---------------------------------------------------------------------- prices


@requires_ffmpeg
def test_estimate_tokens_scales_with_area():
    small = estimate_tokens(png(480, 90))
    whole = estimate_tokens(png(1440, 900))
    assert whole > small
    # 480x90 is 0.043 MP and 1440x900 is 1.296 MP; the price list multiplies both
    # by the same constant, so the ratio is the area ratio.
    assert whole / small == pytest.approx((1440 * 900) / (480 * 90), rel=0.02)


# ------------------------------------------------------------------------ crop


def test_crop_rejects_a_zero_area_region():
    with pytest.raises(VisionError):
        crop_image(b"not really a png", (10, 10, 0, 0))


@requires_ffmpeg
def test_crop_returns_exactly_the_region_asked_for():
    from agent_visual_harness import image_size

    whole = png(1440, 900)
    snip = crop_image(whole, (100, 200, 480, 90))
    assert image_size(snip) == (480, 90)
    assert estimate_tokens(snip) < estimate_tokens(whole)


# ---------------------------------------------------------------------- filter


class _Filter(StubBackend):
    """A stub whose whole purpose is to be threshholded."""

    def __init__(self, probability: float) -> None:
        super().__init__(name="laya-stub", decision_probability=probability)


# -------------------------------------------------------------------- routing


def test_no_model_is_asked_about_a_frame_that_changed_nothing():
    primary = StubBackend()
    report = Triage(primary).run([frame("quiet")], "what happened?")

    assert primary.calls == 0
    assert report.model_calls == 0
    assert report.skipped == 1
    assert report.calls_avoided == 1
    assert "unchanged" in report.answers[0].skipped_reason


def test_a_refactored_session_mostly_skips():
    # The load-bearing claim of the whole design: most frames of a real session
    # carry no new information, and the ones that do are the action anchors.
    frames = [frame(f"quiet{i}") for i in range(20)]
    frames.append(frame("clicked", regions=[(300, 400, 120, 30)], reason="action"))

    primary = StubBackend()
    report = Triage(primary).run(frames, "what happened?")

    assert report.frames == 21
    assert report.model_calls == 1
    assert report.calls_avoided == 20


def test_an_anchor_is_never_skipped_even_when_nothing_moved():
    # A frame at a logged action is evidence. Evidence is not dropped for
    # looking like the frame before it.
    primary = StubBackend()
    report = Triage(primary).run([frame("click", reason="action")], "what happened?")
    assert primary.calls == 1
    assert report.skipped == 0


def test_filter_drops_a_frame_below_the_threshold():
    primary = StubBackend()
    triage = Triage(primary, filter=_Filter(0.10), filter_threshold=0.5)
    report = triage.run([frame("maybe", regions=[(10, 10, 100, 40)])], "what happened?")

    assert primary.calls == 0
    assert report.model_calls == 0
    assert report.filter_calls == 1
    # The probability that dropped it is recorded, not discarded.
    assert report.answers[0].filtered_p == pytest.approx(0.10)
    assert "filtered" in report.answers[0].skipped_reason


def test_filter_passes_a_frame_above_the_threshold():
    primary = StubBackend()
    triage = Triage(primary, filter=_Filter(0.90), filter_threshold=0.5)
    report = triage.run([frame("interesting", regions=[(10, 10, 100, 40)])], "what happened?")

    assert primary.calls == 1
    assert report.model_calls == 1
    assert report.filter_calls == 1


def test_filter_never_drops_an_anchor():
    primary = StubBackend()
    triage = Triage(primary, filter=_Filter(0.01), filter_threshold=0.5)
    report = triage.run([frame("click", reason="action")], "what happened?")
    assert primary.calls == 1
    assert report.skipped == 0


def test_a_broken_filter_does_not_lose_the_frame():
    class Broken(StubBackend):
        def decide(self, state, questions, *, timeout=None):
            raise VisionError("server is down")

    primary = StubBackend()
    report = Triage(primary, filter=Broken()).run(
        [frame("x", regions=[(10, 10, 100, 40)])], "what happened?"
    )
    # A filter that cannot answer is not a reason to drop the frame.
    assert primary.calls == 1
    assert report.answers[0].filtered_p is None


def test_the_filter_is_given_text_not_pixels():
    # A decision model like Laya reads text; handing it image bytes would be
    # silently meaningless rather than an error, so this is worth pinning.
    seen: list[object] = []

    class Recording(StubBackend):
        def decide(self, state, questions, *, timeout=None):
            seen.append(state)
            return super().decide(state, questions, timeout=timeout)

    source = frame("click", regions=[(10, 10, 100, 40)], reason="action")
    source.action = "click: continue to payment"
    source.text = "Your card was declined"
    source.timestamp = 12.5
    Triage(StubBackend(), filter=Recording()).run([source], "q")

    assert len(seen) == 1
    state = seen[0]
    assert isinstance(state, dict)
    assert state["on_screen"] == "Your card was declined"
    assert state["last_action"] == "click: continue to payment"
    # Regions are summarised rather than enumerated: a text model given twenty
    # small boxes learns that a lot happened and nothing about what.
    assert state["changed_regions"] == 1
    assert state["largest_change"] == [10, 10, 100, 40]
    assert "\\x89PNG" not in str(state)


def test_a_caller_can_choose_what_the_filter_is_told():
    seen: list[object] = []

    class Recording(StubBackend):
        def decide(self, state, questions, *, timeout=None):
            seen.append(state)
            return super().decide(state, questions, timeout=timeout)

    Triage(
        StubBackend(),
        filter=Recording(),
        state_for=lambda f: f"only {f.name}",
    ).run([frame("x", regions=[(1, 1, 40, 40)])], "q")

    assert seen == ["only x"]


# ---------------------------------------------------------------------- crops


@requires_ffmpeg
def test_a_changed_region_is_cropped_before_the_model_sees_it():
    primary = StubBackend()
    shot = png(1440, 900)
    report = Triage(primary).run(
        [frame("moved", regions=[(600, 400, 480, 90)], image=shot)], "what changed?"
    )

    answer = report.answers[0]
    assert answer.cropped is True
    assert answer.region is not None
    assert report.cropped == 1
    # The model is charged for the crop, not the full frame.
    assert answer.tokens_after < answer.tokens_before
    assert report.token_ratio < 0.5


@requires_ffmpeg
def test_the_model_receives_the_crop_and_not_the_whole_frame():
    primary = StubBackend()
    whole = png(1440, 900)
    Triage(primary).run([frame("moved", regions=[(10, 10, 200, 60)], image=whole)], "q")

    # The stub records the byte length it was handed; a crop must be smaller.
    assert primary.seen_images
    assert primary.seen_images[0] < len(whole)


def test_crop_can_be_turned_off():
    primary = StubBackend()
    report = Triage(primary, crop=False).run(
        [frame("moved", regions=[(10, 10, 100, 40)])], "q"
    )
    assert report.answers[0].cropped is False
    assert report.cropped == 0


# ------------------------------------------------------------------ escalation


def test_escalation_fires_only_when_the_local_answer_is_empty():
    primary = StubBackend(empty=True)
    remote = StubBackend(name="remote:big")
    report = Triage(primary, escalation=remote).run(
        [frame("hard", reason="action")], "what happened?"
    )

    assert report.escalated == 1
    assert report.answers[0].escalated is True
    assert report.answers[0].backend == "remote:big"


def test_a_good_local_answer_is_never_escalated():
    primary = StubBackend(answers={"q": "the card was declined"})
    remote = StubBackend(name="remote:big")
    report = Triage(primary, escalation=remote).run([frame("easy", reason="action")], "q")

    assert report.escalated == 0
    assert remote.calls == 0
    assert report.answers[0].backend == "stub"


def test_no_escalation_backend_means_nothing_leaves_the_machine():
    report = Triage(StubBackend(empty=True)).run([frame("x", reason="action")], "q")
    assert report.escalated == 0
    assert report.backends["escalation"] is None


# -------------------------------------------------------------------- backends


def test_laya_refuses_to_generate():
    with pytest.raises(VisionError) as excinfo:
        LayaBackend().ask(b"", "describe this")
    assert "does not generate" in str(excinfo.value)


def test_laya_is_described_as_a_non_generating_local_backend():
    described = LayaBackend().describe()
    assert described["local"] is True
    assert described["generates"] is False


def test_remote_backend_is_not_local():
    assert RemoteBackend("some-model").describe()["local"] is False


def test_ollama_backend_names_the_model_it_will_use():
    described = OllamaBackend("qwen3-vl:8b").describe()
    assert described["model"] == "qwen3-vl:8b"
    assert described["local"] is True


def test_a_stub_is_deterministic():
    first = StubBackend().ask(b"", "same question")
    second = StubBackend().ask(b"", "same question")
    assert first == second


def test_question_serialises_layas_wire_shape():
    payload = Question(
        "dept", "choice", "Which team?", {"billing": "money", "other": "the rest"}
    ).to_dict()
    assert payload == {
        "type": "choice",
        "instructions": "Which team?",
        "criteria": {"billing": "money", "other": "the rest"},
    }
    # A question with no criteria must not send a null one: Laya validates it.
    assert Question("urgent", "noul", "Is it urgent?").to_dict() == {
        "type": "noul",
        "instructions": "Is it urgent?",
    }
    assert Question("sev", "score", "How bad?", ["mild", "bad"]).to_dict()["criteria"] == [
        "mild",
        "bad",
    ]


# -------------------------------------------------------------------------- laya


def test_laya_parses_a_noul_as_p_true_not_as_the_reported_side():
    # answer_confidence on a noul is max(p_true, 1 - p_true) — the probability of
    # the side being reported. Gating on that would make a confident "false"
    # indistinguishable from a confident "true", which is the exact bug the
    # probability/threshold split exists to avoid.
    question = Question("worth", "noul", "act on this?")
    confident_no = LayaBackend._decision(
        question, {"type": "noul", "noul": 0.04, "confidence": 0.96, "answer_confidence": 0.96}
    )
    assert confident_no.probability == pytest.approx(0.04)
    assert confident_no.value == pytest.approx(0.04)
    assert confident_no.label == "false"
    assert confident_no.confidence == pytest.approx(0.96)

    confident_yes = LayaBackend._decision(
        question, {"type": "noul", "noul": 0.93, "confidence": 0.93, "answer_confidence": 0.93}
    )
    assert confident_yes.probability == pytest.approx(0.93)
    assert confident_yes.label == "true"


def test_laya_parses_a_choice_on_answer_confidence():
    question = Question("dept", "choice", "which?")
    decision = LayaBackend._decision(
        question,
        {"type": "choice", "choice": "billing", "confidence": 0.79, "answer_confidence": 0.94},
    )
    assert decision.label == "billing"
    # Gate on answer_confidence, keep the entropy number for diagnosis — they are
    # different scales and Laya's docs say so explicitly.
    assert decision.probability == pytest.approx(0.94)
    assert decision.confidence == pytest.approx(0.79)


def test_laya_parses_a_score_as_an_expected_level():
    question = Question("sev", "score", "how bad?", ["mild", "bad", "critical"])
    decision = LayaBackend._decision(
        question, {"type": "score", "score": 2.9, "answer_confidence": 0.79}
    )
    assert decision.value == pytest.approx(2.9)
    # A score is an expected value, not the most likely level; nothing here
    # rounds it into a label.
    assert decision.label is None


def test_laya_survives_an_answer_it_does_not_recognise():
    decision = LayaBackend._decision(Question("q", "noul", "?"), None)
    assert decision.probability == 0.0
    assert decision.label is None


def test_laya_reports_itself_as_non_generating():
    assert LayaBackend().describe()["generates"] is False


def test_laya_decide_posts_to_its_own_wire_path(monkeypatch):
    sent: dict = {}

    def fake_post(url, payload, *, timeout, api_key=None):
        sent["url"] = url
        sent["payload"] = payload
        return {"answers": {"worth": {"type": "noul", "noul": 0.8, "confidence": 0.8}}}

    monkeypatch.setattr("agent_visual_harness.vision._post_json", fake_post)
    decisions = LayaBackend("http://127.0.0.1:9999/").decide(
        {"on_screen": "declined"}, [Question("worth", "noul", "act?")]
    )

    assert sent["url"] == "http://127.0.0.1:9999/v1/systemone"
    assert sent["payload"]["state"] == {"on_screen": "declined"}
    assert sent["payload"]["questions"] == {"worth": {"type": "noul", "instructions": "act?"}}
    assert decisions[0].probability == pytest.approx(0.8)


def test_laya_rejects_a_response_with_no_answers(monkeypatch):
    monkeypatch.setattr(
        "agent_visual_harness.vision._post_json",
        lambda url, payload, *, timeout, api_key=None: {"detail": "unprocessable"},
    )
    with pytest.raises(VisionError):
        LayaBackend().decide("state", [Question("worth", "noul", "act?")])


# ---------------------------------------------------------------- counted tokens


def test_the_count_comes_from_the_model_and_not_the_geometry(monkeypatch):
    """A zero estimate beside a real count is the whole point.

    ``tokens_after`` is derived from pixel area, so an empty image estimates
    zero however the model was actually billed. The two numbers answer different
    questions and are kept apart on purpose; this asserts the second one cannot
    have leaked in from the first.
    """
    monkeypatch.setattr(
        "agent_visual_harness.vision._post_json",
        lambda url, payload, *, timeout, api_key=None: {
            "response": "a decline notice",
            "prompt_eval_count": 1089,
        },
    )
    report = Triage(OllamaBackend()).run(
        [frame("a", regions=[(0, 0, 10, 10)])], "what happened?"
    )

    answer = report.answers[0]
    assert answer.text == "a decline notice"
    assert answer.prompt_tokens == 1089
    assert answer.tokens_after == 0
    assert report.prompt_tokens == 1089


def test_a_server_that_omits_the_count_leaves_it_unset(monkeypatch):
    monkeypatch.setattr(
        "agent_visual_harness.vision._post_json",
        lambda url, payload, *, timeout, api_key=None: {"response": "ok"},
    )
    backend = OllamaBackend()
    backend.ask(b"", "q")

    # None, not 0: "the server did not say" and "the server said zero" are
    # different facts and only one of them is a measurement.
    assert backend.last_prompt_tokens is None


def test_a_stub_claims_nothing_about_tokens():
    report = Triage(StubBackend()).run(
        [frame("a", regions=[(0, 0, 10, 10)])], "q"
    )

    assert report.answers[0].prompt_tokens is None
    assert report.prompt_tokens is None
    assert "prompt_tokens" in report.answers[0].to_dict()
    assert "prompt_tokens" in report.to_dict()


def test_laya_records_the_tokens_it_read(monkeypatch):
    monkeypatch.setattr(
        "agent_visual_harness.vision._post_json",
        lambda url, payload, *, timeout, api_key=None: {
            "answers": {"worth": {"type": "noul", "noul": 0.7, "confidence": 0.7}},
            "usage": {"input_tokens": 42},
        },
    )
    backend = LayaBackend()
    backend.decide("state", [Question("worth", "noul", "act?")])

    assert backend.last_prompt_tokens == 42


def test_the_summary_prints_the_count_beside_the_estimate(monkeypatch):
    monkeypatch.setattr(
        "agent_visual_harness.vision._post_json",
        lambda url, payload, *, timeout, api_key=None: {
            "response": "x",
            "prompt_eval_count": 1089,
        },
    )
    report = Triage(OllamaBackend()).run(
        [frame("a", regions=[(0, 0, 10, 10)])], "q"
    )
    summary = report.summary()

    assert "1,089" in summary
    assert "prompt tokens" in summary
    assert "estimate" in summary


# ---------------------------------------------------------------------- report


def test_report_round_trips_through_json():
    report = Triage(StubBackend()).run(
        [frame("a", regions=[(10, 10, 100, 40)]), frame("b")], "what happened?"
    )
    restored = json.loads(json.dumps(report.to_dict()))
    assert restored["frames"] == 2
    assert restored["model_calls"] == 1
    assert restored["calls_avoided"] == 1
    assert len(restored["answers"]) == 2


def test_report_summary_states_what_it_did_not_send():
    report = Triage(StubBackend()).run([frame("quiet")], "q")
    summary = report.summary()
    assert "never sent" in summary
    assert "frames    1" in summary
