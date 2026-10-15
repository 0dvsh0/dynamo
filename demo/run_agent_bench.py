#!/usr/bin/env python3
"""Price one real question against a whole recorded session, three ways.

This is the benchmark for the model-facing layer: the same 29 keyframes, the
same question, three increasingly cheap ways of answering it.

    whole     every keyframe sent whole, no filter, no crop   (the naive local agent)
    cropped   every keyframe cropped to what moved             (the crop, alone)
    triaged   skip unchanged -> Laya scores -> crop -> Qwen3-VL (the whole design)

It reads the keyframes the reducer already produced (``assets/dynamo_frames``)
and re-derives "what moved" by diffing consecutive frames — so it needs ffmpeg
and the model servers, but no Xvfb, no browser and no recording. That keeps the
number reproducible on a laptop.

    # the generative half: an open-weight VLM, local
    ollama serve &
    ollama pull qwen3-vl:4b

    # the filter half is optional: Laya, on its own server, at POST /v1/systemone
    # (pip install "laya[serve]"; run laya-serve). Without it the run still
    # measures the crop and the model, and says the filter step was not exercised.
    python3 demo/run_agent_bench.py --out demo

Three hard rules, all about not publishing a number that was not measured:

* Without ``--stub`` the script **refuses to write a file** unless a real model
  answered the frames. A benchmark that silently falls back to a stub and reports
  its numbers as measurements is worse than no benchmark.
* A filter that is not answering is recorded as **absent**, never as a stub — so
  the routing column cannot be quietly invented. (``--stub`` is the exception, and
  it writes ``measured: false`` for exactly that reason.)
* With ``--stub`` it writes ``measured: false`` and says why, so the page can
  state that plainly instead of implying otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

# Windows consoles default to cp1252 and cannot encode the arrows in the progress
# lines. Reconfigure before anything is printed, or a long run dies on its own
# output — which is a spectacularly annoying way to lose an hour of inference.
for _stream in (sys.stdout, sys.stderr):
    _reconfigure = getattr(_stream, "reconfigure", None)
    if _reconfigure is not None:
        try:
            _reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # pragma: no cover - detached stream
            pass

sys.path.insert(0, str(REPO / "src"))

from agent_visual_harness import (  # noqa: E402
    DEFAULT_LAYA_ENDPOINT,
    DEFAULT_OLLAMA_ENDPOINT,
    DEFAULT_VLM_MODEL,
    MAX_STATE_CHARS,
    TOKENS_PER_MEGAPIXEL,
    Frame,
    HarnessError,
    LayaBackend,
    OllamaBackend,
    StubBackend,
    Triage,
    estimate_tokens,
    frames_from_manifest,
    have_tesseract,
)
FRAMES_DIR = HERE / "assets" / "dynamo_frames"
MANIFEST = FRAMES_DIR / "manifest.json"
OUTPUT = HERE / "data" / "agent.json"

#: One question, asked of every frame in every configuration, so the only thing
#: that varies between columns is how the frame got there.
QUESTION = (
    "What went wrong on this screen, and which step of the checkout does it "
    "belong to? Answer in one sentence."
)

#: The filter's threshold and the region size below which a diff is treated as
#: glyph-level noise rather than as a reason to look. Both are the defaults
#: Triage ships with; they are named here so the page can state them.
FILTER_THRESHOLD = 0.5
MIN_REGION_PX = 400


def load_frames(limit: int | None = None, *, ocr: bool = False) -> tuple[list[Frame], dict, dict]:
    """Build Frames from the reducer's keyframes, with regions from a real diff.

    The frame loading itself lives in the package (``frames_from_manifest``) so
    ``dynamo triage`` and this benchmark cannot diverge on what a frame is. What
    stays here is the timing, because "reading the screen costs this much and
    involves no model" is a number this page argues from.
    """
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    started = time.monotonic()
    frames = frames_from_manifest(FRAMES_DIR, limit=limit, ocr=ocr)
    load_seconds = time.monotonic() - started

    ocr_info = {
        "available": have_tesseract(),
        "used": bool(ocr and have_tesseract()),
        "load_seconds": round(load_seconds, 3),
        "frames_with_text": sum(1 for f in frames if f.text),
        "max_state_chars": MAX_STATE_CHARS,
    }
    return frames, manifest, ocr_info


def probe_backends(model: str, ollama_endpoint: str, laya_endpoint: str) -> tuple[bool, bool]:
    """Whether each model server is actually answering. Never raises."""
    ollama = OllamaBackend(model, ollama_endpoint)
    reachable = ollama.available()
    if reachable and ollama.models():
        # A server that is up but has never pulled the model is not usable, and
        # saying so here beats a 404 from the first frame.
        reachable = any(model.split(":")[0] in name for name in ollama.models())
    return reachable, LayaBackend(laya_endpoint).available()


def run_config(name: str, frames: list[Frame], *, primary, filter_backend, crop: bool,
               skip_unchanged: bool, progress: bool = True) -> tuple[dict, list[dict]]:
    """Run one configuration over the frames and return its numbers."""
    triage = Triage(
        primary,
        filter=filter_backend,
        crop=crop,
        skip_unchanged=skip_unchanged,
        filter_threshold=FILTER_THRESHOLD,
        min_region_px=MIN_REGION_PX,
    )

    def announce(answer, index: int, total: int) -> None:
        # A 4B VLM on a CPU is tens of seconds a frame. Without this the run is
        # an hour of silence, and silence is indistinguishable from a hang.
        if answer.skipped:
            what = f"skipped — {answer.skipped_reason}"
        else:
            what = f"{answer.tokens_before:>6} → {answer.tokens_after:>6} tok"
            what += ", cropped" if answer.cropped else "        "
            what += f", {answer.elapsed:5.1f}s"
        print(f"  [{index:>3}/{total}] {answer.frame:<20} {what}", flush=True)

    started = time.monotonic()
    report = triage.run(frames, QUESTION, on_answer=announce if progress else None)
    wall = time.monotonic() - started

    payload = report.to_dict()
    payload["wall_seconds"] = round(wall, 3)
    payload["name"] = name
    # The page prints the report's own words rather than a re-typed copy of
    # them, so the summary on the page cannot drift from the summary the library
    # produces.
    payload["summary"] = report.summary()
    return payload, [a.to_dict() for a in report.answers]


def state_example(frames: list[Frame], answers: list[dict]) -> dict | None:
    """What the filter is actually told about one real frame.

    Printed rather than described because "the filter decides over the harness's
    own observation" is a claim a reader can only check by seeing the object.
    """
    probe = Triage(StubBackend())
    for frame, answer in zip(frames, answers):
        if answer.get("region"):
            return {"frame": frame.name, "state": probe.state_for(frame)}
    if frames:
        return {"frame": frames[0].name, "state": probe.state_for(frames[0])}
    return None


def crop_example(frames: list[Frame], answers: list[dict]) -> dict | None:
    """One concrete crop, for the page to show instead of a summary statistic."""
    for frame, answer in zip(frames, answers):
        region = answer.get("region")
        if not region:
            continue
        from agent_visual_harness import image_size

        whole = image_size(frame.image)
        return {
            "frame": frame.name,
            "whole": [whole[0], whole[1]],
            "region": list(region),
            "megapixels_whole": round(whole[0] * whole[1] / 1e6, 3),
            "megapixels_crop": round(region[2] * region[3] / 1e6, 3),
            "tokens_whole": answer["tokens_before"],
            "tokens_crop": answer["tokens_after"],
            "action": frame.action,
        }
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(HERE), help="demo directory to populate")
    parser.add_argument("--model", default=DEFAULT_VLM_MODEL, help="open-weight VLM to use")
    parser.add_argument("--ollama", default=DEFAULT_OLLAMA_ENDPOINT)
    parser.add_argument("--laya", default=DEFAULT_LAYA_ENDPOINT)
    parser.add_argument("--limit", type=int, default=None, help="use only the first N frames")
    parser.add_argument("--ocr", action="store_true",
                        help="read each keyframe with tesseract so the filter gets real "
                             "screen text instead of only the action label")
    parser.add_argument("--stub", action="store_true",
                        help="run the pipeline without a model; writes measured: false")
    parser.add_argument("--quiet", action="store_true",
                        help="do not print a line per frame")
    args = parser.parse_args(argv)

    out = Path(args.out)
    (out / "data").mkdir(parents=True, exist_ok=True)

    print(f"frames   {FRAMES_DIR}")
    frames, manifest, ocr_info = load_frames(args.limit, ocr=args.ocr)
    print(f"loaded   {len(frames)} keyframes "
          f"({sum(1 for f in frames if f.is_anchor)} action anchors, "
          f"{sum(1 for f in frames if f.has_change)} with changed regions)")
    if ocr_info["used"]:
        print(f"ocr      {ocr_info['frames_with_text']} of {len(frames)} frames read "
              f"in {ocr_info['load_seconds']}s, no model involved")

    if args.stub:
        primary_real = filter_real = False
        why = "--stub was passed: no model answered these frames"
    else:
        print("probing  ollama and laya ...")
        primary_real, filter_real = probe_backends(args.model, args.ollama, args.laya)
        why = None
        if not primary_real:
            why = (f"no model answered at {args.ollama} for {args.model} "
                   f"(start it, or run with --stub)")
        if why:
            print(f"\nrefusing to write {OUTPUT.name}: {why}", file=sys.stderr)
            print("A benchmark that silently stubs its model and reports the result "
                  "as a measurement is worse than no benchmark.", file=sys.stderr)
            print("Pass --stub to exercise the pipeline and record measured: false.",
                  file=sys.stderr)
            return 2
        if not filter_real:
            # Not fatal. The crop and the model are still being measured; the
            # filter step simply did not run, and is recorded as such rather than
            # as a stub that "passed everything".
            print("filter   no Laya server answered at "
                  f"{args.laya} — the filter step is skipped and recorded as "
                  "not exercised")

    measured = bool(primary_real)
    filter_measured = bool(filter_real and measured)

    def primary_for() -> object:
        if primary_real:
            return OllamaBackend(args.model, args.ollama)
        return StubBackend(
            name="stub-primary",
            answers={QUESTION: "stub: no model answered these frames"},
        )

    def filter_for() -> object | None:
        """The decision backend, or ``None`` when nothing can decide.

        ``None``, not a stub: a stubbed filter would invent the routing numbers
        this benchmark exists to measure. ``--stub`` is the one exception, and it
        is labelled at every level.
        """
        if filter_measured:
            return LayaBackend(args.laya)
        if args.stub:
            return StubBackend(name="stub-filter", decision_probability=0.9)
        return None

    configurations = {
        "whole": dict(crop=False, skip_unchanged=False, use_filter=False),
        "cropped": dict(crop=True, skip_unchanged=False, use_filter=False),
        "triaged": dict(crop=True, skip_unchanged=True, use_filter=True),
    }

    workflows: dict[str, dict] = {}
    triaged_answers: list[dict] = []
    for name, spec in configurations.items():
        print(f"running  {name} ...")
        payload, answers = run_config(
            name,
            frames,
            primary=primary_for(),
            filter_backend=filter_for() if spec["use_filter"] else None,
            crop=spec["crop"],
            skip_unchanged=spec["skip_unchanged"],
            progress=not args.quiet,
        )
        workflows[name] = payload
        if name == "triaged":
            triaged_answers = answers

    whole, triaged = workflows["whole"], workflows["triaged"]
    headline = {
        "uncropped_tokens": whole["tokens_before"],
        "whole_tokens": whole["tokens_after"],
        "cropped_tokens": workflows["cropped"]["tokens_after"],
        "triaged_tokens": triaged["tokens_after"],
        "frames_to_model_naive": whole["model_calls"],
        "frames_to_model_triaged": triaged["model_calls"],
        "calls_avoided": triaged["calls_avoided"],
        "crop_example": crop_example(frames, triaged_answers),
        "state_example": state_example(frames, triaged_answers),
    }
    if whole["tokens_after"]:
        headline["triaged_vs_whole_pct"] = round(
            100.0 * triaged["tokens_after"] / whole["tokens_after"], 2
        )

    notes = [
        "The keyframe set is already reduced by the token reducer, so most frames "
        "here are action anchors — which are never skipped on purpose. The "
        "'unchanged' saving bites on raw frames, not on a reduced set.",
        "Token counts use a published per-megapixel price list "
        f"({TOKENS_PER_MEGAPIXEL} tokens/MP). They are a model of a price list, "
        "not a tokenizer measurement.",
    ]
    if not filter_measured:
        notes.append(
            "The triaged column ran without its filter step, so it is identical to "
            "the cropped column by construction. Nothing here measured what a "
            "decision model would drop."
        )
    elif ocr_info["used"]:
        notes.append(
            "The filter was given OCR text read off each frame, so it judges the "
            "screen's words."
        )
    else:
        notes.append(
            "The filter was given the action label only — no OCR ran, so it never "
            "saw the screen's text. Pass --ocr to change that."
        )
    notes.append(
        "The filter is a text decision model: it does not see pixels at all, which "
        "is why its input is the harness's own observation rather than the image."
    )

    # What the run would have used, recorded even when a stub answered — so the
    # page can name the model it is about without implying that model ran.
    primary_meta = primary_for().describe()
    primary_meta.setdefault("model", args.model)
    primary_meta.setdefault("endpoint", args.ollama)
    if filter_measured:
        filter_meta = filter_for().describe()
    else:
        filter_meta = {
            "name": "no filter",
            "endpoint": args.laya,
            "local": True,
            "generates": False,
            "present": False,
            "reason": (
                "--stub was passed"
                if args.stub
                else f"no Laya server answered at {args.laya}"
            ),
        }

    results = {
        "meta": {
            "measured": measured,
            "filter_measured": filter_measured,
            "generated_by": "demo/run_agent_bench.py",
            "question": QUESTION,
            "frames": len(frames),
            "action_anchors": sum(1 for f in frames if f.is_anchor),
            "frames_with_regions": sum(1 for f in frames if f.has_change),
            "source": f"assets/dynamo_frames — {len(frames)} keyframes the reducer kept",
            "keyframe_size": f"{manifest['source_size']['width']}x"
                             f"{manifest['source_size']['height']} recording, "
                             f"{manifest['config']['max_size']}px keyframes",
            "primary": primary_meta,
            "filter": filter_meta,
            "escalation": None,
            "tokens_per_megapixel": TOKENS_PER_MEGAPIXEL,
            "filter_threshold": FILTER_THRESHOLD,
            "min_region_px": MIN_REGION_PX,
            "ocr": ocr_info,
            "why_not_measured": why,
        },
        "workflows": workflows,
        "headline": headline,
        "per_frame": triaged_answers,
        "notes": notes,
    }

    target = out / "data" / "agent.json"
    target.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    print()
    print(f"{'whole frames':<22}{whole['tokens_after']:,} tokens  "
          f"{whole['model_calls']} model calls")
    print(f"{'cropped':<22}{workflows['cropped']['tokens_after']:,} tokens  "
          f"{workflows['cropped']['model_calls']} model calls  "
          f"({workflows['cropped']['cropped']} cropped)")
    print(f"{'triaged':<22}{triaged['tokens_after']:,} tokens  "
          f"{triaged['model_calls']} model calls  "
          f"({triaged['calls_avoided']} never sent)")
    if headline.get("triaged_vs_whole_pct") is not None:
        print(f"{'triaged vs whole':<22}{headline['triaged_vs_whole_pct']}% of the tokens")
    print(f"{'measured':<22}{measured}")
    if not measured:
        print(f"\nNOT A MEASUREMENT: {why}")
    print(f"results  : {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
