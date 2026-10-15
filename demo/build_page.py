#!/usr/bin/env python3
"""Render demo/index.html from demo/index.template.html + demo/data/results.json.

The page is generated rather than hand-written so that a number on it can never
drift from the measurement that produced it. Re-run this after
`run_experiments.py`; `check_page.py` then verifies the result.

    python3 demo/build_page.py
"""

from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "index.template.html"
OUTPUT = HERE / "index.html"
RESULTS = HERE / "data" / "results.json"
INSPECTION = HERE / "data" / "inspection.json"
AGENT = HERE / "data" / "agent.json"
MANIFEST = HERE / "assets" / "manifest.json"

STRIP_BLOCK = re.compile(r"<!--STRIP:(\w+)-->.*?<!--/STRIP-->", re.S)
STILLS_BLOCK = re.compile(r"<!--STILLS-->.*?<!--/STILLS-->", re.S)
AGENT_NOTE_BLOCK = re.compile(r"<!--AGENTNOTE-->.*?<!--/AGENTNOTE-->", re.S)
AGENT_COUNTED_BLOCK = re.compile(r"<!--AGENTCOUNTED-->.*?<!--/AGENTCOUNTED-->", re.S)
STATE_BLOCK = re.compile(r"<!--STATE-->.*?<!--/STATE-->", re.S)
PLACEHOLDER = re.compile(r"\{\{([\w.]+)(?:\|(\w+))?\}\}")


# --------------------------------------------------------------------- formatting


def commas(value) -> str:
    return f"{float(value):,.0f}"


def kb(value) -> str:
    return f"{float(value) / 1024:,.1f} KB"


def bytes_kb(value) -> str:
    return f"{int(value):,} B"


def f1(value) -> str:
    return f"{float(value):.1f}"


def f3(value) -> str:
    return f"{float(value):.3f}"


def f2(value) -> str:
    return f"{float(value):.2f}"


def pct(value) -> str:
    """A value already expressed in percent."""
    return f"{float(value):.2f}%"


def frac(value) -> str:
    """A fraction (0-1) expressed as a percentage.

    Separate from :func:`pct` on purpose: guessing which one a number is has
    produced 0.75% where 75% was meant, and the whole point of generating the
    page is that a number cannot be quietly wrong.
    """
    return f"{float(value) * 100:.0f}%"


def frac1(value) -> str:
    """As :func:`frac`, but a decimal — for ratios like 4.66% that would
    otherwise round to a flat and uninformative 5%."""
    return f"{float(value) * 100:.1f}%"


def count(value) -> str:
    """Length of a list, so a page can say "2 of these" without arithmetic."""
    try:
        return f"{len(value):,}"
    except TypeError:
        return "0"


def ratio(a, b) -> str:
    return f"{float(a) / float(b):.1f}×"


def resolve(data: dict, path: str):
    node = data
    for part in path.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        elif isinstance(node, list):
            try:
                node = node[int(part)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return node


FORMATS = {"commas": commas, "kb": kb, "bytes": bytes_kb, "f1": f1,
           "f2": f2, "f3": f3, "pct": pct, "frac": frac, "frac1": frac1,
           "count": count, "ratio": ratio}


def render_value(node, fmt: str | None) -> str:
    if node is None:
        return "—"
    if fmt and fmt in FORMATS:
        try:
            return FORMATS[fmt](node)
        except (TypeError, ValueError):
            return "—"
    if isinstance(node, bool):
        return "yes" if node else "no"
    if isinstance(node, float):
        return f"{node:g}"
    return str(node)


# ------------------------------------------------------------------------- blocks


def sparkline(values: list[float], colour: str, width: int = 600, height: int = 66) -> str:
    """Frame-to-frame deltas. Exact zeros (bit-identical frames) sit on the floor."""
    if not values:
        return ""
    top = max(values) or 1.0
    step = width / max(1, len(values))
    points = " ".join(
        f"{i * step:.1f},{height - 6 - (v / top) * (height - 14):.1f}"
        for i, v in enumerate(values)
    )
    return (
        f'<svg class="spark" viewBox="0 0 {width} {height}" preserveAspectRatio="none" '
        f'aria-hidden="true"><polyline fill="none" stroke="{colour}" stroke-width="2" '
        f'stroke-linejoin="round" points="{points}"/></svg>'
    )


def strip_block(name: str, data: dict) -> str:
    """One filmstrip, with bit-identical repeats flagged where they occur."""
    key = {"screens": "screenshots", "diy": "diy", "dynamo": "dynamo"}[name]
    result = data[key]
    prefix = f"kf_{name}"
    count = data["_strips"][prefix]
    redundancy = result.get("redundancy") or {}
    deltas = redundancy.get("deltas") or []

    cards = []
    for i in range(count):
        # Each workflow's frames are numbered by its own ffmpeg pattern, so the
        # zero padding differs: screenshot_%02d, frame_%03d, frame_%04d.
        suffix = {"screens": f"{i + 1:02d}", "diy": f"{i + 1:03d}",
                  "dynamo": f"{i + 1:04d}"}[name]
        # deltas[i-1] is the change from frame i to frame i+1; frame i+1 is a
        # repeat when that change is exactly zero.
        is_dup = i > 0 and (deltas[i - 1] == 0.0)
        cls = "fr dup" if is_dup else "fr keep"
        cards.append(
            f'<div class="{cls}"><img src="assets/{prefix}_{suffix}.jpg" '
            f'alt="{key} frame {i + 1}" loading="lazy"><div class="cap">#{i + 1}</div></div>'
        )

    colours = {"screens": "#5C6673", "diy": "#FF5F5F", "dynamo": "#C8FF3D"}
    meta = (f'{commas(result["images"])} images · {commas(result["tokens"])} tokens · '
            f'{bytes_kb(result["image_bytes"])} · '
            f'{redundancy.get("duplicates", 0)} bit-identical · '
            f'mean Δ {f2(redundancy.get("mean_delta", 0))} · max Δ {f2(redundancy.get("max_delta", 0))}')
    return (f'<div class="strip-block"><div class="strip-head">'
            f'<h4>{html.escape(NAMES[key])}</h4>'
            f'<span class="meta">{meta}</span></div>'
            f'<div class="strip">{"".join(cards)}</div>'
            f'{sparkline(deltas, colours[name])}'
            f'</div>')


NAMES = {
    "screenshots": "Workflow A — screenshot after every agent action",
    "diy": "Workflow B — DIY ffmpeg, one frame per wall-clock second",
    "dynamo": "Workflow C — Dynamo token reducer",
}


def stills_block(data: dict) -> str:
    stills = data.get("stills") or []
    cards = []
    for still in stills:
        cards.append(
            f'<figure class="still"><img src="{still["file"]}" '
            f'alt="frame at {still["timestamp"]}s: {html.escape(still["action"])}" loading="lazy">'
            f'<figcaption><span class="t">{html.escape(still["action"])}</span>'
            f'<span class="ts">t = {f2(still["timestamp"])}s'
            + (f' · ({still["x"]}, {still["y"]})' if still.get("x") is not None else "")
            + '</span></figcaption></figure>'
        )
    return (f'<p class="lede">Each of these is the decoded frame at that timestamp out of the '
            f'finished recording — not a redraw. The coordinates are what telemetry logged, '
            f'and are where the click rings land in the video.</p>'
            f'<div class="stills">{"".join(cards)}</div>')


def agent_note(agent: dict) -> str:
    """The banner above the model section.

    Generated rather than hand-written for the same reason every number is: a
    page must not be able to imply a model measured something when the run that
    produced it used a stub — nor imply a filter ran when none was answering.
    Which parts of the table are real depends on what was actually up, so the
    sentence has to come out of the data.
    """
    meta = agent.get("meta") or {}
    primary = meta.get("primary") or {}
    filt = meta.get("filter") or {}
    model = html.escape(str(primary.get("model", "a local model")))
    endpoint = html.escape(str(primary.get("endpoint", "")))

    if not meta.get("measured"):
        why = html.escape(str(meta.get("why_not_measured") or "no model was available"))
        return (
            '<p class="insp-note"><b>Not measured — and the page says so.</b> The run behind '
            f'this section wrote <code>measured: false</code>: {why}. Two of the three '
            'columns are still real. <b>Tokens and crop geometry are measured</b> — the '
            'regions came from diffing the actual keyframes, and the prices come from the '
            'actual pixel area of the actual crops. <b>The routing column is not</b> — a '
            'stub filter passed every frame and a stub answered each one, so it shows what '
            'the pipeline <em>does</em>, not what a model <em>would</em> decide. Run '
            '<code>python3 demo/run_agent_bench.py</code> where Ollama is answering and '
            'this banner replaces itself.</p>'
        )

    if meta.get("filter_measured"):
        return (
            '<p class="insp-note"><b>Measured on this machine.</b> Every frame was answered '
            f'by <code>{model}</code> served by <code>{endpoint}</code>, with '
            f'<code>{html.escape(str(filt.get("name", "a filter")))}</code> deciding which '
            'frames were worth asking about. All three columns are that run\'s decisions.</p>'
        )

    reason = html.escape(str(filt.get("reason") or "no filter was answering"))
    return (
        '<p class="insp-note"><b>Measured on this machine — the model half of it.</b> Every '
        f'frame that was sent was answered by <code>{model}</code> served by '
        f'<code>{endpoint}</code>. <b>The filter step did not run</b>: {reason}. Column C '
        'is therefore identical to column B by construction, and no number here says what '
        'a decision model would drop. Start Laya and re-run to fill that in.</p>'
    )


def counted_block(agent: dict) -> str:
    """The model's own token count, printed beside the estimate.

    The table prices frames from pixel area, which is what makes a budget
    checkable before a model exists and identical on every machine. That number
    is a *model of* a price list. Ollama, asked, reports the tokens it really fed
    the model — so a run that had a model available can show both, and a reader
    can see how far to trust the first.

    Empty when no model ran: the note above already says why, and there is
    nothing honest to add here.
    """
    headline = agent.get("headline") or {}
    counted_whole = headline.get("counted_tokens_whole")
    if not counted_whole:
        return ""

    workflows = agent.get("workflows") or {}

    def num(value) -> str:
        return f"{int(value):,}" if value is not None else "—"

    estimate_whole = workflows.get("whole", {}).get("tokens_after") or 0
    estimate_triaged = workflows.get("triaged", {}).get("tokens_after") or 0
    counted_triaged = headline.get("counted_tokens_triaged")
    ratio = headline.get("counted_triaged_vs_whole_pct")

    if ratio is None:
        # A model answered but did not say what it was fed, so only the part that
        # is a count is shown. Nothing is inferred from the estimate.
        return (
            '<div class="vgrid">'
            '<div class="vstat"><span class="k">Every keyframe whole, counted by the model'
            f'</span><span class="v warn">{num(counted_whole)} tokens</span></div>'
            '<div class="vstat"><span class="k">What the table above predicted'
            f'</span><span class="v">{num(estimate_whole)} → {num(estimate_triaged)}'
            '</span></div>'
            '</div>'
        )

    return (
        '<div class="vgrid">'
        '<div class="vstat"><span class="k">Every keyframe whole, counted by the model'
        f'</span><span class="v warn">{num(counted_whole)} tokens</span></div>'
        '<div class="vstat"><span class="k">Cropped to what moved, counted by the model'
        f'</span><span class="v good">{num(counted_triaged)} tokens</span></div>'
        f'<div class="vstat"><span class="k">Counted, cropped vs whole'
        f'</span><span class="v">{ratio}%</span></div>'
        '</div>'
        '<p class="insp-note"><b>Two counts on purpose, and they differ.</b> The table '
        'prices a frame from its pixel area, which is what lets a budget be checked before '
        'a model exists and gives the same answer on every machine. These are what the '
        'model itself reported being fed — <code>prompt_eval_count</code>, straight from '
        'the server. They are not the same measurement, so they are not the same number, '
        'and both are printed because the only way to know how far to trust an estimator '
        'is to put the real figure beside it. The ratio the layer turns on comes out at '
        f'{ratio}% counted, {headline.get("triaged_vs_whole_pct")}% estimated.</p>'
    )


def state_block(agent: dict) -> str:
    """Exactly what a text decision model is handed for one real frame.

    The filter never sees pixels, and "it decides over the harness's own
    observation" is a claim a reader can only check by seeing the object — so the
    object is printed, including when it is mostly empty.
    """
    example = ((agent.get("headline") or {}).get("state_example")) or {}
    state = example.get("state")
    if state is None:
        return '<p class="insp-note">No frame state was recorded for this run.</p>'

    ocr = (agent.get("meta") or {}).get("ocr") or {}
    if ocr.get("used"):
        caption = (
            f'Tesseract read {int(ocr.get("frames_with_text", 0))} of the frames as they '
            f'were loaded, in {html.escape(str(ocr.get("load_seconds", 0)))}s — locally, no '
            'model, no network — and this is what the filter was handed. It judges text, so '
            'the screen had to be turned into text first.'
        )
    else:
        caption = (
            '<b>Note <code>on_screen</code> is empty.</b> OCR did not run on this machine, '
            'so the filter would be judging the action label and nothing about the screen '
            'itself. That is a weaker signal than the design intends, and showing it is more '
            'useful than describing it. Run with <code>--ocr</code> where tesseract is '
            'installed, or set <code>frame.text</code> yourself from whatever read the screen.'
        )

    return (
        '<div class="codecard"><div class="hd">'
        '<span>The filter\'s input, for one real frame</span>'
        f'<span>{html.escape(str(example.get("frame", "")))}</span></div>'
        f'<pre>{html.escape(json.dumps(state, indent=2, sort_keys=True))}</pre></div>'
        f'<p class="insp-note">{caption}</p>'
    )


# -------------------------------------------------------------------------- main

def main() -> int:
    if not TEMPLATE.exists():
        print(f"missing template: {TEMPLATE}", file=sys.stderr)
        return 1
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    if MANIFEST.exists():
        data["manifest"] = json.loads(MANIFEST.read_text(encoding="utf-8"))

    # The inspection capabilities are measured by a separate run, because they
    # need a different page (a canvas) and a different display scale. Merged
    # under one key so the template addresses them as insp.<capability>.
    if not INSPECTION.exists():
        print(f"missing measurements: {INSPECTION} (run demo/run_inspection.py)",
              file=sys.stderr)
        return 1
    data["insp"] = json.loads(INSPECTION.read_text(encoding="utf-8"))

    # And the model-facing layer is measured by a third run, which needs a model
    # server rather than a display. Merged under `agent` so the template
    # addresses it as agent.<workflow>.<number>.
    if not AGENT.exists():
        print(f"missing measurements: {AGENT} (run demo/run_agent_bench.py)",
              file=sys.stderr)
        return 1
    data["agent"] = json.loads(AGENT.read_text(encoding="utf-8"))

    text = TEMPLATE.read_text(encoding="utf-8")
    text = STRIP_BLOCK.sub(lambda m: strip_block(m.group(1), data), text)
    text = STILLS_BLOCK.sub(lambda _: stills_block(data), text)
    text = AGENT_NOTE_BLOCK.sub(lambda _: agent_note(data["agent"]), text)
    text = AGENT_COUNTED_BLOCK.sub(lambda _: counted_block(data["agent"]), text)
    text = STATE_BLOCK.sub(lambda _: state_block(data["agent"]), text)

    missing: list[str] = []

    def substitute(match: re.Match) -> str:
        node = resolve(data, match.group(1))
        if node is None:
            missing.append(match.group(0))
            return "—"
        return render_value(node, match.group(2))

    text = PLACEHOLDER.sub(substitute, text)

    if missing:
        print("unresolved placeholders:", file=sys.stderr)
        for item in sorted(set(missing)):
            print(f"  {item}", file=sys.stderr)
        return 1

    OUTPUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(HERE.parent)}  ({len(text):,} bytes)")
    print(f"  filmstrips : {sum(data['_strips'].values())} thumbnails")
    print(f"  stills     : {len(data.get('stills') or [])}")
    print(f"  assets     : {len(data['assets'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
