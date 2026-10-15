#!/usr/bin/env python3
"""Verify demo/index.html against demo/data/results.json.

The page is generated, but generation is not proof: this walks every asset it
references, confirms the stills and filmstrips match what the run produced, and
checks that the load-bearing claims on the page are the measured ones.

    python3 demo/check_page.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# Windows consoles default to cp1252 and cannot encode the arrows and box
# characters this report prints. Reconfigure before anything is written, or the
# checker fails on the very output that says everything passed.
for _stream in (sys.stdout, sys.stderr):
    _reconfigure = getattr(_stream, "reconfigure", None)
    if _reconfigure is not None:
        try:
            _reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # pragma: no cover - detached stream
            pass

HERE = Path(__file__).resolve().parent
PAGE = HERE / "index.html"
RESULTS = HERE / "data" / "results.json"
INSPECTION = HERE / "data" / "inspection.json"
AGENT = HERE / "data" / "agent.json"
MANIFEST = HERE / "assets" / "manifest.json"


def check_inspection(html: str, problems: list[str]) -> list[str]:
    """Verify the inspection section against its own measurements.

    These claims are the ones a reader is most likely to take on trust — a
    recall percentage, a drift in pixels, a click that either landed or did not
    — so each is checked back against the run that produced it.
    """
    if not INSPECTION.exists():
        problems.append("missing data/inspection.json (run demo/run_inspection.py)")
        return []
    insp = json.loads(INSPECTION.read_text(encoding="utf-8"))

    def must(value: str, label: str) -> None:
        if value and value not in html:
            problems.append(f"page does not state {label}: {value!r}")

    notes: list[str] = []

    cap = insp["capture"]
    must(f"{cap['full']['bytes']:,}", "full screenshot bytes")
    must(f"{cap['snip']['bytes']:,}", "snip bytes")
    must(f"{cap['full']['tokens_flat']:,}", "flat token price")
    must(f"{cap['full']['tokens_sized']:,}", "sized token price")
    must(f"{cap['snip']['tokens_sized']:,}", "snip sized tokens")
    notes.append(
        f"snip              : {cap['snip']['bytes']:,} B vs {cap['full']['bytes']:,} B"
    )
    # The page argues flat pricing gives snips no credit; that must stay true.
    if cap["snip"]["tokens_flat"] != cap["full"]["tokens_flat"]:
        problems.append(
            "capture result changed: the flat per-image price now differs for a "
            "snip, so the page's 'flat pricing saves nothing' claim is stale"
        )

    ocr = insp["ocr"]
    if ocr["dom_has_text"]:
        problems.append(
            "OCR benchmark is invalid: the canvas now exposes text to the DOM, "
            "so this is no longer a test of reading pixels"
        )
    must(f"{ocr['score_panel']['recall_strict'] * 100:.0f}%", "OCR strict recall")
    must(f"{ocr['score_panel']['recall_folded'] * 100:.0f}%", "OCR folded recall")
    must(f"{ocr['elapsed_panel']:.2f}s", "OCR panel timing")
    must(ocr["decline_line_raw"], "the raw OCR transcription of the decline line")
    notes.append(
        f"ocr               : {ocr['score_panel']['recall_strict'] * 100:.0f}% strict"
        f" → {ocr['score_panel']['recall_folded'] * 100:.0f}% folded"
    )
    if ocr["score_panel"]["recall_folded"] < ocr["score_panel"]["recall_strict"]:
        problems.append("folding reduced OCR recall — it can only widen a match")

    diff = insp["diff"]
    if diff["region_count"] == 0:
        problems.append(
            f"diff found {diff['full']['changed_pixels']:,} changed pixels but "
            "reported 0 regions — the region grouper is dropping runs again"
        )
    if diff["control"]["changed_pixels"] != 0:
        problems.append("diff control (image vs itself) is not identical")
    must(f"{diff['full']['changed_pct']:.3f}%", "diff changed percentage")
    notes.append(f"diff              : {diff['full']['changed_pixels']:,} px"
                 f" in {diff['region_count']} regions, control 0")

    som = insp["grounding"]
    if som["marks"] == 0:
        problems.append("grounding drew no marks")
    must(f"({som['resolved_point'][0]}, {som['resolved_point'][1]})", "resolved click point")
    must(f"({som['resolved_point_image'][0]}, {som['resolved_point_image'][1]})",
         "resolved click point in image space")

    # The resolved point must land inside the element it names, in *image* space.
    # resolve() returns pointer space, so compare against the image-space point —
    # a mismatch here is the "transform applied twice" bug, which is otherwise
    # silent: the page still shows a plausible-looking pair of numbers.
    px, py = som["resolved_point_image"]
    bx, by, bw, bh = som["device_box"]
    if not (bx <= px <= bx + bw and by <= py <= by + bh):
        problems.append(
            f"grounding resolved to ({px}, {py}) in image space, outside the "
            f"element it names ({bx}, {by}, {bw}x{bh}) — a coordinate transform "
            "is being applied twice, or clipped"
        )
    if som["resolved_point"] == som["resolved_point_image"]:
        problems.append(
            "grounding returned the same point in both spaces — no transform "
            "was applied, so the page's origin argument no longer holds"
        )
    for item in som["skipped_detail"]:
        if item["box"][2] != 0 and item["box"][3] != 0:
            problems.append(
                f"grounding skipped {item['label']!r} whose box has area — it "
                "should have been drawn"
            )
    for item in som["clipped_at_top_detail"]:
        if item["box"][2] != 0 and item["box"][3] != 0:
            problems.append(
                f"clipped-element probe counted {item['label']!r} which has area"
            )
    notes.append(f"grounding         : {som['marks']} marks, "
                 f"{som['clipped_at_top']} clipped before scrolling, index "
                 f"{som['answer']} -> "
                 f"({som['resolved_point'][0]}, {som['resolved_point'][1]}) pointer / "
                 f"({som['resolved_point_image'][0]}, {som['resolved_point_image'][1]}) image")

    hidpi = insp["hidpi"]
    if hidpi["naive_click_landed"]:
        problems.append(
            "HiDPI experiment: the naive click landed, so the page's 'it misses "
            "silently' claim no longer holds"
        )
    if not hidpi["fixed_click_landed"]:
        problems.append(
            "HiDPI experiment: the normalised click did not land, so the fix is "
            "not actually fixing anything"
        )
    must(f"{hidpi['image_size_2x'][0]}×{hidpi['image_size_2x'][1]}", "the 2x image size")
    notes.append(f"hidpi             : drift {hidpi['drift_px']}, naive="
                 f"{hidpi['naive_click_landed']} normalised={hidpi['fixed_click_landed']}")

    # Every image the section shows must actually be on disk and on the page.
    for name in insp["assets"]:
        if not (HERE / "assets" / "inspection" / name).exists():
            problems.append(f"inspection artifact missing on disk: {name}")
    shown = len(re.findall(r"assets/inspection/[a-z0-9_]+\.png", html))
    if shown < 6:
        problems.append(f"only {shown} inspection images on the page")
    notes.append(f"inspection images : {shown} shown of {len(insp['assets'])} produced")

    for needle, label in (
        ("capture", "the capture command"),
        ("dynamo ocr", "the ocr command"),
        ("dynamo diff", "the diff command"),
        ("devicePixelRatio", "the HiDPI condition"),
        ("canvas", "the canvas surface"),
    ):
        if needle not in html:
            problems.append(f"page missing {label}")

    return notes


def check_agent(html: str, problems: list[str]) -> list[str]:
    """Verify the model-triage section against its own measurements.

    The load-bearing claim in this section is a *disclaimer*. Everything else on
    the page a reader could in principle re-measure; whether a model actually
    answered is the one thing they cannot see, so when the run wrote
    ``measured: false`` the page has to say so, and this fails the build if it
    does not.
    """
    if not AGENT.exists():
        problems.append("missing data/agent.json (run demo/run_agent_bench.py)")
        return []
    agent = json.loads(AGENT.read_text(encoding="utf-8"))
    meta = agent["meta"]
    notes: list[str] = []

    def must(value, label: str) -> None:
        text = str(value)
        if text and text not in html:
            problems.append(f"page does not state {label}: {text!r}")

    measured = bool(meta.get("measured"))
    filter_measured = bool(meta.get("filter_measured"))
    if measured:
        if "Measured on this machine" not in html:
            problems.append(
                "agent.json says a model measured this section, but the page does "
                "not say so — the measurement banner is missing"
            )
    else:
        if "Not measured" not in html:
            problems.append(
                "agent.json says measured: false and the page does not say so — "
                "this is the one claim on the page a reader cannot check"
            )
        if str(meta.get("why_not_measured") or "") not in html:
            problems.append("the page does not say why the section was not measured")
        if "Measured on this machine" in html:
            problems.append("page claims a model measured a run that used a stub")
        # The measured columns and the stubbed ones must be told apart in words,
        # not just implied by a flag.
        if "routing column is not" not in html:
            problems.append(
                "measured: false, but the page does not distinguish the routing "
                "column from the measured ones"
            )
    must(meta["primary"].get("model", ""), "the model the section is about")

    headline = agent["headline"]
    workflows = agent["workflows"]

    if measured and not filter_measured:
        # The filter is recorded as absent, so the page has to admit it — and the
        # two columns have to actually be the same numbers, or one of them is not
        # what it says it is.
        if "filter step did not run" not in html:
            problems.append(
                "the filter did not answer, but the page does not say the filter "
                "step was skipped"
            )
        if workflows["triaged"]["tokens_after"] != workflows["cropped"]["tokens_after"]:
            problems.append(
                "the filter is recorded as absent, yet the triaged and cropped "
                "columns differ — one of them is not what it claims to be"
            )

    for name, label in (("whole", "every keyframe whole"),
                        ("cropped", "cropped"),
                        ("triaged", "triaged")):
        wf = workflows[name]
        must(f"{wf['tokens_after']:,.0f}", f"{label} token count")
        must(str(wf["model_calls"]), f"{label} model calls")
        notes.append(
            f"agent {label:<22}: {wf['tokens_after']:>7,} tokens, "
            f"{wf['model_calls']:>2} calls, {wf['cropped']:>2} cropped, "
            f"{wf['token_ratio'] * 100:5.1f}% of whole"
        )

    must(f"{headline['triaged_vs_whole_pct']:.1f}%", "triaged share of the whole-screen price")

    example = headline.get("crop_example") or {}
    if not example:
        problems.append("agent.json has no crop example for the page to show")
    else:
        must(f"{example['megapixels_whole']:.3f}", "whole-frame megapixels")
        must(f"{example['megapixels_crop']:.3f}", "cropped megapixels")
        must(f"{example['tokens_whole']:,.0f}", "whole-frame token price")
        must(f"{example['tokens_crop']:,.0f}", "cropped token price")
        box = example["region"]
        if box[2] <= 0 or box[3] <= 0:
            problems.append("the crop example has no area")
        if box[2] > example["whole"][0] or box[3] > example["whole"][1]:
            problems.append("the crop example is larger than the frame it came from")
        if box[0] + box[2] > example["whole"][0] or box[1] + box[3] > example["whole"][1]:
            problems.append("the crop example leaves the frame it came from")
        notes.append(
            f"agent crop example            : {example['whole'][0]}×{example['whole'][1]} "
            f"{example['tokens_whole']} tok → {box[2]}×{box[3]} {example['tokens_crop']} tok"
        )

    # A crop can only ever remove pixels from the payload. If a cropped column
    # ever costs more than the whole one, the geometry is wrong somewhere.
    if workflows["cropped"]["tokens_after"] > workflows["whole"]["tokens_after"]:
        problems.append("cropping raised the token count — the crop geometry is wrong")
    if workflows["triaged"]["tokens_after"] > workflows["whole"]["tokens_after"]:
        problems.append("triage raised the token count above sending every frame whole")

    for needle, label in (
        ("Triage", "the Triage entry point"),
        ("LayaBackend", "the filter backend"),
        ("OllamaBackend", "the open-weight backend"),
        ("report.summary()", "the report's own summary"),
        ("Apache-2.0", "the licence of the weights"),
    ):
        if needle not in html:
            problems.append(f"page missing {label}")

    return notes


def main() -> int:
    html = PAGE.read_text(encoding="utf-8")
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    problems: list[str] = []

    # ------------------------------------------------------------- assets
    refs = sorted(set(re.findall(r'(?:src|href)="(?!#|https?:)([^"]+)"', html)))
    for ref in refs:
        if not (HERE / ref).exists():
            problems.append(f"missing asset: {ref}")
    print(f"assets referenced : {len(refs)}")

    for name in ("repro.mp4", "naive.mp4", "dynamo_crash.mp4", "dynamo_demo.gif",
                 "diy_naive_ok.gif", "manifest.json", "telemetry.jsonl",
                 "telemetry_crash.jsonl", "repro_poster.jpg", "naive_poster.jpg",
                 "crash_poster.jpg"):
        path = HERE / "assets" / name
        if not path.exists() or path.stat().st_size == 0:
            problems.append(f"empty or missing artifact: {name}")
    print("artifacts present : 11")

    # --------------------------------------------------------- filmstrips
    for prefix, expected in data["_strips"].items():
        shown = len(re.findall(rf'assets/{prefix}_\d+\.jpg', html))
        if shown != expected:
            problems.append(f"{prefix}: page shows {shown} thumbs, run produced {expected}")
    total_strip = sum(data["_strips"].values())
    print(f"filmstrips        : {total_strip} thumbnails across {len(data['_strips'])} sets")

    # Duplicate flags must match the measured duplicates, exactly.
    expected_dup = sum(
        data[key]["redundancy"]["duplicates"] for key in ("screenshots", "diy", "dynamo")
    )
    shown_dup = len(re.findall(r'class="fr dup"', html))
    if shown_dup != expected_dup:
        problems.append(
            f"{shown_dup} frames flagged duplicate on the page, data says {expected_dup}"
        )
    total_frames = sum(data["_strips"].values())
    shown_frames = len(re.findall(r'class="fr (?:dup|keep)"', html))
    if shown_frames != total_frames:
        problems.append(f"{shown_frames} frames on page, run produced {total_frames}")
    print(f"duplicate flags   : {shown_dup} flagged, matches the deltas")

    # ------------------------------------------------------------- stills
    stills = data.get("stills") or []
    if not stills:
        problems.append("no stills in results.json")
    for still in stills:
        if still["file"] not in html:
            problems.append(f"still not on page: {still['file']}")
    print(f"stills            : {len(stills)} gallery frames")

    # ------------------------------------------------------ numbers on page
    def must(value: str, label: str) -> None:
        if value not in html:
            problems.append(f"page does not state {label}: {value!r}")

    for key, label in (("screenshots", "screenshots"), ("diy", "diy"), ("dynamo", "dynamo")):
        r = data[key]
        must(f"{r['tokens']:,}", f"{label} tokens")
        must(f"{r['image_bytes']:,} B", f"{label} payload")
        must(f"{r['redundancy']['duplicates']}", f"{label} duplicates")
        must(f"{r['redundancy']['mean_delta']:g}", f"{label} mean delta")
        must(f"{r['redundancy']['max_delta']:g}", f"{label} max delta")

    must(f"{data['diy']['video_bytes'] / 1024:,.1f} KB", "diy video size")
    must(f"{data['dynamo']['video_bytes'] / 1024:,.1f} KB", "dynamo video size")
    must(f"{data['diy']['gif_bytes'] / 1024:,.1f} KB", "diy gif size")
    must(f"{data['dynamo']['gif_bytes'] / 1024:,.1f} KB", "dynamo gif size")
    must(f"{manifest['reduction_pct']}", "reduction percentage")
    must(f"{manifest['raw_frame_estimate']:,}", "raw frame estimate")
    must(f"{manifest['keyframes_kept']}", "keyframes kept")
    must(data["crash"]["probe"], "crash ffprobe duration")
    must(data["diy"]["crash"]["raw_error"], "the real ffprobe error")
    must(f'{data["meta"]["width"]}', "display width")

    # ------------------------------------------------------ honesty checks
    # The page must not claim a GIF win it did not measure, and must not claim
    # Dynamo beats screenshots on tokens, because it does not.
    dyn_tokens = data["dynamo"]["tokens"]
    scr_tokens = data["screenshots"]["tokens"]
    if dyn_tokens < scr_tokens:
        problems.append(
            "results changed: Dynamo now beats screenshots on tokens — the page's "
            "honesty note about that needs rewriting"
        )
    if "close to a wash" not in html and data["diy"]["gif_bytes"] < data["dynamo"]["gif_bytes"] * 1.5:
        problems.append("GIF result changed — re-check the 'close to a wash' wording")

    for needle, label in (
        ("<video", "a playable <video> element"),
        ("manifest.json", "the manifest reference"),
        ("moov atom not found", "the real ffprobe error"),
        ("telemetry_crash.jsonl", "the crash telemetry"),
        ("telemetry.jsonl", "the telemetry file"),
        ("Chromium", "the browser identity"),
    ):
        if needle not in html:
            problems.append(f"page missing {label}")

    if "{{" in html:
        problems.append("unresolved placeholder left in the page")

    for note in check_inspection(html, problems):
        print(note)
    for note in check_agent(html, problems):
        print(note)

    print()
    if problems:
        print("FAIL")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("PASS — every asset resolves; every printed number matches results.json;")
    print("       the honesty claims still hold against the data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
