#!/usr/bin/env python3
"""Benchmark the four agent-facing visual primitives on real UI, for real.

This covers the capabilities the checkout benchmark cannot:

    capture    region snips           — static UI verification
    ocr        text off pixels         — canvas / VNC / RDP / BIOS surfaces
    diff       visual regression       — before vs after, with a diff image
    som        set-of-marks grounding  — description -> coordinates
    scale      HiDPI coordinate drift  — the silent-failure case

Everything is measured against the same real Chromium-in-a-real-X-display
setup the main benchmark uses. Nothing is simulated: the canvas log is painted
by a browser, the CSS regression is a real stylesheet change, and the grounding
boxes come out of a live DOM query.

    python3 demo/run_inspection.py --out /path/to/demo
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

from agent_visual_harness import (  # noqa: E402
    Scale,
    Surface,
    VirtualDisplay,
    annotate_boxes,
    compare,
    find,
    fold,
    image_size,
    read_image,
    read_region,
    resolve,
    resolve_box,
)
from serve import start_background  # noqa: E402

#: Same constant the main benchmark uses, so the token columns line up.
TOKENS_PER_IMAGE = 1100

#: ~765 tokens / 1000px per Anthropic's published vision pricing, rounded.
TOKENS_PER_KPX = 765

WIDTH, HEIGHT, VIEWPORT_H = 1440, 900, 800


# ------------------------------------------------------------------ utilities


def human(n: int | float) -> str:
    n = float(n)
    for unit, size in (("MB", 1e6), ("kB", 1e3)):
        if abs(n) >= size:
            return f"{n / size:.2f} {unit}"
    return f"{int(n)} B"


def cost_of(data: bytes) -> dict:
    """What this image costs to hand to a vision model.

    Two numbers, because they disagree and both matter: the flat per-image
    charge the marketing site quotes, and the size-derived one from published
    API pricing. A snip wins on both, but by very different margins.
    """
    width, height = image_size(data)
    mpx = (width * height) / 1e6
    return {
        "bytes": len(data),
        "width": width,
        "height": height,
        "megapixels": round(mpx, 3),
        "tokens_flat": TOKENS_PER_IMAGE,
        "tokens_sized": int(round(mpx * TOKENS_PER_KPX)),
    }


def sh(cmd: list[str], timeout: float = 300.0):
    import subprocess

    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


class Browser:
    """A real, headed Chromium inside the harness display.

    Every element box, every style value and every assertion in this file comes
    from this browser. Nothing is hand-authored, which is what makes the numbers
    worth putting on a slide.
    """

    def __init__(self, display: str, url: str):
        from playwright.sync_api import sync_playwright

        os.environ["DISPLAY"] = display
        self.url = url
        self._pw = sync_playwright().start()
        self.browser = self._pw.chromium.launch(
            headless=False,
            args=[
                "--window-position=0,0",
                f"--window-size={WIDTH},{HEIGHT}",
                "--force-device-scale-factor=1",
                "--no-sandbox",
                "--no-first-run",
                "--disable-infobars",
                "--disable-session-crashed-bubble",
                "--disable-features=Translate,TranslateUI",
                "--disable-blink-features=AutomationControlled",
                "--disable-search-engine-choice-screen",
            ],
        )
        self.context = self.browser.new_context(
            viewport={"width": WIDTH, "height": VIEWPORT_H},
            device_scale_factor=1,
            reduced_motion="no-preference",
        )
        self.page = self.context.new_page()

    # -------------------------------------------------------------- geometry

    def viewport_origin(self) -> tuple[int, int]:
        geo = self.page.evaluate(
            "() => ({sx: window.screenX, sy: window.screenY,"
            " oh: window.outerHeight, ih: window.innerHeight})"
        )
        return int(geo["sx"]), int(geo["sy"]) + int(geo["oh"] - geo["ih"])

    def goto(self, path: str = "/") -> None:
        self.page.goto(self.url.rstrip("/") + path, wait_until="networkidle")
        self.page.wait_for_timeout(500)

    def boxes(self, selector: str) -> list[dict]:
        """Live DOM boxes, in CSS pixels relative to the viewport."""
        return self.page.evaluate(
            """(sel) => Array.from(document.querySelectorAll(sel)).map((el, i) => {
                 const r = el.getBoundingClientRect();
                 const style = getComputedStyle(el);
                 return {
                   box: [Math.round(r.x), Math.round(r.y),
                         Math.round(r.width), Math.round(r.height)],
                   label: (el.innerText || el.textContent || el.id || el.tagName)
                            .trim().split('\\n')[0].slice(0, 40),
                   role: el.tagName.toLowerCase(),
                   disabled: el.disabled === true || style.pointerEvents === 'none',
                 };
               })""",
            selector,
        )

    def dom_text_of(self, selector: str) -> str:
        return self.page.evaluate(
            "(sel) => (document.querySelector(sel) || {}).textContent || ''", selector
        )

    def close(self) -> None:
        for closer in (getattr(self.context, "close", None),
                       getattr(self.browser, "close", None)):
            try:
                if closer:
                    closer()
            except Exception:
                pass
        try:
            self._pw.stop()
        except Exception:
            pass


# ------------------------------------------------------------------ the runs


def experiment_capture(surface: Surface, browser: Browser, assets: Path) -> dict:
    """Region snips: capture the whole screen, or just the thing that matters."""
    browser.goto("/")
    browser.page.fill("#zip", "97")
    browser.page.click("#to-payment")
    browser.page.wait_for_timeout(900)

    full = surface.grab()
    (assets / "capture_full.png").write_bytes(full)

    banner = browser.page.locator("#zip-error, .field.invalid").first.bounding_box()
    ox, oy = browser.viewport_origin()
    box = (
        int(banner["x"]) + ox - 8,
        int(banner["y"]) + oy - 8,
        int(banner["width"]) + 16,
        int(banner["height"]) + 16,
    )
    snip = surface.snip(box)
    (assets / "capture_snip.png").write_bytes(snip)

    full_cost, snip_cost = cost_of(full), cost_of(snip)

    # The real question is not bytes but legibility: can an agent still read it?
    full_ocr = read_image(full)
    snip_ocr = read_image(snip)
    banner_text = browser.page.locator(".field.invalid").first.inner_text().strip()

    return {
        "region": list(box),
        "full": full_cost,
        "snip": snip_cost,
        "byte_ratio": round(snip_cost["bytes"] / max(full_cost["bytes"], 1), 4),
        "token_ratio_flat": round(
            snip_cost["tokens_flat"] / full_cost["tokens_flat"], 4
        ),
        "tokens_saved": full_cost["tokens_flat"] - snip_cost["tokens_flat"],
        "elapsed_full": round(full_ocr.elapsed, 3),
        "elapsed_snip": round(snip_ocr.elapsed, 3),
        "ground_truth": banner_text,
        "ocr_on_full": full_ocr.text.strip(),
        "ocr_on_snip": snip_ocr.text.strip(),
        "needle": banner_text.split("\n")[0].lower()[:18],
        "reads_banner_from_snip": banner_text.split("\n")[0].lower()[:18] in
                                  snip_ocr.text.lower(),
        "reads_banner_from_full": banner_text.split("\n")[0].lower()[:18] in
                                  full_ocr.text.lower(),
        "banner_confidence": round(snip_ocr.confidence, 1),
        "full_confidence": round(full_ocr.confidence, 1),
    }


def experiment_ocr(surface: Surface, browser: Browser, assets: Path) -> dict:
    """OCR: the log is painted to a canvas, so the DOM has nothing to give."""
    browser.goto("/gateway")
    browser.page.wait_for_timeout(600)
    browser.page.click("#probe")
    browser.page.wait_for_timeout(250)

    shot = surface.grab()
    (assets / "ocr_page.png").write_bytes(shot)
    probe_out = browser.dom_text_of("#probe-out")

    # Ground truth straight from the page's own JS, not from the pixels.
    truth = browser.page.evaluate("() => window.__gatewayLog || []")
    log_box = browser.boxes("#log")[0]["box"]
    ox, oy = browser.viewport_origin()
    region = (log_box[0] + ox, log_box[1] + oy, log_box[2], log_box[3])

    # --- the whole screen: what a naive agent would send -------------------
    t0 = time.monotonic()
    full = read_image(shot)
    full_elapsed = time.monotonic() - t0

    # --- just the log panel: what it costs to ask the right question -------
    t0 = time.monotonic()
    panel = read_region(shot, region)
    panel_elapsed = time.monotonic() - t0

    (assets / "ocr_panel.png").write_bytes(surface.snip(region))

    # --- score it, twice: verbatim, and through the glyph confusions -------
    # The tokens that matter are the ones an agent would act on: the decline
    # code, the transaction id, the rule that tripped, the postal code.
    expected_tokens = [
        "DO_NOT_HONOR", "7f3c9ab1", "velocity_exceeded", "FAILED",
        "http.status=402", "DECLINED", "4242", "97209",
    ]

    def score(result) -> dict:
        strict = [t for t in expected_tokens if result.contains(t)]
        folded = [t for t in expected_tokens if result.contains(t, fuzzy=True)]
        return {
            "expected": len(expected_tokens),
            "found_strict": strict,
            "found_folded": folded,
            "rescued_by_folding": [t for t in folded if t not in strict],
            "missed": [t for t in expected_tokens if t not in folded],
            "recall_strict": round(len(strict) / len(expected_tokens), 3),
            "recall_folded": round(len(folded) / len(expected_tokens), 3),
        }

    panel_score, full_score = score(panel), score(full)

    # Word-level agreement against what the page actually painted.
    truth_words = [w for line in truth for w in line["text"].split()]
    got_words = {fold(w.text) for w in panel.words}
    overlap = sum(1 for w in truth_words if fold(w) in got_words)

    # Show the raw transcription of the lines that matter, so the reader can see
    # exactly what the engine got wrong rather than take a score on faith.
    # Ground truth is rendered in the same column layout the canvas paints, so
    # the only differences on the page are OCR errors — which is the argument.
    def painted(marker: str) -> str:
        for entry in truth:
            if marker in entry["text"]:
                return f'{entry["ts"]} {entry["tag"]} {entry["text"]}'
        return ""

    decl_line = panel.line_of("DECLINED", fuzzy=True)
    terminal_line = panel.line_of("FAILED", fuzzy=True)

    box = panel.box_of("DO_NOT_HONOR", fuzzy=True)
    if box:
        (assets / "ocr_error_box.png").write_bytes(
            surface.snip((box[0] - 6, box[1] - 6, box[2] + 12, box[3] + 12))
        )

    return {
        "page": "gateway console — log painted to <canvas>",
        "dom_text_of_canvas": probe_out.strip(),
        "dom_has_text": bool(browser.dom_text_of("#log").strip()),
        "truth_lines": len(truth),
        "truth_words": len(truth_words),
        "panel_region": list(region),
        "elapsed_full_screen": round(full_elapsed, 3),
        "elapsed_panel": round(panel_elapsed, 3),
        "confidence_full": round(full.confidence, 1),
        "confidence_panel": round(panel.confidence, 1),
        "lines_full": full.line_count,
        "lines_panel": panel.line_count,
        "score_panel": panel_score,
        "score_full": full_score,
        "word_recall": round(overlap / max(len(truth_words), 1), 3),
        "words_matched": overlap,
        "panel_cost": cost_of(surface.snip(region)),
        "full_cost": cost_of(shot),
        "decline_line_raw": decl_line.text if decl_line else "",
        "decline_line_truth": painted("DECLINED"),
        "terminal_line_raw": terminal_line.text if terminal_line else "",
        "terminal_line_truth": painted("FAILED"),
        "error_box": list(box) if box else None,
        "sample_text": panel.text[:900],
    }


def experiment_diff(surface: Surface, browser: Browser, assets: Path) -> dict:
    """Visual regression: change one rule, see whether the pixels notice."""
    browser.goto("/")
    browser.page.wait_for_timeout(400)

    before = surface.grab()
    (assets / "diff_before.png").write_bytes(before)

    # A regression of exactly the kind CSS cascades produce: tightening one
    # selector on the step rail nudges a sibling's margin, which moves the
    # form two pixels and reflows the summary column.
    browser.page.add_style_tag(content="""
        .steps li + li { margin-left: 26px; }
        .field label { letter-spacing: 0.04em; }
        .summary { padding: 18px 22px; }
    """)
    browser.page.wait_for_timeout(500)
    after = surface.grab()
    (assets / "diff_after.png").write_bytes(after)

    result = compare(before, after, threshold=0.10, min_changed_pct=0.05)
    if result.diff_image:
        (assets / "diff_overlay.png").write_bytes(result.diff_image)

    # A control: the same image against itself must report nothing at all.
    control = compare(before, before)

    # A targeted snip diff: one field, before and after. Both sides are cut from
    # the full frames so they are guaranteed to be the same crop.
    from agent_visual_harness._ffmpeg import FFMPEG, run_bytes

    def snip_of(data: bytes, b: tuple[int, int, int, int]) -> bytes:
        return run_bytes(
            [FFMPEG, "-hide_banner", "-loglevel", "error", "-f", "image2pipe", "-i", "-",
             "-vf", f"crop={b[2]}:{b[3]}:{b[0]}:{b[1]}", "-c:v", "png",
             "-f", "image2pipe", "-"],
            timeout=60, input_bytes=data,
        )

    field_box = browser.boxes(".field")[0]["box"]
    ox, oy = browser.viewport_origin()
    box = (field_box[0] + ox, field_box[1] + oy, field_box[2], field_box[3])
    field_diff = compare(snip_of(before, box), snip_of(after, box))
    if field_diff.diff_image:
        (assets / "diff_field.png").write_bytes(field_diff.diff_image)

    # How many of the reported regions actually changed? Precision matters as
    # much as recall: a diff that flags everything gets switched off.
    moved = [
        r for r in result.regions
        if r.box[2] * r.box[3] > 400
    ]

    return {
        "full": result.to_dict(),
        "control": control.to_dict(),
        "field": field_diff.to_dict(),
        "field_region": list(box),
        "regions": [r.to_dict() for r in result.regions[:12]],
        "region_count": result.region_count,
        "large_regions": len(moved),
        "verdict": result.verdict(),
        "elapsed": round(result.elapsed, 3),
        "css_injected": [
            ".steps li + li { margin-left: 26px; }",
            ".field label { letter-spacing: 0.04em; }",
            ".summary { padding: 18px 22px; }",
        ],
    }


def experiment_grounding(surface: Surface, browser: Browser, assets: Path) -> dict:
    """Set-of-Marks: turn "the pay button" into a coordinate, not a guess."""
    browser.goto("/")
    browser.page.wait_for_timeout(400)

    # First look at the page as it loads. The action row is below the fold, so
    # the DOM reports zero-height rects for those buttons — measurable evidence
    # of the case the grouper has to refuse rather than draw.
    at_top = browser.boxes("a[href], button, input, .field")
    clipped_at_top = [
        {"label": el["label"], "box": list(el["box"])}
        for el in at_top if el["box"][2] == 0 or el["box"][3] == 0
    ]

    # Then scroll the action row into view, exactly as a real agent does before
    # clicking, so the buttons it wants to ground are actually on screen.
    browser.page.locator("#to-payment").scroll_into_view_if_needed()
    browser.page.wait_for_timeout(400)

    ox, oy = browser.viewport_origin()
    elements = browser.boxes("a[href], button, input, .field")

    # Boxes stay in viewport CSS pixels and Scale does the single transform.
    # Adding the origin here as well would apply it twice and push anything in
    # the lower half of the page past the bottom edge.
    targets = []
    for i, el in enumerate(elements):
        targets.append({
            "index": i + 1,
            "box": list(el["box"]),
            "label": el["label"],
            "role": el["role"],
            "disabled": el["disabled"],
        })

    scale = Scale(origin=(ox, oy))
    shot = surface.grab()
    marks = annotate_boxes(
        shot, [t["box"] for t in targets],
        labels=[t["label"] for t in targets],
        roles=[t["role"] for t in targets],
        scale=scale,
    )
    (assets / "som_screen.png").write_bytes(marks.image)

    # The model's job is to answer with an index. Resolve it to a click.
    # Chosen from the marks that survived, not from the raw DOM list: the
    # clipped elements were dropped and have no index to answer with.
    kept = marks.targets

    def pick(*needles: str) -> int | None:
        for needle in needles:
            for target in kept:
                if needle in target.label.lower():
                    return target.index
        return None

    answer = pick("continue to payment", "pay", "place order") or kept[0].index
    raw_target = next(t for t in kept if t.index == answer)

    resolved = resolve(marks, answer, scale=scale)
    resolved_box = resolve_box(marks, answer)

    # resolve() hands back pointer space; the mark itself is drawn in image
    # pixels. Publishing both stops the two being compared with `==`.
    device_point = scale.to_device(*resolved) if not scale.is_identity else resolved
    offset_y = device_point[1] - resolved[1]

    return {
        "marks": len(marks),
        "targets": len(targets),
        "skipped": marks.skipped_count,
        "skipped_detail": marks.skipped,
        "clipped_at_top": len(clipped_at_top),
        "clipped_at_top_detail": clipped_at_top,
        "screenshot_cost": cost_of(shot),
        "annotated_cost": cost_of(marks.image),
        "legend": marks.compact(),
        "prompt": marks.prompt("Which element continues to payment?"),
        "answer": answer,
        "target_label": raw_target.label,
        "resolved_point": list(resolved),
        "resolved_point_image": list(device_point),
        "resolved_box": list(resolved_box),
        # raw_target.box is already in image pixels; do not transform it again.
        "device_box": list(raw_target.box),
        "space_offset": [device_point[0] - resolved[0], offset_y],
        "dom_point": list(raw_target.centre),
        "dom_box": list(raw_target.box),
        "viewport_origin": [ox, oy],
        "scale": scale.describe(),
        "targets_json": targets,
    }


def experiment_hidpi(surface: Surface, browser: Browser, assets: Path) -> dict:
    """HiDPI drift: the same button, two displays, one silent bug.

    Not arithmetic on paper. A real 2x display is opened, a real click is
    attempted at the coordinate an agent would have read off the screenshot,
    and the page is asked afterwards whether the click landed. Then the
    coordinate is normalised and the same click is retried.
    """
    from playwright.sync_api import sync_playwright  # noqa: F401 - type only

    # --- reference: what the DOM says at 1x -------------------------------
    browser.goto("/gateway")
    browser.page.wait_for_timeout(500)
    ox, oy = browser.viewport_origin()
    button = browser.boxes("#probe")[0]["box"]
    css_box = (button[0] + ox, button[1] + oy, button[2], button[3])
    css_point = (css_box[0] + css_box[2] // 2, css_box[1] + css_box[3] // 2)

    # --- the same page on a 2x display -------------------------------------
    context = browser.context.browser.new_context(
        viewport={"width": WIDTH, "height": VIEWPORT_H},
        device_scale_factor=2,
        reduced_motion="no-preference",
    )
    page2 = context.new_page()
    page2.goto(browser.url.rstrip("/") + "/gateway", wait_until="networkidle")
    page2.wait_for_timeout(700)

    dpr = page2.evaluate("() => window.devicePixelRatio")
    page2.screenshot(path=str(assets / "hidpi_2x_page.png"))

    # Where the button really is in the 2x *image*, found by reading the pixels
    # rather than trusting the DOM. The *word* box, not the line box: a line box
    # spans the whole toolbar, and its centre is nowhere near the button.
    shot2 = (assets / "hidpi_2x_page.png").read_bytes()
    read = read_image(shot2)
    word = find(shot2, "innerHTML", fuzzy=True)

    if word is None:
        raise RuntimeError(
            "OCR could not find the control button on the 2x page; the HiDPI "
            "experiment has no ground truth to measure drift against"
        )
    device_box = list(word.box)

    # The DOM's own opinion, in CSS pixels, scaled to image pixels.
    dom2 = page2.locator("#probe").bounding_box()
    dom_box_2x = [
        int(dom2["x"] * dpr), int(dom2["y"] * dpr),
        int(dom2["width"] * dpr), int(dom2["height"] * dpr),
    ]

    # OCR sees the label, not the control's border — which is exactly why the
    # element boxes in a Set-of-Marks come from the DOM and not from reading
    # pixels. Measured here rather than asserted.

    # What an agent does wrong: use the coordinate it saw, unchanged.
    naive_point = (device_box[0] + device_box[2] // 2, device_box[1] + device_box[3] // 2)
    page2.mouse.click(*naive_point)
    page2.wait_for_timeout(400)
    naive_hit = bool(page2.locator("#probe-out").inner_text().strip())
    (assets / "hidpi_naive_click.png").write_bytes(page2.screenshot())

    # What the harness does right: normalise image pixels back to pointer space.
    scale = Scale(ratio=dpr, origin=(0, 0))
    fixed_point = scale.to_logical(*naive_point)
    page2.reload(wait_until="networkidle")
    page2.wait_for_timeout(700)
    page2.mouse.click(*fixed_point)
    page2.wait_for_timeout(400)
    fixed_hit = bool(page2.locator("#probe-out").inner_text().strip())
    (assets / "hidpi_fixed_click.png").write_bytes(page2.screenshot())

    context.close()

    drift = [abs(naive_point[0] - fixed_point[0]), abs(naive_point[1] - fixed_point[1])]

    return {
        "device_pixel_ratio": dpr,
        "css_box_1x": list(css_box),
        "css_point_1x": list(css_point),
        "dom_box_2x": dom_box_2x,
        "ocr_label_box_2x": device_box,
        "ocr_dom_agreement_px": [
            abs(device_box[0] - dom_box_2x[0]),
            abs(device_box[1] - dom_box_2x[1]),
        ],
        "ocr_confidence_2x": round(read.confidence, 1),
        "ocr_elapsed_2x": round(read.elapsed, 3),
        "ocr_lines_2x": read.line_count,
        "naive_point_2x": list(naive_point),
        "fixed_point_2x": list(fixed_point),
        "drift_px": drift,
        "naive_click_landed": naive_hit,
        "fixed_click_landed": fixed_hit,
        "image_size_2x": list(image_size(shot2)),
        "retina_note": (
            f"{image_size(shot2)[0]}x{image_size(shot2)[1]} image vs "
            f"{WIDTH}x{VIEWPORT_H} pointer space"
        ),
        "scale": scale.describe(),
    }


# ------------------------------------------------------------------------ main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(HERE), help="demo directory to populate")
    parser.add_argument("--display", default=":151", help="display to create")
    parser.add_argument("--port", type=int, default=8098, help="port for the storefront")
    args = parser.parse_args(argv)

    out = Path(args.out)
    assets = out / "assets" / "inspection"
    assets.mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(parents=True, exist_ok=True)

    httpd, url = start_background(args.port)
    print(f"storefront  {url}")

    display = VirtualDisplay(display=args.display, resolution=(WIDTH, HEIGHT), timeout=30)
    display.start()
    os.environ["DISPLAY"] = display.display
    surface = Surface.from_display(display.display, (WIDTH, HEIGHT))

    results: dict = {"meta": {
        "display": f"{WIDTH}x{HEIGHT} Xvfb, headed Chromium",
        "tokens_per_image_flat": TOKENS_PER_IMAGE,
        "tokens_per_megapixel": TOKENS_PER_KPX,
        "page": "Northwind Outfitters — demo storefront",
    }}

    browser = Browser(display.display, url)
    try:
        print("1/5  region snips ...")
        results["capture"] = experiment_capture(surface, browser, assets)
        print("2/5  OCR from a canvas ...")
        results["ocr"] = experiment_ocr(surface, browser, assets)
        print("3/5  visual regression ...")
        results["diff"] = experiment_diff(surface, browser, assets)
        print("4/5  set-of-marks grounding ...")
        results["grounding"] = experiment_grounding(surface, browser, assets)
        print("5/5  HiDPI coordinate drift ...")
        results["hidpi"] = experiment_hidpi(surface, browser, assets)
    finally:
        browser.close()
        display.stop()
        httpd.shutdown()

    results["assets"] = sorted(p.name for p in assets.iterdir())
    target = out / "data" / "inspection.json"
    target.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    c = results["capture"]
    o = results["ocr"]
    d = results["diff"]
    g = results["grounding"]
    print()
    print(f"{'snip bytes / full':<26}{human(c['snip']['bytes'])} / "
          f"{human(c['full']['bytes'])}  ({c['byte_ratio'] * 100:.1f}%)")
    print(f"{'snip tokens (flat)':<26}{c['snip']['tokens_flat']:,} / "
          f"{c['full']['tokens_flat']:,}  (flat pricing ignores size entirely)")
    print(f"{'snip tokens (sized)':<26}{c['snip']['tokens_sized']:,} / "
          f"{c['full']['tokens_sized']:,}  (by megapixels)")
    print(f"{'OCR recall strict':<26}"
          f"{o['score_panel']['recall_strict'] * 100:.0f}%  "
          f"conf {o['confidence_panel']:.0f}%  {o['elapsed_panel']:.2f}s")
    print(f"{'OCR recall folded':<26}"
          f"{o['score_panel']['recall_folded'] * 100:.0f}%  "
          f"(+{len(o['score_panel']['rescued_by_folding'])} rescued by glyph folding)")
    print(f"{'DOM text available?':<26}{o['dom_has_text']}")
    print(f"{'diff changed':<26}{d['full']['changed_pct']:.3f}%  "
          f"{d['region_count']} regions")
    print(f"{'marks drawn':<26}{g['marks']}  answer {g['answer']} -> {g['resolved_point']}")
    h = results["hidpi"]
    print(f"{'2x drift':<26}{h['drift_px']} px — naive click landed: "
          f"{h['naive_click_landed']}, normalised: {h['fixed_click_landed']}")
    print(f"results        : {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())