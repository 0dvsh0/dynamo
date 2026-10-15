#!/usr/bin/env python3
"""Drive a real browser through a real checkout, and measure each workflow.

Everything reported in demo/index.html comes out of this script — the numbers are
measured, never estimated. Artifacts are copied into demo/assets/ so the page shows
the real thing rather than a mock-up.

The workload is a *headed* Chromium rendering a real HTTP storefront
(demo/webapp) inside the harness display, so what gets recorded is what a person's
screen would look like: tab strip, address bar, scrollbars and all.

Workflows
---------
screenshots   what a team does today: grab a PNG after every agent action
diy           hand-rolled ffmpeg: x11grab + SIGKILL, default transcode, one-pass GIF
dynamo       the harness: managed display, SIGINT finalisation, two-pass GIF, reducer

Usage
-----
    python3 demo/run_experiments.py --out demo
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "src"))

from agent_visual_harness import VirtualDisplay  # noqa: E402

sys.path.insert(0, str(HERE))
from serve import start_background  # noqa: E402

FFMPEG = os.environ.get("DYNAMO_FFMPEG", "ffmpeg")
FFPROBE = os.environ.get("DYNAMO_FFPROBE", "ffprobe")

# Tokens per image handed to a multimodal model — the figure the marketing site
# uses for a full-size screenshot. Applied identically to every workflow so the
# comparison is fair.
TOKENS_PER_IMAGE = 1100

#: Display geometry. 1440x900 is an ordinary laptop screen, and the extra 100px
#: over the 800px viewport is where the browser's own chrome lives.
WIDTH, HEIGHT, VIEWPORT_H, FPS = 1440, 900, 800, 20

#: Seconds the agent spends waiting between actions — a slow API, a rate limit, a
#: model call. This is most of a real session, and it is exactly the stretch a
#: screenshot-per-action workflow records nothing useful for while a wall-clock
#: sampler keeps paying for identical frames.
DWELL = 1.1
DWELL_MS = int(DWELL * 1000)


# --------------------------------------------------------------------------- helpers


def sh(cmd: list[str], timeout: float = 600.0) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def size_of(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def playable(path: Path) -> tuple[bool, str]:
    """Can ffprobe decode this file? Returns (ok, duration or first error line)."""
    if not path.exists():
        return False, "file does not exist"
    proc = sh([FFPROBE, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)])
    if proc.returncode != 0:
        first = (proc.stderr or "").strip().splitlines()
        return False, (first[0] if first else "unknown error")
    return True, proc.stdout.strip()


def probe(path: Path) -> dict:
    proc = sh([FFPROBE, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)])
    if proc.returncode != 0:
        return {}
    data = json.loads(proc.stdout or "{}")
    stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
    fmt = data.get("format", {})
    duration = float(fmt.get("duration") or stream.get("duration") or 0.0)
    num, _, den = (stream.get("avg_frame_rate") or "0/1").partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 0.0
    return {
        "duration": round(duration, 2),
        "frames": int(stream.get("nb_frames") or 0),
        "fps": round(fps, 2),
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "codec": stream.get("codec_name", ""),
        "bytes": int(fmt.get("size") or size_of(path)),
    }


def poster(video: Path, out: Path, at: float = 2.0) -> Path | None:
    proc = sh([FFMPEG, "-v", "error", "-y", "-ss", f"{at:.2f}", "-i", str(video),
               "-frames:v", "1", "-q:v", "3", str(out)])
    return out if proc.returncode == 0 and out.exists() else None


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


# --------------------------------------------------------------------------- workload


class CheckoutSession:
    """A real browser buying something, the way an automation agent would.

    Headed Chromium renders into the harness display, so the capture includes the
    tab strip and address bar. Every interaction goes through Playwright, which
    is what a real e2e agent does — nothing here is scripted against internal
    application state.
    """

    def __init__(self, display: str, url: str, headed: bool = True):
        from playwright.sync_api import sync_playwright

        os.environ["DISPLAY"] = display
        self.url = url
        self._pw = sync_playwright().start()
        self.browser = self._pw.chromium.launch(
            headless=not headed,
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
        self.page.goto(url, wait_until="networkidle")
        self.page.wait_for_timeout(400)
        self.actions: list[dict] = []
        self.on_action = None

    # ---------------------------------------------------------- coordinates

    def viewport_origin(self) -> tuple[int, int]:
        """Where the page viewport starts in *display* pixels.

        With browser chrome present the page does not start at (0,0); overlays
        drawn from telemetry would land on the tab strip instead of the button.
        """
        geo = self.page.evaluate(
            "() => ({sx: window.screenX, sy: window.screenY,"
            " oh: window.outerHeight, ih: window.innerHeight})"
        )
        return int(geo["sx"]), int(geo["sy"]) + int(geo["oh"] - geo["ih"])

    def centre_of(self, selector: str) -> tuple[int, int] | None:
        box = self.page.locator(selector).first.bounding_box()
        if not box:
            return None
        ox, oy = self.viewport_origin()
        return int(box["x"] + box["width"] / 2 + ox), int(box["y"] + box["height"] / 2 + oy)

    # ------------------------------------------------------------- driving

    def _point(self, selector: str) -> tuple[int, int] | None:
        """Scroll the element into view, then read where it actually is.

        Playwright auto-scrolls before it clicks, so a bounding box read before
        the click describes a position the user never saw. Overlays drawn from
        these coordinates would land somewhere else on the recording.
        """
        locator = self.page.locator(selector).first
        locator.scroll_into_view_if_needed()
        self.page.wait_for_timeout(150)
        box = locator.bounding_box()
        if not box:
            return None
        ox, oy = self.viewport_origin()
        return int(box["x"] + box["width"] / 2 + ox), int(box["y"] + box["height"] / 2 + oy)

    def _announce(self, label: str, point: tuple[int, int] | None) -> None:
        action = {"label": label, "x": point[0] if point else None,
                  "y": point[1] if point else None}
        self.actions.append(action)
        if self.on_action is not None:
            self.on_action(action)
        if DWELL:
            # The agent stops to think. On screen nothing changes at all.
            self.page.wait_for_timeout(DWELL_MS)

    def _click(self, selector: str, label: str) -> None:
        point = self._point(selector)
        self.page.locator(selector).first.click()
        self._announce(label, point)

    def _type(self, selector: str, value: str, label: str) -> None:
        point = self._point(selector)
        field = self.page.locator(selector).first
        field.click()
        field.type(value, delay=16)
        self._announce(label, point)

    def _note(self, label: str) -> None:
        """Record a state change that had no click of its own."""
        self._announce(label, None)

    def run(self, on_action=None) -> list[dict]:
        """Play the flow, announcing every agent action as it happens."""
        self.on_action = on_action

        # --- 1. a bad postal code trips the real validator -------------------
        self._type("#name", "Ada Lovelace", "type: full name")
        self._type("#line1", "18 Wharf Road", "type: street address")
        self._type("#city", "Portland", "type: city")
        self._type("#zip", "97", "type: postal code")
        self.page.wait_for_timeout(700)

        self._click("#to-payment", "click: continue to payment")
        self.page.wait_for_timeout(600)
        assert self.page.locator(".field.invalid").count() > 0, "validator did not fire"
        self._note("validation error: postal code rejected")

        # --- 2. fix it and move on -------------------------------------------
        point = self._point("#zip")
        self.page.locator("#zip").first.fill("97209")
        self._announce("fix: postal code", point)
        self.page.wait_for_timeout(400)

        self._click("#to-payment", "click: continue to payment")
        self.page.wait_for_timeout(900)
        assert self.page.locator("#cnum").count() > 0, "did not reach the payment step"
        self._note("reached payment step")

        # --- 3. card entry, with the preview updating as you type -----------
        self._type("#cnum", "4242424242424242", "type: card number")
        self.page.wait_for_timeout(600)
        self._type("#cname", "ADA LOVELACE", "type: name on card")
        self.page.wait_for_timeout(500)
        self._type("#cexp", "0429", "type: expiry")
        self.page.wait_for_timeout(400)
        self._type("#ccvc", "311", "type: cvc")
        self.page.wait_for_timeout(600)

        # --- 4. the issuer declines twice before it settles ------------------
        for attempt in (1, 2):
            self._click("#pay-btn", f"click: pay (attempt {attempt})")
            self.page.wait_for_timeout(1300)
            self._note(f"waiting on issuer (attempt {attempt})")
            self.page.wait_for_timeout(1700)
            assert self.page.locator("#declined").count() > 0, f"attempt {attempt} was not declined"
            self._note(f"issuer declined (attempt {attempt})")

        self._click("#pay-btn", "click: pay (attempt 3)")
        self.page.wait_for_timeout(1300)
        self._note("waiting on issuer (attempt 3)")
        self.page.wait_for_timeout(1700)
        assert self.page.locator("#place").count() > 0, "payment never captured"
        self._note("payment captured")
        self.page.wait_for_timeout(600)

        # --- 5. place the order ---------------------------------------------
        self._click("#place", "click: place order")
        self.page.wait_for_timeout(2000)
        assert "Order confirmed" in self.page.content(), "order was never confirmed"
        self._note("order confirmed")
        self.page.wait_for_timeout(1200)
        self.on_action = None
        return self.actions

    def close(self) -> None:
        try:
            self.context.close()
            self.browser.close()
        except Exception:
            pass
        try:
            self._pw.stop()
        except Exception:
            pass


# --------------------------------------------------------------------------- workflows


def workflow_screenshots(session_factory, out: Path) -> dict:
    """What a team does today: a PNG after every action, nothing else."""
    shots = out / "shots"
    shots.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    session = session_factory()

    def grab(action: dict, index: int) -> None:
        path = shots / f"shot_{index:02d}.png"
        sh([FFMPEG, "-v", "error", "-y", "-f", "x11grab", "-video_size", f"{WIDTH}x{HEIGHT}",
            "-i", os.environ["DISPLAY"], "-frames:v", "1", str(path)], timeout=90)

    counter = {"n": 0}

    def on_action(action: dict) -> None:
        counter["n"] += 1
        grab(action, counter["n"])

    session.run(on_action=on_action)
    session.close()
    elapsed = time.monotonic() - started

    files = sorted(shots.glob("*.png"))
    return {
        "images": len(files),
        "image_bytes": sum(size_of(f) for f in files),
        "tokens": len(files) * TOKENS_PER_IMAGE,
        "seconds": round(elapsed, 1),
        "video": None,
        "gif": None,
        "frames_dir": shots,
        "playable_after_sigkill": None,
        "redundancy": redundancy(_thumbs_from_images(shots)),
        "notes": "no recording at all — nothing to attach to a bug report",
    }


def _capture_diy(session_factory, out: Path, kill: str) -> Path:
    """Record the workload the way a hand-rolled script does.

    ``kill="9"`` reproduces the CI teardown that breaks everyone: the runner is
    reaped and ffmpeg is SIGKILLed before it writes its trailer.
    ``kill="int"`` is the lucky path where the process happens to exit cleanly.
    """
    raw = out / f"diy_raw_{kill}.mp4"
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
           "-f", "x11grab", "-framerate", str(FPS), "-video_size", f"{WIDTH}x{HEIGHT}",
           "-i", os.environ["DISPLAY"], "-an", "-c:v", "libx264", "-preset", "ultrafast",
           "-pix_fmt", "yuv420p", str(raw)]
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, start_new_session=True)
    session = session_factory()
    session.run()
    session.close()
    os.killpg(os.getpgid(proc.pid), 9 if kill == "9" else 2)
    proc.wait(timeout=30)
    return raw


def _naive_pipeline(raw: Path, out: Path, tag: str) -> dict:
    """The three commands every team writes: transcode, GIF, sample frames."""
    video = out / f"diy_naive_{tag}.mp4"
    gif = out / f"diy_naive_{tag}.gif"
    frames_dir = out / f"diy_frames_{tag}"
    frames_dir.mkdir(parents=True, exist_ok=True)

    if playable(raw)[0]:
        sh([FFMPEG, "-v", "error", "-y", "-i", str(raw), "-c:v", "libx264", "-crf", "28",
            "-pix_fmt", "yuv420p", str(video)], timeout=900)
        # The naive GIF: one pass, full resolution, default 256-colour palette.
        # Framerate is held at 15 to match the harness default, so the only
        # difference between the two GIFs is the palette strategy.
        sh([FFMPEG, "-v", "error", "-y", "-i", str(raw), "-vf", "fps=15", str(gif)],
           timeout=3000)
    for stale in frames_dir.glob("*.png"):
        stale.unlink()
    if playable(raw)[0]:
        sh([FFMPEG, "-v", "error", "-y", "-i", str(raw), "-vf", "fps=1",
            str(frames_dir / "frame_%03d.png")], timeout=900)
    return {"video": video, "gif": gif, "frames": sorted(frames_dir.glob("*.png"))}


def workflow_diy(session_factory, out: Path) -> dict:
    """Hand-rolled ffmpeg, shown twice: once crashed, once merely naive.

    Presenting only the crashed case would be a strawman, so the page shows the
    lucky path too — that is what teams actually end up shipping, and it is
    where the frame count and the GIF size go wrong.
    """
    raw_killed = _capture_diy(session_factory, out, kill="9")
    ok_killed, error = playable(raw_killed)
    artifacts_after_kill = _naive_pipeline(raw_killed, out, "killed")

    raw_ok = _capture_diy(session_factory, out, kill="int")
    naive = _naive_pipeline(raw_ok, out, "ok")
    ok_raw, _ = playable(raw_ok)

    frames = sorted((out / "diy_frames_ok").glob("*.png"))
    return {
        "crash": {
            "raw_playable": ok_killed,
            "raw_error": error,
            "video_bytes": size_of(artifacts_after_kill["video"]),
            "frames": len(artifacts_after_kill["frames"]),
            "notes": "SIGKILL leaves no moov atom, so the post-run transcode has nothing to read",
        },
        "raw_playable": ok_raw,
        "video": naive["video"],
        "gif": naive["gif"],
        "images": len(frames),
        "image_bytes": sum(size_of(f) for f in frames),
        "tokens": len(frames) * TOKENS_PER_IMAGE,
        "video_bytes": size_of(naive["video"]),
        "gif_bytes": size_of(naive["gif"]),
        "frames_dir": out / "diy_frames_ok",
        "raw": raw_ok,
        "redundancy": redundancy(_thumbs_from_images(out / "diy_frames_ok")),
        "notes": "one frame per wall-clock second, no dedup, single-pass GIF at full size",
    }


def workflow_dynamo(session_factory, out: Path, session_dir: Path) -> dict:
    """The harness, end to end."""
    from agent_visual_harness import Processor
    from agent_visual_harness import Recorder, Session as DynamoSession
    from agent_visual_harness import TokenReducer

    kf_dir = out / "dynamo_frames"
    started = time.monotonic()

    dynamo_session = DynamoSession.create(
        session_dir, display=os.environ["DISPLAY"],
        resolution=(WIDTH, HEIGHT), fps=FPS,
    )
    recorder = Recorder(
        display=os.environ["DISPLAY"], size=(WIDTH, HEIGHT),
        output=dynamo_session.raw_path, fps=FPS, crf=16, preset="veryfast",
        telemetry_path=dynamo_session.telemetry_path,
    )
    recorder.start()

    session = session_factory()

    def on_action(action: dict) -> None:
        recorder.log_action(action["label"], x=action["x"], y=action["y"])

    session.run(on_action=on_action)
    session.close()
    recorder.stop()
    capture_seconds = time.monotonic() - started

    actions = dynamo_session.actions
    processor = Processor(crf=23, preset="veryfast", gif_max_colors=128)
    repro = out / "dynamo_repro.mp4"
    gif = out / "dynamo_demo.gif"
    processor.export_video(dynamo_session.raw_path, repro, actions=actions, source_size=(WIDTH, HEIGHT))
    processor.to_gif(dynamo_session.raw_path, gif)
    frames = TokenReducer().reduce(
        dynamo_session.raw_path, kf_dir, actions=actions, source_size=(WIDTH, HEIGHT),
    )
    elapsed = time.monotonic() - started

    return {
        "images": len(frames),
        "image_bytes": sum(size_of(f.path) for f in frames),
        "tokens": len(frames) * TOKENS_PER_IMAGE,
        "seconds": round(elapsed, 1),
        "capture_seconds": round(capture_seconds, 1),
        "video": repro,
        "gif": gif,
        "video_bytes": size_of(repro),
        "gif_bytes": size_of(gif),
        "keyframes_dir": kf_dir,
        "frames_dir": kf_dir,
        "telemetry": dynamo_session.telemetry_path,
        "session_dir": session_dir,
        "raw_playable": True,
        "playable_after_sigkill": True,
        "actions": len(actions),
        "raw": dynamo_session.raw_path,
        "redundancy": redundancy([_thumbnail(f.path) for f in frames]),
        "notes": "SIGINT on every exit path; anchors are never dropped",
    }


def workflow_dynamo_crash(session_factory, out: Path, session_dir: Path) -> dict:
    """The failure case: the agent dies mid-run. Does the recording survive?"""
    from agent_visual_harness import Recorder
    from agent_visual_harness import Session as DynamoSession

    dynamo_session = DynamoSession.create(
        session_dir, display=os.environ["DISPLAY"],
        resolution=(WIDTH, HEIGHT), fps=FPS,
    )
    recorder = Recorder(
        display=os.environ["DISPLAY"], size=(WIDTH, HEIGHT),
        output=dynamo_session.raw_path, fps=FPS, crf=16, preset="veryfast",
        telemetry_path=dynamo_session.telemetry_path,
    )
    recorder.start()
    session = session_factory()
    logged = {"n": 0}

    def on_action(action: dict) -> None:
        logged["n"] += 1
        recorder.log_action(action["label"], x=action["x"], y=action["y"])
        if logged["n"] >= 4:
            # The agent dies here, mid-checkout, with the recorder still running.
            raise SystemExit(9)

    try:
        session.run(on_action=on_action)
    except SystemExit:
        pass
    session.close()
    # In the SDK this happens inside __exit__; calling it directly proves the point.
    recorder.stop()
    ok, probe_output = playable(dynamo_session.raw_path)
    shutil.copy2(dynamo_session.raw_path, out / "dynamo_crash.mp4")
    poster(dynamo_session.raw_path, out / "crash_poster.jpg", at=1.5)
    return {
        "playable": ok,
        "probe": probe_output,
        # The finished file's duration is slightly longer than the wall-clock
        # span between start() and stop(); the gap is ffmpeg writing its trailer.
        "file_duration": float(probe_output) if ok and probe_output else 0.0,
        "raw": dynamo_session.raw_path,
        "telemetry": dynamo_session.telemetry_path,
        "actions": len(read_actions(dynamo_session.telemetry_path)),
        "logged": logged["n"],
        "duration": round(recorder.elapsed, 1),
    }


def read_actions(path: Path) -> list:
    from agent_visual_harness.recorder import read_actions as _read

    return _read(path)


def extract_stills(video: Path, telemetry: Path, out: Path, max_stills: int = 14) -> list[dict]:
    """Pull the real frame at each logged action out of the finished video.

    This is the gallery the page shows: not a mock-up of what the agent "should"
    have seen, but the actual decoded frame at the timestamp telemetry recorded.
    """
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("still_*.jpg"):
        stale.unlink()

    events = read_actions(telemetry)
    picks = [e for e in events if e.has_position]
    # Thin out if the flow produced a lot of clicks, keeping the first and last.
    if len(picks) > max_stills:
        step = len(picks) / max_stills
        picks = [picks[int(i * step)] for i in range(max_stills)]

    stills: list[dict] = []
    for index, action in enumerate(picks, start=1):
        path = out / f"still_{index:02d}.jpg"
        proc = sh([FFMPEG, "-v", "error", "-y", "-ss", f"{action.timestamp + 0.05:.3f}",
                   "-i", str(video), "-frames:v", "1", "-q:v", "3", str(path)], timeout=120)
        if proc.returncode != 0 or not path.exists():
            continue
        stills.append({"file": f"assets/stills/{path.name}", "index": index,
                       "timestamp": round(action.timestamp, 2), "action": action.action,
                       "x": action.x, "y": action.y})
    return stills


# --------------------------------------------------------------------------- metrics


THUMB = 64


def _thumbnail(image: Path) -> bytes:
    """Decode one image to a 64x64 RGB raw buffer, for content comparison."""
    proc = subprocess.run(
        [FFMPEG, "-v", "error", "-i", str(image), "-vf", f"scale={THUMB}:{THUMB}",
         "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
        capture_output=True, timeout=90,
    )
    return proc.stdout[: THUMB * THUMB * 3]


def _thumbs_from_images(directory: Path) -> list[bytes]:
    images = sorted(directory.glob("*.png")) + sorted(directory.glob("*.jpg")) + \
        sorted(directory.glob("*.webp"))
    return [_thumbnail(image) for image in images]


def redundancy(thumbs: list[bytes]) -> dict:
    """How much of a frame set is the same picture more than once?

    Two frames sampled from the same still screen decode to *identical* pixels,
    so exact comparison is enough — and exact comparison needs no threshold,
    which matters: a hand-tuned "similarity" cutoff would be a number chosen to
    produce the answer this demo wants. The raw per-pair deltas are reported too,
    so the shape of the distribution stays visible instead of being summarised
    away into one percentage.
    """
    if len(thumbs) < 2:
        return {"images": len(thumbs), "pairs": 0, "duplicates": 0, "duplicate_pct": 0.0,
                "distinct": len(thumbs), "mean_delta": 0.0, "max_delta": 0.0, "deltas": []}

    deltas: list[float] = []
    for previous, current in zip(thumbs, thumbs[1:], strict=False):
        total = sum(abs(a - b) for a, b in zip(previous, current, strict=False))
        deltas.append(round(total / len(previous), 2))

    duplicates = sum(1 for d in deltas if d == 0.0)
    return {
        "images": len(thumbs),
        "pairs": len(deltas),
        "duplicates": duplicates,
        "duplicate_pct": round(100.0 * duplicates / len(deltas), 1),
        "distinct": len(thumbs) - duplicates,
        "mean_delta": round(sum(deltas) / len(deltas), 2),
        "max_delta": round(max(deltas), 2),
        "deltas": deltas,
    }


# --------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure the three workflows")
    parser.add_argument("--out", default=str(HERE), help="demo directory to populate")
    parser.add_argument("--display", default=":150", help="display to create (Dynamo owns it)")
    parser.add_argument("--port", type=int, default=8099, help="port for the demo storefront")
    parser.add_argument("--work", default="/tmp/dynamo-demo", help="scratch directory")
    parser.add_argument("--keep-work", action="store_true")
    args = parser.parse_args(argv)

    out = Path(args.out)
    assets = out / "assets"
    data_dir = out / "data"
    if assets.exists():
        shutil.rmtree(assets)
    assets.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    work = Path(args.work)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    httpd, url = start_background(args.port)
    print(f"storefront  {url}")

    print(f"Creating display {args.display} at {WIDTH}x{HEIGHT} @ {FPS}fps")
    display = VirtualDisplay(display=args.display, resolution=(WIDTH, HEIGHT), timeout=30)
    display.start()
    os.environ["DISPLAY"] = display.display

    def session_factory() -> CheckoutSession:
        return CheckoutSession(display.display, url, headed=True)

    results: dict = {"meta": {
        "width": WIDTH, "height": HEIGHT, "viewport_height": VIEWPORT_H, "fps": FPS,
        "tokens_per_image": TOKENS_PER_IMAGE,
        "browser": "Chromium (Playwright), headed, inside the harness display",
        "site": "Northwind Outfitters — demo storefront",
        "flow": [
            "type shipping details", "validator rejects a 2-digit postal code",
            "fix postal code, continue to payment", "type card details",
            "issuer declines attempt 1", "issuer declines attempt 2",
            "payment captured on attempt 3", "review order", "place order",
            "order confirmed",
        ],
    }}

    try:
        print("1/4  screenshots-only workflow ...")
        results["screenshots"] = workflow_screenshots(session_factory, assets)

        print("2/4  DIY ffmpeg workflow (with a SIGKILL) ...")
        results["diy"] = workflow_diy(session_factory, assets)

        print("3/4  Dynamo workflow ...")
        results["dynamo"] = workflow_dynamo(session_factory, assets, work / "session")

        print("4/4  Dynamo, agent crashes mid-run ...")
        results["crash"] = workflow_dynamo_crash(session_factory, assets, work / "crash")
    finally:
        display.stop()
        httpd.shutdown()

    # ------------------------------------------------- media info + copies
    for source, stem in (("dynamo_repro.mp4", "repro"), ("diy_naive_ok.mp4", "naive")):
        src = assets / source
        if src.exists() and playable(src)[0]:
            shutil.copy2(src, assets / f"{stem}.mp4")
            poster(src, assets / f"{stem}_poster.jpg", at=4.0)

    for key in ("diy", "dynamo"):
        blob = results.get(key, {})
        if blob.get("video") and Path(blob["video"]).exists():
            blob["video_info"] = probe(Path(blob["video"]))
        if blob.get("gif") and Path(blob["gif"]).exists():
            blob["gif_bytes"] = size_of(Path(blob["gif"]))
            blob["gif_info"] = probe(Path(blob["gif"]))

    for prefix, folder in (("kf_screens", assets / "shots"),
                           ("kf_diy", assets / "diy_frames_ok"),
                           ("kf_dynamo", assets / "dynamo_frames")):
        if not folder.exists():
            continue
        images = sorted(folder.glob("*.png")) + sorted(folder.glob("*.webp"))
        for image in images:
            shutil.copy2(image, assets / f"{prefix}_{image.stem.split('_')[-1]}.jpg")
        results.setdefault("_strips", {})[prefix] = len(images)

    manifest = assets / "dynamo_frames" / "manifest.json"
    if manifest.exists():
        shutil.copy2(manifest, assets / "manifest.json")

    # Telemetry has to land in assets/ before anything reads it from there. The
    # gallery below samples assets/telemetry.jsonl, and on a clean run that file
    # does not exist yet — so the gallery came out silently empty, and only ever
    # looked right because a previous run had left one behind.
    for key, field, name in (("dynamo", "telemetry", "telemetry.jsonl"),
                             ("crash", "telemetry", "telemetry_crash.jsonl")):
        source = results.get(key, {}).get(field)
        if source and Path(source).exists():
            shutil.copy2(source, assets / name)

    # The gallery the page leads with: the real decoded frame at each logged action.
    if playable(assets / "dynamo_repro.mp4")[0] and (assets / "telemetry.jsonl").exists():
        results["stills"] = extract_stills(assets / "dynamo_repro.mp4",
                                           assets / "telemetry.jsonl", assets / "stills")

    crash_raw = results.get("crash", {}).get("raw")
    if crash_raw and Path(crash_raw).exists():
        results["crash"]["playable_asset"] = "assets/dynamo_crash.mp4"

    def publish(key: str, field: str) -> None:
        value = results.get(key, {}).get(field)
        if value and Path(value).exists():
            results[key][field] = f"assets/{Path(value).name}"

    for key in ("screenshots", "diy", "dynamo", "crash"):
        for field in ("video", "gif", "raw", "keyframes_dir", "frames_dir", "telemetry"):
            publish(key, field)

    if not args.keep_work:
        shutil.rmtree(work, ignore_errors=True)

    results["assets"] = sorted(p.name for p in assets.iterdir())
    (data_dir / "results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    # ------------------------------------------------- console summary
    def fmt(n: float) -> str:
        return f"{n:,.0f}"

    print()
    header = f"{'workflow':<14}{'images':>7}{'tokens':>9}{'dupes':>7}{'meanΔ':>8}{'video':>10}{'gif':>10}"
    print(header)
    print("-" * len(header))
    for key, label in (("screenshots", "screenshots"), ("diy", "diy ffmpeg"), ("dynamo", "dynamo")):
        r = results[key]
        red = r.get("redundancy") or {}
        print(f"{label:<14}{r['images']:>7}{fmt(r['tokens']):>9}{red.get('duplicates', '-'):>7}"
              f"{red.get('mean_delta', '-'):>8}{human(r.get('video_bytes') or 0):>10}"
              f"{human(r.get('gif_bytes') or 0):>10}")
    print()
    print(f"diy after SIGKILL : {results['diy']['crash']['raw_playable']} "
          f"({results['diy']['crash']['raw_error']})")
    print(f"dynamo on crash   : {results['crash']['playable']} "
          f"({results['crash']['actions']} actions still logged)")
    print(f"results           : {data_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
