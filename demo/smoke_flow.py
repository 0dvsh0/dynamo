"""Smoke-test the browser checkout flow on its own, before the long experiment.

Drives one session and reports the telemetry it produced plus the screen offset
used to map page coordinates into display coordinates.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, "/opt/dynamo/src")
sys.path.insert(0, "/mnt/d/coding/dynamo/demo")

from agent_visual_harness import VirtualDisplay, Recorder, Session as DynamoSession  # noqa: E402

import run_experiments as rx  # noqa: E402
from serve import start_background  # noqa: E402

display = VirtualDisplay(display=":153", resolution=(rx.WIDTH, rx.HEIGHT), timeout=30)
display.start()
os.environ["DISPLAY"] = display.display
httpd, url = start_background(8098)
print("storefront:", url, "| display:", display.display, f"{rx.WIDTH}x{rx.HEIGHT}")

work = Path("/tmp/smoke")
if work.exists():
    import shutil

    shutil.rmtree(work)
work.mkdir(parents=True)

session = rx.DynamoSession = None  # avoid confusion in the printout
ds = DynamoSession.create(work / "s", display=display.display,
                          resolution=(rx.WIDTH, rx.HEIGHT), fps=rx.FPS)
rec = Recorder(display=display.display, size=(rx.WIDTH, rx.HEIGHT),
               output=ds.raw_path, fps=rx.FPS, crf=20, preset="veryfast",
               telemetry_path=ds.telemetry_path)
rec.start()

browser = rx.CheckoutSession(display.display, url, headed=True)
print("viewport origin:", browser.viewport_origin())
print("window geometry:", browser.page.evaluate(
    "() => ({sx: window.screenX, sy: window.screenY, ow: window.outerWidth,"
    " oh: window.outerHeight, iw: window.innerWidth, ih: window.innerHeight})"))
print("page title:", browser.page.title())

try:
    actions = browser.run(on_action=lambda a: rec.log_action(a["label"], x=a["x"], y=a["y"]))
    print(f"\ncompleted flow: {len(actions)} positioned actions")
finally:
    browser.close()
    rec.stop()

print("\ntelemetry.jsonl:")
for line in ds.telemetry_path.read_text(encoding="utf-8").splitlines():
    print("  " + line)

info = rx.probe(ds.raw_path)
print(f"\nraw capture: {info}")

# A quick visual proof that the browser chrome is in the frame.
subprocess_ok = rx.poster(ds.raw_path, work / "proof.jpg", at=6.0)
print("proof frame:", subprocess_ok)

display.stop()
httpd.shutdown()
print("\nSMOKE OK ->", work)
