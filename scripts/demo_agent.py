#!/usr/bin/env python3
"""Runnable demonstration of the whole Dynamo pipeline.

Drives a real GUI inside the harness display: logs every click, grabs a
mid-run screenshot, then exports an MP4, a GIF and a VLM keyframe set.

    python scripts/demo_agent.py --out output/

If tkinter is unavailable the demo degrades to repainting the X root window with
``xsetroot``, which still produces genuine scene changes for the reducer to find.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from agent_visual_harness import HarnessError, VisualHarness

STEPS = [
    ("#101820", "idle"),
    ("#c8ff3d", "build started"),
    ("#4fd8ff", "tests running"),
    ("#ff5f5f", "test failed"),
    ("#ffb300", "retrying"),
    ("#43d17a", "tests passed"),
    ("#e8e8e8", "deploying"),
    ("#7a5cff", "shipped"),
]


def run_gui(harness: VisualHarness, duration: float) -> None:
    """Drive a real window when tkinter is present; otherwise repaint the root."""
    # Tk reads DISPLAY from the environment at Tk() construction time and would
    # otherwise attach to whatever the agent shell happened to export (:0).
    os.environ["DISPLAY"] = harness.display
    try:
        import tkinter as tk
    except ImportError:
        _repaint_loop(harness, duration)
        return

    try:
        root = tk.Tk()
    except Exception as exc:  # no Tk compiled in, or no usable Tk build
        print(f"  ! tkinter unavailable ({exc}); falling back to xsetroot", file=sys.stderr)
        _repaint_loop(harness, duration)
        return

    width, height = harness.size
    root.title("Dynamo demo — CI dashboard")
    root.geometry(f"{min(960, width - 80)}x{min(600, height - 80)}+40+40")
    root.configure(bg="#0a0b0d")

    header = tk.Label(
        root, text="agent-e2e / visual-run", bg="#0a0b0d", fg="#c8ff3d",
        font=("TkFixedFont", 22, "bold"), anchor="w", padx=24, pady=16,
    )
    header.pack(fill="x")

    body = tk.Frame(root, bg="#0a0b0d")
    body.pack(fill="both", expand=True, padx=24, pady=8)

    status = tk.Label(
        body, text="idle", bg="#0a0b0d", fg="#e8e8e8",
        font=("TkFixedFont", 18), anchor="w",
    )
    status.pack(fill="x", pady=(0, 12))

    buttons: dict[str, tk.Button] = {}
    for colour, label in STEPS:
        button = tk.Button(
            body, text=label, bg=colour, fg="#0a0b0d", activebackground=colour,
            relief="flat", font=("TkFixedFont", 13), padx=18, pady=10,
        )
        button.pack(side="left", padx=(0, 8))
        buttons[label] = button

    log = tk.Text(body, bg="#121417", fg="#9aa4b2", height=8, relief="flat",
                  font=("TkFixedFont", 11))
    log.pack(fill="both", expand=True, pady=(16, 0))

    def pump(milliseconds: int = 120) -> None:
        """Keep the Tk event loop alive so the window actually repaints."""
        end = time.monotonic() + milliseconds / 1000.0
        while time.monotonic() < end:
            root.update()
            time.sleep(0.01)

    pump(700)
    harness.log_action("focus", target="window")
    log.insert("end", "session started\n")

    deadline = time.monotonic() + duration
    for _colour, label in STEPS:
        if time.monotonic() > deadline:
            break
        status.configure(text=label)
        button = buttons[label]
        button.configure(relief="sunken")
        root.update()

        # Report the click in *display* coordinates, which is what the overlay
        # renderer and the manifest expect.
        x = button.winfo_rootx() + button.winfo_width() // 2
        y = button.winfo_rooty() + button.winfo_height() // 2
        harness.log_action("click", x=x, y=y, target=label)

        log.insert("end", f"{label}\n")
        log.see("end")
        pump(650)
        button.configure(relief="flat")
        root.update()
        pump(150)

    harness.log_action("scroll", dy=-320, target="log")
    pump(400)
    root.destroy()


def _repaint_loop(harness: VisualHarness, duration: float) -> None:
    """Fallback: repaint the X root so the capture has real scene changes."""
    import shutil
    import subprocess

    if not shutil.which("xsetroot"):
        print("  ! no GUI toolkit and no xsetroot; the capture will be static", file=sys.stderr)
        time.sleep(duration)
        return
    per_step = max(0.2, duration / len(STEPS))
    for colour, label in STEPS:
        subprocess.run(
            ["xsetroot", "-display", harness.display, "-solid", colour],
            capture_output=True,
            check=False,
        )
        harness.log_action("click", x=harness.size[0] // 2, y=harness.size[1] // 2, target=label)
        time.sleep(per_step)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dynamo end-to-end demo")
    parser.add_argument("--out", default="output", help="artifact directory (default: output)")
    parser.add_argument("--session", default="dynamo-session", help="session directory")
    parser.add_argument("--display", default=":99", help="display to create")
    parser.add_argument("--resolution", default="1280x800x24")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--duration", type=float, default=8.0, help="seconds of GUI activity")
    parser.add_argument("--scene", type=float, default=0.05)
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Dynamo demo — {args.display} @ {args.resolution}, {args.fps} fps")
    try:
        with VisualHarness(
            display=args.display,
            resolution=args.resolution,
            fps=args.fps,
            session_dir=args.session,
            overlay_clicks=True,
            scene_threshold=args.scene,
        ) as harness:
            print(f"  display {harness.display} ready at {harness.size[0]}x{harness.size[1]}")
            run_gui(harness, args.duration)

            shot = harness.save_screenshot(out / "midrun.png")
            print(f"  mid-run screenshot -> {shot} ({shot.stat().st_size:,} bytes)")

        artifacts = harness.export(
            video_path=out / "repro.mp4",
            gif_path=out / "demo.gif",
            vlm_keyframes_dir=out / "vlm_frames",
        )
    except HarnessError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    print("\n" + artifacts.summary())
    print(f"\nGenerated {len(artifacts.keyframes)} frames for VLM context.")
    print(f"Telemetry: {artifacts.telemetry_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
