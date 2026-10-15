"""``dynamo`` — the command-line half of the harness.

The subcommands mirror the three things you actually do with a recording, plus
the two that answer questions about one:

``dynamo record``
    Bring up a display, start the recorder, run your command inside it, and
    always leave a playable file behind — even if the command is killed.

``dynamo export``
    Compile a finished session into MP4 / WebM / GIF artifacts, optionally with
    click overlays burned into the video.

``dynamo keyframes``
    Produce a token-budgeted, manifest-described keyframe set for a VLM.

``dynamo triage``
    Answer one question about a keyframe set with the cheapest model that can:
    skip what did not change, filter, crop to what moved, and only then ask an
    open-weight VLM.  Needs a model server, and says so rather than stubbing.

``dynamo capture``, ``diff``, ``ocr``, ``som``, ``info``
    The inspection primitives, each usable on its own and none of them needing a
    model at all.

Every argument the Python SDK accepts has a flag here; nothing is SDK-only.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from collections.abc import Sequence

from . import __version__
from ._ffmpeg import FFMPEG, FFPROBE, have_binary, probe
from .coordinates import Scale
from .display import VirtualDisplay
from .errors import HarnessError
from .processor import VALID_DITHERS, Processor
from .recorder import Recorder
from .session import Session
from .token_reducer import FRAME_FORMATS, ReducerConfig, TokenReducer
from .vision import DEFAULT_LAYA_ENDPOINT, DEFAULT_OLLAMA_ENDPOINT, DEFAULT_VLM_MODEL

__all__ = ["main", "build_parser"]

PROGRAM = "dynamo"

GREEN = "\033[92m"
CYAN = "\033[96m"
RED = "\033[91m"
DIM = "\033[2m"
RESET = "\033[0m"


def _colour(text: str, code: str, stream) -> str:
    if not stream.isatty() or os.environ.get("NO_COLOR"):
        return text
    return f"{code}{text}{RESET}"


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _ok(message: str) -> None:
    _log(f"{_colour('✓', GREEN, sys.stderr)} {message}")


def _info(message: str) -> None:
    _log(f"{_colour('·', CYAN, sys.stderr)} {message}")


def _warn(message: str) -> None:
    _log(f"{_colour('!', DIM, sys.stderr)} {message}")


def _fail(message: str) -> None:
    _log(f"{_colour('✗', RED, sys.stderr)} {message}")


# --------------------------------------------------------------------- record


def cmd_record(args: argparse.Namespace) -> int:
    """Start a display + recorder, run a command inside them, then finalise."""
    if not args.command:
        _fail("nothing to run. Usage: dynamo record [options] -- <command> [args...]")
        return 2

    display = VirtualDisplay(
        display=args.display,
        resolution=args.resolution,
        depth=args.depth,
        dpi=args.dpi,
        timeout=args.timeout,
        adopt_existing=args.adopt_existing,
    )
    session_dir = Path(args.session)
    session = Session.create(
        session_dir,
        display=display.display,
        resolution=display.resolution,
        depth=args.depth,
        fps=args.fps,
        crf=args.crf,
        preset=args.preset,
        command=list(args.command),
    )
    _info(f"session  {session_dir}")
    _info(f"display  {display.display}  {display.size}x{display.depth} @ {display.dpi}dpi")

    recorder: Recorder | None = None
    child: subprocess.Popen | None = None
    interrupted = False
    started_at = time.monotonic()

    try:
        display.start()
        _ok(f"display {display.display} ready ({display.size}x{display.depth})")

        if not args.no_record:
            recorder = Recorder(
                display=display.display,
                size=display.resolution,
                output=session.raw_path,
                fps=args.fps,
                crf=args.crf,
                preset=args.preset,
                telemetry_path=session.telemetry_path,
                draw_mouse=0 if args.hide_cursor else 1,
                startup_timeout=args.timeout,
            )
            recorder.start()
            _ok(f"recording at {args.fps} fps -> {session.raw_path.name}")

        env = {**os.environ, "DISPLAY": display.display}
        env.update(_parse_env_pairs(args.env))
        _info(f"running  {' '.join(args.command)}")
        child = subprocess.Popen(list(args.command), env=env)  # noqa: S603 - argv list
        returncode = child.wait()
        _info(f"child exited with {returncode}")
    except HarnessError as exc:
        _fail(str(exc))
        return 1
    except OSError as exc:
        _fail(f"could not run {args.command[0]!r}: {exc}")
        return 127
    except KeyboardInterrupt:
        interrupted = True
        _warn("interrupted — finalising recording")
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:  # pragma: no cover
                child.kill()
        return 130
    finally:
        if recorder is not None:
            try:
                recorder.stop()
                if recorder.was_repaired:
                    _warn("raw capture needed a repair pass to become playable")
                _ok(f"raw capture finalised -> {session.raw_path}")
            except HarnessError as exc:
                _fail(f"could not finalise recording: {exc}")
        display.stop()
        session.mark_finished(
            duration=time.monotonic() - started_at,
            interrupted=interrupted,
            child_returncode=None if interrupted else returncode,
        )

    # Optionally chain straight into artifact compilation.
    if not interrupted and returncode == 0:
        if args.mp4 or args.gif or args.vlm_frames:
            export_args = argparse.Namespace(
                session=str(session_dir),
                mp4=args.mp4,
                webm=None,
                gif=args.gif,
                vlm_frames=args.vlm_frames,
                crf=args.export_crf,
                preset=args.preset,
                max_colors=args.max_colors,
                dither=args.dither,
                gif_fps=args.gif_fps,
                overlay_clicks=args.overlay_clicks,
                scene=args.scene,
                max_size=args.max_size,
                fmt=args.fmt,
            )
            return cmd_export(export_args)

    return returncode


# --------------------------------------------------------------------- export


def cmd_export(args: argparse.Namespace) -> int:
    """Compile a finished session into MP4 / WebM / GIF artifacts."""
    session = Session.load(args.session)
    raw = session.require_recording()
    actions = session.actions
    processor = Processor(
        crf=args.crf,
        preset=args.preset,
        gif_max_colors=args.max_colors,
        gif_dither=args.dither,
        gif_fps=args.gif_fps,
        overlay_clicks=args.overlay_clicks,
    )

    _info(f"session  {session.root}")
    _info(f"raw      {raw.name}  ({len(actions)} actions logged)")
    if not (args.mp4 or args.webm or args.gif or args.vlm_frames):
        _warn("no output requested; pass at least one of --mp4 / --webm / --gif / --vlm-frames")

    try:
        if args.mp4:
            out = processor.export_video(
                raw, args.mp4, actions=actions, source_size=session.resolution, crf=args.crf
            )
            _ok(f"mp4      {out}")
        if args.webm:
            out = processor.to_video(raw, args.webm, crf=args.crf)
            _ok(f"webm     {out}")
        if args.gif:
            out = processor.to_gif(
                raw, args.gif, max_colors=args.max_colors, dither=args.dither, fps=args.gif_fps
            )
            _ok(f"gif      {out}")
        if args.vlm_frames:
            reducer = TokenReducer(
                config=ReducerConfig(
                    scene_threshold=args.scene, max_size=args.max_size, fmt=args.fmt
                )
            )
            keyframes = reducer.reduce(
                raw,
                args.vlm_frames,
                actions=actions,
                source_size=session.resolution,
                session_dir=session.root,
            )
            _ok(f"keyframes {len(keyframes)} kept -> {args.vlm_frames}")
    except HarnessError as exc:
        _fail(str(exc))
        return 1

    return 0


# ------------------------------------------------------------------ keyframes


def cmd_keyframes(args: argparse.Namespace) -> int:
    """Produce a token-budgeted keyframe set for a VLM."""
    session = Session.load(args.session)
    raw = session.require_recording()
    actions = session.actions if args.with_actions else []
    reducer = TokenReducer(
        config=ReducerConfig(
            scene_threshold=args.scene,
            max_size=args.max_size,
            fmt=args.fmt,
            quality=args.quality,
            max_frames=args.max_frames,
            min_gap=args.min_gap,
        )
    )
    try:
        keyframes = reducer.reduce(
            raw, args.out, actions=actions,
            source_size=session.resolution, session_dir=session.root,
        )
    except HarnessError as exc:
        _fail(str(exc))
        return 1

    info = probe(raw)
    raw_estimate = info.nb_frames or int(info.duration * (info.fps or session.fps))
    saved = 100.0 * (1 - len(keyframes) / max(1, raw_estimate))
    print(
        f"{_colour('✓', GREEN, sys.stdout)} {len(keyframes)} keyframes · "
        f"~{raw_estimate:,} frames scanned · {saved:.1f}% fewer · manifest.json written"
    )
    if args.verbose:
        for frame in keyframes:
            print(
                f"  {frame.index:>3}  {frame.timestamp:>8.3f}s  {frame.reason:<6} "
                f"{frame.action or '':<12} {frame.path.name}"
            )
    return 0


def cmd_triage(args: argparse.Namespace) -> int:
    """Answer one question about a keyframe set with the cheapest model that can.

    The report is the output, not the answers alone: how many frames reached a
    generative model, how many were cropped first, and what the whole set would
    have cost uncropped. That is the part a caller cannot get by asking a model
    directly.
    """
    from .vision import (
        LayaBackend,
        OllamaBackend,
        StubBackend,
        Triage,
        frames_from_manifest,
    )

    try:
        frames = frames_from_manifest(args.frames, limit=args.limit, ocr=args.ocr)
    except (HarnessError, OSError, KeyError, ValueError) as exc:
        _fail(str(exc))
        return 1
    if not frames:
        _fail(f"no keyframes in {args.frames}")
        return 1

    filter_backend = None
    if args.stub:
        primary = StubBackend(
            name="stub-primary",
            answers={args.question: "stub: no model answered these frames"},
        )
        if not args.no_filter:
            filter_backend = StubBackend(name="stub-filter")
        _warn("--stub: no model is answering, so none of this is a measurement")
    else:
        primary = OllamaBackend(args.model, args.ollama)
        if not primary.available():
            _fail(
                f"no model answered at {args.ollama}. Start one (ollama serve) or "
                f"pass --stub — this will not silently fall back to a stub."
            )
            return 1
        if not args.no_filter:
            candidate = LayaBackend(args.laya)
            if candidate.available():
                filter_backend = candidate
            else:
                _warn(f"no filter answered at {args.laya}; scoring was skipped")

    triage = Triage(
        primary,
        filter=filter_backend,
        crop=not args.no_crop,
        skip_unchanged=not args.keep_all,
        filter_threshold=args.threshold,
    )

    def announce(answer, index: int, total: int) -> None:
        state = (
            f"skipped: {answer.skipped_reason}"
            if answer.skipped
            else (answer.text[:60] or "(empty answer)")
        )
        print(
            f"  [{index:>3}/{total}] {answer.frame:<20} "
            f"{answer.tokens_before:>6} → {answer.tokens_after:>6} tok  "
            f"{state}",
            flush=True,
        )

    report = triage.run(
        frames, args.question, on_answer=announce if args.verbose else None
    )
    print(report.summary())

    if args.out:
        Path(args.out).write_text(
            json.dumps(report.to_dict(), indent=2), encoding="utf-8"
        )
        print(f"report   : {args.out}")
    return 0


# ---------------------------------------------------------------- inspection


def _resolve_display(args) -> tuple[str, tuple[int, int]]:
    """Work out which display to grab and how big it is."""
    display = getattr(args, "display", None) or os.environ.get("DISPLAY") or ":0"
    width, height = getattr(args, "width", None), getattr(args, "height", None)
    if width and height:
        return display, (width, height)
    if shutil.which("xdotool"):  # pragma: no cover - optional probe
        proc = subprocess.run(["xdotool", "getdisplaygeometry"],
                              capture_output=True, text=True)
        parts = proc.stdout.split()
        if len(parts) == 2 and all(p.isdigit() for p in parts):
            return display, (int(parts[0]), int(parts[1]))
    return display, (width or 1920, height or 1080)


def _parse_box(text: str | None) -> tuple[int, int, int, int] | None:
    if not text:
        return None
    parts = [p for p in str(text).replace(" ", "").split(",") if p]
    if len(parts) != 4:
        raise SystemExit(f"region must be X,Y,W,H — got {text!r}")
    try:
        x, y, w, h = (int(p) for p in parts)
    except ValueError as exc:
        raise SystemExit(f"region must be integers — got {text!r}") from exc
    if w <= 0 or h <= 0:
        raise SystemExit(f"region must have a positive size — got {text!r}")
    return (x, y, w, h)


def _scale_from(args) -> Scale:
    raw = [p for p in str(getattr(args, "origin", "0,0")).split(",") if p]
    origin = (int(raw[0]), int(raw[1]) if len(raw) > 1 else 0)
    return Scale(ratio=float(getattr(args, "scale", 1.0)), origin=origin)


def cmd_capture(args) -> int:
    """Static capture: the whole surface, or one region of it."""
    from .capture import Surface, save_image

    display, size = _resolve_display(args)
    surface = Surface.from_display(display, size)
    box = _parse_box(args.region)

    if box is not None:
        data = surface.snip_scaled(box, scale=_scale_from(args), fmt=args.format)
    else:
        data = surface.grab(fmt=args.format)

    if args.out:
        path = save_image(args.out, data)
        print(f"{_colour('✓', GREEN, sys.stdout)} {len(data):,} B -> {path}")
    else:
        sys.stdout.buffer.write(data)
    return 0


def cmd_ocr(args) -> int:
    """Recover text from pixels: a file, or the live display."""
    from .capture import Surface
    from .ocr import read_image, read_region

    if args.image:
        data = Path(args.image).read_bytes()
    else:
        display, size = _resolve_display(args)
        data = Surface.from_display(display, size).grab()

    box = _parse_box(args.region)
    if box is not None:
        result = read_region(data, box, psm=args.psm, lang=args.lang)
    else:
        result = read_image(data, psm=args.psm, lang=args.lang)

    if args.find:
        from .ocr import fold as _fold

        needle = _fold(args.find) if args.fuzzy else args.find.lower()
        hit = next(
            (w for w in result.words
             if needle in (_fold(w.text) if args.fuzzy else w.text.lower())),
            None,
        )
        if hit is None:
            _fail(f"{args.find!r} not found on screen"
                  + ("" if args.fuzzy else " (try --fuzzy)"))
            return 1
        print(f"{hit.text}\t{hit.box[0]}\t{hit.box[1]}\t{hit.box[2]}\t{hit.box[3]}\t{hit.confidence:.0f}")
        return 0

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    print(result.text.rstrip())
    _log(_colour(f"\n({result.line_count} lines · confidence {result.confidence:.0f}%"
                 f" · {result.elapsed:.2f}s)", DIM, sys.stderr))
    return 0


def cmd_diff(args) -> int:
    """Visual regression diff. Exits 1 when the images differ."""
    from .diff import compare

    try:
        result = compare(
            Path(args.before).read_bytes(),
            Path(args.after).read_bytes(),
            threshold=args.threshold,
            min_changed_pct=args.min_changed_pct,
        )
    except HarnessError as exc:
        _fail(str(exc))
        return 1

    if args.out:
        result.save_diff(args.out)

    if args.json or not args.quiet:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(result.verdict())

    if not result.same:
        _fail(result.verdict())
        return 1
    return 0


def cmd_som(args) -> int:
    """Number the elements on a screen so a model can point at one."""
    from .grounding import annotate_boxes, resolve

    if args.image:
        data = Path(args.image).read_bytes()
    else:
        from .capture import Surface

        display, size = _resolve_display(args)
        data = Surface.from_display(display, size).grab()

    raw = args.boxes.strip()
    payload = json.loads(raw if raw.startswith("[") else Path(raw).read_text(encoding="utf-8"))
    boxes = [tuple(int(v) for v in item[:4]) for item in payload]
    if not boxes:
        _fail("no boxes to annotate")
        return 1

    marks = annotate_boxes(
        data, boxes,
        labels=args.labels.split(",") if args.labels else None,
        roles=args.roles.split(",") if args.roles else None,
        scale=_scale_from(args),
    )
    marks.save(args.out)
    if args.legend:
        Path(args.legend).write_text(marks.compact(), encoding="utf-8")

    print(f"{_colour('✓', GREEN, sys.stdout)} {len(marks)} marks -> {args.out}")
    if args.resolve:
        try:
            x, y = resolve(marks, args.resolve)
            print(f"index {args.resolve} -> ({x}, {y})")
        except ValueError as exc:
            _fail(str(exc))
            return 1
    return 0


# ----------------------------------------------------------------------- info


def cmd_info(args: argparse.Namespace) -> int:
    """Print what Dynamo can see on this machine, and what a session contains."""
    if args.session:
        session = Session.load(args.session)
        payload = session.to_dict()
        try:
            payload["media"] = session.media_info().to_dict()
        except HarnessError as exc:
            payload["media"] = {"error": str(exc)}
        print(json.dumps(payload, indent=2, default=str))
        return 0

    payload = {
        "version": __version__,
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "binaries": {
            "ffmpeg": FFMPEG if have_binary(FFMPEG) else None,
            "ffprobe": FFPROBE if have_binary(FFPROBE) else None,
            "Xvfb": have_binary("Xvfb"),
        },
        "optional": {
            # OCR and grounding are opt-in: they work without these, but these
            # are what make them work *well*, so an agent should be told.
            "tesseract": have_binary("tesseract"),
            "fonts": have_binary("fc-match") or Path("/usr/share/fonts").exists(),
        },
        "capabilities": {
            "record": bool(have_binary(FFMPEG) and have_binary("Xvfb")),
            "export": bool(have_binary(FFMPEG)),
            "capture": bool(have_binary(FFMPEG)),
            "ocr": have_binary("tesseract"),
            "diff": bool(have_binary(FFMPEG)),
            "som": bool(have_binary(FFMPEG)) and (
                have_binary("fc-match") or Path("/usr/share/fonts").exists()
            ),
        },
        "session_dir": os.getcwd(),
    }
    print(json.dumps(payload, indent=2))
    core = payload["capabilities"]
    return 0 if all(core[name] for name in ("record", "export", "capture", "diff")) else 1


# --------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROGRAM,
        description="Dynamo — the AI Visual & Video Harness for coding agents.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  dynamo record --display :99 --fps 24 -- python run_agent.py\n"
            "  dynamo export ./session --mp4 out/repro.mp4 --gif out/demo.gif --overlay-clicks\n"
            "  dynamo keyframes ./session --scene 0.05 --max-size 1280 --format webp --out out/frames/\n"
            "  dynamo triage out/frames/ --question \"what did the user see go wrong?\" --out report.json\n"
            "  dynamo capture --region 0,0,480,90 --out banner.png\n"
            "  dynamo ocr --find ECONNREFUSED\n"
            "  dynamo diff before.png after.png --out diff.png\n"
            "  dynamo som shot.png --boxes boxes.json --labels Pay,Cancel --out som.png\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"{PROGRAM} {__version__}")
    sub = parser.add_subparsers(dest="command_name", metavar="<command>")

    # ---- record
    rec = sub.add_parser("record", help="record a command inside a managed display")
    rec.add_argument("--display", default=":99", help="display number (default: :99)")
    rec.add_argument("--resolution", default="1920x1080x24", help="geometry, e.g. 1920x1080x24")
    rec.add_argument("--depth", type=int, default=24, help="colour depth (default: 24)")
    rec.add_argument("--dpi", type=int, default=96, help="reported DPI (default: 96)")
    rec.add_argument("--fps", type=int, default=24, help="capture framerate (default: 24)")
    rec.add_argument("--crf", type=int, default=16, help="raw capture CRF; lower is better (default: 16)")
    rec.add_argument("--preset", default="veryfast", help="x264 preset (default: veryfast)")
    rec.add_argument("--session", default="dynamo-session", help="session directory")
    rec.add_argument("--timeout", type=float, default=30.0, help="startup timeout in seconds")
    rec.add_argument(
        "--adopt-existing",
        action="store_true",
        help="use an already-running X server on this display instead of starting one",
    )
    rec.add_argument("--no-record", action="store_true", help="only start the display")
    rec.add_argument("--hide-cursor", action="store_true", help="do not draw the pointer into the video")
    rec.add_argument("--env", action="append", default=[], metavar="K=V", help="extra env var for the child")
    rec.add_argument("--mp4", help="also export an MP4 when the child exits cleanly")
    rec.add_argument("--gif", help="also export a GIF when the child exits cleanly")
    rec.add_argument("--vlm-frames", help="also extract keyframes when the child exits cleanly")
    rec.add_argument("--export-crf", type=int, default=23, help="CRF for the chained export")
    rec.add_argument("--max-colors", type=int, default=128, help="GIF palette size")
    rec.add_argument("--dither", default="bayer", choices=VALID_DITHERS, help="GIF dithering")
    rec.add_argument("--gif-fps", type=int, default=15, help="GIF framerate")
    rec.add_argument("--overlay-clicks", action="store_true", help="burn click rings into the MP4")
    rec.add_argument("--scene", type=float, default=0.05, help="scene threshold for keyframes")
    rec.add_argument("--max-size", type=int, default=1280, help="keyframe width cap")
    rec.add_argument(
        "--format", "--fmt", dest="fmt", default="webp",
        choices=sorted(set(FRAME_FORMATS)), help="keyframe format (default: webp)",
    )
    rec.add_argument("command", nargs=argparse.REMAINDER, help="-- <command> [args...]")
    rec.set_defaults(func=cmd_record)

    # ---- export
    exp = sub.add_parser("export", help="compile a session into MP4 / WebM / GIF")
    exp.add_argument("session", help="session directory")
    exp.add_argument("--mp4", help="output MP4 path")
    exp.add_argument("--webm", help="output WebM path")
    exp.add_argument("--gif", help="output GIF path")
    exp.add_argument("--vlm-frames", help="output keyframe directory")
    exp.add_argument("--crf", type=int, default=23, help="H.264 / VP9 quality")
    exp.add_argument("--preset", default="medium", help="x264 preset (default: medium)")
    exp.add_argument("--max-colors", type=int, default=128, help="GIF palette size")
    exp.add_argument("--dither", default="bayer", choices=VALID_DITHERS, help="GIF dithering")
    exp.add_argument("--gif-fps", type=int, default=15, help="GIF framerate")
    exp.add_argument("--overlay-clicks", action="store_true", help="burn click rings into the video")
    exp.add_argument("--scene", type=float, default=0.05, help="scene threshold for keyframes")
    exp.add_argument("--max-size", type=int, default=1280, help="keyframe width cap")
    exp.add_argument(
        "--format", "--fmt", dest="fmt", default="webp",
        choices=sorted(set(FRAME_FORMATS)), help="keyframe format (default: webp)",
    )
    exp.set_defaults(func=cmd_export)

    # ---- keyframes
    kf = sub.add_parser("keyframes", help="extract a token-budgeted keyframe set")
    kf.add_argument("session", help="session directory")
    kf.add_argument("--scene", type=float, default=0.05, help="scene-change threshold (default: 0.05)")
    kf.add_argument("--max-size", type=int, default=1280, help="max frame width (default: 1280)")
    kf.add_argument(
        "--format", "--fmt", dest="fmt", default="webp",
        choices=sorted(set(FRAME_FORMATS)), help="frame format (default: webp)",
    )
    kf.add_argument("--quality", type=int, default=82, help="encoder quality")
    kf.add_argument("--max-frames", type=int, default=None, help="hard cap on frames kept")
    kf.add_argument("--min-gap", type=float, default=0.30, help="dedupe window in seconds")
    kf.add_argument("--out", required=True, help="output directory for frames + manifest.json")
    kf.add_argument(
        "--no-actions",
        dest="with_actions",
        action="store_false",
        help="ignore telemetry anchors and keep scene changes only",
    )
    kf.add_argument("-v", "--verbose", action="store_true", help="list every kept frame")
    kf.set_defaults(func=cmd_keyframes, with_actions=True)

    # ---- capture (static screenshot + region snip)
    cap = sub.add_parser("capture", help="take a screenshot, or a region of one")
    cap.add_argument("--display", default=None, help="display to capture (default: $DISPLAY)")
    cap.add_argument("--width", type=int, default=None, help="surface width, if not probing")
    cap.add_argument("--height", type=int, default=None, help="surface height, if not probing")
    cap.add_argument("--region", help="X,Y,W,H to crop, in logical pixels")
    cap.add_argument("--out", help="write to this file instead of stdout")
    cap.add_argument("--format", default="png", choices=["png", "jpeg", "webp"])
    cap.add_argument("--scale", type=float, default=1.0,
                     help="device pixels per logical pixel (HiDPI)")
    cap.add_argument("--origin", default="0,0",
                     help="device-pixel offset of the logical origin, e.g. 0,85 for browser chrome")
    cap.set_defaults(func=cmd_capture)

    # ---- ocr
    ocr_p = sub.add_parser("ocr", help="read text off pixels")
    ocr_p.add_argument("image", help="image file, or omit to read $DISPLAY")
    ocr_p.add_argument("--display", default=None)
    ocr_p.add_argument("--width", type=int, default=None)
    ocr_p.add_argument("--height", type=int, default=None)
    ocr_p.add_argument("--region", help="X,Y,W,H to read, in pixels")
    ocr_p.add_argument("--psm", type=int, default=6, help="tesseract page-segmentation mode")
    ocr_p.add_argument("--lang", default="eng")
    ocr_p.add_argument("--find", help="locate a word and print its box, then exit")
    ocr_p.add_argument(
        "--fuzzy", action="store_true",
        help="match through the glyphs Tesseract confuses (0/O, 1/l/I, _/space)",
    )
    ocr_p.add_argument("--json", action="store_true", help="emit the full result as JSON")
    ocr_p.set_defaults(func=cmd_ocr)

    # ---- diff
    dif = sub.add_parser("diff", help="visual regression diff of two images")
    dif.add_argument("before")
    dif.add_argument("after")
    dif.add_argument("--threshold", type=float, default=0.10,
                     help="per-pixel luma distance counted as changed (0-1)")
    dif.add_argument("--min-changed-pct", type=float, default=0.05,
                     help="percent of pixels that must change to fail")
    dif.add_argument("--out", help="write the diff image here")
    dif.add_argument("--json", action="store_true")
    dif.add_argument("--quiet", action="store_true", help="only print the verdict")
    dif.set_defaults(func=cmd_diff)

    # ---- som (set-of-marks grounding)
    som = sub.add_parser("som", help="draw numbered element marks for grounding")
    som.add_argument("image", help="image file, or omit to grab $DISPLAY")
    som.add_argument("--display", default=None)
    som.add_argument("--width", type=int, default=None)
    som.add_argument("--height", type=int, default=None)
    som.add_argument("--boxes", required=True,
                     help="JSON file or inline array of [x,y,w,h] boxes")
    som.add_argument("--labels", help="comma-separated labels, one per box")
    som.add_argument("--roles", help="comma-separated roles, one per box")
    som.add_argument("--out", required=True, help="write the annotated image here")
    som.add_argument("--legend", help="also write the text legend here")
    som.add_argument("--scale", type=float, default=1.0)
    som.add_argument("--origin", default="0,0")
    som.add_argument("--resolve", help="resolve this index and print the click point")
    som.set_defaults(func=cmd_som)

    # ---- triage (route frames to a model)
    tri = sub.add_parser(
        "triage",
        help="answer one question about a keyframe set, using the cheapest model that can",
    )
    tri.add_argument("frames", help="keyframes directory, with its manifest.json")
    tri.add_argument("--question", required=True, help="what to ask about each frame")
    tri.add_argument("--model", default=DEFAULT_VLM_MODEL,
                     help="open-weight VLM to use (default: %(default)s)")
    tri.add_argument("--ollama", default=DEFAULT_OLLAMA_ENDPOINT,
                     help="Ollama endpoint (default: %(default)s)")
    tri.add_argument("--laya", default=DEFAULT_LAYA_ENDPOINT,
                     help="Laya server endpoint (default: %(default)s)")
    tri.add_argument("--no-filter", action="store_true",
                     help="skip the decision-model pre-filter")
    tri.add_argument("--no-crop", action="store_true",
                     help="send whole frames instead of cropping to what moved")
    tri.add_argument("--keep-all", action="store_true",
                     help="do not skip frames that changed nothing")
    tri.add_argument("--threshold", type=float, default=0.5,
                     help="filter's probability below which a frame is dropped")
    tri.add_argument("--ocr", action="store_true",
                     help="read screen text first, so the filter judges words not just actions")
    tri.add_argument("--limit", type=int, default=None,
                     help="use only the first N frames")
    tri.add_argument("--out", help="write the full report as JSON here")
    tri.add_argument("--stub", action="store_true",
                     help="run the pipeline with no model (explicitly not a measurement)")
    tri.add_argument("--verbose", action="store_true")
    tri.set_defaults(func=cmd_triage)

    # ---- info
    info_parser = sub.add_parser("info", help="show environment or session details")
    info_parser.add_argument("session", nargs="?", help="session directory (default: environment report)")
    info_parser.set_defaults(func=cmd_info)

    return parser


def _parse_env_pairs(pairs: Sequence[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for pair in pairs:
        key, _, value = str(pair).partition("=")
        if key:
            out[key] = value
    return out


def _utf8_streams() -> None:
    """Make stdout/stderr survive a console that cannot encode our output.

    Messages carry ✓, · and Δ, and Python picks the console encoding from the
    locale — which on Windows is cp1252. A user piping ``dynamo diff`` into a
    file, or an agent capturing the output, would get a UnicodeEncodeError
    *instead of* the diff result: the tool crashes exactly when it has something
    useful to say. ``errors="replace"`` costs a mangled glyph in the worst case
    and never an exit code.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:  # pragma: no branch - always on 3.7+
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # pragma: no cover - already-detached stream
                pass


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``dynamo`` console script."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if getattr(args, "command_name", None) is None:
        parser.print_help()
        return 0

    if args.command_name != "capture":
        # `capture` writes raw image bytes to stdout; re-encoding that would
        # corrupt the image, so it keeps the stream exactly as handed to it.
        _utf8_streams()

    if args.command_name == "record":
        command = list(args.command)
        if command and command[0] == "--":
            command = command[1:]
        args.command = command

    started = time.monotonic()
    try:
        code = int(args.func(args))
    except HarnessError as exc:
        _fail(str(exc))
        return 1
    except KeyboardInterrupt:  # pragma: no cover
        return 130
    if args.command_name != "info" and sys.stderr.isatty():
        _log(_colour(f"\n{dim_elapsed(time.monotonic() - started)}", DIM, sys.stderr))
    return code


def dim_elapsed(seconds: float) -> str:
    return f"done in {seconds:.1f}s"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
