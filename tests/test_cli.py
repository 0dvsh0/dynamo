"""CLI tests — `dynamo record`, `dynamo export`, `dynamo keyframes`, `dynamo info`."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from agent_visual_harness.cli import build_parser, main

from .conftest import requires_ffmpeg, requires_linux, requires_xvfb


@pytest.fixture(scope="module")
def record_display() -> str:
    """A display number for ``dynamo record`` to own.

    The CLI always manages its own X server, so it must never be pointed at the
    session-scoped fixture's display — doing so would (correctly) be refused as
    "already in use".
    """
    from agent_visual_harness import is_display_up

    for number in range(150, 175):
        if not is_display_up(f":{number}"):
            return f":{number}"
    pytest.skip("no free display number for CLI tests")


@pytest.fixture
def cli_env() -> dict:
    """Run the CLI as `python -m agent_visual_harness` in a subprocess."""
    env = dict(os.environ)
    src = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    return env


def run_cli(cli_env: dict, *args: str, cwd: Path | None = None, timeout: float = 300.0):
    proc = subprocess.run(
        [sys.executable, "-m", "agent_visual_harness", *args],
        capture_output=True,
        text=True,
        env=cli_env,
        cwd=str(cwd) if cwd else None,
        timeout=timeout,
    )
    return proc


class TestParser:
    def test_no_arguments_prints_help(self, capsys):
        assert main([]) == 0
        assert "usage: dynamo" in capsys.readouterr().out

    def test_version(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main(["--version"])
        assert exc.value.code == 0
        assert "dynamo" in capsys.readouterr().out

    @pytest.mark.parametrize("command", ["record", "export", "keyframes", "info", "triage"])
    def test_subcommand_help(self, command, capsys):
        with pytest.raises(SystemExit) as exc:
            main([command, "--help"])
        assert exc.value.code == 0
        assert command in capsys.readouterr().out

    def test_record_without_a_command_fails_cleanly(self, capsys):
        assert main(["record"]) == 2
        assert "nothing to run" in capsys.readouterr().err

    def test_export_with_a_missing_session(self, tmp_path: Path, capsys):
        assert main(["export", str(tmp_path / "absent")]) == 1
        assert "does not exist" in capsys.readouterr().err


class TestInfo:
    def test_environment_report(self, cli_env, tmp_path: Path):
        proc = run_cli(cli_env, "info", cwd=tmp_path)
        assert proc.returncode in (0, 1)
        payload = json.loads(proc.stdout)
        assert payload["version"]
        assert "ffmpeg" in payload["binaries"]

    def test_capabilities_are_reported(self, cli_env, tmp_path: Path):
        """An agent should be able to ask what this machine can do, and be told."""
        payload = json.loads(run_cli(cli_env, "info", cwd=tmp_path).stdout)
        assert set(payload["capabilities"]) == {
            "record", "export", "capture", "ocr", "diff", "som",
        }
        assert payload["capabilities"]["capture"] is True
        assert payload["optional"]["tesseract"] == (shutil.which("tesseract") is not None)

    @requires_linux
    @requires_xvfb
    @requires_ffmpeg
    def test_session_report(self, cli_env, tmp_path: Path, synthetic_clip: Path):
        from agent_visual_harness import Session

        session = Session.create(tmp_path / "s", display=":99", resolution=(640, 480), fps=15)
        session.raw_path.write_bytes(synthetic_clip.read_bytes())
        (tmp_path / "s" / "telemetry.jsonl").write_text(
            '{"timestamp": 0.5, "action": "click", "x": 1, "y": 2}\n', encoding="utf-8"
        )
        proc = run_cli(cli_env, "info", str(tmp_path / "s"))
        assert proc.returncode == 0, proc.stderr
        payload = json.loads(proc.stdout)
        assert payload["actions"] == 1
        assert payload["media"]["width"] == 640


@pytest.mark.xfvb
@requires_linux
@requires_xvfb
@requires_ffmpeg
class TestRecordExportKeyframes:
    """The full CLI round trip on a real display."""

    @pytest.fixture
    def painter(self, tmp_path: Path) -> Path:
        """A child script that repaints the X root, creating real scene changes."""
        script = tmp_path / "paint.py"
        script.write_text(
            "import os, subprocess, sys, time\n"
            "colours = sys.argv[1:]\n"
            "for colour in colours:\n"
            "    subprocess.run(['xsetroot', '-display', os.environ['DISPLAY'],\n"
            "                    '-solid', colour], check=False)\n"
            "    time.sleep(0.45)\n",
            encoding="utf-8",
        )
        return script

    def test_record_then_export_then_keyframes(
        self, cli_env, tmp_path: Path, record_display, painter
    ):
        session = tmp_path / "session"
        workdir = tmp_path / "work"
        workdir.mkdir()

        record = run_cli(
            cli_env,
            "record",
            "--display", record_display,
            "--resolution", "640x480x24",
            "--fps", "15",
            "--session", str(session),
            "--preset", "ultrafast",
            "--",
            sys.executable,
            str(painter),
            "#101010", "#f0f0f0", "#1070d0", "#f08010",
            cwd=workdir,
        )
        assert record.returncode == 0, f"record failed:\n{record.stderr}"
        assert (session / "raw.mp4").exists()
        assert (session / "meta.json").exists()

        meta = json.loads((session / "meta.json").read_text(encoding="utf-8"))
        assert meta["display"] == record_display
        assert "finished_at" in meta

        export = run_cli(
            cli_env,
            "export", str(session),
            "--mp4", str(tmp_path / "out.mp4"),
            "--crf", "28",
            "--preset", "ultrafast",
            cwd=workdir,
        )
        assert export.returncode == 0, f"export failed:\n{export.stderr}"
        assert (tmp_path / "out.mp4").exists()

        frames = run_cli(
            cli_env,
            "keyframes", str(session),
            "--scene", "0.05",
            "--max-size", "640",
            "--format", "jpeg",
            "--out", str(tmp_path / "kf"),
            "--verbose",
            cwd=workdir,
        )
        assert frames.returncode == 0, f"keyframes failed:\n{frames.stderr}"
        assert "manifest.json written" in frames.stdout

        manifest = json.loads((tmp_path / "kf" / "manifest.json").read_text(encoding="utf-8"))
        # Four distinct colours across ~2s must collapse to a handful of frames,
        # not the ~30 the raw capture holds.
        assert manifest["scene_frames_detected"] >= 3
        assert 0 < manifest["keyframes_kept"] < manifest["raw_frame_estimate"] / 2
        assert list((tmp_path / "kf").glob("frame_*.jpg"))
        # --verbose prints one line per kept frame: index, timestamp, reason.
        listed = [ln for ln in frames.stdout.splitlines() if "scene" in ln or "action" in ln]
        assert len(listed) == manifest["keyframes_kept"]

    def test_static_screen_with_no_actions_yields_no_keyframes(
        self, cli_env, tmp_path: Path, record_display
    ):
        """The token reducer's whole point: nothing happens, so keep nothing."""
        session = tmp_path / "s"
        proc = run_cli(
            cli_env,
            "record",
            "--display", record_display,
            "--resolution", "640x480x24",
            "--fps", "10",
            "--preset", "ultrafast",
            "--session", str(session),
            "--",
            sys.executable, "-c", "import time; time.sleep(1.0)",
        )
        assert proc.returncode == 0, proc.stderr

        out = run_cli(
            cli_env, "keyframes", str(session), "--out", str(tmp_path / "kf"),
            "--format", "jpeg",
        )
        assert out.returncode == 0, out.stderr
        manifest = json.loads((tmp_path / "kf" / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["scene_frames_detected"] == 0
        assert manifest["action_anchors"] == 0
        assert manifest["keyframes_kept"] == 0
        assert list((tmp_path / "kf").glob("frame_*.jpg")) == []
        assert "0 keyframes" in out.stdout

    def test_record_propagates_the_child_exit_code(self, cli_env, tmp_path: Path, record_display):
        proc = run_cli(
            cli_env,
            "record",
            "--display", record_display,
            "--resolution", "640x480x24",
            "--fps", "10",
            "--preset", "ultrafast",
            "--session", str(tmp_path / "s"),
            "--",
            sys.executable,
            "-c",
            "raise SystemExit(3)",
        )
        assert proc.returncode == 3, proc.stderr
        assert (tmp_path / "s" / "raw.mp4").exists(), "the recording must survive a failing agent"

    def test_record_passes_the_display_to_the_child(self, cli_env, tmp_path: Path, record_display):
        marker = tmp_path / "display.txt"
        proc = run_cli(
            cli_env,
            "record",
            "--display", record_display,
            "--resolution", "640x480x24",
            "--fps", "10",
            "--preset", "ultrafast",
            "--session", str(tmp_path / "s"),
            "--",
            sys.executable,
            "-c",
            f"open({str(marker)!r}, 'w').write(__import__('os').environ['DISPLAY'])",
        )
        assert proc.returncode == 0, proc.stderr
        assert marker.read_text().strip() == record_display

    def test_record_env_flag(self, cli_env, tmp_path: Path, record_display):
        proc = run_cli(
            cli_env,
            "record",
            "--display", record_display,
            "--resolution", "640x480x24",
            "--fps", "10",
            "--preset", "ultrafast",
            "--no-record",
            "--session", str(tmp_path / "s"),
            "--env", "DYNAMO_TEST=hello",
            "--",
            sys.executable,
            "-c",
            "import os,sys; sys.exit(0 if os.environ['DYNAMO_TEST']=='hello' else 1)",
        )
        assert proc.returncode == 0, proc.stderr

    def test_record_chains_into_export(self, cli_env, tmp_path: Path, record_display):
        proc = run_cli(
            cli_env,
            "record",
            "--display", record_display,
            "--resolution", "640x480x24",
            "--fps", "10",
            "--preset", "ultrafast",
            "--session", str(tmp_path / "s"),
            "--mp4", str(tmp_path / "chained.mp4"),
            "--export-crf", "30",
            "--",
            sys.executable,
            "-c",
            "pass",
        )
        assert proc.returncode == 0, proc.stderr
        assert (tmp_path / "chained.mp4").exists()

    def test_record_no_record_skips_capture(self, cli_env, tmp_path: Path, record_display):
        proc = run_cli(
            cli_env,
            "record",
            "--display", record_display,
            "--resolution", "640x480x24",
            "--no-record",
            "--session", str(tmp_path / "s"),
            "--",
            sys.executable,
            "-c",
            "import os,sys; sys.exit(0 if os.environ['DISPLAY'] else 1)",
        )
        assert proc.returncode == 0, proc.stderr
        assert not (tmp_path / "s" / "raw.mp4").exists()

    def test_export_with_overlays_and_gif(self, cli_env, tmp_path: Path, record_display):
        session = tmp_path / "s"
        assert (
            run_cli(
                cli_env,
                "record",
                "--display", record_display,
                "--resolution", "640x480x24",
                "--fps", "10",
                "--preset", "ultrafast",
                "--session", str(session),
                "--",
                sys.executable,
                "-c",
                "pass",
            ).returncode
            == 0
        )
        (session / "telemetry.jsonl").write_text(
            '{"timestamp": 0.2, "action": "click", "x": 100, "y": 100}\n', encoding="utf-8"
        )
        proc = run_cli(
            cli_env,
            "export", str(session),
            "--mp4", str(tmp_path / "o.mp4"),
            "--gif", str(tmp_path / "o.gif"),
            "--overlay-clicks",
            "--max-colors", "64",
            "--dither", "bayer",
            "--preset", "ultrafast",
        )
        assert proc.returncode == 0, proc.stderr
        assert (tmp_path / "o.mp4").exists()
        assert (tmp_path / "o.gif").read_bytes()[:6] in (b"GIF87a", b"GIF89a")

    def test_keyframes_no_actions_flag(self, cli_env, tmp_path: Path, record_display):
        session = tmp_path / "s"
        run_cli(
            cli_env,
            "record",
            "--display", record_display,
            "--resolution", "640x480x24",
            "--fps", "10",
            "--preset", "ultrafast",
            "--session", str(session),
            "--",
            sys.executable, "-c", "pass",
        )
        proc = run_cli(
            cli_env, "keyframes", str(session), "--out", str(tmp_path / "kf"),
            "--no-actions", "--format", "jpeg",
        )
        assert proc.returncode == 0, proc.stderr
        payload = json.loads((tmp_path / "kf" / "manifest.json").read_text(encoding="utf-8"))
        assert payload["action_anchors"] == 0

    def test_export_without_outputs_warns_but_succeeds(
        self, cli_env, tmp_path: Path, record_display
    ):
        session = tmp_path / "s"
        run_cli(
            cli_env,
            "record",
            "--display", record_display,
            "--resolution", "640x480x24",
            "--fps", "10",
            "--preset", "ultrafast",
            "--session", str(session),
            "--",
            sys.executable, "-c", "pass",
        )
        proc = run_cli(cli_env, "export", str(session))
        assert proc.returncode == 0
        assert "no output requested" in proc.stderr


class TestDocumentedInterface:
    """The exact commands shown in the project docs must keep working."""

    @pytest.mark.parametrize(
        "command,flags",
        [
            (
                "keyframes",
                ["--scene", "0.05", "--max-size", "1280", "--format", "webp", "--out", "o/"],
            ),
            ("export", ["--mp4", "a.mp4", "--crf", "23"]),
            (
                "export",
                ["--gif", "a.gif", "--max-colors", "128", "--dither", "bayer", "--overlay-clicks"],
            ),
            ("record", ["--display", ":99", "--resolution", "1920x1080x24", "--fps", "24", "--"]),
            ("capture", ["--region", "0,0,480,90", "--out", "banner.png"]),
            ("diff", ["--threshold", "0.1", "--out", "d.png"]),
            ("som", ["--boxes", "[[1,2,3,4]]", "--out", "som.png", "--resolve", "1"]),
            ("ocr", ["--region", "0,0,100,40", "--find", "ERROR"]),
        ],
        ids=["keyframes", "export-mp4", "export-gif", "record", "capture", "diff", "som", "ocr"],
    )
    def test_documented_flags_are_accepted(self, command, flags, tmp_path: Path):
        parser = build_parser()
        if command == "record":
            argv = [command, *flags]
        elif command == "capture":
            argv = [command, *flags]
        elif command == "diff":
            argv = [command, str(tmp_path / "before.png"), str(tmp_path / "after.png"), *flags]
        else:
            argv = [command, str(tmp_path / "shot.png"), *flags]
        args = parser.parse_args(argv)
        assert args.command_name == command

    @pytest.mark.parametrize("command", ["capture", "diff", "ocr", "som"])
    def test_inspection_commands_reject_unknown_flags(self, command, tmp_path: Path):
        with pytest.raises(SystemExit):
            build_parser().parse_args([command, str(tmp_path / "a.png"), "--nonsense"])


@requires_ffmpeg
class TestCaptureCommand:
    """`dynamo capture` is how a shell-only agent gets a static image."""

    @pytest.mark.xfvb
    @requires_linux
    @requires_xvfb
    def test_grabs_the_display(self, cli_env, tmp_path: Path, display):
        """`capture` does not start a server — it grabs one that is already up."""
        out = tmp_path / "full.png"
        proc = run_cli(cli_env, "capture", "--display", display.display,
                       "--width", "640", "--height", "480", "--out", str(out))
        assert proc.returncode == 0, proc.stderr
        assert out.exists() and out.stat().st_size > 0

    @pytest.mark.xfvb
    @requires_linux
    @requires_xvfb
    def test_region_snippet_is_smaller_than_the_full_frame(self, cli_env, tmp_path: Path,
                                                          display):
        full, small = tmp_path / "full.png", tmp_path / "small.png"
        assert run_cli(cli_env, "capture", "--display", display.display, "--width", "640",
                       "--height", "480", "--out", str(full)).returncode == 0
        assert run_cli(cli_env, "capture", "--display", display.display, "--width", "640",
                       "--height", "480", "--region", "40,40,160,60",
                       "--out", str(small)).returncode == 0
        assert small.stat().st_size < full.stat().st_size

    @pytest.mark.xfvb
    @requires_linux
    @requires_xvfb
    def test_reads_the_display_env_var(self, cli_env, tmp_path: Path, display):
        """With no --display, $DISPLAY is the target — the usual agent setup."""
        env = {**cli_env, "DISPLAY": display.display}
        proc = subprocess.run(
            [sys.executable, "-m", "agent_visual_harness", "capture",
             "--width", "640", "--height", "480", "--out", str(tmp_path / "env.png")],
            capture_output=True, text=True, env=env, timeout=300,
        )
        assert proc.returncode == 0, proc.stderr
        assert (tmp_path / "env.png").exists()

    def test_rejects_a_malformed_region(self, cli_env, tmp_path: Path):
        proc = run_cli(cli_env, "capture", "--region", "1,2,3", "--out",
                       str(tmp_path / "x.png"))
        assert proc.returncode != 0
        assert "X,Y,W,H" in proc.stderr


@requires_ffmpeg
class TestDiffCommand:
    """`dynamo diff` is a visual regression gate: exit 0 means the UI held."""

    def _pair(self, tmp_path: Path) -> tuple[Path, Path]:
        from agent_visual_harness._ffmpeg import FFMPEG, run_bytes

        def solid(colour: str) -> bytes:
            return run_bytes(
                [FFMPEG, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                 f"color=c={colour}:s=120x80", "-frames:v", "1", "-c:v", "png",
                 "-f", "image2pipe", "-"],
                timeout=60,
            )

        before, after = tmp_path / "before.png", tmp_path / "after.png"
        before.write_bytes(solid("#ffffff"))
        after.write_bytes(solid("#000000"))
        return before, after

    def _small_change(self, tmp_path: Path) -> tuple[Path, Path]:
        """Two 200x200 frames differing in a 40x40 corner — 4% of the pixels."""
        from agent_visual_harness._ffmpeg import FFMPEG, run_bytes

        def frame(colour: str, draw: str) -> bytes:
            return run_bytes(
                [FFMPEG, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                 f"color=c={colour}:s=200x200", "-vf", draw,
                 "-frames:v", "1", "-c:v", "png", "-f", "image2pipe", "-"],
                timeout=60,
            )

        box = "drawbox=x=0:y=0:w=40:h=40:color=black@1:t=fill"
        before, after = tmp_path / "b.png", tmp_path / "a.png"
        before.write_bytes(frame("#ffffff", "null"))
        after.write_bytes(frame("#ffffff", box))
        return before, after

    def test_matching_images_exit_zero(self, cli_env, tmp_path: Path):
        before, after = self._pair(tmp_path)
        after.write_bytes(before.read_bytes())
        proc = run_cli(cli_env, "diff", str(before), str(after))
        assert proc.returncode == 0, proc.stderr

    def test_changed_images_exit_one(self, cli_env, tmp_path: Path):
        before, after = self._pair(tmp_path)
        proc = run_cli(cli_env, "diff", str(before), str(after))
        assert proc.returncode == 1
        assert "changed" in proc.stderr

    def test_quiet_prints_only_the_verdict(self, cli_env, tmp_path: Path):
        before, after = self._pair(tmp_path)
        proc = run_cli(cli_env, "diff", str(before), str(after), "--quiet")
        assert proc.stdout.strip().count("\n") == 0
        assert "% of pixels changed" in proc.stdout

    def test_json_output_is_machine_readable(self, cli_env, tmp_path: Path):
        before, after = self._pair(tmp_path)
        payload = json.loads(run_cli(cli_env, "diff", str(before), str(after),
                                     "--json").stdout)
        assert payload["same"] is False
        assert payload["changed_pixels"] > 0

    def test_writes_the_diff_image(self, cli_env, tmp_path: Path):
        before, after = self._pair(tmp_path)
        out = tmp_path / "diff.png"
        run_cli(cli_env, "diff", str(before), str(after), "--out", str(out))
        assert out.exists() and out.stat().st_size > 0

    def test_min_changed_pct_is_the_tolerance_knob(self, cli_env, tmp_path: Path):
        """4% of pixels changed: over the 0.05% default, under a 5% budget."""
        before, after = self._small_change(tmp_path)
        assert run_cli(cli_env, "diff", str(before), str(after)).returncode == 1
        relaxed = run_cli(cli_env, "diff", str(before), str(after),
                          "--min-changed-pct", "5")
        assert relaxed.returncode == 0, relaxed.stderr


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract-ocr is not installed")
@requires_ffmpeg
class TestOCRCommand:
    def _shot(self, tmp_path: Path, text: str) -> Path:
        from agent_visual_harness._ffmpeg import FFMPEG, run_bytes

        path = tmp_path / "shot.png"
        path.write_bytes(run_bytes(
            [FFMPEG, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
             "color=c=white:s=640x160",
             "-vf", f"drawtext=text='{text.replace(':', chr(92) + ':')}':fontsize=32:"
                    "fontcolor=black:x=30:y=60",
             "-frames:v", "1", "-c:v", "png", "-f", "image2pipe", "-"],
            timeout=60,
        ))
        return path

    def test_prints_the_text_it_found(self, cli_env, tmp_path: Path):
        shot = self._shot(tmp_path, "ECONNREFUSED")
        proc = run_cli(cli_env, "ocr", str(shot))
        assert proc.returncode == 0, proc.stderr
        assert "ECONNREFUSED" in proc.stdout

    def test_find_reports_a_box(self, cli_env, tmp_path: Path):
        """`--find` is the cheapest useful question: is this error on screen, and where?"""
        shot = self._shot(tmp_path, "ECONNREFUSED")
        proc = run_cli(cli_env, "ocr", str(shot), "--find", "ECONNREFUSED")
        assert proc.returncode == 0, proc.stderr
        parts = proc.stdout.split()
        assert parts[0] == "ECONNREFUSED"
        assert [int(v) for v in parts[1:5]] == [int(v) for v in parts[1:5]]

    def test_find_exits_one_when_absent(self, cli_env, tmp_path: Path):
        shot = self._shot(tmp_path, "ECONNREFUSED")
        proc = run_cli(cli_env, "ocr", str(shot), "--find", "ZEBRA")
        assert proc.returncode == 1
        assert "not found" in proc.stderr

    def test_json_includes_confidence(self, cli_env, tmp_path: Path):
        payload = json.loads(run_cli(cli_env, "ocr", str(self._shot(tmp_path, "READY")),
                                     "--json").stdout)
        assert payload["confidence"] > 0
        assert payload["engine"] == "tesseract"


@pytest.mark.skipif(
    shutil.which("fc-match") is None, reason="fontconfig is not installed (drawtext)"
)
@requires_ffmpeg
class TestSoMCommand:
    def _shot(self, tmp_path: Path) -> Path:
        from agent_visual_harness._ffmpeg import FFMPEG, run_bytes

        path = tmp_path / "page.png"
        path.write_bytes(run_bytes(
            [FFMPEG, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
             "color=c=white:s=400x200", "-frames:v", "1", "-c:v", "png",
             "-f", "image2pipe", "-"],
            timeout=60,
        ))
        return path

    def test_annotates_and_resolves(self, cli_env, tmp_path: Path):
        """Number in, coordinates out — the whole grounding round trip."""
        out = tmp_path / "som.png"
        proc = run_cli(cli_env, "som", str(self._shot(tmp_path)), "--out", str(out),
                       "--boxes", "[[20,40,120,40],[200,40,120,40]]",
                       "--labels", "Pay,Cancel", "--resolve", "2")
        assert proc.returncode == 0, proc.stderr
        assert out.exists()
        assert "index 2 -> (260, 60)" in proc.stdout

    def test_writes_a_legend(self, cli_env, tmp_path: Path):
        legend = tmp_path / "legend.txt"
        run_cli(cli_env, "som", str(self._shot(tmp_path)),
                "--out", str(tmp_path / "som.png"),
                "--boxes", "[[20,40,120,40]]", "--labels", "Pay",
                "--roles", "button", "--legend", str(legend))
        text = legend.read_text(encoding="utf-8")
        assert "Pay" in text and "<button>" in text

    def test_resolving_an_absent_index_fails(self, cli_env, tmp_path: Path):
        proc = run_cli(cli_env, "som", str(self._shot(tmp_path)),
                       "--out", str(tmp_path / "som.png"),
                       "--boxes", "[[20,40,120,40]]", "--resolve", "9")
        assert proc.returncode == 1
        assert "not on this screen" in proc.stderr

    def test_empty_box_list_fails(self, cli_env, tmp_path: Path):
        proc = run_cli(cli_env, "som", str(self._shot(tmp_path)),
                       "--out", str(tmp_path / "som.png"), "--boxes", "[]")
        assert proc.returncode == 1
        assert "no boxes" in proc.stderr

    def test_fmt_remains_a_short_alias(self, tmp_path: Path):
        args = build_parser().parse_args(
            ["keyframes", str(tmp_path / "s"), "--out", "o", "--fmt", "jpeg"]
        )
        assert args.fmt == "jpeg"

    @pytest.mark.parametrize("command", ["keyframes", "export"])
    def test_unknown_flags_are_rejected(self, command, tmp_path: Path):
        with pytest.raises(SystemExit):
            build_parser().parse_args([command, str(tmp_path / "s"), "--nonsense"])


class TestTriageCommand:
    """`dynamo triage` is the shell's way to ask a model about a keyframe set.

    The command is testable without a model because the routing is the part that
    can be wrong — and because the one thing it must never do is quietly fall
    back to a stub and call the result a measurement.
    """

    @staticmethod
    def _keyframes(tmp_path: Path, count: int = 3) -> Path:
        """A tiny but real keyframe set: real PNGs, a real manifest, real diffs.

        Each frame is the same background with one small filled box that moves,
        so a crop has something small to crop to. Three frames of flat, entirely
        different colours would produce a diff region covering the whole frame —
        true, but useless for checking that cropping shrinks anything.
        """
        from agent_visual_harness._ffmpeg import FFMPEG

        root = tmp_path / "frames"
        root.mkdir(parents=True, exist_ok=True)
        entries = []
        for index in range(1, count + 1):
            name = f"frame_{index:04d}.png"
            box = f"drawbox=x={10 + index * 40}:y=20:w=60:h=20:color=0xff0000:t=fill"
            proc = subprocess.run(
                [FFMPEG, "-hide_banner", "-loglevel", "error",
                 "-f", "lavfi", "-i", "color=c=0x202020:s=320x200:d=1",
                 "-vf", box, "-frames:v", "1", "-c:v", "png", "-y", str(root / name)],
                capture_output=True, check=False,
            )
            assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
            entries.append({
                "index": index, "path": name, "timestamp": float(index),
                "reason": "action", "action": f"type: field {index}",
                "width": 320, "height": 200,
            })
        (root / "manifest.json").write_text(
            json.dumps({"version": 1, "keyframes": entries}), encoding="utf-8"
        )
        return root

    def test_missing_manifest_fails_cleanly(self, tmp_path: Path, capsys):
        empty = tmp_path / "empty"
        empty.mkdir()
        assert main(["triage", str(empty), "--question", "what?"]) == 1
        assert "manifest.json" in capsys.readouterr().err

    @requires_ffmpeg
    def test_stub_runs_the_pipeline_and_says_it_is_not_a_measurement(
        self, tmp_path: Path, capsys
    ):
        root = self._keyframes(tmp_path)
        assert main(["triage", str(root), "--question", "what went wrong?",
                     "--out", str(tmp_path / "report.json"), "--stub"]) == 0

        captured = capsys.readouterr()
        assert "none of this is a measurement" in captured.err
        report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
        assert report["frames"] == 3
        assert report["question"] == "what went wrong?"
        assert len(report["answers"]) == 3

    @requires_ffmpeg
    def test_frames_are_cropped_to_what_changed(self, tmp_path: Path, capsys):
        root = self._keyframes(tmp_path)
        assert main(["triage", str(root), "--question", "what went wrong?",
                     "--out", str(tmp_path / "r.json"), "--stub"]) == 0
        report = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
        # The box moved between frames, so two frames have a small region to crop
        # to and the first has none.
        assert report["cropped"] == 2
        assert report["tokens_after"] < report["tokens_before"]

    @requires_ffmpeg
    def test_without_a_model_it_refuses_rather_than_stubbing(
        self, tmp_path: Path, capsys, monkeypatch
    ):
        # The failure this guards against is the quiet one: no server, so a stub
        # answers, and the numbers get believed.
        root = self._keyframes(tmp_path)
        monkeypatch.setattr(
            "agent_visual_harness.vision.OllamaBackend.available", lambda self: False
        )
        code = main(["triage", str(root), "--question", "what?",
                     "--ollama", "http://127.0.0.1:1"])
        assert code == 1
        assert "no model answered" in capsys.readouterr().err

    @requires_ffmpeg
    def test_keep_all_and_no_filter_are_wired_through(self, tmp_path: Path, capsys):
        root = self._keyframes(tmp_path)
        assert main(["triage", str(root), "--question", "what?", "--stub",
                     "--no-filter", "--keep-all", "--no-crop",
                     "--out", str(tmp_path / "r.json")]) == 0
        report = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
        assert report["cropped"] == 0
        assert report["filter_calls"] == 0
        assert report["tokens_after"] == report["tokens_before"]

    def test_unknown_flags_are_rejected(self, tmp_path: Path):
        with pytest.raises(SystemExit):
            build_parser().parse_args(
                ["triage", str(tmp_path / "f"), "--question", "q", "--nonsense"]
            )


class TestConsoleScript:
    def test_entry_point_is_importable(self):
        from agent_visual_harness.cli import main as entry

        assert callable(entry)

    @pytest.mark.skipif(shutil.which("dynamo") is None, reason="dynamo not installed on PATH")
    def test_installed_console_script_runs(self):
        proc = subprocess.run(["dynamo", "--version"], capture_output=True, text=True)
        assert proc.returncode == 0
        assert "dynamo" in proc.stdout
