# Dynamo — the AI Visual & Video Harness

`agent_visual_harness` is a headless, production-ready visual & video harness for
AI coding agents, browser automation and computer-use sandboxes. It owns the
virtual display, records every session, and compiles it into **VLM-ready
keyframes** and **PR-ready MP4/GIF artifacts** — in one context manager.

Recording gets an agent a *replay*. Four more primitives get it an *answer*:
what error the page showed, which pixels moved, where the button is, and which
coordinate space those pixels are in. Those four are arithmetic — no model
involved, which is why they are fast, reproducible and free.

For the questions arithmetic cannot answer — *what did the decline message say?* —
a fifth layer routes **one frame at a time** to an open-weight model: cropped to
what changed, filtered first, and only when the frame earns the call.

All of it ships in this package with **no runtime dependencies beyond `ffmpeg`**.
The package imports, tests and runs with no model, no server and no network;
`tesseract` is needed only for OCR, and a model server only for the triage layer.

```python
from agent_visual_harness import VisualHarness

with VisualHarness(display=":99", resolution=(1920, 1080), record=True) as harness:
    harness.run_on_display(["python", "run_agent.py"])
    harness.log_action("click", x=500, y=300)
    frame = harness.take_screenshot()          # PNG bytes, mid-run

artifacts = harness.export(
    video_path="output/repro.mp4",
    gif_path="output/demo.gif",
    vlm_keyframes_dir="output/vlm_frames/",
)
print(f"Generated {len(artifacts.keyframes)} frames for VLM context.")
```

Same capabilities from the CLI, for shell-driven agents and CI:

```bash
dynamo record --display :99 --resolution 1920x1080x24 --fps 24 -- python run_agent.py
dynamo export ./dynamo-session --mp4 output/repro.mp4 --crf 23 \
    --gif output/demo.gif --max-colors 128 --dither bayer --overlay-clicks
dynamo keyframes ./dynamo-session --scene 0.05 --max-size 1280 \
    --format webp --out output/vlm_frames/
```

---

## Inspecting a live screen

Everything below works against a running display — no recording required — and
exists because **no operating system ships a headless version of it**.

```bash
# one region, not the whole screen: crops in ffmpeg, before encoding
dynamo capture --region 0,0,480,90 --out banner.png

# text off pixels — for canvas, VNC, RDP and BIOS surfaces that give up nothing
dynamo ocr banner.png --find DO_NOT_HONOR --fuzzy   # exits 1 if absent
dynamo ocr --display :99 --find ECONNREFUSED

# visual regression gate — exits 1 when the UI moved
dynamo diff before.png after.png --out diff.png

# number the elements so a model answers "12" instead of guessing "(740, 318)"
dynamo som shot.png --boxes boxes.json --labels Pay,Cancel --out som.png --resolve 12
```

The same four from Python, plus the coordinate layer:

```python
from agent_visual_harness import Scale, compare, read_region, resolve

# text off pixels, with word boxes in screen coordinates
banner = read_region(harness.take_screenshot(), (0, 294, 1360, 470))
if banner.contains("DO_NOT_HONOR", fuzzy=True):
    left, top, w, h = banner.box_of("DO_NOT_HONOR", fuzzy=True)
    harness.click(left + w // 2, top + h // 2)      # read it, then point at it

# before vs after, with the changed regions and a reviewable overlay
d = compare(yesterday, harness.take_screenshot())
if not d.same:
    d.save_diff("artifacts/diff.png")
    for region in d.regions:
        print(region.box, region.changed_pixels)

# index -> coordinates. resolve() returns *pointer* space; Target.box is
# *image* space. On a HiDPI display or under browser chrome they differ.
marks = harness.ground(boxes, scale=Scale(ratio=device_pixel_ratio))
click_x, click_y = resolve(marks, model_answer)
```

### What each one is honest about

| Primitive | Guarantee | Measured limit |
| --- | --- | --- |
| `capture` | Crops before encoding, so a snip costs KBs not MBs | Flat per-image token pricing gives a snip **no** discount; sized pricing shows ~20× |
| `ocr` | Word-level boxes, so you can point back at what you read | Tesseract reads `DO_NOT_HONOR` as `D0_ NOT HONOR` and `7f3c9ab1` as `7f3c9abl`; no tesseract knob fixes it, so `fuzzy=True` matches through the confusable set while `text` stays verbatim |
| `diff` | Regions, a threshold you choose, and a red-on-dimmed overlay to review | Reads glyph-level slivers (letter-spacing shifts) as regions, not as noise; a control comparison must report zero |
| `som` | Numbered marks, and a wrong answer is a wrong *index* | Refuses zero-area boxes (clipped elements) rather than drawing an unclickable mark; element boxes come from the DOM/CDP, never from OCR |
| `Scale` | One place where logical↔device conversion happens | None — but skipping it is the silent failure: a click lands on empty page and the agent reports success |

---

## Routing a frame to a model

Everything above answers questions with arithmetic, and that is the point. But
*"what did the decline message say?"* is not arithmetic. It needs a model, and
that is where an agent quietly becomes expensive — the obvious implementation
sends the whole screen, every frame, every time.

`Triage` decides, per frame, how little work can still answer the question.
Three levers, none of which is model quality:

1. **No model.** A frame with no changed regions and no logged action carries no
   new information, so it is answered by omission.
2. **A cheap model.** An optional decision backend (Laya) scores the survivors and
   the ones below your threshold are dropped — with the probability that dropped
   them kept on the answer rather than discarded.
3. **The generative model.** Only now is a VLM asked, and it is asked about a
   crop rather than a whole screen whenever the diff can point at one.

Measured against the same 29 keyframes the reducer produced
(`demo/run_agent_bench.py` → `demo/data/agent.json`):

| How the frames reach the model | frames sent | cropped first | visual tokens | of sending all |
| --- | --- | --- | --- | --- |
| A · every keyframe, whole | 29 | 0 | 22,707 | 100% |
| B · cropped to what the diff says moved | 29 | 20 | **11,008** | **48.5%** |
| C · triaged: skip unchanged → filter → crop | 29 | 20 | **11,008** | **48.5%** |

And on one frame: the whole keyframe is 1280×800 → **783 tokens**. The region the
diff reported for that same frame is 578×296 → **131 tokens**, or **6.0× less**.
The model is not asked to ignore the rest of the screen; it is never shown it.

**Being honest about the third column: rows A and B are measurements, row C is
not.** On this keyframe set the skip and filter steps removed *nothing* — because
the reducer had already dropped the redundant frames, and the frames that remain
are action anchors, which are never skipped on purpose. That saving bites on a raw
frame stream, not on an already-reduced one. A stub filter also passed every
frame, so row C's `frames sent` shows what the pipeline *does*, not what a model
*would* decide. The bench **refuses to write the file** without `--stub` unless a
real model answered, and the page states which of the two it is.

```python
from agent_visual_harness import LayaBackend, OllamaBackend, Triage

triage = Triage(
    OllamaBackend("qwen3-vl:4b"),   # open weights (Apache-2.0), local
    filter=LayaBackend(),           # typed decisions, no generation
    crop=True,                      # cut to the diff region before encoding
)
report = triage.run(frames, "what did the user see go wrong?")
print(report.summary())             # includes what it declined to send
```

The report is the argument: `calls_avoided`, `cropped`, `filter_calls`,
`tokens_before` → `tokens_after`, and a per-frame `skipped_reason`. `RemoteBackend`
exists for escalation and is **never constructed implicitly**, so nothing leaves
the machine unless a caller says so — and every frame that does is counted.

**Both halves are Apache-2.0, and that was checked, not assumed.** Qwen3-VL-4B is
Apache-2.0, and Laya's official checkpoints (`laya`, `laya-multilingual`,
`laya-typed-decisions`) are Apache-2.0 as well — so the whole path is clean for a
commercial build. It is also why the primary is Qwen3-VL and not Qwen2.5-VL: the
2.5 3B checkpoints are not Apache-2.0, and a licence is not a detail you want to
discover after shipping. (Community *vision* forks of Laya are not uniformly
licensed — one on the Hub is CC BY-NC-SA — which is a second, practical reason the
filter is handed text rather than pixels: the text checkpoints are the officially
licensed ones.)

Two things this layer does not claim:

- **It does not beat a frontier model at reading an image.** It does not have to.
  A frontier model may well read a hard screenshot better. The win is custody
  (the frame never leaves the machine, which matters when it shows someone's card
  details), determinism at temperature 0, the budget to ask 29 times instead of
  twice, and attribution back to a click at a timestamp — which no screenshot can
  supply however well it is read.
- **The filter is a text model, and an image is not text.** Laya decides over the
  harness's own observation — the action, the timestamp, how much of the screen
  moved, and whatever text OCR read off the pixels. Without OCR it is judging the
  action label alone, which is weaker, and `Triage.state_for(frame)` prints exactly
  what was handed over so you can see which of the two you got.

---

## Why it exists

| Problem | What Dynamo does about it |
| --- | --- |
| A 60-second session at 30 fps is 1,800 frames; feeding all of them to a VLM costs real money to learn nothing | Keeps only scene changes and the exact frames at every logged action |
| DIY ffmpeg scripts race Xvfb startup, lose `moov` atoms to `SIGKILL`, emit 40 MB GIFs | Owns the Xvfb lifecycle with a readiness probe; finalises with `SIGINT` on **every** exit path, including exceptions; two-pass palette GIFs |
| A PNG cannot tell you which click caused the regression | Timestamped `telemetry.jsonl` plus click rings and crosshairs burned into the video |

## Install

```bash
pip install agent-visual-harness
```

Zero runtime dependencies — standard library only. The harness shells out to
`ffmpeg`/`ffprobe` and `Xvfb`, which you install at the OS level:

```bash
apt-get update && apt-get install -y xvfb ffmpeg
```

### Docker

```dockerfile
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
      xvfb ffmpeg x11-utils && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir agent-visual-harness

# Dynamo manages Xvfb itself — no DISPLAY export needed
ENTRYPOINT ["dynamo", "record", "--", "python", "agent.py"]
```

### GitHub Actions

```yaml
name: agent-e2e
on: [pull_request]
jobs:
  visual-run:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: sudo apt-get install -y xvfb ffmpeg tesseract-ocr fonts-dejavu-core
      - run: pip install agent-visual-harness
      - run: dynamo record -- pytest tests/e2e
      - run: dynamo export ./dynamo-session --gif pr/demo.gif --overlay-clicks
      # Visual regression gate: exits 1 when a view moved.
      - run: dynamo diff baseline/checkout.png pr/checkout.png --out pr/diff.png
      - uses: actions/upload-artifact@v4
        with: { name: agent-recording, path: pr/ }
```

## The modules

Each is independently usable, fully typed, and usable on its own.

| Module | Responsibility | Key guarantee |
| --- | --- | --- |
| `display.py` | Xvfb lifecycle, configurable display number / resolution / DPI | Socket probe blocks until X accepts connections; Dynamo never kills a display it does not own |
| `recorder.py` | Non-blocking `ffmpeg -f x11grab` daemon + JSONL telemetry | `SIGINT` on every exit path finalises the `moov` atom; a repair pass catches hard kills |
| `processor.py` | CRF-tuned H.264 / VP9, two-pass GIF, click overlays | `--movflags +faststart`; bayer-dithered palettes small enough for a PR |
| `token_reducer.py` | Scene detection + action anchors, downsampled and deduplicated | One `manifest.json` describes every frame's time, reason and action |
| `harness.py` | The `VisualHarness` context manager that wires it all together | A crashing agent still leaves a playable MP4 and complete telemetry |
| `coordinates.py` | Logical↔device conversion, box helpers, IoU | The single place a scale factor is applied; a silent DPI miss cannot happen elsewhere |
| `capture.py` | Full-surface grabs and region snips, header-only image sizing | Cropping happens in ffmpeg *before* encoding, so a snip costs KBs |
| `ocr.py` | Tesseract TSV parsing into words, lines and boxes | Text stays verbatim; only *matching* folds the glyphs Tesseract confuses |
| `diff.py` | Pixel diff, changed-region grouping, reviewable overlay | Regions survive per-glyph slivers, and the tolerance that tripped is named in the verdict |
| `grounding.py` | Set-of-Marks overlays and index→coordinate resolution | Zero-area boxes are refused, not drawn; an unreadable answer raises rather than defaulting |
| `vision.py` | Decides per frame whether a model is needed, then which: skip, filter, crop, escalate | A failed crop falls back to the whole frame rather than losing it; a missing model server raises instead of silently answering |

## Session layout

`export()` reads only from the session directory, which is what makes
`dynamo record` on one machine and `dynamo export` on another a valid split:

```
dynamo-session/
├── raw.mp4           high-quality intermediate capture (x11grab -> H.264 CRF 16)
├── telemetry.jsonl   one JSON object per logged action
├── meta.json         display, geometry, fps, pid, timestamps
└── display.log       Xvfb stderr, when Dynamo managed the display
```

Each telemetry line:

```json
{"timestamp": 1.45, "action": "click", "x": 420, "y": 680}
```

## Artifacts

| Output | What it is |
| --- | --- |
| `repro.mp4` | CRF 23 H.264 with `+faststart`; monospace text stays legible |
| `demo.gif` | Two-pass `palettegen → paletteuse`, bayer dithering, optional width cap |
| `vlm_frames/` | `frame_0001.webp` … plus `manifest.json` with timing and action labels |
| `telemetry.jsonl` | The action timeline that makes visual state attributable |

## Design notes

**Zero dependencies.** Everything is `subprocess` plus `json`, `socket` and
`dataclasses`. That keeps the Docker image small and means the harness cannot be
broken by a resolver conflict in an agent's own venv.

**Failure semantics are the feature.** An agent that raises at step 3 still gets
a playable MP4, because the context manager finalises in `__exit__` before letting
the exception propagate — and teardown errors never mask the original error.

**Telemetry can never crash the run.** `log_action()` swallows its own I/O
errors. Observability must not be able to take down the thing it observes.

**No data egress.** Recordings, frames and telemetry are written to paths you
name. Nothing is uploaded, and there is no telemetry of the telemetry.

**CPU-only by default.** The harness — capture, record, diff, OCR, export — never
touches a GPU, and neither does the triage plumbing. The optional model backend is
whatever you point it at: `qwen3-vl:4b` runs on a CPU (slowly) or on a GPU, and
nothing in the default path constructs a model, opens a socket or needs a network.

## The demo

`demo/` is a measured comparison, not a mock-up. A **headless-agent script drives a
real Chromium** (Playwright, headed, inside the harness display) through a **real HTTP
storefront** (`demo/webapp`) — filling in a checkout form, tripping the postal-code
validator, getting the card declined twice, retrying, and completing the order. The same
checkout is recorded three different ways.

```bash
# regenerate everything (needs Xvfb, ffmpeg, Playwright; ~8 min)
python3 demo/serve.py &                       # the storefront
python3 demo/run_experiments.py --out demo    # record + measure
python3 demo/run_inspection.py --out demo     # the four primitives, measured
python3 demo/run_agent_bench.py --out demo    # the model-facing layer
python3 demo/build_page.py                    # render index.html from the results
python3 demo/check_page.py                    # assert the page matches the data
```

`run_agent_bench.py` is the odd one out: it runs off the keyframes already on
disk, so it needs `ffmpeg` and a model server but **no Xvfb and no browser**. It
also refuses to write `data/agent.json` unless a real model answered, so the page
cannot end up showing a stub's numbers as a measurement. Add `--stub` to exercise
the pipeline without a model; the page then says so, in those words.

Then open `demo/index.html`.

The page is **generated** from `index.template.html` + `data/results.json`, and
`check_page.py` verifies that every asset resolves, every printed number matches the
measurement, every duplicate flag matches the measured deltas, and the honesty claims
still hold. A demo showing the wrong number is worse than no demo.

### What it measured

| | A · screenshots | B · DIY ffmpeg | C · Dynamo |
| --- | --- | --- | --- |
| images to the model | 24 | 51 | 29 |
| visual tokens | 26,400 | 56,100 | 31,900 |
| bit-identical repeats | 1 | 21 | 1 |
| mean frame-to-frame Δ | 3.29 | 1.75 | **11.36** |
| frame payload | 2.4 MB | 9.5 MB | **919 KB** |
| survives a torn-down CI job | — | `moov atom not found` | **yes** |

The honest reading, which the page states plainly:

- **Screenshots and Dynamo cost about the same in tokens**, because both sample on
  actions rather than wall-clock time. Dynamo's win there is not the token count.
- **Time-driven sampling (DIY) is the real token disaster** — 1.8× the tokens and 21 of
  50 consecutive frame pairs are pixel-identical.
- **Dynamo's win over screenshots is everything screenshots cannot do**: the video, the
  GIF, the click overlays, the timing manifest, a crash-safe recording, and a frame
  payload 2.6× smaller because it ships WebP rather than PNG.
- **The GIF byte comparison is close to a wash on this content** (1.90 MB vs 1.99 MB at
  identical framerate and resolution). The two-pass palette earns its keep on flat and
  dark UI; on bytes alone, with a dense browser UI, it does not.

The model-facing layer is priced separately, above — and where a row of that table
is *not* a measurement, it says so in as many words.

## Development

```bash
pip install -e ".[dev]"

# Linux with Xvfb — the whole suite, including the X11 and OCR tests
apt-get install -y xvfb ffmpeg x11-xserver-utils tesseract-ocr
bash scripts/run_tests.sh -q

# Everywhere else — X11- and OCR-dependent tests skip with a reason.
# 406 collected; on Windows that is 260 passed, 146 skipped, 0 failed.
pytest -q
```

Helper scripts:

| Script | What it does |
| --- | --- |
| `scripts/preflight.sh` | Checks a host for ffmpeg/Xvfb/xsetroot and the `/tmp/.X11-unix` mode |
| `scripts/wsl_setup.sh` | Repairs the WSLg read-only X11 mount that silently breaks Xvfb |
| `scripts/run_tests.sh` | Preflight + full suite, repair included |
| `scripts/demo_agent.py` | Drives a real GUI (tkinter) or falls back to `xsetroot`, then exports |
| `scripts/verify_artifacts.sh` | Re-checks exported artifacts with plain ffmpeg/ffprobe, no package code |
| `scripts/cli_check.sh` | Full `dynamo record → info → export → keyframes` round trip |

> **WSL note.** WSLg mounts `/tmp/.X11-unix` read-only. X cannot bind its unix
> listener there, and when it refuses it keeps running while accepting nothing,
> so every readiness probe times out for no visible reason. `scripts/wsl_setup.sh`
> detects this and anything like it; the only unfixable case is a read-only mount
> that refuses to unmount, which needs `wsl --shutdown` from Windows.

The suite is layered: pure-logic tests (parsing, dedup, sizing, CLI wiring) run
anywhere, and only the capture tests require a live X server.

### Three things the media layer will teach you the hard way

All three are covered by tests, and all three are the kind of bug that only
appears when you run against a real X server and a real encoder:

- **ffmpeg buffers a static screen.** On a display where nothing changes, the
  encoded output is so small that ffmpeg holds it internally and the file stays
  at the 48-byte container header for the entire session. The recorder passes
  `-flush_packets 1` so progress is real, visible, and survives a hard kill.
- **ffmpeg's scene metric is luma-weighted.** A hard cut between two saturated
  colours of similar luma — `red` to `green`, for instance — scores 0 and is
  never detected, at any resolution. That is why `log_action()` anchors exist:
  they are the frames you can rely on, and the detector only finds the rest.
- **A missing `moov` atom is not repairable.** MP4 has no index to rebuild from,
  so a hard kill can be *prevented* (`SIGINT` on every exit path) but not
  *undone*. `repair_media()` says so honestly and never destroys the original.

## Licence

MIT
