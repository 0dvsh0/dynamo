#!/usr/bin/env bash
# Verify the commands documented in README.md actually work.
set -u
export PYTHONPATH=/opt/dynamo/src
WORK=/tmp/readme-check
rm -rf "$WORK"; mkdir -p "$WORK"; cd "$WORK"

python3 - <<'PY'
import sys
sys.path.insert(0, '/opt/dynamo/src')
from agent_visual_harness._ffmpeg import FFMPEG, run_bytes

def solid(colour, w, h):
    return run_bytes(
        [FFMPEG, '-hide_banner', '-loglevel', 'error', '-f', 'lavfi', '-i',
         f'color=c={colour}:s={w}x{h}', '-frames:v', '1', '-c:v', 'png',
         '-f', 'image2pipe', '-'],
        timeout=60,
    )

open('before.png', 'wb').write(solid('#ffffff', 300, 160))
# A real contrast, not a near-white: #ffffff vs #eeeeee differs by only 0.067 in
# luma, which is correctly *below* the default 0.10 threshold and reports "same".
open('after.png', 'wb').write(solid('#c0c0c0', 300, 160))
open('subtle.png', 'wb').write(solid('#eeeeee', 300, 160))
PY

run() { printf '%-46s' "$1"; shift; "$@" >/dev/null 2>&1; echo "exit=$?"; }

echo "== README CLI examples =="
run "diff, identical (expect 0)"    python3 -m agent_visual_harness diff before.png before.png
run "diff, changed (expect 1)"      python3 -m agent_visual_harness diff before.png after.png
run "diff, below threshold (0)"     python3 -m agent_visual_harness diff before.png subtle.png
run "diff, below threshold, strict" python3 -m agent_visual_harness diff before.png subtle.png --threshold 0.01
run "diff --out writes a file"      python3 -m agent_visual_harness diff before.png after.png --out diff.png
run "som --resolve"                 python3 -m agent_visual_harness som before.png --boxes '[[20,40,120,40]]' --labels Pay --out som.png --resolve 1
run "ocr --find absent (expect 1)"  python3 -m agent_visual_harness ocr before.png --find ZEBRA
run "ocr --find absent --fuzzy"     python3 -m agent_visual_harness ocr before.png --find ZEBRA --fuzzy

echo
echo "== README Python examples =="
python3 - <<'PY'
import sys
sys.path.insert(0, '/opt/dynamo/src')
from agent_visual_harness import Scale, annotate_boxes, compare, read_region, resolve

root = '/tmp/readme-check'
yesterday = open(f'{root}/before.png', 'rb').read()
today = open(f'{root}/after.png', 'rb').read()

d = compare(yesterday, today)
d.save_diff(f'{root}/py_diff.png')
print('compare        :', d.verdict())
print('  regions      :', [r.box for r in d.regions])
print('  largest      :', (d.to_dict()['largest_region'] or {}).get('box'))

banner = read_region(today, (0, 0, 300, 160))
print('read_region    :', repr(banner.text.strip()), 'conf', round(banner.confidence))
print('  box_of miss  :', banner.box_of('DO_NOT_HONOR', fuzzy=True))

marks = annotate_boxes(
    today, [[20, 40, 120, 40]], labels=['Pay'], roles=['button'],
    scale=Scale(ratio=2.0),
)
print('ground/resolve :', resolve(marks, 1), '-> pointer space')
print('  target box   :', marks.targets[0].box, '-> image space')
print('  skipped      :', marks.skipped_count)
PY

echo
echo "== produced =="
ls -la "$WORK" | awk 'NR>3 {printf "  %8s  %s\n", $5, $9}'