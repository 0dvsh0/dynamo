#!/usr/bin/env bash
# Independently verify the demo's artifacts with plain ffmpeg/ffprobe —
# nothing from the agent_visual_harness package is used here, so this is a
# genuine external check rather than the code grading its own homework.
set -u
OUT=/tmp/demo/output
SES=/tmp/demo/session
fail=0
ok()   { printf '  \033[92mPASS\033[0m  %s\n' "$1"; }
bad()  { printf '  \033[91mFAIL\033[0m  %s\n' "$1"; fail=1; }
check(){ if [ "$1" = "0" ]; then ok "$2"; else bad "$2"; fi }

echo "== files =="
ls -la "$OUT" "$SES" | sed 's/^/  /'

echo
echo "== repro.mp4 =="
ffprobe -v error -show_entries format=duration,size -show_entries stream=codec_name,width,height,nb_frames \
  -of default=noprint_wrappers=1 "$OUT/repro.mp4" | sed 's/^/  /'
ffmpeg -v error -i "$OUT/repro.mp4" -f null - && ok "decodes end-to-end without errors" || bad "decode errors"

echo
echo "== demo.gif =="
ffprobe -v error -show_entries stream=codec_name,width,height -of default=noprint_wrappers=1 "$OUT/demo.gif" | sed 's/^/  /'
head -c 6 "$OUT/demo.gif" | grep -q GIF && ok "GIF magic bytes" || bad "not a GIF"

echo
echo "== vlm_frames =="
echo "  files: $(ls "$OUT/vlm_frames" | grep -c '^frame_')"
echo "  scratch left behind: $(ls "$OUT/vlm_frames" | grep -c '^_' || true)"
[ "$(ls "$OUT/vlm_frames" | grep -c '^_')" = "0" ] && ok "no scratch files" || bad "scratch files leaked"
python3 - "$OUT/vlm_frames/manifest.json" <<'PY'
import json, sys, pathlib
m = json.load(open(sys.argv[1], encoding="utf-8"))
out = pathlib.Path(sys.argv[1]).parent
print(f"  scene_frames_detected={m['scene_frames_detected']} action_anchors={m['action_anchors']} kept={m['keyframes_kept']}")
print(f"  raw_frame_estimate={m['raw_frame_estimate']} reduction={m['reduction_pct']}%")
missing = [k["path"] for k in m["keyframes"] if not (out / k["path"]).exists()]
empty = [k["path"] for k in m["keyframes"] if (out / k["path"]).stat().st_size == 0]
print(f"  missing={missing} empty={empty}")
print(f"  first 3: {[(k['timestamp'], k['reason'], k['action'], k['width'], k['height']) for k in m['keyframes'][:3]]}")
sys.exit(1 if missing or empty else 0)
PY
check $? "every manifest entry maps to a non-empty file on disk"

echo
echo "== telemetry.jsonl =="
cat "$SES/telemetry.jsonl" | sed 's/^/  /'
python3 -c "
import json,sys
rows=[json.loads(l) for l in open('$SES/telemetry.jsonl') if l.strip()]
assert rows, 'no telemetry'
assert all({'timestamp','action'} <= set(r) for r in rows), rows
assert [r['timestamp'] for r in rows] == sorted(r['timestamp'] for r in rows), 'not chronological'
assert any('x' in r for r in rows), 'no click coordinates logged'
print(f'  {len(rows)} well-formed, chronological events with coordinates')
" && ok "telemetry is valid, ordered, and positioned" || bad "telemetry problems"

echo
echo "== click overlays burned into the video =="
python3 - <<'PY'
import json, pathlib, subprocess
rows = [json.loads(l) for l in open("/tmp/demo/session/telemetry.jsonl") if l.strip()]
first = next((r for r in rows if "x" in r), None)
assert first, "no positioned action to sample"
for name, label in (("repro.mp4", "overlay export"), ("raw.mp4", "raw capture")):
    p = pathlib.Path("/tmp/demo/output" if name == "repro.mp4" else "/tmp/demo/session") / name
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{first['timestamp']+0.05:.2f}", "-i", str(p),
         "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True).stdout
    hit = sum(1 for i in range(0, len(out) - 3, 3)
              if out[i] > 200 and out[i+1] > 230 and out[i+2] < 120)  # volt-lime ring
    print(f"  {label:<14} {name:<10} volt-lime pixels at t={first['timestamp']:.2f}s: {hit}")
PY

echo
[ "$fail" = "0" ] && echo "artifacts: VERIFIED" || echo "artifacts: PROBLEMS FOUND"
exit "$fail"
