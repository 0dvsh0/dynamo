#!/usr/bin/env bash
# End-to-end CLI round trip using the installed `dynamo` console script,
# exactly as the documentation describes it.
set -u
W=/tmp/cli-check
rm -rf $W; mkdir -p $W
cd $W

cat > paint.sh <<'EOS'
#!/usr/bin/env bash
for c in "#101010" "#f0f0f0" "#1070d0" "#f08010" "#20b060"; do
  xsetroot -display "$DISPLAY" -solid "$c"
  sleep 0.5
done
EOS
chmod +x paint.sh

echo "== 1. dynamo record =="
dynamo record --display :141 --resolution 800x600x24 --fps 20 --session ./session \
  -- ./paint.sh
echo "  exit=$?"

echo
echo "== 2. dynamo info =="
dynamo info ./session

echo
echo "== 3. dynamo export (documented flags) =="
dynamo export ./session \
  --mp4 ./out/repro.mp4 --crf 23 \
  --gif ./out/demo.gif --max-colors 128 --dither bayer \
  --overlay-clicks --vlm-frames ./out/vlm_frames
echo "  exit=$?"

echo
echo "== 4. dynamo keyframes (documented flags) =="
dynamo keyframes ./session --scene 0.05 --max-size 1280 --format webp --out ./out/frames/ -v
echo "  exit=$?"

echo
echo "== 5. artifacts =="
find ./out ./session -type f | sort | while read -r f; do
  printf '  %8s  %s\n' "$(stat -c %s "$f")" "$f"
done

echo
echo "== 6. token reduction =="
python3 - <<'PY'
import json
m = json.load(open("./out/frames/manifest.json", encoding="utf-8"))
print(f"  raw frames (estimated) : {m['raw_frame_estimate']}")
print(f"  scene changes detected : {m['scene_frames_detected']}")
print(f"  keyframes kept         : {m['keyframes_kept']}")
print(f"  reduction              : {m['reduction_pct']}%")
PY
