"""Re-extract the still gallery for the run that already happened.

`run_experiments.py` does this as part of a full run; this is the same code path
so the existing artifacts can be refreshed without re-recording 50 seconds of
browser five times over.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, "/mnt/d/coding/dynamo/demo")
sys.path.insert(0, "/mnt/d/coding/dynamo/src")

import run_experiments as rx  # noqa: E402

demo = Path("/mnt/d/coding/dynamo/demo")
assets = demo / "assets"
results_path = demo / "data" / "results.json"

stills = rx.extract_stills(assets / "dynamo_repro.mp4", assets / "telemetry.jsonl",
                           assets / "stills")
results = json.loads(results_path.read_text(encoding="utf-8"))
results["stills"] = stills
results["assets"] = sorted(p.name for p in assets.iterdir())
results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
print(f"extracted {len(stills)} stills")
for s in stills:
    print(f"  {s['index']:>2}  t={s['timestamp']:>6.2f}s  {s['action']}")
