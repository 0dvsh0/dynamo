"""Print the README's comparison table straight from the measured data.

Hand-copying numbers into prose is how a demo ends up quoting a run that no
longer exists. This reads demo/data/*.json and emits the rows verbatim, so the
README can be updated by pasting rather than by retyping.
"""

import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def load(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as fh:
        return json.load(fh)


results = load("results.json")
agent = load("agent.json")

cols = [("screenshots", "A · screenshots"), ("diy", "B · DIY ffmpeg"), ("dynamo", "C · Dynamo")]

print("| | " + " | ".join(label for _, label in cols) + " |")
print("| --- | " + " | ".join("---" for _ in cols) + " |")


def row(label, getter):
    cells = []
    for key, _ in cols:
        value = getter(results.get(key, {}))
        cells.append("—" if value is None else str(value))
    print(f"| {label} | " + " | ".join(cells) + " |")


def red(d, key):
    """The redundancy block, which is where the frame statistics live."""
    block = d.get("redundancy") or {}
    return block.get(key)


row("images to the model", lambda d: d.get("images"))
row("visual tokens", lambda d: f"{d.get('tokens', 0):,}")
row("bit-identical repeats", lambda d: red(d, "duplicates"))
row("consecutive frame pairs compared", lambda d: red(d, "pairs"))
row("mean frame-to-frame Δ", lambda d: red(d, "mean_delta"))
row("image payload", lambda d: f"{d.get('image_bytes', 0) / 1e6:.1f} MB")

print()
print("prose ratios (the sentences, not the table):")
shots, diy, dyn = results["screenshots"], results["diy"], results["dynamo"]


def ratio(a, b):
    return None if not b else a / b


t_ratio = ratio(diy.get("tokens", 0), shots.get("tokens", 0))
b_ratio = ratio(shots.get("image_bytes", 0), dyn.get("image_bytes", 1))
print(f"  DIY vs screenshots tokens   : {t_ratio:.2f}x" if t_ratio else "  DIY tokens ratio: n/a")
print(f"  screenshots vs dynamo payload: {b_ratio:.2f}x smaller" if b_ratio else "  payload ratio: n/a")

for name, block in (("screenshots", shots), ("diy", diy), ("dynamo", dyn)):
    r = block.get("redundancy") or {}
    print(
        f"  {name:<12} dups {r.get('duplicates')} of {r.get('pairs')} pairs, "
        f"mean delta {r.get('mean_delta')}"
    )
    if "gif_bytes" in block:
        print(
            f"  {name:<12} gif {block['gif_bytes'] / 1e6:.2f} MB, "
            f"video {block.get('video_bytes', 0) / 1e6:.2f} MB"
        )

diy_gif, dyn_gif = diy.get("gif_bytes"), dyn.get("gif_bytes")
if diy_gif and dyn_gif:
    lead = "DIY is larger by" if diy_gif > dyn_gif else "Dynamo is larger by"
    print(
        f"  GIF comparison: {diy_gif / 1e6:.2f} MB vs {dyn_gif / 1e6:.2f} MB -> {lead} "
        f"{abs(diy_gif - dyn_gif) / 1e6:.2f} MB ({abs(diy_gif / dyn_gif - 1) * 100:.0f}%)"
    )

print()
print("raw keys per workflow:")
for key, _ in cols:
    print(f"  {key}: {sorted(results.get(key, {}).keys())}")

print()
print("model layer:")
for name in ("whole", "cropped", "triaged"):
    w = agent["workflows"][name]
    print(
        f"  {name:<8} sent={w['model_calls']:<3} cropped={w['cropped']:<3} "
        f"est={w['tokens_after']:<7,} counted={w['prompt_tokens']:,} "
        f"nonempty={w['answered_nonempty']}"
    )
head = agent["headline"]
for k in (
    "tokens_whole", "tokens_cropped", "tokens_triaged", "triaged_vs_whole_pct",
    "counted_tokens_whole", "counted_tokens_cropped", "counted_tokens_triaged",
    "counted_triaged_vs_whole_pct",
):
    print(f"  headline.{k} = {head.get(k)}")
print(f"  frames = {agent['meta']['frames']}, anchors = {agent['meta']['action_anchors']}, "
      f"with regions = {agent['meta']['frames_with_regions']}")
print(f"  question = {agent['meta']['question']}")
