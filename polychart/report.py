"""Offline paired analysis: reads the harvest + ledger, writes a status report.

No API calls, so it can run on a short cycle without competing with the runners for
the rate limiter. Conditions come from fleet.jsonl, whose entries are derived from the
launch requests/responses in queue.log - never inferred from steps_per_epoch, which
cannot distinguish 0 from 2k augmentation (21-22 vs 23) and previously mislabelled
three augmented arms as controls.

Usage: python -m polychart.report
"""

from __future__ import annotations

import json
import pathlib
import time
from math import comb

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNDIR = ROOT / "data" / "runs"
OUT = RUNDIR / "report.txt"


def _sign_p(pos: int, n: int) -> float:
    if n == 0:
        return 1.0
    k = max(pos, n - pos)
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)


def build():
    harvest = []
    hp = RUNDIR / "category_harvest.jsonl"
    if hp.exists():
        for line in hp.read_text().splitlines():
            try:
                harvest.append(json.loads(line))
            except Exception:
                pass
    led = {}
    lp = RUNDIR / "fleet.jsonl"
    if lp.exists():
        for line in lp.read_text().splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("job"):
                led[r["job"][:8]] = r

    arms, unmapped = {}, 0
    for r in harvest:
        if r.get("category") is None:
            continue
        m = led.get(r.get("short", ""))
        if not m:
            unmapped += 1
            continue
        arm = "control" if (m.get("domain", 0), m.get("general", 0)) == (0, 0) else "aug"
        dose = m.get("domain", 0) + m.get("general", 0)
        arms.setdefault(r["dataset"], {}).setdefault(arm, []).append((r["category"], dose))

    lines = ["AutoScientist augmentation study - {}".format(time.strftime("%Y-%m-%d %H:%M")),
             "=" * 68, ""]
    lines.append("Per-seed category win rates (our launches only):")
    for ds in sorted(arms):
        parts = []
        for k in sorted(arms[ds]):
            vals = sorted(v for v, _ in arms[ds][k])
            parts.append("{}: {}".format(k, vals))
        lines.append("  {:<32} {}".format(ds, " | ".join(parts)))

    pairs = []
    for ds, a in arms.items():
        if "control" in a and "aug" in a:
            c = sum(v for v, _ in a["control"]) / len(a["control"])
            g = sum(v for v, _ in a["aug"]) / len(a["aug"])
            pairs.append((ds, c, g, len(a["control"]), len(a["aug"])))

    lines += ["", "VERIFIED WITHIN-SEED PAIRS: {}".format(len(pairs))]
    for ds, c, g, nc, na in sorted(pairs):
        lines.append("  {:<32} control {:5.1f} (n={}) -> aug {:5.1f} (n={})  delta {:+5.1f}".format(
            ds, c, nc, g, na, g - c))
    if pairs:
        pos = sum(1 for _, c, g, _, _ in pairs if g > c)
        n = len(pairs)
        mean = sum(g - c for _, c, g, _, _ in pairs) / n
        lines += ["", "  {}/{} positive | mean delta {:+.1f} | sign test p={:.4f}".format(
            pos, n, mean, _sign_p(pos, n))]
        need = 6 - n
        if _sign_p(n, n) > 0.05 and need > 0:
            lines.append("  (n={} floor is p={:.3f}; {} more all-positive pairs reach p<0.05)".format(
                n, _sign_p(n, n), need))

    # incomplete seeds: what is missing to finish a pair
    lines += ["", "Seeds missing an arm:"]
    for ds, a in sorted(arms.items()):
        if "control" in a and "aug" in a:
            continue
        lines.append("  {:<32} has {} - needs {}".format(
            ds, "+".join(sorted(a)), "aug" if "control" in a else "control"))
    if unmapped:
        lines.append("")
        lines.append("NOTE: {} category readings could not be mapped to a launch record "
                     "(pre-automation jobs; excluded).".format(unmapped))

    total_cat = sum(1 for r in harvest if r.get("category") is not None)
    lines += ["", "Captured: {} category readings, {} ledger records, {} harvest rows".format(
        total_cat, len(led), len(harvest))]
    return "\n".join(lines)


def main():
    txt = build()
    RUNDIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
