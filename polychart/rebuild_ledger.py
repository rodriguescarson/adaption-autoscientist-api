"""Rebuild the launch ledger from queue.log, which is the authoritative record.

The queue runner prints `launching <label> (d=.. g=.. <preset>)` immediately before
its launch call, and launch() prints the raw `launch: 201 {json}` response containing
the finetune_job_id. So every successful launch is fully recoverable from the log,
with no API calls and no rate-limit exposure - unlike querying the platform, which is
what corrupted the ledger in the first place.

Pairs each `launching` line with the next `launch: 201` line, rewrites fleet.jsonl,
and reports any label that was attempted but never produced a 201 (those are the ones
still genuinely pending).

Usage: python -m polychart.rebuild_ledger [--write]
"""

from __future__ import annotations

import json
import pathlib
import re
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNDIR = ROOT / "data" / "runs"
LEDGER = RUNDIR / "fleet.jsonl"

LAUNCHING = re.compile(r"^\s*launching (\S+) \(d=(\d+) g=(\d+) (\S+)\)")
RESP = re.compile(r'launch: 201 (\{.*)')
# A stale-job return is ALSO a 201: the platform hands back the pre-existing job when
# the dataset is busy. queue_runner detects that and refuses to record it, but this
# parser used to record the label against that stale job id anyway - which both marked
# the condition as "already launched" (so it was never retried) and attributed someone
# else's job to our arm. Any launching/201 pair followed by this line is discarded.
STALE = re.compile(r'STALE job returned')


GAP = 35  # launch() prints ~20 lines (config/validate/cost) before the 201


def parse(log: pathlib.Path):
    """Walk the log; a `launching` line claims a `launch: 201` only if it appears
    within GAP lines. The adjacency requirement matters because two queue_runner
    instances were briefly appending to the same log, so lines interleave and a
    naive next-match rule assigns one job id to two different conditions."""
    out, pending, attempted = [], None, []
    if not log.exists():
        return out, attempted
    for lineno, line in enumerate(log.read_text(errors="replace").splitlines()):
        m = LAUNCHING.match(line)
        if m:
            if pending:
                attempted.append(pending[0])   # previous one never got a 201
            pending = (m.group(1), int(m.group(2)), int(m.group(3)), m.group(4), lineno)
            continue
        if STALE.search(line):
            # the 201 just recorded (if any) was a pre-existing job, not our launch
            if out and lineno - out[-1].get('_line', -999) <= GAP:
                out.pop()
            pending = None
            continue
        r = RESP.search(line)
        if r and pending and lineno - pending[4] > GAP:
            attempted.append(pending[0])       # too far away to be this launch's reply
            pending = None
        if r and pending:
            body = r.group(1)
            jid = None
            mm = re.search(r'"finetune_job_id":\s*"([0-9a-f-]+)"', body)
            if mm:
                jid = mm.group(1)
            ds = None
            dm = re.search(r'"dataset_id":\s*"([0-9a-f-]+)"', body)
            if dm:
                ds = dm.group(1)
            st = None
            sm = re.search(r'"status":\s*"(\w+)"', body)
            if sm:
                st = sm.group(1)
            if jid:
                out.append({"label": pending[0], "domain": pending[1],
                            "general": pending[2], "preset": pending[3],
                            "job": jid, "dataset_id": ds, "status": st,
                            "source": "queue.log", "_line": lineno})
            pending = None
    if pending:
        attempted.append(pending[0])
    return out, attempted


def main():
    write = "--write" in sys.argv
    recovered, never = parse(RUNDIR / "queue.log")
    byjob = {}
    for r in recovered:
        byjob.setdefault(r["job"], []).append(r)
    ambiguous = {j: v for j, v in byjob.items() if len({(x["label"], x["domain"]) for x in v}) > 1}
    recovered = [v[0] for j, v in byjob.items() if j not in ambiguous]
    print("recovered {} unambiguous launches from queue.log".format(len(recovered)))
    if ambiguous:
        print("  AMBIGUOUS (excluded, interleaved log): {}".format(
            {j[:8]: sorted({x["label"] for x in v}) for j, v in ambiguous.items()}))
    paid = [r for r in recovered if r["domain"] or r["general"]]
    print("  paid (augmented) arms: {} => {} credits".format(
        len(paid), sum((r["domain"] + r["general"]) // 100 for r in paid)))
    for r in recovered:
        print("   {:<34} {} d={}".format(r["label"], r["job"][:8], r["domain"]))
    if never:
        print("attempted but no 201 (still pending): {}".format(sorted(set(never))))

    if not write:
        print("\n(dry run; pass --write to merge into fleet.jsonl)")
        return

    existing = []
    if LEDGER.exists():
        for line in LEDGER.read_text().splitlines():
            if line.strip():
                try:
                    existing.append(json.loads(line))
                except Exception:
                    pass
    # Drop the placeholder rows written before real ids were known.
    existing = [r for r in existing if not r.get("id_is_prefix_only")]
    have = {r.get("job") for r in existing if r.get("job")}
    added = 0
    for r in recovered:
        if r["job"] in have:
            continue
        r["at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        existing.append(r)
        have.add(r["job"])
        added += 1
    LEDGER.write_text("\n".join(json.dumps(r) for r in existing) + "\n")
    print("\nledger rewritten: {} records ({} new)".format(len(existing), added))


if __name__ == "__main__":
    main()
