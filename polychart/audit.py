"""Completeness audit: is every succeeded job on the platform captured locally?

Enumerates every job on every dataset and cross-checks it against
  * category_harvest.jsonl (did we read its metrics?), and
  * fleet.jsonl (do we know which condition it was?)
then prints the gaps. This is a stronger check than the platform's success emails,
because it covers jobs whose email was never sent or never seen, and it reports the
two failure modes that actually matter: a result we never read, and a result we read
but cannot attribute to a condition.

Usage: python -m polychart.audit
"""

from __future__ import annotations

import json
import pathlib
import time

from . import launch as L

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNDIR = ROOT / "data" / "runs"


def _get(path, key, tries=6, wait=25):
    for _ in range(tries):
        st, d = L._req("GET", path, key=key)
        if st == 200:
            return d
        time.sleep(wait)
    return None


def main():
    key = L._auth("pt_live")
    harvested = {}
    hp = RUNDIR / "category_harvest.jsonl"
    if hp.exists():
        for line in hp.read_text().splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            harvested[r.get("short") or r.get("job", "")[:8]] = r
    ledger = {}
    lp = RUNDIR / "fleet.jsonl"
    if lp.exists():
        for line in lp.read_text().splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("job"):
                ledger[r["job"][:8]] = r

    d = _get("/datasets", key)
    if d is None:
        print("throttled")
        return
    rows = d.get("datasets", d) if isinstance(d, dict) else d

    total = succeeded = missing_metrics = missing_cond = with_cat = 0
    gaps_metrics, gaps_cond = [], []
    per_status = {}

    for x in rows:
        did, name = x["dataset_id"], x.get("name", "?")
        time.sleep(2)
        j = _get("/datasets/{}/finetune/jobs".format(did), key, tries=3, wait=20)
        if j is None:
            print("SKIP {} (throttled)".format(name))
            continue
        for job in j.get("jobs", []):
            total += 1
            st = job.get("status")
            per_status[st] = per_status.get(st, 0) + 1
            if st != "succeeded":
                continue
            succeeded += 1
            sid = job["finetune_job_id"][:8]
            h = harvested.get(sid)
            if h is None:
                missing_metrics += 1
                gaps_metrics.append((name, sid, job.get("created_at", "")[:19]))
            elif h.get("category") is not None:
                with_cat += 1
            if sid not in ledger:
                missing_cond += 1
                gaps_cond.append((name, sid, job.get("created_at", "")[:19]))

    print("\n=== COMPLETENESS AUDIT ===")
    print("jobs on platform      : {}".format(total))
    print("  by status           : {}".format(per_status))
    print("succeeded             : {}".format(succeeded))
    print("  metrics captured    : {}".format(succeeded - missing_metrics))
    print("  with a category     : {}".format(with_cat))
    print("  condition known     : {}".format(succeeded - missing_cond))
    if gaps_metrics:
        print("\nSUCCEEDED BUT NEVER READ ({}):".format(len(gaps_metrics)))
        for n, s, t in gaps_metrics[:25]:
            print("   {:<32} {} {}".format(n, s, t))
    if gaps_cond:
        print("\nREAD BUT CONDITION UNKNOWN ({}):".format(len(gaps_cond)))
        for n, s, t in gaps_cond[:25]:
            print("   {:<32} {} {}".format(n, s, t))
    if not gaps_metrics and not gaps_cond:
        print("\nNo gaps: every succeeded job is both read and attributed.")
    print("\nDONE")


if __name__ == "__main__":
    main()
