"""Attribute orphaned jobs via training_experiment_id.

AutoScientist runs a RECIPE SEARCH: a single launch spawns several training jobs
grouped under one `training_experiment_id`, and the platform selects a best (hence
the `best_job_config` wrapper). That is why there have always been more jobs than
launches, and why many jobs never receive a category eval.

The consequence for attribution is decisive. A job's own augmentation setting is not
retrievable, but `GET /datasets/{ds}/finetune/jobs/{job}/config` returns the
experiment id, and every job in an experiment came from the SAME launch. So any
orphan that shares an experiment with a job whose condition we recorded at launch
time inherits that condition - by evidence, not by inference from a proxy such as
steps-per-epoch (which was disproven: 0-aug and 2k-aug both land at 21-23 steps).

This walks every dataset, maps job -> experiment, and writes attributions for
orphans whose experiment contains a ledger-known job. Conflicts (an experiment
containing two different recorded conditions) are reported and skipped.

Usage: python -m polychart.attribute [--write]
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

from . import launch as L

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNDIR = ROOT / "data" / "runs"
LEDGER = RUNDIR / "fleet.jsonl"
EXPMAP = RUNDIR / "experiment_map.json"


def _get(path, key, tries=5, wait=20):
    for _ in range(tries):
        st, d = L._req("GET", path, key=key)
        if st == 200:
            return d
        time.sleep(wait)
    return None


def main():
    write = "--write" in sys.argv
    key = L._auth("pt_live")

    ledger = {}
    if LEDGER.exists():
        for line in LEDGER.read_text().splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("job"):
                ledger[r["job"][:8]] = r

    harvest = {}
    hp = RUNDIR / "category_harvest.jsonl"
    if hp.exists():
        for line in hp.read_text().splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            harvest[r.get("short")] = r

    cached = {}
    if EXPMAP.exists():
        cached = json.loads(EXPMAP.read_text())

    d = _get("/datasets", key)
    if d is None:
        print("throttled")
        return
    rows = d.get("datasets", d) if isinstance(d, dict) else d

    # job -> experiment, per dataset
    exp_of, ds_of = dict(cached.get("exp_of", {})), dict(cached.get("ds_of", {}))
    for x in rows:
        did, name = x["dataset_id"], x.get("name", "?")
        time.sleep(2)
        j = _get("/datasets/{}/finetune/jobs".format(did), key, tries=3, wait=15)
        if j is None:
            continue
        for job in j.get("jobs", []):
            if job.get("status") != "succeeded":
                continue
            sid = job["finetune_job_id"][:8]
            ds_of[sid] = name
            if sid in exp_of:
                continue
            # only spend a call when this job could actually matter: it either has a
            # category reading to rescue, or it is a ledger anchor for its experiment
            if sid not in harvest and sid not in ledger:
                continue
            time.sleep(2)
            c = _get("/datasets/{}/finetune/jobs/{}/config".format(did, job["finetune_job_id"]),
                     key, tries=3, wait=15)
            e = (c or {}).get("best_job_config", {}).get("training_experiment_id")
            if e:
                exp_of[sid] = e
                print("MAP {} {} -> {}".format(name[:26], sid, e[:8]), flush=True)

    EXPMAP.write_text(json.dumps({"exp_of": exp_of, "ds_of": ds_of}, indent=1))

    # experiment -> the recorded condition(s) of its ledger-known members
    cond = {}
    for sid, e in exp_of.items():
        r = ledger.get(sid)
        if not r:
            continue
        cond.setdefault(e, set()).add((r.get("domain", 0), r.get("general", 0)))

    added, conflicts = [], []
    for sid, e in exp_of.items():
        if sid in ledger or sid not in harvest:
            continue
        c = cond.get(e)
        if not c:
            continue
        if len(c) > 1:
            conflicts.append((sid, e, sorted(c)))
            continue
        dom, gen = next(iter(c))
        added.append({"label": "exp-attributed-{}".format(sid), "dataset": ds_of.get(sid),
                      "domain": dom, "general": gen, "preset": "llama70b", "job": sid,
                      "status": "succeeded", "attribution": "training_experiment_id",
                      "experiment": e})

    print("\n=== EXPERIMENT-BASED ATTRIBUTION ===")
    print("jobs mapped to an experiment : {}".format(len(exp_of)))
    print("orphans now attributable     : {}".format(len(added)))
    for a in added:
        cat = harvest.get(a["job"], {}).get("category")
        print("  {:<30} {} d={} g={} cat={}".format(
            a["dataset"] or "?", a["job"], a["domain"], a["general"], cat))
    if conflicts:
        print("CONFLICTS (experiment holds two recorded conditions; skipped):")
        for sid, e, c in conflicts:
            print("  {} exp {} -> {}".format(sid, e[:8], c))
    if write and added:
        with LEDGER.open("a") as f:
            for a in added:
                a["at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                f.write(json.dumps(a) + "\n")
        print("\nwrote {} attributions to the ledger".format(len(added)))
    elif not write:
        print("\n(dry run; pass --write to append)")


if __name__ == "__main__":
    main()
