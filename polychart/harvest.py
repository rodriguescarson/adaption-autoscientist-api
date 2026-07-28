"""Re-poll jobs whose category eval is still in flight, until it settles.

`domain_eval_status` of `provisioning` / `running` means the held-out category eval
is still going and WILL produce a number; `None` means it was never attempted and
never will. So a single read of a freshly-finished job systematically undercounts
category coverage - the number shows up minutes to hours later.

This walks every dataset, finds succeeded jobs with a non-terminal eval status, and
re-polls them on a slow loop, appending each resolved number to category_harvest.jsonl.

Usage: python -m polychart.harvest [max_minutes]
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

from . import launch as L

RUNDIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "runs"
OUT = RUNDIR / "category_harvest.jsonl"
# "eval_failed" is a TERMINAL eval outcome, never a pending one.
PENDING = {"provisioning", "running", "pending", "queued", "in_progress"}


def _recorded():
    if not OUT.exists():
        return set()
    out = set()
    for line in OUT.read_text().splitlines():
        try:
            out.add(json.loads(line)["job"])
        except Exception:
            pass
    return out


def main():
    max_min = float(sys.argv[1]) if len(sys.argv) > 1 else 50.0
    deadline = time.time() + max_min * 60
    key = L._auth("pt_live")
    have = _recorded()

    while time.time() < deadline:
        st, d = L._req("GET", "/datasets", key=key)
        if st != 200:
            time.sleep(120)
            continue
        rows = (d.get("datasets", d) if isinstance(d, dict) else d) or []
        still_pending = 0
        for x in rows:
            if time.time() >= deadline:
                break
            did, name = x["dataset_id"], x.get("name", "?")
            time.sleep(3)
            s2, j = L._req("GET", "/datasets/{}/finetune/jobs".format(did), key=key)
            if s2 != 200:
                continue
            for job in j.get("jobs", []):
                if job.get("status") != "succeeded":
                    continue
                jid = job["finetune_job_id"]
                if jid in have:
                    continue
                time.sleep(3)
                m = L._metrics(did, jid, key)
                status = m.get("domain_eval_status")
                cat = (m.get("domain_eval") or {}).get(
                    "win_rates", {}).get("finetuned_model_win_rate_pct")
                if status in PENDING:
                    still_pending += 1
                    continue
                # steps_per_epoch fingerprints the augmentation level (the API never
                # exposes it directly): ~21-22 = no augmentation, ~27 = 2k combined,
                # ~49-52 = 8k. Captured here because we already hold the job record,
                # which avoids a second rate-limited pass just to classify old runs.
                te = job.get("training_events") or []
                spe = None
                if te and te[0].get("epoch"):
                    try:
                        spe = round(1 / te[0]["epoch"])
                    except Exception:
                        spe = None
                RUNDIR.mkdir(parents=True, exist_ok=True)
                OUT.open("a").write(json.dumps({
                    "job": jid, "short": jid[:8], "dataset": name,
                    "steps_per_epoch": spe,
                    "aug_guess": (None if spe is None else
                                  ("none" if spe <= 23 else
                                   ("~2k" if spe <= 35 else "~8k+"))),
                    "on_data": m.get("win_rates", {}).get("finetuned_model_win_rate_pct"),
                    "category": cat, "status": status,
                    "at": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
                have.add(jid)
                print("HARVEST {} {} spe={} cat={} ({})".format(
                    name, jid[:8], spe, cat, status), flush=True)
        print("  pass done; {} evals still in flight".format(still_pending), flush=True)
        # Do NOT exit when nothing is pending: jobs are still training and will enter
        # the eval queue later, and a rate-limited pass can undercount to zero. Just
        # idle and look again; the time budget ends the process.
        time.sleep(600 if still_pending else 900)


if __name__ == "__main__":
    main()
