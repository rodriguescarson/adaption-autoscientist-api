"""Drive the six new category entries from processed dataset to published entry.

One pass does whatever each category is currently ready for, then exits, so it is
safe to call repeatedly from a loop:

    awaiting_input  -> POST /run with the column mapping (starts Adaptive Data)
    processing      -> nothing, wait
    succeeded       -> launch a control arm, then a +2k augmented arm
    has both arms   -> publish the dataset to HuggingFace and Kaggle

Credit discipline: the control arm is free (only augmentation is billed), and the
augmented arm costs ~20. The guard skips a paid launch when the balance cannot
cover it, so the free work keeps draining even at zero balance.

Usage: python -m polychart.category_pipeline [--status]
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

from . import launch as L
from .export_ds import publish

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNDIR = ROOT / "data" / "runs"
REC = RUNDIR / "category_datasets.json"
LEDGER = RUNDIR / "fleet.jsonl"
STATE = RUNDIR / "category_state.json"

CATEGORIES = ["market-analysis-news", "science", "agriculture",
              "hr", "personal-finance", "math-and-code"]
COLS = {"prompt": "prompt", "completion": "response"}
TERMINAL = {"succeeded", "failed", "cancelled", "canceled", "error", "eval_failed"}

# Set to 0 once the credit top-up lands to resume paid launches.
CREDIT_HOLD_FLOOR = 140


def _state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def _save(st: dict):
    STATE.write_text(json.dumps(st, indent=1))


def _ledger_labels() -> set:
    out = set()
    if LEDGER.exists():
        for line in LEDGER.read_text().splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("label"):
                out.add(r["label"])
    return out


def _record(label, ds_name, ds_id, dom, job, status):
    with LEDGER.open("a") as f:
        f.write(json.dumps({"label": label, "dataset": ds_name, "dataset_id": ds_id,
                            "domain": dom, "general": 0, "preset": "llama70b",
                            "job": job, "status": status,
                            "at": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")


def _balance(ds_id, key):
    st, c = L._req("GET", "/datasets/{}/finetune/calculate".format(ds_id), key=key)
    return (c or {}).get("availableCredits") if st == 200 else None


def main():
    key = L._auth("pt_live")
    ids = json.loads(REC.read_text()) if REC.exists() else {}
    st_all = _state()
    done_labels = _ledger_labels()
    only_status = "--status" in sys.argv

    for cat in CATEGORIES:
        ds_id = ids.get(cat)
        if not ds_id:
            print("{:<24} not uploaded yet".format(cat))
            continue
        s, d = L._req("GET", "/datasets/{}".format(ds_id), key=key)
        if s != 200:
            print("{:<24} status query {}".format(cat, s))
            continue
        status = d.get("status")
        print("{:<24} {:<16} rows={}".format(cat, status, d.get("row_count")))
        if only_status:
            continue

        if status == "awaiting_input":
            r, _ = L._req("POST", "/datasets/{}/run".format(ds_id),
                          {"column_mapping": COLS}, key=key)
            print("    -> run {}".format(r))
            continue
        if status != "succeeded":
            continue  # still processing

        # dataset is processed: make sure both training arms exist
        sj, jj = L._req("GET", "/datasets/{}/finetune/jobs".format(ds_id), key=key)
        if sj != 200:
            continue
        busy = any(j.get("status") not in TERMINAL for j in jj.get("jobs", []))
        if busy:
            print("    busy (a job is running)")
            continue

        ctrl, aug = "{}-control".format(cat), "{}-domain2k".format(cat)
        for label, dom in ((ctrl, 0), (aug, 2000)):
            if label in done_labels:
                continue
            if dom:
                bal = _balance(ds_id, key)
                # HOLD FLOOR: a credit top-up has been requested, so do not spend the
                # last of the balance on a subset of categories. Six categories need
                # ~120cr between them; launching three now would leave the other three
                # unentered. Wait for the refill and launch them together.
                if bal is not None and bal < CREDIT_HOLD_FLOOR:
                    print("    HOLD {} (balance {} < floor {}; awaiting top-up)".format(
                        label, bal, CREDIT_HOLD_FLOOR))
                    continue
                if bal is not None and bal < dom // 100:
                    print("    SKIP {} (needs {}cr, balance {})".format(label, dom // 100, bal))
                    continue
            try:
                res = L.launch(ds_id, "llama70b", key, columns=COLS, epochs=3, lr=None,
                               do_launch=True, aug_domain=dom, aug_general=0)
            except SystemExit as e:
                print("    {} rejected ({})".format(label, e))
                break
            except Exception as e:
                print("    {} error {}".format(label, str(e)[:90]))
                break
            jid = (res or {}).get("finetune_job_id")
            if jid:
                _record(label, cat, ds_id, dom, jid, (res or {}).get("status"))
                print("    launched {} -> {}".format(label, jid[:8]))
            break  # one launch per pass; the dataset is now busy

        # publish once the dataset has produced at least one succeeded job
        if not st_all.get(cat, {}).get("published"):
            if any(j.get("status") == "succeeded" for j in jj.get("jobs", [])):
                for target in ("huggingface", "kaggle"):
                    code, body = publish(ds_id, target, key, private=False)
                    print("    publish {} -> {}".format(target, code))
                    time.sleep(3)
                st_all.setdefault(cat, {})["published"] = True
                _save(st_all)

    print("pass complete")


if __name__ == "__main__":
    main()
