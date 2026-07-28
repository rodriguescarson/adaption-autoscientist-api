"""Build an Adaption Interface for every dataset that has a trained model.

An Interface is a hosted app generated from a dataset + its training experiment,
served publicly on modal.host. Two of our twenty datasets had one; the rest showed
"Create" in the dashboard. Each is a submission asset, and building them is a third
Adaption product in the entry (Adaptive Data -> AutoScientist -> Interfaces).

Needs a training_experiment_id, which is not on the dataset record. It comes from
any succeeded job's config:  GET /datasets/{ds}/finetune/jobs/{job}/config
    -> best_job_config.training_experiment_id

Then the build flow (captured from the UI):
    POST /chat/sessions                       {"origin":"autoscientist"}
    POST /chat/sessions/{s}/apps              {dataset_id, experiment_id}
    POST /chat/sessions/{s}/apps/{a}/messages {content, app_build_self_contained}
    GET  /chat/sessions/{s}/apps/{a}/versions -> frontend_url

Usage:
  python -m polychart.build_interfaces list
  python -m polychart.build_interfaces build [max_n]
  python -m polychart.build_interfaces status
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

from . import launch as L

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNDIR = ROOT / "data" / "runs"
REC = RUNDIR / "interfaces.json"

# The app brief. Kept task-specific rather than generic so the generated app is a
# chart-honesty tool, which is what the dataset actually trained for.
PROMPT = (
    "A chart honesty checker. Paste the plotted values and how the chart is drawn "
    "(chart type, axis range, whether it starts at zero, whether the axis is inverted), "
    "and the app extracts the true values, computes Tufte's Lie Factor, names the "
    "distortion mechanism, and rates severity from honest to extremely manipulative. "
    "Support comparing two renderings of the same data and ranking them by visual "
    "honesty. Built for editors and data-visualisation reviewers deciding which "
    "rendering is safe to publish."
)


def _load():
    return json.loads(REC.read_text()) if REC.exists() else {}


def _save(d):
    RUNDIR.mkdir(parents=True, exist_ok=True)
    REC.write_text(json.dumps(d, indent=1))


def experiment_for(ds_id: str, key: str) -> str | None:
    """Find a training_experiment_id via any succeeded job on the dataset."""
    st, j = L._req("GET", "/datasets/{}/finetune/jobs".format(ds_id), key=key)
    if st != 200:
        return None
    for job in j.get("jobs", []):
        if job.get("status") != "succeeded":
            continue
        time.sleep(2)
        s, c = L._req("GET", "/datasets/{}/finetune/jobs/{}/config".format(
            ds_id, job["finetune_job_id"]), key=key)
        if s == 200:
            e = (c or {}).get("best_job_config", {}).get("training_experiment_id")
            if e:
                return e
    return None


def build_one(name: str, ds_id: str, key: str):
    exp = experiment_for(ds_id, key)
    if not exp:
        print("  {:<32} no trained experiment yet".format(name))
        return None
    st, d = L._req("POST", "/chat/sessions", {"origin": "autoscientist"}, key=key)
    if st not in (200, 201):
        print("  {:<32} session -> {}".format(name, st))
        return None
    sid = d["session_id"]
    st, d = L._req("POST", "/chat/sessions/{}/apps".format(sid),
                   {"dataset_id": ds_id, "experiment_id": exp}, key=key)
    if st not in (200, 201):
        print("  {:<32} app -> {}".format(name, st))
        return None
    app = d["app_id"]
    st, _ = L._req("POST", "/chat/sessions/{}/apps/{}/messages".format(sid, app),
                   {"content": PROMPT, "app_build_self_contained": False}, key=key)
    print("  {:<32} building  session={} app={} ({})".format(
        name, sid[:8], app[:8], st))
    return {"session_id": sid, "app_id": app, "experiment_id": exp, "dataset_id": ds_id}


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    key = L._auth("pt_live")
    rec = _load()

    st, d = L._req("GET", "/datasets", key=key)
    if st != 200:
        print("datasets -> {}".format(st))
        return
    rows = d.get("datasets", d) if isinstance(d, dict) else d
    # only datasets big enough to have trained, newest first
    cands = [x for x in rows if (x.get("row_count") or 0) >= 1000]

    if cmd == "list":
        for x in cands:
            mark = "HAS" if x["dataset_id"] in rec else "--"
            print("  {:<3} {:<34} rows={}".format(mark, x["name"], x.get("row_count")))
        return

    if cmd == "status":
        for ds, r in rec.items():
            s, d2 = L._req("GET", "/chat/sessions/{}/apps/{}".format(
                r["session_id"], r["app_id"]), key=key)
            if s == 200:
                print("  {:<38} built={} {}".format(
                    ds[:38], d2.get("is_built"), d2.get("api_preview_url") or ""))
            time.sleep(2)
        return

    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    made = 0
    for x in cands:
        if made >= limit:
            break
        if x["dataset_id"] in rec:
            continue
        r = build_one(x["name"], x["dataset_id"], key)
        if r:
            rec[x["dataset_id"]] = r
            _save(rec)
            made += 1
            time.sleep(8)
    print("built {} interface(s)".format(made))


if __name__ == "__main__":
    main()
