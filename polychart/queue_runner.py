"""Work a queue of finetune conditions, keeping the platform's 5-job limit saturated.

Measured platform limits:
  * one active job per DATASET, and
  * a global cap of 5 in-flight finetunes ("You already have 5 fine-tune jobs in
    flight - wait for one to finish").

Rather than polling 20 datasets to count in-flight jobs (expensive under the
aggressive rate limiter), we use the launch response itself as the governor: a
429 mentioning "in flight" means the fleet is full, so wait and retry.

The queue is idempotent across restarts: every successful launch is appended to
fleet.jsonl, and anything already recorded there is skipped. Background tasks get
killed roughly hourly, so just re-run this; it picks up where it left off.

Usage: python -m polychart.queue_runner [max_minutes]
"""

from __future__ import annotations

# NOTE: "eval_failed" is TERMINAL. The platform leaves jobs in this state after the
# held-out eval errors; treating it as in-flight made four zombie jobs on one dataset
# look permanently busy and inflated the in-flight count, which blocked every launch.


import calendar
import json
import pathlib
import sys
import time

from . import launch as L

RUNDIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "runs"
LOG = RUNDIR / "fleet.jsonl"
COLS = {"prompt": "original_prompt", "completion": "fused_generation"}

# (label, dataset, domain, general, preset)
# Free controls first (0 credits), then the paired augmented arms, then the new
# multimodal architecture. Paired control+augmented on the same seed is what turns
# the one-seed augmentation finding into a replicated, cross-seed result.
# PAIRED CROSS-SEED DESIGN. For each usable seed: a free control (0 augmentation)
# and a +2,000-domain arm (20cr). Every dataset is its own block, so the comparison
# is within-seed and the augmentation effect is not confounded with seed difficulty.
# ~17 usable seeds x 20cr = ~340cr, which buys a paired N~17 study instead of one
# 259cr extreme-dose cell. Datasets under the platform's 1,000-row training minimum
# (polychart-scout-*, 15 rows) are excluded.
_SEEDS = [
    "chart_data_qa_pairs",
    "extreme_poverty_chart_qa",
    "chart_qa_obesity_renewables",
    "chart_qa_health_metrics",
    "chart_qa_co2_schooling",
    "chart_qa_population_schooling",
    "chart_qa_with_axis_tricks",
    "energy_and_carbon_qa",
    # The four polychart-abl-* seeds are EXCLUDED: their Adaptive-Data processing is
    # stuck (dataset page shows "Data is being processed" since 07/22, prompt and
    # completion lengths both 0 words), so the derived original_prompt /
    # fused_generation columns never materialised and every launch fails with
    # "AutoScientist fine-tuning failed. No successful evaluation was produced."
    # Re-add them only after their processing is retried and completes.
    "tb_mortality_chart_qa",
    "chart_qa_hdi_co2_trends",
    "electricity_demand_chart_qa",
    "pop_growth_chart_qa",
]

def _short(s):
    return s.replace("chart_qa_", "").replace("_chart_qa", "").replace("polychart-", "")


# All controls first, then all augmented arms. Interleaving them per-seed would
# serialize on the same dataset (one job per dataset) and stall the whole queue;
# this way the 5 in-flight slots are always filled by *different* datasets.
# Seeds that ALREADY have a control with a returned category number. Their augmented
# arm is the highest-value job in the queue, because it is the one that completes a
# usable within-seed pair. Everything else only builds toward a pair.
_HAVE_CONTROL_CATEGORY = [
    "tb_mortality_chart_qa",          # controls 60.7, 66.2
    "extreme_poverty_chart_qa",       # control 63.8
    "chart_data_qa_pairs",            # controls 52.1, 59.3
    "chart_qa_with_axis_tricks",      # 75.8, 78.1 (config unverified, still worth pairing)
    "energy_and_carbon_qa",           # 62.5 (config unverified)
]

# ORDERING IS A SPEND-RATE DECISION. Only 5 jobs run at once and each takes ~2.5h, so
# slots - not credits - are the scarce resource. A free control occupies a slot while
# burning 0 credits, so front-loading free work starves the budget. Order is therefore:
#   1. paid arms that COMPLETE a pair (highest scientific value per credit)
#   2. the two controls still needed to close a pair (free, but pair-critical)
#   3. paid arms on the remaining seeds
#   4. high-dose 8k arms - 80cr per slot, i.e. 4x the burn rate of a 2k arm, and they
#      extend the dose-response curve rather than just replicating it
#   5. replicates, then extra control rounds last
QUEUE = [("{}-domain2k".format(_short(s)), s, 2000, 0, "llama70b")
         for s in _HAVE_CONTROL_CATEGORY]
QUEUE += [("{}-control".format(_short(s)), s, 0, 0, "llama70b")
          for s in ("chart_qa_with_axis_tricks", "pop_growth_chart_qa")]
QUEUE += [("{}-domain2k".format(_short(s)), s, 2000, 0, "llama70b")
          for s in _SEEDS if s not in _HAVE_CONTROL_CATEGORY]
QUEUE += [("{}-domain8k".format(_short(s)), s, 8000, 0, "llama70b") for s in _SEEDS]
QUEUE += [("{}-control".format(_short(s)), s, 0, 0, "llama70b") for s in _SEEDS]
# Second control round. The held-out category eval fires for only some jobs and cannot
# be triggered (domain_eval_status None = never attempted, and it never resolves), so a
# single control per seed often yields no category number and the pair is unusable.
# Controls are free, so replicating them costs only a slot, and it doubles as the
# within-cell variance estimate the effect size currently lacks.
QUEUE += [("{}-control2".format(_short(s)), s, 0, 0, "llama70b") for s in _SEEDS]
QUEUE += [("{}-control3".format(_short(s)), s, 0, 0, "llama70b") for s in _SEEDS]
# Second augmented replicate per seed. Category noise is sd 3-5, so a single draw per
# arm cannot separate a ~13-point effect from chance; a second augmented draw halves
# the standard error of each seed's delta and guards against one lucky/unlucky eval
# deciding a seed's sign in the paired test.
QUEUE += [("{}-domain2k-B".format(_short(s)), s, 2000, 0, "llama70b") for s in _SEEDS]
# Dose-response extension: 8k on the seeds where a 2k pair already exists, to test
# whether the lift keeps climbing or plateaus (fertility suggested a plateau: 2k 81.8
# vs 8k 84.9/76.0).
# The atlas seed is image-bearing, so it needs a vision base; Scout has never been
# tried on it (only gemma-VLM has).
QUEUE.append(("atlas-scout-2k", "chart_lie_factor_analysis", 2000, 0, "scout"))

# Blocks above intentionally overlap (a seed can appear in both the pair-critical list
# and the all-seeds list). Deduplicate by label, keeping FIRST occurrence so the
# priority ordering above is preserved; a repeated label could otherwise be launched
# twice before the first launch is recorded.
def _dedup(q):
    seen, out = set(), []
    for item in q:
        if item[0] in seen:
            continue
        seen.add(item[0])
        out.append(item)
    return out


QUEUE = _dedup(QUEUE)

# SUBMISSION-MAXIMISING RUNS. The scored metric is the held-out category win rate, and our
# best model is fertility suggested-full at 85.9. axis_tricks (84.9) and obesity (84.4)
# reached nearly that on only 2k augmentation, so Adaption's full suggested dose is the
# best shot at a higher submitted number. ~259cr each. Placed FIRST because they are the
# only runs that can move the number we are actually judged on.
QUEUE = [("axis_tricks-suggested", "chart_qa_with_axis_tricks", 17872, 8000, "llama70b"),
         ("obesity-suggested", "chart_qa_obesity_renewables", 17872, 8000, "llama70b"),
         ("health_metrics-domain2k2", "chart_qa_health_metrics", 2000, 0, "llama70b"),
        ] + QUEUE


# FREE EPOCH SWEEP. Base training is voucher-covered, so epochs cost zero credits - and
# with the balance at 0 this is the only lever left that can still move the number. The
# platform's recipe engine recommends 3 epochs to everyone (a competitor reported being
# given the identical Llama-3.3-70B / r=64 / 3-epoch recipe), so it is almost certainly
# untuned across the whole field. Suggestive prior: on axis_tricks the one legacy 10-epoch
# run scored the highest category of that seed's 3-epoch runs (78.1 vs 75.8/70.9/68.0),
# though its augmentation is unknown so that comparison is confounded. These arms are
# controls (0 augmentation) on seeds that already have a control, so epochs is the ONLY
# thing that varies against an existing same-seed baseline.
QUEUE = [("{}-ep{}".format(_short(s), ep), s, 0, 0, "llama70b", ep) for s, ep in (
    ("tb_mortality_chart_qa", 8),
    ("extreme_poverty_chart_qa", 8),
    ("chart_data_qa_pairs", 8),
    ("tb_mortality_chart_qa", 12),
    ("extreme_poverty_chart_qa", 12),
)] + QUEUE


# VERIFIED-CONTROL round. Four seeds have an augmented arm plus several control-looking
# category readings whose launch confirmations were lost to interleaved logs. The API
# exposes NO augmentation or row-count field on either the job record or the metrics
# record (verified: job keys and metrics keys both lack any such field), so those
# readings cannot be attributed on evidence - only by inference, which already produced
# one wrong result here. Controls are free, so the honest fix is to re-run them under a
# fresh label with the fixed recorder (job id taken from the launch response) rather
# than guess. These are the pairs that decide p<0.05.
QUEUE = [("{}-controlV".format(_short(s)), s, 0, 0, "llama70b") for s in (
    "chart_qa_with_axis_tricks",
    "pop_growth_chart_qa",
    "chart_qa_obesity_renewables",
    "chart_qa_population_schooling",
)] + QUEUE



def _done_labels():
    if not LOG.exists():
        return set()
    out = set()
    for line in LOG.read_text().splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("label"):
            out.add(r["label"])
    return out


def _record(label, name, dom, gen, preset, job, status, epochs=3):
    RUNDIR.mkdir(parents=True, exist_ok=True)
    LOG.open("a").write(json.dumps({
        "label": label, "dataset": name, "domain": dom, "general": gen,
        "preset": preset, "epochs": epochs, "job": job, "status": status,
        "at": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")


def main():
    max_min = float(sys.argv[1]) if len(sys.argv) > 1 else 50.0
    deadline = time.time() + max_min * 60
    key = L._auth("pt_live")
    done = _done_labels()
    todo = [q for q in QUEUE if q[0] not in done]
    print("queue: {} pending, {} already launched".format(len(todo), len(done)), flush=True)

    # Cycle through the queue, launching whatever is currently launchable. A busy
    # dataset is skipped (not waited on) so one slow seed never stalls the rest.
    while todo and time.time() < deadline:
        progressed = False
        for item in list(todo):
            if time.time() >= deadline:
                break
            label, name, dom, gen, preset = item[:5]
            n_epochs = item[5] if len(item) > 5 else 3
            try:
                ds = L.resolve(name, key)
            except Exception as e:
                print("  {} resolve failed: {}".format(label, e), flush=True)
                todo.remove(item)
                continue
            # A dataset already busy with its own job can never accept another. FAIL
            # CLOSED: if this check is throttled we must NOT launch. Launching on a busy
            # dataset does not error - the platform returns the EXISTING job with a 201,
            # which we would then record as the new condition. That silently mislabels
            # the run and means the intended condition never executes at all.
            st, d = L._req("GET", "/datasets/{}/finetune/jobs".format(ds), key=key)
            if st != 200:
                continue
            if any(j.get("status") not in
                   ("succeeded", "failed", "cancelled", "canceled", "error", "eval_failed")
                   for j in d.get("jobs", [])):
                continue
            # Credit guard. Only augmentation is billed (base training is voucher
            # covered), so free controls always proceed; a paid arm is skipped when the
            # balance cannot cover it, which lets the queue keep draining free work
            # right down to an empty balance instead of stalling on an unaffordable item.
            cost = (dom + gen) // 100
            if cost:
                cst, cdat = L._req("GET", "/datasets/{}/finetune/calculate".format(ds), key=key)
                bal = (cdat or {}).get("availableCredits") if cst == 200 else None
                if bal is not None and bal < cost:
                    print("  SKIP {} - needs {}cr, balance {}".format(label, cost, bal), flush=True)
                    todo.remove(item)
                    continue
                if bal is not None:
                    print("  [credits {}] ".format(bal), end="", flush=True)
            print("  launching {} (d={} g={} {})".format(label, dom, gen, preset), flush=True)
            try:
                res = L.launch(ds, preset, key, columns=COLS, epochs=n_epochs, lr=None,
                               do_launch=True, aug_domain=dom, aug_general=gen)
            except SystemExit as e:
                # launch() calls sys.exit("launch failed") on a non-201 (e.g. a 429
                # throttle or a full fleet). That must NOT kill the queue runner.
                res = None
                print("    launch rejected ({}), will retry".format(e), flush=True)
            except Exception as e:
                res = None
                print("    exception: {}".format(e), flush=True)
            # Record straight from the launch response. Re-querying the job list is
            # unreliable under the rate limiter: a FAILED query left a *successful*
            # launch unrecorded, so the item stayed queued and was relaunched later,
            # double-spending credits. That is what produced the stray duplicate 8k run.
            job = None
            if isinstance(res, dict) and res.get("finetune_job_id"):
                # Guard against the platform handing back a PRE-EXISTING job: if the
                # returned created_at is not within the last couple of minutes, this
                # launch did not actually create anything, so leave the item queued
                # instead of recording a run that never happened.
                fresh = True
                ts = res.get("created_at")
                if ts:
                    try:
                        # created_at is UTC. time.mktime() would interpret it as LOCAL
                        # time, which skews by the UTC offset and made every fresh job
                        # look hours old - the guard then discarded our own successful
                        # launches. calendar.timegm() interprets the struct as UTC.
                        made = calendar.timegm(time.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S"))
                        fresh = abs(time.time() - made) < 300
                    except Exception:
                        fresh = True
                if not fresh:
                    print("    STALE job returned ({}, created {}) - dataset was busy; "
                          "leaving {} queued".format(res["finetune_job_id"][:8], ts, label),
                          flush=True)
                    time.sleep(60)
                    continue
                job = {"finetune_job_id": res["finetune_job_id"],
                       "status": res.get("status", "pending")}
            else:
                time.sleep(12)
                st2, d2 = L._req("GET", "/datasets/{}/finetune/jobs".format(ds), key=key)
                if st2 == 200:
                    for j in d2.get("jobs", []):
                        if j.get("status") not in ("succeeded", "failed", "cancelled",
                                                   "canceled", "error"):
                            job = j
                            break
            if job:
                _record(label, name, dom, gen, preset,
                        job.get("finetune_job_id"), job.get("status"), n_epochs)
                print("    OK {} {}".format(job.get("finetune_job_id", "")[:8],
                                            job.get("status")), flush=True)
                todo.remove(item)
                progressed = True
                time.sleep(20)
                continue
            # No job appeared: the 5-job fleet is full or we were throttled. Leave the
            # item queued and move on; the next cycle retries it.
            print("    no slot (fleet full / throttled)", flush=True)
            time.sleep(120)
        if not progressed:
            # Nothing launchable this pass: everything is busy or the fleet is full.
            print("  full pass with no launches; {} left, waiting".format(len(todo)), flush=True)
            time.sleep(300)
    if todo:
        print("time budget reached; {} still queued, re-run to continue".format(len(todo)),
              flush=True)
    else:
        print("queue drained", flush=True)


if __name__ == "__main__":
    main()
