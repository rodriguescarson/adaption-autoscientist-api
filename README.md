# adaption-autoscientist-api

An unofficial, headless Python client for **Adaption's AutoScientist**, built by
reverse-engineering the web app's network traffic in July 2026, **before the official API
shipped**. Stdlib only, no dependencies.

It drove **176 fine-tuning jobs across 86 launches on 13 datasets** for the AutoScientist
Challenge 2026, which is where the findings below come from.

> The dataset generator and the study's category configurations live in a separate private
> repository until the challenge closes on 10 August 2026. This repo is the platform client:
> the part that is useful to other people regardless of what they train.

---

## Why it still exists now that the official API is out

Adaption published their [official API docs](https://docs.adaptionlabs.ai/guides/autoscientist-api/)
on 28 July 2026. The official SDK (`pip install adaption`) is the right default for launching
runs, and it exposes things this client does not, notably `max_iterations` and
`target_win_rate` for iterative search.

This client remains useful because it covers surface the SDK does not:

| Capability | Official SDK | This client |
|---|---|---|
| Launch, poll, download, cancel | ✅ | ✅ |
| `max_iterations` / `target_win_rate` | ✅ | — |
| Recommended hyperparameters | ✅ | — |
| **Per-job metrics: on-dataset AND held-out category win rate** | — | ✅ |
| **Per-job config → `training_experiment_id`** | — | ✅ |
| **Publish a dataset to HuggingFace / Kaggle** | — | ✅ |
| **Hosted Interface generation** (sessions → apps → versions) | — | ✅ |
| **App recommendations** | — | ✅ |
| **Four-step fine-tune config wizard** | — | ✅ |

The metrics gap is the important one. The SDK surfaces `best_win_rate`; this client reads the
full metrics payload including `domain_eval`, the held-out category evaluation. Those two
numbers are not interchangeable (see below).

---

## Platform behaviours worth knowing

Found the hard way, at volume. None are visible in the interface, and each will corrupt an
automated experiment without raising an error.

**1. The displayed win rate is measured on the training distribution.** The held-out category
figure is a different number and they can move in opposite directions. From a matched pair
differing only in augmentation:

| | on-dataset (displayed) | held-out category |
|---|---|---|
| baseline | **87.2** | 66.3 |
| +2,000 domain rows | **81.0** | **77.3** |

The better model is 6.2 points worse on the displayed number. Three zero-augmentation controls
scored 83.4 / 83.6 / 83.7 on-dataset and 52.1 / 63.8 / 60.7 held-out. **If you optimise the
displayed metric you can select the weaker model.**

**2. One launch is not one model.** A launch spawns several training jobs sharing a
`training_experiment_id` and the platform reports a best. 86 launches produced 176 jobs here.
The official SDK now names this `max_iterations`. An A/B test that assumes one launch equals
one model is comparing best-of-N against best-of-M.

**3. Launching onto a busy dataset returns the existing job with HTTP 201**, not an error. A
client that records the returned id attributes another run to its own condition, and the
condition it intended never executes.

**4. `eval_failed` is terminal but reads as in-flight.** Clients enumerating in-flight work
count these forever and stall against the five-job concurrency cap.

**5. A finished run's augmentation setting is not retrievable.** Not from the job record,
metrics, `/jobs/{id}/config` (hyperparameters only), or any experiments route. **Only the
launch request knows.** Record your conditions at submission time; you cannot reconstruct them
afterwards.

**6. Undocumented limits:** 1,000-row training minimum, five concurrent fine-tunes.

**7. Image-bearing datasets never receive the held-out category evaluation.** Six fine-tunes
across two vision bases, zero evaluations, one explicitly `skipped`. Text-only datasets on the
same account received it normally.

---

## Modules

```
polychart/
  launch.py             fine-tune wizard, launch, metrics, auth, rate-limit backoff
  upload_category.py    upload: initiate → PUT → complete → poll → run
  export_ds.py          publish a dataset to HuggingFace / Kaggle
  interfaces.py         hosted app generation (sessions → apps → build → versions)
  build_interfaces.py   build an Interface for every trained dataset
  queue_runner.py       condition queue honouring the 5-job cap and per-dataset limit
  category_pipeline.py  drives a dataset from processed to published
  harvest.py            re-polls late-arriving held-out evaluations
  attribute.py          recovers a job's condition via training_experiment_id
  rebuild_ledger.py     reconstructs a launch ledger from run logs
  report.py             paired within-seed analysis + sign test
  audit.py              completeness audit: every succeeded job read and attributed
  supervise.sh          keeps the long-running daemons alive
```

## Use

```bash
cp .env.example .env     # add your own ADAPTION_API_KEY
python3 -m polychart.audit      # every succeeded job, read and attributed
python3 -m polychart.report     # paired within-seed analysis
```

`.env` is gitignored and has never been committed. No credential appears in this repository
or its history.

## Note on the findings

The full study (paired within-seed design, 13 seeds) is being written up as a paper. The
headline is that **Adaptive Data augmentation genuinely works** — it improved the held-out
category on 13 of 13 seeds, mean +14 points, two-sided sign test p=0.0002, and Adaption's
advertised "+16 points from crossing 20,000 datapoints" reproduced at +17.4. The behaviours
above are reported as an audit layer that platform should ship, not as an argument that it
does not work.

### A retraction

An earlier draft of that paper claimed the platform's advertised scaling benefit *reverses*,
citing a matched pair where the score fell from 76 to 72. That was one unpaired comparison.
Repeating it properly — within seed, across 13 seeds and 176 jobs — reversed the conclusion:
13 of 13 positive, and the advertised gain reproduced at +17.4. **The claim is withdrawn.**

It is recorded here rather than deleted because it is the same error this client exists to
catch. Held-out category evaluations fire on roughly a third of jobs, so a single pair is one
draw against a metric with sd 3–5 noise, on seeds whose difficulty spans 44.5 to 85.9 — about
three times the effect being measured. A cross-seed comparison mostly reports which seeds
landed in which arm. That is why `report.py` pairs within seed and uses a sign test on
direction rather than a t-test on magnitude.

## License

MIT.
