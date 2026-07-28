"""Automate an AutoScientist finetune launch over the internal REST API.

Reverse-engineered from a captured browser session (docs/adaption-api.md). The
public `adaption` SDK exposes only `/datasets`; the finetune wizard the UI drives
lives under `/datasets/{id}/finetune/*` and was assumed unreachable (finding A6).
It is not: the long-lived `pt_live_` key in .env authenticates those endpoints too,
so the whole launch runs headless with no hourly JWT.

The wizard is four config writes then one launch:

  POST /finetune/config   training method  (instruction, size=all)
  POST /finetune/config   columns          (prompt/completion mapping)
  POST /finetune/config   augmentation     (0 / 0  -- never augment, it dilutes)
  POST /finetune/config   recipe           (base model + LoRA params)
  POST /finetune/hyperparams/validate      (server checks the params)
  GET  /finetune/calculate                 (credit cost)
  POST /finetune/launch                    (model + hyperparams + idempotency_key)

Everything else in the capture (images, posthog/hubspot telemetry, OPTIONS
preflights, cached GETs) is noise and dropped.

Safe by default: without --launch the script configures, validates, and prints the
credit cost, then STOPS before spending anything. Add --launch to actually fire.

  python -m polychart.launch --list
  python -m polychart.launch --dataset fertility_rate_qa_pairs --preset llama70b
  python -m polychart.launch --dataset fertility_rate_qa_pairs --preset llama70b --launch
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import urllib.error
import urllib.request
import uuid

ROOT = pathlib.Path(__file__).resolve().parent.parent
BASE = "https://api.prod.adaptionlabs.ai/api/v1"

# Base-model presets. `modules` differs by architecture: Scout is MoE and needs the
# explicit expert/feed-forward list (all-linear is rejected for it); dense Llama
# takes all-linear. lr and size are the platform's own values for each.
PRESETS = {
    # v5's winning config. Together only defines finetune batch sizes for the
    # "-Reference" variant; the plain meta-llama id fails with "max batch size not
    # defined". v5's adapter reported the -Reference base, so this is the real one.
    "llama70b": {
        "model": "meta-llama/Llama-3.3-70B-Instruct-Reference",
        "base_model_size": "70B",
        "lora_trainable_modules": "all-linear",
        "learning_rate": 0.0001,
    },
    # The captured run (Scout 109B). lr 5e-5 as the platform set it.
    "scout": {
        "model": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "base_model_size": "109B",
        "lora_trainable_modules": (
            "k_proj,o_proj,q_proj,v_proj,"
            "shared_expert.gate_proj,shared_expert.up_proj,shared_expert.down_proj,"
            "feed_forward.gate_proj,feed_forward.up_proj,feed_forward.down_proj"),
        "learning_rate": 0.00005,
    },
    # Vision-language. The atlas dataset (chart_lie_factor_analysis) carries an image
    # column, so the platform rejects text-only bases for it. Three earlier runs used
    # this model with NO augmentation and scored 43.8 / 50.5 / 53.9 on-data, which is
    # the control arm for testing whether the augmentation lift also holds multimodally.
    "gemma27b_vlm": {
        "model": "google/gemma-3-27b-it-VLM",
        "base_model_size": "27B",
        "lora_trainable_modules": "all-linear",
        "learning_rate": 0.0001,
    },
}

# Fixed parts of the recipe, shared across presets (the v5 / captured values).
COMMON_HP = {
    "n_epochs": 3, "batch_size": "max", "lora": True, "lora_r": 64,
    "lora_alpha": 128, "lora_dropout": 0, "lr_scheduler_type": "cosine",
    "min_lr_ratio": 0.1, "scheduler_num_cycles": 0.5, "warmup_ratio": 0.05,
    "max_grad_norm": 1, "weight_decay": 0.02, "n_evals": 5,
    "training_method": "sft", "train_on_inputs": False,
}


def _env() -> dict:
    return dict(re.findall(r"^([A-Z_]+)=(.*)$", (ROOT / ".env").read_text(), re.M))


def _login(env: dict) -> str:
    """Trade email + password for a fresh session JWT (the UI's own auth path)."""
    email, pw = env.get("EMAIL", "carson@celabe.com"), env.get("PASSWORD", "")
    if not pw:
        sys.exit("PASSWORD not in .env (needed for --auth login)")
    st, resp = _req("POST", "/auth/login", {"email": email, "password": pw}, key=None)
    tok = (resp or {}).get("access_token") or (resp or {}).get("token")
    if not (200 <= st < 300) or not tok:
        sys.exit("login failed: {} {}".format(st, json.dumps(resp)[:200]))
    return tok


def _auth(mode: str) -> str:
    """Return a bearer token. Both the long-lived pt_live_ key and a login JWT work
    on every finetune endpoint (verified); pt_live_ is the zero-friction default."""
    env = _env()
    if mode == "login":
        return _login(env)
    k = env.get("ADAPTION_API_KEY", "")
    if not k.startswith("pt_"):
        sys.exit("ADAPTION_API_KEY (pt_live_...) not found in .env")
    return k


def _req(method: str, path: str, body=None, key: str = None):
    import time
    data = json.dumps(body).encode() if body is not None else None
    headers = {"accept": "*/*"}
    if key:
        headers["Authorization"] = "Bearer " + key
    if data is not None:
        headers["content-type"] = "application/json"
    for attempt in range(5):
        req = urllib.request.Request(BASE + path, data=data, headers=headers, method=method)
        try:
            r = urllib.request.urlopen(req, timeout=60)
            raw = r.read().decode("utf-8", "replace")
            return r.status, (json.loads(raw) if raw.strip() else {})
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 4:            # rate limited: back off and retry
                time.sleep(5 * (attempt + 1))
                continue
            raw = e.read().decode("utf-8", "replace")
            try:
                return e.code, json.loads(raw)
            except Exception:
                return e.code, {"_raw": raw[:300]}


def list_datasets(key: str) -> list:
    _, d = _req("GET", "/datasets", key=key)
    return d.get("datasets", [])


def resolve(name_or_id: str, key: str) -> str:
    if re.fullmatch(r"[0-9a-f-]{36}", name_or_id):
        return name_or_id
    hits = [x for x in list_datasets(key) if x.get("name") == name_or_id]
    if not hits:
        sys.exit("no dataset named {!r} (use --list)".format(name_or_id))
    if len(hits) > 1:
        sys.exit("{} datasets named {!r}; pass the id".format(len(hits), name_or_id))
    return hits[0]["dataset_id"]


def _config(ds: str, payload: dict, key: str, label: str):
    st, resp = _req("POST", "/datasets/{}/finetune/config".format(ds), payload, key)
    ok = 200 <= st < 300
    print("  {:<22} {} {}".format(label, st, "" if ok else json.dumps(resp)[:200]))
    return ok


def aug_step(ds: str, key: str) -> dict:
    """The augmentation step, including the platform's suggested domain/general counts."""
    _, r = _req("GET", "/datasets/{}/finetune/steps/finetune_augmentation".format(ds), key=key)
    return r


def launch(ds: str, preset: str, key: str, *, columns: dict, epochs: int,
           lr: float | None, do_launch: bool, aug_domain: int = 0, aug_general: int = 0,
           hp_overrides: dict | None = None) -> dict:
    p = PRESETS[preset]
    hp = dict(COMMON_HP)
    hp["n_epochs"] = epochs
    hp["learning_rate"] = lr if lr is not None else p["learning_rate"]
    hp["base_model_size"] = p["base_model_size"]
    hp["lora_trainable_modules"] = p["lora_trainable_modules"]
    # Recipe knobs (rank, alpha, warmup, weight decay, train_on_inputs, ...) cost no
    # credits to vary, since only augmentation rows are billed. This is what makes a
    # hyperparameter sweep free.
    if hp_overrides:
        hp.update(hp_overrides)

    print("dataset  :", ds)
    print("model    :", p["model"], "| size", p["base_model_size"])
    print("recipe   : r={lora_r} alpha={lora_alpha} {mods} | {ep} epochs | lr {lr}".format(
        mods=p["lora_trainable_modules"] if p["lora_trainable_modules"] == "all-linear"
        else "explicit-moe-modules", ep=hp["n_epochs"], lr=hp["learning_rate"], **hp))
    print("columns  : prompt={prompt} completion={completion}".format(**columns))
    print("\n[1/6] configuring the wizard")
    _config(ds, {"finetune_training_method": {
        "finetune_training_method_value": {"value": "instruction"},
        "finetune_model_size_constraint_value": {"value": "all"}}}, key, "training method")
    _config(ds, {"finetune_columns": {
        "finetune_column_selections": {"value": columns}}}, key, "columns")
    _config(ds, {"finetune_augmentation": {
        "finetune_augmentation_domain_rows": {"value": aug_domain},
        "finetune_augmentation_general_rows": {"value": aug_general},
        "suggested_general_augmentation_count": {"value": None},
        "suggested_domain_augmentation_count": {"value": None}}}, key,
        "augmentation D={} G={}".format(aug_domain, aug_general))
    _config(ds, {"finetune_recipe": {
        "finetune_recipe_algorithm": {"value": "lora"},
        "finetune_base_model": {"value": p["model"]},
        "finetune_recipe_lora": {"value": True},
        "finetune_recipe_lora_r": {"value": hp["lora_r"]},
        "finetune_recipe_lora_alpha": {"value": hp["lora_alpha"]},
        "finetune_recipe_lora_trainable_modules": {"value": p["lora_trainable_modules"]},
        "finetune_recipe_lr_scheduler_type": {"value": "cosine"},
        "finetune_recipe_warmup_ratio": {"value": 0.05},
        "finetune_recipe_max_grad_norm": {"value": 1},
        "finetune_recipe_weight_decay": {"value": 0.02},
        "finetune_recipe_n_epochs": {"value": hp["n_epochs"]}}}, key, "recipe")

    print("\n[2/6] validating hyperparameters")
    # Matches the captured validate body exactly: base_model_size, no model string.
    st, vresp = _req("POST", "/datasets/{}/finetune/hyperparams/validate".format(ds),
                     {**{k: hp[k] for k in COMMON_HP}, "learning_rate": hp["learning_rate"],
                      "lora_trainable_modules": hp["lora_trainable_modules"],
                      "base_model_size": hp["base_model_size"]}, key)
    print("  validate: {} {}".format(st, "OK" if 200 <= st < 300 else json.dumps(vresp)[:300]))
    if not (200 <= st < 300):
        sys.exit("validation failed; not launching")

    print("\n[3/6] credit cost (aug D={} G={})".format(aug_domain, aug_general))
    st, calc = _req("GET", "/datasets/{}/finetune/calculate"
                    "?augmentation_domain_rows={}&augmentation_general_rows={}".format(
                        ds, aug_domain, aug_general), key=key)
    print("  {}".format(json.dumps(calc)[:400]))

    if not do_launch:
        print("\nDRY RUN: configured + validated. Re-run with --launch to fire the job.")
        return {"dry_run": True, "cost": calc}

    idem = str(uuid.uuid4())
    body = {
        "model": p["model"],
        "hyperparams": {**{k: hp[k] for k in COMMON_HP}, "learning_rate": hp["learning_rate"],
                        "lora_trainable_modules": hp["lora_trainable_modules"],
                        "base_model_size": hp["base_model_size"]},
        "idempotency_key": idem}
    print("\n[4/6] LAUNCHING (idempotency_key={})".format(idem))
    st, resp = _req("POST", "/datasets/{}/finetune/launch".format(ds), body, key)
    if st in (401, 403):
        print("  auth rejected on launch; retrying with a fresh login JWT")
        st, resp = _req("POST", "/datasets/{}/finetune/launch".format(ds), body, _login(_env()))
    print("  launch: {} {}".format(st, json.dumps(resp)[:400]))
    if 200 <= st < 300:
        exp = resp.get("training_experiment_id") or resp.get("id") or resp.get("experiment_id")
        print("\nLAUNCHED. experiment:", exp or "(see response above)")
    else:
        sys.exit("launch failed")
    return resp


def status(ds: str, key: str):
    """Show the current finetune job(s) for a dataset."""
    st, d = _req("GET", "/datasets/{}".format(ds), key=key)
    jobs = d.get("finetune_jobs") or d.get("finetuneJobs") or []
    if not jobs and d.get("finetune_job_id"):
        jobs = [d]
    print("dataset:", d.get("name", ds), "| status:", d.get("status", "?"))
    for j in jobs:
        print("  job {}  {}  model={}  {}".format(
            (j.get("finetune_job_id") or "?")[:8], j.get("status", "?"),
            j.get("model", "?"), j.get("error_message") or ""))
    if not jobs:
        print("  (no finetune job fields on the dataset object; raw keys: {})".format(list(d.keys())[:12]))


def _latest_job(ds, key, evaluated=False):
    """Newest job (optionally newest that has evaluation), plus the full job list."""
    _, d = _req("GET", "/datasets/{}/finetune/jobs".format(ds), key=key)
    jobs = d.get("jobs", [])
    if evaluated:
        j = next((j for j in jobs if j.get("status") == "succeeded"), None)
    else:
        j = jobs[0] if jobs else None
    return j, jobs


def _metrics(ds, job_id, key):
    _, m = _req("GET", "/datasets/{}/finetune/jobs/{}/metrics".format(ds, job_id), key=key)
    return m


def results(names_or_ids, key):
    """Both competition panels per dataset: on-dataset and held-out category win rate."""
    print("{:<28} {:<10} {:>8} {:>8}  {}".format("dataset", "status", "on-data", "category", "model"))
    for n in names_or_ids:
        ds = resolve(n, key)
        job, jobs = _latest_job(ds, key, evaluated=True)
        if not job:
            act = next((j for j in jobs if j.get("status") != "succeeded"), None)
            print("{:<28} {:<10} {:>8} {:>8}".format(n[:28], (act or {}).get("status", "none"), "-", "-"))
            continue
        m = _metrics(ds, job["finetune_job_id"], key)
        on = m.get("win_rates", {}).get("finetuned_model_win_rate_pct", "-")
        cat = (m.get("domain_eval") or {}).get("win_rates", {}).get("finetuned_model_win_rate_pct", "-")
        print("{:<28} {:<10} {:>8} {:>8}  {}".format(
            n[:28], job["status"], on, cat, job["model"].split("/")[-1]))


def metrics_detail(name, key):
    """Full metrics for a dataset's latest job: quality, both win panels, loss curve."""
    ds = resolve(name, key)
    job, _ = _latest_job(ds, key, evaluated=True)
    if not job:
        sys.exit("no completed job for " + name)
    m = _metrics(ds, job["finetune_job_id"], key)
    q = m.get("dataset_quality", {})
    w = m.get("win_rates", {})
    dw = (m.get("domain_eval") or {}).get("win_rates", {})
    print("job     :", job["finetune_job_id"][:8], "|", job["model"].split("/")[-1])
    print("quality : {} -> {} (grade {}->{})".format(
        q.get("before_score"), q.get("after_score"), q.get("grade_before"), q.get("grade_after")))
    print("on-data : base {}% / adapted {}%".format(
        w.get("base_model_win_rate_pct"), w.get("finetuned_model_win_rate_pct")))
    print("category: base {}% / adapted {}%  ({})".format(
        dw.get("base_model_win_rate_pct"), dw.get("finetuned_model_win_rate_pct"),
        (m.get("domain_eval") or {}).get("domain", "?")))
    tc = m.get("training_curves", [])
    if tc:
        last = tc[-1]
        print("curve   : {} steps, final train_loss {:.3f} val_loss {}".format(
            len(tc), last.get("train_loss", 0),
            round(last["val_loss"], 3) if last.get("val_loss") else "n/a"))


def download_weights(name, key, out=None):
    ds = resolve(name, key)
    job, _ = _latest_job(ds, key, evaluated=True)
    if not job:
        sys.exit("no completed job for " + name)
    jid = job["finetune_job_id"]
    out = out or "weights/{}_{}.tar.gz".format(name.replace("/", "_")[:24], jid[:8])
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    print("downloading weights: job {} -> {}".format(jid[:8], out))
    req = urllib.request.Request(
        BASE + "/datasets/{}/finetune/jobs/{}/download".format(ds, jid),
        headers={"Authorization": "Bearer " + key, "accept": "*/*"})
    got = 0
    with urllib.request.urlopen(req, timeout=600) as r, open(out, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
    print("done: {:.0f} MB -> {}".format(got / 1e6, out))


def export_links(name, key):
    ds = resolve(name, key)
    _, e = _req("GET", "/datasets/{}/export-links".format(ds), key=key)
    print("HF    :", e.get("huggingface") or "(not exported yet)")
    print("Kaggle:", e.get("kaggle") or "(not exported yet)")


def main():
    ap = argparse.ArgumentParser(description="Launch an AutoScientist finetune headless.")
    ap.add_argument("--list", action="store_true", help="list datasets and exit")
    ap.add_argument("--status", metavar="DATASET", help="show finetune job status for a dataset")
    ap.add_argument("--results", nargs="+", metavar="DATASET", help="on-data + category win rate per dataset")
    ap.add_argument("--metrics", metavar="DATASET", help="full metrics (quality, both panels, loss curve)")
    ap.add_argument("--download", metavar="DATASET", help="download the trained weights")
    ap.add_argument("--out", help="output path for --download")
    ap.add_argument("--export-links", dest="export_links", metavar="DATASET",
                    help="show HF/Kaggle adapted-dataset export URLs")
    ap.add_argument("--dataset", help="dataset name or id")
    ap.add_argument("--preset", default="llama70b", choices=list(PRESETS), help="base-model preset")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=None, help="override learning rate")
    ap.add_argument("--prompt-col", default="original_prompt")
    ap.add_argument("--completion-col", default="fused_generation")
    ap.add_argument("--aug-domain", type=int, default=0, help="domain augmentation rows")
    ap.add_argument("--aug-general", type=int, default=0, help="general augmentation rows")
    ap.add_argument("--launch", action="store_true", help="actually fire (default is dry run)")
    ap.add_argument("--auth", default="pt_live", choices=["pt_live", "login"],
                    help="pt_live_ key (default) or email+password login JWT")
    args = ap.parse_args()

    key = _auth(args.auth)
    if args.list:
        for x in list_datasets(key):
            print("  {}  {:<14} {}".format(x["dataset_id"], x.get("status", "?"), x.get("name", "?")))
        return
    if args.status:
        status(resolve(args.status, key), key)
        return
    if args.results:
        results(args.results, key)
        return
    if args.metrics:
        metrics_detail(args.metrics, key)
        return
    if args.download:
        download_weights(args.download, key, args.out)
        return
    if args.export_links:
        export_links(args.export_links, key)
        return
    if not args.dataset:
        ap.error("--dataset required (or use --list)")

    ds = resolve(args.dataset, key)
    launch(ds, args.preset, key,
           columns={"prompt": args.prompt_col, "completion": args.completion_col},
           epochs=args.epochs, lr=args.lr, do_launch=args.launch,
           aug_domain=args.aug_domain, aug_general=args.aug_general)


if __name__ == "__main__":
    main()
