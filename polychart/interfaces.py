"""Build an Adaption Interface (hosted app) for a trained dataset, headlessly.

Flow, captured from the UI:

    POST /training-experiments/{exp}/app-recommendations  {"count":3}
        -> candidate app ideas, each with a recommendation_id
    POST /chat/sessions                                    {"origin":"autoscientist"}
        -> session_id
    POST /chat/sessions/{sid}/apps                         {dataset_id, experiment_id}
        -> app_id
    POST /chat/sessions/{sid}/apps/{app}/messages
        {content, app_build_self_contained, recommendation_id}
        -> triggers the build (async, a few minutes)
    GET  /chat/sessions/{sid}/apps/{app}/versions
        -> versions[].frontend_url, a live public app on modal.host
    GET  /chat/sessions/{sid}/apps/{app}
        -> is_built, api_preview_url

Note the path root is `/training-experiments/{id}`, not `/experiments/{id}`; the
latter 404s, which is why an earlier probe concluded no experiment route existed.

Usage:
  python -m polychart.interfaces recommend <experiment_id>
  python -m polychart.interfaces build <dataset_id> <experiment_id> ["prompt"]
  python -m polychart.interfaces status <session_id> <app_id>
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

from . import launch as L

ROOT = pathlib.Path(__file__).resolve().parent.parent
REC = ROOT / "data" / "runs" / "interfaces.json"


def _auth():
    """Interfaces endpoints may need the session JWT rather than the pt_live key."""
    key = L._auth("pt_live")
    st, _ = L._req("POST", "/chat/sessions", {"origin": "autoscientist"}, key=key)
    if st in (200, 201):
        return key
    print("  pt_live rejected on /chat/sessions ({}); falling back to login JWT".format(st))
    return L._auth("login")


def recommend(exp: str, key: str, count: int = 3):
    st, d = L._req("POST", "/training-experiments/{}/app-recommendations".format(exp),
                   {"count": count}, key=key)
    if st not in (200, 201):
        print("  recommendations -> {} {}".format(st, str(d)[:200]))
        return []
    recs = d if isinstance(d, list) else d.get("recommendations", d.get("items", []))
    for i, r in enumerate(recs):
        if isinstance(r, dict):
            print("  [{}] {} :: {}".format(
                i, r.get("recommendation_id", "?")[:8],
                (r.get("title") or r.get("content") or "")[:110]))
    return recs


def build(dataset: str, exp: str, key: str, prompt: str | None = None,
          recommendation_id: str | None = None):
    st, d = L._req("POST", "/chat/sessions", {"origin": "autoscientist"}, key=key)
    if st not in (200, 201):
        print("session failed: {} {}".format(st, str(d)[:200]))
        return None
    sid = d["session_id"]
    print("  session {}".format(sid))

    st, d = L._req("POST", "/chat/sessions/{}/apps".format(sid),
                   {"dataset_id": dataset, "experiment_id": exp}, key=key)
    if st not in (200, 201):
        print("  app create failed: {} {}".format(st, str(d)[:200]))
        return None
    app = d["app_id"]
    print("  app {}".format(app))

    body = {"content": prompt, "app_build_self_contained": False}
    if recommendation_id:
        body["recommendation_id"] = recommendation_id
    st, d = L._req("POST", "/chat/sessions/{}/apps/{}/messages".format(sid, app),
                   body, key=key)
    print("  build triggered -> {}".format(st))
    return {"session_id": sid, "app_id": app}


def status(sid: str, app: str, key: str):
    st, d = L._req("GET", "/chat/sessions/{}/apps/{}".format(sid, app), key=key)
    if st != 200:
        print("  status -> {}".format(st))
        return None
    print("  is_built={} url={}".format(d.get("is_built"), d.get("api_preview_url")))
    s2, v = L._req("GET", "/chat/sessions/{}/apps/{}/versions".format(sid, app), key=key)
    if s2 == 200:
        for ver in (v.get("versions") or []):
            print("  version {} -> {}".format(ver.get("version_id", "")[:8],
                                              ver.get("frontend_url")))
            if ver.get("summary"):
                print("     {}".format(ver["summary"][:160]))
    return d


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    key = _auth()
    if cmd == "recommend":
        recommend(sys.argv[2], key)
    elif cmd == "build":
        ds, exp = sys.argv[2], sys.argv[3]
        prompt = sys.argv[4] if len(sys.argv) > 4 else None
        r = build(ds, exp, key, prompt)
        if r:
            cur = json.loads(REC.read_text()) if REC.exists() else {}
            cur[ds] = r
            REC.write_text(json.dumps(cur, indent=1))
            print("  recorded -> {}".format(REC))
    elif cmd == "status":
        status(sys.argv[2], sys.argv[3], key)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
