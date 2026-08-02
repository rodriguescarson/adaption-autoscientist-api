"""Publish an Adaption Interface to a permanent URL, and revive an expired one.

Two endpoints, neither in the official documentation, both reachable with a
`pt_live_` API key:

    GET  /chat/slug-available?slug=<slug>&appId=<app>
         -> {"available": true|false}

    POST /chat/sessions/<session>/apps/<app>/publish   {"slug": "<slug>"}
         -> {"slug": …, "url": "https://<slug>.adaptionlabs.app",
             "status": "active", "version_id": …, "unpublished_at": null}

Why this matters more than it looks. The build flow (see interfaces.py) returns a
`*.w.modal.host` preview URL, and that URL is an ephemeral sandbox: ours answered
200 on 29 July and were refusing TCP connections by 2 August, DNS still resolving,
container gone. Anything that quotes a preview URL — a README, a demo link, a
competition submission judged after its own deadline — is quoting a link that dies
in a few days, silently, with no notification.

`publish` returns a URL on Adaption's own domain that does not depend on a warm
container. That is the link to hand to anyone.

If a preview URL has already expired there is no start/deploy/restart endpoint
(all 404). Posting any message to the app rebuilds it and mints a fresh version,
which `publish` can then pin. `revive()` does that.

Publishing the same app under a second slug does not retire the first: both
remain active.

    python -m polychart.publish list
    python -m polychart.publish publish <session_id> <app_id> <slug>
    python -m polychart.publish revive  <session_id> <app_id>
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request

from . import launch as L

REBUILD_PROMPT = ("Redeploy this app so the preview URL is live again. "
                  "Keep the existing functionality unchanged.")
BUILD_WAIT = 200


def slug_available(slug: str, app_id: str, key: str) -> bool | None:
    """True/False, or None when the platform did not answer."""
    st, d = L._req("GET", "/chat/slug-available?slug={}&appId={}".format(slug, app_id), key=key)
    if st != 200:
        return None
    return (d or {}).get("available")


def publish(session_id: str, app_id: str, slug: str, key: str):
    """Publish to https://<slug>.adaptionlabs.app. Returns the response dict."""
    st, d = L._req("POST", "/chat/sessions/{}/apps/{}/publish".format(session_id, app_id),
                   {"slug": slug}, key=key)
    if st not in (200, 201):
        raise RuntimeError("publish returned {}: {}".format(st, json.dumps(d)[:200]))
    return d


def versions(session_id: str, app_id: str, key: str):
    st, d = L._req("GET", "/chat/sessions/{}/apps/{}/versions".format(session_id, app_id), key=key)
    if st != 200:
        raise RuntimeError("versions returned {}".format(st))
    return sorted((d or {}).get("versions", []),
                  key=lambda x: x.get("created_at", ""), reverse=True)


def revive(session_id: str, app_id: str, key: str, wait: int = BUILD_WAIT):
    """Rebuild an app whose preview URL has stopped serving, and return the new one."""
    st, _ = L._req("POST", "/chat/sessions/{}/apps/{}/messages".format(session_id, app_id),
                   {"content": REBUILD_PROMPT, "app_build_self_contained": False}, key=key)
    if st not in (200, 201):
        raise RuntimeError("rebuild message returned {}".format(st))
    time.sleep(wait)
    vs = versions(session_id, app_id, key)
    return vs[0].get("frontend_url") if vs else None


def is_serving(url: str, timeout: int = 25) -> bool:
    """A published URL can exist and still not answer; check rather than assume."""
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return 200 <= r.status < 400
    except Exception:
        return False


def main():
    key = L._auth("pt_live")
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    if cmd == "publish":
        sid, app, slug = sys.argv[2], sys.argv[3], sys.argv[4]
        avail = slug_available(slug, app, key)
        if avail is False:
            print("slug taken:", slug)
            return
        d = publish(sid, app, slug, key)
        print(json.dumps(d, indent=1))
        print("serving:", is_serving(d["url"]))
    elif cmd == "revive":
        sid, app = sys.argv[2], sys.argv[3]
        url = revive(sid, app, key)
        print("new preview url:", url, "| serving:", is_serving(url) if url else False)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
