"""Publish a dataset to HuggingFace / Kaggle through Adaption's upload endpoints.

The export is NOT the `export-links` route (that only reads back links that already
exist, and returns nulls until a publish has happened). Publishing is:

    POST /datasets/{id}/upload/huggingface
    POST /datasets/{id}/upload/kaggle

as multipart/form-data. The form carries optional card images (banner, quality_gain,
grade, percentile_chart, cover_image) plus `isPrivate`. The images are decoration for
the dataset card; the publish succeeds without them, so we send only `isPrivate`
unless files are supplied.

This publishes PUBLICLY by default (isPrivate=false), because the challenge
submission requires public dataset links.

Usage:
  python -m polychart.export_ds <dataset_name> [huggingface|kaggle|both] [--private]
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
import uuid

from . import launch as L

BASE = "https://api.prod.adaptionlabs.ai/api/v1"


def _multipart(fields: dict) -> tuple[bytes, str]:
    """Build a multipart/form-data body from simple text fields."""
    boundary = "----polychart{}".format(uuid.uuid4().hex)
    out = []
    for k, v in fields.items():
        out.append("--{}\r\n".format(boundary).encode())
        out.append('Content-Disposition: form-data; name="{}"\r\n\r\n'.format(k).encode())
        out.append("{}\r\n".format(v).encode())
    out.append("--{}--\r\n".format(boundary).encode())
    return b"".join(out), "multipart/form-data; boundary={}".format(boundary)


def publish(ds_id: str, target: str, key: str, private: bool = False):
    body, ctype = _multipart({"isPrivate": "true" if private else "false"})
    url = "{}/datasets/{}/upload/{}".format(BASE, ds_id, target)
    req = urllib.request.Request(url, data=body, method="POST", headers={
        "Authorization": "Bearer {}".format(key),
        "Content-Type": ctype,
        "Accept": "*/*",
        "Origin": "https://adaptionlabs.ai",
        "Referer": "https://adaptionlabs.ai/",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36",
    })
    try:
        r = urllib.request.urlopen(req, timeout=180)
        return r.status, r.read().decode("utf-8", "replace")[:400]
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")[:400]
    except Exception as e:
        return "ERR", str(e)[:200]


def main():
    name = sys.argv[1]
    target = sys.argv[2] if len(sys.argv) > 2 else "both"
    private = "--private" in sys.argv
    key = L._auth("pt_live")
    ds = L.resolve(name, key)
    print("{} -> {}".format(name, ds))

    targets = ["huggingface", "kaggle"] if target == "both" else [target]
    for t in targets:
        st, body = publish(ds, t, key, private)
        print("  {:<12} -> {} {}".format(t, st, body.replace("\n", " ")[:260]), flush=True)
        time.sleep(4)

    # read the links back
    for _ in range(6):
        s, d = L._req("GET", "/datasets/{}/export-links".format(ds), key=key)
        if s == 200:
            print("  links: {}".format(json.dumps(d)))
            break
        time.sleep(15)


if __name__ == "__main__":
    main()
