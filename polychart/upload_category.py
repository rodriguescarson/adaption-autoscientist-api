"""Upload a category parquet to Adaption and start Adaptive Data on it.

Flow (documented by the platform's own /snippets/dataset/{id} endpoint):

    POST /datasets/upload/initiate   -> {upload_url, s3_key}
    PUT  <upload_url>                 (raw file body)
    POST /datasets/upload/complete   -> {dataset_id}
    poll /datasets/{id}/status        until row_count populates
    POST /datasets/{id}/run           with column_mapping, starts Adaptive Data

The last step is the one that is easy to miss: a freshly uploaded dataset sits at
status `awaiting_input` with a null column mapping and does nothing at all until
`/run` is called with one. Four of our earlier datasets sat in that state for five
days and were misdiagnosed as "stuck processing".

Usage:
  python -m polychart.upload_category market-analysis-news
"""

from __future__ import annotations

import json
import pathlib
import sys
import time
import urllib.error
import urllib.request

from . import launch as L

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "categories"
COLUMN_MAPPING = {"prompt": "prompt", "completion": "response"}


def _put(url: str, data: bytes) -> int:
    req = urllib.request.Request(url, data=data, method="PUT",
                                 headers={"Content-Type": "application/octet-stream"})
    try:
        return urllib.request.urlopen(req, timeout=300).status
    except urllib.error.HTTPError as e:
        return e.code


def upload(category: str, key: str) -> str | None:
    pq = OUT / category / "{}.parquet".format(category)
    if not pq.exists():
        print("missing {} - run build_category first".format(pq))
        return None
    blob = pq.read_bytes()
    name = "polychart-{}".format(category)
    print("{}: {} bytes".format(name, len(blob)))

    st, d = L._req("POST", "/datasets/upload/initiate",
                   {"name": name, "file_format": "parquet"}, key=key)
    if st not in (200, 201, 202):
        print("  initiate failed: {} {}".format(st, str(d)[:200]))
        return None
    url = d.get("upload_url")
    s3key = d.get("s3_key")
    if not s3key and url:
        # The platform's own documented snippet reads `.s3_key` from this response,
        # but the API returns ONLY `upload_url`. The key is the presigned URL's path
        # minus the leading slash and the query string.
        from urllib.parse import urlparse, unquote
        s3key = unquote(urlparse(url).path).lstrip("/")
    print("  initiate ok (s3_key={})".format((s3key or "")[:48]))

    code = _put(url, blob)
    print("  PUT -> {}".format(code))
    if code not in (200, 201, 204):
        return None

    st, d = L._req("POST", "/datasets/upload/complete",
                   {"s3_key": s3key, "name": name, "file_format": "parquet",
                    "file_size_bytes": len(blob)}, key=key)
    # 202 Accepted is the success path here, not just 200/201.
    if st not in (200, 201, 202):
        print("  complete failed: {} {}".format(st, str(d)[:200]))
        return None
    ds = d.get("dataset_id")
    print("  dataset_id {}".format(ds))

    # wait for ingestion to count rows
    for _ in range(40):
        time.sleep(15)
        s2, r = L._req("GET", "/datasets/{}/status".format(ds), key=key)
        if s2 != 200:
            continue
        status, rows = r.get("status"), r.get("row_count")
        if status == "failed":
            print("  ingestion FAILED: {}".format(str(r)[:200]))
            return None
        if rows:
            print("  ingested: status={} rows={}".format(status, rows))
            break

    # start Adaptive Data. Without this the dataset never leaves awaiting_input.
    for _ in range(5):
        s3, r = L._req("POST", "/datasets/{}/run".format(ds),
                       {"column_mapping": COLUMN_MAPPING}, key=key)
        if s3 in (200, 201, 202):
            print("  RUN started: {}".format(str(r)[:160]))
            return ds
        print("  run -> {} {}".format(s3, str(r)[:160]))
        time.sleep(20)
    return ds


def main():
    cat = sys.argv[1]
    key = L._auth("pt_live")
    ds = upload(cat, key)
    if ds:
        rec = ROOT / "data" / "runs" / "category_datasets.json"
        cur = json.loads(rec.read_text()) if rec.exists() else {}
        cur[cat] = ds
        rec.write_text(json.dumps(cur, indent=1))
        print("recorded {} -> {}".format(cat, ds))


if __name__ == "__main__":
    main()
