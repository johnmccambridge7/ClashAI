"""Download original base layout images from the scraped JSON.

Reads runs/base_layouts/th{N}.json and pulls every `original_url` into
runs/base_layouts/th{N}_images/{category}/th{N}_{category}_{id}.jpg.

- Skips files that already exist with non-trivial size.
- Concurrent downloads with a small worker pool to stay polite.
- Retries with backoff on transient errors.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
import time
from pathlib import Path

import requests

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


def download_one(rec: dict, out_root: Path, retries: int = 3) -> tuple[str, int]:
    url = rec["original_url"]
    cat = rec["category"]
    fname = Path(url).name
    dest = out_root / cat / fname
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 5_000:
        return ("skip", dest.stat().st_size)

    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=60)
            r.raise_for_status()
            if len(r.content) < 5_000:
                raise RuntimeError(f"suspiciously small payload ({len(r.content)} bytes)")
            dest.write_bytes(r.content)
            return ("ok", len(r.content))
        except Exception as exc:
            last_exc = exc
            time.sleep(1.0 * (attempt + 1))
    return (f"fail:{last_exc}", 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("runs/base_layouts/th4.json"))
    ap.add_argument("--out", type=Path, default=Path("runs/base_layouts/th4_images"))
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    data = json.loads(args.manifest.read_text())
    images = data["images"]
    args.out.mkdir(parents=True, exist_ok=True)

    ok = skip = fail = 0
    total_bytes = 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(download_one, rec, args.out): rec for rec in images}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            rec = futs[fut]
            status, n = fut.result()
            if status == "ok":
                ok += 1
                total_bytes += n
            elif status == "skip":
                skip += 1
                total_bytes += n
            else:
                fail += 1
                print(f"  FAIL {rec['original_url']}: {status}", file=sys.stderr)
            if i % 25 == 0 or i == len(images):
                print(
                    f"  {i}/{len(images)}  ok={ok} skip={skip} fail={fail}"
                    f"  size_so_far={total_bytes / 1e6:.1f} MB",
                    file=sys.stderr,
                )

    print(
        f"\nDone. ok={ok} skip={skip} fail={fail} "
        f"total_size={total_bytes / 1e6:.1f} MB  out={args.out}",
        file=sys.stderr,
    )
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
