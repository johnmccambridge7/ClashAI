"""Scrape clashofclans-layouts.com to enumerate base layout image URLs.

Usage:
    python3 scripts/scrape_base_layouts.py --th 4 --pages 17 \
        --out runs/base_layouts/th4.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE = "https://clashofclans-layouts.com"
LIST_URL = BASE + "/plans/th_{th}/page_{page}/"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

# Matches /pics/th{N}_plans/{category}/{size}/th{N}_{category}_{id}.jpg
IMG_RE = re.compile(
    r"/pics/th(?P<th>\d+)_plans/(?P<category>[a-z0-9_]+)/(?P<size>[a-z]+)/"
    r"th\d+_[a-z0-9_]+_(?P<id>\d+)\.(?P<ext>jpg|png)",
    re.IGNORECASE,
)


def fetch(url: str, retries: int = 3, sleep: float = 1.0) -> str:
    last_exc = None
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()
            return r.text
        except Exception as exc:
            last_exc = exc
            time.sleep(sleep * (attempt + 1))
    raise RuntimeError(f"failed to fetch {url}: {last_exc}")


def extract_images(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    found: dict[tuple, dict] = {}

    candidates: list[str] = []
    for tag in soup.find_all(["img", "source", "a"]):
        for attr in ("src", "data-src", "data-original", "srcset", "href"):
            val = tag.get(attr)
            if not val:
                continue
            # srcset can contain multiple comma-separated URLs
            for piece in re.split(r"[,\s]+", val):
                if piece:
                    candidates.append(piece)

    for raw in candidates:
        url = urljoin(BASE, raw.split("?")[0])
        m = IMG_RE.search(url)
        if not m:
            continue
        key = (m["th"], m["category"], m["size"], m["id"])
        if key in found:
            continue
        th_n = int(m["th"])
        cat = m["category"]
        bid = int(m["id"])
        ext = m["ext"].lower()
        found[key] = {
            "url": url,
            "th": th_n,
            "category": cat,
            "size": m["size"],
            "id": bid,
            "ext": ext,
            "original_url": (
                f"{BASE}/pics/th{th_n}_plans/{cat}/original/"
                f"th{th_n}_{cat}_{bid}.{ext}"
            ),
        }
    return list(found.values())


def scrape(th: int, pages: int, delay: float) -> list[dict]:
    all_imgs: dict[tuple, dict] = {}
    for page in range(1, pages + 1):
        url = LIST_URL.format(th=th, page=page)
        print(f"[page {page}/{pages}] {url}", file=sys.stderr)
        html = fetch(url)
        page_imgs = extract_images(html)
        for img in page_imgs:
            key = (img["th"], img["category"], img["size"], img["id"])
            img.setdefault("pages", [])
            existing = all_imgs.get(key)
            if existing:
                existing["pages"].append(page)
            else:
                img["pages"] = [page]
                all_imgs[key] = img
        print(f"  -> {len(page_imgs)} image refs on page", file=sys.stderr)
        time.sleep(delay)
    return list(all_imgs.values())


def summarize(images: list[dict]) -> dict:
    by_cat: Counter = Counter()
    by_size: Counter = Counter()
    by_ext: Counter = Counter()
    id_range: dict[str, list[int]] = defaultdict(list)
    samples: dict[str, str] = {}

    for img in images:
        by_cat[img["category"]] += 1
        by_size[img["size"]] += 1
        by_ext[img["ext"]] += 1
        id_range[img["category"]].append(img["id"])
        samples.setdefault(f"{img['category']}/{img['size']}", img["url"])

    return {
        "total_unique": len(images),
        "by_category": dict(by_cat),
        "by_size": dict(by_size),
        "by_ext": dict(by_ext),
        "id_range_per_category": {
            cat: {"min": min(ids), "max": max(ids), "count": len(set(ids))}
            for cat, ids in id_range.items()
        },
        "sample_urls": samples,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--th", type=int, default=4)
    ap.add_argument("--pages", type=int, default=17)
    ap.add_argument("--delay", type=float, default=0.6)
    ap.add_argument("--out", type=Path, default=Path("runs/base_layouts/th4.json"))
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    images = scrape(args.th, args.pages, args.delay)
    summary = summarize(images)

    args.out.write_text(json.dumps({"summary": summary, "images": images}, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {len(images)} image records to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
