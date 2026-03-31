"""
Download congress member photos from unitedstates.github.io for all legislators
who served in the 116th Congress (2019-01-03) or later.

Images are saved to data/congress_images/<bioguide>.jpg.
Falls back through available sizes: original -> 450x550 -> 225x275.
Already-downloaded images are skipped (safe to re-run).
"""

import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

CONGRESS_116_START = "2019-01-03"
OUT_DIR = os.path.join(os.path.dirname(__file__), "../data/congress_images")
BASE_URL = "https://unitedstates.github.io/images/congress/{}/{}.jpg"
SIZES = ["original", "450x550", "225x275"]
WORKERS = 20

LEGISLATOR_FILES = [
    os.path.join(os.path.dirname(__file__), "../data/legislators-current.json"),
    os.path.join(os.path.dirname(__file__), "../data/legislators-historical.json"),
]


def served_116th_or_later(terms):
    for t in terms:
        if t.get("end", "9999-99-99") >= CONGRESS_116_START:
            return True
    return False


def collect_bioguides():
    bioguides = set()
    for fname in LEGISLATOR_FILES:
        with open(fname) as f:
            for leg in json.load(f):
                bg = leg.get("id", {}).get("bioguide")
                if bg and served_116th_or_later(leg.get("terms", [])):
                    bioguides.add(bg)
    return bioguides


def fetch_with_fallback(bg):
    dest = os.path.join(OUT_DIR, f"{bg}.jpg")
    for size in SIZES:
        url = BASE_URL.format(size, bg)
        try:
            urllib.request.urlretrieve(url, dest)
            return (bg, size, True)
        except Exception:
            pass
    return (bg, None, False)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    bioguides = collect_bioguides()
    to_download = [bg for bg in sorted(bioguides)
                   if not os.path.exists(os.path.join(OUT_DIR, f"{bg}.jpg"))]

    print(f"Total legislators: {len(bioguides)}")
    print(f"Already downloaded: {len(bioguides) - len(to_download)}")
    print(f"To download: {len(to_download)}")

    ok = fail = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = {pool.submit(fetch_with_fallback, bg): bg for bg in to_download}
        for i, fut in enumerate(as_completed(futs), 1):
            bg, size, success = fut.result()
            if success:
                ok += 1
            else:
                fail += 1
                print(f"  FAIL {bg}")
            if i % 100 == 0:
                print(f"  progress: {i}/{len(to_download)} (ok={ok}, fail={fail})")

    print(f"\nDone. downloaded={ok}, failed={fail}")


if __name__ == "__main__":
    main()
