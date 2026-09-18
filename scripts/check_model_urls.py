#!/usr/bin/env python3
"""Check that every model download listed in services/model_store.py still works.

Run weekly by .github/workflows/check-model-urls.yml; also fine to run by hand:

    python scripts/check_model_urls.py

Each model is checked with a HEAD request, so nothing big is downloaded. The
file must still exist and still be the size the app expects: the app rejects a
download of the wrong size, so a file replaced upstream would break users even
though its URL still answers. Gated models (SAM3) link to a web page instead
of a file, so for those it only checks that the page loads.

Exits 1 if anything is wrong.
"""
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from services import model_store  # noqa: E402  (standard library only, no PyQt6)

ATTEMPTS = 3
RETRY_DELAY_SECONDS = 10


def remote_size(url):
    """Size in bytes the server reports for url, or None if it doesn't say."""
    request = urllib.request.Request(
        url, headers={"User-Agent": "SegmentME-model-downloader"}, method="HEAD")
    with urllib.request.urlopen(request, timeout=30) as response:
        return int(response.headers.get("Content-Length") or 0) or None


def find_problem(spec):
    """None if the model's download is fine, otherwise a short reason."""
    for attempt in range(1, ATTEMPTS + 1):
        try:
            size = remote_size(spec.download_url)
        except urllib.error.HTTPError as exc:
            problem, retry = f"HTTP {exc.code} {exc.reason}", exc.code >= 500 or exc.code == 429
        except (urllib.error.URLError, TimeoutError) as exc:
            problem, retry = f"could not connect ({getattr(exc, 'reason', exc)})", True
        else:
            if spec.gated or not spec.size_bytes or size in (None, spec.size_bytes):
                return None
            return f"file changed: server reports {size:,} bytes, the app expects {spec.size_bytes:,}"
        if not retry or attempt == ATTEMPTS:
            return problem
        time.sleep(RETRY_DELAY_SECONDS)


def main():
    failures = 0
    for spec in model_store.MODELS.values():
        problem = find_problem(spec)
        if problem is None:
            print(f"OK    {spec.label:<14} {spec.download_url}")
            continue
        failures += 1
        print(f"FAIL  {spec.label:<14} {spec.download_url}\n      {problem}")
        print(f"::error title=Model download broken::{spec.label}: {problem}")
    total = len(model_store.MODELS)
    print(f"\n{total - failures}/{total} model downloads OK")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
