"""Small HTTP layer: retrying, resumable, atomic file downloads.

Every download is written to ``<dest>.part`` and renamed into place only once complete, so an
existing ``dest`` always means "fully downloaded" and a re-run never fetches it again. A leftover
``.part`` is resumed with an HTTP Range request when the server supports it.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Protocol

log = logging.getLogger("data_prep")

# Open-i and cancerimagingarchive.net sit behind a WAF that resets connections from non-browser
# user agents (plain curl/python-requests UAs get "connection reset"), so identify as a browser.
USER_AGENT = "Mozilla/5.0 (compatible; radintake-data-prep/0.1)"
CHUNK = 1 << 20


class Session(Protocol):  # the subset of requests.Session we use (mocked in unit tests)
    def get(self, url: str, **kwargs): ...


def make_session() -> Session:
    import requests

    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    return s


class DownloadError(RuntimeError):
    pass


def download_file(session: Session, url: str, dest: Path, *, params: dict | None = None,
                  retries: int = 5, backoff: float = 2.0, timeout: float = 120.0,
                  sleep=time.sleep) -> bool:
    """Download ``url`` to ``dest``. Returns True if bytes were fetched, False if already present."""
    if dest.exists():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with session.get(url, params=params, headers=headers, stream=True, timeout=timeout) as r:
                if r.status_code == 416:  # range past the end: the .part is already complete
                    break
                if r.status_code not in (200, 206):
                    raise DownloadError(f"HTTP {r.status_code} for {url}")
                mode = "ab" if (have and r.status_code == 206) else "wb"
                with part.open(mode) as fh:
                    for block in r.iter_content(CHUNK):
                        if block:
                            fh.write(block)
                expected = r.headers.get("Content-Length")
                got = part.stat().st_size - (have if mode == "ab" else 0)
                if expected is not None and int(expected) != got:
                    raise DownloadError(f"short read for {url}: {got} of {expected} bytes")
            break
        except Exception as exc:  # network errors, short reads, HTTP 5xx
            last_err = exc
            if attempt == retries:
                raise DownloadError(f"giving up on {url} after {retries} attempts: {exc}") from exc
            wait = backoff * attempt
            log.warning("download failed (%s), retry %d/%d in %.0fs", exc, attempt, retries, wait)
            sleep(wait)
    else:  # pragma: no cover - loop always breaks or raises
        raise DownloadError(str(last_err))
    if not part.exists() or part.stat().st_size == 0:
        raise DownloadError(f"empty download for {url}")
    part.replace(dest)
    return True
