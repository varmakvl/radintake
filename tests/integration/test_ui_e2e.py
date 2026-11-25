"""Playwright smoke test of the review UI served at /ui (real app on a free local port, offline).

Synthetic fixtures only (tests/integration/fixtures.py). Every browser request is recorded and must
go to localhost: the UI is self-contained and nothing leaves the machine.
"""

from __future__ import annotations

import os
import socket
import threading
import time
from urllib.parse import urlparse

import httpx
import pytest

from .fixtures import build_corpus

pytestmark = pytest.mark.integration
sync_api = pytest.importorskip("playwright.sync_api")

LOCAL = {"127.0.0.1", "localhost"}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    import uvicorn

    os.environ.setdefault("RADINTAKE_OFFLINE", "1")
    os.environ["RADINTAKE_DATA_DIR"] = str(tmp_path_factory.mktemp("ui_data"))
    from radintake.api.app import create_app

    port = _free_port()
    srv = uvicorn.Server(uvicorn.Config(create_app(), host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    base = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            if httpx.get(base + "/health", timeout=2).status_code in (200, 503):
                break
        except httpx.HTTPError:
            time.sleep(0.2)
    yield base
    srv.should_exit = True
    th.join(timeout=10)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    folder = tmp_path_factory.mktemp("ui_inputs")
    build_corpus(folder)
    return folder


def _api_case(base, path, timeout=900):
    with open(path, "rb") as fh:
        cid = httpx.post(base + "/cases", files={"file": (path.name, fh)}).json()["case_id"]
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = httpx.get(f"{base}/cases/{cid}").json()["status"]
        if st in ("ok", "needs_review", "error"):
            return cid
        time.sleep(1)
    pytest.fail("case did not finish")


def test_review_ui_flow(server, corpus):
    requests: list[str] = []
    queued = _api_case(server, corpus / "report_injected_multipage.pdf")  # lands in the review queue
    with sync_api.sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page()
        page.on("request", lambda r: requests.append(r.url))
        images: list[int] = []
        page.on("response", lambda r: images.append(r.status) if "/pages/1.png" in r.url else None)
        page.goto(server + "/ui/")
        assert page.locator("h1").inner_text() == "radintake"

        # upload -> poll -> page image + boxes
        page.set_input_files("#file", str(corpus / "report_clean.png"))
        page.click("#upload-form button")
        # (page.evaluate / wait_for_function would violate the UI's strict CSP, so wait on selectors)
        page.wait_for_selector("#case-status:text-matches('^(ok|needs_review)$')", timeout=900_000)
        page.wait_for_selector("#overlay .box")
        assert page.locator("#overlay .box").count() >= 1
        assert images and images[0] == 200  # the PHI-masked page image loaded

        # review queue
        page.click("nav a[data-view=queue]")
        page.wait_for_selector("#queue-table tbody tr")
        assert queued in page.locator("#queue-table tbody").inner_text()

        # query -> cited answer (or abstention) is shown
        page.click("nav a[data-view=query]")
        page.fill("#question", "Which studies report a calcified granuloma in the right upper lobe?")
        page.click("#query-form button")
        page.wait_for_selector("#answer-text")
        assert page.locator("#answer .citation").count() >= 1 or "Abstained" in page.locator("#answer-text").inner_text()
        browser.close()

    assert requests
    hosts = {urlparse(u).hostname for u in requests if urlparse(u).scheme in ("http", "https")}
    assert hosts <= LOCAL, f"non-local requests: {sorted(hosts - LOCAL)}"


def test_ui_csp_and_masked_page_route(server):
    r = httpx.get(server + "/ui/")
    assert r.status_code == 200 and "default-src 'self'" in r.headers["content-security-policy"]
    assert "http://" not in r.text and "https://" not in r.text
    assert httpx.get(server + "/cases/nope/pages/1.png").status_code == 404
