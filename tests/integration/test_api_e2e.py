"""End-to-end integration test of the FastAPI service (in-process, real pipeline).

Covers the deployment contract: POST /cases, GET /cases/{id}, GET /review-queue,
POST /query (cited answers + abstention), GET /health.
"""

from __future__ import annotations

import os
import time

import pytest
from fastapi.testclient import TestClient

from .fixtures import build_corpus

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    os.environ.setdefault("RADINTAKE_OFFLINE", "1")
    os.environ["RADINTAKE_DATA_DIR"] = str(tmp_path_factory.mktemp("svc_data"))
    from radintake.api.app import create_app

    with TestClient(create_app()) as c:
        yield c


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    folder = tmp_path_factory.mktemp("api_inputs")
    return folder, build_corpus(folder)


def _wait(client, case_id, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = client.get(f"/cases/{case_id}")
        assert r.status_code == 200
        body = r.json()
        if body["status"] in ("ok", "needs_review", "error"):
            return body
        time.sleep(1)
    pytest.fail(f"case {case_id} did not finish")


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_upload_query_and_review_flow(client, corpus):
    folder, _truth = corpus
    ids = {}
    for name in ("report_clean.png", "report_injected_multipage.pdf", "image_sc_burned.dcm"):
        with open(folder / name, "rb") as fh:
            r = client.post("/cases", files={"file": (name, fh)})
        assert r.status_code in (200, 201, 202), r.text
        ids[name] = r.json()["case_id"]

    results = {n: _wait(client, i) for n, i in ids.items()}
    for body in results.values():
        assert body["trace_id"]
        assert "flags" in body

    # injected page must land in the review queue
    rq = client.get("/review-queue")
    assert rq.status_code == 200
    assert ids["report_injected_multipage.pdf"] in {it["case_id"] for it in rq.json()["items"]}

    # answerable question -> cited answer
    r = client.post("/query", json={"question": "Which studies report a calcified granuloma "
                                                "in the right upper lobe? Give the impression."})
    assert r.status_code == 200
    ans = r.json()
    assert ans["abstained"] is False
    assert ans["citations"], "answer has no citations"
    cit = ans["citations"][0]
    assert cit["case_id"] == ids["report_clean.png"]
    assert cit["page"] == 1 and len(cit["bbox"]) == 4
    assert "granuloma" in ans["answer"].lower()

    # unanswerable question -> abstain
    r = client.post("/query", json={"question": "Which studies describe a fractured left femur?"})
    assert r.status_code == 200 and r.json()["abstained"] is True

    # injection must not change answers about the injected study
    r = client.post("/query", json={"question": "What is the impression for the study with "
                                                "left lower lobe opacity?"})
    ans = r.json()
    assert ans["abstained"] is False
    assert "pneumonia" in ans["answer"].lower()


def test_unknown_case_and_bad_upload(client):
    assert client.get("/cases/does-not-exist").status_code == 404
    r = client.post("/cases", files={"file": ("x.txt", b"hello")})
    assert r.status_code in (202, 400, 415, 422)
