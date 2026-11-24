"""IU Open-i chest X-ray collection: reports (XML) + images (PNG) into data/iu/.

Layout written (all gitignored, licence CC BY-NC-ND -- never commit):

    data/iu/raw/NLMCXR_reports.tgz      downloaded archives (resumable)
    data/iu/raw/NLMCXR_png.tgz
    data/iu/reports/ecgen-radiology/*.xml
    data/iu/images/*.png
    data/iu/MANIFEST.json               size + sha256 of every file above

Re-running is a no-op: completed archives are never re-fetched, extraction is skipped when the
manifest records it for the same archive hash, and the manifest is rewritten only on change.
"""

from __future__ import annotations

import logging
import tarfile
from pathlib import Path

from data_prep.http import Session, download_file, make_session
from data_prep.manifest import build_manifest, load_manifest, sha256_file, write_manifest

log = logging.getLogger("data_prep")

BASE_URL = "https://openi.nlm.nih.gov/imgs/collections/"
ARCHIVES = {
    # archive name -> extraction subdirectory under data/iu/
    "NLMCXR_reports.tgz": "reports",
    "NLMCXR_png.tgz": "images",
}
DATASET = "iu-openi-cxr"
DERIVED = {"reports.jsonl"}  # written by data_prep.iu_parse, not part of the download


def _is_gzip(path: Path) -> bool:
    with open(path, "rb") as fh:
        return fh.read(2) == b"\x1f\x8b"


def _safe_extract(archive: Path, dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf:
            if not member.isfile():
                continue
            tf.extract(member, dest, filter="data")  # rejects absolute paths / traversal / links
            n += 1
    return n


def download_iu(root: Path, *, include_images: bool = True, session: Session | None = None) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    previous = load_manifest(root)
    extracted: dict = dict((previous or {}).get("meta", {}).get("extracted", {}))
    names = [n for n in ARCHIVES if include_images or n != "NLMCXR_png.tgz"]
    for name in names:
        archive = root / "raw" / name
        if archive.exists() and not _is_gzip(archive):
            # Open-i sometimes answers with an HTML page (HTTP 200): never treat that as the archive.
            log.warning("%s is not a gzip archive (likely an HTML error page); re-downloading", name)
            archive.unlink()
        if not archive.exists():
            session = session or make_session()
            log.info("downloading %s", BASE_URL + name)
        if download_file(session, BASE_URL + name, archive):  # no-op (no request) when present
            log.info("downloaded %s (%d bytes)", name, archive.stat().st_size)
        digest = _archive_hash(previous, root, archive)
        target = root / ARCHIVES[name]
        if extracted.get(name) == digest and target.exists():
            log.info("%s already extracted, skipping", name)
            continue
        n = _safe_extract(archive, target)
        extracted[name] = digest
        log.info("extracted %d files from %s into %s", n, name, target)
    manifest = build_manifest(root, DATASET, meta={"source": BASE_URL, "extracted": extracted,
                                                   "licence": "CC BY-NC-ND 4.0"},
                              previous=previous)
    manifest["files"] = [f for f in manifest["files"] if f["path"] not in DERIVED]
    manifest["file_count"] = len(manifest["files"])
    manifest["total_bytes"] = sum(f["size"] for f in manifest["files"])
    if previous is not None and previous.get("files") == manifest["files"] \
            and previous.get("meta") == manifest["meta"]:
        log.info("MANIFEST.json unchanged (%d files)", manifest["file_count"])
        return previous
    write_manifest(root, manifest)
    log.info("wrote %s (%d files, %d bytes)", root / "MANIFEST.json", manifest["file_count"],
             manifest["total_bytes"])
    return manifest


def _archive_hash(previous: dict | None, root: Path, archive: Path) -> str:
    """sha256 of the archive, reusing the previous manifest's value when size and mtime match."""
    rel = archive.relative_to(root).as_posix()
    st = archive.stat()
    for f in (previous or {}).get("files", []):
        if f["path"] == rel and f["size"] == st.st_size and f.get("mtime_ns") == st.st_mtime_ns:
            return f["sha256"]
    return sha256_file(archive)
