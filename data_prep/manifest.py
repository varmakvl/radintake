"""MANIFEST.json helpers shared by the IU and TCIA downloaders.

A manifest lists every data file under a dataset root with its size and sha256, so a later
``verify`` can prove the local copy is intact without re-downloading anything.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

MANIFEST_NAME = "MANIFEST.json"
# Files that are bookkeeping, not data: never listed in the manifest.
_EXCLUDED_NAMES = {MANIFEST_NAME}
_EXCLUDED_SUFFIXES = {".part"}


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def iter_data_files(root: Path) -> list[Path]:
    """All data files under ``root`` (sorted, posix-relative order), excluding bookkeeping files."""
    files = [
        p
        for p in root.rglob("*")
        if p.is_file() and p.name not in _EXCLUDED_NAMES and p.suffix not in _EXCLUDED_SUFFIXES
    ]
    return sorted(files, key=lambda p: p.relative_to(root).as_posix())


def build_manifest(root: Path, dataset: str, meta: dict | None = None,
                   previous: dict | None = None) -> dict:
    """Hash every data file under ``root``.

    ``previous`` (an earlier manifest) lets unchanged files (same size and mtime) reuse their hash,
    so re-running a download over 1+ GB of intact data stays cheap.
    """
    prev = {f["path"]: f for f in (previous or {}).get("files", [])}
    entries = []
    for path in iter_data_files(root):
        rel = path.relative_to(root).as_posix()
        st = path.stat()
        old = prev.get(rel)
        if old and old.get("size") == st.st_size and old.get("mtime_ns") == st.st_mtime_ns:
            digest = old["sha256"]
        else:
            digest = sha256_file(path)
        entries.append({"path": rel, "size": st.st_size, "sha256": digest, "mtime_ns": st.st_mtime_ns})
    return {
        "dataset": dataset,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "file_count": len(entries),
        "total_bytes": sum(e["size"] for e in entries),
        "meta": meta or {},
        "files": entries,
    }


def load_manifest(root: Path) -> dict | None:
    path = root / MANIFEST_NAME
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(root: Path, manifest: dict) -> Path:
    path = root / MANIFEST_NAME
    tmp = path.with_suffix(".json.part")
    tmp.write_text(json.dumps(manifest, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def verify_manifest(root: Path) -> list[str]:
    """Re-hash every listed file. Returns a list of problems (empty == intact).

    Problems: manifest missing, a listed file missing, or a size / sha256 mismatch. Files on disk
    that are not listed (e.g. derived outputs such as reports.jsonl) are reported by
    :func:`unlisted_files` and are not failures.
    """
    manifest = load_manifest(root)
    if manifest is None:
        return [f"{root / MANIFEST_NAME}: missing (run the download first)"]
    problems: list[str] = []
    for entry in manifest["files"]:
        rel = entry["path"]
        path = root / rel
        if not path.is_file():
            problems.append(f"{rel}: missing")
            continue
        if path.stat().st_size != entry["size"]:
            problems.append(f"{rel}: size {path.stat().st_size} != {entry['size']}")
            continue
        if sha256_file(path) != entry["sha256"]:
            problems.append(f"{rel}: sha256 mismatch")
    return problems


def unlisted_files(root: Path, ignore: set[str] | None = None) -> list[str]:
    """Data files present under ``root`` but absent from its manifest."""
    manifest = load_manifest(root) or {"files": []}
    listed = {e["path"] for e in manifest["files"]} | (ignore or set())
    return [
        p.relative_to(root).as_posix()
        for p in iter_data_files(root)
        if p.relative_to(root).as_posix() not in listed
    ]
