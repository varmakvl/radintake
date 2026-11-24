"""Version lock for everything that changes model behaviour: prompts, tool/output schemas, configs.

    python -m radintake.eval.versions            # check (exit 1 on drift)
    python -m radintake.eval.versions --update   # rewrite eval/VERSIONS.lock after a deliberate change

Rules (enforced by ``tests/unit/test_versions.py`` and recorded in every run's ``run.json``):

* prompts (``prompts/<id>_vN.md``) and schemas (``schemas/<name>.vN.json``) are immutable once locked:
  changing the content of a locked file is an error - copy it to the next version (``_v2`` /
  ``.v2``) and point the config at it. Adding a new version is fine (``--update`` records it).
* configs (``configs/*.yaml``) may change, but only together with a lock update, so every behaviour
  change is a visible diff in ``eval/VERSIONS.lock``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from radintake.config import REPO_ROOT

LOCK = REPO_ROOT / "eval" / "VERSIONS.lock"
GROUPS = {"prompts": ("prompts", "*.md"), "schemas": ("schemas", "*.json"), "configs": ("configs", "*.yaml")}
IMMUTABLE = ("prompts", "schemas")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def current() -> dict[str, dict[str, str]]:
    return {g: {p.name: _sha(p) for p in sorted((REPO_ROOT / d).glob(pat))} for g, (d, pat) in GROUPS.items()}


def lock_hash(lock: dict | None = None) -> str:
    lock = lock if lock is not None else current()
    return hashlib.sha256(json.dumps(lock, sort_keys=True).encode()).hexdigest()


def check(lock_path: Path = LOCK) -> list[str]:
    if not lock_path.is_file():
        return [f"{lock_path} missing: run python -m radintake.eval.versions --update"]
    locked = json.loads(lock_path.read_text(encoding="utf-8"))
    now = current()
    problems = []
    for g, files in now.items():
        for name, h in files.items():
            old = locked.get(g, {}).get(name)
            if old is None:
                problems.append(f"{g}/{name} is new and not locked (run --update)")
            elif old != h:
                hint = "copy it to a new version instead of editing" if g in IMMUTABLE else "run --update"
                problems.append(f"{g}/{name} changed since it was locked ({hint})")
        for name in set(locked.get(g, {})) - set(files):
            problems.append(f"{g}/{name} is locked but missing")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m radintake.eval.versions")
    ap.add_argument("--update", action="store_true")
    a = ap.parse_args(argv)
    if a.update:
        LOCK.parent.mkdir(parents=True, exist_ok=True)
        LOCK.write_text(json.dumps(current(), indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {LOCK} ({lock_hash()[:12]})")
        return 0
    problems = check()
    for p in problems:
        print("DRIFT", p)
    print("versions: OK" if not problems else f"versions: {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
