"""Dataset download / verify entry point.

    python -m data_prep.download iu [--no-images]   # IU Open-i reports + PNGs -> data/iu/
    python -m data_prep.download verify iu          # re-hash data/iu against MANIFEST.json

Exit code of ``verify`` is 0 when every listed file is intact and 1 otherwise.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from data_prep.manifest import unlisted_files, verify_manifest

DATA_ROOT = Path("data")
DATASETS = ("iu", "tcia")
_DERIVED = {"iu": {"reports.jsonl"}, "tcia": {"summary.json"}}


def _verify(name: str, root: Path) -> int:
    problems = verify_manifest(root)
    extra = unlisted_files(root, ignore=_DERIVED.get(name))
    if extra:
        print(f"note: {len(extra)} file(s) under {root} not in manifest (first: {extra[0]})")
    if problems:
        for p in problems[:50]:
            print(f"FAIL {p}")
        print(f"verify {name}: {len(problems)} problem(s) under {root}")
        return 1
    print(f"verify {name}: OK ({root})")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m data_prep.download", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_iu = sub.add_parser("iu", help="download IU Open-i reports + images")
    p_iu.add_argument("--no-images", action="store_true", help="reports only (skip the 1.3 GB PNGs)")
    p_v = sub.add_parser("verify", help="re-hash a dataset against its MANIFEST.json")
    p_v.add_argument("dataset", choices=DATASETS)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.cmd == "iu":
        from data_prep.download_iu import download_iu

        download_iu(args.data_root / "iu", include_images=not args.no_images)
        return 0
    if args.cmd == "verify":
        return _verify(args.dataset, args.data_root / args.dataset)
    return 2  # pragma: no cover


if __name__ == "__main__":
    sys.exit(main())
