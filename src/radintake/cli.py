"""Batch CLI entry point.

Contract (see README "Batch CLI output schema" and schemas/cli_output.v1.json):

    radintake batch --in <folder> --out <folder>

Writes exactly one ``<input-file-name>.json`` per input file in ``--out``, validating
against ``schemas/cli_output.v1.json``. Unsupported or corrupt inputs produce a JSON
with ``status: "error"`` instead of crashing the batch. Exit code is 0 when the batch
ran (even with per-file errors), non-zero only for invalid arguments.

This is a stub: the sprint implements the pipeline behind it.
"""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="radintake")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("batch", help="Process every file in a folder")
    b.add_argument("--in", dest="inp", required=True, help="Input folder")
    b.add_argument("--out", dest="out", required=True, help="Output folder (one JSON per input)")
    b.add_argument("--config", default=None, help="Optional config YAML")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "batch":
        raise NotImplementedError("batch pipeline not implemented yet")
    return 2


if __name__ == "__main__":
    sys.exit(main())
