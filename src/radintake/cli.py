"""Batch CLI entry point.

Contract (see README "Batch CLI output schema" and schemas/cli_output.v1.json):

    radintake batch --in <folder> --out <folder> [--config <yaml>]

Writes exactly one ``<input-file-name>.json`` per input file in ``--out``, validating
against ``schemas/cli_output.v1.json``. Unsupported or corrupt inputs produce a JSON
with ``status: "error"`` instead of crashing the batch. Exit code is 0 when the batch
ran (even with per-file errors), non-zero only for invalid arguments.

Traces go to ``RADINTAKE_TRACE_DIR`` (default ``<out>/_traces``). Logs are JSON lines on
stderr with PHI scrubbed; stdout carries only a count summary (no file names, no PHI).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

EXIT_OK = 0
EXIT_BAD_ARGS = 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="radintake")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("batch", help="Process every file in a folder")
    b.add_argument("--in", dest="inp", required=True, help="Input folder")
    b.add_argument("--out", dest="out", required=True, help="Output folder (one JSON per input)")
    b.add_argument("--config", default=None, help="Optional config YAML")
    return p


def list_inputs(folder: Path) -> list[Path]:
    """Regular files directly inside ``folder``, sorted (sub-folders are not descended)."""
    return [p for p in sorted(folder.iterdir()) if p.is_file()]


def run_batch(inp: Path, out: Path, config_path: str | None = None) -> dict[str, int]:
    """Thin wrapper over :meth:`radintake.sdk.Pipeline.process_batch` (one JSON per input)."""
    from radintake.config import load_config
    from radintake.logging import configure_logging, get_logger
    from radintake.sdk import Pipeline

    cfg = load_config(config_path)
    configure_logging(cfg.log_level)
    log = get_logger("radintake.cli")
    out.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    files = list_inputs(inp)
    with Pipeline(cfg, output_dir=out, trace_dir=cfg.trace_dir or (out / "_traces"),
                  data_dir=cfg.data_dir) as pipe:
        run_id = pipe.run_id
        log.info("batch_start", run_id=run_id, n_inputs=len(files), offline=cfg.offline)
        pipe.process_batch(files, on_result=lambda r: counts.update([r.status]))
    summary = {"inputs": len(files), **{k: counts.get(k, 0) for k in ("ok", "needs_review", "error")},
               "run_id": run_id}
    log.info("batch_done", **summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as exc:  # argparse already printed usage
        return int(exc.code or EXIT_BAD_ARGS)
    if args.cmd == "batch":
        inp, out = Path(args.inp), Path(args.out)
        if not inp.is_dir():
            print(f"radintake: error: --in is not a folder: {args.inp}", file=sys.stderr)
            return EXIT_BAD_ARGS
        if args.config and not Path(args.config).is_file():
            print(f"radintake: error: --config file not found: {args.config}", file=sys.stderr)
            return EXIT_BAD_ARGS
        if out.exists() and not out.is_dir():
            print(f"radintake: error: --out exists and is not a folder: {args.out}", file=sys.stderr)
            return EXIT_BAD_ARGS
        summary = run_batch(inp, out, args.config)
        print(json.dumps(summary))
        return EXIT_OK
    return EXIT_BAD_ARGS


if __name__ == "__main__":
    sys.exit(main())
