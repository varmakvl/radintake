"""Fresh held-out evaluation set (golden v2) from IU reports never used in v1.

    python -m data_prep.heldout [--reports data/iu/reports.jsonl] [--seed 241] [--n-docs 112]
                                [--rendered data/generated_v2] [--corpus data/corpus_v2] [--splits splits]
                                [--select-only]

Why a second golden set: golden v1 (``splits/golden_ids.txt``) was scored once by the v1 final run
and is now spent. v2 is a fresh set, built and locked the same way, and scored once by the v2 final
run (``python -m radintake.eval.run --split golden_v2 --final``, guarded by
``eval/GOLDEN_V2_USED.json``).

Steps (reproducible, seed 241):

1. **Exclude v1.** Every report id in ``splits/dev_ids.txt`` and ``splits/golden_ids.txt`` (330)
   is out; the candidate pool is the remaining renderable IU reports.
2. **Near-duplicate check against all of v1.** The method of ``data_prep.split`` (exact pass +
   MinHash/LSH at Jaccard 0.8, every candidate re-checked with the exact Jaccard, union-find) runs
   over v1 + the candidate pool; a candidate whose dup group holds ANY v1 report is dropped, so no
   dup group spans v1 and v2. The drop count is recorded.
3. **Plan + render + degrade** with the v1 code and composition rules (``data_prep.render`` plan:
   25 % multi-page, 40 % handwriting, injected docs; ``data_prep.generate``: tiers dealt
   round-robin, PNG/JPEG/PDF/TIFF) under the new seed. Doc ids are ``G0001..`` so they can never
   collide with v1's ``D0001..``.
4. **Committed outputs (ids + hashes only, no report text):** ``splits/golden_v2_ids.txt``
   (``<doc_id>\\t<report_id>``), ``splits/golden_v2.lock`` (sha256 of the ids file, LF endings),
   ``splits/golden_v2_dedup.json`` (method, counts, v2-internal dup groups) and the golden v2
   section of ``splits/SPLIT.md``. Rendered pages and GT stay under gitignored ``data/``.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path

from data_prep.split import NUM_PERM, SHINGLE, THRESHOLD, dup_groups, golden_hash

SEED = 241
N_DOCS = 112          # same size as golden v1 (112 docs / 145 pages)
N_INJECTED = 8        # >= 5 injected docs (v1 golden held 4)
DOC_PREFIX = "G"
SECTION_MARKER = "<!-- golden_v2 -->"
TIERS = ("clean_scan", "poor_scan", "phone_photo")


def v1_report_ids(splits: Path) -> set[str]:
    out = set()
    for name in ("dev_ids.txt", "golden_ids.txt"):
        for ln in (splits / name).read_text(encoding="utf-8").splitlines():
            if ln.strip():
                out.add(ln.split("\t")[1].strip())
    return out


def select(reports: list[dict], v1: set[str]) -> tuple[list[dict], dict, list[list[str]]]:
    """Candidate reports (not in v1, not near-duplicate of v1) + dedup stats + v2-internal dup groups."""
    from data_prep.render import renderable

    by_id = {r["report_id"]: r for r in reports}
    missing = v1 - set(by_id)
    if missing:
        raise RuntimeError(f"{len(missing)} v1 report ids not in the reports file")
    pool = [r for r in reports if renderable(r) and r["report_id"] not in v1]
    universe = {i: by_id[i] for i in v1} | {r["report_id"]: r for r in pool}
    group, stats = dup_groups(universe)
    tainted = {g for rid, g in group.items() if rid in v1}
    kept = [r for r in pool if group[r["report_id"]] not in tainted]
    members = defaultdict(list)
    for r in kept:
        members[group[r["report_id"]]].append(r["report_id"])
    stats = {"v1_reports": len(v1), "candidate_pool": len(pool), "dropped_v1_dup_group": len(pool) - len(kept),
             "kept": len(kept), "universe_groups_size_gt1": stats["groups_size_gt1"]}
    return kept, stats, [sorted(v) for v in members.values() if len(v) > 1]


def plan(kept: list[dict], n_docs: int = N_DOCS, seed: int = SEED):
    from data_prep.render import plan_documents

    specs = plan_documents(kept, n_docs, seed, n_injected=N_INJECTED)
    return [replace(s, doc_id=f"{DOC_PREFIX}{k + 1:04d}") for k, s in enumerate(specs)]


def write_split(corpus: Path, splits: Path, seed: int, dedup: dict, internal_groups: list[list[str]],
                chosen: set[str]) -> dict:
    manifest = list(csv.DictReader((corpus / "manifest.csv").read_text(encoding="utf-8").splitlines()))
    rows = sorted(manifest, key=lambda r: r["doc_id"])
    ids_path = splits / "golden_v2_ids.txt"
    ids_path.write_text("".join(f"{r['doc_id']}\t{r['report_id']}\n" for r in rows), encoding="utf-8", newline="\n")
    lock = golden_hash(ids_path)
    (splits / "golden_v2.lock").write_text(lock + "\n", encoding="utf-8", newline="\n")
    groups = [sorted(set(g) & chosen) for g in internal_groups]
    groups = sorted((g for g in groups if len(g) > 1), key=lambda v: (-len(v), v[0]))
    (splits / "golden_v2_dedup.json").write_text(json.dumps(
        {"method": "exact + MinHash LSH", "threshold": THRESHOLD, "seed": seed, **dedup,
         "groups": groups}, indent=1) + "\n", encoding="utf-8", newline="\n")
    tally = {t: (sum(1 for r in rows if r["tier"] == t), sum(int(r["n_pages"]) for r in rows if r["tier"] == t))
             for t in TIERS}
    fmts = defaultdict(int)
    for r in rows:
        fmts[r["format"]] += 1
    flag = lambda k: sum(int(r[k]) for r in rows)
    pages = sum(int(r["n_pages"]) for r in rows)
    hw_pages = sum(int(r["n_pages"]) for r in rows if int(r["handwriting"]))
    md = [SECTION_MARKER, ("## Golden v2 (held-out set, `python -m data_prep.heldout`, seed "
          f"{seed})"), "",
          ("A fresh evaluation set built after golden v1 was spent; scored once by the v2 final run "
          "(`python -m radintake.eval.run --split golden_v2 --final`, guard `eval/GOLDEN_V2_USED.json`)."), "",
          (f"- Selection: IU reports not in `dev_ids.txt` / `golden_ids.txt` ({dedup['v1_reports']} v1 reports "
           f"excluded); {dedup['candidate_pool']} renderable candidates."),
          (f"- Near-duplicate check against ALL v1 reports (same method as above: exact + MinHash {NUM_PERM} perms, "
           f"{SHINGLE}-gram shingles, LSH at Jaccard {THRESHOLD}, exact re-check, union-find over v1 + candidates): "
           f"{dedup['dropped_v1_dup_group']} candidates dropped because their dup group touches a v1 report; "
           f"{dedup['kept']} kept. Shared v1/v2 dup groups: 0."),
          (f"- Plan/render/degrade: v1 code and rules, seed {seed}, {len(rows)} docs (ids `G0001..`), "
          f"{N_INJECTED} injection slots; v2-internal dup groups: {len(groups)} (`golden_v2_dedup.json`)."), "",
          "| Tier | Docs | Pages |", "|---|---|---|",
          *(f"| {t} | {tally[t][0]} | {tally[t][1]} |" for t in TIERS),
          f"| **total** | {len(rows)} | {pages} |", "",
          "Formats: " + ", ".join(f"{k} {fmts[k]}" for k in sorted(fmts)) + ".", "",
          (f"Flags: {flag('handwriting')} handwriting docs ({hw_pages} of {pages} pages, "
           f"{100 * hw_pages / max(pages, 1):.0f} %), {flag('injected')} injected docs, "
           f"{flag('multipage')} multi-page docs."), "",
          f"Golden v2 list lock (sha256 of `golden_v2_ids.txt`, LF line endings): `{lock}`.", ""]
    split_md = splits / "SPLIT.md"
    head = split_md.read_text(encoding="utf-8").split(SECTION_MARKER)[0].rstrip("\n") + "\n\n"
    split_md.write_text(head + "\n".join(md), encoding="utf-8", newline="\n")
    return {"docs": len(rows), "pages": pages, "tiers": tally, "formats": dict(fmts),
            "handwriting_docs": flag("handwriting"), "handwriting_pages": hw_pages, "injected": flag("injected"),
            "multipage": flag("multipage"), "lock": lock}


def main(argv: list[str] | None = None) -> int:
    from data_prep.generate import build
    from data_prep.render import render_documents

    ap = argparse.ArgumentParser(prog="python -m data_prep.heldout")
    ap.add_argument("--reports", type=Path, default=Path("data/iu/reports.jsonl"))
    ap.add_argument("--splits", type=Path, default=Path("splits"))
    ap.add_argument("--rendered", type=Path, default=Path("data/generated_v2"))
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus_v2"))
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n-docs", type=int, default=N_DOCS)
    ap.add_argument("--select-only", action="store_true", help="print the selection/dedup stats; render nothing")
    a = ap.parse_args(argv)
    reports = [json.loads(ln) for ln in a.reports.read_text(encoding="utf-8").splitlines() if ln.strip()]
    kept, dedup, internal = select(reports, v1_report_ids(a.splits))
    specs = plan(kept, a.n_docs, a.seed)
    if a.select_only:
        print(json.dumps({"dedup": dedup, "planned_docs": len(specs)}, indent=1))
        return 0
    a.rendered.mkdir(parents=True, exist_ok=True)
    (a.rendered / "plan.json").write_text(json.dumps([asdict(s) for s in specs], indent=1), encoding="utf-8")
    render_documents(specs, a.rendered)
    build(a.rendered, a.corpus, a.seed)
    out = write_split(a.corpus, a.splits, a.seed, dedup, internal, {s.report_id for s in specs})
    print(json.dumps({"dedup": dedup, **out}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
