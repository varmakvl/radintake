"""Near-duplicate grouping and the dev / frozen golden split (by report, stratified by tier).

    python -m data_prep.split [--corpus data/corpus] [--reports data/iu/reports.jsonl] [--out splits]

De-duplication. IU has many near-identical normal reports (audit: 24% exact duplicates). Two
reports in different splits would leak boilerplate into the golden scores, so:

1. normalise Findings + Impression (lowercase, drop the ``XXXX`` de-id token, digits, punctuation);
2. exact pass: identical normalised text -> same group;
3. near pass: MinHash (128 permutations) over character 5-gram shingles + LSH at Jaccard 0.8;
   every LSH candidate pair is re-checked with the exact Jaccard before it is merged;
4. union-find over both passes -> dup groups. A group never crosses the split.

Split. ``StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=140)`` with groups = dup
groups and y = quality tier; the first fold whose documents hold >= 100 pages is the golden set,
the rest is dev. Outputs (committed - ids only, no report text):

    splits/dev_ids.txt, splits/golden_ids.txt   "<doc_id>\t<report_id>" per line
    splits/dup_groups.json                       groups (size > 1) of corpus report ids
    splits/golden.lock                           sha256 of golden_ids.txt (checked by tests/eval)
    splits/SPLIT.md                              method, seed, thresholds, counts per tier
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

SEED = 140
THRESHOLD = 0.8
NUM_PERM = 128
SHINGLE = 5
_NORM_RE = re.compile(r"[^a-z ]+")


def normalise(r: dict) -> str:
    text = f"{r.get('findings') or ''} {r.get('impression') or ''}".replace("XXXX", " ").lower()
    return " ".join(_NORM_RE.sub(" ", text).split())


def shingles(text: str, k: int = SHINGLE) -> set[str]:
    if len(text) <= k:
        return {text}
    return {text[i:i + k] for i in range(len(text) - k + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a or b else 1.0


class UnionFind:
    def __init__(self, items):
        self.p = {i: i for i in items}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def dup_groups(reports: dict[str, dict], threshold: float = THRESHOLD) -> tuple[dict[str, str], dict]:
    """report_id -> group id (smallest member id); plus stats."""
    from datasketch import MinHash, MinHashLSH

    ids = sorted(reports)
    norm = {i: normalise(reports[i]) for i in ids}
    sh = {i: shingles(norm[i]) for i in ids}
    uf = UnionFind(ids)
    by_text = defaultdict(list)
    for i in ids:
        by_text[norm[i]].append(i)
    exact_pairs = 0
    for members in by_text.values():
        for m in members[1:]:
            uf.union(members[0], m)
            exact_pairs += 1
    lsh = MinHashLSH(threshold=threshold, num_perm=NUM_PERM)
    mh = {}
    for i in ids:
        m = MinHash(num_perm=NUM_PERM, seed=SEED)
        for s in sh[i]:
            m.update(s.encode("utf-8"))
        mh[i] = m
        lsh.insert(i, m)
    cand = verified = 0
    near_scores = []
    for i in ids:
        for j in lsh.query(mh[i]):
            if j <= i:
                continue
            cand += 1
            jac = jaccard(sh[i], sh[j])
            if jac >= threshold:
                verified += 1
                near_scores.append(jac)
                uf.union(i, j)
    group = {i: uf.find(i) for i in ids}
    sizes = Counter(group.values())
    stats = {"reports": len(ids), "exact_pairs": exact_pairs, "lsh_candidates": cand, "verified_near_pairs": verified,
             "groups": len(sizes), "groups_size_gt1": sum(1 for s in sizes.values() if s > 1),
             "reports_in_dup_groups": sum(s for s in sizes.values() if s > 1), "largest_group": max(sizes.values()),
             "near_pairs_jaccard_0.8_0.9": sum(1 for s in near_scores if s < 0.9)}
    return group, stats


def split(manifest: list[dict], group: dict[str, str], seed: int = SEED, min_golden_pages: int = 100):
    from sklearn.model_selection import StratifiedGroupKFold

    y = [m["tier"] for m in manifest]
    g = [group[m["report_id"]] for m in manifest]
    skf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    for _, gold_idx in skf.split(manifest, y, g):
        pages = sum(int(manifest[i]["n_pages"]) for i in gold_idx)
        if pages >= min_golden_pages:
            golden = {manifest[i]["doc_id"] for i in gold_idx}
            return [m for m in manifest if m["doc_id"] not in golden], [m for m in manifest if m["doc_id"] in golden]
    raise RuntimeError("no fold reaches the minimum golden page count")


def _write_ids(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(f"{r['doc_id']}\t{r['report_id']}\n" for r in sorted(rows, key=lambda r: r["doc_id"])),
                    encoding="utf-8")


def golden_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def run(corpus: Path, reports_path: Path, out: Path, seed: int = SEED) -> dict:
    manifest = list(csv.DictReader((corpus / "manifest.csv").read_text(encoding="utf-8").splitlines()))
    allr = {json.loads(line)["report_id"]: json.loads(line)
            for line in reports_path.read_text(encoding="utf-8").splitlines() if line.strip()}
    reps = {m["report_id"]: allr[m["report_id"]] for m in manifest}
    group, stats = dup_groups(reps)
    dev, gold = split(manifest, group, seed)
    out.mkdir(parents=True, exist_ok=True)
    _write_ids(out / "dev_ids.txt", dev)
    _write_ids(out / "golden_ids.txt", gold)
    (out / "golden.lock").write_text(golden_hash(out / "golden_ids.txt") + "\n", encoding="utf-8")
    members = defaultdict(list)
    for rid, gid in group.items():
        members[gid].append(rid)
    groups = sorted((sorted(v) for v in members.values() if len(v) > 1), key=lambda v: (-len(v), v[0]))
    (out / "dup_groups.json").write_text(json.dumps({"method": "exact + MinHash LSH", "threshold": THRESHOLD,
                                                     "groups": groups}, indent=1), encoding="utf-8")

    def tally(rows):
        return {t: (sum(1 for r in rows if r["tier"] == t), sum(int(r["n_pages"]) for r in rows if r["tier"] == t))
                for t in ("clean_scan", "poor_scan", "phone_photo")}

    td, tg = tally(dev), tally(gold)
    flags = lambda rows, k: sum(int(r[k]) for r in rows)
    md = ["# Dataset split", "",
          "Generated by `python -m data_prep.split` (seed 140). Split by report: every page of a document,",
          "and every member of a near-duplicate group, is on one side only.", "",
          "## De-duplication", "",
          "- Normalised text: Findings + Impression, lowercase, `XXXX` / digits / punctuation removed.",
          (f"- Exact pass, then MinHash ({NUM_PERM} permutations, character {SHINGLE}-gram shingles) + LSH at "
          f"Jaccard {THRESHOLD}; each LSH candidate re-checked with the exact Jaccard; union-find over both."),
          *(f"- {k.replace('_', ' ')}: {v}" for k, v in stats.items()), "",
          "## Split (StratifiedGroupKFold, 3 folds, groups = dup groups, y = tier; golden = first fold with >= 100 pages)", "",
          "| Tier | Dev docs | Dev pages | Golden docs | Golden pages |", "|---|---|---|---|---|",
          *(f"| {t} | {td[t][0]} | {td[t][1]} | {tg[t][0]} | {tg[t][1]} |" for t in td),
          f"| **total** | {len(dev)} | {sum(v[1] for v in td.values())} | {len(gold)} | {sum(v[1] for v in tg.values())} |", "",
          (f"Golden flags: {flags(gold, 'handwriting')} handwriting docs, {flags(gold, 'injected')} injected docs, "
          f"{flags(gold, 'multipage')} multi-page docs. Dev: {flags(dev, 'handwriting')} / {flags(dev, 'injected')} / "
          f"{flags(dev, 'multipage')}."), "",
          f"Golden list lock (sha256 of `golden_ids.txt`, LF line endings): `{(out / 'golden.lock').read_text().strip()}`.",
          "The golden set is scored once, by the final evaluation run only; all tuning uses dev.", ""]
    (out / "SPLIT.md").write_text("\n".join(md), encoding="utf-8")
    return {"dedup": stats, "dev": td, "golden": tg}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m data_prep.split")
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    ap.add_argument("--reports", type=Path, default=Path("data/iu/reports.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("splits"))
    a = ap.parse_args(argv)
    print(json.dumps(run(a.corpus, a.reports, a.out), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
