"""Split manifest: tuning / validation / calibration / test / adversarial.

Hard constraints enforced by ``verify``:
* a normalized question text appears in at most one split;
* a gold PMID appears in at most one split (no document leakage between tuning and test);
* pinned records (the thesis 24 → ``test``) keep their split.

The manifest is a JSON file keyed by record id so that later code never re-derives splits.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from evaluation.datasets.loaders import QueryRecord, normalize_question

SPLITS = ("tuning", "validation", "calibration", "test", "adversarial")
DEFAULT_FRACTIONS = {"tuning": 0.40, "validation": 0.20, "calibration": 0.15, "test": 0.25}
MANIFEST_DIR = Path(__file__).resolve().parent / "datasets" / "splits"


class SplitError(ValueError):
    pass


@dataclass
class SplitManifest:
    version: str
    seed: int
    assignments: dict[str, str]  # record id -> split
    pinned: dict[str, str] = field(default_factory=dict)
    notes: str = ""

    def split_of(self, record_id: str) -> str | None:
        return self.assignments.get(record_id)

    def ids(self, split: str) -> list[str]:
        return sorted(i for i, s in self.assignments.items() if s == split)

    def counts(self) -> dict[str, int]:
        c: dict[str, int] = defaultdict(int)
        for s in self.assignments.values():
            c[s] += 1
        return dict(sorted(c.items()))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.version,
            "seed": self.seed,
            "notes": self.notes,
            "pinned": dict(sorted(self.pinned.items())),
            "assignments": dict(sorted(self.assignments.items())),
        }
        path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> SplitManifest:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            version=d["version"],
            seed=d["seed"],
            assignments=d["assignments"],
            pinned=d.get("pinned", {}),
            notes=d.get("notes", ""),
        )


def _groups_by_pmid(records: list[QueryRecord]) -> list[list[QueryRecord]]:
    """Union records that share any gold PMID, so a document never straddles two splits."""
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    first_by_pmid: dict[str, int] = {}
    for i, r in enumerate(records):
        find(i)
        for p in r.gold_pmids:
            if p in first_by_pmid:
                union(first_by_pmid[p], i)
            else:
                first_by_pmid[p] = i
    groups: dict[int, list[QueryRecord]] = defaultdict(list)
    for i, r in enumerate(records):
        groups[find(i)].append(r)
    return list(groups.values())


def build_manifest(
    records: Iterable[QueryRecord],
    *,
    seed: int = 20261004,
    fractions: Mapping[str, float] = DEFAULT_FRACTIONS,
    pinned: Mapping[str, str] | None = None,
    version: str = "v1",
    notes: str = "",
) -> SplitManifest:
    """Stratify by dataset × question_type; assign whole PMID-connected groups to one split.

    ``pinned`` maps record ids to a split and is applied first; any record sharing a gold PMID
    or question text with a pinned record goes to the same split.
    """
    records = list(records)
    if abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise SplitError("fractions must sum to 1")
    pinned = dict(pinned or {})
    rng = random.Random(seed)

    by_norm: dict[str, list[QueryRecord]] = defaultdict(list)
    for r in records:
        by_norm[normalize_question(r.question)].append(r)
    groups = _groups_by_pmid(records)

    assignments: dict[str, str] = {}
    # 1. pinned groups
    for g in groups:
        pinned_splits = {pinned[r.id] for r in g if r.id in pinned}
        if len(pinned_splits) > 1:
            raise SplitError(f"conflicting pins inside one PMID group: {pinned_splits}")
        if pinned_splits:
            s = pinned_splits.pop()
            for r in g:
                assignments[r.id] = s
    # 2. remaining groups, stratified by the (dataset, question_type) of their first record
    strata: dict[tuple[str, str], list[list[QueryRecord]]] = defaultdict(list)
    for g in groups:
        if g[0].id in assignments:
            continue
        strata[(g[0].dataset, g[0].question_type)].append(g)
    order = [s for s in SPLITS if s in fractions]
    for key in sorted(strata):
        gs = strata[key]
        rng.shuffle(gs)
        n = len(gs)
        cuts, acc = [], 0.0
        for s in order:
            acc += fractions[s]
            cuts.append(round(acc * n))
        start = 0
        for s, end in zip(order, cuts, strict=True):
            for g in gs[start:end]:
                for r in g:
                    assignments[r.id] = s
            start = end
    m = SplitManifest(
        version=version, seed=seed, assignments=assignments, pinned=pinned, notes=notes
    )
    verify(m, records)
    return m


def verify(manifest: SplitManifest, records: Iterable[QueryRecord]) -> None:
    """Raise SplitError on any text or PMID overlap across splits, or a broken pin."""
    records = list(records)
    by_id = {r.id: r for r in records}
    for rid, s in manifest.pinned.items():
        if manifest.assignments.get(rid) != s:
            actual = manifest.assignments.get(rid)
            raise SplitError(f"pinned record {rid} is in {actual!r}, not {s!r}")
    text_split: dict[str, str] = {}
    pmid_split: dict[str, str] = {}
    for rid, s in manifest.assignments.items():
        r = by_id.get(rid)
        if r is None:
            continue
        if s not in SPLITS:
            raise SplitError(f"unknown split {s!r} for {rid}")
        key = normalize_question(r.question)
        if text_split.setdefault(key, s) != s:
            raise SplitError(f"question text in two splits: {r.question[:60]!r}")
        for p in r.gold_pmids:
            if pmid_split.setdefault(p, s) != s:
                raise SplitError(f"gold PMID {p} in two splits ({pmid_split[p]} and {s})")
