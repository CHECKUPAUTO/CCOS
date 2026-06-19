#!/usr/bin/env python3
"""Causal-validation harness for CCOS — Phases 1 & 2 (offline, no LLM).

This script grounds CCOS's central claim — *that failure pressure propagated
over the causal graph pulls the files a fix must touch into a bounded working
set* — in the repository's own Git history, and measures it quantitatively.

For every bug-fix commit ``N`` it reconstructs the pre-fix world and asks
whether CCOS, starting from one changed file as the fault symptom, recovers the
rest of the fix within a node budget ``K``:

  Phase 1 — DATA MINING & FAULT INJECTION
    * scan ``git log`` for fixes (keywords: fix, bug, crash, issue, …);
    * the files the fix changed (and that already existed at the parent) are the
      ground truth ``F_target``;
    * check out the parent commit ``N-1`` in a throwaway worktree and build a
      CCOS snapshot of it (``ccos analyze``);
    * choose the highest-out-degree changed file as the fault root ``n_fail``
      and inject it (``ccos failure … --max-nodes K --json``).

  Phase 2 — CONSTRAINT-COVERAGE RATIO
    For each budget ``K`` the surviving WorkingSet_K is scored with

        R_cov = | F_target ∩ WorkingSet_K | / | F_target |

    and every scenario is logged as one JSON line (commit, parent, n_fail, K,
    R_cov, sizes, weights). The aggregate (arithmetic and *geometric* mean — the
    objective Phase 3 will optimise) is printed as a table.

Design: standard library only; every subprocess call is timed out and its
stdout/stderr captured; ``--dry-run`` validates the pipeline on a single commit.
The scoring weights are passed through the environment (``CCOS_W_*`` /
``CCOS_FAILURE_DECAY``), so Phase 3 can wrap this module in an optimiser without
touching it.

Usage::

    python scripts/causal_validation/validate.py --dry-run
    python scripts/causal_validation/validate.py --limit 25 --k 20 50 100 \
        --out scripts/causal_validation/results.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# Commit-subject substrings that mark a fix (case-insensitive).
DEFAULT_KEYWORDS = ["fix", "bug", "crash", "issue", "regression", "panic", "hotfix"]


# --------------------------------------------------------------------------- #
# Small subprocess helpers (timed out, output captured).
# --------------------------------------------------------------------------- #
class StepError(RuntimeError):
    """A subprocess returned non-zero or timed out; the scenario is skipped."""


def git(repo: Path, *args: str, timeout: float = 60.0) -> str:
    """Run ``git -C repo <args>`` and return stdout (raises on failure)."""
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise StepError(f"git {' '.join(args)}: {proc.stderr.strip()}")
    return proc.stdout


def run_ccos(
    ccos_bin: str, args: list[str], cwd: Path | None, env: dict | None, timeout: float
) -> subprocess.CompletedProcess:
    """Invoke the CCOS binary; caller inspects returncode/stdout."""
    return subprocess.run(
        [ccos_bin, *args],
        cwd=str(cwd) if cwd else None,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# --------------------------------------------------------------------------- #
# Phase 1 — mine fix commits into scenarios.
# --------------------------------------------------------------------------- #
@dataclass
class Scenario:
    commit: str
    parent: str
    subject: str
    target_files: list[str]  # repo-relative .rs paths changed by the fix


def is_fix(subject: str, keywords: list[str]) -> bool:
    low = subject.lower()
    return any(k in low for k in keywords)


def file_exists_at(repo: Path, rev: str, path: str) -> bool:
    """True if ``path`` exists in tree ``rev`` (so it is a node at the parent)."""
    proc = subprocess.run(
        ["git", "-C", str(repo), "cat-file", "-e", f"{rev}:{path}"],
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0


def mine_scenarios(
    repo: Path, subdir: str, keywords: list[str], limit: int, scan: int
) -> list[Scenario]:
    """Return up to ``limit`` fix scenarios with a non-empty ``F_target``."""
    grep_args = []
    for k in keywords:
        grep_args += ["--grep", k]
    # %x1f = unit separator, a safe field delimiter for arbitrary subjects.
    log = git(
        repo,
        "log",
        "--no-merges",
        "-i",
        *grep_args,
        "--pretty=format:%H%x1f%P%x1f%s",
        "-n",
        str(scan),
    )
    scenarios: list[Scenario] = []
    for line in log.splitlines():
        if not line.strip():
            continue
        commit, parents, subject = (line.split("\x1f", 2) + ["", ""])[:3]
        if not is_fix(subject, keywords):
            continue
        parent = parents.split()[0] if parents.strip() else ""
        if not parent:
            continue  # root commit — nothing to diff against
        try:
            diff = git(repo, "diff", "--name-only", parent, commit, "--", subdir)
        except StepError:
            continue
        targets = [
            p
            for p in diff.splitlines()
            if p.endswith(".rs") and file_exists_at(repo, parent, p)
        ]
        if not targets:
            continue
        scenarios.append(Scenario(commit, parent, subject.strip(), sorted(targets)))
        if len(scenarios) >= limit:
            break
    return scenarios


# --------------------------------------------------------------------------- #
# Worktrees (isolated pre-fix checkouts).
# --------------------------------------------------------------------------- #
def add_worktree(repo: Path, rev: str) -> Path:
    wt = Path(tempfile.mkdtemp(prefix="ccos-wt-"))
    git(repo, "worktree", "add", "--detach", "--force", str(wt), rev, timeout=120.0)
    return wt


def remove_worktree(repo: Path, wt: Path) -> None:
    try:
        git(repo, "worktree", "remove", "--force", str(wt), timeout=60.0)
    except StepError:
        pass
    # Belt and braces: drop the dir and prune the registry.
    subprocess.run(["rm", "-rf", str(wt)], capture_output=True)
    subprocess.run(
        ["git", "-C", str(repo), "worktree", "prune"], capture_output=True
    )


# --------------------------------------------------------------------------- #
# CCOS interaction.
# --------------------------------------------------------------------------- #
def analyze_snapshot(
    ccos_bin: str, worktree: Path, subdir: str, out: Path, cap: int, env: dict, timeout: float
) -> None:
    """Build a CCOS snapshot of ``worktree/subdir`` at ``out`` (ids ``file:<subdir>/…``)."""
    proc = run_ccos(
        ccos_bin,
        ["analyze", subdir, "--max-nodes", str(cap), "--out", str(out)],
        cwd=worktree,
        env=env,
        timeout=timeout,
    )
    if proc.returncode != 0 or not out.exists():
        raise StepError(f"analyze failed: {proc.stderr.strip()[:200]}")


def load_graph(snap: Path) -> tuple[set[str], Counter]:
    """Return (node-id set, out-degree counter) from a kernel snapshot JSON."""
    data = json.loads(snap.read_text())
    graph = data["graph"]
    nodes = set(graph["nodes"].keys())
    outdeg: Counter = Counter(e["source"] for e in graph["edges"])
    return nodes, outdeg


def pick_origin(target_ids: list[str], nodes: set[str], outdeg: Counter) -> str | None:
    """Choose the present changed file with the most outgoing causal edges."""
    present = [nid for nid in target_ids if nid in nodes]
    if not present:
        return None
    # Highest out-degree, ties broken by id for determinism.
    return max(sorted(present), key=lambda nid: outdeg.get(nid, 0))


def failure_working_set(
    ccos_bin: str,
    snap: Path,
    origin: str,
    k: int,
    depth: int,
    env: dict,
    timeout: float,
    bidirectional: bool = False,
) -> dict:
    args = ["failure", str(snap), origin, "--depth", str(depth), "--max-nodes", str(k), "--json"]
    if bidirectional:
        args.append("--bidirectional")
    proc = run_ccos(ccos_bin, args, cwd=None, env=env, timeout=timeout)
    if proc.returncode != 0:
        raise StepError(f"failure failed: {proc.stderr.strip()[:200]}")
    return json.loads(proc.stdout)


# --------------------------------------------------------------------------- #
# Phase 2 — the coverage metric.
# --------------------------------------------------------------------------- #
def r_cov(target_ids: list[str], working_set: list[str]) -> float:
    targets = set(target_ids)
    if not targets:
        return 0.0
    return len(targets & set(working_set)) / len(targets)


def geomean(xs: list[float], eps: float = 1e-6) -> float:
    if not xs:
        return 0.0
    return math.exp(sum(math.log(max(x, eps)) for x in xs) / len(xs))


# --------------------------------------------------------------------------- #
# Orchestrator.
# --------------------------------------------------------------------------- #
@dataclass
class Args:
    repo: Path
    subdir: str
    keywords: list[str]
    limit: int
    ks: list[int]
    depth: int
    cap: int
    timeout: float
    ccos_bin: str
    out: Path | None
    dry_run: bool
    bidirectional: bool = False
    weights: dict = field(default_factory=dict)


def ensure_binary(repo: Path, ccos_bin: str | None, no_build: bool) -> str:
    if ccos_bin:
        return ccos_bin
    candidate = repo / "target" / "release" / "ccos"
    if candidate.exists() or no_build:
        return str(candidate)
    print("[build] cargo build --release …", file=sys.stderr)
    proc = subprocess.run(
        ["cargo", "build", "--release", "--quiet"], cwd=str(repo), text=True
    )
    if proc.returncode != 0 or not candidate.exists():
        raise SystemExit("ccos: could not build release binary; pass --ccos-bin")
    return str(candidate)


def child_env(weights: dict) -> dict:
    env = dict(os.environ)
    keymap = {
        "w_base": "CCOS_W_BASE",
        "w_failure": "CCOS_W_FAILURE",
        "w_recency": "CCOS_W_RECENCY",
        "w_access": "CCOS_W_ACCESS",
        "failure_decay": "CCOS_FAILURE_DECAY",
    }
    for k, v in weights.items():
        if k in keymap:
            env[keymap[k]] = str(v)
    return env


def run(args: Args) -> list[dict]:
    repo = args.repo
    env = child_env(args.weights)
    print(f"[mine] scanning {repo} for fix commits …", file=sys.stderr)
    scenarios = mine_scenarios(
        repo, args.subdir, args.keywords, args.limit, scan=max(args.limit * 6, 60)
    )
    print(f"[mine] {len(scenarios)} scenario(s) with non-empty F_target", file=sys.stderr)

    records: list[dict] = []
    for i, sc in enumerate(scenarios, 1):
        target_ids = [f"file:{p}" for p in sc.target_files]
        tag = f"{sc.commit[:8]} ({i}/{len(scenarios)})"
        wt = None
        try:
            wt = add_worktree(repo, sc.parent)
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
                snap = Path(tf.name)
            analyze_snapshot(args.ccos_bin, wt, args.subdir, snap, args.cap, env, args.timeout)
            nodes, outdeg = load_graph(snap)
            origin = pick_origin(target_ids, nodes, outdeg)
            if origin is None:
                print(f"[skip] {tag}: no changed file is a graph node", file=sys.stderr)
                continue
            present = sum(1 for t in target_ids if t in nodes)
            for k in args.ks:
                res = failure_working_set(
                    args.ccos_bin,
                    snap,
                    origin,
                    k,
                    args.depth,
                    env,
                    args.timeout,
                    bidirectional=args.bidirectional,
                )
                cov = r_cov(target_ids, res["working_set"])
                rec = {
                    "commit": sc.commit,
                    "parent": sc.parent,
                    "subject": sc.subject,
                    "n_fail": origin,
                    "target_files": sc.target_files,
                    "targets_present": present,
                    "k": k,
                    "depth": args.depth,
                    "r_cov": round(cov, 4),
                    "working_set_size": res["working_set_size"],
                    "nodes_before": res["nodes_before"],
                    "weights": res["weights"],
                }
                records.append(rec)
                if args.dry_run:
                    print(json.dumps(rec, indent=2))
            print(
                f"[ok]   {tag}: |F|={len(target_ids)} present={present} "
                f"origin={origin} "
                + " ".join(
                    f"R_cov@{r['k']}={r['r_cov']}" for r in records if r["commit"] == sc.commit
                ),
                file=sys.stderr,
            )
            snap.unlink(missing_ok=True)
        except (StepError, subprocess.TimeoutExpired, json.JSONDecodeError) as e:
            print(f"[skip] {tag}: {e}", file=sys.stderr)
        finally:
            if wt is not None:
                remove_worktree(repo, wt)
    return records


def summarise(records: list[dict], ks: list[int]) -> None:
    print("\n=== Phase 2 — Constraint-Coverage Ratio (CCOS) ===")
    n_scen = len({r["commit"] for r in records})
    print(f"scenarios scored: {n_scen}   measurements: {len(records)}\n")
    print(f"  {'K':>6}  {'n':>4}  {'mean R_cov':>11}  {'geo R_cov':>10}  {'perfect':>8}")
    for k in ks:
        rs = [r["r_cov"] for r in records if r["k"] == k]
        if not rs:
            continue
        amean = sum(rs) / len(rs)
        gmean = geomean(rs)
        perfect = sum(1 for x in rs if x >= 0.999) / len(rs)
        print(f"  {k:>6}  {len(rs):>4}  {amean:>11.3f}  {gmean:>10.3f}  {perfect:>7.0%}")
    print("\n(geo R_cov is the Phase-3 optimisation objective.)")


def parse_args(argv: list[str]) -> Args:
    here = Path(__file__).resolve()
    default_repo = here.parents[2]  # scripts/causal_validation/validate.py -> repo root
    p = argparse.ArgumentParser(description="CCOS causal-validation harness (Phases 1-2).")
    p.add_argument("--repo", type=Path, default=default_repo, help="target Git repo")
    p.add_argument("--subdir", default="src", help="source subdir to analyze (default: src)")
    p.add_argument("--keywords", nargs="+", default=DEFAULT_KEYWORDS)
    p.add_argument("--limit", type=int, default=20, help="max scenarios to score")
    p.add_argument("--k", dest="ks", nargs="+", type=int, default=[20, 50, 100])
    p.add_argument("--depth", type=int, default=3, help="failure propagation depth")
    p.add_argument("--cap", type=int, default=100000, help="analyze --max-nodes (capture full graph)")
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--ccos-bin", default=None, help="path to ccos binary (default: build release)")
    p.add_argument("--no-build", action="store_true")
    p.add_argument("--out", type=Path, default=None, help="write per-scenario JSONL here")
    p.add_argument("--dry-run", action="store_true", help="process a single commit, verbosely")
    p.add_argument(
        "--bidirectional",
        action="store_true",
        help="propagate failure in both edge directions (reach upstream causes)",
    )
    # Optional weight overrides (also useful for a Phase-3 wrapper).
    for name in ("w_base", "w_failure", "w_recency", "w_access", "failure_decay"):
        p.add_argument(f"--{name.replace('_', '-')}", type=float, default=None)
    ns = p.parse_args(argv)
    weights = {
        n: getattr(ns, n)
        for n in ("w_base", "w_failure", "w_recency", "w_access", "failure_decay")
        if getattr(ns, n) is not None
    }
    ccos_bin = ensure_binary(ns.repo, ns.ccos_bin, ns.no_build)
    return Args(
        repo=ns.repo,
        subdir=ns.subdir,
        keywords=ns.keywords,
        limit=1 if ns.dry_run else ns.limit,
        ks=ns.ks,
        depth=ns.depth,
        cap=ns.cap,
        timeout=ns.timeout,
        ccos_bin=ccos_bin,
        out=ns.out,
        dry_run=ns.dry_run,
        bidirectional=ns.bidirectional,
        weights=weights,
    )


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    records = run(args)
    if not records:
        print("no scenarios scored (try --limit higher or widen --keywords)", file=sys.stderr)
        return 1
    summarise(records, args.ks)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"\n[out] {len(records)} records -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
