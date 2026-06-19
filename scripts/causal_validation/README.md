# CCOS causal-validation harness

A closed-loop harness that tests CCOS's central claim against the repository's
own Git history, **without an LLM**. It is the empirical counterpart to the
synthetic `ccos experiment` / `ccos eval` benchmarks.

> **Claim under test.** When a fault is injected at one file, CCOS's failure
> propagation pulls the *other files a real fix had to touch* into a bounded
> working set — better than a budget-blind baseline.

## Phases

| Phase | What it does | Status |
| ----- | ------------ | ------ |
| **1 — Mine & inject** | Scan `git log` for fix commits; the changed `.rs` files that already existed at the parent are the ground truth `F_target`; check out the parent in a throwaway worktree; build a CCOS snapshot; inject a fault at the highest-out-degree changed file `n_fail` (`ccos failure … --json`). | ✅ `validate.py` |
| **2 — Coverage** | For each node budget `K`, score `R_cov = \|F_target ∩ WorkingSet_K\| / \|F_target\|`; report the arithmetic and **geometric** mean over scenarios. | ✅ `validate.py` |
| **3 — Optimise weights** | Treat the 5 scoring/decay knobs (now `CCOS_W_*` / `CCOS_FAILURE_DECAY`) as hyperparameters; maximise geo-mean `R_cov` under **leave-one-commit-out** cross-validation. | ⏳ planned (wraps this module) |
| **4 — End-to-end vs RAG** | Real LLM, real patch-and-test loop; resolution rate, tokens, iterations. | ⏳ planned (needs a model — e.g. local Ollama on the Jetson) |

Phases 3–4 deliberately reuse Phase 1–2: the weights flow through the
environment, so an optimiser can wrap `validate.py` unchanged.

## Requirements

Standard-library Python 3.9+ and a built `ccos` binary. No third-party packages
(Phase 3 will add `optuna`/`scipy`, optional).

## Usage

```bash
# Validate the pipeline on a single commit (verbose, no aggregation):
python scripts/causal_validation/validate.py --dry-run

# Score a batch and write per-scenario JSONL:
python scripts/causal_validation/validate.py --limit 25 --k 20 50 100 \
    --out scripts/causal_validation/results.jsonl

# Point at a larger external repo (recommended — see caveat below):
python scripts/causal_validation/validate.py --repo /path/to/some-rust-project \
    --subdir src --limit 50
```

Key flags: `--repo` (target repo, default = this one), `--subdir` (default
`src`), `--k` (budgets), `--depth` (propagation depth), `--keywords`,
`--ccos-bin` (skip the release build), and `--w-failure …` / `--failure-decay …`
to override weights for a single run.

## How a scenario is scored

```
fix commit N ──diff──> F_target = changed .rs files present at N-1
     │
     └─ checkout N-1 (worktree) ─> ccos analyze ─> snapshot (file:<path> nodes)
            │
            └─ n_fail = changed file with max out-degree
                   └─ ccos failure snap n_fail --max-nodes K --json
                          └─ WorkingSet_K ──> R_cov = |F_target ∩ WS_K| / |F_target|
```

Robustness: every subprocess is timed out and captured; a scenario that fails to
analyze, or whose changed files aren't graph nodes, is skipped (not fatal);
worktrees are always cleaned up.

## First findings — and the improvement they drove

Run against **this** repository (a young prototype), the harness finds only a
handful of qualifying commits. The first run exposed a real limitation and the
fix for it is itself measured here — the loop earning its keep.

**Baseline (downstream-only propagation):** `R_cov ≈ 0.30`, **flat across K** —
in every scenario only the seed file is recovered (`R_cov = 1/|F_target|`). The
cause was structural: co-changed files are typically *upstream importers* of the
fault node, linked only through `dep:` hubs, whereas `propagate_failure` walked
`source → target` only, so the pressure never reached them.

**Fix:** (a) `ccos analyze` now resolves intra-crate imports into `file→file`
edges (`link_module_imports`), and (b) `ccos failure --bidirectional` propagates
pressure in both directions (`propagate_failure_bidirectional`). Re-running with
`--bidirectional`:

| K | downstream-only | bidirectional |
| --- | --- | --- |
| 20  | 0.333 | 0.278 |
| 50  | 0.333 | **0.444** |
| 100 | 0.333 | **0.611** (33% fully covered) |

So bidirectional propagation roughly **doubles coverage at a moderate/large
budget**, at the cost of **diluting** it at a very tight one (`K=20`): marking
neighbours on both sides floods a small working set and can evict a target. A
real, falsifiable trade-off — exactly what the harness is for.

**Caveat, unchanged:** `n ≈ 3` is far too small to conclude anything. This
methodology is SWE-bench-shaped; it only becomes meaningful on a large, mature
codebase with many fix commits. Point `--repo` at one and re-run with and without
`--bidirectional`.
