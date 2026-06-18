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

## First findings (and caveats)

Run against **this** repository (a young prototype) the harness finds only a
handful of qualifying commits, and reports `R_cov ≈ 0.30` (geometric mean),
**flat across K** — in every scenario only the seed file is recovered.

That is a genuine, non-trivial result, not a rigged one:

* **It is honest.** Coverage equals `1/|F_target|` every time — CCOS recovers
  the fault file but **none** of its co-changed siblings within budget.
* **It localises a real limitation.** Co-changed files are typically *upstream
  importers* linked only through dependency hubs (`dep:crate`), whereas failure
  pressure currently flows **downstream only** (`propagate_failure` walks
  `source → target`). So the metric is measuring a direction mismatch, and it
  hands Phase 3 a concrete hypothesis to test: propagate (or also propagate)
  *upstream* toward causes — the direction `ccos blame`'s `source_set` already
  walks.
* **The sample is far too small to conclude anything** (`n ≈ 3`). This
  methodology is SWE-bench-shaped: it only becomes meaningful on a large, mature
  codebase with many real fix commits. Point `--repo` at one.

In other words: the loop works, the measurement is falsifiable, and it already
earns its keep by surfacing a testable improvement instead of a vanity number.
