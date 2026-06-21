# Thor handoff — confirm generalisation, then test sufficiency (Q7)

## Where we are

The "context assembly" core of CCOS was failing on real code; three measured fixes (the
triad) plus a budget-balancing fix landed and are validated on **CCOS's own src** *and* the
**`syn` crate** (independent):

- **#1 granularity** — no node carries a whole file (region blow-up 15× → ~1×).
- **#2 degree-aware propagation** — a hub distributes pressure, no flood (518 → 37 on CCOS).
- **#3 anchor proximity** — near neighbours outrank distant component noise.
- **budget balancing** — header cap + per-file cap, so deps fit a fixed budget regardless of
  anchor size (`syn` item.rs: 1/7 → **7/7** at budget 2048).

At a size-appropriate budget, `recall around <failing file>` returns the file **and all its
real dependencies** in ~a few % of the cost of dumping the crate. Details +
before/after numbers: `docs/FIELD_CAMPAIGN_H.md`, `docs/DESIGN_symbol_granularity.md`,
`docs/DESIGN_recall_budget.md`.

**Two questions remain — that's your run:**

1. **Generalisation.** Does this hold on a *third* independent repo (not CCOS, not `syn`)?
2. **Sufficiency (the decisive one — paper Phase 4 / Q7).** At an *equal token budget*, does
   the CCOS causal window make your local LLM **resolve** a bug better/cheaper than a naive
   file dump? Coverage and frugality are necessary; *this* is the claim that matters.

## Setup

```sh
cargo build --release            # produces ./target/release/ccos
```

Both helper scripts drive the `ccos mcp` server over stdio; they need only Python 3.

## Campaign I — generalisation (structural, model-free)

Pick **2–3 real Rust crates you have** with a mostly **flat `src/*.rs`** layout (CCOS's
cross-file linking is strongest there; sub-`mod.rs` trees are a separate known limit). Good
picks: `ripgrep`'s `grep-matcher` / `grep-regex`, `bat`'s `src`, `fd`'s `src`, or any crate
in `~/.cargo/registry/src`.

```sh
python3 scripts/ccos_campaign_probe.py <crate>/src --budget 2048 --out corpus_I/<crate>.json
python3 scripts/ccos_campaign_probe.py <crate>/src --budget 8192 --out corpus_I/<crate>-8k.json
```

It prints, per crate: the #1 blow-up factor, and per anchor the #2 flood and #3 coverage
(`deps_in_win`, `window_tok`, `% all-src`, `noise`). **Fill this grid** (one row per anchor):

| crate | anchor | #deps | deps_in_win | window_tok | % all-src | noise | blow-up |
| ----- | ------ | ----- | ----------- | ---------- | --------- | ----- | ------- |

**Expected (honest):** blow-up ~0.5–1.5×; coverage = all deps once the budget fits the
anchor's dep count (a high-fan-out file like `lib.rs` needs more budget — that *is* the
budget-scaling signal); window a few % of all-src; noise small at a tight budget. If a crate
breaks linking (deps never appear even at large budget), that's a **parser finding** — note
the crate and a file, it justifies the sub-`mod.rs` parser work.

## Campaign J — sufficiency / Q7 (the decisive one)

Mine **8–12 real bug-fix commits** (prefer ones where the fix touches a *different* file than
the failing test — multi-file bugs are where a dump misses the cause and CCOS shouldn't).
For each: `git checkout` the tree **before** the fix, confirm `cargo test` is red, note the
**failing file**.

Build the two **equal-budget** contexts:

```sh
cargo test 2>&1 | tee red.txt          # capture the red output
python3 scripts/ccos_resolution_context.py <crate>/src <failing_file>.rs \
    --budget 4096 --cargo-output red.txt --outdir corpus_J/<bug-hash>
# -> context_ccos.txt and context_baseline.txt, both ~4096 tokens
```

`context_ccos.txt` = the causal region around the failing file. `context_baseline.txt` = the
failing file + siblings dumped and truncated to the **same** budget (the naive agent).

Then, for **each** context, with the **same** prompt to your local LLM:

> "Here is the context. The test in `<failing_file>` fails. Return a unified diff that fixes
> the bug."

apply the diff, run `cargo test`, and record. **Fill this grid:**

| bug (hash) | failing file | cause file | same file? | resolved_ccos | resolved_baseline | tokens_ccos | tokens_baseline |
| ---------- | ------------ | ---------- | ---------- | ------------- | ----------------- | ----------- | --------------- |

**The honest hypothesis:** equal *resolution* on single-file bugs (the dump has the cause
too), but a **CCOS win on multi-file bugs** — where the baseline truncates away the cause
file and CCOS's region still carries it. If resolution is *equal everywhere* at equal budget,
that is a real (negative-for-the-thesis) result and we report it. If CCOS resolves multi-file
bugs the dump can't, that is the sufficiency evidence the paper needs.

## Bring back

- `corpus_I/` (the JSON files + your filled Campaign I grid).
- `corpus_J/` (per bug: the two context files, the model's two diffs, the two `cargo test`
  results, and the filled grid). Keep each bug's commit hash in the dir name.

Then `tar czf corpus_handoff.tar.gz corpus_I corpus_J` and send it — we analyse together.
Note your model id and the Jetson power mode (`scripts/jetson_repro_env.sh`) so the numbers
are reproducible.
