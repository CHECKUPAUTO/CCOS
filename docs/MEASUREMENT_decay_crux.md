# Measurement — Decay crux (knowledge half-life)

*Why a stale, never-reaffirmed dissent should stop deadlocking a claim.*

Reproduce: `cargo run --release --example decay_crux`

Slice 2 of the Q-Page contested-knowledge work, building on `qbelief` (slice 1). `MemoryGraph::qbelief`
weighs every assertion **equally and forever**, so a single old objection that was never revisited
keeps a claim looking *eternally contested* even after fresh evidence has arrived on the other side.
`MemoryGraph::qbelief_decayed(claim, half_life)` fades each evidence edge by `0.5^(age / half_life)`
— `age` is the clock ticks since the edge was asserted — so a fresh (re-)assertion outweighs an
ageing one. It is **lazy and pure** (computed on demand from each edge's `created_at` vs the current
`clock`, no stored decay state), so it is deterministic and `replay == live` holds.

## Fixture

A claim is **contested at `t = 0`** (one `Supports`, one `Contradicts` edge). The support is then
re-affirmed by a *fresh* assertion at time `T`, while the contradiction is **never revisited** — a
one-off objection that turned out to be wrong. We sweep `T` (the objection's age) at `half_life = 10`
ticks and compare plain vs decayed belief.

## Result (measured)

```
    T      PLAIN qbelief         DECAYED qbelief
         belief  conflict      belief  conflict   (objection age = T)
    0      0.50     1.00       0.50     1.00
    5      0.50     1.00       0.54     0.83
   10      0.50     1.00       0.57     0.67
   20      0.50     1.00       0.62     0.40
   40      0.50     1.00       0.65     0.12
   80      0.50     1.00       0.67     0.01
```

## Reading

**Plain `qbelief` is frozen.** One un-reaffirmed objection counts as much as the fresh support
forever, so the claim reads as a *permanent deadlock* — `conflict 1.00`, `belief 0.50` — no matter
how stale the objection is. An agent relying on it would keep flagging this claim for resolution
indefinitely, even though the only thing keeping it contested is a single ancient objection nobody
stood behind.

**With decay, the claim resolves on its own.** The objection's weight halves every 10 ticks, so as
its age `T` grows the fresh support wins: `conflict` collapses `1.00 → 0.01` and `belief` climbs
`0.50 → 0.67`. This is the **knowledge half-life** — recent evidence outweighs stale, unrefreshed
evidence, so memory is not held hostage by old objections. Reinforcement is just re-assertion: a
fresh edge restores full weight.

Note that `conflict` moves here **only because the two surfaces age differently** (the support is
fresh, the contradiction is old). `conflict` is a scale-free *balance*, so equally-aged evidence
would leave it unchanged — decay relaxes `belief` toward the prior (the magnitudes shrink) and tips
`conflict` only when one side is fresher than the other. The measurement isolates exactly that
asymmetry, which is where decay earns its keep.

## What this does and does not claim

- **Does:** show that an un-decayed belief over an append-only evidence log cannot forget — a stale
  minority objection deadlocks a claim forever — and that an `0.5^(age/half_life)` decay, computed
  lazily and deterministically from edge timestamps, lets fresh evidence resolve it without
  mutating or deleting history (the `Contradicts` edge is still in the log and the audit chain; only
  its *current weight* fades).
- **Does not:** prescribe a half-life. The right `half_life` is domain-dependent (code/logic: long;
  prices/logs: short) and is a caller parameter, not a fixed constant. Choosing or learning it per
  evidence class, and decaying on the *retrieval* path, are follow-ups.
