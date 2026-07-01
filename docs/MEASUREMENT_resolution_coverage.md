# Resolution coverage — the call/data-flow resolver across Rust path shapes

> Reproduce: `cargo run --release --example resolution_coverage`

The call graph (`Calls`: fn → fn) and data-flow graph (`DataFlow`: fn → `static`/`const`) grew slice
by slice, each closing one path shape under the same discipline: **resolve-uniquely-or-skip**, so a
wrong edge is never invented. This measurement is the arc's capstone — it enumerates every shape the
resolver now handles (tagged with the slice that added it), asserts the shapes it *deliberately
skips*, and reports the structural yield on CCOS's own `src/`. Two runs are bit-identical.

## What resolves — 10/10 idioms

| shape | example | slice |
|---|---|---|
| crate-rooted call | `crate::m::f()` | call crux |
| imported fn | `use m::f; f()` | Tier A |
| imported module | `use crate::m; m::f()` | Slice 2 |
| **local submodule path, no `use`** | `mod m; m::f()` | **#122** |
| **nested submodule path, no `use`** | `a::b::f()` | **#122** |
| **typed receiver method (incl. `&T`)** | `fn r(x: &T) { x.bar() }` | **#23 + #126** |
| `Self` method | `self.helper()` | #20 |
| bare const (globally unique) | `FOO` | data-flow Slice 1 |
| **import-scoped const** | `use m::MAX; MAX` | **#113** |
| **renamed const import** | `use m::MAX as L; L` | **#124** |

The bolded rows are this arc's additions. Two are worth spelling out:

- **Non-`use` module paths (#122).** `submod::f()` with no import is among the commonest idioms in
  real Rust, and it produced *no edge*. The fix resolves the prefix as a submodule of the **caller's
  own crate only** — module-file-must-exist (exact, no ancestor shortening), then symbol-must-exist.
  An adversarial multi-agent review of the first design *empirically confirmed* that also trying an
  external-crate interpretation minted false edges (a symbol-less local `mod sib` fell through to a
  same-named extern crate — an edge Rust's shadow rule forbids) and stole valid type-method edges
  (an extern crate named like a type). The external reading was excluded; both defects have
  regression tests.
- **Reference-typed receivers (#126).** `fn r(x: &T) { x.bar() }` — the pervasive `&self`-taking
  pattern — inferred no receiver type because `&T` is not a `Type::Path`. Peeling leading `&`/`&mut`
  down to the referent reuses the exact #23 gates (`&dyn Tr`, `&Box<T>`, `&[T]`, generic params all
  still skip), so it adds recall with zero new false-edge surface. *Building this measurement is what
  surfaced the gap* — the first run printed `✗ MISSED` on its own fixture.

## What skips — 3/3 precision holds

| shape | why skipping is correct |
|---|---|
| bare `othercrate::f()` (no `use`) | a local `mod` shadows a same-named extern crate — linking into an unrelated crate can contradict Rust name resolution (the #122 review's confirmed false edge); cross-crate calls resolve via an explicit `use` |
| `f()` with two defs, no import | globally ambiguous — either link is a guess |
| `nope::f()` (unknown module) | no module file — never fall back to a same-named free fn |

## At scale — CCOS's own `src/` (48 files)

```
2474 call references parsed   →  963 fn→fn `Calls` edges resolved (deduped)
  80 const/static references  →   43 `DataFlow` edges resolved (deduped)
```

The `&T` receiver fix alone moved the same corpus from 903 → **963** call edges (+60, ≈ 7 % — the
single largest recall gain of the arc, from a four-line peel). The parsed-but-unresolved remainder is
dominated by calls into `std`/external crates, method chains on non-inferable receivers
(`f().bar()`), and macro paths — all **correctly** unresolved: the graph never asserts a causal edge
it cannot prove, so the structure an agent walks is always trustworthy. This is the representation a
vector RAG index does not have — relatedness can rank `db.rs` near `api.rs`, but only the resolved
edge says *`api::handle` calls `repo::fetch` calls `db::timeout_ms`*, deterministically and
replay-exact.

## Provenance

The arc landed as #113 (import-scoped bare consts) → #122 (non-`use` local module paths; designed and
adversarially reviewed via multi-agent workflows) → #124 (renamed const imports) → #126 (`&T`
receivers + this measurement). Each slice: crux tests first, full-suite regression (543 lib tests at
the arc's close), clippy `-D warnings`, and a bit-identical two-run determinism check.
