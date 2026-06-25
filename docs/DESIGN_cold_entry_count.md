# Slice 5c — bounding the COLD *entry count* (design + measured verdict)

> Reproduce: `cargo run --release --example cold_count`.

Slices 5/5b/[u8;32] bound the per-entry *size* of the COLD tier: a demoted node is a
compact `DeepHusk` (a 40-byte body-blob stub + the neighbour ids). What remains `O(N)`
is the **count** — one resident husk per node ever demoted. This is the last resident
term that grows without bound. Slice 5c asks: can we bound the count too, and is it
worth it?

## Measurement

120 000 nodes, every entry deep-spilled to a husk:

| metric | value |
|---|---|
| resident, logical (`cold_resident_bytes`) | **174 B / husk** |
| resident, actual (`VmRSS` delta) | **~1 567 B / husk** |
| 1 GiB reached at | ~6.2 M husks (logical) / **~685 K husks (VmRSS)** |

Two things stand out:

1. **The actual cost is ~9× the logical.** A husk is logically 174 B, but each one
   costs ~1.5 KB of real RSS. The gap is **allocation overhead**: every husk is a
   `BTreeMap` node holding a `DeepHusk` whose `adj: Vec<NodeId>` is a heap `Vec` of
   heap `String`s — roughly `degree + 2` small allocations per husk, each with its own
   allocator header, alignment slack and fragmentation. The logical byte count ignores
   all of that; RSS doesn't.
2. **It bites earlier than the husk size suggests.** At ~1.5 KB/husk the tier alone
   reaches 1 GiB at **~685 K cold nodes** — reachable by a large monorepo or a
   long-running daemon that accumulates history. So the count *is* worth bounding for
   large/long-lived deployments (though not for a typical < 100 K-node project, which
   sits at tens of MB).

## Why bounding the count is hard — the `cold_neighbours` tension

`cold_neighbours(id)` (region paging) must find every cold node adjacent to `id`. It
reads each husk's resident `adj`. To answer it **without a resident entry per node**,
the adjacency must live somewhere queryable:

- Keep it **resident** → `Ω(N · degree)` RAM. That is exactly the variable part the
  measurement shows dominates the husk — so keeping it resident *is* the `O(N)` we are
  trying to remove. No win.
- Move it **to disk** → `cold_neighbours` must read the adjacency from disk. A naïve
  per-call full scan is `O(N)` disk I/O. Answering it in `O(1)` disk reads needs a
  **bidirectional, node-keyed on-disk adjacency index** (so `cold_neighbours(id)` is one
  lookup), which means maintaining symmetric adjacency under demotion/page-in with
  on-disk read-modify-writes.

In other words, a truly `O(1)`-resident COLD tier is an **embedded on-disk index**
(B-tree / LSM for `NodeId → {body stub, adjacency}`, with only a bounded working-set
cache resident). That is database-grade: correct disk I/O, a cache eviction policy,
symmetric-adjacency maintenance, crash-consistency to keep `replay == live`. It is the
right end state, but a large, high-risk undertaking that should not be built without an
explicit decision.

## Two levers (smallest first)

**Lever 1 — collapse the per-husk allocation overhead (no disk, no DB).** Replace
`adj: Vec<NodeId>` with a single length-prefixed `Box<[u8]>` of concatenated ids: **one**
allocation per husk instead of `degree + 1`, and `cold_neighbours` decodes it in place.
This attacks the measured 9× gap directly — pulling the ~1.5 KB actual toward the 174 B
logical — and is contained (husk format + `cold_neighbours` + the enforce loop + serde),
verifiable by the existing hardening/round-trip suites plus a resident-RSS assertion. It
does **not** bound the count, but it cuts the constant by a large factor, pushing the
1-GiB threshold from ~685 K toward a few million nodes.

A stronger variant of Lever 1 is to **intern `NodeId`s** (a shared string table; husks
hold `u32` handles). That cuts both the per-id `String` allocations *and* the bytes
(4-byte handle vs a ~35-byte id), but it touches `NodeId` across the kernel and is a
much wider change.

**Lever 2 — bound the count with an on-disk husk+adjacency index (database-grade).**
The full fix above. Bounds resident COLD to `O(working set)` regardless of `N`, at the
cost of the embedded-index machinery and disk-backed `cold_neighbours`.

## Verdict

The husk **count** is a real `O(N)` that reaches 1 GiB at ~685 K cold nodes — worth
addressing for very large or long-running deployments, but not urgent for typical
projects. The **measured-dominant cost is allocation overhead, not algorithmic** — so
**Lever 1 (pack the adjacency) is the high-return, low-risk next step**, and **Lever 2
(the on-disk index) is the eventual `O(1)`-resident end state**, to be built only when a
deployment's scale justifies the database-grade complexity. This document pins the
design so that build is a decision, not a discovery.
