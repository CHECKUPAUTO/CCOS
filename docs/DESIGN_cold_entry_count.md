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
| resident, logical (`cold_resident_bytes`) | **~150–174 B / husk** |
| 1 GiB reached at | **~6–7 M husks** |

> **Methodology correction.** An earlier draft also reported a `VmRSS`-delta of
> ~1 567 B/husk and concluded "~685 K nodes for 1 GiB / a 9× overhead". That figure was
> **wrong**: `examples/cold_count.rs` materializes the *whole* graph (full nodes + their
> content) before deep-spilling, so the process-RSS delta measures that transient **build
> peak**, not the husk steady state. The honest per-husk metric is the logical
> `cold_resident_bytes` (~150–174 B), so the tier reaches 1 GiB at **~6–7 M cold nodes** —
> still worth bounding for very large / long-lived deployments, but *less* urgent than the
> peak-inclusive number implied, and comfortably tens of MB for a typical < 100 K-node
> project.

The husk's variable part is its resident **adjacency** (neighbour ids) — exactly what
`cold_neighbours` reads without disk. The fixed part is the `BTreeMap` node + the
`DeepHusk` stub. Two costs follow: the *bytes* (bounded by the logical figure above) and
the *allocation count* (each husk previously held a `Vec` + a `String` per id — many tiny
allocations, which inflate real RSS and fragment a long-running heap even though the
logical byte count ignores them).

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

**Lever 1 — collapse the per-husk allocation overhead (no disk, no DB). ✅ Built.**
`DeepHusk.adj` is now a single length-prefixed `Box<[u8]>` of concatenated ids
(`pack_adj` / `unpack_adj`) instead of a `Vec<NodeId>`: **one** allocation per husk
instead of `degree + 1`, and `cold_neighbours` decodes it in place. A `serde` adapter
keeps the snapshot a plain array of id strings, so the on-disk form is byte-identical.
This cuts the logical husk ~16 % (174 → ~146 B) and, more importantly, the *allocation
count* — the steady-state allocator-overhead and long-running-heap fragmentation the
build-peak RSS measurement couldn't isolate. It does **not** bound the count.

A stronger variant of Lever 1 is to **intern `NodeId`s** (a shared string table; husks
hold `u32` handles). That cuts both the per-id `String` allocations *and* the bytes
(4-byte handle vs a ~35-byte id), but it touches `NodeId` across the kernel and is a
much wider change.

**Lever 2 — bound the count with an on-disk husk+adjacency index (database-grade).**
The full fix above. Bounds resident COLD to `O(working set)` regardless of `N`, at the
cost of the embedded-index machinery and disk-backed `cold_neighbours`.

## Verdict

The husk **count** is a real `O(N)`, but — once the build-peak measurement error is
corrected — a comfortably slow one: ~150–174 B/husk reaches 1 GiB only at ~6–7 M cold
nodes, tens of MB for a typical project. **Lever 1 (pack the adjacency) is built**: it
shrinks the bytes ~16 % and, more importantly, drops the per-husk allocation count to
one, the steady-state and fragmentation win. **Lever 2 (the on-disk index) remains the
eventual `O(1)`-resident end state**, to be built only when a deployment's scale
genuinely justifies the database-grade complexity (disk I/O, cache eviction, symmetric
adjacency, crash-consistency for `replay == live`). This document pins that design so
the build is a decision, not a discovery.
