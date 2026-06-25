# COLD-tier resident RAM — measure before bounding it (slice 5 input)

> Reproduce: `cargo run --release --example cold_ram`.

Slice 3 spills a demoted node's **content** to disk, but its **metadata** (id, label,
archived edges, spill-hash stub) stays resident — an `O(N)` footprint we documented
but never quantified. Before choosing how to bound it (slice 5), measure it.

The benchmark builds N realistic nodes (files + 3 symbols each + `Contains`/`DependsOn`
edges), demotes the whole graph to COLD, spills every content blob to disk, and reports
the stuck resident bytes vs the offloaded disk bytes, cross-checked against process
`VmRSS`.

## Result

| files | cold nodes | resident/node | disk/node | **res : disk** |
|------:|-----------:|--------------:|----------:|---------------:|
| 2000  |      8000  |      494 B    |   179 B   |   **2.76×**    |
| 8000  |     32000  |      496 B    |   179 B   |   **2.77×**    |
| 30000 |    120000  |      500 B    |   179 B   |   **2.78×**    |

(`VmRSS` delta is ≈1300–1700 B/node — the logical figure plus allocator slack and
the live `nodes`/`edges` vectors. The logical 500 B/node is the part slice 5 controls.)

## What it says — and how it settles the lossless-vs-lossy fork

1. **The stuck metadata (~500 B/node) is ~2.8× larger than the content we spilled to
   disk (179 B/node).** For *code* — where a symbol's content is small — **spilling
   content barely dents resident RAM**; the metadata dominates. (Slice 3 is still a
   real win for large blobs and for *disk*-unboundedness, but on a code graph it is
   not where the RAM is.)
2. **~60 % of that metadata is edges.** Each cold entry archives its incident edges
   (~5 here, ~60 B each ≈ 300 B of the 500 B).
3. Therefore:
   - **Lossy edge-contraction (the discussion's idea) is the *wrong* lever here.**
     Contracting a hub of degree *d* replaces *d* edges with up to *d²* bridge edges —
     it would **inflate** the single biggest component of the resident cost on exactly
     the nodes (hubs) that matter. The data argues against it.
   - **The clean win is lossless *full-entry* spill**: archive the edges and metadata
     to the on-disk store too (extending slice 3 from content to the whole `ColdNode`),
     keeping only a minimal `NodeId → handle` resident. That attacks the dominant cost,
     stays non-destructive (“page don’t drop”), and reuses the content-addressed spill
     machinery already built.

## Is it urgent?

At ~500 B/node the COLD metadata reaches **1 GiB at ~2.1 M nodes** (logical), or
~700 K nodes by `VmRSS`. Typical repos (< ~100 K nodes) sit at tens of MB — tolerable.
So slice 5 is a **scaling** fix for very large monorepos or long-running daemons that
accumulate history, not a present bottleneck — and when built, it should be **lossless
full-entry spill**, not lossy contraction.

## Slice 5 result — deep-spill (built; lossless, off by default)

`set_cold_resident_budget(Some(b))` deep-spills the coldest entries: each one's
`label` + full `edges` move to the same content-addressed store as one blob, and only
the neighbour **ids** (`ColdNode::adj`) stay resident — enough for `cold_neighbours`
and region paging to keep working without touching disk. Everything faults back,
hash-verified, on `page_in`. A guard skips any entry that wouldn't shrink (two 64-byte
hash stubs costing more than the bytes they replace), so the budget is approached
best-effort, never by dropping a node.

On the 120 K-node fixture above (content already spilled by slice 3), deep-spill cuts
resident metadata **~11 %**, archiving the 30 K edge-bearing file nodes. The honest
ceiling: the 90 K symbols have their one edge archived under their file (nothing to
shrink), and the remainder is the **irreducible per-entry floor** — the `ColdNode`
struct + id + content-hash stub — which deep-spill leaves resident by design. Cutting
*that* floor (a compact husk replacing the full struct) is the next lever, beyond this
slice. The win scales with edge density: hub- and edge-rich tiers shed far more, still
losslessly, and crucially **without** the bridge-edge blow-up that lossy contraction
would inflict on those same hubs.
