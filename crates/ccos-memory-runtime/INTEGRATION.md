# Automatic integration — `ccos-memory-runtime` → CCOS

> **For an automated integrator (Claude Opus 4.8) working inside the CCOS repo.**
> Follow this file top to bottom. It is self-sufficient: the crate to integrate
> **is this package** — you place it, wire two manifest edits, verify, and open a
> draft PR. Do **not** redesign the crate.

## What this crate is

A **zero-dependency** cognitive-memory middleware:

- `traits::MemoryProvider` — the neutral backend interface.
- `telemetry::{MemoryState, MemoryBackend, CompressionResult}` — the neutral
  vocabulary the runtime + monitors share.
- `AlignedMemoryPage` / `AlignedPayload` — a 128-byte, 128-aligned opaque page.
- `backend::slha_adapter::SlhaAdapter` — a **self-contained** SLHAv2 tile backend.

Lifecycle owned by the runtime: **HOT** (latent + residual) → **WARM** (residual
removed, latent kept, 96 B) → **COLD** (paged out; persistence external to CCOS).

## Core principle — no `scirust` dependency, anywhere

`scirust` (the SLHAv2 kernel) is a **development toolbox**, not a runtime
dependency. Where we need a piece of it — here, the 128-byte tile ABI and its
grouped-INT4 decode — we **vendor** (copy) that minimal logic into this crate.
Result: **CCOS works without scirust, and this SLHA option works without
scirust.** Verified: `cargo tree` prints only `ccos-memory-runtime`.

## Steps

1. **Place the crate.** Copy this entire package into the CCOS repo at
   `crates/ccos-memory-runtime/` (i.e. `Cargo.toml`, `src/`, `tests/`, and this
   file may travel with it). Keep the files byte-for-byte.

2. **Make CCOS a Cargo workspace.** In the **root** `Cargo.toml`, add:
   ```toml
   [workspace]
   members = ["crates/ccos-memory-runtime"]
   ```

3. **Wire SLHAv2 as an OPT-IN feature of the `ccos` crate.** In the **root**
   `Cargo.toml`:
   ```toml
   # under [dependencies]
   ccos-memory-runtime = { path = "crates/ccos-memory-runtime", optional = true }

   # under [features]
   slhav2 = ["dep:ccos-memory-runtime"]
   ```
   Do **not** add `slhav2` to `default` (keep `default = ["syn-parser"]`).

## Hard rules

- Do **not** modify any existing CCOS `.rs` file. Only add this crate and edit
  the root `Cargo.toml`.
- The default CCOS build stays behaviorally unchanged.
- No `scirust` in any configuration (including `--features slhav2`).

## Verify (all must pass)

```bash
cargo build                                   # default: unchanged
cargo build --features slhav2                 # pulls only the zero-dep runtime
cargo tree --features slhav2 | grep -i scirust && echo "LEAK — FAIL" || echo "no scirust anywhere: OK"
cargo test -p ccos-memory-runtime             # 5/5 pass
cargo clippy -p ccos-memory-runtime --all-targets -- -D warnings
cargo fmt && cargo fmt --check
git status                                    # only: crates/ccos-memory-runtime/** + root Cargo.toml
```

## Deliver

Commit on a feature branch and open a **draft** PR:

> **Title:** `feat: SLHAv2 memory backend as opt-in `slhav2` feature (zero-dep ccos-memory-runtime)`
>
> **Body:** Default build unchanged. `slhav2` is opt-in and pulls **no** scirust
> (verified via `cargo tree`). WARM footprint = 96 B. The adapter is
> self-contained and exposes no backend-specific type in its public API.

## Design notes (settled — keep as-is)

- **Zero dependencies.** The grouped-INT4 decode is vendored from scirust's tile
  ABI; scirust itself is never pulled in.
- **WARM = 96 B** (latent 64 + metadata 32; only the 32-byte residual is freed).
  Not 64 — 64 is the latent size, not the WARM footprint.
- **Restore is approximate** (latent-only; the residual is gone after WARM).
  `RestoreMode::Exact` is a reserved forward-compat variant.
- **Page lookup is O(n)** over a `Vec` (constraint: no HashMap). Fine at
  page-cache scale.
- **Zero `unsafe`** (`#![forbid(unsafe_code)]`); `cpu_cycles` is an elapsed-ns
  proxy — wire a real TSC/PMU counter later if true cycles are needed.
- **Feature name:** `slhav2`. If you prefer to avoid confusion with the existing
  runtime license flag `Feature::SlhAv2Embeddings`, rename the Cargo feature to
  `slha-backend` consistently (manifest only).

## Optional future extension (do NOT enable now)

If exact kernel parity (bit-identical dequant, NF4 codebook, SIMD scoring) is
ever wanted, add an **opt-in** `scirust-kernel` feature that pulls `scirust` and
routes `decode_latent` through `scirust::…::dequant_latent`. It must remain
**off by default** so the zero-dependency guarantee holds.
