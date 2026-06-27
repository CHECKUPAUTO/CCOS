# Deploying CCOS

CCOS ships as a single Rust binary (`ccos`) that also hosts the MCP server. This is the recommended
path for running it on a server, behind an AI agent.

> **First command on any new host:** `ccos doctor` — it reports the build profile, compiled features,
> active parser, license status, and any deployment warnings.

## 1. Build

The `ccos` binary **requires the `llm` feature** (it drives the async MCP server + runtime), so a bare
`cargo build --release` produces **no binary at all**. Build with the deployment features:

```sh
cargo build --release --features llm,license
```

| feature | gives you | default |
|---|---|---|
| `syn-parser` | the accurate `syn` AST parser | **on** |
| `llm` | the `ccos` binary itself + MCP server + Ollama backend | required for the bin |
| `license` | the offline ed25519 Pro-license verifier (`tensions` / `audit` + a Pro tier) | recommended |
| `learned-embed` | the LSA semantic re-ranker | optional |
| `mimalloc` | a faster allocator (benchmarking only) | optional |

## 2. Install

```sh
install -m 755 target/release/ccos /usr/local/bin/ccos
ccos doctor
```

`ccos doctor` (or `ccos doctor --json` for machines) prints, e.g.:

```
ccos doctor — deployment self-check

  version      0.3.0
  build        release
  target       x86_64-linux
  parser       syn AST (accurate)
  features     llm=yes license=yes syn-parser=yes learned-embed=no mimalloc=no
  mcp          ready  (ccos mcp <workspace>)

  license
    verifier   ed25519 (compiled in)
    vendor key placeholder (fail-closed)
    tier       community
    token      none

  ⚠ 1 warning(s):
    - embedded public key is the all-zero placeholder — Pro is fail-closed until a vendor key is set …
```

## 3. MCP server

Point your MCP gateway at the **installed release** binary and a workspace path:

```
/usr/local/bin/ccos mcp /var/lib/ccos/workspace.ccos
```

> ⚠️ Do **not** point the gateway at `target/debug/ccos`: a debug build is slower and may diverge from
> your installed release. `ccos doctor` flags a debug build. Keep the MCP command and your install on
> the same release binary.

The workspace is one `.ccos` snapshot + a `.oplog` timeline sidecar, shared with the CLI.

## 4. Pro license (optional)

The public key shipped in the source tree is an **all-zero placeholder**, so the build is
**fail-closed** — it licenses nothing — until you bake in your own vendor key:

```sh
# 1. Generate a keypair (keep the seed SECRET; it never goes in the repo).
cargo run --features license --example license_sign -- keygen
#    → paste the printed LICENSE_PUBLIC_KEY into src/license.rs, then rebuild.

# 2. Sign a license for a customer (perpetual = omit --days).
CCOS_LICENSE_SIGNING_SEED=<64-hex-seed> \
  cargo run --features license --example license_sign -- sign --licensee "Acme Corp" --days 365

# 3. Install the token on the host — env var or file.
export CCOS_LICENSE="<token>"                 # or: write it to ~/.config/ccos/license
ccos doctor                                   # tier should now read: PRO
```

Verification is **fully offline** — no network, no telemetry — so a customer can run air-gapped.

## 5. Durability (what survives a crash / power cut)

Every checkpoint is **crash- and power-safe**. `util::write_durable` writes a temp file, `fsync`s it,
**atomically renames** it into place, then `fsync`s the parent directory — the snapshot is never left
half-written, and the hash-chained event log detects any tampering on reload. Durability is at
**checkpoint granularity**: the agent / MCP flow checkpoints to the workspace, so both the causal
memory and the replayable timeline survive a restart or a sudden power loss.

## One-shot

`scripts/install.sh` does build → install → `ccos doctor` in one step
(`PREFIX=/usr/local/bin CCOS_FEATURES=llm,license sh scripts/install.sh`).
