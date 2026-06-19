# CCOS external memory interface

A single, documented façade for using CCOS as an agent's **external working
memory**: write code and failure signals in, recall a bounded, causally-coherent
context window out, and keep an auditable, hash-chained state on disk.

It is the in-process Rust surface (`ccos::external_memory`). A network server or a
stdio CLI can be layered on top later — both would call exactly this API — but the
façade is the contract.

- [Why](#why)
- [Quick start](#quick-start)
- [The contract](#the-contract)
- [Recall strategies](#recall-strategies)
- [Node identity](#node-identity)
- [Persistence & integrity](#persistence--integrity)
- [A typical agent loop](#a-typical-agent-loop)
- [Guarantees](#guarantees)
- [Limitations](#limitations)

## Why

An LLM agent's context window is a scarce, bounded resource. CCOS manages it like
an OS manages memory: source is parsed into a **causal graph**, nodes are scored
(importance · failure-pressure · recency · use), and a bounded window is paged in.
The external-memory façade exposes that machinery as a handful of verbs an agent
actually needs — `ingest`, `signal_failure`, `recall`, `verify`, `checkpoint` —
instead of the kernel's a-dozen internal types.

## Quick start

```rust
use ccos::external_memory::{CcosMemory, ExternalMemory, Recall};

// Open (or create) a persistent workspace memory.
let mut mem = CcosMemory::open("workspace.ccos")?;

// Write the workspace in.
mem.ingest_source("src/db.rs", &std::fs::read_to_string("src/db.rs")?);
mem.ingest_source("src/api.rs", &std::fs::read_to_string("src/api.rs")?);

// The agent's task failed at a test that exercises db.rs — tell the memory.
let affected = mem.signal_failure("file:src/db.rs", 3)?;   // propagate 3 hops

// Recall a bounded, causally-coherent window for the model (≤ 2048 tokens).
let window = mem.recall(&Recall::around("file:src/db.rs"), 2048);
for item in &window.items {
    println!("{:.3}  {:<28}  ({} chars)", item.score, item.uri, item.content.len());
}

// Persist; the hash chain stays verifiable.
mem.checkpoint()?;
assert!(mem.verify().valid);
# Ok::<(), ccos::external_memory::MemoryError>(())
```

Add the dependency (path or git) and use the crate as `ccos`.

## The contract

`ExternalMemory` is the stable trait; `CcosMemory` is the implementation.

| Operation | Signature | Meaning |
| --------- | --------- | ------- |
| `ingest_source` | `(&mut self, uri, source) -> IngestReport` | parse a file, fold the delta into the graph, extend the hash chain. Re-ingesting identical text is a no-op delta. |
| `signal_failure` | `(&mut self, node, depth) -> Result<usize, _>` | mark `node` failing (severity `0.95`) and propagate downstream up to `depth` hops; returns affected count. |
| `recall` | `(&self, &Recall, budget_tokens) -> RecallWindow` | select a bounded window under a strategy (below). |
| `verify` | `(&self) -> Integrity` | check both hash chains are intact. |
| `stats` | `(&self) -> MemoryStats` | counts (nodes / edges / events / files / clock). |
| `checkpoint` | `(&self) -> Result<(), _>` | persist the whole state to the bound path. |

Inherent helpers on `CcosMemory` (not in the trait):

| Method | Meaning |
| ------ | ------- |
| `open(path)` / `new()` | load-or-create / empty in-memory |
| `checkpoint_to(path)` | persist to an explicit path and bind it |
| `impact(node, depth)` | downstream **blast radius** (`Vec<Reached>`) |
| `causes(node, depth)` | upstream **causes** (`Vec<Reached>`) |
| `tick()` | advance the logical clock (recency decay) |
| `graph()` | read-only access to the raw `MemoryGraph` (escape hatch) |

### Returned types

- `IngestReport { uri, nodes_added, nodes_removed, edges_added }`
- `RecallWindow { strategy, items: Vec<RecallItem>, tokens }`
- `RecallItem { uri, score, kind, content }` — `content` is the file's ingested
  source when known, else the node's own content.
- `Integrity { valid, events, errors }`
- `MemoryStats { nodes, edges, events, files, clock }`
- `MemoryError` — `NodeNotFound(id)`, `Io`, `Serde`, `NoPath` (implements
  `std::error::Error`).

All result types derive `Serialize`, so a server/CLI layer can return them as JSON
verbatim.

## Recall strategies

```rust
Recall::working_set()        // hottest nodes globally by causal score
Recall::around("file:src/db.rs")  // the causal region anchored on a node
Recall::task("fix db timeout")    // lexical entry point → its region
```

- **`WorkingSet`** — the globally hottest nodes. Use when there is no specific
  anchor (a fresh session, a "what matters now?" query).
- **`Around(uri)`** — the causal **region** containing the anchor (the active
  file, the failing test). This is the workspace-anchored recall and the one to
  prefer for a focused task: selection is driven by *structure*, not by matching
  the query text, so it is robust when the task description is vague or
  misleading. If the anchor belongs to no region, its k-hop causal neighbourhood
  (causes + impact) is used instead.
- **`Task(text)`** — when all you have is free text: a simple lexical entry point
  (token overlap on node labels/content) is picked, then expanded to its region.
  Weaker than `Around` (it trusts the query); prefer `Around` whenever you have a
  real anchor.

Every strategy is bounded by `budget_tokens` (≈ 4 chars/token) and is
deterministic: ties break on the node id.

## Node identity

Node ids are namespaced strings:

| Prefix | Example | Meaning |
| ------ | ------- | ------- |
| `file:` | `file:src/db.rs` | a source file |
| `mod:` | `mod:src/db.rs:tests` | a module |
| `sym:` | `sym:src/db.rs:query` | a symbol (fn/const/struct/…) |
| `use:` | `use:src/db.rs:std::io` | an import |
| `dep:` | `dep:serde` | an external dependency |

`ingest_source("src/db.rs", …)` creates `file:src/db.rs` (and the module/symbol
nodes under it). `signal_failure` and `recall` accept either a full node id or a
**bare path** — `file:` is assumed when no known prefix is present, so
`signal_failure("src/db.rs", 3)` and `signal_failure("file:src/db.rs", 3)` are
equivalent.

## Persistence & integrity

`open(path)` loads an existing checkpoint or starts empty with `path` bound;
`checkpoint()` writes the whole state (graph + both logs + retained sources) as
one JSON file. Every `ingest_source` appends to a canonical **SHA-256 hash chain**
over the event's replayable content, so:

```rust
let report = mem.verify();          // Integrity { valid, events, errors }
assert!(report.valid);              // tampering with the checkpoint is detectable
```

A checkpoint **round-trips**: reloading reproduces the graph and the chain
(covered by `checkpoint_roundtrips_through_a_file`).

## A typical agent loop

```text
on session start:        for each workspace file → ingest_source(uri, src)
on a tool/test failure:  signal_failure(failing_file, depth=3)
before each model call:  window = recall(Around(active_file), budget)
                         feed window.items (uri + content) to the model
on edit:                 ingest_source(uri, new_src)   // O(Δ) re-index
periodically:            checkpoint(); assert verify().valid
```

`recall(Around(active_file), budget)` is the key step: it returns the failing
file *and the causally-related files the fix is likely to touch*, within the token
budget — the thing a flat top-k retriever misses.

A runnable version of this loop is [`scripts/agent_demo.py`](../scripts/agent_demo.py):
it ingests a small workspace whose bug's cause is two files away and lexically
dissimilar, recalls the causal region, signals the failure, and (if
`OLLAMA_ENDPOINT` is set) asks a local model to propose a fix. It runs offline —
the value is observable without a model.

## Driving it from any language (`ccos memory`)

The same façade is exposed as a **stdio JSON-Lines** command, so any language can
use CCOS as memory via a subprocess — no server to run:

```bash
printf '%s\n' \
  '{"op":"ingest","uri":"src/db.rs","source":"pub fn query() {}"}' \
  '{"op":"failure","node":"file:src/db.rs","depth":3}' \
  '{"op":"recall","strategy":"around","anchor":"file:src/db.rs","budget":2048}' \
  '{"op":"verify"}' \
  | ccos memory --path workspace.ccos
```

One request object per line in, one JSON response per line out. The workspace is
loaded from `--path` (default `workspace.ccos`) and checkpointed back if any
request mutated it. Operations: `ingest` (`uri`, `source`), `failure` (`node`,
`depth`), `recall` (`strategy` ∈ `around` / `task` / `working_set`, plus
`anchor` / `text`, `budget`), `impact` / `causes` (`node`, `depth`), `verify`,
`stats`. Responses are the `Serialize` types above verbatim; errors are
`{"error":"…"}`.

From Python:

```python
import subprocess, json

def mem(reqs, path="workspace.ccos"):
    inp = "\n".join(json.dumps(r) for r in reqs)
    out = subprocess.run(["ccos", "memory", "--path", path],
                         input=inp, capture_output=True, text=True).stdout
    return [json.loads(l) for l in out.splitlines()]

mem([{"op": "ingest", "uri": "src/db.rs", "source": open("src/db.rs").read()}])
win = mem([{"op": "recall", "strategy": "around",
            "anchor": "file:src/db.rs", "budget": 2048}])[0]
for it in win["items"]:
    print(it["score"], it["uri"])
```

## Guarantees

- **Deterministic recall** — a total order on `(score, uri)`; same memory, same
  budget ⇒ same window.
- **Tamper-evidence** — `verify()` covers the whole history via the hash chain.
- **Round-trip** — checkpoint then reload reproduces the graph and the chain.
- **`edges ⊆ nodes × nodes`** — the kernel never holds dangling edges.

## Limitations

- Retained source is held per ingested file (so `recall` can return content); for
  very large workspaces this is memory the kernel graph itself does not keep.
- `Around`/`Task` recompute region clustering per call (correctness over speed);
  fine for interactive use, not for tight inner loops on huge graphs.
- `Task` uses a deliberately simple lexical entry point (no embeddings) — it is a
  convenience, not a semantic retriever; prefer `Around` with a real anchor.
- A **stdio JSON CLI** (`ccos memory`) ships on top of this façade (see above); a
  network/HTTP server is not included, but would layer on the same trait.
