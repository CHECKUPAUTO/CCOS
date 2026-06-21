# CCOS editor clients — the attentional shield

Thin clients over **`ccos focus --json`**. When a `cargo test` run fails, instead of the
raw 50-line backtrace they show the **causal region** — the likely root cause first, then the
symptom, then the rest — and hide the noise. All the intelligence is in the `ccos` binary
(the triad + budget balancing + the `focus_view` cause/symptom tagging); the clients only
*capture the failure → call `ccos focus` → render*.

Build the binary first: `cargo build --release` (then put `target/release/ccos` on `PATH`,
or set the `ccos.binary` / `opts.ccos` path).

## The contract (write your own client against this)

```
ccos focus <src> --input <cargo-output> --workspace <ws.ccos> --json
```
emits:
```json
{
  "message": "…panicked at src/writer.rs:3:78",
  "symptom_files": ["src/lib.rs", "src/writer.rs"],
  "workspace_files": 3,
  "reparsed_files": 0,
  "tokens": 874,
  "entries": [
    { "file": "src/config.rs", "role": "cause",   "score": 0.67, "content": "pub fn buffer_size() -> usize { 0 }" },
    { "file": "src/writer.rs",  "role": "symptom", "score": 0.72, "content": "…" }
  ]
}
```
`role` is `cause` (top file pulled in *causally*, not named by the trace — the likely root),
`symptom` (a file the trace names), or `context`. `--workspace` makes re-parse **O(Δ)**
(only changed files), so calling `focus` on every save stays ~instant (~15 ms warm).

## Neovim (`editors/nvim/`)

```lua
-- lazy.nvim
{ dir = "/path/to/CCOS/editors/nvim", config = function()
    require("ccos_focus").setup({ ccos = "ccos", keymap = "<leader>cf" })
  end }
```
`:CcosFocus` (or `<leader>cf`) runs the test command, then floats the shield; `<CR>` on a file
opens it, `q`/`<Esc>` closes. Config: `ccos`, `src`, `workspace`, `budget`, `test_cmd`, `keymap`.

## VS Code (`editors/vscode/`)

```sh
cd editors/vscode && npm install && npm run compile
```
Press **F5** to launch an Extension Development Host, then run **“CCOS: Focus on failure”** from
the command palette. It runs the test command and opens a side panel, *CCOS Attentional Shield*;
clicking the **likely cause** opens that file. Settings under `ccos.*` (binary, src, workspace,
budget, testCommand).

## Status — honest

- **Backend** (`ccos focus`, `--json`, `--workspace`): verified end-to-end on a multi-file bug
  (the cross-file cause is surfaced and tagged), unit-tested, in the main CI gate.
- **VS Code render logic** (`src/render.ts`, the cause-first ordering + HTML/CSP/escaping): the
  pure part is **node-tested against real `ccos focus --json` output** (compile `render.ts`, feed
  a live payload — see the commit). The editor glue (`extension.ts`) uses the standard VS Code
  API and needs `npm install` + the editor to run; it is **not** executed in CI.
- **Neovim plugin**: written against the same verified contract; **not** checked by a Lua
  interpreter here. Treat both clients as MVPs to F5/`:CcosFocus`, not shipped extensions.

## Known limits (inherited from the kernel)

The causal graph is structural, not semantic; the default parser is a line-based Rust heuristic
(`--features syn-parser` for a real AST). `focus` keys files by the `src/…` tail to match
`cargo`'s paths — a multi-crate workspace needs path-mapping. The “likely cause” is a heuristic
(the top non-trace file): right when the cause is a pulled-in dependency, and harmless on a
single-file bug (it just shows the symptom region). See `docs/DESIGN_focus_shield.md`.
