use crate::memory::{EdgeType, MemoryGraph, NodeId, NodeType};
use crate::util::sha256_hex;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParseResult {
    pub file_path: String,
    pub file_hash: String,
    pub modules: Vec<ModuleDecl>,
    pub use_statements: Vec<UseStatement>,
    pub symbols: Vec<Symbol>,
    /// In-body call-sites (Slice 1: single-segment free-function calls). `serde(default)` +
    /// skip-if-empty keeps the serialized form unchanged for the common (heuristic / no-call) case.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub call_sites: Vec<CallSite>,
    /// In-body references to module-level `static`/`const` items (data-flow Slice 1). Same
    /// skip-if-empty serde contract — empty on the heuristic path and for call-free bodies.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub data_refs: Vec<DataRef>,
    pub generated_nodes: usize,
    pub generated_edges: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleDecl {
    pub name: String,
    pub line: usize,
    pub is_public: bool,
    pub children: Vec<ModuleDecl>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UseStatement {
    pub full_path: String,
    pub line: usize,
    pub is_import: bool,
    pub components: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Symbol {
    pub name: String,
    pub line: usize,
    pub kind: SymbolKind,
}

/// A **call-site**: a `callee()` invocation found inside `caller`'s body. `callee` is the
/// `::`-joined call path — a bare `foo` (Slice 1) or a qualified `a::b::foo` (Slice 2); method
/// calls `x.bar()` are Slice 3. Resolved to a definition symbol by
/// [`crate::memory::MemoryGraph::resolve_symbol_calls`], which adds a `caller → callee`
/// [`EdgeType::Calls`](crate::memory::EdgeType) edge — the fn→fn structure that imports alone
/// miss. Only emitted on the `syn` (real-AST) path; empty on the heuristic fallback.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallSite {
    pub caller: String,
    pub callee: String,
    pub line: usize,
}

/// A **data-reference**: `reader`'s body mentions `name`, an identifier in `SCREAMING_SNAKE_CASE`
/// (the Rust convention for a `static`/`const`). Resolved to a `static`/`const` definition symbol
/// by [`crate::memory::MemoryGraph::resolve_data_flow`], which adds a `reader → item`
/// [`EdgeType::DataFlow`](crate::memory::EdgeType) edge — the shared-global-state channel that call
/// and import edges miss. Only emitted on the `syn` (real-AST) path; empty on the heuristic fallback.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataRef {
    pub reader: String,
    pub name: String,
    pub line: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum SymbolKind {
    Function,
    Struct,
    Enum,
    Trait,
    Impl,
    Const,
    Static,
    Type,
    Macro,
    Other,
}

#[derive(Debug, Clone)]
pub struct ASTParser;

impl ASTParser {
    pub fn new() -> Self {
        Self
    }

    pub fn parse_source(&self, file_path: &str, source_code: &str) -> ParseResult {
        let hash = Self::compute_hash(source_code);
        let (modules, use_statements, symbols, call_sites, data_refs) =
            Self::extract_all(source_code);

        ParseResult {
            file_path: file_path.to_string(),
            file_hash: hash,
            generated_nodes: modules.len() + use_statements.len() + symbols.len(),
            generated_edges: use_statements.len() + modules.len(),
            modules,
            use_statements,
            symbols,
            call_sites,
            data_refs,
        }
    }

    /// Extract modules / `use` statements / symbols from source.
    ///
    /// With the `syn-parser` feature enabled, this parses a real Rust AST (which
    /// captures nested-module bodies, multi-line signatures, grouped `use` and
    /// impl methods). If the feature is off — or the source does not parse as
    /// valid Rust — it falls back to the zero-dependency line-based heuristic.
    #[allow(clippy::type_complexity)]
    fn extract_all(
        source: &str,
    ) -> (
        Vec<ModuleDecl>,
        Vec<UseStatement>,
        Vec<Symbol>,
        Vec<CallSite>,
        Vec<DataRef>,
    ) {
        #[cfg(feature = "syn-parser")]
        {
            if let Some(parsed) = syn_ast::parse(source) {
                return parsed;
            }
        }
        // Heuristic fallback emits no call-sites or data-refs (it does not parse expression
        // trees), so the call/data-flow graphs simply stay empty on that path — a build can only
        // ever *omit* those edges.
        (
            Self::extract_modules(source),
            Self::extract_uses(source),
            Self::extract_symbols(source),
            Vec::new(),
            Vec::new(),
        )
    }

    /// Build the causal graph for one file, storing **granular** content on every
    /// node so recall never spends a whole file's budget on a single node (see
    /// `docs/DESIGN_symbol_granularity.md`): the file node carries a *header*
    /// (path + one signature line per symbol), each symbol node carries its own
    /// source span, modules carry their declaration line, and `use` nodes the
    /// import line. The whole-file source is kept by the caller (`ExternalMemory`)
    /// for explicit retrieval, not duplicated into every node.
    pub fn update_memory_graph(&self, result: &ParseResult, source: &str, graph: &mut MemoryGraph) {
        let lines: Vec<&str> = source.lines().collect();
        let line_at = |ln: usize| {
            lines
                .get(ln.saturating_sub(1))
                .map(|l| l.trim().to_string())
                .unwrap_or_default()
        };

        // File node = a thin header: the path and a signature line per symbol,
        // capped at `header_symbol_cap()` lines so a huge file (syn's `item.rs` has
        // ~88 symbols) does not spend a third of a recall budget on its index alone
        // (see `docs/DESIGN_recall_budget.md`). Capped-out symbols are still their
        // own span nodes; the header just teases the first N.
        let file_id = NodeId(format!("file:{}", result.file_path));
        let cap = header_symbol_cap();
        let mut header = format!(
            "// {} — {} symbols\n",
            result.file_path,
            result.symbols.len()
        );
        let mut shown = 0usize;
        for s in &result.symbols {
            if shown >= cap {
                break;
            }
            let sig = line_at(s.line);
            if !sig.is_empty() {
                header.push_str(&sig);
                header.push('\n');
                shown += 1;
            }
        }
        if result.symbols.len() > shown {
            header.push_str(&format!("// … (+{} more)\n", result.symbols.len() - shown));
        }
        graph.upsert_node(
            file_id.clone(),
            result.file_path.clone(),
            header,
            NodeType::Module,
        );

        // Module nodes = their declaration line only; the items inside become their
        // own symbol nodes, so carrying the body here would just duplicate them.
        for module in &result.modules {
            let mod_id = NodeId(format!("mod:{}:{}", result.file_path, module.name));
            graph.upsert_node(
                mod_id.clone(),
                module.name.clone(),
                line_at(module.line),
                NodeType::Module,
            );
            graph.add_edge(file_id.clone(), mod_id.clone(), 0.9, EdgeType::Contains);
            self.add_module_tree(graph, &mod_id, &module.children, &result.file_path, &lines);
        }

        // Use statements = the import line itself.
        for use_stmt in &result.use_statements {
            let use_id = NodeId(format!("use:{}:{}", result.file_path, use_stmt.full_path));
            graph.upsert_node(
                use_id.clone(),
                format!("use {}", use_stmt.full_path),
                line_at(use_stmt.line),
                NodeType::Symbol,
            );
            graph.add_edge(file_id.clone(), use_id.clone(), 0.5, EdgeType::DependsOn);

            // Create dependency edges based on use path components
            if let Some(root) = use_stmt.components.first() {
                let dep_id = NodeId(format!("dep:{}", root));
                graph.upsert_node(
                    dep_id.clone(),
                    root.clone(),
                    format!("External dependency: {}", root),
                    NodeType::Symbol,
                );
                graph.add_edge(use_id.clone(), dep_id, 0.7, EdgeType::DependsOn);
            }
        }

        // Symbol nodes = the symbol's own source span (the granular recall unit).
        for symbol in &result.symbols {
            let sym_id = NodeId(format!("sym:{}:{}", result.file_path, symbol.name));
            let (start, end) = symbol_span(&lines, symbol.line);
            // Clamp into the slice before indexing: with `--features syn-parser`
            // a span line can land past EOF (trailing-newline / CRLF edge cases),
            // and `lines[start-1..end]` would otherwise panic out of bounds.
            let body = if lines.is_empty() {
                String::new()
            } else {
                let s = start.clamp(1, lines.len());
                let e = end.clamp(s, lines.len());
                lines[s - 1..e].join("\n")
            };
            graph.upsert_node(sym_id.clone(), symbol.name.clone(), body, NodeType::Symbol);
            graph.add_edge(file_id.clone(), sym_id.clone(), 0.6, EdgeType::Contains);
            // A `static`/`const` is the only valid `DataFlow` target; mark it so the resolver can
            // tell it apart from a function (the graph node stores `NodeType`, not `SymbolKind`).
            if matches!(symbol.kind, SymbolKind::Static | SymbolKind::Const) {
                graph.mark_data_symbol(sym_id.clone());
            }
        }

        // Hand this file's in-body call-sites to the graph; they are resolved into `caller →
        // callee` Calls edges by the whole-graph `resolve_symbol_calls` pass once every file is
        // ingested (a call may target a symbol defined in a not-yet-seen file). Replaces any
        // prior entry for this file, so a re-ingest re-states (never duplicates) its calls.
        graph.set_pending_calls(
            &result.file_path,
            result
                .call_sites
                .iter()
                .map(|c| (c.caller.clone(), c.callee.clone(), c.line))
                .collect(),
        );
        // Likewise hand over this file's `static`/`const` references, resolved into `reader → item`
        // DataFlow edges by the whole-graph `resolve_data_flow` pass after call resolution.
        graph.set_pending_data_refs(
            &result.file_path,
            result
                .data_refs
                .iter()
                .map(|d| (d.reader.clone(), d.name.clone(), d.line))
                .collect(),
        );
    }

    fn add_module_tree(
        &self,
        graph: &mut MemoryGraph,
        parent_id: &NodeId,
        children: &[ModuleDecl],
        file_path: &str,
        lines: &[&str],
    ) {
        for child in children {
            let child_id = NodeId(format!("mod:{}:{}", file_path, child.name));
            let decl = lines
                .get(child.line.saturating_sub(1))
                .map(|l| l.trim().to_string())
                .unwrap_or_default();
            graph.upsert_node(child_id.clone(), child.name.clone(), decl, NodeType::Module);
            graph.add_edge(
                parent_id.clone(),
                child_id.clone(),
                0.85,
                EdgeType::Contains,
            );

            if !child.children.is_empty() {
                self.add_module_tree(graph, &child_id, &child.children, file_path, lines);
            }
        }
    }

    fn compute_hash(source: &str) -> String {
        sha256_hex(source)
    }

    fn extract_modules(source: &str) -> Vec<ModuleDecl> {
        let mut modules = Vec::new();

        for (line_num, line) in source.lines().enumerate() {
            let trimmed = line.trim();
            let stripped = strip_comments(trimmed);

            if stripped.is_empty() {
                continue;
            }

            let is_pub = stripped.starts_with("pub mod ");
            let is_mod = (stripped.starts_with("mod ") || stripped.starts_with("pub mod "))
                && (stripped.ends_with(';') || stripped.ends_with('{'));

            if is_mod {
                let name = if is_pub {
                    stripped
                        .strip_prefix("pub mod ")
                        .unwrap_or("")
                        .trim()
                        .split([' ', '{', ';'])
                        .next()
                        .unwrap_or("")
                } else {
                    stripped
                        .strip_prefix("mod ")
                        .unwrap_or("")
                        .trim()
                        .split([' ', '{', ';'])
                        .next()
                        .unwrap_or("")
                };

                if name.is_empty() || name.contains("//") {
                    continue;
                }

                modules.push(ModuleDecl {
                    name: name.to_string(),
                    line: line_num + 1,
                    is_public: is_pub,
                    children: Vec::new(),
                });
            }
        }
        modules
    }

    fn extract_uses(source: &str) -> Vec<UseStatement> {
        let mut uses = Vec::new();

        for (line_num, line) in source.lines().enumerate() {
            let trimmed = line.trim();
            let stripped = strip_comments(trimmed);

            if stripped.is_empty() {
                continue;
            }

            if stripped.starts_with("use ") {
                let raw_path = stripped.strip_prefix("use ").unwrap_or("").trim();
                let path = raw_path.trim_end_matches(';').trim();

                let components: Vec<String> = path
                    .split("::")
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();

                let full_path = components.join("::");

                if !full_path.is_empty() {
                    uses.push(UseStatement {
                        full_path,
                        line: line_num + 1,
                        is_import: true,
                        components,
                    });
                }
            }
        }
        uses
    }

    fn extract_symbols(source: &str) -> Vec<Symbol> {
        let mut symbols = Vec::new();
        let mut seen = HashSet::new();

        for (line_num, line) in source.lines().enumerate() {
            let trimmed = line.trim();
            let stripped = strip_comments(trimmed);

            if stripped.is_empty() {
                continue;
            }

            let kind_and_name = if stripped.starts_with("fn ") {
                stripped.strip_prefix("fn ").and_then(|rest| {
                    let name = rest
                        .split(['(', '<', '{', ';'])
                        .next()
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    if name.is_empty() || name.starts_with("//") {
                        None
                    } else {
                        Some((SymbolKind::Function, name))
                    }
                })
            } else if stripped.starts_with("pub fn ") {
                stripped.strip_prefix("pub fn ").and_then(|rest| {
                    let name = rest
                        .split(['(', '<', '{', ';'])
                        .next()
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    if name.is_empty() {
                        None
                    } else {
                        Some((SymbolKind::Function, name))
                    }
                })
            } else if stripped.starts_with("struct ") {
                stripped.strip_prefix("struct ").and_then(|rest| {
                    let name = rest
                        .split(['<', '{', '(', ';'])
                        .next()
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    if name.is_empty() {
                        None
                    } else {
                        Some((SymbolKind::Struct, name))
                    }
                })
            } else if stripped.starts_with("pub struct ") {
                stripped.strip_prefix("pub struct ").and_then(|rest| {
                    let name = rest
                        .split(['<', '{', '(', ';'])
                        .next()
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    if name.is_empty() {
                        None
                    } else {
                        Some((SymbolKind::Struct, name))
                    }
                })
            } else if stripped.starts_with("enum ") {
                stripped.strip_prefix("enum ").and_then(|rest| {
                    let name = rest
                        .split(['<', '{', ';'])
                        .next()
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    if name.is_empty() {
                        None
                    } else {
                        Some((SymbolKind::Enum, name))
                    }
                })
            } else if stripped.starts_with("pub enum ") {
                stripped.strip_prefix("pub enum ").and_then(|rest| {
                    let name = rest
                        .split(['<', '{', ';'])
                        .next()
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    if name.is_empty() {
                        None
                    } else {
                        Some((SymbolKind::Enum, name))
                    }
                })
            } else if stripped.starts_with("trait ") {
                stripped.strip_prefix("trait ").and_then(|rest| {
                    let name = rest
                        .split(['<', '{', ';'])
                        .next()
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    if name.is_empty() {
                        None
                    } else {
                        Some((SymbolKind::Trait, name))
                    }
                })
            } else if stripped.starts_with("pub trait ") {
                stripped.strip_prefix("pub trait ").and_then(|rest| {
                    let name = rest
                        .split(['<', '{', ';'])
                        .next()
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    if name.is_empty() {
                        None
                    } else {
                        Some((SymbolKind::Trait, name))
                    }
                })
            } else if stripped.starts_with("impl ") {
                stripped.strip_prefix("impl ").and_then(|rest| {
                    let name = rest
                        .split(['<', '{', ' ', ';'])
                        .next()
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    if name.is_empty() {
                        None
                    } else {
                        Some((SymbolKind::Impl, name))
                    }
                })
            } else if stripped.starts_with("const ") {
                stripped.strip_prefix("const ").and_then(|rest| {
                    let name = rest
                        .split([':', '=', ';'])
                        .next()
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    if name.is_empty() {
                        None
                    } else {
                        Some((SymbolKind::Const, name))
                    }
                })
            } else if stripped.starts_with("pub const ") {
                stripped.strip_prefix("pub const ").and_then(|rest| {
                    let name = rest
                        .split([':', '=', ';'])
                        .next()
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    if name.is_empty() {
                        None
                    } else {
                        Some((SymbolKind::Const, name))
                    }
                })
            } else if stripped.starts_with("static ") {
                stripped.strip_prefix("static ").and_then(|rest| {
                    let name = rest
                        .split([':', '=', ';'])
                        .next()
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    if name.is_empty() {
                        None
                    } else {
                        Some((SymbolKind::Static, name))
                    }
                })
            } else if stripped.starts_with("type ") {
                stripped.strip_prefix("type ").and_then(|rest| {
                    let name = rest
                        .split(['=', ';', '<'])
                        .next()
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    if name.is_empty() {
                        None
                    } else {
                        Some((SymbolKind::Type, name))
                    }
                })
            } else if stripped.starts_with("macro_rules!") {
                stripped.strip_prefix("macro_rules!").and_then(|rest| {
                    let name = rest
                        .trim()
                        .split(['{', '(', ' '])
                        .next()
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    if name.is_empty() {
                        None
                    } else {
                        Some((SymbolKind::Macro, name))
                    }
                })
            } else {
                None
            };

            if let Some((kind, name)) = kind_and_name {
                if seen.insert((kind.clone(), name.clone())) {
                    symbols.push(Symbol {
                        name,
                        line: line_num + 1,
                        kind,
                    });
                }
            }
        }
        symbols.sort_by_key(|s| s.line);
        symbols
    }
}

/// Strip comments from a single source line, ignoring `//` and `/* … */` that
/// appear inside string literals. Inline block comments (`a(); /* c */ b();`)
/// are removed in place. A block comment left open at end-of-line is dropped to
/// the line end — multi-line `/* … */` spans are a known limitation of the
/// line-based parser (see `ROADMAP.md`).
fn strip_comments(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut in_string = false;
    let mut escaped = false;
    let mut chars = line.chars().peekable();

    while let Some(c) = chars.next() {
        if in_string {
            out.push(c);
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_string = false;
            }
            continue;
        }
        match c {
            '"' => {
                in_string = true;
                out.push(c);
            }
            // Line comment: the rest of the line is dropped.
            '/' if chars.peek() == Some(&'/') => break,
            // Block comment: skip until the closing `*/` (or end of line).
            '/' if chars.peek() == Some(&'*') => {
                chars.next(); // consume '*'
                let mut prev = '\0';
                for d in chars.by_ref() {
                    if prev == '*' && d == '/' {
                        break;
                    }
                    prev = d;
                }
            }
            _ => out.push(c),
        }
    }
    out.trim().to_string()
}

impl Default for ASTParser {
    fn default() -> Self {
        Self::new()
    }
}

/// Max signature lines a file-header node lists. Default 24; override with
/// `CCOS_HEADER_SYMBOLS`. Caps the header footprint of very large files so it
/// cannot dominate a recall budget; the omitted symbols remain their own nodes.
fn header_symbol_cap() -> usize {
    std::env::var("CCOS_HEADER_SYMBOLS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|x| *x >= 1)
        .unwrap_or(24)
}

/// Inclusive 1-based `[start, end]` line span of the item beginning at
/// `start_line`. Brace-matched for `{}`-bodied items (fn/struct/enum/trait/impl);
/// semicolon-terminated for the rest (const/static/type/use); a lone start line
/// otherwise. Capped at end-of-file. `//`-comment and string aware via
/// [`strip_comments`]; braces inside strings and multi-line `/* … */` share the
/// line parser's documented fragility — `--features syn-parser` parses exactly.
fn symbol_span(lines: &[&str], start_line: usize) -> (usize, usize) {
    let n = lines.len();
    if start_line == 0 || start_line > n {
        // Out-of-range start (e.g. a syn span past EOF): return an in-bounds
        // lone line so callers can slice without panicking.
        let line = start_line.clamp(1, n.max(1));
        return (line, line);
    }
    let s0 = start_line - 1; // 0-based
                             // Within a short signature window, find the body's opening brace — or a
                             // semicolon that terminates a brace-less item (const/static/type/use).
    let mut open = None;
    for (off, line) in lines[s0..(s0 + 8).min(n)].iter().enumerate() {
        let stripped = strip_comments(line);
        if stripped.contains('{') {
            open = Some(s0 + off);
            break;
        }
        if stripped.trim_end().ends_with(';') {
            return (start_line, s0 + off + 1);
        }
    }
    let Some(open) = open else {
        return (start_line, start_line);
    };
    let mut depth: i32 = 0;
    for (i, line) in lines.iter().enumerate().skip(open) {
        for c in strip_comments(line).chars() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        return (start_line, i + 1);
                    }
                }
                _ => {}
            }
        }
    }
    (start_line, n)
}

/// Real-AST parsing via `syn` (enabled by the `syn-parser` feature). Produces
/// the same `(modules, uses, symbols)` triple as the heuristic parser, but
/// accurately: it descends into nested-module bodies and impl blocks, expands
/// grouped `use` trees, handles multi-line signatures, and ignores comments
/// natively. Returns `None` on a syntax error so the caller can fall back.
#[cfg(feature = "syn-parser")]
mod syn_ast {
    use super::{CallSite, DataRef, ModuleDecl, Symbol, SymbolKind, UseStatement};
    use proc_macro2::Span;
    use std::collections::HashSet;
    use syn::spanned::Spanned;
    use syn::visit::Visit;

    #[allow(clippy::type_complexity)]
    pub fn parse(
        source: &str,
    ) -> Option<(
        Vec<ModuleDecl>,
        Vec<UseStatement>,
        Vec<Symbol>,
        Vec<CallSite>,
        Vec<DataRef>,
    )> {
        let file = syn::parse_file(source).ok()?;
        let mut out = Collected::default();
        walk(&file.items, &mut out);
        Some((out.modules, out.uses, out.symbols, out.calls, out.data_refs))
    }

    #[derive(Default)]
    struct Collected {
        modules: Vec<ModuleDecl>,
        uses: Vec<UseStatement>,
        symbols: Vec<Symbol>,
        calls: Vec<CallSite>,
        data_refs: Vec<DataRef>,
    }

    /// Collects single-segment free-function call-sites from a function body, in source
    /// (document) order — a pure function of the AST, so call extraction is deterministic.
    struct CallVisitor<'a> {
        caller: String,
        /// Method / associated-fn names defined on the enclosing impl or trait (empty for free
        /// functions). A `self.m()` / `Self::m()` is captured ONLY when `m` is in this set, so a
        /// Deref- or blanket-trait-provided method (not defined here) is never mislinked to a
        /// same-named free function in this module.
        own_methods: &'a std::collections::HashSet<String>,
        /// Names bound *locally* in this function — parameters, `let`s, and fn-local `const`/`static`
        /// items. A `SCREAMING_SNAKE` data-reference whose name is locally bound is skipped: it
        /// denotes the local (invisible as a graph symbol), not a same-named module-level
        /// `static`/`const`, so resolving it global-unique would be a false edge.
        local_bound: &'a std::collections::HashSet<String>,
        calls: Vec<CallSite>,
        data_refs: Vec<DataRef>,
    }
    impl<'a, 'ast> Visit<'ast> for CallVisitor<'a> {
        fn visit_expr_call(&mut self, node: &'ast syn::ExprCall) {
            if let syn::Expr::Path(p) = &*node.func {
                // Capture the call path as a `::`-joined callee: bare `foo` (Slice 1) and qualified
                // `a::b::foo` (Slice 2). `<T>::x` (qself) and `::abs::x` (leading `::`) are skipped.
                // A `Self::assoc()` is captured only when `assoc` is defined on this type (Slice 3) —
                // else it is a trait/blanket assoc fn that must not match a same-named free fn.
                if p.qself.is_none() && p.path.leading_colon.is_none() {
                    if let Some(last) = p.path.segments.last() {
                        let segs = p
                            .path
                            .segments
                            .iter()
                            .map(|s| s.ident.to_string())
                            .collect::<Vec<_>>();
                        let is_self = segs.first().is_some_and(|s| s == "Self");
                        if !is_self || self.own_methods.contains(&last.ident.to_string()) {
                            self.calls.push(CallSite {
                                caller: self.caller.clone(),
                                callee: segs.join("::"),
                                line: line_of(last.ident.span()),
                            });
                        }
                    }
                }
            }
            // Always recurse so calls nested in args, closures, match arms, blocks are seen.
            syn::visit::visit_expr_call(self, node);
        }
        fn visit_expr_method_call(&mut self, node: &'ast syn::ExprMethodCall) {
            // Slice 3 — `self.method()`: the receiver type is the enclosing impl. Capture as
            // `Self::method` ONLY when `method` is defined on this type here, so a Deref/blanket
            // method (absent from `own_methods`) is skipped, not mislinked to a same-named free fn.
            // Only a bare `self` receiver; `self.field.m()` / `x.m()` have an unknown receiver type.
            if let syn::Expr::Path(p) = &*node.receiver {
                if p.qself.is_none()
                    && p.path.is_ident("self")
                    && self.own_methods.contains(&node.method.to_string())
                {
                    self.calls.push(CallSite {
                        caller: self.caller.clone(),
                        callee: format!("Self::{}", node.method),
                        line: line_of(node.method.span()),
                    });
                }
            }
            syn::visit::visit_expr_method_call(self, node);
        }
        fn visit_expr_path(&mut self, node: &'ast syn::ExprPath) {
            // Data-flow Slice 1 — a bare `SCREAMING_SNAKE` value reference is, by Rust convention, a
            // `static`/`const` use; capture it UNLESS the name is locally bound (a param / `let` /
            // fn-local `const` — `local_bound`), which would denote the local, not a same-named
            // global (the false edge the scope guard closes). Casing excludes PascalCase
            // types/variants and snake_case fns/locals; qualified `m::CONST` is a later slice.
            // (Known residual: a bare SCREAMING-cased enum variant brought in by `use` can still
            // coincide with a global const — rare, since variants are conventionally PascalCase.)
            if node.qself.is_none()
                && node.path.leading_colon.is_none()
                && node.path.segments.len() == 1
            {
                let seg = &node.path.segments[0];
                let name = seg.ident.to_string();
                if is_screaming_snake(&name) && !self.local_bound.contains(&name) {
                    self.data_refs.push(DataRef {
                        reader: self.caller.clone(),
                        name,
                        line: line_of(seg.ident.span()),
                    });
                }
            }
            syn::visit::visit_expr_path(self, node);
        }
    }

    /// A `static`/`const`-style identifier: at least one ASCII upper-case letter and nothing but
    /// upper-case letters, digits, and underscores (the Rust `SCREAMING_SNAKE_CASE` convention).
    fn is_screaming_snake(s: &str) -> bool {
        s.chars().any(|c| c.is_ascii_uppercase())
            && s.chars()
                .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
    }

    /// Names bound *locally* in a function — its parameters plus every `let`, fn-local
    /// `const`/`static`, closure parameter, and pattern binding in its body. Used to suppress
    /// data-references that denote a local rather than a module-level `static`/`const`.
    /// **Conservative**: collects across the whole body regardless of nested scope, so it can only
    /// ever drop a data-edge, never invent one — exactly the precision-first trade we want.
    fn local_bound_names(sig: &syn::Signature, block: &syn::Block) -> HashSet<String> {
        struct BindingCollector {
            names: HashSet<String>,
        }
        impl<'ast> Visit<'ast> for BindingCollector {
            fn visit_pat_ident(&mut self, p: &'ast syn::PatIdent) {
                self.names.insert(p.ident.to_string());
                syn::visit::visit_pat_ident(self, p);
            }
            fn visit_item_const(&mut self, c: &'ast syn::ItemConst) {
                self.names.insert(c.ident.to_string());
                syn::visit::visit_item_const(self, c);
            }
            fn visit_item_static(&mut self, s: &'ast syn::ItemStatic) {
                self.names.insert(s.ident.to_string());
                syn::visit::visit_item_static(self, s);
            }
        }
        let mut c = BindingCollector {
            names: HashSet::new(),
        };
        for input in &sig.inputs {
            if let syn::FnArg::Typed(pt) = input {
                c.visit_pat(&pt.pat);
            }
        }
        c.visit_block(block);
        c.names
    }

    fn collect_calls(
        caller: &str,
        own_methods: &std::collections::HashSet<String>,
        sig: &syn::Signature,
        block: &syn::Block,
        calls_out: &mut Vec<CallSite>,
        refs_out: &mut Vec<DataRef>,
    ) {
        let local_bound = local_bound_names(sig, block);
        let mut v = CallVisitor {
            caller: caller.to_string(),
            own_methods,
            local_bound: &local_bound,
            calls: Vec::new(),
            data_refs: Vec::new(),
        };
        v.visit_block(block);
        calls_out.append(&mut v.calls);
        refs_out.append(&mut v.data_refs);
    }

    /// 1-based source line; the `span-locations` feature guarantees real spans.
    fn line_of(span: Span) -> usize {
        span.start().line
    }

    fn is_pub(vis: &syn::Visibility) -> bool {
        matches!(vis, syn::Visibility::Public(_))
    }

    fn push_sym(out: &mut Collected, ident: &syn::Ident, kind: SymbolKind) {
        out.symbols.push(Symbol {
            name: ident.to_string(),
            line: line_of(ident.span()),
            kind,
        });
    }

    /// Walk a list of items at one scope. Nested modules become `children` of
    /// their parent; symbols and `use`s from nested scopes are surfaced into the
    /// flat lists (matching the line parser, which sees every line).
    fn walk(items: &[syn::Item], out: &mut Collected) {
        for item in items {
            match item {
                syn::Item::Mod(m) => {
                    let mut child = Collected::default();
                    if let Some((_, inner)) = &m.content {
                        walk(inner, &mut child);
                    }
                    out.uses.append(&mut child.uses);
                    out.symbols.append(&mut child.symbols);
                    out.modules.push(ModuleDecl {
                        name: m.ident.to_string(),
                        line: line_of(m.ident.span()),
                        is_public: is_pub(&m.vis),
                        children: child.modules,
                    });
                }
                syn::Item::Use(u) => {
                    flatten_use(&u.tree, String::new(), line_of(u.span()), &mut out.uses);
                }
                syn::Item::Fn(f) => {
                    push_sym(out, &f.sig.ident, SymbolKind::Function);
                    // A free function has no `self`/`Self` methods in scope → empty own-method set.
                    collect_calls(
                        &f.sig.ident.to_string(),
                        &HashSet::new(),
                        &f.sig,
                        &f.block,
                        &mut out.calls,
                        &mut out.data_refs,
                    );
                }
                syn::Item::Struct(s) => push_sym(out, &s.ident, SymbolKind::Struct),
                syn::Item::Enum(e) => push_sym(out, &e.ident, SymbolKind::Enum),
                syn::Item::Trait(t) => {
                    push_sym(out, &t.ident, SymbolKind::Trait);
                    let methods: HashSet<String> = t
                        .items
                        .iter()
                        .filter_map(|ti| match ti {
                            syn::TraitItem::Fn(m) => Some(m.sig.ident.to_string()),
                            _ => None,
                        })
                        .collect();
                    for ti in &t.items {
                        if let syn::TraitItem::Fn(method) = ti {
                            push_sym(out, &method.sig.ident, SymbolKind::Function);
                            if let Some(body) = &method.default {
                                collect_calls(
                                    &method.sig.ident.to_string(),
                                    &methods,
                                    &method.sig,
                                    body,
                                    &mut out.calls,
                                    &mut out.data_refs,
                                );
                            }
                        }
                    }
                }
                syn::Item::Const(c) => push_sym(out, &c.ident, SymbolKind::Const),
                syn::Item::Static(s) => push_sym(out, &s.ident, SymbolKind::Static),
                syn::Item::Type(t) => push_sym(out, &t.ident, SymbolKind::Type),
                syn::Item::Macro(m) => {
                    if let Some(ident) = &m.ident {
                        push_sym(out, ident, SymbolKind::Macro);
                    }
                }
                syn::Item::Impl(i) => {
                    let methods: HashSet<String> = i
                        .items
                        .iter()
                        .filter_map(|ii| match ii {
                            syn::ImplItem::Fn(m) => Some(m.sig.ident.to_string()),
                            _ => None,
                        })
                        .collect();
                    for ii in &i.items {
                        if let syn::ImplItem::Fn(method) = ii {
                            push_sym(out, &method.sig.ident, SymbolKind::Function);
                            collect_calls(
                                &method.sig.ident.to_string(),
                                &methods,
                                &method.sig,
                                &method.block,
                                &mut out.calls,
                                &mut out.data_refs,
                            );
                        }
                    }
                }
                _ => {}
            }
        }
    }

    /// Expand a (possibly grouped) `use` tree into one `UseStatement` per leaf
    /// path, e.g. `use a::{b, c::d}` → `a::b` and `a::c::d`.
    fn flatten_use(tree: &syn::UseTree, prefix: String, line: usize, out: &mut Vec<UseStatement>) {
        let join = |p: &str, s: &str| {
            if p.is_empty() {
                s.to_string()
            } else {
                format!("{p}::{s}")
            }
        };
        match tree {
            syn::UseTree::Path(p) => {
                flatten_use(&p.tree, join(&prefix, &p.ident.to_string()), line, out)
            }
            syn::UseTree::Name(n) => push_use(join(&prefix, &n.ident.to_string()), line, out),
            // A renamed import `use a::b as c` binds `b` under the name `c`. We record the ORIGINAL
            // path (`a::b`) so import-linking resolves the real module. Call-resolution by the alias
            // `c` is deferred: a single path string cannot carry alias→real-module, and recording
            // the alias instead (`a::c`) would mislink a call `c::foo()` to a real sibling module
            // `c` if one exists (a wrong-existing-target edge). Proper alias support is future work.
            syn::UseTree::Rename(r) => push_use(join(&prefix, &r.ident.to_string()), line, out),
            syn::UseTree::Glob(_) => push_use(join(&prefix, "*"), line, out),
            syn::UseTree::Group(g) => {
                for t in &g.items {
                    flatten_use(t, prefix.clone(), line, out);
                }
            }
        }
    }

    fn push_use(full_path: String, line: usize, out: &mut Vec<UseStatement>) {
        let components: Vec<String> = full_path.split("::").map(str::to_string).collect();
        out.push(UseStatement {
            full_path,
            line,
            is_import: true,
            components,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_modules_basic() {
        let source = "mod foo;\nmod bar;\npub mod baz;";
        let modules = ASTParser::extract_modules(source);
        assert_eq!(modules.len(), 3);
        assert!(modules.iter().any(|m| m.name == "foo"));
        assert!(modules.iter().any(|m| m.name == "bar"));
        assert!(modules.iter().any(|m| m.name == "baz"));
    }

    #[test]
    fn test_extract_uses_basic() {
        let source = "use std::collections::HashMap;\nuse crate::foo::bar;";
        let uses = ASTParser::extract_uses(source);
        assert_eq!(uses.len(), 2);
        assert_eq!(uses[0].full_path, "std::collections::HashMap");
    }

    #[test]
    fn test_extract_symbols_functions() {
        let source = "fn main() {}\nfn helper() {}\npub fn public_fn() {}";
        let symbols = ASTParser::extract_symbols(source);
        assert!(symbols
            .iter()
            .any(|s| s.name == "main" && s.kind == SymbolKind::Function));
        assert!(symbols.iter().any(|s| s.name == "helper"));
        assert!(symbols.iter().any(|s| s.name == "public_fn"));
    }

    #[test]
    fn test_strip_comments() {
        assert_eq!(strip_comments("use foo; // bar"), "use foo;");
        assert_eq!(
            strip_comments("let x = \"//not_a_comment\"; // real"),
            "let x = \"//not_a_comment\";"
        );
    }

    #[test]
    fn test_strip_block_comments() {
        // Inline block comment removed in place.
        assert_eq!(
            strip_comments("pub fn a() {} /* fn fake() {} */"),
            "pub fn a() {}"
        );
        // Block comment between tokens.
        assert_eq!(strip_comments("mod /* x */ real;"), "mod  real;");
        // Unterminated block comment dropped to end of line.
        assert_eq!(strip_comments("use foo; /* trailing"), "use foo;");
        // `/*` inside a string is not a comment.
        assert_eq!(
            strip_comments("let u = \"http:/*not*/\";"),
            "let u = \"http:/*not*/\";"
        );
    }

    #[test]
    fn test_block_comment_hides_fake_symbols() {
        // A function hidden in a block comment must NOT be extracted as a symbol.
        let source = "fn real() {}\n/* fn fake() {} */ struct Keep;";
        let symbols = ASTParser::extract_symbols(source);
        assert!(symbols.iter().any(|s| s.name == "real"));
        assert!(symbols.iter().any(|s| s.name == "Keep"));
        assert!(
            !symbols.iter().any(|s| s.name == "fake"),
            "symbol inside a block comment must be ignored"
        );
    }

    #[test]
    fn test_extract_structs_and_enums() {
        let source = "struct Foo;\nenum Bar { A, B }\npub struct Baz<T> {}";
        let symbols = ASTParser::extract_symbols(source);
        assert!(symbols
            .iter()
            .any(|s| s.name == "Foo" && s.kind == SymbolKind::Struct));
        assert!(symbols
            .iter()
            .any(|s| s.name == "Bar" && s.kind == SymbolKind::Enum));
        assert!(symbols.iter().any(|s| s.name == "Baz"));
    }

    #[test]
    fn test_graph_update_from_parse() {
        let source = "mod foo;\nuse std::io;\nfn main() {}";
        let parser = ASTParser::new();
        let result = parser.parse_source("test.rs", source);
        let mut graph = MemoryGraph::default();
        parser.update_memory_graph(&result, source, &mut graph);
        assert!(graph.node_count() > 3);
    }

    #[test]
    fn file_header_caps_its_symbol_list() {
        let mut src = String::new();
        for i in 0..50 {
            src.push_str(&format!("pub fn f{i}() {{}}\n"));
        }
        let parser = ASTParser::new();
        let result = parser.parse_source("t.rs", &src);
        let mut graph = MemoryGraph::default();
        parser.update_memory_graph(&result, &src, &mut graph);
        let header = &graph
            .nodes
            .get(&NodeId("file:t.rs".to_string()))
            .expect("file node")
            .content;
        // Default cap is 24 lines + a "(+N more)" marker, not all 50 signatures.
        assert!(
            header.contains("(+26 more)"),
            "header must note the omitted symbols: {header}"
        );
        assert!(
            !header.contains("f49"),
            "header must not list every symbol of a large file"
        );
    }

    #[test]
    fn symbol_span_brace_matches_multiline_and_single_line() {
        let src = "pub fn a() {\n    let x = 1;\n    x\n}\nfn b() {}\nconst K: u8 = 3;";
        let lines: Vec<&str> = src.lines().collect();
        assert_eq!(
            symbol_span(&lines, 1),
            (1, 4),
            "multi-line fn closes at its brace"
        );
        assert_eq!(
            symbol_span(&lines, 5),
            (5, 5),
            "one-line fn is a single line"
        );
        assert_eq!(
            symbol_span(&lines, 6),
            (6, 6),
            "a const ends at its semicolon line"
        );
    }

    #[test]
    fn symbol_span_keeps_nested_braces() {
        let src = "fn f() {\n    if x { a(); }\n    loop { break; }\n}";
        let lines: Vec<&str> = src.lines().collect();
        assert_eq!(
            symbol_span(&lines, 1),
            (1, 4),
            "nested braces must not close the span early"
        );
    }

    #[test]
    fn symbol_span_clamps_a_line_past_eof() {
        // A start line beyond EOF (a `--features syn-parser` span edge case) must
        // return an in-bounds span so `lines[start-1..end]` never panics.
        let lines = vec!["fn a() {}"];
        let (s, e) = symbol_span(&lines, 9);
        assert!(s >= 1 && s <= lines.len() && e >= s && e <= lines.len());
        // Empty input must not panic either.
        let empty: Vec<&str> = vec![];
        let _ = symbol_span(&empty, 3);
    }

    #[test]
    fn symbol_node_carries_its_span_and_file_node_is_a_header() {
        let src = "pub fn small() -> u8 { 7 }\npub fn big() {\n    let _ = 1;\n    let _ = 2;\n}";
        let parser = ASTParser::new();
        let result = parser.parse_source("t.rs", src);
        let mut graph = MemoryGraph::default();
        parser.update_memory_graph(&result, src, &mut graph);

        let small = graph
            .nodes
            .get(&NodeId("sym:t.rs:small".to_string()))
            .expect("small symbol node");
        assert_eq!(small.content, "pub fn small() -> u8 { 7 }");

        let big = graph
            .nodes
            .get(&NodeId("sym:t.rs:big".to_string()))
            .expect("big symbol node");
        assert!(big.content.starts_with("pub fn big()") && big.content.contains("let _ = 2;"));

        // The file node is a header (signatures), never the embedded bodies.
        let file = graph
            .nodes
            .get(&NodeId("file:t.rs".to_string()))
            .expect("file node");
        assert!(
            file.content.contains("pub fn small"),
            "header lists signatures"
        );
        assert!(
            !file.content.contains("let _ = 2;"),
            "file header must not embed symbol bodies"
        );
    }
}

/// Tests exercising the real-AST path (only compiled with `--features syn-parser`).
#[cfg(all(test, feature = "syn-parser"))]
mod syn_tests {
    use super::*;

    #[test]
    fn syn_captures_nested_module_tree() {
        let src = "pub mod outer { mod inner { fn deep() {} } }";
        let r = ASTParser::new().parse_source("t.rs", src);
        let outer = r
            .modules
            .iter()
            .find(|m| m.name == "outer")
            .expect("outer module");
        assert!(outer.is_public);
        assert!(
            outer.children.iter().any(|c| c.name == "inner"),
            "nested module must be a child (heuristic parser cannot do this)"
        );
        // The deeply-nested function is still surfaced into the flat symbol list.
        assert!(r.symbols.iter().any(|s| s.name == "deep"));
    }

    #[test]
    fn syn_captures_multiline_signature() {
        // The `fn` line does not end in `{`, so the line parser would miss it.
        let src = "fn wide(\n    a: i32,\n    b: i32,\n) -> i32 {\n    a + b\n}";
        let r = ASTParser::new().parse_source("t.rs", src);
        assert!(r
            .symbols
            .iter()
            .any(|s| s.name == "wide" && s.kind == SymbolKind::Function));
    }

    #[test]
    fn syn_expands_grouped_use() {
        let src = "use std::collections::{HashMap, HashSet};";
        let r = ASTParser::new().parse_source("t.rs", src);
        let paths: Vec<&str> = r
            .use_statements
            .iter()
            .map(|u| u.full_path.as_str())
            .collect();
        assert!(paths.contains(&"std::collections::HashMap"));
        assert!(paths.contains(&"std::collections::HashSet"));
    }

    #[test]
    fn syn_captures_impl_methods() {
        let src = "struct S;\nimpl S {\n    fn a(&self) {}\n    fn b(&self) {}\n}";
        let r = ASTParser::new().parse_source("t.rs", src);
        assert!(r
            .symbols
            .iter()
            .any(|s| s.name == "S" && s.kind == SymbolKind::Struct));
        assert!(r
            .symbols
            .iter()
            .any(|s| s.name == "a" && s.kind == SymbolKind::Function));
        assert!(r.symbols.iter().any(|s| s.name == "b"));
    }

    #[test]
    fn syn_captures_self_method_call_not_other_receivers() {
        // `self.helper()` is captured as `Self::helper`; an arbitrary receiver `x.helper()` is not
        // captured (unknown type — Slice 3+).
        let src = "struct T;\nimpl T {\n  fn run(&self, x: T) { self.helper(); x.helper(); }\n  fn helper(&self) {}\n}";
        let r = ASTParser::new().parse_source("t.rs", src);
        assert!(
            r.call_sites
                .iter()
                .any(|c| c.caller == "run" && c.callee == "Self::helper"),
            "self.helper() is captured as Self::helper, got {:?}",
            r.call_sites
                .iter()
                .map(|c| (&c.caller, &c.callee))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            r.call_sites
                .iter()
                .filter(|c| c.callee.ends_with("helper"))
                .count(),
            1,
            "only the self-receiver method call is captured, not x.helper()"
        );
    }

    #[test]
    fn syn_self_method_skips_deref_or_external_method() {
        // `self.len()` where the type has NO `len` method (it would come from Deref / a trait) is
        // NOT captured — so it can never be mislinked to the same-named free `len`. This is the
        // exact false-edge the own-method-set guard closes.
        let src =
            "struct W;\nimpl W { fn run(&self) -> usize { self.len() } }\nfn len() -> usize { 0 }";
        let r = ASTParser::new().parse_source("t.rs", src);
        assert!(
            !r.call_sites.iter().any(|c| c.callee == "Self::len"),
            "self.len() (not a method of W) is not captured, got {:?}",
            r.call_sites.iter().map(|c| &c.callee).collect::<Vec<_>>()
        );
    }

    #[test]
    fn syn_captures_screaming_snake_data_refs_only() {
        // A `SCREAMING_SNAKE` value reference is captured as a data-ref; snake_case (fn/local) and
        // PascalCase (type/variant) are not. The const *definition* is an item, not a value path,
        // so only the *use* inside `reader` is captured.
        let src = "const MAX_SIZE: usize = 10;\nfn reader() -> usize { let _c = Config; MAX_SIZE + helper() }\nfn helper() -> usize { 0 }\nstruct Config;";
        let r = ASTParser::new().parse_source("t.rs", src);
        assert!(
            r.data_refs
                .iter()
                .any(|d| d.reader == "reader" && d.name == "MAX_SIZE"),
            "SCREAMING_SNAKE reference is captured, got {:?}",
            r.data_refs.iter().map(|d| &d.name).collect::<Vec<_>>()
        );
        assert!(
            !r.data_refs
                .iter()
                .any(|d| d.name == "helper" || d.name == "Config"),
            "snake_case (fn) and PascalCase (type) references are not data-refs"
        );
    }

    #[test]
    fn data_flow_end_to_end_cross_file() {
        // The whole wiring through the real parser: config.rs defines `const MAX_LIMIT`, api.rs's
        // `reader` references it. update_memory_graph marks the const + records the ref, and
        // resolve_data_flow links them across files — the channel imports/calls never see.
        let p = ASTParser::new();
        let mut g = MemoryGraph::new(0.0, usize::MAX);
        for (path, src) in [
            ("src/config.rs", "pub const MAX_LIMIT: usize = 10;"),
            ("src/api.rs", "fn reader() -> usize { MAX_LIMIT + 1 }"),
        ] {
            let r = p.parse_source(path, src);
            p.update_memory_graph(&r, src, &mut g);
        }
        g.resolve_data_flow();
        let edges: Vec<(String, String)> = g
            .edges()
            .iter()
            .filter(|e| e.edge_type == EdgeType::DataFlow)
            .map(|e| (e.source.0.clone(), e.target.0.clone()))
            .collect();
        assert_eq!(
            edges,
            vec![(
                "sym:src/api.rs:reader".to_string(),
                "sym:src/config.rs:MAX_LIMIT".to_string()
            )],
            "reader → MAX_LIMIT cross-file data-flow edge, end to end"
        );
    }

    #[test]
    fn syn_data_refs_skip_locally_bound_names() {
        // A `SCREAMING_SNAKE` name bound locally — a parameter, a fn-local `const`, or a `let` — is
        // NOT a data-ref: it denotes the local, so resolving it global-unique to a same-named global
        // would be a false edge. Only the genuinely-free `MAX_LIMIT` is captured.
        let src = "fn r(PARAM_X: u8) -> u8 {\n  const LOCAL_C: u8 = 1;\n  let LET_V = 2u8;\n  PARAM_X + LOCAL_C + LET_V + MAX_LIMIT\n}";
        let r = ASTParser::new().parse_source("t.rs", src);
        let names: Vec<String> = r.data_refs.iter().map(|d| d.name.clone()).collect();
        assert!(
            names.contains(&"MAX_LIMIT".to_string()),
            "the free global reference is captured, got {names:?}"
        );
        for bound in ["PARAM_X", "LOCAL_C", "LET_V"] {
            assert!(
                !names.iter().any(|n| n == bound),
                "{bound} is locally bound — it must not be a data-ref (got {names:?})"
            );
        }
    }

    #[test]
    fn syn_captures_free_function_call_sites() {
        let src = "fn caller() { helper(); ns::deep(); recur(); }\nfn helper() {}\nfn recur() { recur(); }";
        let r = ASTParser::new().parse_source("t.rs", src);
        // bare free-function calls are captured with their caller; a qualified call keeps its full
        // `::`-joined path (Slice 2). Method calls `x.bar()` are still not captured (Slice 3).
        assert!(r
            .call_sites
            .iter()
            .any(|c| c.caller == "caller" && c.callee == "helper"));
        assert!(r
            .call_sites
            .iter()
            .any(|c| c.caller == "recur" && c.callee == "recur"));
        assert!(
            r.call_sites
                .iter()
                .any(|c| c.caller == "caller" && c.callee == "ns::deep"),
            "a qualified-path call keeps its full path (Slice 2)"
        );
    }

    #[test]
    fn syn_falls_back_on_invalid_syntax() {
        // Not valid Rust → syn returns None → heuristic parser handles it, no panic.
        let src = "fn broken( this is not rust {{{";
        let r = ASTParser::new().parse_source("t.rs", src);
        // Should not panic; result is whatever the heuristic produced.
        let _ = r.symbols.len();
    }

    #[test]
    fn syn_ignores_commented_out_code() {
        let src = "fn real() {}\n// fn commented() {}\n/* fn blocked() {} */";
        let r = ASTParser::new().parse_source("t.rs", src);
        assert!(r.symbols.iter().any(|s| s.name == "real"));
        assert!(!r.symbols.iter().any(|s| s.name == "commented"));
        assert!(!r.symbols.iter().any(|s| s.name == "blocked"));
    }
}
