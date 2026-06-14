use crate::memory::{EdgeType, MemoryGraph, NodeId, NodeType};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParseResult {
    pub file_path: String,
    pub file_hash: String,
    pub modules: Vec<ModuleDecl>,
    pub use_statements: Vec<UseStatement>,
    pub symbols: Vec<Symbol>,
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
pub struct ASTParser {
    file_hashes: HashMap<String, String>,
}

impl ASTParser {
    pub fn new() -> Self {
        Self {
            file_hashes: HashMap::new(),
        }
    }

    pub fn parse_source(&mut self, file_path: &str, source_code: &str) -> ParseResult {
        let hash = Self::compute_hash(source_code);
        self.file_hashes
            .insert(file_path.to_string(), hash.clone());

        let modules = Self::extract_modules(source_code);
        let use_statements = Self::extract_uses(source_code);
        let symbols = Self::extract_symbols(source_code);

        ParseResult {
            file_path: file_path.to_string(),
            file_hash: hash,
            generated_nodes: modules.len() + use_statements.len() + symbols.len(),
            generated_edges: use_statements.len() + modules.len(),
            modules,
            use_statements,
            symbols,
        }
    }

    pub fn update_memory_graph(&self, result: &ParseResult, graph: &mut MemoryGraph) {
        let file_id = NodeId(format!("file:{}", result.file_path));
        graph.upsert_node(
            file_id.clone(),
            result.file_path.clone(),
            format!("Source file at {}", result.file_path),
            NodeType::Module,
        );

        // Add module nodes
        for module in &result.modules {
            let mod_id = NodeId(format!("mod:{}:{}", result.file_path, module.name));
            graph.upsert_node(
                mod_id.clone(),
                module.name.clone(),
                format!(
                    "{}module {} at line {}",
                    if module.is_public { "pub " } else { "" },
                    module.name,
                    module.line
                ),
                NodeType::Module,
            );
            graph.add_edge(file_id.clone(), mod_id.clone(), 0.9, EdgeType::Contains);

            // Process nested modules
            self.add_module_tree(graph, &mod_id, &module.children, &result.file_path);
        }

        // Add use statements as edges
        for use_stmt in &result.use_statements {
            let use_id = NodeId(format!("use:{}:{}", result.file_path, use_stmt.full_path));
            graph.upsert_node(
                use_id.clone(),
                format!("use {}", use_stmt.full_path),
                format!("Import: {} at line {}", use_stmt.full_path, use_stmt.line),
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

        // Add symbol nodes
        for symbol in &result.symbols {
            let sym_id = NodeId(format!("sym:{}:{}", result.file_path, symbol.name));
            graph.upsert_node(
                sym_id.clone(),
                symbol.name.clone(),
                format!("{:?} {} at line {}", symbol.kind, symbol.name, symbol.line),
                NodeType::Symbol,
            );
            graph.add_edge(file_id.clone(), sym_id.clone(), 0.6, EdgeType::Contains);
        }
    }

    fn add_module_tree(
        &self,
        graph: &mut MemoryGraph,
        parent_id: &NodeId,
        children: &[ModuleDecl],
        file_path: &str,
    ) {
        for child in children {
            let child_id = NodeId(format!("mod:{}:{}", file_path, child.name));
            graph.upsert_node(
                child_id.clone(),
                child.name.clone(),
                format!(
                    "{}module {} at line {}",
                    if child.is_public { "pub " } else { "" },
                    child.name,
                    child.line
                ),
                NodeType::Module,
            );
            graph.add_edge(parent_id.clone(), child_id.clone(), 0.85, EdgeType::Contains);

            if !child.children.is_empty() {
                self.add_module_tree(graph, &child_id, &child.children, file_path);
            }
        }
    }

    fn compute_hash(source: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(source.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    pub fn get_file_hash(&self, file_path: &str) -> Option<&String> {
        self.file_hashes.get(file_path)
    }

    pub fn is_file_changed(&self, file_path: &str, new_source: &str) -> bool {
        let new_hash = Self::compute_hash(new_source);
        match self.get_file_hash(file_path) {
            Some(old_hash) => old_hash != &new_hash,
            None => true,
        }
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
                stripped
                    .strip_prefix("fn ")
                    .and_then(|rest| {
                        let name = rest
                            .split(|c: char| c == '(' || c == '<' || c == '{' || c == ';')
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
                stripped
                    .strip_prefix("pub fn ")
                    .and_then(|rest| {
                        let name = rest
                            .split(|c: char| c == '(' || c == '<' || c == '{' || c == ';')
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
                stripped
                    .strip_prefix("struct ")
                    .and_then(|rest| {
                        let name = rest
                            .split(|c: char| c == '<' || c == '{' || c == '(' || c == ';')
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
                stripped
                    .strip_prefix("pub struct ")
                    .and_then(|rest| {
                        let name = rest
                            .split(|c: char| c == '<' || c == '{' || c == '(' || c == ';')
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
                stripped
                    .strip_prefix("enum ")
                    .and_then(|rest| {
                        let name = rest
                            .split(|c: char| c == '<' || c == '{' || c == ';')
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
                stripped
                    .strip_prefix("pub enum ")
                    .and_then(|rest| {
                        let name = rest
                            .split(|c: char| c == '<' || c == '{' || c == ';')
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
                        .split(|c: char| c == '<' || c == '{' || c == ';')
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
                        .split(|c: char| c == '<' || c == '{' || c == ';')
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
                        .split(|c: char| c == '<' || c == '{' || c == ' ' || c == ';')
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
                        .split(|c: char| c == ':' || c == '=' || c == ';')
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
                        .split(|c: char| c == ':' || c == '=' || c == ';')
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
                        .split(|c: char| c == ':' || c == '=' || c == ';')
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
                        .split(|c: char| c == '=' || c == ';' || c == '<')
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
                        .split(|c: char| c == '{' || c == '(' || c == ' ')
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
        symbols.sort_by(|a, b| a.line.cmp(&b.line));
        symbols
    }
}

fn strip_comments(line: &str) -> String {
    let bytes = line.as_bytes();
    let mut in_string = false;
    let mut escaped = false;

    for (i, &ch) in bytes.iter().enumerate() {
        if escaped {
            escaped = false;
            continue;
        }
        if ch == b'\\' && in_string {
            escaped = true;
            continue;
        }
        if ch == b'"' {
            in_string = !in_string;
            continue;
        }
        if !in_string && ch == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
            return line[..i].trim_end().to_string();
        }
    }
    line.to_string()
}

impl Default for ASTParser {
    fn default() -> Self {
        Self::new()
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
        assert!(symbols.iter().any(|s| s.name == "main" && s.kind == SymbolKind::Function));
        assert!(symbols.iter().any(|s| s.name == "helper"));
        assert!(symbols.iter().any(|s| s.name == "public_fn"));
    }

    #[test]
    fn test_strip_comments() {
        assert_eq!(strip_comments("use foo; // bar"), "use foo;");
        assert_eq!(strip_comments("let x = \"//not_a_comment\"; // real"), "let x = \"//not_a_comment\";");
    }

    #[test]
    fn test_extract_structs_and_enums() {
        let source = "struct Foo;\nenum Bar { A, B }\npub struct Baz<T> {}";
        let symbols = ASTParser::extract_symbols(source);
        assert!(symbols.iter().any(|s| s.name == "Foo" && s.kind == SymbolKind::Struct));
        assert!(symbols.iter().any(|s| s.name == "Bar" && s.kind == SymbolKind::Enum));
        assert!(symbols.iter().any(|s| s.name == "Baz"));
    }

    #[test]
    fn test_graph_update_from_parse() {
        let source = "mod foo;\nuse std::io;\nfn main() {}";
        let mut parser = ASTParser::new();
        let result = parser.parse_source("test.rs", source);
        let mut graph = MemoryGraph::default();
        parser.update_memory_graph(&result, &mut graph);
        assert!(graph.node_count() > 3);
    }
}
