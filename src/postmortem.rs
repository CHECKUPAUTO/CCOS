//! # Post-mortem — an interactive time-travel debugger for an agent session
//!
//! The [`agent_session`](crate::agent_session) module makes a cognitive timeline
//! *replayable*; this module makes it *walkable by hand*. Point it at a persisted
//! workspace (the `<workspace>.oplog` written by `ccos mcp`) — or the built-in
//! [`demo_session`] of a session that drifts — and step through the agent's
//! history: move a cursor across the timeline, inspect the recalled context window
//! at any past step, and diff two points to see how the working set drifted as
//! failures accumulated.
//!
//! It is a thin REPL over [`AgentSession::replay_to`] /
//! [`AgentSession::recall_what_if`]: every command reconstructs state
//! deterministically from the op-log, so the post-mortem is exact and side-effect
//! free (it never mutates the session).
//!
//! Run with `ccos postmortem [workspace.ccos]`. Commands: `timeline`, `goto N`,
//! `next`/`prev`, `recall [budget]`, `around <anchor> [budget]`, `task <text…>`,
//! `diff A B`, `stats`, `help`, `quit`.

use crate::agent_session::AgentSession;
use crate::external_memory::{ExternalMemory, Recall, RecallWindow};
use crate::memory::{MemoryGraph, NodeId};

/// What a [`Debugger::command`] decides the REPL should do next.
pub enum Outcome {
    /// Print this text (may be empty — print nothing) and keep going.
    Print(String),
    /// Leave the debugger.
    Quit,
}

const HELP: &str = "\
commands:
  timeline | tl            the cognitive journal (▸ marks the cursor)
  goto N   | g N           move the cursor to step N (time-travel position)
  next | n   /  prev | p    move the cursor one step
  recall [budget] | r      working-set window as of the cursor
  around <anchor> [budget] | a   region window anchored on a node/file, at the cursor
  task <text…> | t         free-text recall at the cursor
  diff A B | d             which files entered/left the working set between A and B
  energy A B | e           node-level Δscore + failure-pressure between A and B
  stats | s                memory counts at the cursor
  help | h | ?             this help
  quit | q                 leave";

/// An interactive walk over an [`AgentSession`]'s recorded timeline.
pub struct Debugger {
    session: AgentSession,
    /// Current time-travel position (logical step `0..=len`).
    cursor: usize,
}

impl Debugger {
    /// Open a debugger positioned at the end of the timeline ("now").
    pub fn new(session: AgentSession) -> Self {
        let cursor = session.len();
        Debugger { session, cursor }
    }

    /// The current cursor position.
    pub fn cursor(&self) -> usize {
        self.cursor
    }

    /// Parse and run one REPL command line.
    pub fn command(&mut self, line: &str) -> Outcome {
        let mut it = line.split_whitespace();
        let Some(cmd) = it.next() else {
            return Outcome::Print(String::new());
        };
        let rest: Vec<&str> = it.collect();
        let num = |s: &str| s.parse::<usize>().ok();
        let len = self.session.len();

        match cmd {
            "quit" | "q" | "exit" => Outcome::Quit,
            "help" | "h" | "?" => Outcome::Print(HELP.to_string()),
            "timeline" | "tl" => Outcome::Print(self.render_timeline()),
            "len" => Outcome::Print(format!("{len} steps")),
            "stats" | "s" => Outcome::Print(self.render_at_cursor()),
            "goto" | "g" => match rest.first().and_then(|s| num(s)) {
                Some(n) => {
                    self.cursor = n.min(len);
                    Outcome::Print(self.render_at_cursor())
                }
                None => Outcome::Print("usage: goto <step>".to_string()),
            },
            "next" | "n" => {
                self.cursor = (self.cursor + 1).min(len);
                Outcome::Print(self.render_at_cursor())
            }
            "prev" | "p" => {
                self.cursor = self.cursor.saturating_sub(1);
                Outcome::Print(self.render_at_cursor())
            }
            "recall" | "r" => {
                let budget = rest.first().and_then(|s| num(s)).unwrap_or(2048);
                Outcome::Print(self.render_recall(Recall::working_set(), budget))
            }
            "around" | "a" => match rest.first() {
                Some(anchor) => {
                    let budget = rest.get(1).and_then(|s| num(s)).unwrap_or(2048);
                    Outcome::Print(self.render_recall(Recall::around(*anchor), budget))
                }
                None => Outcome::Print("usage: around <anchor> [budget]".to_string()),
            },
            "task" | "t" => {
                if rest.is_empty() {
                    Outcome::Print("usage: task <text…>".to_string())
                } else {
                    Outcome::Print(self.render_recall(Recall::task(rest.join(" ")), 2048))
                }
            }
            "diff" | "d" => {
                match (
                    rest.first().and_then(|s| num(s)),
                    rest.get(1).and_then(|s| num(s)),
                ) {
                    (Some(a), Some(b)) => Outcome::Print(self.render_diff(a, b)),
                    _ => Outcome::Print("usage: diff <step-a> <step-b>".to_string()),
                }
            }
            "energy" | "pressure" | "e" => {
                match (
                    rest.first().and_then(|s| num(s)),
                    rest.get(1).and_then(|s| num(s)),
                ) {
                    (Some(a), Some(b)) => Outcome::Print(self.render_energy(a, b)),
                    _ => Outcome::Print("usage: energy <step-a> <step-b>".to_string()),
                }
            }
            other => Outcome::Print(format!("unknown command '{other}' (try 'help')")),
        }
    }

    /// The journal, with a marker on the line at the cursor.
    fn render_timeline(&self) -> String {
        let mut out = String::new();
        for line in self.session.timeline() {
            // Lines read "t=<n>  <op>"; mark the one at the cursor.
            let at_cursor = line
                .strip_prefix("t=")
                .and_then(|r| r.split_whitespace().next())
                .and_then(|n| n.parse::<usize>().ok())
                == Some(self.cursor);
            out.push_str(if at_cursor { "▸ " } else { "  " });
            out.push_str(&line);
            out.push('\n');
        }
        out.push_str(&format!(
            "(cursor at step {} of {})",
            self.cursor,
            self.session.len()
        ));
        out
    }

    /// Memory counts at the cursor.
    fn render_at_cursor(&self) -> String {
        let st = self.session.replay_to(self.cursor).stats();
        format!(
            "cursor → t={}/{}   nodes={} edges={} files={}",
            self.cursor,
            self.session.len(),
            st.nodes,
            st.edges,
            st.files
        )
    }

    /// A recalled window as of the cursor (a time-travel what-if).
    fn render_recall(&self, recall: Recall, budget: usize) -> String {
        let win = self.session.recall_what_if(self.cursor, &recall, budget);
        let mut out = format!(
            "t={}/{}  {}  ({} items, ~{} tokens)\n",
            self.cursor,
            self.session.len(),
            win.strategy,
            win.items.len(),
            win.tokens
        );
        for it in win.items.iter().take(12) {
            out.push_str(&format!(
                "  {:.3}  {:<8}  {}\n",
                it.score,
                trunc(&it.kind, 8),
                it.uri
            ));
        }
        out.pop(); // drop the trailing newline
        out
    }

    /// How the working set drifted between two steps (files that entered / left).
    fn render_diff(&self, a: usize, b: usize) -> String {
        let files = |step: usize| -> std::collections::BTreeSet<String> {
            window_files(
                &self
                    .session
                    .recall_what_if(step, &Recall::working_set(), 8192),
            )
        };
        let (fa, fb) = (files(a), files(b));
        let entered: Vec<&String> = fb.difference(&fa).collect();
        let left: Vec<&String> = fa.difference(&fb).collect();
        let fmt = |v: &[&String]| {
            if v.is_empty() {
                "—".to_string()
            } else {
                v.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
            }
        };
        format!(
            "working set drift  t={a} → t={b}\n  entered: {}\n  left:    {}",
            fmt(&entered),
            fmt(&left)
        )
    }

    /// Node-level **energy/pressure drift**: how each AST node's causal score (and
    /// failure pressure) moved between two steps — the migration of "heat" through
    /// the graph as failures propagate. Sorted by the magnitude of the score delta.
    fn render_energy(&self, a: usize, b: usize) -> String {
        let (ma, mb) = (self.session.replay_to(a), self.session.replay_to(b));
        let (ga, gb) = (ma.graph(), mb.graph());
        let score = |g: &MemoryGraph, id: &NodeId| {
            g.nodes
                .get(id)
                .map(|n| g.compute_node_score(n))
                .unwrap_or(0.0)
        };
        let fail = |g: &MemoryGraph, id: &NodeId| {
            g.nodes.get(id).map(|n| n.failure_relevance).unwrap_or(0.0)
        };

        let mut ids: std::collections::BTreeSet<NodeId> = std::collections::BTreeSet::new();
        ids.extend(ga.nodes.keys().cloned());
        ids.extend(gb.nodes.keys().cloned());

        let mut rows: Vec<(f64, String)> = ids
            .iter()
            .map(|id| {
                let d = score(gb, id) - score(ga, id);
                (
                    d,
                    format!(
                        "{:+.3}  {:<30}  fail {:.2}→{:.2}",
                        d,
                        trunc(&id.0, 30),
                        fail(ga, id),
                        fail(gb, id)
                    ),
                )
            })
            .filter(|(d, _)| d.abs() > 1e-9)
            .collect();
        rows.sort_by(|x, y| {
            y.0.abs()
                .partial_cmp(&x.0.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut out = format!("node energy/pressure drift  t={a} → t={b}  (Δscore, top movers)\n");
        if rows.is_empty() {
            out.push_str("  (no change)");
            return out;
        }
        for (_, line) in rows.iter().take(12) {
            out.push_str("  ");
            out.push_str(line);
            out.push('\n');
        }
        out.pop();
        out
    }
}

/// The `file:` nodes of a window.
fn window_files(win: &RecallWindow) -> std::collections::BTreeSet<String> {
    win.items
        .iter()
        .filter(|i| i.uri.starts_with("file:"))
        .map(|i| i.uri.clone())
        .collect()
}

/// Truncate for column display.
fn trunc(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_string()
    } else {
        s.chars().take(n).collect()
    }
}

/// A built-in session that **drifts**: a small import chain plus an unrelated file,
/// then a failure on the entrypoint and a page-fault on the deep cause — so the
/// working set visibly migrates toward `db.rs` as the post-mortem walks forward.
pub fn demo_session() -> AgentSession {
    let mut s = AgentSession::new();
    s.ingest("src/db.rs", "pub fn timeout() -> i64 { 30 }\n");
    s.ingest(
        "src/repo.rs",
        "use crate::db;\npub fn fetch() -> i64 { db::timeout() }\n",
    );
    s.ingest(
        "src/api.rs",
        "use crate::repo;\npub fn handle() -> i64 { repo::fetch() }\n",
    );
    s.ingest(
        "src/util.rs",
        "pub fn fmt_date() -> &'static str { \"\" }\n",
    );
    s.recall(Recall::working_set(), 2048);
    // The drift begins: the entrypoint is failing…
    let _ = s.signal_failure("file:src/api.rs", 3);
    s.recall(Recall::around("file:src/api.rs"), 2048);
    // …and a panic points at the deep cause, pulling the hot set toward db.rs.
    let panic = "thread 'main' panicked at src/db.rs:1:14:\nattempt to add with overflow\n";
    s.page_fault(panic, 800);
    s.ingest(
        "src/api.rs",
        "use crate::repo;\npub fn handle() -> i64 { repo::fetch() + 1 }\n",
    );
    s
}

/// Run the interactive REPL until `quit`/EOF. Prompts and the banner go to stderr;
/// command output goes to stdout (so a piped session captures clean results).
pub fn serve(session: AgentSession) {
    use std::io::{BufRead, Write};
    const PROMPT: &str = "ccos⏪ ";
    let mut dbg = Debugger::new(session);
    eprintln!(
        "CCOS post-mortem — interactive time-travel debugger ({} steps). Type 'help'.",
        dbg.session.len()
    );
    eprint!("{PROMPT}");
    let _ = std::io::stderr().flush();

    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        match dbg.command(&line) {
            Outcome::Quit => break,
            Outcome::Print(text) => {
                let mut out = stdout.lock();
                if !text.is_empty() {
                    let _ = writeln!(out, "{text}");
                }
                let _ = out.flush();
            }
        }
        eprint!("{PROMPT}");
        let _ = std::io::stderr().flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn out(o: Outcome) -> String {
        match o {
            Outcome::Print(s) => s,
            Outcome::Quit => "<quit>".to_string(),
        }
    }

    #[test]
    fn demo_session_has_the_expected_length() {
        // 5 ingests/recalls + failure + recall + page_fault: a non-trivial timeline.
        assert!(demo_session().len() >= 8);
    }

    #[test]
    fn cursor_starts_at_now() {
        let d = Debugger::new(demo_session());
        assert_eq!(d.cursor(), d.session.len());
    }

    #[test]
    fn goto_then_recall_reflects_that_point_in_history() {
        let mut d = Debugger::new(demo_session());
        out(d.command("goto 1")); // only db.rs ingested so far
        let r = out(d.command("recall 2000"));
        assert!(r.contains("file:src/db.rs"), "step-1 window has db.rs: {r}");
        assert!(
            !r.contains("file:src/api.rs"),
            "step-1 predates api.rs: {r}"
        );
    }

    #[test]
    fn diff_shows_the_working_set_drift() {
        let mut d = Debugger::new(demo_session());
        let end = d.session.len();
        let report = out(d.command(&format!("diff 1 {end}")));
        assert!(report.contains("entered:"));
        // By the end the import chain has joined the hot set.
        assert!(
            report.contains("file:src/api.rs") || report.contains("file:src/repo.rs"),
            "drift surfaces the dependents: {report}"
        );
    }

    #[test]
    fn energy_shows_pressure_rising_on_the_cause() {
        let mut d = Debugger::new(demo_session());
        let end = d.session.len();
        // Before the failure (t=4) vs after the page-fault (end): the deep cause
        // db.rs should gain energy and failure pressure.
        let report = out(d.command(&format!("energy 4 {end}")));
        assert!(
            report.contains("file:src/db.rs"),
            "db.rs surfaces as a mover: {report}"
        );
        assert!(
            report.contains("fail 0.00→"),
            "pressure column present: {report}"
        );
    }

    #[test]
    fn next_and_prev_move_the_cursor() {
        let mut d = Debugger::new(demo_session());
        out(d.command("goto 2"));
        out(d.command("next"));
        assert_eq!(d.cursor(), 3);
        out(d.command("prev"));
        assert_eq!(d.cursor(), 2);
    }

    #[test]
    fn goto_clamps_past_the_end() {
        let mut d = Debugger::new(demo_session());
        out(d.command("goto 99999"));
        assert_eq!(d.cursor(), d.session.len());
    }

    #[test]
    fn timeline_marks_the_cursor() {
        let mut d = Debugger::new(demo_session());
        out(d.command("goto 1"));
        let tl = out(d.command("timeline"));
        assert!(tl.contains("▸ t=1"), "cursor marked at t=1: {tl}");
        assert!(tl.contains("cursor at step 1"));
    }

    #[test]
    fn quit_and_unknown_commands() {
        let mut d = Debugger::new(demo_session());
        assert!(matches!(d.command("quit"), Outcome::Quit));
        assert!(out(d.command("frobnicate")).contains("unknown command"));
        assert!(out(d.command("help")).contains("time-travel position"));
    }
}
