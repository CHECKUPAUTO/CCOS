# Campagne H — Le test décisif : bugs multi-fichiers (CCOS vs dump)

Le corpus de 16 runs a montré que les vrais bugs testés étaient **mono-fichier**, où
CCOS est à parité avec un dump du fichier. L'avantage de frugalité de CCOS est un
phénomène **multi-fichier** — non encore testé. Cette campagne le teste : des bugs où
**le `cargo test` panique dans un fichier, mais la cause est dans un *autre*.**

## La question décisive

> Quand le symptôme et la cause sont dans des fichiers différents, CCOS ramène-t-il la
> **cause** dans le contexte, en **bien moins de tokens** que dumper tout le `src/` ?

- **Baseline-symptôme** : dumper le seul fichier du panic → **rate la cause** (autre
  fichier).
- **Baseline-tout** : dumper tout `src/` → contient la cause mais **cher**.
- **CCOS** : `recall around <fichier-du-panic>` → la **région causale**, qui inclut le
  fichier-cause par l'arête `use crate::…`, frugalement. ← l'hypothèse à prouver.

## Protocole de mesure (model-free, robuste — par bug Hk)

1. `cargo new --lib hk`, écris les fichiers du bug (ci-dessous), `cd hk`.
2. `cargo test` → **capture la sortie rouge**. Note le **fichier-symptôme** (la
   localisation du panic, `src/X.rs:ligne`).
3. Ingère **tous** les fichiers `src/` dans un workspace neuf `corpus_H/Hk/` (l'agent
   les aurait lus).
4. `page_fault {output:"<sortie cargo test brute>"}` → CCOS pressurise le symptôme.
5. **CCOS** : `recall {around, anchor:"src/X.rs", budget:2048}` → enregistre :
   - `cause_in_window` : `file:src/<cause>.rs` est-il dans la fenêtre ? *(le booléen clé)*
   - `ccos_tokens` : tokens de la fenêtre.
6. **Baselines** : `all_tokens` = somme (chars/4) de tous les `src/*.rs` ;
   `symptom_tokens` = celui du seul fichier-symptôme.
7. Export : `ccos postmortem corpus_H/Hk --json > corpus_H/Hk.json`.

**Tableau visé** : `bug | sauts | cause_in_CCOS | ccos_tokens | all_tokens | ratio`.

**Critère de succès** : pour les bugs ≤ 3 sauts, la cause **est** dans la fenêtre CCOS,
à `ccos_tokens ≪ all_tokens`. Pour H4 (3 sauts) on teste la limite. Pour H5 (leurre),
CCOS inclut la **vraie** cause et **pas** le décoy.

*(Optionnel — moitié suffisante)* : envoie au LLM local la fenêtre CCOS vs le dump-tout
**au même budget**, « corrige le fichier en cause », et note par `cargo test`.

---

## Les bugs (copier-coller, prêts à injecter)

> Dans chaque crate, `src/lib.rs` déclare les `pub mod …` et porte le test. Le **panic
> surgit dans le fichier-symptôme**, la **cause est ailleurs**.

### H1 — Constante fausse, 1 saut  *(symptôme `writer.rs` → cause `config.rs`)*
```rust
// src/config.rs
pub fn buffer_size() -> usize { 0 }              // BUG : devrait être 8
// src/writer.rs
use crate::config;
pub fn render() -> u8 {
    let mut buf = vec![0u8; config::buffer_size()];
    buf[0] = 42;                                 // panic : index out of bounds (len 0)
    buf[0]
}
// src/lib.rs
pub mod config; pub mod writer;
#[cfg(test)] mod tests { #[test] fn renders() { assert_eq!(crate::writer::render(), 42); } }
```

### H2 — Off-by-one dans un helper, 1 saut  *(`access.rs` → `idx.rs`)*
```rust
// src/idx.rs
pub fn last_index(len: usize) -> usize { len }   // BUG : devrait être len - 1
// src/access.rs
use crate::idx;
pub fn last(v: &[i32]) -> i32 { v[idx::last_index(v.len())] }   // index out of bounds
// src/lib.rs
pub mod idx; pub mod access;
#[cfg(test)] mod tests { #[test] fn t() { assert_eq!(crate::access::last(&[1,2,3]), 3); } }
```

### H3 — Diviseur nul, 2 sauts  *(`engine.rs` → `math.rs` → `settings.rs`)*
```rust
// src/settings.rs
pub fn divisor() -> i64 { 0 }                    // BUG : devrait être 2
// src/math.rs
use crate::settings;
pub fn d() -> i64 { settings::divisor() }
// src/engine.rs
use crate::math;
pub fn run() -> i64 { 100 / math::d() }          // panic : divide by zero
// src/lib.rs
pub mod settings; pub mod math; pub mod engine;
#[cfg(test)] mod tests { #[test] fn t() { assert_eq!(crate::engine::run(), 50); } }
```

### H4 — Cause à 3 sauts (teste la limite de propagation)  *(`api.rs` → ctrl → repo → `store.rs`)*
```rust
// src/store.rs
pub fn capacity() -> usize { 0 }                 // BUG : devrait être 4
// src/repo.rs
use crate::store; pub fn cap() -> usize { store::capacity() }
// src/ctrl.rs
use crate::repo; pub fn limit() -> usize { repo::cap() }
// src/api.rs
use crate::ctrl;
pub fn first() -> u8 { let v = vec![0u8; ctrl::limit()]; v[0] }   // panic : index out of bounds
// src/lib.rs
pub mod store; pub mod repo; pub mod ctrl; pub mod api;
#[cfg(test)] mod tests { #[test] fn t() { assert_eq!(crate::api::first(), 0); } }
```

### H5 — Le leurre (structure vs lexical)  *(symptôme `handler.rs` → vraie cause `config.rs`, décoy `handler_helpers.rs`)*
```rust
// src/config.rs
pub fn timeout() -> u64 { 0 }                    // BUG : la vraie cause (devrait être 200)
// src/handler.rs
use crate::config;
pub fn handle() -> u64 { 1000 / config::timeout() }   // panic : divide by zero
// src/handler_helpers.rs
// DÉCOY : lexicalement proche de handler.rs, MAIS hors du chemin causal.
pub fn handle_format(s: &str) -> String { format!("handled: {s}") }
// src/lib.rs
pub mod config; pub mod handler; pub mod handler_helpers;
#[cfg(test)] mod tests { #[test] fn t() { assert_eq!(crate::handler::handle(), 5); } }
```
**Attendu (vérifié en preview)** : CCOS `around handler.rs` **atteint la vraie cause**
`config.rs` (via `use crate::config`) ✅. Mais le décoy `handler_helpers.rs` **est aussi
tiré dans la fenêtre** — car `pub mod handler_helpers;` dans `lib.rs` connecte les
modules frères en une seule région via la racine. C'est la **précision faible** connue du
paper (recall 0.97 / précision 0.06), confirmée sur du vrai code. La vraie question de H5
n'est donc pas « le décoy est-il exclu ? » (il ne l'est pas) mais : **sous budget serré
ou avec la pression d'échec, `config.rs` est-il classé au-dessus du décoy ?** (mesure le
rang/score des deux). Un baseline lexical, lui, attraperait le décoy *à la place* de la
cause — CCOS, au moins, a la cause.

---

## Preview sur fichiers-jouets (ce repo, binaire 0.3.0) — fait

Exécuté ici via le serveur MCP (workspace neuf par bug, `cargo test` réel →
`page_fault` → `recall around <symptôme>`, budget 2048). **Tous les bugs paniquent
bien dans le fichier-symptôme attendu.**

| Bug | sauts | cause dans CCOS ? | rang de la cause | ccos_tokens | all_tokens | ratio |
| --- | ----- | ----------------- | ---------------- | ----------- | ---------- | ----- |
| H1  | 1 | ✅ oui | **#1** (0.875) | 173 | 71 | 0.41× |
| H2  | 1 | ✅ oui | top (0.875) | 139 | 60 | 0.43× |
| H3  | 2 | ✅ oui | top (0.875) | 167 | 69 | 0.41× |
| H4  | 3 | ✅ oui (chaîne `api→ctrl→repo→store` entière) | présent (0.779/0.569) | 202 | 91 | 0.45× |
| H5  | 1 (+ décoy) | ✅ oui ; décoy **présent mais classé dernier** | **#1** (0.875) ; décoy 0.569 | 201 | 92 | 0.46× |

**Trois lectures, honnêtes :**

1. **Couverture — PROUVÉE à 1, 2 et 3 sauts.** La cause d'un *autre* fichier est dans
   la fenêtre à chaque fois, y compris la chaîne à 3 sauts de H4 (avec décroissance
   visible du score, mais présente). C'est exactement ce que le corpus mono-fichier
   n'avait jamais montré.
2. **Classement sous pression — meilleur que prévu.** En H5, le décoy lexical
   `handler_helpers.rs` *entre* bien dans la région (sur-connexion par la racine de
   module `pub mod`), **mais le score causal le classe bon dernier** ; la vraie cause
   `config.rs` est **#1**. Sous budget serré (200) l'ordre est
   `config.rs > handler.rs > … > handler_helpers.rs`. Un baseline lexical classerait le
   décoy *haut* (préfixe « handler »). Le score fait le tri que la topologie seule ne
   fait pas. ✅
3. **Frugalité — PAS démontrée sur des jouets (ratio 0.41–0.46×, CCOS *plus gros*).** Sur
   des fichiers de 60–90 tokens, la fenêtre duplique le source : un même fichier apparaît
   dans son nœud `file:`, ses nœuds `sym:` et ses nœuds `use:` (chacun portant **tout** le
   fichier — granularité fichier, pas symbole). Dumper `src/` est donc *moins* cher que la
   région. **La frugalité exige (a) de vrais gros fichiers** (où `all-src ≫ région`) **et
   (b) la granularité symbole** (qu'un nœud `sym:` ne porte que sa fonction). C'est le
   point qui relie directement H à l'item roadmap Q2.

➡️ **Conclusion preview** : le *quoi* (la cause multi-fichier est atteinte et bien classée)
est acquis sur des jouets ; le *combien* (frugalité) ne se mesure que sur du vrai code
volumineux. Je l'ai mesuré — section suivante.

## Réalité sur vrai code (le `src/` de CCOS, 32 fichiers, 130 287 tokens) — la correction

Le protocole H, appliqué non plus à des crates-jouets mais au **propre `src/` de CCOS**
(32 fichiers réels, vraies arêtes `use crate::`, fichiers de 700 à 17 000 tokens),
**renverse le résultat des jouets**. Le binaire 0.3.0, via MCP (ingest des 32 fichiers →
`signal_failure file:src/mcp.rs` → `recall around file:src/mcp.rs`) donne :

| profondeur | budget | symptôme dans la fenêtre ? | deps couvertes | ce que contient la fenêtre |
| ---------- | ------ | -------------------------- | -------------- | -------------------------- |
| **3 (défaut)** | 2 048 → 32 768 | ❌ non | **0/2** | 1 nœud = le **hub `memory.rs`** entier (9 747 tok) |
| 1 | 2 048 | ❌ non | 1/2 | 1 fichier entier (`external_memory.rs`) |
| 1 | **32 768** | ✅ oui | 2/2 | les 3 fichiers entiers (20 405 tok uniques) |

**Trois causes racines, chacune mesurée (plus des hypothèses) :**

1. **Granularité fichier — le blocant n°1 (Q2), désormais prouvé.** Chaque nœud
   `sym:`/`use:`/`mod:` porte **tout son fichier**. Un seul nœud (`sym:memory.rs:MemoryGraph`
   = 9 747 tok) **dépasse à lui seul un budget de 2 048**. La fenêtre ne tient donc qu'1
   nœud = 1 fichier dupliqué. La **région complète** (budget ∞) autour de *n'importe quel*
   ancre = **les 32 fichiers, ~2 000 000 tokens** pour 130 287 tokens uniques : un facteur
   **15× de duplication** (les ~6 symboles de `main.rs` portent chacun les 64 655 chars de
   `main.rs`). Tant que `sym:` ne porte pas seulement *sa* fonction, le budget est brûlé.
2. **Propagation à profondeur fixe — elle inonde les graphes denses. ✅ CORRIGÉ.**
   `signal_failure` sur **un** fichier pressurisait **518 / 1037 nœuds** (depth 3) : la
   moitié du dépôt, et le hub finissait par surclasser le symptôme. Correctif livré :
   **propagation consciente du degré** — un nœud *distribue* sa pression sur ses arêtes
   (`damp = failure_fanout / out_degree`, no-op sous `failure_fanout=6`) au lieu de la
   *répliquer*, plus un **cutoff au plancher de paging** (on arrête de relayer une pression
   qui ne peut plus rien pager). Résultat mesuré : **flood 518 → 37 nœuds (14×)** ; le
   **défaut `depth=3` atteint maintenant le symptôme + 2/2 deps en 2035 tokens** (identique
   à depth=1 — la tension profondeur/densité est **résolue**) ; le symptôme `mcp.rs` repasse
   **#1** (0.850) et le hub `memory.rs` sort de la fenêtre. La **réuse-clé** : sur une chaîne
   creuse `a→b→c→d` (degré 1), `damp=1`, donc la cause à 3 sauts `d.rs` est **toujours**
   atteinte — le degré-conscient préserve la portée profonde là où baisser la profondeur
   l'aurait perdue. (`CCOS_FAILURE_FANOUT` ajustable.)
3. **Domination du hub & bruit de composante. ✅ CORRIGÉ.** Diagnostic initial : `memory.rs`
   (utilisé par presque tout) surclassait le symptôme. La **partie hub a été réglée par #2**
   (le symptôme `mcp.rs` est repassé #1). Restait une imprécision **mesurée plus finement** :
   la région de `mcp.rs` est **31 fichiers sur 32** (le vrai graphe `use crate::` de CCOS est
   quasi entièrement connexe). Ce **n'était donc pas** un pont `dep:`/std (le clustering les
   exclut déjà) mais une **composante dense sans terme de proximité** : sans pression d'échec
   forte, la *récence* fait qu'un nœud lointain (`use:commands_demo`, ~3 sauts) **égalait** les
   vraies deps (1 saut). Correctif livré : **décroissance par proximité** dans `around`/`task`
   — score `×= proximity_decay^(distance au point d'ancrage)`, distance par BFS bidirectionnel
   **qui ne relaie pas à travers les hubs `dep:`**. Résultat : le bruit `commands_demo` (rang 3
   à 0.444) **sort** ; le top devient symptôme → ses 2 deps réelles → ses propres symboles. Le
   symptôme + 2/2 deps tiennent toujours en 2045 tokens, et la chaîne creuse garde ses 3 sauts.
   (`CCOS_PROXIMITY_DECAY` / `CCOS_PROXIMITY_HOPS` ajustables.)

**Le verdict frugalité, sans fard (diagnostic d'origine)** : les 3 fichiers pertinents =
**20 405 tokens**. CCOS exigeait un budget de **32 768** pour les livrer (taxe de
duplication) **et seulement** à `depth=1` ; au **défaut `depth=3`, il ne les atteignait sous
aucun budget testé** — il servait le hub.

> **MISE À JOUR (les trois causes corrigées).** Granularité symbole (#1) + propagation
> degré-consciente (#2) + proximité d'ancrage (#3). La même mesure donne maintenant, au
> **réglage par défaut, budget 2048** : **symptôme #1, ses 2 deps réelles, ses symboles —
> en ~2045 tokens, sans bruit de composante.** Bilan avant→après : region complète 15× → 1.2× ;
> flood 518 → 37 ; couverture deps 0/2 → 2/2 ; le nœud non pertinent `commands_demo` évincé.
> CCOS livre les bons fichiers **et dit lesquels**, frugalement, par défaut, sur du vrai code.

Ce n'était pas un échec de la machinerie (event-sourcing, time-travel, persistance, audit
fonctionnent) : c'était l'**assemblage de contexte** — le cœur de la thèse — qui ne
survivait pas à l'échelle réelle. **Les trois leviers sont livrés**, chacun avec test de
régression et chiffres avant/après ; ces items sont passés de « roadmap spéculative » à
« corrigés et mesurés ». Reste à **confirmer sur un autre dépôt** (Thor : ripgrep/bat/fd) que
le gain n'est pas propre au `src/` de CCOS.

➡️ Le test de Thor sur un **autre** vrai dépôt (ripgrep/bat/fd) reste utile comme
confirmation indépendante, mais la conclusion est déjà nette ici.

## Grille de résultats à remplir (vrai code, Thor)

Refais les 5 bugs **mais greffés sur de vrais fichiers volumineux** (insère le mauvais
constant dans un module réel de `ripgrep`/`bat`/`fd` ; le symptôme panique ailleurs). Là,
`all_tokens` = plusieurs milliers, et on verra si la région reste petite.

| Bug | sauts symptôme→cause | cause dans CCOS ? | ccos_tokens | all_tokens | ratio |
| --- | -------------------- | ----------------- | ----------- | ---------- | ----- |
| H1  | 1 | ? | ? | ? | ? |
| H2  | 1 | ? | ? | ? | ? |
| H3  | 2 | ? | ? | ? | ? |
| H4  | 3 | ? | ? | ? | ? |
| H5  | 1 (+ décoy) | ? (rang cause vs décoy) | ? | ? | ? |

**Lecture** : si, sur du vrai code, CCOS ramène la cause d'un autre fichier à quelques
centaines de tokens là où le dump-tout en coûte plusieurs milliers — **c'est la preuve de
frugalité que le corpus n'avait pas encore montrée.** Si le ratio reste ≤ 1 même sur de
gros fichiers, **la granularité symbole (Q2) devient l'item bloquant**, pas un détail.

Ramène `corpus_H/` complet — on le déroule ensemble.
