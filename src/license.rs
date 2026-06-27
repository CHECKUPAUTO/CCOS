//! **Zero-knowledge, offline license gating** for CCOS *Pro* features.
//!
//! Design constraints (by the project owner):
//! - **Nothing leaves the host.** No network calls, no telemetry, no phone-home. A license is a
//!   locally-verified, signed token — the engine holds a **public key**, the vendor signs with the
//!   matching **private key**, and verification is a pure offline signature check. A customer can
//!   run CCOS fully air-gapped.
//! - **The core is never gated, never degraded.** Ingestion, the causal graph, and the Q-Page
//!   belief / decay / propagation primitives are always available in the free **community** tier. An
//!   unlicensed engine is *not* made "vague" or silently wrong — it simply **gates the advanced
//!   features and logs, explicitly, how to obtain a key**. (This is the fail-closed / announced
//!   model — the deliberately-deceptive "degrade confidence under an invalid license" idea is *not*
//!   implemented here, by design.)
//! - **The dollar funds the user's own control surface**, not surveillance: the Pro features are
//!   per-source authority weighting, cognitive-tension visualization in the logs, and audit-report
//!   generation — tools the operator points *at their own system*.
//!
//! This module is the **gate**: tiers, the feature set, and the explicit-logging policy. The gate and
//! the verifier are **pure** — the single [`load_license_blob`] helper is the one explicit, opt-in I/O
//! entry point (an env var or a local file; never a network call). The public-key signature check
//! ([`LicenseVerifier`]) is pluggable; the bundled ed25519 verifier ([`Ed25519Verifier`]) is provided
//! behind the `license` cargo feature so the default build pulls in no cryptography.

use std::fmt;

/// A licensed (*Pro*) capability. The **core** of CCOS is never one of these — only advanced,
/// operator-facing tooling is gated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Feature {
    /// Per-source **custom authority weighting** (vs. the uniform default authority).
    CustomAuthorityWeights,
    /// **Cognitive-tension visualization** in the logs (rendering `qbelief` conflict per claim).
    TensionVisualization,
    /// **Audit-report generation** (belief / conflict / provenance of the knowledge base).
    AuditReports,
}

impl Feature {
    /// Stable human-readable name (used in logs and errors).
    pub fn name(self) -> &'static str {
        match self {
            Feature::CustomAuthorityWeights => "custom-authority-weights",
            Feature::TensionVisualization => "tension-visualization",
            Feature::AuditReports => "audit-reports",
        }
    }

    /// Every Pro feature — for enumerating the gate.
    pub const ALL: [Feature; 3] = [
        Feature::CustomAuthorityWeights,
        Feature::TensionVisualization,
        Feature::AuditReports,
    ];
}

impl fmt::Display for Feature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// The active licensing tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// Free — the full core, no Pro features.
    Community,
    /// Licensed — Pro features unlocked.
    Pro,
}

/// A **verified** license. Only a [`LicenseVerifier`] produces one (from a signed token); it is never
/// fabricated from untrusted input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct License {
    /// Who the license was issued to (for the audit trail / logs).
    pub licensee: String,
    /// Expiry in unix seconds; `None` = perpetual.
    pub expires_at: Option<u64>,
}

impl License {
    /// Whether the license is still in force at `now` (unix seconds).
    pub fn is_valid_at(&self, now: u64) -> bool {
        self.expires_at.is_none_or(|e| now <= e)
    }
}

/// Why a Pro action was refused (or how verification failed). A refusal **never** degrades the core.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LicenseError {
    /// No license present — running in the free community tier.
    NoLicense,
    /// The license is past its expiry.
    Expired,
    /// Malformed token or bad signature — never trusted.
    Invalid(String),
    /// A Pro `feature` was requested without an active license.
    FeatureLocked(Feature),
}

impl fmt::Display for LicenseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LicenseError::NoLicense => write!(f, "no license present (community tier)"),
            LicenseError::Expired => write!(f, "license expired"),
            LicenseError::Invalid(why) => write!(f, "invalid license: {why}"),
            LicenseError::FeatureLocked(feat) => write!(
                f,
                "the Pro feature '{feat}' requires an active license (the core is unaffected)"
            ),
        }
    }
}

impl std::error::Error for LicenseError {}

/// Verifies a license **entirely locally** — no network, no telemetry, no data leaves the host. An
/// implementation MUST be pure (an offline signature + format + expiry check only): this is the
/// zero-knowledge contract that lets a customer run CCOS air-gapped. `now` is unix seconds, supplied
/// by the caller so the verifier itself reads no clock.
pub trait LicenseVerifier {
    fn verify(&self, blob: &[u8], now: u64) -> Result<License, LicenseError>;
}

/// The default verifier: it holds no public key, so every input is unlicensed → community tier. It
/// pulls in no cryptography; the real public-key (`ed25519`) verifier lives behind the `license`
/// cargo feature and also implements [`LicenseVerifier`].
#[derive(Debug, Default, Clone, Copy)]
pub struct CommunityVerifier;

impl LicenseVerifier for CommunityVerifier {
    fn verify(&self, _blob: &[u8], _now: u64) -> Result<License, LicenseError> {
        Err(LicenseError::NoLicense)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Offline ed25519 verifier + signed-token format (behind the `license` feature)
// ─────────────────────────────────────────────────────────────────────────────

/// The vendor's **ed25519 public key**, baked into the binary. A license token is signed by the
/// matching private key — held only by the vendor, never in this tree — and verification is a pure
/// offline signature check against this constant. A deployment with its own key replaces these 32
/// bytes with its own public key (its private half then signs that deployment's licenses). An unset
/// value (the placeholder below) or any non-point makes [`Ed25519Verifier`] license **nothing** →
/// community tier, so a build that never set a real key fails **closed**, never open.
///
/// Regenerate with `cargo run --features license --example license_sign keygen`.
#[cfg(feature = "license")]
pub const LICENSE_PUBLIC_KEY: [u8; 32] = [0u8; 32];

/// The signed-token payload: who, and until when. Compact-JSON + base64url is the token's first
/// segment.
#[cfg(feature = "license")]
#[derive(serde::Serialize, serde::Deserialize)]
struct TokenPayload {
    /// Licensee (organisation / deployment name) — surfaced in the audit log.
    licensee: String,
    /// Expiry, unix seconds. Absent = perpetual.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    exp: Option<u64>,
}

/// URL-safe base64 **without padding** (RFC 4648 §5: `-`/`_`, no `=`). Hand-rolled so the `license`
/// feature's only new dependency is the ed25519 primitive — the same reason CCOS hand-rolls its hex.
#[cfg(feature = "license")]
fn b64url_encode(bytes: &[u8]) -> String {
    const A: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = *chunk.get(1).unwrap_or(&0) as u32;
        let b2 = *chunk.get(2).unwrap_or(&0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(A[(n >> 18) as usize & 63] as char);
        out.push(A[(n >> 12) as usize & 63] as char);
        if chunk.len() > 1 {
            out.push(A[(n >> 6) as usize & 63] as char);
        }
        if chunk.len() > 2 {
            out.push(A[n as usize & 63] as char);
        }
    }
    out
}

/// Inverse of [`b64url_encode`]. `None` on any non-alphabet byte or a truncated group.
#[cfg(feature = "license")]
fn b64url_decode(s: &str) -> Option<Vec<u8>> {
    let val = |c: u8| -> Option<u32> {
        Some(match c {
            b'A'..=b'Z' => (c - b'A') as u32,
            b'a'..=b'z' => (c - b'a' + 26) as u32,
            b'0'..=b'9' => (c - b'0' + 52) as u32,
            b'-' => 62,
            b'_' => 63,
            _ => return None,
        })
    };
    let s = s.as_bytes();
    let mut out = Vec::with_capacity(s.len() / 4 * 3 + 3);
    for chunk in s.chunks(4) {
        if chunk.len() < 2 {
            return None; // a lone trailing char encodes no full byte
        }
        let mut n = 0u32;
        for (i, &c) in chunk.iter().enumerate() {
            n |= val(c)? << (18 - 6 * i);
        }
        out.push((n >> 16) as u8);
        if chunk.len() > 2 {
            out.push((n >> 8) as u8);
        }
        if chunk.len() > 3 {
            out.push(n as u8);
        }
    }
    Some(out)
}

/// Sign a license token with the 32-byte ed25519 **signing seed** (private key material): emits
/// `base64url(payload).base64url(signature)`, the signature taken over the first segment's ASCII
/// bytes (JWT convention). Vendor-side tooling and the tests use this; the engine only ever *verifies*.
#[cfg(feature = "license")]
pub fn sign_token(signing_seed: &[u8; 32], licensee: &str, exp: Option<u64>) -> String {
    use ed25519_dalek::{Signer, SigningKey};
    let payload = TokenPayload {
        licensee: licensee.to_string(),
        exp,
    };
    let json = serde_json::to_vec(&payload).expect("payload serialises");
    let signing_input = b64url_encode(&json);
    let sk = SigningKey::from_bytes(signing_seed);
    let sig = sk.sign(signing_input.as_bytes());
    format!("{signing_input}.{}", b64url_encode(&sig.to_bytes()))
}

/// The offline **ed25519 license verifier**: a pure signature + format check against a public key
/// (the baked-in [`LICENSE_PUBLIC_KEY`] by default). No I/O, no clock, no network — the zero-knowledge
/// contract that lets a customer run air-gapped. An unset / invalid embedded key licenses nothing.
#[cfg(feature = "license")]
#[derive(Clone)]
pub struct Ed25519Verifier {
    key: Option<ed25519_dalek::VerifyingKey>,
}

#[cfg(feature = "license")]
impl Default for Ed25519Verifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "license")]
impl Ed25519Verifier {
    /// Verifier bound to the baked-in vendor key ([`LICENSE_PUBLIC_KEY`]). The all-zero placeholder
    /// shipped in this open tree means *no key was set* → it licenses nothing, so the default build is
    /// **fail-closed**: a deployment must paste its own public key (via the `license_sign keygen` tool)
    /// before any token can unlock Pro.
    pub fn new() -> Self {
        if LICENSE_PUBLIC_KEY == [0u8; 32] {
            return Self { key: None };
        }
        Self::with_public_key(&LICENSE_PUBLIC_KEY)
    }

    /// Verifier bound to an explicit public key — the tests sign with a throwaway keypair and verify
    /// against its public half, never the embedded vendor key.
    pub fn with_public_key(public_key: &[u8; 32]) -> Self {
        Self {
            key: ed25519_dalek::VerifyingKey::from_bytes(public_key).ok(),
        }
    }
}

#[cfg(feature = "license")]
impl LicenseVerifier for Ed25519Verifier {
    /// Verify `blob` (a `payload.sig` token, tolerant of trailing whitespace from a file) and return
    /// the encoded [`License`] on a good signature. Temporal validity is **not** checked here — a
    /// signature-valid but expired token still parses, and [`Licensing::tier`] reports it as community
    /// (so the CLI can say *expired on X* while keeping the licensee for the audit log). `now` is thus
    /// unused; the check is pure signature + format.
    fn verify(&self, blob: &[u8], _now: u64) -> Result<License, LicenseError> {
        let key = self
            .key
            .as_ref()
            .ok_or_else(|| LicenseError::Invalid("no embedded public key".into()))?;
        let token = std::str::from_utf8(blob)
            .map_err(|_| LicenseError::Invalid("token is not UTF-8".into()))?
            .trim();
        let (signing_input, sig_b64) = token
            .split_once('.')
            .ok_or_else(|| LicenseError::Invalid("token is not payload.signature".into()))?;
        let sig_bytes = b64url_decode(sig_b64)
            .filter(|s| s.len() == 64)
            .ok_or_else(|| LicenseError::Invalid("signature is not 64 base64url bytes".into()))?;
        let sig_array: [u8; 64] = sig_bytes.try_into().expect("length checked to be 64");
        let sig = ed25519_dalek::Signature::from_bytes(&sig_array);
        use ed25519_dalek::Verifier;
        key.verify(signing_input.as_bytes(), &sig)
            .map_err(|_| LicenseError::Invalid("bad signature".into()))?;
        let json = b64url_decode(signing_input)
            .ok_or_else(|| LicenseError::Invalid("payload is not base64url".into()))?;
        let payload: TokenPayload = serde_json::from_slice(&json)
            .map_err(|e| LicenseError::Invalid(format!("payload JSON: {e}")))?;
        Ok(License {
            licensee: payload.licensee,
            expires_at: payload.exp,
        })
    }
}

/// Load a license token from the host — **the one explicit I/O entry point** (the gate and verifier
/// are pure). Order: the `$CCOS_LICENSE` env var (the token text inline — handy in containers / CI),
/// else the file at `$CCOS_LICENSE_FILE`, else the XDG default `$XDG_CONFIG_HOME/ccos/license` (or
/// `~/.config/ccos/license`). Returns `None` when nothing is present → the community tier. Never
/// fails: an unreadable or absent file is simply "no license".
pub fn load_license_blob() -> Option<Vec<u8>> {
    if let Ok(token) = std::env::var("CCOS_LICENSE") {
        let token = token.trim();
        if !token.is_empty() {
            return Some(token.as_bytes().to_vec());
        }
    }
    let path = std::env::var_os("CCOS_LICENSE_FILE")
        .map(std::path::PathBuf::from)
        .or_else(default_license_path)?;
    std::fs::read(path).ok()
}

/// `$XDG_CONFIG_HOME/ccos/license`, else `$HOME/.config/ccos/license`.
fn default_license_path() -> Option<std::path::PathBuf> {
    use std::path::PathBuf;
    if let Some(xdg) = std::env::var_os("XDG_CONFIG_HOME") {
        return Some(PathBuf::from(xdg).join("ccos").join("license"));
    }
    std::env::var_os("HOME").map(|home| {
        PathBuf::from(home)
            .join(".config")
            .join("ccos")
            .join("license")
    })
}

/// Current unix time in seconds — a convenience for callers that gate features (the verifier itself
/// never reads a clock; `now` is always passed in). Saturates to 0 before the epoch.
pub fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

impl Licensing {
    /// Determine the active licensing from the host: load any local token ([`load_license_blob`]) and
    /// verify it with the compiled-in verifier. With the `license` feature that is the offline
    /// [`Ed25519Verifier`]; without it there is no verifier, so the result is always the community
    /// tier (the core is never gated). Pure beyond the single [`load_license_blob`] read; the one
    /// place CLI commands and the session obtain their licensing.
    pub fn detect(now: u64) -> Self {
        let Some(blob) = load_license_blob() else {
            return Self::community();
        };
        #[cfg(feature = "license")]
        {
            Self::from_blob(&Ed25519Verifier::new(), &blob, now)
        }
        #[cfg(not(feature = "license"))]
        {
            let _ = (blob, now);
            Self::community()
        }
    }
}

/// Runtime license state and the **feature gate**. Holds an optional verified [`License`] and never
/// performs I/O itself. Cloneable and cheap; a single instance is threaded through the engine.
#[derive(Debug, Clone, Default)]
pub struct Licensing {
    license: Option<License>,
}

impl Licensing {
    /// The free community tier — the full core, no Pro features.
    pub fn community() -> Self {
        Self { license: None }
    }

    /// A licensed engine from an already-verified [`License`] (produced by a [`LicenseVerifier`]).
    pub fn licensed(license: License) -> Self {
        Self {
            license: Some(license),
        }
    }

    /// Verify `blob` with `verifier` and build the licensing state. On **any** failure it falls back
    /// to the community tier — a missing or invalid license must never break the core, only gate Pro.
    pub fn from_blob(verifier: &impl LicenseVerifier, blob: &[u8], now: u64) -> Self {
        match verifier.verify(blob, now) {
            Ok(license) => Self::licensed(license),
            Err(_) => Self::community(),
        }
    }

    /// The active tier at `now` (an expired license reads as community).
    pub fn tier(&self, now: u64) -> Tier {
        match &self.license {
            Some(l) if l.is_valid_at(now) => Tier::Pro,
            _ => Tier::Community,
        }
    }

    /// The licensee, if any (for the audit log).
    pub fn licensee(&self) -> Option<&str> {
        self.license.as_ref().map(|l| l.licensee.as_str())
    }

    /// Whether `feature` is unlocked at `now`. Every advanced feature is Pro in this design, so this
    /// is simply "is the tier Pro".
    pub fn allows(&self, _feature: Feature, now: u64) -> bool {
        matches!(self.tier(now), Tier::Pro)
    }

    /// **Gate a Pro `feature`.** `Ok(())` when unlocked; otherwise it emits one explicit system-log
    /// line — stating that the core is fully functional and that an annual, **locally-verified**
    /// license unlocks the feature — and returns [`LicenseError::FeatureLocked`]. There is **no**
    /// silent downgrade and no side effect beyond that log: the caller decides what to do with the
    /// refusal (typically: skip the Pro path, keep the core result).
    pub fn require(&self, feature: Feature, now: u64) -> Result<(), LicenseError> {
        if self.allows(feature, now) {
            Ok(())
        } else {
            eprintln!(
                "[ccos] license: Pro feature '{feature}' is locked — the core (ingestion, causal \
                 graph, Q-Page belief/decay/propagation) is fully functional. An annual license \
                 unlocks it and is verified entirely locally (no data leaves your infrastructure)."
            );
            Err(LicenseError::FeatureLocked(feature))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NOW: u64 = 1_000;

    fn license(expires_at: Option<u64>) -> License {
        License {
            licensee: "acme-corp".to_string(),
            expires_at,
        }
    }

    #[test]
    fn community_gates_every_pro_feature_without_degrading() {
        let l = Licensing::community();
        assert_eq!(l.tier(NOW), Tier::Community);
        assert_eq!(l.licensee(), None);
        for f in Feature::ALL {
            assert!(!l.allows(f, NOW));
            assert_eq!(l.require(f, NOW), Err(LicenseError::FeatureLocked(f)));
        }
    }

    #[test]
    fn valid_license_unlocks_every_pro_feature() {
        let l = Licensing::licensed(license(Some(NOW + 100)));
        assert_eq!(l.tier(NOW), Tier::Pro);
        assert_eq!(l.licensee(), Some("acme-corp"));
        for f in Feature::ALL {
            assert!(l.allows(f, NOW));
            assert!(l.require(f, NOW).is_ok());
        }
    }

    #[test]
    fn expired_license_falls_back_to_community() {
        let l = Licensing::licensed(license(Some(NOW - 1)));
        assert_eq!(l.tier(NOW), Tier::Community);
        assert!(!l.allows(Feature::AuditReports, NOW));
    }

    #[test]
    fn perpetual_license_never_expires() {
        let l = Licensing::licensed(license(None));
        assert_eq!(l.tier(u64::MAX), Tier::Pro);
    }

    #[test]
    fn community_verifier_is_zero_knowledge_and_never_licenses() {
        // The default verifier holds no key and reaches no network — any blob is community.
        let s = Licensing::from_blob(&CommunityVerifier, b"any-token-at-all", NOW);
        assert_eq!(s.tier(NOW), Tier::Community);
    }

    // ── ed25519 verifier + token format (behind the `license` feature) ────────
    // A throwaway TEST key: its public half is derived at runtime and passed to
    // `with_public_key`, never the embedded vendor key — so no production private
    // key lives in the tree.
    #[cfg(feature = "license")]
    const TEST_SEED: [u8; 32] = [7u8; 32];

    #[cfg(feature = "license")]
    fn test_verifier() -> Ed25519Verifier {
        let sk = ed25519_dalek::SigningKey::from_bytes(&TEST_SEED);
        Ed25519Verifier::with_public_key(&sk.verifying_key().to_bytes())
    }

    #[cfg(feature = "license")]
    #[test]
    fn b64url_round_trips_without_padding() {
        let cases: [&[u8]; 8] = [
            b"",
            b"f",
            b"fo",
            b"foo",
            b"foob",
            b"fooba",
            b"foobar",
            &[0, 255, 1, 254],
        ];
        for case in cases {
            assert_eq!(b64url_decode(&b64url_encode(case)).as_deref(), Some(case));
        }
        assert!(!b64url_encode(b"any payload here").contains('='));
    }

    #[cfg(feature = "license")]
    #[test]
    fn signed_token_verifies_to_pro_and_unlocks_features() {
        let token = sign_token(&TEST_SEED, "acme-corp", Some(NOW + 1000));
        let s = Licensing::from_blob(&test_verifier(), token.as_bytes(), NOW);
        assert_eq!(s.tier(NOW), Tier::Pro);
        assert_eq!(s.licensee(), Some("acme-corp"));
        for f in Feature::ALL {
            assert!(s.require(f, NOW).is_ok());
        }
    }

    #[cfg(feature = "license")]
    #[test]
    fn perpetual_signed_token_is_pro_forever() {
        let token = sign_token(&TEST_SEED, "forever-inc", None);
        let s = Licensing::from_blob(&test_verifier(), token.as_bytes(), NOW);
        assert_eq!(s.tier(u64::MAX), Tier::Pro);
    }

    #[cfg(feature = "license")]
    #[test]
    fn trailing_whitespace_from_a_file_is_tolerated() {
        let token = format!("{}\n", sign_token(&TEST_SEED, "acme", None));
        assert!(test_verifier().verify(token.as_bytes(), NOW).is_ok());
    }

    #[cfg(feature = "license")]
    #[test]
    fn tampered_payload_is_rejected_and_falls_back_to_community() {
        let token = sign_token(&TEST_SEED, "acme-corp", Some(NOW + 1000));
        let mut bytes = token.into_bytes();
        bytes[0] ^= 0b1; // flip a payload char → signature no longer matches
        let v = test_verifier();
        assert!(matches!(
            v.verify(&bytes, NOW),
            Err(LicenseError::Invalid(_))
        ));
        assert_eq!(
            Licensing::from_blob(&v, &bytes, NOW).tier(NOW),
            Tier::Community
        );
    }

    #[cfg(feature = "license")]
    #[test]
    fn a_token_signed_by_another_key_is_rejected() {
        let token = sign_token(&[9u8; 32], "impostor", None); // different seed
        let v = test_verifier(); // expects TEST_SEED's public half
        assert!(matches!(
            v.verify(token.as_bytes(), NOW),
            Err(LicenseError::Invalid(_))
        ));
    }

    #[cfg(feature = "license")]
    #[test]
    fn malformed_tokens_are_invalid_and_never_panic() {
        let v = test_verifier();
        for bad in ["", "no-dot", "not.base64url-!!", "only.", ".only"] {
            assert!(v.verify(bad.as_bytes(), NOW).is_err(), "rejects {bad:?}");
        }
    }

    #[cfg(feature = "license")]
    #[test]
    fn unset_embedded_key_fails_closed_to_community() {
        // The placeholder key shipped in this tree licenses nothing — even a well-formed token
        // signed by some key is refused, so the default build is fail-closed (a vendor must paste
        // their own public key). Holds while LICENSE_PUBLIC_KEY is the all-zero placeholder.
        let token = sign_token(&TEST_SEED, "acme", None);
        let s = Licensing::from_blob(&Ed25519Verifier::new(), token.as_bytes(), NOW);
        assert_eq!(s.tier(NOW), Tier::Community);
    }

    #[cfg(feature = "license")]
    #[test]
    fn expired_signed_token_reads_community_but_keeps_licensee() {
        let token = sign_token(&TEST_SEED, "lapsed-llc", Some(NOW - 1));
        let s = Licensing::from_blob(&test_verifier(), token.as_bytes(), NOW);
        // Valid signature (licensee retained for the audit log) but past expiry, so the
        // tier is community — gated, never silently degraded.
        assert_eq!(s.licensee(), Some("lapsed-llc"));
        assert_eq!(s.tier(NOW), Tier::Community);
        assert!(!s.allows(Feature::AuditReports, NOW));
    }
}
