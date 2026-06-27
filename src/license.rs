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
//! This module is the **gate**: tiers, the feature set, and the explicit-logging policy. It performs
//! **no I/O itself** (a caller loads the local license file and passes the bytes in). The actual
//! public-key signature check ([`LicenseVerifier`]) is pluggable; the bundled `ed25519` verifier is
//! provided behind the `license` cargo feature so the default build pulls in no cryptography.

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
}
