//! The engine-mode axis of the explicit-instantiation lattice.
//!
//! Rust has no stable `extern template`, so every instruction specialization is
//! instantiated once per concrete [`crate::WarpEngine`] mode.  Before this
//! module each instruction file repeated the mode list — engine type, feature
//! gate, and a local `type NumSimWarp = ...` alias — once per instruction
//! family.  The list now lives here exactly once: adding an engine mode is one
//! row in [`engine_mode_rows`], and no instruction module names a
//! `native_analysis` mode type any more.
//!
//! Callers pass the per-mode entry macro plus an optional bracketed argument
//! group, which the entry macro destructures:
//!
//! ```text
//! for_each_engine_mode!(test_visible, my_family_for_mode, [Ty, Space, Sem]);
//! for_each_engine_mode!(test_visible, my_family_for_mode);
//! ```

/// The mode rows themselves. `$sync_gate` is the only per-family variation:
/// see [`for_each_engine_mode`].
macro_rules! engine_mode_rows {
    ($sync_gate:meta, $entry:ident $(, $args:tt)?) => {
        #[cfg(any(test, not(feature = "analysis-core")))]
        $entry!($crate::WarpEngine<$crate::engine_mode::NumSimMode> $(, $args)?);
        #[cfg($sync_gate)]
        $entry!($crate::WarpEngine<$crate::sync_check::SyncCheckMode> $(, $args)?);
        #[cfg(feature = "racecheck")]
        $entry!($crate::WarpEngine<$crate::race_check::RaceCheckMode> $(, $args)?);
    };
}
pub(crate) use engine_mode_rows;

/// Instantiate one instruction family for every engine mode.
///
/// The two selectors differ only in whether the synccheck row is also
/// instantiated in unit-test builds that enable no analysis feature. Both
/// spellings predate this table; keeping them distinct reproduces the
/// instantiated set exactly rather than silently widening or narrowing it.
///
/// * `test_visible` — synccheck row is present under `cfg(test)` as well.
/// * `analysis_only` — synccheck row requires the `analysis-core` feature.
macro_rules! for_each_engine_mode {
    (test_visible, $($rest:tt)*) => {
        $crate::runtime::instructions::mode_axis::engine_mode_rows!(
            any(test, all(feature = "analysis-core", not(feature = "racecheck"))),
            $($rest)*
        );
    };
    (analysis_only, $($rest:tt)*) => {
        $crate::runtime::instructions::mode_axis::engine_mode_rows!(
            all(feature = "analysis-core", not(feature = "racecheck")),
            $($rest)*
        );
    };
}
pub(crate) use for_each_engine_mode;
