//! Opaque v2 error transport.

use std::fmt;

/// Engine failure returned by v2 calls.
///
/// The payload is intentionally opaque: adding a checker diagnostic or an
/// engine-internal error kind does not expand the generated-artifact ABI.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EngineError(crate::EngineError);

impl EngineError {
    pub(crate) fn message(message: impl Into<String>) -> Self {
        Self(crate::EngineError::message(message))
    }
}

impl From<crate::EngineError> for EngineError {
    fn from(error: crate::EngineError) -> Self {
        Self(error)
    }
}

impl From<EngineError> for crate::EngineError {
    fn from(error: EngineError) -> Self {
        error.0
    }
}

impl fmt::Display for EngineError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

impl std::error::Error for EngineError {}
