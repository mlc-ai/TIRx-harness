use crate::DynamicOpId;

/// Exact async transaction identity, independent of completion scheduling order.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AsyncTokenId {
    issue_operation: DynamicOpId,
    token_ordinal: u32,
}

impl AsyncTokenId {
    pub fn new(issue_operation: DynamicOpId, token_ordinal: u32) -> Self {
        Self {
            issue_operation,
            token_ordinal,
        }
    }

    pub const fn issue_operation(&self) -> &DynamicOpId {
        &self.issue_operation
    }

    pub const fn token_ordinal(&self) -> u32 {
        self.token_ordinal
    }
}
