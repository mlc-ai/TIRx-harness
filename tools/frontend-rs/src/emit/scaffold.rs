//! Artifact split of large regions into private helpers.  Capture attempts,
//! thresholds and restore points decide the emitted text.

use tvm::tirx::{
    AttrStmtObj, BindObj, ForObj, IfThenElseObj, ScopeIdDefStmtObj, SeqStmtObj, Stmt, WhileObj,
};
use tvm::tvm_ffi::object::ObjectRef;
use tvm::tvm_ffi::{Map, ObjectIdentity, ObjectRefCore, String as FfiString};

use super::super::analyze::util::{ffi_error, ffi_text, ident, int_imm, oref, AResult, Failure};
use super::{Emitter, RustValue, SplitArgument, SplitEffects, SplitHelper, Uniformity, Variables};
use crate::tables::is_integer_rust_type;

macro_rules! threshold_value {
    (usize, $value:expr) => {
        usize::try_from($value.max(0)).unwrap_or(usize::MAX)
    };
    (i64, $value:expr) => {
        $value
    };
}

macro_rules! split_thresholds {
    ($($field:ident: $kind:ident = $default:expr, unsplit = $unsplit:expr;)*) => {
        #[derive(Clone, Debug)]
        pub struct SplitThresholds { $(pub $field: $kind,)* }

        impl SplitThresholds {
            /// Analysis checks the emitted body without extracting helpers.
            pub fn unsplit() -> Self { Self { $($field: $unsplit,)* } }

            /// One definition owns fields, defaults, and the Python override names.
            pub fn defaults() -> tvm::tvm_ffi::Result<Map<FfiString, i64>> {
                Ok([$(
                    (FfiString::from(format!("_{}", stringify!($field).to_ascii_uppercase())), $default),
                )*].into_iter().collect())
            }

            pub fn parse(values: Map<FfiString, i64>) -> AResult<Self> {
                let value = |field: &str| -> AResult<i64> {
                    let name = format!("_{}", field.to_ascii_uppercase());
                    values.get(&FfiString::from(name.clone()))?.ok_or_else(|| {
                        Failure::Ffi(ffi_error(&format!("missing split threshold {name}")))
                    })
                };
                let thresholds = Self { $(
                    $field: threshold_value!($kind, value(stringify!($field))?),
                )* };
                if values.len() != [$(stringify!($field),)*].len() {
                    return Err(Failure::Ffi(ffi_error("unknown split threshold")));
                }
                Ok(thresholds)
            }
        }
    };
}

split_thresholds! {
    root_uniform_if_split_min_lines: usize = 1024, unsplit = usize::MAX;
    root_sync_split_min_lines: usize = 384, unsplit = usize::MAX;
    root_sync_split_target_lines: usize = 512, unsplit = usize::MAX;
    root_async_split_min_lines: usize = 384, unsplit = usize::MAX;
    root_async_split_target_lines: usize = 512, unsplit = usize::MAX;
    inner_uniform_if_split_min_lines: usize = 768, unsplit = usize::MAX;
    inner_async_split_min_lines: usize = 384, unsplit = usize::MAX;
    inner_varying_if_split_min_lines: usize = 64, unsplit = usize::MAX;
    inner_for_body_split_min_lines: usize = 128, unsplit = usize::MAX;
    // Serial loops up to this extent may split each iteration body.
    max_transfer_split_for_extent: i64 = 4, unsplit = 0;
    large_transfer_for_body_split_min_lines: usize = 2048, unsplit = usize::MAX;
    max_large_transfer_for_body_split_extent: i64 = 128, unsplit = 0;
    inner_async_split_target_lines: usize = 64, unsplit = usize::MAX;
    analysis_inner_async_split_target_lines: usize = 64, unsplit = usize::MAX;
}

fn static_positive_for_extent(loop_stmt: &ForObj) -> Option<i64> {
    int_imm(&oref(loop_stmt.extent.clone())).filter(|extent| *extent > 0)
}

pub fn allows_bounded_transfer_split(loop_stmt: &ForObj, thresholds: &SplitThresholds) -> bool {
    static_positive_for_extent(loop_stmt)
        .is_some_and(|extent| extent <= thresholds.max_transfer_split_for_extent)
}

fn allows_large_transfer_body_split(loop_stmt: &ForObj, thresholds: &SplitThresholds) -> bool {
    static_positive_for_extent(loop_stmt)
        .is_some_and(|extent| extent <= thresholds.max_large_transfer_for_body_split_extent)
}

fn is_rust_identifier(text: &str) -> bool {
    let mut chars = text.chars();
    match chars.next() {
        Some(first) if first.is_ascii_alphabetic() || first == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Statement kinds that never join a captured run.
fn is_run_excluded(stmt: &Stmt) -> bool {
    stmt.as_node::<AttrStmtObj>().is_some()
        || stmt.as_node::<SeqStmtObj>().is_some()
        || stmt.as_node::<ScopeIdDefStmtObj>().is_some()
        || stmt.as_node::<BindObj>().is_some()
}

pub struct Checkpoint {
    next_control: i64,
    next_temp: i64,
    next_split_helper: i64,
    next_sync_helper: i64,
    next_root_async_helper: i64,
    next_inner_if_helper: i64,
    next_inner_varying_if_helper: i64,
    next_inner_for_helper: i64,
    next_inner_async_helper: i64,
    split_helper_count: usize,
    raw_tcgen: super::raw_tcgen::State,
    variables: Variables,
    dynamic_buffers_len: usize,
    dynamic_buffers: Vec<(tvm::tirx::BufferVar, String)>,
    split_effects: Vec<SplitEffects>,
    recorded_store_count: usize,
}

pub struct CapturedSplitBody {
    pub lines: Vec<String>,
    pub effects: SplitEffects,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RunKind {
    RootSync,
    RootAsync,
    InnerAsync,
}

#[derive(Clone, Copy)]
enum IfSplitKind {
    RootUniform,
    InnerUniform,
    InnerVarying,
}

/// `re.search(rf"\\b{name}\\b", body)`.
fn contains_word(body: &str, name: &str) -> bool {
    let is_word = |c: char| c.is_alphanumeric() || c == '_';
    body.match_indices(name).any(|(at, _)| {
        body[..at].chars().next_back().map_or(true, |c| !is_word(c))
            && body[at + name.len()..].chars().next().map_or(true, |c| !is_word(c))
    })
}

impl CapturedSplitBody {
    fn into_helper(self, name: String, arguments: Vec<SplitArgument>) -> SplitHelper {
        let is_async = self.effects.suspend_count > 0;
        // Keep the helper signature and every call derived from one argument
        // list. Unused borrowed runtime handles need neither a parameter nor
        // an argument; owned handles retain their clone/drop behavior.
        let arguments = if is_async {
            arguments
        } else {
            let body = self.lines.join("\n");
            arguments
                .into_iter()
                .filter(|argument| {
                    !((argument.name == "physical" || argument.name == "services")
                        && argument.rust_type.starts_with('&')
                        && argument.call_code == format!("&{}", argument.name)
                        && !contains_word(&body, &argument.name))
                })
                .collect()
        };
        SplitHelper {
            name,
            is_async,
            inline_never: true,
            arguments,
            body: self.lines,
        }
    }
}

impl<'a> Emitter<'a> {
    // ------------------------------------------------------------------
    // Split effects.
    // ------------------------------------------------------------------

    /// Enter a split effect scope.
    pub fn push_split_effects(&mut self, borrow_non_copy: bool) {
        let mut outer_variables = Variables::default();
        for key in self.variables.keys() {
            let value = self.lookup_variable(&key).expect("bound variable");
            outer_variables.set(key, value);
        }
        let effects = SplitEffects {
            outer_variables,
            outer_dynamic_references: self
                .dynamic_buffers
                .iter()
                .map(|(_, reference)| reference.clone())
                .collect(),
            outer_loop_live_masks: self.loop_live_masks.clone(),
            borrow_non_copy,
            arguments: Vec::new(),
            variable_replacements: Variables::default(),
            used_dynamic_references: Vec::new(),
            loop_live_mask_replacements: Vec::new(),
            allow_outer_loop_live_mutation: self
                .outer_loop_live_split_permissions
                .last()
                .copied()
                .unwrap_or(false),
            suspend_count: 0,
            unsafe_reason: None,
        };
        self.split_effects.push(effects);
    }

    pub fn pop_split_effects(&mut self) -> SplitEffects {
        self.split_effects.pop().expect("split effects")
    }

    fn split_argument_name(&mut self, at: usize, value: &RustValue) -> String {
        let is_reserved = |emitter: &Self, name: &str| -> bool {
            [
                "ctx",
                "physical",
                "buffers",
                "mbarriers",
                "named_barriers",
                "rendezvous",
                "cluster_barriers",
                "cta_reduce",
                "tcgen",
                "async_groups",
                "ordering",
                "services",
            ]
            .contains(&name)
                || emitter.split_effects[at]
                    .outer_dynamic_references
                    .iter()
                    .any(|reference| reference == name)
                || emitter.split_effects[at].argument(name).is_some()
        };
        if is_rust_identifier(&value.code)
            && value.code != "false"
            && value.code != "true"
            && !is_reserved(self, &value.code)
        {
            return value.code.clone();
        }
        loop {
            let candidate = self.control_name("split_arg");
            if !is_reserved(self, &candidate) {
                return candidate;
            }
        }
    }

    fn capture_split_value(
        &mut self,
        at: usize,
        variable: &ObjectRef,
        value: RustValue,
    ) -> RustValue {
        if let Some(existing) = self.split_effects[at].variable_replacements.get(variable) {
            return existing.clone();
        }
        let variable_name = match super::super::analyze::util::as_var(variable) {
            Some(var) => ffi_text(&var.name),
            None => String::new(),
        };
        let argument_name = self.split_argument_name(at, &value);
        let borrow_non_copy = self.split_effects[at].borrow_non_copy;
        let clone_source = if is_rust_identifier(&value.code) {
            value.code.clone()
        } else {
            format!("({})", value.code)
        };
        let (rust_type, call_code) = if value.is_mask {
            ("WarpMask".to_owned(), value.code.clone())
        } else if value.rust_type == "PhysicalPtr" {
            if borrow_non_copy {
                if !is_rust_identifier(&value.code) {
                    self.split_effects[at].note_unsafe(format!(
                        "outer physical pointer {:?} is not materialized",
                        &variable_name
                    ));
                    return value;
                }
                ("&PhysicalPtr".to_owned(), format!("&{}", value.code))
            } else {
                ("PhysicalPtr".to_owned(), format!("{clone_source}.clone()"))
            }
        } else if value.uniformity == Uniformity::Varying {
            let value_type = format!("WarpValue<{}>", value.rust_type);
            if borrow_non_copy {
                if !is_rust_identifier(&value.code) {
                    self.split_effects[at].note_unsafe(format!(
                        "outer varying value {:?} is not materialized",
                        &variable_name
                    ));
                    return value;
                }
                (format!("&{value_type}"), format!("&{}", value.code))
            } else {
                (value_type, format!("{clone_source}.clone()"))
            }
        } else if is_integer_rust_type(&value.rust_type)
            || matches!(value.rust_type.as_str(), "bool" | "f32" | "F32x4" | "U64x2")
        {
            (value.rust_type.clone(), value.code.clone())
        } else {
            self.split_effects[at].note_unsafe(format!(
                "outer variable {:?} has unsupported Rust type {}",
                &variable_name, value.rust_type
            ));
            return value;
        };
        self.split_effects[at].set_argument(SplitArgument {
            name: argument_name.clone(),
            rust_type,
            call_code,
            snapshot_before_control: value.requires_statement,
        });
        let replacement = RustValue {
            code: argument_name,
            requires_statement: false,
            ..value
        };
        self.split_effects[at]
            .variable_replacements
            .set(variable.clone(), replacement.clone());
        replacement
    }

    pub fn resolve_split_variable(&mut self, variable: &ObjectRef, value: RustValue) -> RustValue {
        let mut replacement = value;
        for at in 0..self.split_effects.len() {
            if self.split_effects[at].outer_variables.contains(variable) {
                replacement = self.capture_split_value(at, variable, replacement);
            }
        }
        replacement
    }

    pub fn record_dynamic_buffer_use(&mut self, reference: &str) {
        for effects in &mut self.split_effects {
            if effects
                .outer_dynamic_references
                .iter()
                .any(|candidate| candidate == reference)
                && !effects
                    .used_dynamic_references
                    .iter()
                    .any(|candidate| candidate == reference)
            {
                effects.used_dynamic_references.push(reference.to_owned());
            }
        }
    }

    pub fn mark_suspend(&mut self) {
        for effects in &mut self.split_effects {
            effects.suspend_count += 1;
        }
    }

    pub fn split_loop_live_mask_lvalue(&mut self, live_mask: &str) -> String {
        let mut code = live_mask.to_owned();
        let mut is_reference = false;
        for at in 0..self.split_effects.len() {
            if !self.split_effects[at]
                .outer_loop_live_masks
                .iter()
                .any(|candidate| candidate == live_mask)
            {
                continue;
            }
            if !self.split_effects[at].allow_outer_loop_live_mutation {
                self.split_effects[at].note_unsafe("helper would mutate an outer loop live mask");
                continue;
            }
            let existing = self.split_effects[at]
                .loop_live_mask_replacements
                .iter()
                .find(|(mask, _)| mask == live_mask)
                .map(|(_, replacement)| replacement.clone());
            let replacement = match existing {
                Some(replacement) => replacement,
                None => {
                    let replacement = self.control_name("split_loop_live_mask");
                    let call_code = if is_reference {
                        format!("&mut *{code}")
                    } else {
                        format!("&mut {code}")
                    };
                    self.split_effects[at].set_argument(SplitArgument {
                        name: replacement.clone(),
                        rust_type: "&mut WarpMask".to_owned(),
                        call_code,
                        snapshot_before_control: false,
                    });
                    self.split_effects[at]
                        .loop_live_mask_replacements
                        .push((live_mask.to_owned(), replacement.clone()));
                    replacement
                }
            };
            code = replacement;
            is_reference = true;
        }
        if is_reference {
            format!("*{code}")
        } else {
            live_mask.to_owned()
        }
    }

    // ------------------------------------------------------------------
    // Checkpoints.
    // ------------------------------------------------------------------

    pub fn partition_checkpoint(&self) -> Checkpoint {
        Checkpoint {
            next_control: self.next_control,
            next_temp: self.next_temp,
            next_split_helper: self.next_split_helper,
            next_sync_helper: self.next_sync_helper,
            next_root_async_helper: self.next_root_async_helper,
            next_inner_if_helper: self.next_inner_if_helper,
            next_inner_varying_if_helper: self.next_inner_varying_if_helper,
            next_inner_for_helper: self.next_inner_for_helper,
            next_inner_async_helper: self.next_inner_async_helper,
            split_helper_count: self.split_helpers.len(),
            raw_tcgen: self.raw_tcgen.clone(),
            variables: self.variables.clone(),
            dynamic_buffers_len: self.dynamic_buffers.len(),
            dynamic_buffers: self.dynamic_buffers.clone(),
            split_effects: self.split_effects.clone(),
            recorded_store_count: self.recorded_store_count,
        }
    }

    pub(super) fn restore_failed_partition(&mut self, checkpoint: &Checkpoint) {
        self.split_effects = checkpoint.split_effects.clone();
        self.restore_partition_checkpoint(checkpoint);
    }

    pub fn restore_partition_checkpoint(&mut self, checkpoint: &Checkpoint) {
        self.next_control = checkpoint.next_control;
        self.next_temp = checkpoint.next_temp;
        self.next_split_helper = checkpoint.next_split_helper;
        self.next_sync_helper = checkpoint.next_sync_helper;
        self.next_root_async_helper = checkpoint.next_root_async_helper;
        self.next_inner_if_helper = checkpoint.next_inner_if_helper;
        self.next_inner_varying_if_helper = checkpoint.next_inner_varying_if_helper;
        self.next_inner_for_helper = checkpoint.next_inner_for_helper;
        self.next_inner_async_helper = checkpoint.next_inner_async_helper;
        self.split_helpers.truncate(checkpoint.split_helper_count);
        self.raw_tcgen = checkpoint.raw_tcgen.clone();
        self.variables = checkpoint.variables.clone();
        self.dynamic_buffers = checkpoint.dynamic_buffers.clone();
        assert_eq!(self.split_effects.len(), checkpoint.split_effects.len());
        for (effects, saved) in self
            .split_effects
            .iter_mut()
            .zip(checkpoint.split_effects.iter())
        {
            effects.arguments = saved.arguments.clone();
            effects.variable_replacements = saved.variable_replacements.clone();
            effects.loop_live_mask_replacements = saved.loop_live_mask_replacements.clone();
            effects.used_dynamic_references = saved.used_dynamic_references.clone();
            effects.suspend_count = saved.suspend_count;
            effects.unsafe_reason = saved.unsafe_reason.clone();
        }
        self.recorded_store_count = checkpoint.recorded_store_count;
    }

    fn partition_state_escaped(&self, checkpoint: &Checkpoint) -> bool {
        !self.variables.equals(&checkpoint.variables)
            || self.dynamic_buffers.len() != checkpoint.dynamic_buffers_len
    }

    fn nested_split_lines(&self, checkpoint: &Checkpoint) -> usize {
        self.split_helpers[checkpoint.split_helper_count..]
            .iter()
            .map(|helper| helper.body.len())
            .sum()
    }

    // ------------------------------------------------------------------
    // Suspend reachability.
    // ------------------------------------------------------------------

    fn subtree_may_suspend(&mut self, stmt: &Stmt) -> AResult<bool> {
        let key = oref(stmt.clone());
        if let Some(cached) = self.subtree_may_suspend_cache.get(&key) {
            return Ok(*cached);
        }
        let found = crate::post_order_nodes(key.clone())?.iter().any(|node| {
            node.as_node::<ForObj>().is_some()
                || node.as_node::<WhileObj>().is_some()
                || self.suspend_source_nodes.contains(&node)
        });
        self.subtree_may_suspend_cache.insert(key, found);
        Ok(found)
    }

    fn inner_async_suffix_may_suspend(
        &mut self,
        statements: &[Stmt],
        start: usize,
    ) -> AResult<bool> {
        for stmt in &statements[start..] {
            if is_run_excluded(stmt) {
                break;
            }
            if self.subtree_may_suspend(stmt)? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    // ------------------------------------------------------------------
    // Helpers and calls.
    // ------------------------------------------------------------------

    fn split_arguments(&self, effects: &SplitEffects) -> Option<Vec<SplitArgument>> {
        if effects.unsafe_reason.is_some() {
            return None;
        }
        let owned = [
            ("physical", "PhysicalMemory".to_owned(), "physical.clone()"),
            (
                "buffers",
                format!("Arc<Kernel{}Buffers>", self.kernel_index),
                "buffers.clone()",
            ),
            (
                "services",
                "KernelRuntimeServices".to_owned(),
                "services.clone()",
            ),
        ];
        let mut arguments: Vec<SplitArgument> = owned
            .iter()
            .map(|(name, rust_type, call_code)| {
                if effects.borrow_non_copy {
                    SplitArgument {
                        name: (*name).to_owned(),
                        rust_type: format!("&{rust_type}"),
                        call_code: format!("&{name}"),
                        snapshot_before_control: false,
                    }
                } else {
                    SplitArgument {
                        name: (*name).to_owned(),
                        rust_type: rust_type.clone(),
                        call_code: (*call_code).to_owned(),
                        snapshot_before_control: false,
                    }
                }
            })
            .collect();
        let mut existing: Vec<String> = arguments
            .iter()
            .map(|argument| argument.name.clone())
            .collect();
        for reference in &effects.outer_dynamic_references {
            if existing.contains(reference) || !effects.used_dynamic_references.contains(reference)
            {
                continue;
            }
            arguments.push(if effects.borrow_non_copy {
                SplitArgument {
                    name: reference.clone(),
                    rust_type: "&RuntimeBuffer".to_owned(),
                    call_code: format!("&{reference}"),
                    snapshot_before_control: false,
                }
            } else {
                SplitArgument {
                    name: reference.clone(),
                    rust_type: "RuntimeBuffer".to_owned(),
                    call_code: format!("{reference}.clone()"),
                    snapshot_before_control: false,
                }
            });
            existing.push(reference.clone());
        }
        for argument in &effects.arguments {
            if existing.contains(&argument.name) {
                return None;
            }
            arguments.push(argument.clone());
            existing.push(argument.name.clone());
        }
        Some(arguments)
    }

    pub fn snapshot_split_call_arguments(&mut self, helper: &SplitHelper) -> Vec<String> {
        let mut snapshots = Vec::new();
        for argument in &helper.arguments {
            if !argument.snapshot_before_control {
                snapshots.push(argument.call_code.clone());
                continue;
            }
            let snapshot = self.control_name("split_call_arg");
            self.emit_line(&format!(
                "let {snapshot}: {} = {};",
                argument.rust_type, argument.call_code
            ));
            snapshots.push(snapshot);
        }
        snapshots
    }

    pub fn emit_split_call(&mut self, helper: &SplitHelper, call_codes: Option<Vec<String>>) {
        if helper.is_async {
            self.mark_suspend();
        }
        let call_codes = call_codes.unwrap_or_else(|| {
            helper
                .arguments
                .iter()
                .map(|argument| argument.call_code.clone())
                .collect()
        });
        assert_eq!(call_codes.len(), helper.arguments.len());
        let wrapper = if helper.is_async {
            "NumSimModuleFuture("
        } else {
            ""
        };
        self.emit_line(&format!("ctx = {wrapper}{}(", helper.name));
        self.indent += 1;
        self.emit_line("ctx,");
        self.emit_line("warp,");
        for call_code in &call_codes {
            self.emit_line(&format!("{call_code},"));
        }
        self.indent -= 1;
        self.emit_line(if helper.is_async { ")).await?;" } else { ")?;" });
    }

    /// Capture generated lines and effects, restoring the parent emission frame
    /// even when the body returns an error. Partition rollback belongs to the caller.
    fn capture_split<T>(
        &mut self,
        arm_depth: usize,
        borrow_non_copy: bool,
        emit: impl FnOnce(&mut Self) -> AResult<T>,
    ) -> AResult<(T, CapturedSplitBody)> {
        let parent_lines = std::mem::take(&mut self.lines);
        let parent_indent = self.indent;
        let parent_split_arm_depth = self.split_arm_depth;
        self.indent = 1;
        self.split_arm_depth = arm_depth;
        self.push_split_effects(borrow_non_copy);
        let result = emit(self);
        let effects = self.pop_split_effects();
        let lines = std::mem::replace(&mut self.lines, parent_lines);
        self.indent = parent_indent;
        self.split_arm_depth = parent_split_arm_depth;
        Ok((result?, CapturedSplitBody { lines, effects }))
    }

    fn capture_split_arm(
        &mut self,
        stmt: &Stmt,
        initial_stmt: Option<&Stmt>,
        borrow_non_copy: bool,
    ) -> AResult<CapturedSplitBody> {
        let (_, body) =
            self.capture_split(self.split_arm_depth + 1, borrow_non_copy, |emitter| {
                emitter.emit_scoped(stmt, initial_stmt)
            })?;
        Ok(body)
    }

    // ------------------------------------------------------------------
    // Root sequence.
    // ------------------------------------------------------------------

    pub fn emit_root_sequence(
        &mut self,
        statements: &[Stmt],
        previous_sequence_stmt: Option<&Stmt>,
    ) -> AResult<()> {
        let snapshot = self.scope_snapshot();
        let result = (|| -> AResult<()> {
            let mut previous = previous_sequence_stmt.cloned();
            let mut index = 0;
            while index < statements.len() {
                let mut consumed =
                    self.emit_root_run(statements, index, previous.as_ref(), RunKind::RootSync)?;
                if consumed == 0 {
                    consumed = self.emit_root_run(
                        statements,
                        index,
                        previous.as_ref(),
                        RunKind::RootAsync,
                    )?;
                }
                if consumed > 0 {
                    previous = Some(statements[index + consumed - 1].clone());
                    index += consumed;
                    continue;
                }
                let stmt = &statements[index];
                self.emit_stmt(stmt, previous.as_ref())?;
                previous = Some(stmt.clone());
                index += 1;
            }
            Ok(())
        })();
        self.restore_scope(snapshot);
        result
    }

    fn reemit_run(
        &mut self,
        statements: &[Stmt],
        start: usize,
        consumed: usize,
        previous_stmt: Option<&Stmt>,
    ) -> AResult<()> {
        let mut current_previous = previous_stmt.cloned();
        for stmt in &statements[start..start + consumed] {
            self.emit_stmt(stmt, current_previous.as_ref())?;
            current_previous = Some(stmt.clone());
        }
        Ok(())
    }

    /// The run kinds share capture and rollback; their suspend/transfer rules
    /// and size thresholds remain explicit at the admission points below.
    fn capture_run(
        &mut self,
        statements: &[Stmt],
        start: usize,
        previous_stmt: Option<&Stmt>,
        kind: RunKind,
    ) -> AResult<(usize, CapturedSplitBody)> {
        let arm_depth = match kind {
            RunKind::RootAsync => 1,
            RunKind::InnerAsync => 2,
            RunKind::RootSync => self.split_arm_depth,
        };
        let target_lines = match kind {
            RunKind::RootSync => self.split_thresholds.root_sync_split_target_lines,
            RunKind::RootAsync => self.split_thresholds.root_async_split_target_lines,
            RunKind::InnerAsync => self.inner_async_split_target_lines(),
        };
        self.capture_split(arm_depth, kind != RunKind::RootSync, |emitter| {
            let mut consumed = 0;
            let mut open_transfer_guards = 0;
            let mut previous = previous_stmt.cloned();
            for (index, stmt) in statements.iter().enumerate().skip(start) {
                if is_run_excluded(stmt) {
                    break;
                }
                let checkpoint = emitter.partition_checkpoint();
                let line_start = emitter.lines.len();
                let suspend_before = emitter.split_effects.last().expect("effects").suspend_count;
                emitter.control_depth += 1;
                let result = emitter.emit_stmt(stmt, previous.as_ref());
                emitter.control_depth -= 1;
                result?;
                let suspend_count = emitter.split_effects.last().expect("effects").suspend_count;
                if (kind == RunKind::RootSync && suspend_count != suspend_before)
                    || emitter.partition_state_escaped(&checkpoint)
                {
                    emitter.restore_partition_checkpoint(&checkpoint);
                    emitter.lines.truncate(line_start);
                    break;
                }
                consumed += 1;
                previous = Some(stmt.clone());
                if kind == RunKind::InnerAsync
                    && emitter.contains_current_loop_transfer(stmt)?
                    && index + 1 < statements.len()
                {
                    emitter.emit_line("if !ctx.active_mask().is_empty() {");
                    emitter.indent += 1;
                    open_transfer_guards += 1;
                }
                let can_stop = match kind {
                    RunKind::RootSync => true,
                    RunKind::RootAsync => suspend_count > 0,
                    // The caller guards the remainder with the returned active mask,
                    // so a split may close local transfer guards at this boundary
                    // instead of pulling the whole loop tail into one helper.
                    RunKind::InnerAsync => true,
                };
                if emitter.lines.len() >= target_lines && can_stop {
                    break;
                }
            }
            for _ in 0..open_transfer_guards {
                emitter.indent -= 1;
                emitter.emit_line("}");
            }
            Ok(consumed)
        })
    }

    fn publish_run(
        &mut self,
        kind: RunKind,
        body: CapturedSplitBody,
        arguments: Vec<SplitArgument>,
    ) {
        let (prefix, counter) = match kind {
            RunKind::RootSync => ("root_sync", &mut self.next_sync_helper),
            RunKind::RootAsync => ("root_async", &mut self.next_root_async_helper),
            RunKind::InnerAsync => ("inner_async", &mut self.next_inner_async_helper),
        };
        let name = format!("kernel_{}_{prefix}_{}", self.kernel_index, *counter);
        *counter += 1;
        let helper = body.into_helper(name, arguments);
        self.split_helpers.push(helper.clone());
        self.emit_split_call(&helper, None);
    }

    fn emit_root_run(
        &mut self,
        statements: &[Stmt],
        start: usize,
        previous_stmt: Option<&Stmt>,
        kind: RunKind,
    ) -> AResult<usize> {
        if is_run_excluded(&statements[start]) {
            return Ok(0);
        }
        let checkpoint = self.partition_checkpoint();
        let (consumed, body) = self.capture_run(statements, start, previous_stmt, kind)?;
        if consumed == 0 {
            self.restore_partition_checkpoint(&checkpoint);
            return Ok(0);
        }
        let arguments = self.split_arguments(&body.effects);
        let accepted = match kind {
            RunKind::RootSync => {
                body.lines.len() >= self.split_thresholds.root_sync_split_min_lines
            }
            RunKind::RootAsync => {
                body.lines.len() + self.nested_split_lines(&checkpoint)
                    >= self.split_thresholds.root_async_split_min_lines
                    && body.effects.suspend_count > 0
            }
            RunKind::InnerAsync => unreachable!("inner runs have their own admission rules"),
        };
        if accepted {
            if let Some(arguments) = arguments {
                self.publish_run(kind, body, arguments);
                return Ok(consumed);
            }
        }
        self.restore_partition_checkpoint(&checkpoint);
        self.reemit_run(statements, start, consumed, previous_stmt)?;
        Ok(consumed)
    }

    // ------------------------------------------------------------------
    // Inner sequences.
    // ------------------------------------------------------------------

    fn inner_async_split_target_lines(&self) -> usize {
        if self.analysis_capable {
            self.split_thresholds.analysis_inner_async_split_target_lines
        } else {
            self.split_thresholds.inner_async_split_target_lines
        }
    }

    fn emit_inner_async_run(
        &mut self,
        statements: &[Stmt],
        start: usize,
        previous_stmt: Option<&Stmt>,
    ) -> AResult<usize> {
        if is_run_excluded(&statements[start]) {
            return Ok(0);
        }
        if !self.inner_async_suffix_may_suspend(statements, start)?
            && self.lines.len() < self.inner_async_split_target_lines()
        {
            return Ok(0);
        }
        let identity = oref(statements[start].clone());
        if let Some(failed_run) = self.failed_inner_async_splits.get(&identity).cloned() {
            let end = (start + failed_run.len()).min(statements.len());
            let current_run: Vec<ObjectIdentity> =
                statements[start..end].iter().map(ident).collect();
            if current_run == failed_run {
                return self.emit_inner_async_fallback(
                    statements,
                    start,
                    failed_run.len(),
                    previous_stmt,
                );
            }
        }
        let group_checkpoint = self.partition_checkpoint();
        let parent_line_count = self.lines.len();
        let (consumed, body) =
            self.capture_run(statements, start, previous_stmt, RunKind::InnerAsync)?;
        if consumed == 0 {
            self.restore_partition_checkpoint(&group_checkpoint);
            return Ok(0);
        }
        let arguments = self.split_arguments(&body.effects);
        let min_lines = if parent_line_count >= self.inner_async_split_target_lines() {
            1
        } else {
            self.split_thresholds.inner_async_split_min_lines
        };
        if body.lines.len() >= min_lines {
            if let Some(arguments) = arguments {
                self.publish_run(RunKind::InnerAsync, body, arguments);
                return Ok(consumed);
            }
        }
        let run: Vec<ObjectIdentity> = statements[start..start + consumed]
            .iter()
            .map(ident)
            .collect();
        if !self
            .failed_inner_async_splits
            .insert(identity.clone(), run.clone())
        {
            *self
                .failed_inner_async_splits
                .get_mut(&identity)
                .expect("failed run") = run;
        }
        self.restore_partition_checkpoint(&group_checkpoint);
        self.emit_inner_async_fallback(statements, start, consumed, previous_stmt)
    }

    fn emit_inner_async_fallback(
        &mut self,
        statements: &[Stmt],
        start: usize,
        consumed: usize,
        previous_stmt: Option<&Stmt>,
    ) -> AResult<usize> {
        let fallback = &statements[start..start + consumed];
        let mut any_transfer = false;
        for stmt in fallback {
            if self.contains_current_loop_transfer(stmt)? {
                any_transfer = true;
                break;
            }
        }
        if any_transfer {
            self.emit_loop_sequence_items(fallback, previous_stmt)?;
        } else {
            self.reemit_run(statements, start, consumed, previous_stmt)?;
        }
        Ok(consumed)
    }

    pub fn emit_inner_sequence(
        &mut self,
        statements: &[Stmt],
        previous_sequence_stmt: Option<&Stmt>,
    ) -> AResult<()> {
        let snapshot = self.scope_snapshot();
        let result = (|| -> AResult<()> {
            let mut previous = previous_sequence_stmt.cloned();
            let mut index = 0;
            while index < statements.len() {
                let consumed = self.emit_inner_async_run(statements, index, previous.as_ref())?;
                let end = if consumed > 0 {
                    index + consumed
                } else {
                    self.emit_stmt(&statements[index], previous.as_ref())?;
                    index + 1
                };
                let emitted = &statements[index..end];
                previous = emitted.last().cloned();
                index = end;
                let mut any_transfer = false;
                for stmt in emitted {
                    if self.contains_current_loop_transfer(stmt)? {
                        any_transfer = true;
                        break;
                    }
                }
                if !any_transfer {
                    continue;
                }
                let remainder = &statements[index..];
                if !remainder.is_empty() {
                    self.emit_line("if !ctx.active_mask().is_empty() {");
                    self.indent += 1;
                    self.emit_inner_sequence(remainder, previous.as_ref())?;
                    self.indent -= 1;
                    self.emit_line("}");
                }
                return Ok(());
            }
            Ok(())
        })();
        self.restore_scope(snapshot);
        result
    }

    // ------------------------------------------------------------------
    // Branch and loop splits.
    // ------------------------------------------------------------------

    fn try_capture_if_split(
        &mut self,
        branch: &IfThenElseObj,
        initial_stmt: Option<&Stmt>,
        kind: IfSplitKind,
    ) -> AResult<Option<(SplitHelper, Option<SplitHelper>)>> {
        let (borrow_non_copy, min_lines) = match kind {
            IfSplitKind::RootUniform => {
                (false, self.split_thresholds.root_uniform_if_split_min_lines)
            }
            IfSplitKind::InnerUniform => {
                (true, self.split_thresholds.inner_uniform_if_split_min_lines)
            }
            IfSplitKind::InnerVarying => {
                (true, self.split_thresholds.inner_varying_if_split_min_lines)
            }
        };
        let checkpoint = self.partition_checkpoint();
        let mut bodies =
            vec![self.capture_split_arm(&branch.then_case, initial_stmt, borrow_non_copy)?];
        if let Some(else_case) = &branch.else_case {
            bodies.push(self.capture_split_arm(else_case, initial_stmt, borrow_non_copy)?);
        }
        let arguments: Option<Vec<_>> = bodies
            .iter()
            .map(|body| self.split_arguments(&body.effects))
            .collect();
        let line_count: usize = bodies.iter().map(|body| body.lines.len()).sum();
        if line_count + self.nested_split_lines(&checkpoint) < min_lines || arguments.is_none() {
            self.restore_partition_checkpoint(&checkpoint);
            return Ok(None);
        }
        let (prefix, counter) = match kind {
            IfSplitKind::RootUniform => ("root_if", &mut self.next_split_helper),
            IfSplitKind::InnerUniform => ("inner_if", &mut self.next_inner_if_helper),
            IfSplitKind::InnerVarying => {
                ("inner_varying_if", &mut self.next_inner_varying_if_helper)
            }
        };
        let name = format!("kernel_{}_{prefix}_{}", self.kernel_index, *counter);
        *counter += 1;
        let mut helpers = bodies
            .into_iter()
            .zip(arguments.expect("split arguments"))
            .zip(["then", "else"])
            .map(|((body, arguments), suffix)| {
                body.into_helper(format!("{name}_{suffix}"), arguments)
            });
        let then_helper = helpers.next().expect("then helper");
        let else_helper = helpers.next();
        self.split_helpers.push(then_helper.clone());
        if let Some(helper) = &else_helper {
            self.split_helpers.push(helper.clone());
        }
        Ok(Some((then_helper, else_helper)))
    }

    fn emit_if_split_calls(
        &mut self,
        condition: &RustValue,
        then_helper: &SplitHelper,
        else_helper: Option<&SplitHelper>,
    ) {
        self.emit_line(&format!("if {} {{", condition.code));
        self.indent += 1;
        self.emit_split_call(then_helper, None);
        self.indent -= 1;
        if let Some(else_helper) = else_helper {
            self.emit_line("} else {");
            self.indent += 1;
            self.emit_split_call(else_helper, None);
            self.indent -= 1;
        }
        self.emit_line("}");
    }

    pub fn try_emit_root_uniform_if_split(
        &mut self,
        branch: &IfThenElseObj,
        condition: &RustValue,
        initial_stmt: Option<&Stmt>,
    ) -> AResult<bool> {
        if self.control_depth != 0 {
            return Ok(false);
        }
        let Some((then_helper, else_helper)) =
            self.try_capture_if_split(branch, initial_stmt, IfSplitKind::RootUniform)?
        else {
            return Ok(false);
        };
        self.emit_if_split_calls(condition, &then_helper, else_helper.as_ref());
        Ok(true)
    }

    pub fn try_emit_inner_uniform_if_split(
        &mut self,
        stmt: &Stmt,
        branch: &IfThenElseObj,
        condition: &RustValue,
        initial_stmt: Option<&Stmt>,
    ) -> AResult<bool> {
        if self.split_arm_depth < 1 || self.control_depth < 2 {
            return Ok(false);
        }
        let identity = oref(stmt.clone());
        if self.failed_inner_uniform_if_splits.contains(&identity) {
            return Ok(false);
        }
        let Some((then_helper, else_helper)) =
            self.try_capture_if_split(branch, initial_stmt, IfSplitKind::InnerUniform)?
        else {
            self.failed_inner_uniform_if_splits.add(identity);
            return Ok(false);
        };
        self.emit_if_split_calls(condition, &then_helper, else_helper.as_ref());
        Ok(true)
    }

    pub fn try_capture_inner_varying_if_split(
        &mut self,
        branch: &IfThenElseObj,
        _condition: &RustValue,
        initial_stmt: Option<&Stmt>,
    ) -> AResult<Option<(SplitHelper, Option<SplitHelper>)>> {
        if self.split_arm_depth != 1 {
            return Ok(None);
        }
        self.try_capture_if_split(branch, initial_stmt, IfSplitKind::InnerVarying)
    }

    pub fn try_capture_inner_for_body_split(
        &mut self,
        _stmt: &Stmt,
        loop_stmt: &ForObj,
    ) -> AResult<Option<SplitHelper>> {
        let has_loop_transfer = self.contains_current_loop_transfer(&loop_stmt.body)?;
        let allow_loop_transfer = self
            .outer_loop_live_split_permissions
            .last()
            .copied()
            .unwrap_or(false);
        let allow_large_transfer_body = has_loop_transfer
            && !allows_bounded_transfer_split(loop_stmt, &self.split_thresholds)
            && allows_large_transfer_body_split(loop_stmt, &self.split_thresholds);
        if self.split_arm_depth < 1
            || (has_loop_transfer && !(allow_loop_transfer || allow_large_transfer_body))
        {
            return Ok(None);
        }
        let checkpoint = self.partition_checkpoint();
        let allow_nested_for_split = self
            .nested_for_statement_split_permissions
            .last()
            .copied()
            .unwrap_or(false);
        self.outer_loop_live_split_permissions
            .push(allow_loop_transfer || allow_large_transfer_body);
        self.nested_for_statement_split_permissions
            .push(allow_nested_for_split);
        let body = self.capture_split_arm(&loop_stmt.body, None, true);
        self.nested_for_statement_split_permissions.pop();
        self.outer_loop_live_split_permissions.pop();
        let body = body?;
        let arguments = self.split_arguments(&body.effects);
        let nested_split_lines = self.nested_split_lines(&checkpoint);
        let split_min_lines = if has_loop_transfer
            && !allows_bounded_transfer_split(loop_stmt, &self.split_thresholds)
        {
            self.split_thresholds
                .large_transfer_for_body_split_min_lines
        } else {
            self.split_thresholds.inner_for_body_split_min_lines
        };
        if body.lines.len() + nested_split_lines < split_min_lines || arguments.is_none() {
            self.restore_partition_checkpoint(&checkpoint);
            return Ok(None);
        }
        let helper = body.into_helper(
            format!(
                "kernel_{}_inner_for_{}",
                self.kernel_index, self.next_inner_for_helper
            ),
            arguments.expect("arguments"),
        );
        self.next_inner_for_helper += 1;
        self.split_helpers.push(helper.clone());
        Ok(Some(helper))
    }

    pub fn try_capture_inner_for_statement_split(
        &mut self,
        stmt: &Stmt,
        loop_stmt: &ForObj,
    ) -> AResult<Option<SplitHelper>> {
        let identity = oref(stmt.clone());
        if self.split_arm_depth != 1
            || (self.loop_depth > 0
                && !self
                    .nested_for_statement_split_permissions
                    .last()
                    .copied()
                    .unwrap_or(false))
            || self.capturing_for_statements.contains(&identity)
            || !self.contains_current_loop_transfer(&loop_stmt.body)?
        {
            return Ok(None);
        }
        let checkpoint = self.partition_checkpoint();
        self.capturing_for_statements.add(identity.clone());
        let body = self.capture_split_arm(stmt, None, true);
        self.capturing_for_statements.remove(&identity);
        let body = body?;
        let arguments = self.split_arguments(&body.effects);
        let nested_split_lines = self.nested_split_lines(&checkpoint);
        if body.lines.len() + nested_split_lines
            < self.split_thresholds.inner_for_body_split_min_lines
            || arguments.is_none()
        {
            self.restore_partition_checkpoint(&checkpoint);
            return Ok(None);
        }
        let helper = body.into_helper(
            format!(
                "kernel_{}_inner_for_{}",
                self.kernel_index, self.next_inner_for_helper
            ),
            arguments.expect("arguments"),
        );
        self.next_inner_for_helper += 1;
        self.split_helpers.push(helper.clone());
        Ok(Some(helper))
    }
}
