//! Direct native emission of TIRx tile operations, plus the tile scope sync
//! and the per-lane load helpers the tile lowering is the only consumer of.

use crate::tvm_compat::int_value;
use tvm::analysis::Analyzer;
use tvm::ir::{IntImm, IntImmObj, PrimExpr, PrimType, TensorLoadObj};
use tvm::prim::Cast;
use tvm::prim::{CastObj, NotObj, SelectObj};
use tvm::tirx::{BufferVar, ComposeLayout, Layout, TileLayout, TilePrimitiveCallObj};
use tvm::tvm_ffi::object::ObjectRef;
use tvm::tvm_ffi::{Any, ObjectRefCast, ObjectRefCore};

use super::super::analyze::layout::{
    axis_name, expr_any, int_any, op_binary, tile_layout_is_trivial, Extent,
};
use super::super::analyze::memory::MemorySpace;
use super::super::analyze::shapes::structural_equal;
use super::super::analyze::tile_forms::{
    any_value, tile_op_name, AnyValue, ParsedTileCall, TileOpKind, TileOperand, TileRegion,
    TileScalar,
};
use super::super::analyze::topology_operands;
use super::super::analyze::util::{
    as_buffer, as_var, buffer_dtype, buffer_name, buffer_scope, dtype_of, ffi_error, int_imm_expr,
    kind, oref, same, unsupported, upper_first, AResult, Failure,
};
use super::abi;
use super::abi::{element_ref, empty_table_state, mapped_view, named_buffer, table_state};
use super::tile_common::{
    int64_imm, tile_logical_region_indices, tir_var, v2_tile_scope, var_name,
};
use super::NestedLoadSite;
use super::{Emitter, RustValue, Uniformity};
use crate::schema::Schema;
use crate::tables::is_integer_dtype;
use crate::tables::{json_string, round_marker, v2_memory_type_rust};

fn rust_scalar_by_dtype(schema: &Schema, dtype: &str) -> Option<&'static str> {
    if schema.high_precision && crate::tables::is_promoted_float(dtype) {
        return Some("f64");
    }
    Some(match dtype {
        "bool" => "bool",
        "int8" => "i8",
        "int16" => "i16",
        "int32" => "i32",
        "int64" => "i64",
        "uint8" => "u8",
        "uint16" => "u16",
        "uint32" => "u32",
        "uint64" => "u64",
        "float16" | "bfloat16" | "float32" | "float8_e4m3fn" | "float8_e8m0fnu"
        | "float4_e2m1fn" => "f32",
        "float64" => "f64",
        _ => return None,
    })
}

fn zero_literal_by_rust_type(rust_type: &str) -> AResult<&'static str> {
    Ok(match rust_type {
        "bool" => "false",
        "i8" => "0_i8",
        "i16" => "0_i16",
        "i32" => "0_i32",
        "i64" => "0_i64",
        "u8" => "0_u8",
        "u16" => "0_u16",
        "u32" => "0_u32",
        "u64" => "0_u64",
        "f32" => "0.0_f32",
        "f64" => "0.0_f64",
        other => {
            return Err(Failure::Ffi(ffi_error(&format!(
                "no tile zero literal for Rust type {other}"
            ))))
        }
    })
}

const OWNER_DRIVEN_DTYPES: [&str; 3] = ["float16", "bfloat16", "float32"];
const OWNER_DRIVEN_REGISTER_COPY_DTYPES: [&str; 4] =
    ["float16", "bfloat16", "float32", "float8_e4m3fn"];

fn is_owner_region_min_expr_node(schema: &Schema, node_kind: &str) -> bool {
    schema.integer_binary_node_kinds.contains(node_kind)
        || matches!(node_kind, "IntImm" | "Var" | "Cast")
}

fn is_owner_scalar_expr_node(schema: &Schema, node_kind: &str) -> bool {
    schema.integer_binary_node_kinds.contains(node_kind)
        || matches!(
            node_kind,
            "IntImm"
                | "FloatImm"
                | "Var"
                | "TensorLoad"
                | "Cast"
                | "LT"
                | "LE"
                | "GT"
                | "GE"
                | "EQ"
                | "NE"
                | "And"
                | "Or"
                | "Not"
                | "Select"
        )
}

fn is_integer_like_dtype(dtype: &str) -> bool {
    dtype.starts_with("int") || dtype.starts_with("uint")
}

/// `_static_int`: the analyzer-simplified literal, or `None`.
fn static_int(analyzer: &Analyzer, value: &PrimExpr) -> AResult<Option<i64>> {
    Ok(int_imm_expr(&analyzer.simplify(value)?))
}

fn same_expression(analyzer: &Analyzer, lhs: &PrimExpr, rhs: &PrimExpr) -> AResult<bool> {
    Ok(analyzer.can_prove_equal(lhs, rhs)?)
}

fn same_region(analyzer: &Analyzer, lhs: &TileRegion, rhs: &TileRegion) -> AResult<bool> {
    if !(same(lhs.buffer.as_var(), rhs.buffer.as_var())
        && lhs.dtype == rhs.dtype
        && lhs.extents == rhs.extents
        && lhs.mins.len() == rhs.mins.len())
    {
        return Ok(false);
    }
    for (left, right) in lhs.mins.iter().zip(rhs.mins.iter()) {
        if !same_expression(analyzer, left, right)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn int_literal(node: &ObjectRef) -> Option<i64> {
    node.as_node::<IntImmObj>().and_then(|imm| int_value(imm).ok())
}

/// Values and load indices, never buffer
/// definitions or types.
fn visit_value_expression(
    schema: &Schema,
    expr: &ObjectRef,
    visit: &mut dyn FnMut(&ObjectRef) -> AResult<()>,
) -> AResult<()> {
    visit(expr)?;
    if let Some(load) = expr.as_node::<TensorLoadObj>() {
        for index in load.indices.iter() {
            visit_value_expression(schema, &oref(index), visit)?;
        }
        return Ok(());
    }
    let Some(node_kind) = kind(expr) else {
        return Ok(());
    };
    if !is_owner_scalar_expr_node(schema, node_kind) || as_var(expr).is_some() {
        return Ok(());
    }
    if let Some(cast) = expr.as_node::<CastObj>() {
        return visit_value_expression(schema, &oref(cast.value.clone()), visit);
    }
    if let Some((a, b)) = topology_operands(expr) {
        visit_value_expression(schema, &a, visit)?;
        return visit_value_expression(schema, &b, visit);
    }
    if let Some(not) = expr.as_node::<NotObj>() {
        return visit_value_expression(schema, &oref(not.a.clone()), visit);
    }
    if let Some(select) = expr.as_node::<SelectObj>() {
        visit_value_expression(schema, &oref(select.condition.clone()), visit)?;
        visit_value_expression(schema, &oref(select.true_value.clone()), visit)?;
        return visit_value_expression(schema, &oref(select.false_value.clone()), visit);
    }
    Ok(())
}

fn is_owner_region_min_expression(schema: &Schema, expr: &PrimExpr) -> AResult<bool> {
    let mut safe = true;
    visit_value_expression(schema, &oref(expr.clone()), &mut |node| {
        let dtype = dtype_of(node).unwrap_or_default();
        if !safe
            || !kind(node).is_some_and(|kind| is_owner_region_min_expr_node(schema, kind))
            || !is_integer_like_dtype(&dtype)
        {
            safe = false;
        }
        Ok(())
    })?;
    Ok(safe)
}

/// Pure integer address expressions backed only by private state.
fn is_owner_register_copy_min_expression(schema: &Schema, expr: &PrimExpr) -> AResult<bool> {
    let mut safe = true;
    visit_value_expression(schema, &oref(expr.clone()), &mut |node| {
        let dtype = dtype_of(node).unwrap_or_default();
        let node_kind = kind(node).unwrap_or("");
        if !safe
            || !(is_owner_region_min_expr_node(schema, node_kind) || node_kind == "TensorLoad")
            || !is_integer_like_dtype(&dtype)
        {
            safe = false;
            return Ok(());
        }
        if let Some(load) = node.as_node::<TensorLoadObj>() {
            let Some(source) = super::super::analyze::util::as_buffer(&oref(load.source.clone()))
            else {
                safe = false;
                return Ok(());
            };
            let scope = buffer_scope(&source);
            if scope != "local" && scope != "local_scalar" {
                safe = false;
            }
        }
        Ok(())
    })?;
    Ok(safe)
}

fn iter_axis_name(item: &tvm::tirx::Iter) -> AResult<String> {
    axis_name(&item.axis)
}

fn tile_layout_of(buffer: &BufferVar) -> Option<TileLayout> {
    buffer
        .buffer_type()
        .layout
        .clone()
        .and_then(|layout| layout.try_cast::<TileLayout>().ok())
}

fn canonical_tile_layout(layout: &TileLayout) -> AResult<Option<TileLayout>> {
    Ok(Layout::from(layout.clone())
        .canonicalize()?
        .try_cast::<TileLayout>()
        .ok())
}

/// A Python `int` handed to `tirx.Cast(dtype, value)`: TVM converts it to an
/// `int32` literal.
fn python_int_expr(value: i64) -> AResult<PrimExpr> {
    Ok(IntImm::new("int32", value)?.into())
}

fn cast_to(dtype: &str, value: PrimExpr) -> AResult<PrimExpr> {
    Ok(Cast::new(PrimType::new(dtype)?, value)?.into())
}

fn add_expr(lhs: &PrimExpr, rhs: &PrimExpr) -> AResult<PrimExpr> {
    op_binary("_OpAdd", expr_any(lhs), expr_any(rhs))
}

fn py_str_sorted_list(items: &[String]) -> String {
    let mut sorted: Vec<&String> = items.iter().collect();
    sorted.sort();
    format!(
        "[{}]",
        sorted
            .iter()
            .map(|item| (item).to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )
}

/// `type(value).__name__ == "TensorLoad"` and friends over `TileScalar.expr`.
fn scalar_node(scalar: &TileScalar) -> AResult<Option<ObjectRef>> {
    Ok(match any_value(&scalar.expr)? {
        AnyValue::Object(node) => Some(node),
        _ => None,
    })
}

/// The elementwise result emitters of the tile lowerings.
#[derive(Clone, Copy)]
enum ResultEmitter {
    Zero,
    Fill,
    Arithmetic {
        instruction: &'static str,
        narrow_method: &'static str,
    },
    FloatBinary {
        instruction: &'static str,
        f32_variant: &'static str,
        f64_variant: &'static str,
        error_message: &'static str,
    },
    Fma,
    Transcendental {
        rust_method: &'static str,
        direct_instruction: Option<&'static str>,
    },
    Reciprocal,
    Silu,
}

#[derive(Clone, Copy)]
enum IdentityBuilder {
    Sum,
    Max,
    Min,
}

#[derive(Clone, Copy)]
struct ReductionLowering {
    identity_builder: IdentityBuilder,
    instruction: &'static str,
    narrow_method: &'static str,
    sum_reduction: bool,
    packed_identity: &'static str,
}

enum TileLowering {
    Copy {
        cast: bool,
        register_copy: bool,
    },
    CopyAsync,
    Elementwise {
        result: ResultEmitter,
        owner_driven: bool,
    },
    Reduction(ReductionLowering),
    Gemm,
    GemmAsync,
}

/// Public TIRx op -> exactly one bound lowering.
fn tile_lowering(kind: TileOpKind) -> TileLowering {
    match kind {
        TileOpKind::Copy | TileOpKind::PermuteLayout => TileLowering::Copy {
            cast: false,
            register_copy: true,
        },
        TileOpKind::CopyAsync => TileLowering::CopyAsync,
        TileOpKind::Cast => TileLowering::Copy {
            cast: true,
            register_copy: false,
        },
        TileOpKind::Zero => TileLowering::Elementwise {
            result: ResultEmitter::Zero,
            owner_driven: false,
        },
        TileOpKind::Sqrt => TileLowering::Elementwise {
            result: ResultEmitter::Transcendental {
                rust_method: "sqrt",
                direct_instruction: Some("sqrt"),
            },
            owner_driven: false,
        },
        TileOpKind::Fill => TileLowering::Elementwise {
            result: ResultEmitter::Fill,
            owner_driven: false,
        },
        TileOpKind::Add => TileLowering::Elementwise {
            result: ResultEmitter::Arithmetic {
                instruction: "add",
                narrow_method: "wrapping_add",
            },
            owner_driven: true,
        },
        TileOpKind::Sub => TileLowering::Elementwise {
            result: ResultEmitter::Arithmetic {
                instruction: "sub",
                narrow_method: "wrapping_sub",
            },
            owner_driven: true,
        },
        TileOpKind::Mul => TileLowering::Elementwise {
            result: ResultEmitter::Arithmetic {
                instruction: "mul",
                narrow_method: "wrapping_mul",
            },
            owner_driven: true,
        },
        TileOpKind::Maximum => TileLowering::Elementwise {
            result: ResultEmitter::FloatBinary {
                instruction: "max",
                f32_variant: "v2::reg::variant::F32",
                f64_variant: "v2::reg::variant::F64",
                error_message:
                    "Native maximum tile lowering requires uniform floating-point values",
            },
            owner_driven: false,
        },
        // CUDA's scalar fallback emits ordinary division for every requested mode.
        TileOpKind::Fdiv => TileLowering::Elementwise {
            result: ResultEmitter::FloatBinary {
                instruction: "div",
                f32_variant: "v2::reg::variant::F32Rn",
                f64_variant: "v2::reg::variant::F64Rn",
                error_message: "Native fdiv tile lowering requires uniform f32 or f64 values",
            },
            owner_driven: true,
        },
        TileOpKind::Fma => TileLowering::Elementwise {
            result: ResultEmitter::Fma,
            owner_driven: true,
        },
        TileOpKind::Reciprocal => TileLowering::Elementwise {
            result: ResultEmitter::Reciprocal,
            owner_driven: false,
        },
        TileOpKind::Silu => TileLowering::Elementwise {
            result: ResultEmitter::Silu,
            owner_driven: false,
        },
        TileOpKind::Exp => TileLowering::Elementwise {
            result: ResultEmitter::Transcendental {
                rust_method: "exp",
                direct_instruction: None,
            },
            owner_driven: false,
        },
        TileOpKind::Exp2 => TileLowering::Elementwise {
            result: ResultEmitter::Transcendental {
                rust_method: "exp2",
                direct_instruction: None,
            },
            owner_driven: false,
        },
        TileOpKind::Log2 => TileLowering::Elementwise {
            result: ResultEmitter::Transcendental {
                rust_method: "log2",
                direct_instruction: Some("lg2"),
            },
            owner_driven: false,
        },
        TileOpKind::Sum => TileLowering::Reduction(ReductionLowering {
            identity_builder: IdentityBuilder::Sum,
            instruction: "add",
            narrow_method: "wrapping_add",
            sum_reduction: true,
            packed_identity: "0.0_f32",
        }),
        TileOpKind::Max => TileLowering::Reduction(ReductionLowering {
            identity_builder: IdentityBuilder::Max,
            instruction: "max",
            narrow_method: "max",
            sum_reduction: false,
            packed_identity: "f32::NEG_INFINITY",
        }),
        TileOpKind::Min => TileLowering::Reduction(ReductionLowering {
            identity_builder: IdentityBuilder::Min,
            instruction: "min",
            narrow_method: "min",
            sum_reduction: false,
            packed_identity: "f32::INFINITY",
        }),
        TileOpKind::Gemm => TileLowering::Gemm,
        TileOpKind::GemmAsync => TileLowering::GemmAsync,
    }
}

// ----------------------------------------------------------------------
// Buffer loader overrides installed by the tile lowering.
// ----------------------------------------------------------------------

/// `Layout.is_swizzle()`: a ComposeLayout over a trivial tile.
fn layout_is_swizzle(layout: &Layout) -> AResult<bool> {
    match layout.clone().try_cast::<ComposeLayout>() {
        Ok(compose) => tile_layout_is_trivial(&compose.tile_layout()?),
        Err(_) => Ok(false),
    }
}

/// `region.buffer.shape` as primitive expressions.
fn buffer_shape(buffer: &BufferVar) -> Vec<PrimExpr> {
    buffer.buffer_type().shape.iter().collect()
}

/// `len(layout.replica) == 0 and len(layout.offset) == 0`.
fn tile_layout_is_pure_shard(layout: &TileLayout) -> AResult<bool> {
    Ok(layout.replica()?.is_empty() && layout.offset()?.iter().next().is_none())
}

fn space_marker(space: MemorySpace) -> Option<&'static str> {
    match space {
        MemorySpace::Global => Some("v2::Global"),
        MemorySpace::Shared => Some("v2::Shared"),
        MemorySpace::Local | MemorySpace::Register => Some("v2::Local"),
        MemorySpace::Tmem => None,
    }
}

impl<'a> Emitter<'a> {
    /// The `buffers.<field>` spelling of a buffer.
    fn tile_buffer_field(&self, buffer: &BufferVar) -> AResult<String> {
        let code = self.buffer_code(buffer)?;
        match code.storage_index {
            Some(index) => Ok(format!("buffers.buffers[{index}]")),
            None => unsupported(format!(
                "buffer:{}:dynamic DeclBuffer view is used before declaration",
                self.plan_of(code).name
            )),
        }
    }

    fn tile_stateful(
        &mut self,
        function: &str,
        source_op_id: Option<i64>,
        arguments: &[String],
        variant: Option<&str>,
        context: Option<&str>,
        await_result: bool,
    ) -> String {
        let site = self.v2_site(source_op_id);
        abi::warp_call(
            function,
            &site,
            arguments,
            variant,
            context,
            await_result,
            true,
        )
    }

    // ------------------------------------------------------------------
    // Tile scope sync and its barrier helper.
    // ------------------------------------------------------------------

    /// `emit_tile_scope_sync`: a tile completion boundary without a sync call.
    fn emit_tile_scope_sync(
        &mut self,
        scope: &str,
        source_op_id: i64,
        actual_barrier: bool,
    ) -> AResult<()> {
        if scope == "thread" {
            return Ok(());
        }
        if scope == "warp" {
            let line = format!(
                "{};",
                self.tile_stateful(
                    "warp::bar_warp_sync",
                    Some(source_op_id),
                    &[abi::splat("u32::MAX")],
                    None,
                    None,
                    true,
                )
            );
            self.emit_suspend_line(&line);
            return Ok(());
        }
        if scope == "warpgroup" && !actual_barrier {
            let line = format!(
                "{};",
                self.tile_stateful(
                    "collective::participate",
                    Some(source_op_id),
                    &[],
                    Some("v2::collective::scope::Warpgroup"),
                    None,
                    true,
                )
            );
            self.emit_suspend_line(&line);
            return Ok(());
        }
        if scope == "warpgroup" || scope == "cta" {
            return self.emit_scope_barrier(scope, source_op_id);
        }
        unsupported(format!(
            "tile semantic completion is not implemented for {:?}",
            scope
        ))
    }

    /// One fresh synthetic site per boundary.
    fn tile_emit_scope_sync(&mut self, op: &ParsedTileCall, actual_barrier: bool) -> AResult<()> {
        let site = self.synthetic_site_id() as i64;
        self.emit_tile_scope_sync(&op.exec_scope, site, actual_barrier)
    }

    fn tile_emit_scope_participation(
        &mut self,
        op: &ParsedTileCall,
        source_op_id: i64,
    ) -> AResult<()> {
        if op.exec_scope == "thread" {
            return Ok(());
        }
        if !matches!(op.exec_scope.as_str(), "warp" | "warpgroup" | "cta") {
            return unsupported(format!(
                "Tile participation is not implemented for {:?}",
                &op.exec_scope
            ));
        }
        let site = abi::site(source_op_id as u64);
        let line = format!(
            "{};",
            abi::warp_call(
                "collective::participate",
                &site,
                &[],
                Some(&format!(
                    "v2::collective::scope::{}",
                    upper_first(&op.exec_scope)
                )),
                None,
                true,
                true,
            )
        );
        self.emit_suspend_line(&line);
        Ok(())
    }

    // ------------------------------------------------------------------
    // Scalar operands, loops and coordinates.
    // ------------------------------------------------------------------

    /// Exact source matching.
    fn tile_emit_explicit_expression(&mut self, scalar: &TileScalar) -> AResult<RustValue> {
        let Some(node) = scalar_node(scalar)? else {
            return Err(Failure::Ffi(ffi_error(
                "tile scalar operand is not an IR node",
            )));
        };
        self.with_load_site(Some(NestedLoadSite::Exact), |emitter| {
            emitter.emit_expr(&node)
        })
    }

    fn tile_owner_slot_loop(
        &mut self,
        extent: i64,
        body: impl FnOnce(&mut Self, &[PrimExpr]) -> AResult<()>,
    ) -> AResult<()> {
        let suffix = self.control_name("tile_owner");
        let row_name = format!("tile_owner_rows_{suffix}");
        let slot_name = format!("tile_owner_slot_{suffix}");
        self.emit_line(&format!(
            "let {row_name} = WarpValue::from_fn(|lane| (((ctx.warp_id_in_cta() % {}) * WARP_SIZE + lane) as i64));",
            self.warps_per_warpgroup
        ));
        self.emit_line(&format!("for {slot_name} in 0_i64..{extent}_i64 {{"));
        self.indent += 1;
        let (row_node, row) = tir_var(&row_name, "int64")?;
        let (slot_node, slot) = tir_var(&slot_name, "int64")?;
        let snapshot = self.scope_snapshot();
        self.variables.set(
            row_node,
            RustValue::new(row_name.clone(), "i64", Uniformity::Varying),
        );
        self.variables.set(
            slot_node,
            RustValue::new(slot_name.clone(), "i64", Uniformity::Uniform),
        );
        let result = body(self, &[row, slot]);
        self.restore_scope(snapshot);
        self.indent -= 1;
        self.emit_line("}");
        result
    }

    fn tile_region_indices_from_coordinates(
        &self,
        region: &TileRegion,
        coordinates: &[PrimExpr],
    ) -> AResult<Vec<PrimExpr>> {
        if coordinates.len() != region.extents.len() {
            return unsupported("Internal tile coordinate rank mismatch");
        }
        let mut indices = Vec::new();
        for (minimum, coordinate) in region.mins.iter().zip(coordinates.iter()) {
            let dtype = dtype_of(&oref(minimum.clone()))?;
            let cast = cast_to(&dtype, coordinate.clone())?;
            indices.push(add_expr(minimum, &cast)?);
        }
        Ok(indices)
    }

    fn tile_region_indices(
        &mut self,
        region: &TileRegion,
        linear: &PrimExpr,
    ) -> AResult<Vec<PrimExpr>> {
        let coordinates = self.tile_coordinates(linear, &region.extents)?;
        self.tile_region_indices_from_coordinates(region, &coordinates)
    }

    fn tile_broadcast_region_indices(
        &mut self,
        destination: &TileRegion,
        region: &TileRegion,
        linear: &PrimExpr,
    ) -> AResult<Vec<PrimExpr>> {
        if destination.extents.len() < region.extents.len() {
            return unsupported("Internal tile broadcast rank mismatch");
        }
        let rank_padding = destination.extents.len() - region.extents.len();
        let destination_coordinates = self.tile_coordinates(linear, &destination.extents)?;
        let mut source_coordinates = Vec::new();
        for (axis, extent) in region.extents.iter().enumerate() {
            source_coordinates.push(if *extent == 1 {
                int64_imm(0)?
            } else {
                destination_coordinates[rank_padding + axis].clone()
            });
        }
        self.tile_region_indices_from_coordinates(region, &source_coordinates)
    }

    // ------------------------------------------------------------------
    // Region loads, stores and masks.
    // ------------------------------------------------------------------

    fn tile_load_owner_aligned_region(
        &mut self,
        region: &TileRegion,
        logical_coordinates: &[PrimExpr],
        source_op_id: i64,
    ) -> AResult<RustValue> {
        let indices = tile_logical_region_indices(region, logical_coordinates)?;
        let index = self.physical_index(&region.buffer, &indices)?;
        let field = self.tile_buffer_field(&region.buffer)?;
        let memory_dtype = if !self.ctx.schema.high_precision && region.dtype == "float8_e4m3fn" {
            "uint8"
        } else {
            region.dtype.as_str()
        };
        let marker = match v2_memory_type_rust(self.ctx.schema, memory_dtype) {
            Ok(marker) => marker,
            Err(Failure::Unsupported { .. }) => {
                return unsupported(format!(
                    "Owner-driven tile load is not implemented for {}",
                    region.dtype
                ))
            }
            Err(error) => return Err(error),
        };
        let load_site = self.implicit_source_op_id(&region.buffer, source_op_id)?;
        let space = self.memory_plan.resolve(&region.buffer)?.space;
        let Some(space_marker) = space_marker(space) else {
            return unsupported(format!(
                "Owner-driven tile load has no scalar memory form for {}",
                space.value()
            ));
        };
        let raw = self.control_name("tile_owner_load_raw");
        let itemsize = self
            .ctx
            .inspect_layout(&region.buffer, &self.bindings)?
            .itemsize;
        let logical_name = self.logical_buffer_name(&region.buffer)?;
        let address = abi::buffer_address(
            space_marker,
            &field,
            &format!("({})", index.code),
            itemsize,
            &logical_name,
        );
        let load = self.tile_stateful(
            "mem::ld",
            load_site,
            &[address],
            Some(&format!("v2::mem::variant::Ld<{marker}, {space_marker}>")),
            None,
            false,
        );
        self.emit_line(&format!("let {raw} = v2_register_out({load});"));
        let mut name = raw.clone();
        if !self.ctx.schema.high_precision && region.dtype == "float8_e4m3fn" {
            name = self.control_name("tile_owner_load");
            self.emit_line(&format!(
                "let {name} = WarpValue::from_fn(|lane| float8_e4m3fn_bits_to_f32({raw}[lane]));"
            ));
        }
        Ok(RustValue::new(
            name,
            crate::tables::precision_scalar_type(self.ctx.schema, &region.dtype, "f32"),
            Uniformity::Varying,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn tile_store_region_at_indices(
        &mut self,
        region: &TileRegion,
        indices: &[PrimExpr],
        value: RustValue,
        access_mask: Option<&str>,
        owner_aligned: bool,
        source_op_id: i64,
    ) -> AResult<()> {
        let Some(rust_type) = rust_scalar_by_dtype(self.ctx.schema, &region.dtype) else {
            return unsupported(format!(
                "Tile store lowering is not implemented for {}",
                region.dtype
            ));
        };
        let value = self.as_warp_value(value);
        if value.rust_type != rust_type {
            return unsupported(format!(
                "Tile store for {} produced Rust value {}",
                region.dtype, value.rust_type
            ));
        }
        let mut access_mask = access_mask.unwrap_or("ctx.active_mask()").to_owned();
        if !owner_aligned {
            access_mask = self.physical_access_mask(&region.buffer, indices, &access_mask)?;
        }
        let field = self.tile_buffer_field(&region.buffer)?;
        let store_site = self.implicit_source_op_id(&region.buffer, source_op_id)?;
        let space = self.memory_plan.resolve(&region.buffer)?.space;
        if region.memory_scope == "tmem" {
            let (lane, tcol, allocated_addr) =
                self.tmem_coordinates(&region.buffer, indices, None)?;
            let access_marker = self.tmem_access_marker(&region.buffer)?;
            let memory_dtype =
                if region.dtype == "float8_e4m3fn" || region.dtype == "float8_e8m0fnu" {
                    "uint8"
                } else {
                    region.dtype.as_str()
                };
            let mut stored_value = value.code.clone();
            if region.dtype == "float8_e4m3fn" {
                stored_value = self.control_name("tile_float8_store_bits");
                self.emit_line(&format!(
                    "let {stored_value} = WarpValue::from_fn(|lane| f32_to_float8_e4m3fn_bits({}[lane]));",
                    value.code
                ));
            } else if region.dtype == "float8_e8m0fnu" {
                stored_value = self.control_name("tile_float8_store_bits");
                self.emit_line(&format!(
                    "let {stored_value} = WarpValue::from_fn(|lane| f32_to_float8_e8m0fnu_bits({}[lane]));",
                    value.code
                ));
            }
            let marker = v2_memory_type_rust(self.ctx.schema, memory_dtype)?;
            let logical_name = self.logical_buffer_name(&region.buffer)?;
            let operands = [
                named_buffer("v2::Tmem", &field, &logical_name),
                abi::register(&lane.code),
                abi::register(&tcol.code),
                abi::register(&allocated_addr.code),
                abi::register(&stored_value),
            ]
            .join(", ");
            let invocation = self.tile_stateful(
                "tmem::write",
                store_site,
                &[format!("({operands})")],
                Some(&format!(
                    "v2::tmem::variant::Access<{marker}, {access_marker}>"
                )),
                Some(&abi::context(&format!(
                    "ctx.with_active_mask({access_mask})"
                ))),
                false,
            );
            self.emit_line(&format!("{invocation};"));
            return Ok(());
        }
        let index = self.physical_index(&region.buffer, indices)?;
        let Some(space_marker) = space_marker(space) else {
            return unsupported(format!(
                "Tile store has no scalar memory form for {}",
                space.value()
            ));
        };
        let memory_dtype = if !self.ctx.schema.high_precision
            && (region.dtype == "float8_e4m3fn" || region.dtype == "float8_e8m0fnu")
        {
            "uint8"
        } else {
            region.dtype.as_str()
        };
        let mut stored_value = value.code.clone();
        if !self.ctx.schema.high_precision && region.dtype == "float8_e4m3fn" {
            stored_value = self.control_name("tile_float8_store_bits");
            self.emit_line(&format!(
                "let {stored_value} = WarpValue::from_fn(|lane| f32_to_float8_e4m3fn_bits({}[lane]));",
                value.code
            ));
        } else if !self.ctx.schema.high_precision && region.dtype == "float8_e8m0fnu" {
            stored_value = self.control_name("tile_float8_store_bits");
            self.emit_line(&format!(
                "let {stored_value} = WarpValue::from_fn(|lane| f32_to_float8_e8m0fnu_bits({}[lane]));",
                value.code
            ));
        }
        let marker = v2_memory_type_rust(self.ctx.schema, memory_dtype)?;
        let itemsize = self
            .ctx
            .inspect_layout(&region.buffer, &self.bindings)?
            .itemsize;
        let logical_name = self.logical_buffer_name(&region.buffer)?;
        let address = abi::buffer_address(
            space_marker,
            &field,
            &format!("({})", index.code),
            itemsize,
            &logical_name,
        );
        let invocation = self.tile_stateful(
            "mem::st",
            store_site,
            &[format!("({address}, {})", abi::register(&stored_value))],
            Some(&format!("v2::mem::variant::St<{marker}, {space_marker}>")),
            Some(&abi::context(&format!(
                "ctx.with_active_mask({access_mask})"
            ))),
            false,
        );
        self.emit_line(&format!("{invocation};"));
        Ok(())
    }

    fn tile_store_region(
        &mut self,
        region: &TileRegion,
        linear: &PrimExpr,
        value: RustValue,
        access_mask: Option<&str>,
        source_op_id: i64,
    ) -> AResult<()> {
        let indices = self.tile_region_indices(region, linear)?;
        self.tile_store_region_at_indices(region, &indices, value, access_mask, false, source_op_id)
    }

    /// `set(inspect_buffer_layout(buffer).physical_axes) - {"m"}` is non-empty.
    fn tile_has_owner_axes(&self, buffer: &BufferVar) -> AResult<bool> {
        let info = self.ctx.inspect_layout(buffer, &self.bindings)?;
        Ok(info.physical_axes.iter().any(|axis| axis != "m"))
    }

    fn tile_elementwise_operation_mask(
        &mut self,
        op: &ParsedTileCall,
        regions: &[TileRegion],
        linear: &PrimExpr,
        base_mask: &str,
    ) -> AResult<String> {
        let mut masks = Vec::new();
        for (position, region) in regions.iter().enumerate() {
            if !self.tile_has_owner_axes(&region.buffer)? {
                continue;
            }
            let indices = if position == 0 {
                self.tile_region_indices(region, linear)?
            } else {
                self.tile_broadcast_region_indices(&op.destination, region, linear)?
            };
            masks.push(self.physical_access_mask(&region.buffer, &indices, base_mask)?);
        }
        if masks.is_empty() {
            return Ok(base_mask.to_owned());
        }
        if masks.len() == 1 {
            return Ok(masks.remove(0));
        }
        let name = self.control_name("tile_owner_mask");
        self.emit_line(&format!("let {name} = {};", masks.join(" & ")));
        Ok(name)
    }

    fn tile_distributed_owner_axes(&self, region: &TileRegion) -> AResult<Vec<String>> {
        if region.memory_scope != "local" {
            return Ok(Vec::new());
        }
        let info = self.ctx.inspect_layout(&region.buffer, &self.bindings)?;
        Ok(info
            .physical_axes
            .iter()
            .filter(|axis| matches!(axis.as_str(), "laneid" | "wid_in_wg" | "tid_in_wg"))
            .cloned()
            .collect())
    }

    fn tile_same_distributed_owner_mapping(
        &self,
        left: &TileRegion,
        right: &TileRegion,
    ) -> AResult<bool> {
        if left.extents != right.extents || left.mins.len() != right.mins.len() {
            return Ok(false);
        }
        let analyzer = &self.ctx.analyzer;
        for (left_min, right_min) in left.mins.iter().zip(right.mins.iter()) {
            if !analyzer.can_prove_equal(left_min, right_min)? {
                return Ok(false);
            }
        }
        let left_shape = buffer_shape(&left.buffer);
        let right_shape = buffer_shape(&right.buffer);
        if left_shape.len() != right_shape.len() {
            return Ok(false);
        }
        for (left_extent, right_extent) in left_shape.iter().zip(right_shape.iter()) {
            if !analyzer.can_prove_equal(left_extent, right_extent)? {
                return Ok(false);
            }
        }
        let canonical = |region: &TileRegion| -> AResult<Any> {
            let Some(layout) = region.buffer.buffer_type().layout.clone() else {
                return Err(Failure::Ffi(ffi_error(
                    "'NoneType' object has no attribute 'canonicalize'",
                )));
            };
            Ok(Any::from(layout.canonicalize()?))
        };
        structural_equal(&canonical(left)?, &canonical(right)?)
    }

    fn tile_needs_distributed_owner_transport(&self, regions: &[TileRegion]) -> AResult<bool> {
        let mut distributed = Vec::new();
        for region in regions {
            if !self.tile_distributed_owner_axes(region)?.is_empty() {
                distributed.push(region);
            }
        }
        if distributed.len() < 2 {
            return Ok(false);
        }
        let first = distributed[0];
        for region in &distributed[1..] {
            if !self.tile_same_distributed_owner_mapping(first, region)? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn tile_semantic_element_owner_mask(
        &mut self,
        op: &ParsedTileCall,
        regions: &[TileRegion],
        linear: &PrimExpr,
    ) -> AResult<String> {
        if regions.iter().any(|region| region.memory_scope == "local") {
            return Ok("ctx.active_mask()".to_owned());
        }
        for region in regions {
            for minimum in &region.mins {
                let value = self.emit_expr(&oref(minimum.clone()))?;
                let value = self.as_i64(value)?;
                if value.uniformity == Uniformity::Varying {
                    let rendered_label = json_string(&format!(
                        "warp tile region minimum for {}",
                        buffer_name(&region.buffer)
                    ));
                    self.emit_line(&format!(
                        "let _ = require_uniform_i64(&{}, ctx.active_mask(), {rendered_label})?;",
                        value.code
                    ));
                }
            }
        }
        for region in regions {
            if self.tile_has_owner_axes(&region.buffer)? {
                return Ok("ctx.active_mask()".to_owned());
            }
        }
        let (thread_count, current_scope_warp) = match op.exec_scope.as_str() {
            "warp" => (32, "0_usize".to_owned()),
            "warpgroup" => (
                self.warps_per_warpgroup * 32,
                format!("ctx.warp_id_in_cta() % {}_usize", self.warps_per_warpgroup),
            ),
            "cta" => (self.warps_per_cta * 32, "ctx.warp_id_in_cta()".to_owned()),
            _ => return Ok("ctx.active_mask()".to_owned()),
        };
        let owner_thread = self.control_name("tile_semantic_owner_thread");
        let owner_mask = self.control_name("tile_semantic_owner_mask");
        self.emit_line(&format!(
            "let {owner_thread} = usize::try_from({} % {thread_count}_i64).map_err(|_| EngineError::message(\"negative tile semantic owner\"))?;",
            var_name(linear)?
        ));
        self.emit_line(&format!(
            "let {owner_mask} = if {owner_thread} / WARP_SIZE == {current_scope_warp} {{ WarpMask::from_bits(1_u32 << ({owner_thread} % WARP_SIZE)) }} else {{ WarpMask::EMPTY }};"
        ));
        Ok(owner_mask)
    }
}

// ----------------------------------------------------------------------
// Owner-driven and owner-transported forms.
// ----------------------------------------------------------------------

impl<'a> Emitter<'a> {
    fn tile_canonical_tid_owner_layout(&self, region: &TileRegion) -> AResult<Option<TileLayout>> {
        if region.memory_scope != "local" || region.extents.len() != 2 {
            return Ok(None);
        }
        if region.extents[0] != 128 || region.extents[1] <= 1 {
            return Ok(None);
        }
        if int_literal(&oref(region.mins[0].clone())) != Some(0) {
            return Ok(None);
        }
        for minimum in &region.mins {
            if !is_owner_region_min_expression(self.ctx.schema, minimum)? {
                return Ok(None);
            }
        }
        let analyzer = &self.ctx.analyzer;
        let mut shape = Vec::new();
        for extent in buffer_shape(&region.buffer) {
            shape.push(static_int(analyzer, &extent)?);
        }
        if shape.len() != 2 || shape[0] != Some(128) {
            return Ok(None);
        }
        let Some(columns) = shape[1] else {
            return Ok(None);
        };
        if columns <= 1 {
            return Ok(None);
        }
        let Some(layout) = tile_layout_of(&region.buffer) else {
            return Ok(None);
        };
        let Some(canonical) = canonical_tile_layout(&layout)? else {
            return Ok(None);
        };
        if !tile_layout_is_pure_shard(&canonical)? {
            return Ok(None);
        }
        let shard: Vec<tvm::tirx::Iter> = canonical.shard()?.iter().collect();
        if shard.len() != 2 {
            return Ok(None);
        }
        let (owner, local) = (&shard[0], &shard[1]);
        if iter_axis_name(owner)? != "tid_in_wg"
            || static_int(analyzer, &owner.extent)? != Some(128)
            || static_int(analyzer, &owner.stride)? != Some(1)
            || iter_axis_name(local)? != "m"
            || static_int(analyzer, &local.extent)? != Some(columns)
            || static_int(analyzer, &local.stride)? != Some(1)
        {
            return Ok(None);
        }
        let info = self.ctx.inspect_layout(&region.buffer, &self.bindings)?;
        if info.physical_axes != ["m", "tid_in_wg"] || info.element_count != Some(columns) {
            return Ok(None);
        }
        Ok(Some(canonical))
    }

    fn tile_scalar_operands_are_independent(
        &self,
        op: &ParsedTileCall,
        occupied_backings: &[usize],
    ) -> AResult<bool> {
        for operand in &op.operands {
            let Some(scalar) = operand.scalar() else {
                continue;
            };
            let Some(node) = scalar_node(scalar)? else {
                continue;
            };
            let mut safe = true;
            let mut failure: Option<Failure> = None;
            visit_value_expression(self.ctx.schema, &node, &mut |node| {
                let node_kind = kind(node).unwrap_or("");
                if !safe || !is_owner_scalar_expr_node(self.ctx.schema, node_kind) {
                    safe = false;
                    return Ok(());
                }
                if node_kind != "TensorLoad" {
                    return Ok(());
                }
                match self.tile_scalar_load_is_independent(node, occupied_backings) {
                    Ok(value) => safe = value,
                    Err(error) => {
                        safe = false;
                        if failure.is_none() {
                            failure = Some(error);
                        }
                    }
                }
                Ok(())
            })?;
            if let Some(error) = failure {
                return Err(error);
            }
            if !safe {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// The `TensorLoad` arm of the scalar-operand independence visitor.
    fn tile_scalar_load_is_independent(
        &self,
        node: &ObjectRef,
        occupied_backings: &[usize],
    ) -> AResult<bool> {
        let load = node.as_node::<TensorLoadObj>().expect("TensorLoad");
        let Some(source) = as_buffer(&oref(load.source.clone())) else {
            return Err(Failure::Ffi(ffi_error("TensorLoad source is not a buffer")));
        };
        let scope = buffer_scope(&source);
        let info = self.ctx.inspect_layout(&source, &self.bindings)?;
        let plan = self.memory_plan.resolve(&source)?;
        let canonical = match tile_layout_of(&source) {
            Some(layout) => canonical_tile_layout(&layout)?,
            None => None,
        };
        let analyzer = &self.ctx.analyzer;
        let mut indices_zero = true;
        for index in load.indices.iter() {
            if int_literal(&oref(index)) != Some(0) {
                indices_zero = false;
            }
        }
        let Some(canonical) = canonical else {
            return Ok(false);
        };
        let shard: Vec<tvm::tirx::Iter> = canonical.shard()?.iter().collect();
        Ok(matches!(scope.as_str(), "local" | "local_scalar")
            && OWNER_DRIVEN_DTYPES.contains(&buffer_dtype(&source).as_str())
            && matches!(info.shape.as_slice(), [Extent::Static(1)])
            && info.physical_axes == ["m"]
            && info.element_count == Some(1)
            && tile_layout_is_pure_shard(&canonical)?
            && shard.len() == 1
            && iter_axis_name(&shard[0])? == "m"
            && static_int(analyzer, &shard[0].extent)? == Some(1)
            && static_int(analyzer, &shard[0].stride)? == Some(1)
            && plan
                .backing_index
                .is_some_and(|backing| !occupied_backings.contains(&backing))
            && indices_zero)
    }

    fn tile_owner_driven_extent(&self, op: &ParsedTileCall, enabled: bool) -> AResult<Option<i64>> {
        if !enabled || !self.analysis_capable {
            return Ok(None);
        }
        if op.exec_scope != "warpgroup" {
            return Ok(None);
        }
        if op.attr_bool("snapshot_source") || op.attr_bool("zero_fill_invalid_source") {
            return Ok(None);
        }
        let mut regions: Vec<&TileRegion> = vec![&op.destination];
        regions.extend(op.operands.iter().filter_map(TileOperand::region));
        if regions
            .iter()
            .any(|region| !OWNER_DRIVEN_DTYPES.contains(&region.dtype.as_str()))
        {
            return Ok(None);
        }
        let baseline = regions[0];
        let Some(baseline_layout) = self.tile_canonical_tid_owner_layout(baseline)? else {
            return Ok(None);
        };
        let analyzer = &self.ctx.analyzer;
        for region in &regions[1..] {
            let Some(layout) = self.tile_canonical_tid_owner_layout(region)? else {
                return Ok(None);
            };
            if !structural_equal(&Any::from(layout), &Any::from(baseline_layout.clone()))? {
                return Ok(None);
            }
            if region.extents != baseline.extents || region.mins.len() != baseline.mins.len() {
                return Ok(None);
            }
            for (candidate, expected) in region.mins.iter().zip(baseline.mins.iter()) {
                if !same_expression(analyzer, candidate, expected)? {
                    return Ok(None);
                }
            }
        }
        let mut plans = Vec::new();
        for region in &regions {
            plans.push(self.memory_plan.resolve(&region.buffer)?);
        }
        if plans.iter().any(|plan| plan.backing_index.is_none()) {
            return Ok(None);
        }
        let destination_backing = plans[0].backing_index;
        for (operand, source_plan) in op
            .operands
            .iter()
            .filter_map(TileOperand::region)
            .zip(plans[1..].iter())
        {
            if source_plan.backing_index == destination_backing
                && !same_region(analyzer, &op.destination, operand)?
            {
                return Ok(None);
            }
        }
        let occupied: Vec<usize> = plans.iter().filter_map(|plan| plan.backing_index).collect();
        if !self.tile_scalar_operands_are_independent(op, &occupied)? {
            return Ok(None);
        }
        Ok(Some(baseline.extents[1]))
    }

    fn tile_owner_driven_register_copy_extent(
        &self,
        op: &ParsedTileCall,
        source: &TileRegion,
        enabled: bool,
    ) -> AResult<Option<i64>> {
        if !enabled
            || op.exec_scope != "warpgroup"
            || source.memory_scope != "local"
            || op.destination.memory_scope != "shared"
            || source.dtype != op.destination.dtype
            || !OWNER_DRIVEN_REGISTER_COPY_DTYPES.contains(&source.dtype.as_str())
        {
            return Ok(None);
        }
        if op.attr_bool("zero_fill_invalid_source") {
            return Ok(None);
        }
        if self.tile_canonical_tid_owner_layout(source)?.is_none() {
            return Ok(None);
        }
        if source.logical_shape() != op.destination.logical_shape() {
            return Ok(None);
        }
        for minimum in &op.destination.mins {
            if !is_owner_register_copy_min_expression(self.ctx.schema, minimum)? {
                return Ok(None);
            }
        }
        let destination_info = self
            .ctx
            .inspect_layout(&op.destination.buffer, &self.bindings)?;
        let Some(destination_layout) = op.destination.buffer.buffer_type().layout.clone() else {
            return Err(Failure::Ffi(ffi_error(
                "'NoneType' object has no attribute 'is_swizzle'",
            )));
        };
        if destination_info.physical_axes != ["m"]
            || destination_layout
                .clone()
                .try_cast::<ComposeLayout>()
                .is_err()
            || !layout_is_swizzle(&destination_layout)?
        {
            return Ok(None);
        }
        let source_plan = self.memory_plan.resolve(&source.buffer)?;
        let destination_plan = self.memory_plan.resolve(&op.destination.buffer)?;
        // Private source ownership plus distinct backings makes the generic
        // all-coordinate snapshot unnecessary: each lane can copy its own row.
        if source_plan.backing_index.is_none()
            || destination_plan.backing_index.is_none()
            || source_plan.backing_index == destination_plan.backing_index
        {
            return Ok(None);
        }
        Ok(Some(source.extents[1]))
    }

    fn tile_unique_owner_location(
        &mut self,
        op: &ParsedTileCall,
        region: &TileRegion,
        indices: &[PrimExpr],
    ) -> AResult<(String, String)> {
        let owners = self.physical_owner_coordinates(&region.buffer, indices)?;
        let mut axes: Vec<String> = owners.iter().map(|(axis, _)| axis.clone()).collect();
        axes.sort();
        let supported =
            axes == ["laneid"] || axes == ["tid_in_wg"] || axes == ["laneid", "wid_in_wg"];
        if !supported {
            return unsupported(format!(
                "Canonical owner transport requires exactly one physical owner thread; got owner axes {} for {}",
                py_str_sorted_list(&axes),
                buffer_name(&region.buffer)
            ));
        }
        let mut values: Vec<(String, RustValue)> = Vec::new();
        for (axis, expression) in owners {
            let value = self.emit_expr(&oref(expression))?;
            let value = self.as_i64(value)?;
            if value.uniformity != Uniformity::Uniform {
                return unsupported(
                    "Canonical owner transport requires lane-uniform logical region coordinates",
                );
            }
            values.push((axis, value));
        }
        let value_of = |axis: &str| -> String {
            values
                .iter()
                .find(|(name, _)| name == axis)
                .map(|(_, value)| value.code.clone())
                .expect("owner coordinate")
        };
        if axes == ["laneid"] {
            if op.exec_scope != "warp" {
                return unsupported(
                    "A laneid-only layout is replicated across warps and cannot be remapped as a unique warpgroup owner",
                );
            }
            let lane = self.control_name("tile_owner_lane");
            self.emit_line(&format!(
                "let {lane} = usize::try_from({}).map_err(|_| EngineError::message(\"negative tile owner lane\"))?;",
                value_of("laneid")
            ));
            self.emit_line(&format!("if {lane} >= WARP_SIZE {{"));
            self.indent += 1;
            self.emit_line(
                "return Err(EngineError::message(\"tile owner lane is outside the warp\"));",
            );
            self.indent -= 1;
            self.emit_line("}");
            return Ok(("ctx.warp_id_in_cta()".to_owned(), lane));
        }
        if op.exec_scope != "warpgroup" {
            return unsupported(
                "Cross-warp canonical owner transport currently requires warpgroup exec scope",
            );
        }
        let warps = self.warps_per_warpgroup;
        let warpgroup_base = format!("(ctx.warp_id_in_cta() / {warps}_usize) * {warps}_usize");
        if axes == ["tid_in_wg"] {
            let thread = self.control_name("tile_owner_thread");
            self.emit_line(&format!(
                "let {thread} = usize::try_from({}).map_err(|_| EngineError::message(\"negative tile owner thread\"))?;",
                value_of("tid_in_wg")
            ));
            self.emit_line(&format!("if {thread} >= {}_usize {{", warps * 32));
            self.indent += 1;
            self.emit_line(
                "return Err(EngineError::message(\"tile owner thread is outside the warpgroup\"));",
            );
            self.indent -= 1;
            self.emit_line("}");
            return Ok((
                format!("{warpgroup_base} + {thread} / WARP_SIZE"),
                format!("{thread} % WARP_SIZE"),
            ));
        }
        let owner_warp = self.control_name("tile_owner_warp");
        let owner_lane = self.control_name("tile_owner_lane");
        self.emit_line(&format!(
            "let {owner_warp} = usize::try_from({}).map_err(|_| EngineError::message(\"negative tile owner warp\"))?;",
            value_of("wid_in_wg")
        ));
        self.emit_line(&format!(
            "let {owner_lane} = usize::try_from({}).map_err(|_| EngineError::message(\"negative tile owner lane\"))?;",
            value_of("laneid")
        ));
        self.emit_line(&format!(
            "if {owner_warp} >= {warps}_usize || {owner_lane} >= WARP_SIZE {{"
        ));
        self.indent += 1;
        self.emit_line(
            "return Err(EngineError::message(\"tile owner coordinate is outside the warpgroup\"));",
        );
        self.indent -= 1;
        self.emit_line("}");
        Ok((format!("{warpgroup_base} + {owner_warp}"), owner_lane))
    }

    fn tile_load_unique_owner_region(
        &mut self,
        op: &ParsedTileCall,
        region: &TileRegion,
        linear: &PrimExpr,
        required_mask: &str,
    ) -> AResult<RustValue> {
        let indices = self.tile_broadcast_region_indices(&op.destination, region, linear)?;
        let (target_warp, target_lane) = self.tile_unique_owner_location(op, region, &indices)?;
        let physical_index =
            self.physical_index_at_lane(&region.buffer, &indices, "0_usize", None, false)?;
        let Some(rust_type) = rust_scalar_by_dtype(self.ctx.schema, &region.dtype) else {
            return unsupported(format!(
                "Canonical owner transport is not implemented for {}",
                region.dtype
            ));
        };
        let (storage_type, decoder) = match region.dtype.as_str() {
            "float16" => ("u16", Some("fp16_bits_to_f32")),
            "bfloat16" => ("u16", Some("bf16_bits_to_f32")),
            "float8_e4m3fn" => ("u8", Some("float8_e4m3fn_bits_to_f32")),
            "float8_e8m0fnu" => ("u8", Some("float8_e8m0fnu_bits_to_f32")),
            _ => (rust_type, None),
        };
        let zero = zero_literal_by_rust_type(rust_type)?;
        let scalar = self.control_name("tile_owner_value");
        let buffer_ref = self.buffer_ref(&region.buffer)?;
        self.emit_line(&format!(
            "let {scalar} = if {required_mask}.is_empty() {{ {zero} }} else {{"
        ));
        self.indent += 1;
        let loaded = self.control_name("tile_owner_value");
        let mut load = format!(
            "read_frontend_register_element_at_thread::<{storage_type}>(&physical, v2_context(ctx), &{buffer_ref}, {target_warp}, {}, {target_lane})?",
            physical_index.code
        );
        if self.ctx.schema.high_precision && crate::tables::is_promoted_float(&region.dtype) {
            let marker = v2_memory_type_rust(self.ctx.schema, &region.dtype)?;
            load = format!(
                "read_frontend_register_element_at_thread::<<{marker} as v2::mem::MemoryType>::Storage>(&physical, v2_context(ctx), &{buffer_ref}, {target_warp}, {}, {target_lane})?.value",
                physical_index.code
            );
        } else if let Some(decoder) = decoder {
            load = format!("{decoder}({load})");
        }
        self.emit_line(&format!("let {loaded} = {load};"));
        self.emit_line(&loaded);
        self.indent -= 1;
        self.emit_line("};");
        let value = self.control_name("tile_owner_values");
        self.emit_line(&format!("let {value} = WarpValue::splat({scalar});"));
        Ok(RustValue::new(value, rust_type, Uniformity::Varying))
    }

    fn tile_snapshot_transported_operands(
        &mut self,
        op: &ParsedTileCall,
    ) -> AResult<Vec<(String, &'static str)>> {
        let mut snapshots: Vec<(String, &'static str)> = Vec::new();
        for (operand_index, operand) in op.operands.iter().enumerate() {
            let Some(rust_type) = rust_scalar_by_dtype(self.ctx.schema, operand.dtype()) else {
                return unsupported(format!(
                    "Canonical owner transport is not implemented for {}",
                    operand.dtype()
                ));
            };
            let name = self.control_name(&format!("tile_operand_{operand_index}_snapshot"));
            self.emit_line(&format!(
                "let mut {name}: Vec<WarpValue<{rust_type}>> = Vec::with_capacity({}_usize);",
                op.destination.element_count()
            ));
            snapshots.push((name, rust_type));
        }
        let snapshot_names = snapshots.clone();
        self.tile_linear_loop(
            "tile_transport_element",
            op.destination.element_count(),
            |emitter, linear| {
                let destination_indices = emitter.tile_region_indices(&op.destination, linear)?;
                let destination_mask = emitter.physical_access_mask(
                    &op.destination.buffer,
                    &destination_indices,
                    "ctx.active_mask()",
                )?;
                for (operand, (snapshot, rust_type)) in
                    op.operands.iter().zip(snapshot_names.iter())
                {
                    let value = match operand {
                        TileOperand::Region(region)
                            if !emitter.tile_distributed_owner_axes(region)?.is_empty() =>
                        {
                            emitter.tile_load_unique_owner_region(
                                op,
                                region,
                                linear,
                                &destination_mask,
                            )?
                        }
                        TileOperand::Region(region) => {
                            let operand_indices = emitter.tile_broadcast_region_indices(
                                &op.destination,
                                region,
                                linear,
                            )?;
                            let loaded = emitter.emit_buffer_load(
                                &region.buffer,
                                &operand_indices,
                                Some(destination_mask.clone()),
                                None,
                                false,
                                None,
                                None,
                                None,
                            )?;
                            emitter.as_warp_value(loaded)
                        }
                        TileOperand::Scalar(scalar) => {
                            let Some(node) = scalar_node(scalar)? else {
                                return Err(Failure::Ffi(ffi_error(
                                    "tile scalar operand is not an IR node",
                                )));
                            };
                            let loaded = emitter.emit_expr(&node)?;
                            emitter.as_warp_value(loaded)
                        }
                    };
                    if value.rust_type != *rust_type {
                        return unsupported(
                            "Canonical owner transport operand changed Rust scalar type",
                        );
                    }
                    emitter.emit_line(&format!("{snapshot}.push({});", value.code));
                }
                Ok(())
            },
        )?;
        Ok(snapshots)
    }

    fn tile_emit_owner_transported_copy_or_cast(
        &mut self,
        op: &ParsedTileCall,
        cast: bool,
        source_op_id: i64,
    ) -> AResult<()> {
        self.tile_emit_scope_sync(op, true)?;
        let snapshots = self.tile_snapshot_transported_operands(op)?;
        self.tile_emit_scope_sync(op, true)?;
        let (snapshot, rust_type) = snapshots[0].clone();
        self.tile_linear_loop(
            "tile_transport_restore",
            op.destination.element_count(),
            |emitter, linear| {
                let destination_indices = emitter.tile_region_indices(&op.destination, linear)?;
                let destination_mask = emitter.physical_access_mask(
                    &op.destination.buffer,
                    &destination_indices,
                    "ctx.active_mask()",
                )?;
                let mut value = RustValue::new(
                    format!("{snapshot}[{} as usize].clone()", var_name(linear)?),
                    rust_type,
                    Uniformity::Varying,
                );
                if cast {
                    value =
                        emitter.coerce_dtype(value, &op.destination.dtype, "tile_owner_cast")?;
                }
                emitter.tile_store_region_at_indices(
                    &op.destination,
                    &destination_indices,
                    value,
                    Some(&destination_mask),
                    false,
                    source_op_id,
                )
            },
        )?;
        self.tile_emit_scope_sync(op, true)
    }

    fn tile_emit_owner_transported_elementwise(
        &mut self,
        op: &ParsedTileCall,
        result: ResultEmitter,
        source_op_id: i64,
    ) -> AResult<()> {
        let rounding_mode = op.attr_str("rounding_mode").unwrap_or("rn").to_owned();
        let ftz = op.attr_bool("ftz");
        self.tile_emit_scope_sync(op, true)?;
        let snapshots = self.tile_snapshot_transported_operands(op)?;
        self.tile_emit_scope_sync(op, true)?;
        self.tile_linear_loop(
            "tile_transport_restore",
            op.destination.element_count(),
            |emitter, linear| {
                let linear_name = var_name(linear)?;
                let destination_indices = emitter.tile_region_indices(&op.destination, linear)?;
                let destination_mask = emitter.physical_access_mask(
                    &op.destination.buffer,
                    &destination_indices,
                    "ctx.active_mask()",
                )?;
                let operands: Vec<RustValue> = snapshots
                    .iter()
                    .map(|(snapshot, rust_type)| {
                        RustValue::new(
                            format!("{snapshot}[{} as usize].clone()", linear_name),
                            *rust_type,
                            Uniformity::Varying,
                        )
                    })
                    .collect();
                let value = emitter.tile_emit_result(
                    op,
                    result,
                    &operands,
                    &rounding_mode,
                    ftz,
                    source_op_id,
                )?;
                emitter.tile_store_region_at_indices(
                    &op.destination,
                    &destination_indices,
                    value,
                    Some(&destination_mask),
                    false,
                    source_op_id,
                )
            },
        )?;
        self.tile_emit_elementwise_completion(op)
    }

    fn tile_emit_whole_tile_copy(
        &mut self,
        op: &ParsedTileCall,
        source: &TileRegion,
        zero_fill_invalid: bool,
        source_op_id: i64,
    ) -> AResult<()> {
        if source.dtype != op.destination.dtype {
            return unsupported(
                "Tx.copy requires equal source/destination dtypes; use Tx.cast for conversion",
            );
        }
        let snapshot = self.scope_snapshot();
        let result = (|| -> AResult<()> {
            let source = self.tile_snapshot_copy_region_mins(source, "source", source_op_id)?;
            let destination =
                self.tile_snapshot_copy_region_mins(&op.destination, "destination", source_op_id)?;
            let (destination_view, source_view, shape, element, source_space, destination_space) =
                self.build_mapped_copy_views(&source, &destination, source_op_id)?;
            let Some(scope) = v2_tile_scope(&op.exec_scope) else {
                return unsupported(format!(
                    "Tx.copy has no v2 scope specialization for {:?}",
                    &op.exec_scope
                ));
            };
            let register_copy =
                (destination.memory_scope == "local") != (source.memory_scope == "local");
            let sync = if register_copy {
                "v2::tile::variant::NoSnapshotSync"
            } else {
                "v2::tile::variant::SnapshotSync"
            };
            let fill = if zero_fill_invalid {
                "v2::tile::variant::ZeroFill"
            } else {
                "v2::tile::variant::NoFill"
            };
            let variant = format!(
                "v2::tile::variant::Copy<{shape}, {element}, {source_space}, {destination_space}, {scope}, {sync}, {fill}>"
            );
            let line = format!(
                "{};",
                self.tile_stateful(
                    "tile::copy",
                    Some(source_op_id),
                    &[format!("&{destination_view}"), format!("&{source_view}")],
                    Some(&variant),
                    None,
                    true,
                )
            );
            self.emit_suspend_line(&line);
            Ok(())
        })();
        self.restore_scope(snapshot);
        result
    }

    /// Dynamic tile origins are evaluated once under the tile operation mask.
    fn tile_snapshot_copy_region_mins(
        &mut self,
        region: &TileRegion,
        role: &str,
        source_op_id: i64,
    ) -> AResult<TileRegion> {
        let mut snapshots = Vec::new();
        for (axis, minimum) in region.mins.iter().enumerate() {
            let dtype = dtype_of(&oref(minimum.clone()))?;
            if !is_integer_dtype(&dtype) {
                return unsupported(format!(
                    "Tx.copy {role} minimum {axis} must be integer, got {:?}",
                    &dtype
                ));
            }
            if kind(&oref(minimum.clone())) == Some("IntImm") {
                snapshots.push(minimum.clone());
                continue;
            }
            let site = if self.analysis_capable {
                Some(source_op_id)
            } else {
                None
            };
            let value = self.with_load_site(Some(NestedLoadSite::Site(site)), |emitter| {
                emitter.emit_expr(&oref(minimum.clone()))
            })?;
            let name = self.control_name(&format!("tile_copy_{role}_min_{axis}"));
            self.emit_line(&format!("let {name} = ({}).clone();", value.code));
            let (node, expr) = tir_var(&name, &dtype)?;
            self.variables.set(
                node,
                RustValue::new(name, value.rust_type.clone(), value.uniformity),
            );
            snapshots.push(expr);
        }
        Ok(TileRegion {
            mins: snapshots,
            ..region.clone()
        })
    }

    fn tile_emit_copy_or_cast(
        &mut self,
        op: &ParsedTileCall,
        cast: bool,
        owner_driven: bool,
        register_copy: bool,
        source_op_id: i64,
    ) -> AResult<()> {
        let Some(source) = op.operands.first().and_then(TileOperand::region) else {
            return Err(Failure::Ffi(ffi_error("tile copy source is not a region")));
        };
        let source = source.clone();
        let zero_fill_invalid = op.attr_bool("zero_fill_invalid_source");
        let regions = [op.destination.clone(), source.clone()];
        if self.tile_needs_distributed_owner_transport(&regions)? {
            return self.tile_emit_owner_transported_copy_or_cast(op, cast, source_op_id);
        }
        let register_copy_owner_extent =
            self.tile_owner_driven_register_copy_extent(op, &source, register_copy)?;
        if let Some(extent) = register_copy_owner_extent {
            return self.tile_owner_slot_loop(extent, |emitter, logical_coordinates| {
                let value = emitter.tile_load_owner_aligned_region(
                    &source,
                    logical_coordinates,
                    source_op_id,
                )?;
                let destination_indices =
                    tile_logical_region_indices(&op.destination, logical_coordinates)?;
                emitter.tile_store_region_at_indices(
                    &op.destination,
                    &destination_indices,
                    value,
                    Some("ctx.active_mask()"),
                    true,
                    source_op_id,
                )
            });
        }
        let owner_extent = self.tile_owner_driven_extent(op, owner_driven)?;
        if let Some(extent) = owner_extent {
            return self.tile_owner_slot_loop(extent, |emitter, logical_coordinates| {
                let value = emitter.tile_load_owner_aligned_region(
                    &source,
                    logical_coordinates,
                    source_op_id,
                )?;
                let destination_indices =
                    tile_logical_region_indices(&op.destination, logical_coordinates)?;
                emitter.tile_store_region_at_indices(
                    &op.destination,
                    &destination_indices,
                    value,
                    Some("ctx.active_mask()"),
                    true,
                    source_op_id,
                )
            });
        }
        if !cast {
            return self.tile_emit_whole_tile_copy(op, &source, zero_fill_invalid, source_op_id);
        }
        self.tile_linear_loop(
            "tile_element",
            op.destination.element_count(),
            |emitter, linear| {
                let owner_mask = emitter.tile_semantic_element_owner_mask(op, &regions, linear)?;
                let access_mask =
                    emitter.tile_elementwise_operation_mask(op, &regions, linear, &owner_mask)?;
                let source_indices =
                    emitter.tile_broadcast_region_indices(&op.destination, &source, linear)?;
                let value = emitter.emit_buffer_load(
                    &source.buffer,
                    &source_indices,
                    Some(access_mask.clone()),
                    None,
                    zero_fill_invalid,
                    None,
                    None,
                    None,
                )?;
                let value = emitter.coerce_dtype(value, &op.destination.dtype, "tile_cast")?;
                emitter.tile_store_region(
                    &op.destination,
                    linear,
                    value,
                    Some(&access_mask),
                    source_op_id,
                )
            },
        )?;
        self.tile_emit_elementwise_completion(op)
    }
}

// ----------------------------------------------------------------------
// Elementwise results and lowering.
// ----------------------------------------------------------------------

/// `v2::reg::variant::F32Arithmetic<round, subnormal, NoSat>`.
fn f32_arithmetic_variant(rounding_mode: &str, ftz: bool) -> AResult<String> {
    let Some(round) = round_marker(rounding_mode) else {
        return unsupported(format!("Unknown normalized rounding mode {rounding_mode}"));
    };
    let subnormal = if ftz { "Ftz" } else { "PreserveSubnormal" };
    Ok(format!(
        "v2::reg::variant::F32Arithmetic<v2::reg::variant::{round}, v2::reg::variant::{subnormal}, v2::reg::variant::NoSat>"
    ))
}

fn all_rust_type(operands: &[RustValue], rust_type: &str) -> bool {
    operands
        .iter()
        .all(|operand| operand.rust_type == rust_type)
}

impl<'a> Emitter<'a> {
    /// One exact scalar ABI call for one logical tile element.
    fn tile_emit_elementwise_register_call(
        &mut self,
        op: &ParsedTileCall,
        instruction: &str,
        variant: &str,
        operands: &[RustValue],
        result_type: &str,
        source_op_id: i64,
    ) -> AResult<RustValue> {
        let arguments: Vec<String> = operands
            .iter()
            .map(|operand| abi::register(&operand.code))
            .collect();
        let abi_args = if arguments.len() == 1 {
            arguments[0].clone()
        } else {
            format!("({})", arguments.join(", "))
        };
        let result = self.control_name(&format!("tile_{}", op.kind.value()));
        let site = self.v2_site(Some(source_op_id));
        let call = abi::call(
            &format!("reg::{instruction}"),
            &["v2_context(ctx)".to_owned(), site, abi_args],
            &[variant.to_owned()],
            true,
            false,
        );
        self.emit_line(&format!("let {result} = v2_register_out({call});"));
        Ok(RustValue::new(result, result_type, Uniformity::Varying))
    }

    fn tile_elementwise_type(&self, op: &ParsedTileCall) -> AResult<&'static str> {
        match rust_scalar_by_dtype(self.ctx.schema, &op.destination.dtype) {
            Some(result_type) => Ok(result_type),
            None => unsupported(format!(
                "Native {} tile lowering does not support {}",
                op.kind.value(),
                op.destination.dtype
            )),
        }
    }

    fn tile_emit_elementwise_atom(
        &mut self,
        op: &ParsedTileCall,
        result_type: &str,
        atom: &str,
    ) -> AResult<RustValue> {
        let result = self.control_name(&format!("tile_{}", op.kind.value()));
        self.emit_line(&format!(
            "let {result} = WarpValue::from_fn(|lane| {atom});"
        ));
        Ok(RustValue::new(result, result_type, Uniformity::Varying))
    }

    /// The bound `ResultEmitter`.
    fn tile_emit_result(
        &mut self,
        op: &ParsedTileCall,
        result: ResultEmitter,
        operands: &[RustValue],
        rounding_mode: &str,
        ftz: bool,
        source_op_id: i64,
    ) -> AResult<RustValue> {
        match result {
            ResultEmitter::Zero => {
                if !operands.is_empty() {
                    return unsupported("Normalized tile.zero unexpectedly retained operands");
                }
                let result_type = self.tile_elementwise_type(op)?;
                let atom = zero_literal_by_rust_type(result_type)?;
                if result_type == "bool" {
                    return self.tile_emit_elementwise_atom(op, result_type, atom);
                }
                let marker = upper_first(result_type);
                let zero = RustValue::new(
                    format!("WarpValue::splat({atom})"),
                    result_type,
                    Uniformity::Uniform,
                );
                self.tile_emit_elementwise_register_call(
                    op,
                    "mov",
                    &format!("v2::reg::variant::{marker}"),
                    &[zero],
                    result_type,
                    source_op_id,
                )
            }
            ResultEmitter::Fill => {
                let result_type = self.tile_elementwise_type(op)?;
                if operands.len() != 1 || operands[0].rust_type != result_type {
                    return unsupported(
                        "Normalized tile.fill value does not match the destination scalar type",
                    );
                }
                if result_type == "bool" {
                    let atom = format!("{}[lane]", operands[0].code);
                    return self.tile_emit_elementwise_atom(op, result_type, &atom);
                }
                let marker = upper_first(result_type);
                self.tile_emit_elementwise_register_call(
                    op,
                    "mov",
                    &format!("v2::reg::variant::{marker}"),
                    operands,
                    result_type,
                    source_op_id,
                )
            }
            ResultEmitter::Arithmetic {
                instruction,
                narrow_method,
            } => {
                let result_type = self.tile_elementwise_type(op)?;
                if all_rust_type(operands, "f32") {
                    let variant = f32_arithmetic_variant(rounding_mode, ftz)?;
                    return self.tile_emit_elementwise_register_call(
                        op,
                        instruction,
                        &variant,
                        operands,
                        "f32",
                        source_op_id,
                    );
                }
                if all_rust_type(operands, "f64") {
                    return self.tile_emit_elementwise_register_call(
                        op,
                        instruction,
                        "v2::reg::variant::F64Rn",
                        operands,
                        "f64",
                        source_op_id,
                    );
                }
                if matches!(result_type, "i16" | "i32" | "i64" | "u16" | "u32" | "u64")
                    && all_rust_type(operands, result_type)
                {
                    let marker = upper_first(result_type);
                    return self.tile_emit_elementwise_register_call(
                        op,
                        instruction,
                        &format!("v2::reg::variant::{marker}"),
                        operands,
                        result_type,
                        source_op_id,
                    );
                }
                if all_rust_type(operands, result_type) {
                    let atom = format!(
                        "{}[lane].{narrow_method}({}[lane])",
                        operands[0].code, operands[1].code
                    );
                    return self.tile_emit_elementwise_atom(op, result_type, &atom);
                }
                unsupported(format!(
                    "Native {} tile lowering received incompatible scalar types",
                    op.kind.value()
                ))
            }
            ResultEmitter::FloatBinary {
                instruction,
                f32_variant,
                f64_variant,
                error_message,
            } => {
                if all_rust_type(operands, "f32") {
                    return self.tile_emit_elementwise_register_call(
                        op,
                        instruction,
                        f32_variant,
                        operands,
                        "f32",
                        source_op_id,
                    );
                }
                if all_rust_type(operands, "f64") {
                    return self.tile_emit_elementwise_register_call(
                        op,
                        instruction,
                        f64_variant,
                        operands,
                        "f64",
                        source_op_id,
                    );
                }
                unsupported(error_message)
            }
            ResultEmitter::Fma => {
                if all_rust_type(operands, "f32") {
                    let variant = f32_arithmetic_variant(rounding_mode, ftz)?;
                    return self.tile_emit_elementwise_register_call(
                        op,
                        "fma",
                        &variant,
                        operands,
                        "f32",
                        source_op_id,
                    );
                }
                if all_rust_type(operands, "f64") {
                    return self.tile_emit_elementwise_register_call(
                        op,
                        "fma",
                        "v2::reg::variant::F64Rn",
                        operands,
                        "f64",
                        source_op_id,
                    );
                }
                unsupported("Native fma tile lowering requires uniform f32 or f64 values")
            }
            ResultEmitter::Transcendental {
                rust_method,
                direct_instruction,
            } => {
                let uniform = operands
                    .first()
                    .is_some_and(|first| all_rust_type(operands, &first.rust_type));
                if !uniform {
                    return unsupported(format!(
                        "Native {} tile lowering requires one uniform float type",
                        op.kind.value()
                    ));
                }
                let result_type = operands[0].rust_type.clone();
                if result_type != "f32" && result_type != "f64" {
                    return unsupported(format!(
                        "Native {} tile lowering requires f32 or f64 values",
                        op.kind.value()
                    ));
                }
                let has_scale = op.attr_bool("has_scale");
                let has_bias = op.attr_bool("has_bias");
                if let Some(direct) = direct_instruction {
                    if !has_scale && !has_bias {
                        let suffix = if direct == "sqrt" { "Rn" } else { "" };
                        let marker = format!("{}{suffix}", upper_first(&result_type));
                        return self.tile_emit_elementwise_register_call(
                            op,
                            direct,
                            &format!("v2::reg::variant::{marker}"),
                            operands,
                            &result_type,
                            source_op_id,
                        );
                    }
                }
                let lane_values: Vec<String> = operands
                    .iter()
                    .map(|operand| format!("{}[lane]", operand.code))
                    .collect();
                let mut value = lane_values[0].clone();
                let mut next_operand = 1;
                if has_scale {
                    value = format!("(({value}) * ({}))", lane_values[next_operand]);
                    next_operand += 1;
                }
                if has_bias {
                    value = format!("(({value}) + ({}))", lane_values[next_operand]);
                    next_operand += 1;
                }
                if next_operand != lane_values.len() {
                    return unsupported(format!(
                        "Normalized tile.{} operand contract is inconsistent",
                        op.kind.value()
                    ));
                }
                let atom = format!("({value}).{rust_method}()");
                self.tile_emit_elementwise_atom(op, &result_type, &atom)
            }
            ResultEmitter::Reciprocal => {
                for (result_type, variant) in [
                    ("f32", "v2::reg::variant::F32Rn"),
                    ("f64", "v2::reg::variant::F64Rn"),
                ] {
                    if all_rust_type(operands, result_type) {
                        let one = RustValue::new(
                            format!("WarpValue::splat(1.0_{result_type})"),
                            result_type,
                            Uniformity::Uniform,
                        );
                        return self.tile_emit_elementwise_register_call(
                            op,
                            "div",
                            variant,
                            &[one, operands[0].clone()],
                            result_type,
                            source_op_id,
                        );
                    }
                }
                unsupported("Native reciprocal tile lowering requires uniform f32 or f64 values")
            }
            ResultEmitter::Silu => {
                let uniform = operands
                    .first()
                    .is_some_and(|first| all_rust_type(operands, &first.rust_type));
                if !uniform {
                    return unsupported(
                        "Native silu tile lowering requires one uniform float type",
                    );
                }
                let result_type = operands[0].rust_type.clone();
                if result_type != "f32" && result_type != "f64" {
                    return unsupported("Native silu tile lowering requires f32 or f64 values");
                }
                let value = format!("{}[lane]", operands[0].code);
                let one = if result_type == "f32" {
                    "1.0_f32"
                } else {
                    "1.0_f64"
                };
                let atom = format!("({value}) / ({one} + (-({value})).exp())");
                self.tile_emit_elementwise_atom(op, &result_type, &atom)
            }
        }
    }

    fn tile_emit_elementwise_completion(&mut self, op: &ParsedTileCall) -> AResult<()> {
        if op.attr_str("storage_scope") != Some("shared") {
            return Ok(());
        }
        self.tile_emit_scope_sync(op, false)
    }

    fn tile_emit_elementwise(
        &mut self,
        op: &ParsedTileCall,
        result: ResultEmitter,
        owner_driven: bool,
        source_op_id: i64,
    ) -> AResult<()> {
        let rounding_mode = op.attr_str("rounding_mode").unwrap_or("rn").to_owned();
        let mut regions: Vec<TileRegion> = vec![op.destination.clone()];
        regions.extend(op.operands.iter().filter_map(TileOperand::region).cloned());
        if self.tile_needs_distributed_owner_transport(&regions)? {
            return self.tile_emit_owner_transported_elementwise(op, result, source_op_id);
        }
        let ftz = op.attr_bool("ftz");
        let numeric_path = op.attr_str("numeric_path").unwrap_or("scalar_f32");
        if (numeric_path == "packed_f32x2") != ftz {
            return unsupported(format!(
                "Internal elementwise numerical contract is inconsistent: path={numeric_path}, ftz={}",
                if ftz { "True" } else { "False" }
            ));
        }
        let owner_extent = self.tile_owner_driven_extent(op, owner_driven)?;
        if let Some(extent) = owner_extent {
            return self.tile_owner_slot_loop(extent, |emitter, logical_coordinates| {
                let mut operands = Vec::new();
                for operand in &op.operands {
                    let value = match operand {
                        TileOperand::Region(region) => emitter.tile_load_owner_aligned_region(
                            region,
                            logical_coordinates,
                            source_op_id,
                        )?,
                        TileOperand::Scalar(scalar) => {
                            emitter.tile_emit_explicit_expression(scalar)?
                        }
                    };
                    operands.push(emitter.as_warp_value(value));
                }
                let value = emitter.tile_emit_result(
                    op,
                    result,
                    &operands,
                    &rounding_mode,
                    ftz,
                    source_op_id,
                )?;
                let destination_indices =
                    tile_logical_region_indices(&op.destination, logical_coordinates)?;
                emitter.tile_store_region_at_indices(
                    &op.destination,
                    &destination_indices,
                    value,
                    Some("ctx.active_mask()"),
                    true,
                    source_op_id,
                )
            });
        }
        self.tile_linear_loop(
            "tile_element",
            op.destination.element_count(),
            |emitter, linear| {
                let owner_mask = emitter.tile_semantic_element_owner_mask(op, &regions, linear)?;
                let access_mask =
                    emitter.tile_elementwise_operation_mask(op, &regions, linear, &owner_mask)?;
                let mut operands = Vec::new();
                for operand in &op.operands {
                    let loaded = match operand {
                        TileOperand::Region(region) => {
                            let operand_indices = emitter.tile_broadcast_region_indices(
                                &op.destination,
                                region,
                                linear,
                            )?;
                            emitter.emit_buffer_load(
                                &region.buffer,
                                &operand_indices,
                                Some(access_mask.clone()),
                                None,
                                false,
                                None,
                                None,
                                None,
                            )?
                        }
                        TileOperand::Scalar(scalar) => {
                            let Some(node) = scalar_node(scalar)? else {
                                return Err(Failure::Ffi(ffi_error(
                                    "tile scalar operand is not an IR node",
                                )));
                            };
                            emitter.emit_expr(&node)?
                        }
                    };
                    operands.push(emitter.as_warp_value(loaded));
                }
                let value = emitter.tile_emit_result(
                    op,
                    result,
                    &operands,
                    &rounding_mode,
                    ftz,
                    source_op_id,
                )?;
                emitter.tile_store_region(
                    &op.destination,
                    linear,
                    value,
                    Some(&access_mask),
                    source_op_id,
                )
            },
        )?;
        self.tile_emit_elementwise_completion(op)
    }
}

// ----------------------------------------------------------------------
// Reductions.
// ----------------------------------------------------------------------

fn reduction_identity(schema: &Schema, builder: IdentityBuilder, dtype: &str) -> AResult<String> {
    let dtype = if schema.high_precision && crate::tables::is_promoted_float(dtype) {
        "float64"
    } else {
        dtype
    };
    let Some(rust_type) = rust_scalar_by_dtype(schema, dtype) else {
        return Err(Failure::Ffi(ffi_error(&format!("KeyError: {:?}", dtype))));
    };
    let float_maximum = |dtype: &str| -> AResult<&'static str> {
        Ok(match dtype {
            "float16" => "65504.0_f32",
            "bfloat16" => "f32::from_bits(0x7f7f_0000_u32)",
            "float32" => "f32::MAX",
            "float64" => "f64::MAX",
            other => return Err(Failure::Ffi(ffi_error(&format!("KeyError: {:?}", other)))),
        })
    };
    Ok(match builder {
        IdentityBuilder::Sum => zero_literal_by_rust_type(rust_type)?.to_owned(),
        IdentityBuilder::Max => {
            if is_integer_dtype(dtype) {
                format!("{rust_type}::MIN")
            } else {
                format!("-({})", float_maximum(dtype)?)
            }
        }
        IdentityBuilder::Min => {
            if is_integer_dtype(dtype) {
                format!("{rust_type}::MAX")
            } else {
                float_maximum(dtype)?.to_owned()
            }
        }
    })
}

fn narrow_integer_reduction_atom(
    method: &str,
    dtype: &str,
    lhs: &str,
    rhs: &str,
) -> AResult<String> {
    if dtype != "int8" && dtype != "uint8" {
        return Err(Failure::Ffi(ffi_error(
            "only widened 8-bit reductions use a native Rust atom",
        )));
    }
    Ok(format!("({lhs}).{method}({rhs})"))
}

impl<'a> Emitter<'a> {
    fn tile_reduction_source_indices(
        &mut self,
        source: &TileRegion,
        axes: &[i64],
        output_linear: &PrimExpr,
        reduction_linear: &PrimExpr,
        output_replication: i64,
    ) -> AResult<Vec<PrimExpr>> {
        let spatial_extents: Vec<i64> = source
            .extents
            .iter()
            .enumerate()
            .filter(|(axis, _)| !axes.contains(&(*axis as i64)))
            .map(|(_, extent)| *extent)
            .collect();
        let reduction_extents: Vec<i64> = axes
            .iter()
            .map(|axis| source.extents[*axis as usize])
            .collect();
        let spatial_linear;
        let spatial_ref: &PrimExpr = if output_replication != 1 {
            let name = self.control_name("tile_spatial_output");
            self.emit_line(&format!(
                "let {name}: i64 = {} / {output_replication}_i64;",
                var_name(output_linear)?
            ));
            let (node, expr) = tir_var(&name, "int64")?;
            self.variables
                .set(node, RustValue::new(name, "i64", Uniformity::Uniform));
            spatial_linear = expr;
            &spatial_linear
        } else {
            output_linear
        };
        let spatial_coordinates = self.tile_coordinates(spatial_ref, &spatial_extents)?;
        let reduction_coordinates = self.tile_coordinates(reduction_linear, &reduction_extents)?;
        let mut spatial_iter = spatial_coordinates.into_iter();
        let mut coordinates = Vec::new();
        for axis in 0..source.extents.len() {
            match axes.iter().position(|candidate| *candidate == axis as i64) {
                Some(position) => coordinates.push(reduction_coordinates[position].clone()),
                None => coordinates.push(spatial_iter.next().expect("spatial coordinate")),
            }
        }
        self.tile_region_indices_from_coordinates(source, &coordinates)
    }

    /// The exact register instruction
    /// of one reduction step, or `None` for the widened 8-bit expansion.
    #[allow(clippy::too_many_arguments)]
    fn tile_emit_reduction_combine(
        &mut self,
        op: &ParsedTileCall,
        dtype: &str,
        lhs: RustValue,
        rhs: RustValue,
        instruction: &str,
        sum_reduction: bool,
        source_op_id: i64,
    ) -> AResult<Option<RustValue>> {
        if dtype == "int8" || dtype == "uint8" {
            return Ok(None);
        }
        let dtype = if self.ctx.schema.high_precision && crate::tables::is_promoted_float(dtype) {
            "float64"
        } else {
            dtype
        };
        let Some(rust_type) = rust_scalar_by_dtype(self.ctx.schema, dtype) else {
            return Err(Failure::Ffi(ffi_error(&format!("KeyError: {:?}", dtype))));
        };
        if dtype == "float16" || dtype == "bfloat16" {
            let low = if dtype == "float16" { "F16" } else { "Bf16" };
            let mut encoded = Vec::new();
            for operand in [lhs, rhs] {
                encoded.push(self.tile_emit_elementwise_register_call(
                    op,
                    "cvt",
                    &format!(
                        "v2::reg::variant::Cvt<v2::reg::variant::F32, v2::reg::variant::{low}, v2::reg::variant::Rn>"
                    ),
                    &[operand],
                    "u16",
                    source_op_id,
                )?);
            }
            let arithmetic_marker = if sum_reduction {
                format!("{low}Rn")
            } else {
                low.to_owned()
            };
            let reduced = self.tile_emit_elementwise_register_call(
                op,
                instruction,
                &format!("v2::reg::variant::{arithmetic_marker}"),
                &encoded,
                "u16",
                source_op_id,
            )?;
            return Ok(Some(self.tile_emit_elementwise_register_call(
                op,
                "cvt",
                &format!("v2::reg::variant::Cvt<v2::reg::variant::{low}, v2::reg::variant::F32>"),
                &[reduced],
                "f32",
                source_op_id,
            )?));
        }
        let marker = if dtype == "float32" {
            if sum_reduction { "F32Rn" } else { "F32" }.to_owned()
        } else if dtype == "float64" {
            if sum_reduction { "F64Rn" } else { "F64" }.to_owned()
        } else {
            upper_first(rust_type)
        };
        Ok(Some(self.tile_emit_elementwise_register_call(
            op,
            instruction,
            &format!("v2::reg::variant::{marker}"),
            &[lhs, rhs],
            rust_type,
            source_op_id,
        )?))
    }

    fn tile_emit_reduction_completion(&mut self, op: &ParsedTileCall) -> AResult<()> {
        if op.attr_str("storage_scope") != Some("shared") {
            return Ok(());
        }
        self.tile_emit_scope_sync(op, false)
    }

    #[allow(clippy::too_many_arguments)]
    fn tile_emit_local_collective_reduction(
        &mut self,
        op: &ParsedTileCall,
        source: &TileRegion,
        width: i64,
        output_replication: i64,
        identity: &str,
        lowering: &ReductionLowering,
        source_op_id: i64,
    ) -> AResult<()> {
        let reduction_count: i64 = op
            .axes
            .iter()
            .map(|axis| source.extents[*axis as usize])
            .product();
        let Some(rust_type) = rust_scalar_by_dtype(self.ctx.schema, &source.dtype) else {
            return Err(Failure::Ffi(ffi_error(&format!(
                "KeyError: {:?}",
                &source.dtype
            ))));
        };
        self.tile_linear_loop("tile_output", op.destination.element_count(), |emitter, output_linear| {
            let destination_indices = emitter.tile_region_indices(&op.destination, output_linear)?;
            let destination_mask = emitter.physical_access_mask(
                &op.destination.buffer,
                &destination_indices,
                "ctx.active_mask()",
            )?;
            let accumulator = emitter.control_name(&format!("tile_{}_acc", op.kind.value()));
            if op.accum {
                let initial = emitter.emit_buffer_load(
                    &op.destination.buffer,
                    &destination_indices,
                    Some(destination_mask.clone()),
                    None,
                    false,
                    None,
                    None,
                    None,
                )?;
                if initial.rust_type != rust_type {
                    return unsupported(format!("Reduction accumulator must lower to {rust_type}"));
                }
                emitter.emit_line(&format!("let mut {accumulator} = {};", initial.code));
            } else {
                emitter.emit_line(&format!("let mut {accumulator} = WarpValue::splat({identity});"));
            }
            emitter.tile_linear_loop("tile_reduction", reduction_count, |emitter, reduction_linear| {
                let indices = emitter.tile_reduction_source_indices(
                    source,
                    &op.axes,
                    output_linear,
                    reduction_linear,
                    output_replication,
                )?;
                let owners = emitter.physical_owner_coordinates(&source.buffer, &indices)?;
                let owner_axes: Vec<&str> = owners.iter().map(|(axis, _)| axis.as_str()).collect();
                if owner_axes != ["laneid"] {
                    return unsupported(
                        "Local collective reduction source must have one laneid owner",
                    );
                }
                let source_lane_value = emitter.emit_expr(&oref(owners[0].1.clone()))?;
                let source_lane_value = emitter.as_i64(source_lane_value)?;
                if source_lane_value.uniformity != Uniformity::Uniform {
                    return unsupported(
                        "Local collective reduction source owner must be lane-uniform",
                    );
                }
                let source_lane = emitter.control_name("tile_reduction_source_lane");
                emitter.emit_line(&format!(
                    "let {source_lane} = usize::try_from({}).map_err(|_| EngineError::message(\"negative reduction source lane\"))?;",
                    source_lane_value.code
                ));
                emitter.emit_line(&format!("if {source_lane} >= WARP_SIZE {{"));
                emitter.indent += 1;
                emitter.emit_line(
                    "return Err(EngineError::message(\"reduction source lane outside warp\"));",
                );
                emitter.indent -= 1;
                emitter.emit_line("}");
                let value = emitter.emit_buffer_load_at_lane(
                    &source.buffer,
                    &indices,
                    &source_lane,
                    None,
                    false,
                    None,
                    None,
                    None,
                )?;
                if value.rust_type != rust_type || value.uniformity != Uniformity::Uniform {
                    return unsupported(format!("Reduction source must lower to {rust_type}"));
                }
                let combined = emitter.tile_emit_reduction_combine(
                    op,
                    &source.dtype,
                    RustValue::new(accumulator.clone(), rust_type, Uniformity::Varying),
                    RustValue::new(
                        format!("WarpValue::splat({})", value.code),
                        rust_type,
                        Uniformity::Varying,
                    ),
                    lowering.instruction,
                    lowering.sum_reduction,
                    source_op_id,
                )?;
                emitter.emit_line(&format!("for lane in {destination_mask} {{"));
                emitter.indent += 1;
                emitter.emit_line(&format!(
                    "if lane / {width}_usize == {source_lane} / {width}_usize {{"
                ));
                emitter.indent += 1;
                match combined {
                    None => {
                        let atom = narrow_integer_reduction_atom(
                            lowering.narrow_method,
                            &source.dtype,
                            &format!("{accumulator}[lane]"),
                            &value.code,
                        )?;
                        emitter.emit_line(&format!("{accumulator}[lane] = {atom};"));
                    }
                    Some(combined) => {
                        emitter.emit_line(&format!("{accumulator}[lane] = {}[lane];", combined.code));
                    }
                }
                emitter.indent -= 1;
                emitter.emit_line("}");
                emitter.indent -= 1;
                emitter.emit_line("}");
                Ok(())
            })?;
            emitter.tile_store_region_at_indices(
                &op.destination,
                &destination_indices,
                RustValue::new(accumulator, rust_type, Uniformity::Varying),
                Some(&destination_mask),
                false,
                source_op_id,
            )
        })
    }

    fn tile_emit_reduction(
        &mut self,
        op: &ParsedTileCall,
        lowering: ReductionLowering,
        source_op_id: i64,
    ) -> AResult<()> {
        let Some(source) = op.operands.first().and_then(TileOperand::region) else {
            return Err(Failure::Ffi(ffi_error(
                "tile reduction source is not a region",
            )));
        };
        let source = source.clone();
        if !self.ctx.schema.high_precision && op.attr_str("dispatch") == Some("3input_maxmin") {
            return self.tile_emit_3input_maxmin(
                op,
                &source,
                lowering.instruction,
                lowering.packed_identity,
                source_op_id,
            );
        }
        let collective = op.attr_bool("collective");
        let collective_width = op.attr_int("collective_width").unwrap_or(32);
        if collective && ![1, 2, 4, 8, 16, 32].contains(&collective_width) {
            return unsupported(format!(
                "Invalid normalized warp collective width {collective_width}"
            ));
        }
        let output_replication = op.attr_int("output_replication").unwrap_or(1);
        if op.attr_str("storage_scope") == Some("local") && collective {
            let identity =
                reduction_identity(self.ctx.schema, lowering.identity_builder, &source.dtype)?;
            return self.tile_emit_local_collective_reduction(
                op,
                &source,
                collective_width,
                output_replication,
                &identity,
                &lowering,
                source_op_id,
            );
        }
        let reduction_count: i64 = op
            .axes
            .iter()
            .map(|axis| source.extents[*axis as usize])
            .product();
        let output_count = op.destination.element_count();
        let identity = reduction_identity(self.ctx.schema, lowering.identity_builder, &source.dtype)?;
        let Some(rust_type) = rust_scalar_by_dtype(self.ctx.schema, &source.dtype) else {
            return Err(Failure::Ffi(ffi_error(&format!(
                "KeyError: {:?}",
                &source.dtype
            ))));
        };
        self.tile_linear_loop("tile_output", output_count, |emitter, output_linear| {
            let destination_indices =
                emitter.tile_region_indices(&op.destination, output_linear)?;
            let owner_regions = [op.destination.clone(), source.clone()];
            let owner_mask =
                emitter.tile_semantic_element_owner_mask(op, &owner_regions, output_linear)?;
            let destination_mask = emitter.physical_access_mask(
                &op.destination.buffer,
                &destination_indices,
                &owner_mask,
            )?;
            let accumulator = emitter.control_name(&format!("tile_{}_acc", op.kind.value()));
            if op.accum {
                let initial = emitter.emit_buffer_load(
                    &op.destination.buffer,
                    &destination_indices,
                    Some(destination_mask.clone()),
                    None,
                    false,
                    None,
                    None,
                    None,
                )?;
                if initial.rust_type != rust_type {
                    return unsupported(format!("Reduction accumulator must lower to {rust_type}"));
                }
                emitter.emit_line(&format!("let mut {accumulator} = {};", initial.code));
            } else {
                emitter.emit_line(&format!(
                    "let mut {accumulator} = WarpValue::splat({identity});"
                ));
            }
            emitter.tile_linear_loop(
                "tile_reduction",
                reduction_count,
                |emitter, reduction_linear| {
                    let indices = emitter.tile_reduction_source_indices(
                        &source,
                        &op.axes,
                        output_linear,
                        reduction_linear,
                        output_replication,
                    )?;
                    let source_mask = emitter.physical_access_mask(
                        &source.buffer,
                        &indices,
                        &destination_mask,
                    )?;
                    let value = emitter.emit_buffer_load(
                        &source.buffer,
                        &indices,
                        Some(source_mask.clone()),
                        None,
                        false,
                        None,
                        None,
                        None,
                    )?;
                    if value.rust_type != rust_type {
                        return unsupported(format!("Reduction source must lower to {rust_type}"));
                    }
                    let update_mask = emitter.control_name("tile_reduction_update_mask");
                    emitter.emit_line(&format!(
                        "let {update_mask} = {source_mask} & {destination_mask};"
                    ));
                    let combined = emitter.tile_emit_reduction_combine(
                        op,
                        &source.dtype,
                        RustValue::new(accumulator.clone(), rust_type, Uniformity::Varying),
                        value.clone(),
                        lowering.instruction,
                        lowering.sum_reduction,
                        source_op_id,
                    )?;
                    emitter.emit_line(&format!("for lane in {update_mask} {{"));
                    emitter.indent += 1;
                    match combined {
                        None => {
                            let atom = narrow_integer_reduction_atom(
                                lowering.narrow_method,
                                &source.dtype,
                                &format!("{accumulator}[lane]"),
                                &format!("{}[lane]", value.code),
                            )?;
                            emitter.emit_line(&format!("{accumulator}[lane] = {atom};"));
                        }
                        Some(combined) => {
                            emitter.emit_line(&format!(
                                "{accumulator}[lane] = {}[lane];",
                                combined.code
                            ));
                        }
                    }
                    emitter.indent -= 1;
                    emitter.emit_line("}");
                    Ok(())
                },
            )?;
            emitter.tile_store_region_at_indices(
                &op.destination,
                &destination_indices,
                RustValue::new(accumulator, rust_type, Uniformity::Varying),
                Some(&destination_mask),
                false,
                source_op_id,
            )
        })?;
        self.tile_emit_reduction_completion(op)
    }

    fn tile_update_maxmin_bucket(
        &mut self,
        op: &ParsedTileCall,
        buckets: &str,
        bucket: usize,
        operands: &[RustValue],
        instruction: &str,
        source_op_id: i64,
    ) -> AResult<RustValue> {
        let marker = if operands.len() == 3 {
            "F32ThreeSource"
        } else {
            "F32"
        };
        let reduced = self.tile_emit_elementwise_register_call(
            op,
            instruction,
            &format!("v2::reg::variant::{marker}"),
            operands,
            "f32",
            source_op_id,
        )?;
        self.emit_line(&format!(
            "for lane in ctx.active_mask() {{ {buckets}[{bucket}][lane] = {}[lane]; }}",
            reduced.code
        ));
        Ok(reduced)
    }

    fn tile_emit_3input_maxmin(
        &mut self,
        op: &ParsedTileCall,
        source: &TileRegion,
        instruction: &str,
        identity: &str,
        source_op_id: i64,
    ) -> AResult<()> {
        if op.axes != [0] {
            return unsupported("3input_maxmin normalized with an invalid reduction");
        }
        let reduction_count = source.extents[0];
        let num_full_chunks = reduction_count / 8;
        let remainder = reduction_count % 8;
        let buckets = self.control_name(&format!("packed_{}_buckets", op.kind.value()));
        self.emit_line(&format!(
            "let mut {buckets}: [WarpValue<f32>; 4] = std::array::from_fn(|_| WarpValue::splat({identity}));"
        ));
        let bucket_value = |buckets: &str, bucket: usize| -> RustValue {
            RustValue::new(
                format!("{buckets}[{bucket}].clone()"),
                "f32",
                Uniformity::Varying,
            )
        };
        let load_element = |emitter: &mut Self, coordinate: PrimExpr| -> AResult<RustValue> {
            let indices = emitter.tile_region_indices_from_coordinates(source, &[coordinate])?;
            emitter.emit_buffer_load(
                &source.buffer,
                &indices,
                None,
                None,
                false,
                None,
                None,
                None,
            )
        };
        for index in 0..8 {
            let value = load_element(self, python_int_expr(index)?)?;
            if value.rust_type != "f32" {
                return unsupported("3input_maxmin source must lower to f32");
            }
            let bucket = (index / 2) as usize;
            if index % 2 == 0 {
                self.emit_line(&format!(
                    "for lane in ctx.active_mask() {{ {buckets}[{bucket}][lane] = {}[lane]; }}",
                    value.code
                ));
            } else {
                let mut operands = vec![bucket_value(&buckets, bucket), value];
                if op.accum && bucket == 0 {
                    let initial = self.emit_buffer_load(
                        &op.destination.buffer,
                        &op.destination.mins,
                        None,
                        None,
                        false,
                        None,
                        None,
                        None,
                    )?;
                    if initial.rust_type != "f32" {
                        return unsupported("3input_maxmin accumulator must lower to f32");
                    }
                    operands.push(initial);
                }
                self.tile_update_maxmin_bucket(
                    op,
                    &buckets,
                    bucket,
                    &operands,
                    instruction,
                    source_op_id,
                )?;
            }
        }
        self.tile_linear_loop(
            "packed_maxmin_chunk",
            num_full_chunks - 1,
            |emitter, chunk| {
                let base = op_binary(
                    "_OpMul",
                    expr_any(&op_binary("_OpAdd", expr_any(chunk), int_any(1))?),
                    int_any(8),
                )?;
                for pair in 0..4i64 {
                    let left_coordinate = op_binary("_OpAdd", expr_any(&base), int_any(2 * pair))?;
                    let right_coordinate = op_binary(
                        "_OpAdd",
                        expr_any(&op_binary("_OpAdd", expr_any(&base), int_any(2 * pair))?),
                        int_any(1),
                    )?;
                    let left = load_element(emitter, left_coordinate)?;
                    let right = load_element(emitter, right_coordinate)?;
                    if left.rust_type != "f32" || right.rust_type != "f32" {
                        return unsupported("3input_maxmin source must lower to f32");
                    }
                    let operands = vec![bucket_value(&buckets, pair as usize), left, right];
                    emitter.tile_update_maxmin_bucket(
                        op,
                        &buckets,
                        pair as usize,
                        &operands,
                        instruction,
                        source_op_id,
                    )?;
                }
                Ok(())
            },
        )?;
        let remainder_base = num_full_chunks * 8;
        for index in 0..remainder {
            let value = load_element(self, python_int_expr(remainder_base + index)?)?;
            if value.rust_type != "f32" {
                return unsupported("3input_maxmin source must lower to f32");
            }
            let operands = vec![bucket_value(&buckets, 0), value];
            self.tile_update_maxmin_bucket(op, &buckets, 0, &operands, instruction, source_op_id)?;
        }
        let operands = vec![bucket_value(&buckets, 0), bucket_value(&buckets, 1)];
        self.tile_update_maxmin_bucket(op, &buckets, 0, &operands, instruction, source_op_id)?;
        let operands = vec![
            bucket_value(&buckets, 0),
            bucket_value(&buckets, 2),
            bucket_value(&buckets, 3),
        ];
        let final_value =
            self.tile_update_maxmin_bucket(op, &buckets, 0, &operands, instruction, source_op_id)?;
        let indices = op.destination.mins.clone();
        self.tile_store_region_at_indices(
            &op.destination,
            &indices,
            final_value,
            None,
            false,
            source_op_id,
        )
    }
}

// ----------------------------------------------------------------------
// Synchronous warp GEMM and the lowering dispatcher.
// ----------------------------------------------------------------------

impl<'a> Emitter<'a> {
    fn tile_implicit_region_index_load(&self, execution_lane: &str) -> NestedLoadSite {
        // Both the numeric and the analysis-capable forms drop the source node
        // and carry no site when the operations are not instrumented.
        NestedLoadSite::AtLaneSite(execution_lane.to_owned(), None)
    }

    fn tile_warp_gemm_mapped_view(&mut self, region: &TileRegion, role: &str) -> AResult<String> {
        let shape = region.logical_shape();
        if shape.len() != 2 {
            return unsupported(format!("gemm {role} mapped operand must be rank 2"));
        }
        let info = self.ctx.inspect_layout(&region.buffer, &self.bindings)?;
        let mut axes = info.physical_axes.clone();
        axes.sort();
        axes.dedup();
        if axes != ["laneid", "m"] {
            return unsupported(format!(
                "gemm {role} requires a lane-distributed register TileLayout, got {:?}",
                &info.physical_axes
            ));
        }
        let table = self.control_name(&format!("warp_gemm_{role}_map"));
        let unowned = element_ref("v2::Register", "unowned", &["None".to_owned()]);
        self.emit_line(&format!(
            "let mut {table} = vec![{unowned}; {}_usize * 32_usize];",
            region.element_count()
        ));
        let itemsize = info.itemsize;
        self.tile_linear_loop(
            &format!("warp_gemm_{role}_element"),
            region.element_count(),
            |emitter, linear| {
                let coordinates = emitter.tile_coordinates(linear, &shape)?;
                let indices = tile_logical_region_indices(region, &coordinates)?;
                let linear_usize = emitter.control_name(&format!("warp_gemm_{role}_linear"));
                emitter.emit_line(&format!(
                    "let {linear_usize} = usize::try_from({}).map_err(|_| EngineError::message(\"negative warp GEMM logical index\"))?;",
                    var_name(linear)?
                ));
                let lane = emitter.control_name(&format!("warp_gemm_{role}_lane"));
                emitter.emit_line(&format!("for {lane} in ctx.active_mask() {{"));
                emitter.indent += 1;
                let index_load = emitter.tile_implicit_region_index_load(&lane);
                let owner = emitter.physical_access_predicate_at_lane(
                    &region.buffer,
                    &indices,
                    &lane,
                    Some(index_load.clone()),
                )?;
                let physical_index = emitter.physical_index_at_lane(
                    &region.buffer,
                    &indices,
                    &lane,
                    Some(index_load),
                    false,
                )?;
                let byte_offset = emitter.control_name(&format!("warp_gemm_{role}_byte_offset"));
                emitter.emit_line(&format!(
                    "let {byte_offset}: i128 = i128::from({}).checked_mul({itemsize}_i128).ok_or_else(|| EngineError::message(\"warp GEMM byte offset overflow\"))?;",
                    physical_index.code
                ));
                let slot = format!("{linear_usize} * 32_usize + {lane}");
                let in_bounds = element_ref(
                    "v2::Register",
                    "in_bounds",
                    &[byte_offset, "None".to_owned()],
                );
                emitter.emit_line(&format!(
                    "{table}[{slot}] = if {owner} {{ {in_bounds} }} else {{ {unowned} }};"
                ));
                emitter.indent -= 1;
                emitter.emit_line("}");
                Ok(())
            },
        )?;
        let view = self.control_name(&format!("warp_gemm_{role}_view"));
        let buffer_ref = self.buffer_ref(&region.buffer)?;
        let buffer_name = self.logical_buffer_name(&region.buffer)?;
        let line = mapped_view(
            &view,
            "v2::Register",
            &buffer_ref,
            &buffer_name,
            "NumSimRegisterMap",
            &table_state(&shape, &table),
        );
        self.emit_line(&line);
        Ok(view)
    }

    /// A whole canonical PTX fragment after layout validation.
    fn tile_fixed_warp_mma_fragment_k(
        &self,
        region: &TileRegion,
        rows: i64,
        cols: i64,
        role: &str,
    ) -> AResult<Option<i64>> {
        for minimum in &region.mins {
            if int_literal(&oref(minimum.clone())) != Some(0) {
                return Ok(None);
            }
        }
        let info = self.ctx.inspect_layout(&region.buffer, &self.bindings)?;
        let (mma_k, expected_elements) = if role == "a" && rows == 16 && (cols == 8 || cols == 16) {
            (cols, cols / 2)
        } else if role == "b" && rows == 8 && (cols == 8 || cols == 16) {
            (cols, cols / 4)
        } else if (role == "c" || role == "d") && (rows, cols) == (16, 8) {
            (16, 4)
        } else {
            return Ok(None);
        };
        Ok(if info.element_count == Some(expected_elements) {
            Some(mma_k)
        } else {
            None
        })
    }

    /// Only the allocation for a
    /// compile-time-proven fragment map.
    fn tile_warp_gemm_canonical_view(
        &mut self,
        region: &TileRegion,
        role: &str,
    ) -> AResult<String> {
        let shape = region.logical_shape();
        if shape.len() != 2 {
            return unsupported(format!("gemm {role} canonical operand must be rank 2"));
        }
        let view = self.control_name(&format!("warp_gemm_{role}_view"));
        let buffer_ref = self.buffer_ref(&region.buffer)?;
        let buffer_name = self.logical_buffer_name(&region.buffer)?;
        let line = mapped_view(
            &view,
            "v2::Register",
            &buffer_ref,
            &buffer_name,
            "NumSimRegisterMap",
            &empty_table_state(&shape, "v2::Register"),
        );
        self.emit_line(&line);
        Ok(view)
    }

    fn tile_emit_gemm(&mut self, op: &ParsedTileCall, source_op_id: i64) -> AResult<()> {
        let regions: Vec<&TileRegion> =
            op.operands.iter().filter_map(TileOperand::region).collect();
        let (Some(left), Some(right), Some(accumulator)) =
            (regions.first(), regions.get(1), regions.get(2))
        else {
            return Err(Failure::Ffi(ffi_error(
                "tile gemm operands are not regions",
            )));
        };
        let (left, right, accumulator) =
            ((*left).clone(), (*right).clone(), (*accumulator).clone());
        let attr_int = |name: &str| -> AResult<i64> {
            op.attr_int(name)
                .ok_or_else(|| Failure::Ffi(ffi_error(&format!("KeyError: {:?}", name))))
        };
        let m = attr_int("m")?;
        let n = attr_int("n")?;
        let k = attr_int("k")?;
        let mma_k = attr_int("mma_k")?;
        let beta = attr_int("beta")?;
        let trans_a = op.attr_bool("trans_a");
        let trans_b = op.attr_bool("trans_b");
        let input_marker = match left.dtype.as_str() {
            "float16" => "v2::reg::variant::F16",
            "bfloat16" => "v2::reg::variant::Bf16",
            other => return Err(Failure::Ffi(ffi_error(&format!("KeyError: {:?}", other)))),
        };
        let canonical = !self.ctx.schema.high_precision
            && m == 16
            && n == 8
            && k == mma_k
            && self.tile_fixed_warp_mma_fragment_k(&left, m, k, "a")? == Some(mma_k)
            && self.tile_fixed_warp_mma_fragment_k(&right, n, k, "b")? == Some(mma_k)
            && self
                .tile_fixed_warp_mma_fragment_k(&op.destination, m, n, "d")?
                .is_some()
            && self
                .tile_fixed_warp_mma_fragment_k(&accumulator, m, n, "c")?
                .is_some();
        let view = |emitter: &mut Self, region: &TileRegion, role: &str| -> AResult<String> {
            if canonical {
                emitter.tile_warp_gemm_canonical_view(region, role)
            } else {
                emitter.tile_warp_gemm_mapped_view(region, role)
            }
        };
        let destination_view = view(self, &op.destination, "d")?;
        let left_view = view(self, &left, "a")?;
        let right_view = view(self, &right, "b")?;
        let accumulator_view = view(self, &accumulator, "c")?;
        let mapping = if canonical {
            ", v2::tile::variant::CanonicalMmaM16N8"
        } else {
            ""
        };
        let variant = format!(
            "v2::tile::variant::MmaSync<{input_marker}, {m}, {n}, {k}, {mma_k}, {:?}, {:?}, {:?}{mapping}>",
            trans_a,
            trans_b,
            beta == 1);
        let line = format!(
            "{};",
            self.tile_stateful(
                "tile::gemm",
                Some(source_op_id),
                &[
                    format!("&{destination_view}"),
                    format!("&{left_view}"),
                    format!("&{right_view}"),
                    format!("&{accumulator_view}"),
                ],
                Some(&variant),
                None,
                false,
            )
        );
        self.emit_line(&line);
        Ok(())
    }

    /// Lower one public TIRx tile op through its dedicated entry.
    pub fn emit_tile_call(&mut self, node: &ObjectRef, source_op_id: i64) -> AResult<()> {
        let Some(call) = node.as_node::<TilePrimitiveCallObj>() else {
            return Err(Failure::Ffi(ffi_error(
                "emit_tile_call expects a TilePrimitiveCall",
            )));
        };
        let op_name = tile_op_name(call)?;
        let Some(kind) = TileOpKind::from_op_name(&op_name) else {
            return unsupported(format!("Tile lowering is not registered for {op_name}"));
        };
        let lowering = tile_lowering(kind);
        if self.ctx.schema.high_precision
            && matches!(kind, TileOpKind::CopyAsync | TileOpKind::GemmAsync)
        {
            return unsupported(format!("high precision does not model {op_name}; asynchronous copies and TCGEN instructions are unsupported"));
        }
        let plan = self.plan;
        // Tile analysis has already reported rejected source calls. The
        // diagnostic walk skips those calls and continues with later statements.
        let Some(op) = self
            .op_ids
            .get(node)
            .and_then(|op_id| plan.tile_calls[*op_id as usize].as_ref())
        else {
            if self.diagnostics.enabled && self.op_ids.get(node).is_some() {
                return Err(Failure::Recorded);
            }
            return Err(Failure::Ffi(ffi_error(
                "emit_tile_call expects a tile call the analysis resolved",
            )));
        };
        if self.ctx.schema.high_precision {
            for region in std::iter::once(&op.destination)
                .chain(op.operands.iter().filter_map(TileOperand::region))
            {
                let space = self.memory_plan.resolve(&region.buffer)?.space;
                super::high_precision::validate_memory(
                    self,
                    &region.dtype,
                    space,
                    op.attr_bool("zero_fill_invalid_source"),
                )?;
            }
        }
        self.record_global_write(&op.destination.buffer)?;
        self.with_load_site(Some(NestedLoadSite::ByBuffer(source_op_id)), |emitter| {
            emitter.tile_emit_scope_participation(&op, source_op_id)?;
            match lowering {
                TileLowering::Copy {
                    cast,
                    register_copy,
                } => emitter.tile_emit_copy_or_cast(&op, cast, true, register_copy, source_op_id),
                TileLowering::CopyAsync => {
                    if super::tile_tcgen05::is_tcgen_transfer(&op) {
                        return emitter.emit_tile_tcgen_transfer(&op, source_op_id);
                    }
                    emitter.emit_tile_async_copy(&op, source_op_id)
                }
                TileLowering::Elementwise {
                    result,
                    owner_driven,
                } => emitter.tile_emit_elementwise(&op, result, owner_driven, source_op_id),
                TileLowering::Reduction(reduction) => {
                    emitter.tile_emit_reduction(&op, reduction, source_op_id)
                }
                TileLowering::Gemm => emitter.tile_emit_gemm(&op, source_op_id),
                TileLowering::GemmAsync => emitter.emit_tile_gemm_async(&op, source_op_id),
            }
        })
    }
}
