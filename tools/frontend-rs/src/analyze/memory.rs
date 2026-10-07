//! Physical allocation/view planning.

use tvm::ir::TensorRegionObj;
use tvm::ir::{CallObj, Expr, PrimExpr, TensorLoadObj, Var};
use tvm::tirx::{
    AllocBufferObj, AttrStmtObj, BindObj, BufferStoreObj, BufferVar, DeclBufferObj, PrimFunc, Stmt,
};
use tvm::tvm_ffi::object::ObjectRef;
use tvm::tvm_ffi::{Any, Map, ObjectRefCore};

use super::buffers::BufferBindings;
use super::layout::{expr_any, int_any, op_binary, Extent, LayoutInfo};
use super::shapes::{map_from_pairs, structural_equal, Scalar, ShapeExpressionContext};
use super::util::{
    buffer_name, buffer_ref, buffer_scope, ffi_text, int_imm, oref, same, unsupported, AResult,
    IdSet,
};
use super::Ctx;
use crate::decode::projected_buffer;

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub enum MemorySpace {
    Global,
    Shared,
    Local,
    Register,
    Tmem,
}

impl MemorySpace {
    pub fn value(self) -> &'static str {
        match self {
            MemorySpace::Global => "global",
            MemorySpace::Shared => "shared",
            MemorySpace::Local => "local",
            MemorySpace::Register => "register",
            MemorySpace::Tmem => "tmem",
        }
    }
}

pub struct BackingPlan {
    pub index: usize,
    pub field: String,
    pub space: MemorySpace,
    pub byte_len: Option<i64>,
    pub byte_alignment: Option<i64>,
    pub input_name: Option<String>,
    pub tmem_lanes: Option<i64>,
    pub tmem_columns: Option<i64>,
}

pub struct BufferPlan {
    pub buffer: BufferVar,
    pub name: String,
    pub backing_index: Option<usize>,
    pub space: MemorySpace,
    pub layout: LayoutInfo,
    pub is_parameter: bool,
    pub dynamic_data_var: Option<Expr>,
    pub referenced: bool,
}

impl BufferPlan {
    /// A dynamic shape or changed declared layout identifies different view geometry.
    pub fn view_geometry_changed(&self, rhs: &Self) -> AResult<bool> {
        let left = &self.layout;
        let right = &rhs.layout;
        let left_layout = self.buffer.buffer_type().layout.clone();
        let right_layout = rhs.buffer.buffer_type().layout.clone();
        let same_layout = match (left_layout, right_layout) {
            (Some(left), Some(right)) => structural_equal(&Any::from(left), &Any::from(right))?,
            (None, None) => true,
            _ => false,
        };
        if !same_layout {
            return Ok(true);
        }
        let same_shape =
            left.shape.len() == right.shape.len()
                && left.shape.iter().zip(right.shape.iter()).all(
                    |(a, b)| matches!((a, b), (Extent::Static(x), Extent::Static(y)) if x == y),
                );
        Ok(!(same_shape
            && left.dtype == right.dtype
            && left.itemsize == right.itemsize
            && left.signature == right.signature
            && left.physical_axes == right.physical_axes
            && left.explicit_strides == right.explicit_strides
            && left.packed_nibble_offset == right.packed_nibble_offset))
    }
}

pub struct MemoryPlan {
    pub backings: Vec<BackingPlan>,
    pub buffers: Vec<BufferPlan>,
}

impl MemoryPlan {
    pub fn resolve(&self, buffer: &BufferVar) -> AResult<&BufferPlan> {
        for candidate in &self.buffers {
            if same(candidate.buffer.as_var(), buffer.as_var()) {
                return Ok(candidate);
            }
        }
        unsupported(format!(
            "buffer:{}:has no declared physical view",
            buffer_name(buffer)
        ))
    }
}

pub fn space_of(scope: &str, name: &str) -> AResult<MemorySpace> {
    match scope {
        "global" | "param" => Ok(MemorySpace::Global),
        "shared" | "shared.dyn" => Ok(MemorySpace::Shared),
        "local_scalar" => Ok(MemorySpace::Local),
        "local" | "register" | "reg" => Ok(MemorySpace::Register),
        _ if scope == "tmem" || scope.starts_with("tmem.") => Ok(MemorySpace::Tmem),
        _ => unsupported(format!(
            "buffer:{name}:unsupported memory scope {:?}",
            scope
        )),
    }
}

/// Buffers used outside their allocation/declaration statements.
pub fn referenced_buffers(
    nodes: &[ObjectRef],
    include_tensor_load_sources: bool,
) -> AResult<Vec<BufferVar>> {
    let mut referenced: Vec<BufferVar> = Vec::new();
    let mut seen = IdSet::default();
    let append = |buffer: Option<BufferVar>, seen: &mut IdSet, out: &mut Vec<BufferVar>| {
        if let Some(buffer) = buffer {
            if seen.add(buffer_ref(&buffer)) {
                out.push(buffer);
            }
        }
    };
    for node in nodes {
        if node.as_node::<AllocBufferObj>().is_some() || node.as_node::<DeclBufferObj>().is_some() {
            // Allocations and declarations record no reference here.
        } else if let Some(store) = node.as_node::<BufferStoreObj>() {
            append(Some(store.buffer.clone()), &mut seen, &mut referenced);
        } else if let Some(region) = node.as_node::<TensorRegionObj>() {
            append(
                super::util::as_buffer(&oref(region.source.clone())),
                &mut seen,
                &mut referenced,
            );
        }
        if include_tensor_load_sources {
            if let Some(load) = node.as_node::<TensorLoadObj>() {
                append(
                    super::util::as_buffer(&oref(load.source.clone())),
                    &mut seen,
                    &mut referenced,
                );
            }
        }
        if node.as_node::<CallObj>().is_some() {
            append(projected_buffer(node)?, &mut seen, &mut referenced);
        }
    }
    Ok(referenced)
}

pub fn bind_map(statements: &[Stmt]) -> Map<Var, Expr> {
    map_from_pairs(statements.iter().filter_map(|statement| {
        statement
            .as_node::<BindObj>()
            .map(|bind| (bind.var.clone(), bind.value.clone()))
    }))
}

/// `math.prod(buffer.shape)`: Python multiplies from the integer 1 upward.
pub fn shape_product(buffer: &BufferVar) -> AResult<Scalar> {
    let mut product: Any = int_any(1);
    for extent in buffer.buffer_type().shape.iter() {
        product = expr_any(&op_binary("_OpMul", product, expr_any(&extent))?);
    }
    Ok(Scalar::Expr(PrimExpr::try_from(product)?))
}

fn buffer_byte_end_expression(buffer: &BufferVar, info: &LayoutInfo) -> AResult<Scalar> {
    if info.element_count.is_some() && info.dynamic_elem_offset.is_none() {
        return Ok(Scalar::Int(info.byte_end()?));
    }
    let physical_span = match info.element_count {
        Some(count) => Scalar::Int(count),
        None => shape_product(buffer)?,
    };
    let base = match &info.dynamic_elem_offset {
        Some(offset) => Scalar::Expr(offset.clone()),
        None => Scalar::Int(info.elem_offset),
    };
    let sum = Scalar::binary("_OpAdd", &base, &physical_span, |a, b| a + b)?;
    Scalar::binary("_OpMul", &sum, &Scalar::Int(info.itemsize), |a, b| a * b)
}

struct BackingBuilder {
    key: ObjectRef,
    parameter: Option<BufferVar>,
    allocation: Option<BufferVar>,
    buffers: Vec<(BufferVar, bool)>,
}

fn append_buffer(builder: &mut BackingBuilder, buffer: &BufferVar, is_parameter: bool) {
    if builder
        .buffers
        .iter()
        .any(|(candidate, _)| same(candidate.as_var(), buffer.as_var()))
    {
        return;
    }
    builder.buffers.push((buffer.clone(), is_parameter));
}

fn find_pool_size(pool_sizes: &[(ObjectRef, i64)], key: &ObjectRef) -> Option<i64> {
    pool_sizes
        .iter()
        .find(|(candidate, _)| same(candidate, key))
        .map(|(_, size)| *size)
}

pub(in crate::analyze) fn build_memory_plan(
    ctx: &Ctx,
    func: &PrimFunc,
    statements: &[Stmt],
    nodes: &[ObjectRef],
    buffer_bindings: &BufferBindings,
) -> AResult<MemoryPlan> {
    let analyzer = &ctx.analyzer;
    let mut builders: Vec<BackingBuilder> = Vec::new();
    let mut tmem_buffers: Vec<BufferVar> = Vec::new();
    let mut dynamic_views: Vec<(BufferVar, Expr, MemorySpace)> = Vec::new();
    let referenced = referenced_buffers(nodes, false)?;
    let dynamic_referenced = referenced_buffers(nodes, true)?;
    let mut pool_sizes: Vec<(ObjectRef, i64)> = Vec::new();
    let mut dyn_smem_bytes: Option<i64> = None;
    for statement in statements {
        let Some(attr) = statement.as_node::<AttrStmtObj>() else {
            continue;
        };
        let attr_key = ffi_text(&attr.attr_key);
        if attr_key == "tirx.dyn_smem_bytes" {
            let Some(size) = int_imm(&oref(attr.value.clone())) else {
                return unsupported("tirx.dyn_smem_bytes must be a non-negative IntImm");
            };
            if size < 0 {
                return unsupported("tirx.dyn_smem_bytes cannot be negative");
            }
            if let Some(previous) = dyn_smem_bytes {
                if previous != size {
                    return unsupported(format!(
                        "tirx.dyn_smem_bytes has conflicting capacities {previous} and {size}"
                    ));
                }
            }
            dyn_smem_bytes = Some(size);
            continue;
        }
        if attr_key != "tirx.pool_max_bytes" {
            continue;
        }
        let Some(size) = int_imm(&oref(attr.value.clone())) else {
            return unsupported(
                "AttrStmt(tirx.pool_max_bytes) must bind a buffer root to a non-negative IntImm",
            );
        };
        let node = ObjectRef::try_from(attr.node.clone())
            .map_err(|_| super::util::ffi_error("pool_max_bytes node is not an object"))?;
        let pool_key = match projected_buffer(&node)? {
            Some(source) => buffer_bindings.storage_key(&source)?,
            None => node,
        };
        if size < 0 {
            return unsupported("tirx.pool_max_bytes cannot be negative");
        }
        let previous = find_pool_size(&pool_sizes, &pool_key);
        if let Some(previous) = previous {
            if previous != size {
                return unsupported(format!(
                    "tirx.pool_max_bytes has conflicting capacities {previous} and {size}"
                ));
            }
        } else {
            pool_sizes.push((pool_key, size));
        }
    }
    let bindings = bind_map(statements);
    let shape_expressions = ShapeExpressionContext::build(analyzer, statements, buffer_bindings)?;
    for parameter in crate::buffer_parameters(func.clone())?.iter() {
        let buffer = BufferVar::try_from(&parameter)?;
        let key = buffer_bindings.storage_key(&buffer)?;
        match builders.iter_mut().find(|item| same(&item.key, &key)) {
            None => builders.push(BackingBuilder {
                key,
                parameter: Some(buffer.clone()),
                allocation: None,
                buffers: vec![(buffer.clone(), true)],
            }),
            Some(builder) => {
                if builder.parameter.is_some() {
                    return unsupported(format!(
                        "buffer:{}:multiple parameters share one TIR data variable",
                        buffer_name(&buffer)
                    ));
                }
                builder.parameter = Some(buffer.clone());
                append_buffer(builder, &buffer, true);
            }
        }
    }
    for statement in statements {
        let (buffer, is_alloc, data) = if let Some(alloc) = statement.as_node::<AllocBufferObj>() {
            (alloc.buffer.clone(), true, None)
        } else if let Some(decl) = statement.as_node::<DeclBufferObj>() {
            (decl.buffer.clone(), false, Some(decl.data.clone()))
        } else {
            continue;
        };
        let scope_name = buffer_scope(&buffer);
        if scope_name == "tmem" || scope_name.starts_with("tmem.") {
            if is_alloc {
                return unsupported(format!(
                    "buffer:{}:TMEM must be declared, not allocated",
                    buffer_name(&buffer)
                ));
            }
            if !tmem_buffers
                .iter()
                .any(|candidate| same(candidate.as_var(), buffer.as_var()))
            {
                tmem_buffers.push(buffer.clone());
            }
            continue;
        }
        if let Some(data) = &data {
            if projected_buffer(&oref(data.clone()))?.is_none() {
                let info = ctx.inspect_layout(&buffer, &bindings)?;
                if info.element_count.is_none() {
                    return unsupported(format!(
                        "buffer:{}:pointer-derived view has a runtime shape",
                        buffer_name(&buffer)
                    ));
                }
                let nominal_space = space_of(&scope_name, &buffer_name(&buffer))?;
                if nominal_space == MemorySpace::Tmem {
                    return unsupported(format!(
                        "buffer:{}:generic pointers cannot derive TMEM views",
                        buffer_name(&buffer)
                    ));
                }
                dynamic_views.push((buffer.clone(), data.clone(), nominal_space));
                continue;
            }
        }
        let key = buffer_bindings.storage_key(&buffer)?;
        let position = match builders.iter().position(|item| same(&item.key, &key)) {
            Some(position) => position,
            None => {
                builders.push(BackingBuilder {
                    key,
                    parameter: None,
                    allocation: None,
                    buffers: Vec::new(),
                });
                builders.len() - 1
            }
        };
        let builder = &mut builders[position];
        append_buffer(builder, &buffer, false);
        if is_alloc {
            if let Some(allocation) = &builder.allocation {
                if !same(allocation.as_var(), buffer.as_var()) {
                    return unsupported(format!(
                        "buffer:{}:multiple AllocBuffer nodes share one data variable",
                        buffer_name(&buffer)
                    ));
                }
            }
            builder.allocation = Some(buffer.clone());
        }
    }

    let mut backing_plans: Vec<BackingPlan> = Vec::new();
    let mut buffer_plans: Vec<BufferPlan> = Vec::new();
    for builder in &builders {
        if builder.parameter.is_none() && builder.allocation.is_none() {
            let names: Vec<String> = builder
                .buffers
                .iter()
                .map(|(buffer, _)| buffer_name(buffer))
                .collect();
            return unsupported(format!(
                "buffer:{}:DeclBuffer data variable has no parameter or AllocBuffer owner",
                names.join(",")
            ));
        }
        if builder.parameter.is_some() && builder.allocation.is_some() {
            return unsupported(format!(
                "buffer:{}:data variable is both parameter and allocated",
                buffer_name(builder.allocation.as_ref().unwrap())
            ));
        }
        let mut infos: Vec<(BufferVar, bool, LayoutInfo, MemorySpace)> = Vec::new();
        for (buffer, is_parameter) in &builder.buffers {
            let info = ctx.inspect_layout(buffer, &bindings)?;
            let space = space_of(&buffer_scope(buffer), &buffer_name(buffer))?;
            infos.push((buffer.clone(), *is_parameter, info, space));
        }
        let mut spaces: Vec<MemorySpace> = infos.iter().map(|item| item.3).collect();
        spaces.sort();
        spaces.dedup();
        if spaces.len() != 1 {
            let details: Vec<String> = infos
                .iter()
                .map(|(buffer, _, _, space)| format!("{}:{}", buffer_name(buffer), space.value()))
                .collect();
            return unsupported(format!(
                "buffer:{}:views sharing one data variable cross memory spaces",
                details.join(",")
            ));
        }
        let space = spaces[0];
        let parameter = builder.parameter.as_ref();
        if let Some(parameter) = parameter {
            if space != MemorySpace::Global {
                return unsupported(format!(
                    "buffer:{}:parameter scope must be global",
                    buffer_name(parameter)
                ));
            }
        }
        if parameter.is_none() && space == MemorySpace::Global {
            return unsupported(format!(
                "buffer:{}:internal global allocation is unsupported",
                buffer_name(builder.allocation.as_ref().unwrap())
            ));
        }
        let index = backing_plans.len();
        let input_name = parameter.map(buffer_name);
        if parameter.is_none()
            && infos
                .iter()
                .any(|(_, _, info, _)| info.element_count.is_none())
        {
            return unsupported(format!(
                "buffer:{}:internal allocation has a runtime shape",
                buffer_name(builder.allocation.as_ref().unwrap())
            ));
        }
        let owner: BufferVar = match parameter {
            Some(parameter) => parameter.clone(),
            None => builder.allocation.clone().unwrap(),
        };
        let owner_info = infos
            .iter()
            .find(|(buffer, _, _, _)| same(buffer.as_var(), owner.as_var()))
            .map(|(_, _, info, _)| info.clone())
            .expect("owner layout");
        if owner_info.dynamic_elem_offset.is_some() {
            return unsupported(format!(
                "buffer:{}:allocation owner cannot have a runtime elem_offset",
                buffer_name(&owner)
            ));
        }
        let owner_start = owner_info.elem_offset * owner_info.itemsize;
        let mut pool_byte_len = find_pool_size(&pool_sizes, &builder.key);
        if pool_byte_len.is_none()
            && dyn_smem_bytes.is_some()
            && buffer_scope(&owner) == "shared.dyn"
        {
            pool_byte_len = dyn_smem_bytes;
        }
        let owner_end: Option<i64>;
        if let Some(pool_len) = pool_byte_len {
            if parameter.is_some() || space != MemorySpace::Shared {
                return unsupported(
                    "tirx.pool_max_bytes requires an internal shared-memory allocation",
                );
            }
            if owner_start != 0
                || (owner_info.element_count.is_some() && owner_info.byte_end()? > pool_len)
            {
                return unsupported(format!(
                    "buffer:{}:declared allocation exceeds tirx.pool_max_bytes={pool_len}",
                    buffer_name(&owner)
                ));
            }
            owner_end = Some(pool_len);
        } else {
            let mut end = match owner_info.element_count {
                None => None,
                Some(_) => Some(owner_info.byte_end()?),
            };
            let zero_extent_pool_owner = parameter.is_none()
                && space == MemorySpace::Shared
                && owner_start == 0
                && owner_info.element_count == Some(0);
            if zero_extent_pool_owner {
                let mut evidenced: Vec<i64> = Vec::new();
                for (buffer, _, info, _) in &infos {
                    if same(buffer.as_var(), owner.as_var()) {
                        continue;
                    }
                    if info.dynamic_elem_offset.is_none() && info.element_count.is_some() {
                        let byte_end = info.byte_end()?;
                        if byte_end > 0 {
                            evidenced.push(byte_end);
                        }
                    }
                }
                if evidenced.is_empty() {
                    return unsupported(format!(
                        "buffer:{}:zero-extent shared pool owner has no positive static view span or declared pool capacity",
                        buffer_name(&owner)
                    ));
                }
                end = evidenced.iter().copied().max();
            }
            owner_end = end;
        }
        let owner_end_expression: Scalar = match owner_end {
            Some(end) => Scalar::Int(end),
            None => shape_expressions.resolve(
                analyzer,
                &buffer_byte_end_expression(&owner, &owner_info)?,
                buffer_bindings,
            )?,
        };
        for (buffer, is_parameter, info, _) in &infos {
            if same(buffer.as_var(), owner.as_var()) {
                continue;
            }
            if info.dynamic_elem_offset.is_some() {
                continue;
            }
            if info.explicit_strides.is_some() {
                continue;
            }
            let runtime_sized_global_alias = !is_parameter
                && space == MemorySpace::Global
                && owner_end.is_some()
                && info.element_count.is_none()
                && info.itemsize == owner_info.itemsize;
            let launch_bounded_global_alias = parameter.is_some()
                && !is_parameter
                && space == MemorySpace::Global
                && owner_end.is_none();
            if owner_end.is_none() || info.element_count.is_none() {
                let view_start = info.elem_offset * info.itemsize;
                let view_end_expression = shape_expressions.resolve(
                    analyzer,
                    &buffer_byte_end_expression(buffer, info)?,
                    buffer_bindings,
                )?;
                let mut proven = view_start >= owner_start;
                if proven {
                    let condition = op_binary(
                        "_OpLE",
                        view_end_expression.any(),
                        owner_end_expression.any(),
                    )?;
                    proven = analyzer.can_prove(&condition)?;
                }
                if !proven && !(runtime_sized_global_alias || launch_bounded_global_alias) {
                    return unsupported(format!(
                        "buffer:{}:cannot prove alias bounds within runtime-sized owner {}",
                        buffer_name(buffer),
                        buffer_name(&owner)
                    ));
                }
                continue;
            }
            let view_start = info.elem_offset * info.itemsize;
            let view_end = info.byte_end()?;
            let end = owner_end.unwrap();
            if view_start < owner_start || view_end > end {
                if !referenced
                    .iter()
                    .any(|candidate| same(candidate.as_var(), buffer.as_var()))
                {
                    continue;
                }
                return unsupported(format!(
                    "buffer:{}:byte range [{view_start}, {view_end}) exceeds owner {} range [{owner_start}, {end})",
                    buffer_name(buffer),
                    buffer_name(&owner)
                ));
            }
        }
        let owner_alignment = i64::from(owner.buffer_type().data_alignment);
        backing_plans.push(BackingPlan {
            index,
            field: format!("backing_{index}"),
            space,
            byte_len: owner_end,
            byte_alignment: if owner_alignment > 0 {
                Some(owner_alignment)
            } else {
                None
            },
            input_name,
            tmem_lanes: None,
            tmem_columns: None,
        });
        for (buffer, is_parameter, info, _) in &infos {
            buffer_plans.push(BufferPlan {
                buffer: buffer.clone(),
                name: buffer_name(buffer),
                backing_index: Some(index),
                space,
                layout: info.clone(),
                is_parameter: *is_parameter,
                dynamic_data_var: None,
                referenced: true,
            });
        }
    }

    for (key, _) in &pool_sizes {
        if !builders.iter().any(|builder| same(&builder.key, key)) {
            return unsupported(
                "tirx.pool_max_bytes refers to a data variable without an allocation owner",
            );
        }
    }

    if !tmem_buffers.is_empty() {
        let mut infos: Vec<(BufferVar, LayoutInfo)> = Vec::new();
        for buffer in &tmem_buffers {
            infos.push((buffer.clone(), ctx.inspect_layout(buffer, &bindings)?));
        }
        let dynamic_base = infos.iter().any(|(_, info)| {
            info.allocated_addr_static.is_none() || info.tmem_tcol_base_static.is_none()
        });
        let mut static_column_end: i64 = 0;
        for (_, info) in &infos {
            if let Some(addr) = info.allocated_addr_static {
                let columns = addr
                    + ((info.elem_offset + info.tmem_tcol_span_elements.unwrap_or(0))
                        * info.itemsize
                        * 8
                        + 31)
                        / 32;
                static_column_end = static_column_end.max(columns);
            }
        }
        let tmem_columns = static_column_end.max(if dynamic_base { 512 } else { 0 });
        if tmem_columns <= 0 || tmem_columns > 512 {
            return unsupported(format!(
                "TMEM physical column span {tmem_columns} is outside 1..512"
            ));
        }
        let backing_index = backing_plans.len();
        backing_plans.push(BackingPlan {
            index: backing_index,
            field: format!("backing_{backing_index}"),
            space: MemorySpace::Tmem,
            byte_len: Some(128 * tmem_columns * 4),
            byte_alignment: None,
            input_name: None,
            tmem_lanes: Some(128),
            tmem_columns: Some(tmem_columns),
        });
        for (buffer, info) in infos {
            buffer_plans.push(BufferPlan {
                buffer: buffer.clone(),
                name: buffer_name(&buffer),
                backing_index: Some(backing_index),
                space: MemorySpace::Tmem,
                layout: info,
                is_parameter: false,
                dynamic_data_var: None,
                referenced: true,
            });
        }
    }

    for (buffer, data_var, space) in dynamic_views {
        let info = ctx.inspect_layout(&buffer, &bindings)?;
        let referenced_view = dynamic_referenced
            .iter()
            .any(|candidate| same(candidate.as_var(), buffer.as_var()));
        buffer_plans.push(BufferPlan {
            buffer: buffer.clone(),
            name: buffer_name(&buffer),
            backing_index: None,
            space,
            layout: info,
            is_parameter: false,
            dynamic_data_var: Some(data_var),
            referenced: referenced_view,
        });
    }

    Ok(MemoryPlan {
        backings: backing_plans,
        buffers: buffer_plans,
    })
}
