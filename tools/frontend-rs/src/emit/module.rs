//! `artifact_template.emit_rust_module`: the generated module around the
//! kernel bodies (host bindings, physical allocation, launch phases).

use tvm::tirx::BufferVar;

use super::super::analyze::cuda_arch::cuda_arch;
use super::super::analyze::frontend::KernelPlan;
use super::super::analyze::memory::MemorySpace;
use super::super::analyze::shapes::Scalar;
use super::super::analyze::util::{as_var, ident, oref, same, unsupported, AResult};
use super::super::analyze::Ctx;
use super::module_template::MODULE_TEMPLATE;
use super::{BufferCode, EmitOptions, Emitter, RustValue, SplitHelper, Uniformity, Variables};
use crate::decode::projected_buffer;
use crate::tables::{json_string, phase_binding_name};

pub struct ModuleRequest {
    pub analysis_capable: bool,
    pub analysis_checker: Option<String>,
    pub abi_version: i64,
    pub split_thresholds: super::scaffold::SplitThresholds,
}

fn fill(template: &str, pairs: &[(&str, &str)]) -> String {
    let keys: Vec<_> = pairs
        .iter()
        .map(|(key, value)| (format!("__NUMSIM_{}__", key.to_ascii_uppercase()), *value))
        .collect();
    let mut output = String::with_capacity(template.len());
    let mut rest = template;
    while let Some(start) = rest.find("__NUMSIM_") {
        output.push_str(&rest[..start]);
        rest = &rest[start..];
        let Some(end) = rest[9..].find("__") else {
            break;
        };
        let end = 9 + end + 2;
        let token = &rest[..end];
        output.push_str(
            keys.iter()
                .find(|(key, _)| key == token)
                .map_or(token, |(_, value)| *value),
        );
        rest = &rest[end..];
    }
    output.push_str(rest);
    output
}

#[cfg(test)]
mod template_tests {
    #[test]
    fn replacements_are_literal_and_unbound_artifact_markers_survive() {
        assert_eq!(
            super::fill(
                "__NUMSIM_BODY__ __NUMSIM_INDEX__ __NUMSIM_MODULE__",
                &[("body", "literal __NUMSIM_INDEX__"), ("index", "7")],
            ),
            "literal __NUMSIM_INDEX__ 7 __NUMSIM_MODULE__"
        );
    }
}

fn align_up(value: i64, alignment: i64) -> i64 {
    ((value + alignment - 1) / alignment) * alignment
}

fn optional(value: Option<i64>) -> String {
    match value {
        Some(value) => value.to_string(),
        None => "None".to_owned(),
    }
}

fn split_future_type(function_name: &str) -> String {
    format!("NumSimFuture_{function_name}")
}

/// Bind wrappers to final callees, after shared-block renaming is complete.
pub fn render_split_future_calls(body: &str) -> String {
    const CALL: &str = "ctx = NumSimModuleFuture(";
    body.split('\n')
        .map(|line| {
            let indent = line.len() - line.trim_start_matches([' ', '\t']).len();
            let Some(tail) = line[indent..].strip_prefix(CALL) else {
                return line.to_owned();
            };
            let name_len = tail
                .char_indices()
                .take_while(|(at, c)| c.is_ascii_alphabetic() || *c == '_' || (*at > 0 && c.is_ascii_digit()))
                .count();
            if name_len == 0 || !tail[name_len..].starts_with('(') {
                return line.to_owned();
            }
            let name = &tail[..name_len];
            format!(
                "{}ctx = {}({name}({}",
                &line[..indent],
                split_future_type(name),
                &tail[name_len + 1..]
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

impl SplitHelper {
    fn render_function(
        &self,
        body: &str,
        warp_engine_type: &str,
        name: &str,
        parameters: &[(String, String)],
        optimize_size: bool,
    ) -> String {
        let mut attributes = Vec::new();
        if self.inline_never {
            attributes.push("#[inline(never)]");
        }
        if optimize_size {
            attributes.push("#[optimize(size)]");
        }
        let attribute = if attributes.is_empty() {
            String::new()
        } else {
            format!("{}\n", attributes.join("\n"))
        };
        let async_prefix = if self.is_async { "async " } else { "" };
        let declarations: String = parameters
            .iter()
            .map(|(name, rust_type)| format!("    {name}: {rust_type},\n"))
            .collect();
        format!(
            "{attribute}pub(super) {async_prefix}fn {name}(\n    mut ctx: WarpContext,\n    warp: &mut {warp_engine_type},\n{declarations}) -> Result<WarpContext, EngineError> {{\n{body}\n    Ok(ctx)\n}}"
        )
    }

    /// One private module per split helper.
    pub fn render_module(
        &self,
        warp_engine_type: &str,
        name: Option<&str>,
        parameters: Option<&[(String, String)]>,
        body: Option<&str>,
        optimize_size: bool,
    ) -> String {
        let function_name = name.unwrap_or(&self.name);
        let module_name = format!("numsim_split_{function_name}");
        let rendered_body = render_split_future_calls(&match body {
            Some(body) => body.to_owned(),
            None => self.body.join("\n"),
        });
        let (local_future, future_import) = if self.is_async {
            (
                "numsim_local_future!(pub(super));",
                format!(
                    "\nuse {module_name}::NumSimModuleFuture as {};",
                    split_future_type(function_name)
                ),
            )
        } else {
            ("", String::new())
        };
        let own_parameters: Vec<(String, String)>;
        let parameters = match parameters {
            Some(parameters) => parameters,
            None => {
                own_parameters = self
                    .arguments
                    .iter()
                    .map(|argument| (argument.name.clone(), argument.rust_type.clone()))
                    .collect();
                &own_parameters
            }
        };
        let function = self.render_function(
            &rendered_body,
            warp_engine_type,
            function_name,
            parameters,
            optimize_size,
        );
        format!(
            "mod {module_name} {{\n    use super::*;\n    {local_future}\n{function}\n}}\nuse {module_name}::{function_name};{future_import}"
        )
    }
}

struct KernelItem {
    index: usize,
    tmem_columns: i64,
    name: String,
    clusters: i64,
    ctas_per_cluster: i64,
    warps_per_cta: i64,
    warp_function: String,
    analysis_warp_function: String,
    execute_function: String,
    prepare_function: String,
    body: String,
    warp_engine_type: String,
    split_helpers: String,
    private_metadata_declarations: String,
    fields: String,
    /// The buffer handles localized per cluster (empty without shared memory).
    shared_handle_fields: Vec<String>,
    readonly_proxy: bool,
    setup: String,
    initializers: String,
    setmaxnreg_calling_initial_count: Option<i64>,
    fixed_trace_statically_eligible: bool,
    global_memory_model_enabled: bool,
    racecheck_write_seed: String,
}

impl<'a> Emitter<'a> {
    fn dynamic_layout_view_byte_len(&self, buffer: &BufferVar) -> AResult<i64> {
        let plan = self.memory_plan.resolve(buffer)?;
        let name = plan.name.clone();
        if plan.layout.dynamic_layout_elem_offset.is_none() {
            return unsupported(format!(
                "buffer:{name}:does not have a runtime-shifted physical layout"
            ));
        }
        let Some(backing_index) = plan.backing_index else {
            return unsupported(format!(
                "buffer:{name}:runtime-shifted layout has no static backing"
            ));
        };
        let child_start = plan.layout.elem_offset * plan.layout.itemsize;
        let mut current = buffer.clone();
        loop {
            let parent = match self.buffer_bindings.declared_data(&current) {
                Some(data) => projected_buffer(&oref(data))?,
                None => None,
            };
            let Some(parent) = parent else {
                return unsupported(format!(
                    "buffer:{name}:runtime-shifted layout has no declared static parent"
                ));
            };
            let parent_plan = self.memory_plan.resolve(&parent)?;
            if parent_plan.backing_index != Some(backing_index) {
                return unsupported(format!(
                    "buffer:{name}:runtime-shifted layout crosses physical backings"
                ));
            }
            let layout = &parent_plan.layout;
            if layout.dynamic_elem_offset.is_none()
                && layout.dynamic_layout_elem_offset.is_none()
                && layout.element_count.is_some()
            {
                let parent_start = layout.elem_offset * layout.itemsize;
                let parent_end = layout.byte_end()?;
                if child_start < parent_start || child_start >= parent_end {
                    return unsupported(format!(
                        "buffer:{name}:runtime-shifted layout starts outside its declared static parent"
                    ));
                }
                return Ok(parent_end - child_start);
            }
            current = parent;
        }
    }

    fn append_converted_host_integer(
        &mut self,
        setup: &mut Vec<String>,
        host_variables: &Variables,
        name: &str,
        expression: &tvm::ir::PrimExpr,
        field: &str,
        target_type: &str,
    ) -> AResult<()> {
        let (lines, value) =
            self.emit_host_integer(&oref(expression.clone()), host_variables, field)?;
        setup.extend(lines.iter().map(|line| format!("        {line}")));
        setup.push(format!(
            "        let {name}_value: {} = {};\n        let {name} = {target_type}::try_from({name}_value).map_err(|_| {{\n            PyValueError::new_err(format!(\n                {},\n                {name}_value,\n            ))\n        }})?;",
            value.rust_type,
            value.code,
            json_string(&format!("{field} is negative or too large: {{}}"))
        ));
        Ok(())
    }
}

/// Render one kernel of the generated module from its finished emitter.
fn render_kernel(
    emitter: &mut Emitter<'_>,
    index: usize,
    count: usize,
    body: String,
    request: &ModuleRequest,
) -> AResult<KernelItem> {
    let plan = emitter.plan;
    let topology = &plan.topology;
    let warp_engine_type = if request.analysis_checker.as_deref() == Some("racecheck") {
        "RaceCheckWarpEngine"
    } else if request.analysis_capable {
        "SyncCheckWarpEngine"
    } else {
        "NumSimWarpEngine"
    };
    let mut setup: Vec<String> = Vec::new();
    parameter_setup(emitter, &mut setup);
    let host_integer_variables = host_integer_variables(emitter);
    derived_shape_setup(emitter, &host_integer_variables, &mut setup)?;
    tensor_map_setup(emitter, &host_integer_variables, index, count, &mut setup)?;
    let is_sm107a = cuda_arch(emitter.func)?.as_deref() == Some("sm_107a");
    let tmem_columns: i64 = if is_sm107a { 576 } else { 512 };
    let shared_descriptor_address_bits = if is_sm107a { 19 } else { 18 };
    let mut private_metadata_declarations: Vec<String> = Vec::new();
    let backings = backing_setup(
        emitter,
        tmem_columns,
        shared_descriptor_address_bits,
        index,
        &mut setup,
        &mut private_metadata_declarations,
    )?;
    let buffer_initializers = buffer_setup(
        emitter,
        &backings,
        index,
        count,
        &mut setup,
        &mut private_metadata_declarations,
    )?;
    persistent_buffer_setup(emitter, &buffer_initializers, &mut setup);
    address_candidate_setup(emitter, &mut setup)?;
    let single = count == 1;
    let warp_function = if single {
        "warp_main".to_owned()
    } else {
        format!("kernel_{index}_warp_main")
    };
    let (split_helpers, body) = if !request.analysis_capable {
        super::shared_blocks::render_shared_splits(&emitter.split_helpers, &body, warp_engine_type)?
    } else {
        let modules: Vec<String> = emitter
            .split_helpers
            .iter()
            .map(|helper| helper.render_module(warp_engine_type, None, None, None, true))
            .collect();
        (modules.join("\n\n"), render_split_future_calls(&body))
    };
    let struct_fields = buffer_struct_fields(emitter);
    let mut shared_handle_fields: Vec<String> = Vec::new();
    if backings.shared_virtual_end != 0 {
        shared_handle_fields.push("buffers".to_owned());
        if emitter.memory_candidates.shared {
            shared_handle_fields.push("raw_shared_candidates".to_owned());
        }
        if emitter.memory_candidates.generic {
            shared_handle_fields.push("raw_generic_candidates".to_owned());
        }
    }
    let analysis_warp_function = if request.analysis_capable {
        warp_function.clone()
    } else if single {
        "warp_main_analysis_unavailable".to_owned()
    } else {
        format!("kernel_{index}_warp_main_analysis_unavailable")
    };
    let racecheck_write_seed = match &emitter.written_global_buffers {
        None => "allocation_ids.to_vec()".to_owned(),
        Some(indices) => {
            let indices = indices.iter().map(|index| format!("{index}_usize"))
                .collect::<Vec<_>>().join(", ");
            let seed = format!(r#"[{indices}].map(|index: usize| match &buffers.buffers[index] {{
                    RuntimeBuffer::Global(view) => view.allocation(),
                    _ => unreachable!("statically global buffer"),
                }}).to_vec()"#);
            let mut additions = Vec::new();
            for index in &emitter.written_global_parameters {
                let code = emitter.pointers.iter()
                    .find(|p| same(&p.variable, &emitter.params[*index]))
                    .expect("pointer parameter");
                let pointer = format!("buffers.{}", code.field);
                additions.push(format!("match {pointer}.buffer() {{ RuntimeBuffer::Global(view) => writes.push(view.allocation()), _ => writes.extend_from_slice(&allocation_ids), }};"));
            }
            for index in &emitter.written_tensor_maps {
                let code = emitter.raw_tma.tensor_maps.iter()
                    .find(|p| same(&p.variable, &emitter.params[*index]))
                    .expect("TensorMap parameter");
                additions.push(format!("writes.push(buffers.{}.allocation());", code.field));
            }
            if additions.is_empty() {
                seed
            } else {
                format!("{{ let mut writes = {seed}; {} writes }}", additions.join("\n"))
            }
        }
    };
    Ok(KernelItem {
        racecheck_write_seed,
        index,
        tmem_columns,
        name: plan.name.clone(),
        clusters: topology.clusters,
        ctas_per_cluster: topology.ctas_per_cluster,
        warps_per_cta: topology.warps_per_cta,
        warp_function,
        analysis_warp_function,
        execute_function: if single {
            "execute_kernel".to_owned()
        } else {
            format!("execute_kernel_{index}")
        },
        prepare_function: format!("prepare_kernel_{index}_buffers"),
        body,
        warp_engine_type: warp_engine_type.to_owned(),
        split_helpers,
        private_metadata_declarations: private_metadata_declarations.join("\n"),
        fields: struct_fields
            .iter()
            .map(|(field, rust_type)| format!("    {field}: {rust_type},"))
            .collect::<Vec<_>>()
            .join("\n"),
        shared_handle_fields,
        readonly_proxy: plan.uses_readonly_proxy,
        setup: setup.join("\n"),
        initializers: struct_fields
            .iter()
            .map(|(field, _)| match field.as_str() {
                "buffers" => "buffers: persistent_buffers",
                field => field,
            })
            .collect::<Vec<_>>()
            .join(", "),
        setmaxnreg_calling_initial_count: emitter
            .sync
            .calling_initial_count(&emitter.plan.topology)?,
        // Without analysis-capable slices the fixed-trace eligibility is the
        // empty slice's: statically eligible with no read-only globals.
        fixed_trace_statically_eligible: !(request.analysis_capable
            && super::super::analyze::tile_forms::fixed_trace_has_unknown_call(
                emitter.ctx,
                &emitter.op_ids,
                &emitter.plan.tile_calls,
            )?),
        global_memory_model_enabled: true,
    })
}

/// The `Kernel{index}Buffers` fields and their Rust types, in declaration order.
fn buffer_struct_fields(emitter: &Emitter<'_>) -> Vec<(String, String)> {
    let typed = |field: &str, rust_type: &str| (field.to_owned(), rust_type.to_owned());
    let mut fields = vec![typed("buffers", "Vec<RuntimeBuffer>")];
    if emitter.memory_candidates.shared {
        fields.push(typed("raw_shared_candidates", "Vec<RuntimeBuffer>"));
    }
    if emitter.memory_candidates.generic {
        fields.push(typed("raw_generic_candidates", "Vec<RuntimeBuffer>"));
    }
    for scalar in &emitter.scalars {
        fields.push(typed(&scalar.field, &scalar.rust_type));
    }
    for pointer in &emitter.pointers {
        fields.push(typed(&pointer.field, "PhysicalPtr"));
    }
    for shape in &emitter.shape_vars {
        fields.push(typed(&shape.field, "usize"));
    }
    for tensor_map in &emitter.raw_tma.tensor_maps {
        fields.push(typed(&tensor_map.field, "RuntimeTensorMap"));
    }
    if emitter.raw_tma.uses_registry {
        fields.push(typed("tensor_map_registry", "RuntimeTensorMapRegistry"));
    }
    if emitter.tmem.implicit {
        fields.push(typed("implicit_tmem", "RuntimeBuffer"));
    }
    for (_, field) in &emitter.raw_tcgen.shared_descriptor_domains {
        fields.push(typed(field, "v2::DescriptorDomain<v2::Shared>"));
    }
    fields
}

/// The buffer-struct field of a shape variable.
fn shape_field<'e>(emitter: &'e Emitter<'_>, variable: &tvm::ir::Var) -> Option<&'e str> {
    emitter
        .shape_vars
        .iter()
        .find(|shape| ident(&shape.variable) == ident(variable))
        .map(|shape| shape.field.as_str())
}

/// Extract the scalar, pointer and shape parameters, validating scalars that
/// are also shape variables.
fn parameter_setup(emitter: &Emitter<'_>, setup: &mut Vec<String>) {
    for scalar in &emitter.scalars {
        setup.push(format!(
            "        let {} = extract_scalar_{}(\n            inputs,\n            \"{}\",\n            \"{}\",\n        )?;",
            scalar.field, scalar.rust_type, scalar.binding_name, scalar.dtype
        ));
    }
    for pointer in &emitter.pointers {
        setup.push(format!(
            "        let {} = extract_pointer(\n            inputs,\n            \"{}\",\n            \"{}\",\n            {},\n            physical.global(),\n            allocation_ids,\n        )?;",
            pointer.field, pointer.binding_name, pointer.dtype, pointer.itemsize
        ));
    }
    for shape in &emitter.shape_vars {
        let sources: Vec<String> = shape
            .sources
            .iter()
            .map(|(name, axis)| format!("(\"{name}\", {axis})"))
            .collect();
        setup.push(format!(
            "        let {} = extract_shape_extent(\n            inputs,\n            \"{}\",\n            &[{}],\n        )?;",
            shape.field,
            shape.name,
            sources.join(", ")
        ));
    }
    for scalar in &emitter.scalars {
        if let Some(shape_field) = shape_field(emitter, &scalar.variable) {
            setup.push(format!(
                "        validate_shape_scalar(\n            \"{}\",\n            {},\n            {},\n        )?;",
                scalar.name, scalar.field, shape_field
            ));
        }
    }
}

/// The scalar and shape parameters as host integers.
fn host_integer_variables(emitter: &Emitter<'_>) -> Variables {
    let mut host_integer_variables = Variables::default();
    for scalar in &emitter.scalars {
        host_integer_variables.set(
            oref(scalar.variable.clone()),
            RustValue::new(
                scalar.field.clone(),
                scalar.rust_type.clone(),
                Uniformity::Uniform,
            ),
        );
    }
    for shape in &emitter.shape_vars {
        let key = oref(shape.variable.clone());
        if !host_integer_variables.contains(&key) {
            host_integer_variables.set(
                key,
                RustValue::new(
                    format!("{} as {}", shape.field, shape.rust_type),
                    shape.rust_type.clone(),
                    Uniformity::Uniform,
                ),
            );
        }
    }
    host_integer_variables
}

/// The buffer-struct setup local of a buffer extent that is an expression.
fn derived_shape_field(buffer_index: usize, axis: usize) -> String {
    format!("derived_shape_{buffer_index}_{axis}")
}

/// Compute every buffer extent that is an expression rather than a variable.
fn derived_shape_setup(
    emitter: &mut Emitter<'_>,
    host_integer_variables: &Variables,
    setup: &mut Vec<String>,
) -> AResult<()> {
    for buffer_index in 0..emitter.buffers.len() {
        let plan_name = emitter
            .plan_of(&emitter.buffers[buffer_index].1)
            .name
            .clone();
        let shape: Vec<Scalar> = emitter.buffers[buffer_index].1.shape.clone();
        for (axis, extent) in shape.iter().enumerate() {
            let Scalar::Expr(expression) = extent else {
                continue;
            };
            if as_var(&oref(expression.clone())).is_some() {
                continue;
            }
            emitter.append_converted_host_integer(
                setup,
                host_integer_variables,
                &derived_shape_field(buffer_index, axis),
                expression,
                &format!("{plan_name}:shape[{axis}]"),
                "usize",
            )?;
        }
    }
    Ok(())
}

/// Extract the TensorMap parameters and, for raw TensorMap calls, their
/// descriptor registry.
fn tensor_map_setup(
    emitter: &mut Emitter<'_>,
    host_integer_variables: &Variables,
    index: usize,
    count: usize,
    setup: &mut Vec<String>,
) -> AResult<()> {
    let plan = emitter.plan;
    let tensor_map_codes: Vec<(String, String, String)> = emitter
        .raw_tma
        .tensor_maps
        .iter()
        .map(|tensor_map| {
            (
                tensor_map.name.clone(),
                tensor_map.binding_name.clone(),
                tensor_map.field.clone(),
            )
        })
        .collect();
    for ((name, binding_name, field), spec) in tensor_map_codes.iter().zip(plan.tensor_maps.iter())
    {
        if spec.implicit {
            emitter.implicit_tensor_map_setup(
                setup,
                host_integer_variables,
                name,
                binding_name,
                field,
                index,
                count,
            )?;
            continue;
        }
        setup.push(format!(
            "        let {} = extract_tensor_map(\n            inputs,\n            \"{}\",\n            physical.global(),\n            allocation_ids,\n        )?;",
            field, binding_name
        ));
    }
    if !emitter.raw_tma.uses_registry {
        return Ok(());
    }
    let mut names: Vec<String> = emitter
        .pointers
        .iter()
        .map(|pointer| pointer.binding_name.clone())
        .chain(
            emitter
                .buffers
                .iter()
                .filter_map(|(_, code)| code.input_name.clone()),
        )
        .collect();
    names.sort();
    names.dedup();
    setup.push(
        "        let tensor_map_registry = extract_tensor_map_descriptor_registry(".to_owned(),
    );
    setup.push("            inputs, physical.global(), allocation_ids,".to_owned());
    setup.push("            &[".to_owned());
    for name in &names {
        setup.push(format!("                {},", json_string(name)));
    }
    setup.push("            ],".to_owned());
    setup.push("            &[".to_owned());
    for tensor_map in &emitter.raw_tma.tensor_maps {
        setup.push(format!(
            "                (\"{}\", {}.clone()),",
            tensor_map.binding_name, tensor_map.field
        ));
    }
    setup.push("            ],".to_owned());
    setup.push("        )?;".to_owned());
    Ok(())
}

const SHARED_BACKING_FIELD: &str = "numsim_cta_shared_backing";

/// Where `backing_setup` placed the shared and warp-private backings.
struct BackingLayout {
    /// `(backing index, virtual base)` of every shared backing.
    shared_virtual_bases: Vec<(usize, i64)>,
    /// The end of the shared virtual address span.
    shared_virtual_end: i64,
    local_backings: Vec<usize>,
    register_backings: Vec<usize>,
}

/// Allocate the shared, TMEM, warp-private and implicit TMEM backings.
fn backing_setup(
    emitter: &Emitter<'_>,
    tmem_columns: i64,
    shared_descriptor_address_bits: i64,
    index: usize,
    setup: &mut Vec<String>,
    private_metadata_declarations: &mut Vec<String>,
) -> AResult<BackingLayout> {
    let kernel_name = &emitter.plan.name;
    let mut local_backings: Vec<usize> = Vec::new();
    let mut register_backings: Vec<usize> = Vec::new();
    let mut shared_virtual_bases: Vec<(usize, i64)> = Vec::new();
    let mut shared_virtual_end: i64 = 0;
    for backing in &emitter.memory_plan.backings {
        if backing.space != MemorySpace::Shared {
            continue;
        }
        shared_virtual_end = align_up(
            shared_virtual_end,
            16.max(backing.byte_alignment.unwrap_or(1)),
        );
        shared_virtual_bases.push((backing.index, shared_virtual_end));
        shared_virtual_end += backing.byte_len.unwrap_or(0);
    }
    if shared_virtual_end > (1i64 << shared_descriptor_address_bits) {
        return unsupported(format!(
            "kernel:{kernel_name}:shared-memory virtual address span {shared_virtual_end} exceeds the {shared_descriptor_address_bits}-bit descriptor address space"
        ));
    }
    if shared_virtual_end != 0 {
        setup.push(format!(
            "        let {SHARED_BACKING_FIELD} = Arc::new(allocate_cta_shared(&physical, topology, {shared_virtual_end}).map_err(|error| PyValueError::new_err(error.to_string()))?);"
        ));
    }
    for backing in &emitter.memory_plan.backings {
        match backing.space {
            MemorySpace::Global | MemorySpace::Shared => {}
            MemorySpace::Local => local_backings.push(backing.index),
            MemorySpace::Register => register_backings.push(backing.index),
            MemorySpace::Tmem => setup.push(format!(
                "        let {} = Arc::new(allocate_cta_tmem(physical, topology, {}, {}).map_err(|error| PyValueError::new_err(error.to_string()))?);",
                backing.field,
                optional(backing.tmem_lanes),
                optional(backing.tmem_columns)
            )),
        }
    }
    for (backings, name, accessor) in [
        (&local_backings, "local", "local"),
        (&register_backings, "register", "registers"),
    ] {
        if backings.is_empty() {
            continue;
        }
        let backing_lengths_name = format!(
            "KERNEL_{index}_{}_BACKING_BYTE_LENGTHS",
            name.to_uppercase()
        );
        let byte_lengths: Vec<String> = backings
            .iter()
            .map(|at| {
                format!(
                    "{}_usize",
                    optional(emitter.memory_plan.backings[*at].byte_len)
                )
            })
            .collect();
        private_metadata_declarations.push(format!(
            "static {backing_lengths_name}: &[usize] = &[{}];",
            byte_lengths.join(", ")
        ));
        setup.push(format!(
            "        let mut {name}_backings =\n            Vec::with_capacity({backing_lengths_name}.len());\n        for &byte_len in {backing_lengths_name} {{\n            {name}_backings.push(\n                allocate_warp_private(physical.{accessor}(), topology, byte_len)\n                .map(Arc::new)\n                .map_err(|error| PyValueError::new_err(error.to_string()))?,\n            );\n        }}"
        ));
    }
    if emitter.tmem.implicit {
        setup.extend([
            format!(
                "        let implicit_tmem_allocations = Arc::new(allocate_cta_tmem(physical, topology, 128, {tmem_columns}).map_err(|error| PyValueError::new_err(error.to_string()))?);"
            ),
            "        let implicit_tmem = runtime_buffer_tmem(".to_owned(),
            format!("            implicit_tmem_allocations, 128, {tmem_columns}, 0, 4,"),
            "        );".to_owned(),
        ]);
    }
    Ok(BackingLayout {
        shared_virtual_bases,
        shared_virtual_end,
        local_backings,
        register_backings,
    })
}

enum Initializer {
    Field(String),
    Private(&'static str, usize),
}

/// Create the runtime view of every persistent buffer, returning the
/// initializer of each storage index.
fn buffer_setup(
    emitter: &Emitter<'_>,
    backings: &BackingLayout,
    index: usize,
    count: usize,
    setup: &mut Vec<String>,
    private_metadata_declarations: &mut Vec<String>,
) -> AResult<Vec<(usize, Initializer)>> {
    let mut buffer_initializers: Vec<(usize, Initializer)> = Vec::new();
    let mut local_descriptors: Vec<(usize, i64, String)> = Vec::new();
    let mut register_descriptors: Vec<(usize, i64, String)> = Vec::new();
    for (buffer_index, (buffer, code)) in emitter.buffers.iter().enumerate() {
        let plan = emitter.plan_of(code);
        if plan.dynamic_data_var.is_some() {
            continue;
        }
        let storage_index = code.storage_index.expect("storage index");
        let field = code.field.clone();
        match plan.space {
            MemorySpace::Global => {
                setup.push(global_buffer_view(
                    emitter,
                    code,
                    buffer_index,
                    index,
                    count,
                )?);
                buffer_initializers.push((storage_index, Initializer::Field(field)));
            }
            MemorySpace::Shared => {
                setup.push(shared_buffer_view(emitter, backings, buffer, code)?);
                buffer_initializers.push((storage_index, Initializer::Field(field)));
            }
            MemorySpace::Local | MemorySpace::Register => {
                let backing_index = plan.backing_index.expect("backing");
                let (descriptors, positions, name) = if plan.space == MemorySpace::Local {
                    (&mut local_descriptors, &backings.local_backings, "local")
                } else {
                    (
                        &mut register_descriptors,
                        &backings.register_backings,
                        "register",
                    )
                };
                let position = positions
                    .iter()
                    .position(|at| *at == backing_index)
                    .expect("private backing");
                let descriptor_index = descriptors.len();
                descriptors.push((
                    position,
                    plan.layout.elem_offset * plan.layout.itemsize,
                    optional(
                        plan.layout
                            .element_count
                            .map(|count| count * plan.layout.itemsize),
                    ),
                ));
                buffer_initializers
                    .push((storage_index, Initializer::Private(name, descriptor_index)));
            }
            MemorySpace::Tmem => {
                let backing = &emitter.memory_plan.backings[plan.backing_index.expect("backing")];
                setup.push(format!(
                    "        let {field} = runtime_buffer_tmem(\n            {}.clone(),\n            {},\n            {},\n            {},\n            {},\n        );",
                    backing.field,
                    optional(plan.layout.tmem_lane_span),
                    optional(plan.layout.tmem_tcol_span_elements),
                    plan.layout.elem_offset,
                    plan.layout.itemsize
                ));
                buffer_initializers.push((storage_index, Initializer::Field(field)));
            }
        }
    }
    for (descriptors, name, constructor) in [
        (&local_descriptors, "local", "runtime_buffer_local"),
        (&register_descriptors, "register", "runtime_buffer_register"),
    ] {
        if descriptors.is_empty() {
            continue;
        }
        let descriptors_name = format!("KERNEL_{index}_{}_BUFFER_DESCRIPTORS", name.to_uppercase());
        let rendered: Vec<String> = descriptors
            .iter()
            .map(|(backing, offset, length)| {
                format!("({backing}_usize, {offset}_usize, {length}_usize)")
            })
            .collect();
        private_metadata_declarations.push(format!(
            "static {descriptors_name}: &[(usize, usize, usize)] = &[{}];",
            rendered.join(", ")
        ));
        setup.push(format!(
            "        let mut {name}_buffers = Vec::with_capacity({descriptors_name}.len());\n        for &(backing, byte_offset, byte_len) in {descriptors_name} {{\n            {name}_buffers.push({constructor}(\n                {name}_backings[backing].clone(),\n                byte_offset,\n                byte_len,\n            ));\n        }}"
        ));
    }
    Ok(buffer_initializers)
}

/// The runtime view of a global buffer: its own parameter, or an alias of the
/// parameter that owns its backing.
fn global_buffer_view(
    emitter: &Emitter<'_>,
    code: &BufferCode,
    buffer_index: usize,
    index: usize,
    count: usize,
) -> AResult<String> {
    let plan = emitter.plan_of(code);
    let field = &code.field;
    let mut expected_shape_fields: Vec<String> = Vec::new();
    for (axis, extent) in code.shape.iter().enumerate() {
        expected_shape_fields.push(match extent {
            Scalar::Int(value) => value.to_string(),
            Scalar::Expr(expression) => match as_var(&oref(expression.clone())) {
                Some(variable) => shape_field(emitter, &variable)
                    .expect("shape field")
                    .to_owned(),
                None => derived_shape_field(buffer_index, axis),
            },
        });
    }
    let expected_shape = expected_shape_fields.join(", ");
    let physical_byte_len = plan
        .layout
        .element_count
        .map(|count| count * plan.layout.itemsize);
    if plan.is_parameter {
        let physical_byte_len_code = match physical_byte_len {
            None => "None".to_owned(),
            Some(len) => format!("Some({len}_usize)"),
        };
        return Ok(format!(
            "        let {field} = runtime_buffer_global(extract_buffer(\n            inputs,\n            \"{}\",\n            \"{}\",\n            {},\n            &[{expected_shape}],\n            {},\n            {physical_byte_len_code},\n            physical.global(),\n            allocation_ids,\n        )?);",
            code.input_name.clone().unwrap_or_default(),
            plan.layout.dtype,
            plan.layout.itemsize,
            plan.layout.elem_offset * plan.layout.itemsize
        ));
    }
    let backing = &emitter.memory_plan.backings[plan.backing_index.expect("backing")];
    let Some(backing_input_name) = &backing.input_name else {
        return unsupported(format!(
            "buffer:{}:global alias has no parameter owner",
            plan.name
        ));
    };
    let runtime_view_base = plan.layout.dynamic_elem_offset.is_some()
        || plan.layout.dynamic_layout_elem_offset.is_some();
    let (view_byte_offset, physical_byte_len_code) = if runtime_view_base {
        (
            0,
            match backing.byte_len {
                None => "None".to_owned(),
                Some(len) => format!("Some({len}_usize)"),
            },
        )
    } else if physical_byte_len.is_none() {
        let mut code = format!("{}_usize", plan.layout.itemsize);
        for extent in &expected_shape_fields {
            let overflow_message =
                json_string(&format!("buffer '{}' byte length overflow", plan.name));
            code = format!(
                "({code}).checked_mul({extent}).ok_or_else(|| PyValueError::new_err({overflow_message}))?"
            );
        }
        (
            plan.layout.elem_offset * plan.layout.itemsize,
            format!("Some({code})"),
        )
    } else {
        (
            plan.layout.elem_offset * plan.layout.itemsize,
            format!("Some({}_usize)", physical_byte_len.unwrap()),
        )
    };
    Ok(format!(
        "        let {field} = runtime_buffer_global(extract_buffer_alias(\n            inputs,\n            \"{}\",\n            {view_byte_offset},\n            {physical_byte_len_code},\n            physical.global(),\n            allocation_ids,\n        )?);",
        phase_binding_name(index as i64, backing_input_name, count as i64)
    ))
}

/// The runtime view of a shared buffer inside the shared virtual address span.
fn shared_buffer_view(
    emitter: &Emitter<'_>,
    backings: &BackingLayout,
    buffer: &BufferVar,
    code: &BufferCode,
) -> AResult<String> {
    let plan = emitter.plan_of(code);
    let field = &code.field;
    let shared_virtual_end = backings.shared_virtual_end;
    let backing = &emitter.memory_plan.backings[plan.backing_index.expect("backing")];
    let base = backings
        .shared_virtual_bases
        .iter()
        .find(|(at, _)| *at == backing.index)
        .map(|(_, base)| *base)
        .expect("shared base");
    let (virtual_base, view_byte_len): (i64, String) = if plan.layout.dynamic_elem_offset.is_some()
    {
        (base, optional(backing.byte_len))
    } else if plan.layout.dynamic_layout_elem_offset.is_some() {
        (
            base + plan.layout.elem_offset * plan.layout.itemsize,
            emitter.dynamic_layout_view_byte_len(buffer)?.to_string(),
        )
    } else {
        let len = if plan.layout.explicit_strides.is_some() {
            optional(
                backing
                    .byte_len
                    .map(|len| len - plan.layout.elem_offset * plan.layout.itemsize),
            )
        } else {
            optional(
                plan.layout
                    .element_count
                    .map(|count| count * plan.layout.itemsize),
            )
        };
        (base + plan.layout.elem_offset * plan.layout.itemsize, len)
    };
    Ok(format!(
        "        let {field} = runtime_buffer_shared(\n            {SHARED_BACKING_FIELD}.clone(),\n            {virtual_base},\n            {view_byte_len},\n            {shared_virtual_end},\n            {virtual_base},\n        );"
    ))
}

/// Collect the persistent buffers in storage order, extending each run of
/// consecutive private descriptors at once.
fn persistent_buffer_setup(
    emitter: &Emitter<'_>,
    buffer_initializers: &[(usize, Initializer)],
    setup: &mut Vec<String>,
) {
    let persistent_buffer_count = emitter
        .buffers
        .iter()
        .filter(|(_, code)| code.storage_index.is_some())
        .count();
    let initializer_at = |storage_index: usize| -> &Initializer {
        buffer_initializers
            .iter()
            .find(|(at, _)| *at == storage_index)
            .map(|(_, initializer)| initializer)
            .expect("persistent buffer initializer")
    };
    setup.push(format!(
        "        let mut persistent_buffers = Vec::with_capacity({persistent_buffer_count});"
    ));
    let mut at = 0;
    while at < persistent_buffer_count {
        match initializer_at(at) {
            Initializer::Field(field) => {
                setup.push(format!("        persistent_buffers.push({field});"));
                at += 1;
            }
            Initializer::Private(name, first) => {
                let mut run_end = at + 1;
                while run_end < persistent_buffer_count {
                    match initializer_at(run_end) {
                        Initializer::Private(other, position)
                            if other == name && *position == first + run_end - at => {}
                        _ => break,
                    }
                    run_end += 1;
                }
                let run_count = run_end - at;
                setup.push(format!(
                    "        persistent_buffers.extend({name}_buffers[{first}..{}].iter().cloned());",
                    first + run_count
                ));
                at = run_end;
            }
        }
    }
}

/// Select the raw address candidates and the shared descriptor domains from
/// the persistent buffers.
fn address_candidate_setup(emitter: &Emitter<'_>, setup: &mut Vec<String>) -> AResult<()> {
    let selected_buffers = |indices: &[usize]| -> String {
        let rendered: Vec<String> = indices
            .iter()
            .map(|storage_index| format!("{storage_index}_usize"))
            .collect();
        format!(
            "select_runtime_buffers(&persistent_buffers, &[{}])",
            rendered.join(", ")
        )
    };
    if emitter.memory_candidates.shared {
        let candidates = selected_buffers(&emitter.raw_shared_candidate_storage_indices()?);
        setup.push(format!("        let raw_shared_candidates = {candidates};"));
    }
    if emitter.memory_candidates.generic {
        let candidates = selected_buffers(&emitter.raw_generic_candidate_storage_indices()?);
        let mut extra_candidates: Vec<String> = emitter
            .pointers
            .iter()
            .map(|pointer| format!("{}.buffer().clone()", pointer.field))
            .collect();
        if emitter.raw_tma.uses_registry {
            extra_candidates.extend(emitter.raw_tma.tensor_maps.iter().map(|tensor_map| {
                format!(
                    "tensor_map_registry.parameter_address({}).map_err(|error| PyValueError::new_err(error.to_string()))?.buffer().clone()",
                    json_string(&tensor_map.binding_name)
                )
            }));
        }
        setup.push(format!(
            "        let mut raw_generic_candidates = {candidates};"
        ));
        if !extra_candidates.is_empty() {
            setup.push(format!(
                "        raw_generic_candidates.extend([{}]);",
                extra_candidates.join(", ")
            ));
        }
    }
    for (indices, field) in &emitter.raw_tcgen.shared_descriptor_domains {
        setup.push(format!(
            "        let {field} = v2_descriptor_domain::<v2::Shared>({});",
            selected_buffers(indices)
        ));
    }
    Ok(())
}

fn fixed_trace_fast_path_expression(item: &KernelItem) -> String {
    if !item.fixed_trace_statically_eligible {
        return "false".to_owned();
    }
    "true".to_owned()
}

fn local_shared_handles_method(item: &KernelItem) -> String {
    let Some((first, rest)) = item.shared_handle_fields.split_first() else {
        return String::new();
    };
    let chained: String = rest
        .iter()
        .map(|field| format!(".chain(local.{field}.iter_mut())"))
        .collect();
    format!(
        "impl Kernel{}Buffers {{\n    fn with_local_shared_handles(&self) -> Self {{\n        let mut local = self.clone();\n        localize_shared_buffer_handles(local.{first}.iter_mut(){chained});\n        local\n    }}\n}}",
        item.index
    )
}

fn cluster_buffers_setup(item: &KernelItem) -> &'static str {
    if item.shared_handle_fields.is_empty() {
        ""
    } else {
        "let mut cluster_buffers = std::collections::BTreeMap::new();"
    }
}

fn cluster_buffers_select(item: &KernelItem) -> &'static str {
    if item.shared_handle_fields.is_empty() {
        ""
    } else {
        "let buffers = cluster_buffers\n            .entry(artifact_warp_context(&warp).cluster_id())\n            .or_insert_with(|| Arc::new(buffers.with_local_shared_handles()))\n            .clone();"
    }
}

fn setmaxnreg_policy_expression(item: &KernelItem) -> String {
    match item.setmaxnreg_calling_initial_count {
        None => "None".to_owned(),
        Some(count) => format!("Some({count}_u32)"),
    }
}

const PREPARE_TEMPLATE: &str = r#"__NUMSIM_SPLIT_PREFIX__#[inline(never)]
fn __NUMSIM_PREPARE_FUNCTION__(
    inputs: &Bound<'_, PyDict>,
    physical: &PhysicalMemory,
    allocation_ids: &[AllocationId],
    topology: LaunchTopology,
) -> PyResult<Arc<Kernel__NUMSIM_INDEX__Buffers>> {
    physical.global().set_readonly_proxy_tracking(false)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
__NUMSIM_SETUP__
    physical.global().set_readonly_proxy_tracking(__NUMSIM_READONLY_PROXY__)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    Ok(Arc::new(Kernel__NUMSIM_INDEX__Buffers { __NUMSIM_INITIALIZERS__ }))
}"#;

const WARP_TEMPLATE: &str = r#"#[inline(never)]
__NUMSIM_WARP_OPTIMIZATION__
async fn __NUMSIM_WARP_FUNCTION__(
    warp: __NUMSIM_WARP_ENGINE_TYPE__,
    physical: PhysicalMemory,
    buffers: Arc<Kernel__NUMSIM_INDEX__Buffers>,
    services: KernelRuntimeServices,
) -> Result<(), EngineError> {
    let mut warp_engine = warp;
    let warp = &mut warp_engine;
    let mut ctx = artifact_warp_context(warp);
__NUMSIM_BODY__
    Ok(())
}"#;

const EXECUTE_TEMPLATE: &str = r#"fn __NUMSIM_EXECUTE_FUNCTION__(
    physical: PhysicalMemory,
    buffers: Arc<Kernel__NUMSIM_INDEX__Buffers>,
    selection: LaunchSelection,
    max_workers: usize,
    execution_policy: ExecutionPolicy,
) -> Result<ExecutionStats, String> {
    let topology = physical.topology();
    let execution_policy = execution_policy
        .with_setmaxnreg_calling_initial_count(__NUMSIM_SETMAXNREG_POLICY__);
    __NUMSIM_CLUSTER_BUFFERS_SETUP__
    run_kernel_launch_ordered(
        &physical,
        __NUMSIM_INDEX___usize,
        selection,
        max_workers,
        execution_policy,
        |warp, services| {
            __NUMSIM_CLUSTER_BUFFERS_SELECT__
            NumSimModuleFuture(__NUMSIM_WARP_FUNCTION__(
                warp,
                physical.clone(),
                buffers.clone(),
                services,
            ))
        },
    )
    .map_err(|error| error.to_string())
}"#;

const RUN_PHASE_TEMPLATE: &str = r#"    if selected_phase.is_none() || selected_phase >= Some(__NUMSIM_INDEX___usize) {
        {
        let topology = LaunchTopology::new(
            __NUMSIM_CLUSTERS__,
            __NUMSIM_CTAS_PER_CLUSTER__,
            __NUMSIM_WARPS_PER_CTA__,
        )
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let physical = PhysicalMemory::with_global(topology, global.clone());
        let selection = extract_phase_selection(
            subset,
            __NUMSIM_INDEX___usize,
            __NUMSIM_KERNEL_COUNT___usize,
            topology,
        )?;
        let buffers = __NUMSIM_PREPARE_FUNCTION__(
            inputs,
            &physical,
            &allocation_ids,
            topology,
        )?;
        let execution_policy = execution_policy.with_tmem_column_capacity(
            __NUMSIM_TMEM_COLUMNS__,
        );
        run_result.run_phase(
            py,
            __NUMSIM_INDEX___usize,
            __NUMSIM_NAME__,
            topology,
            {
                let physical = physical.clone();
                let buffers = buffers.clone();
                move || {
                    __NUMSIM_EXECUTE_FUNCTION__(
                        physical,
                        buffers,
                        selection,
                        max_workers,
                        execution_policy,
                    )
                }
            },
        )?;
        run_result.record_uninitialized_read_reviews(
            __NUMSIM_INDEX___usize,
            __NUMSIM_NAME__,
            physical.take_uninitialized_read_reviews(),
        );
        }
    }"#;

const SYNC_PHASE_TEMPLATE: &str = r#"        __NUMSIM_INDEX___usize => {
            let topology = LaunchTopology::new(
                __NUMSIM_CLUSTERS__,
                __NUMSIM_CTAS_PER_CLUSTER__,
                __NUMSIM_WARPS_PER_CTA__,
            )
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
            let physical = PhysicalMemory::with_global(topology, global.clone());
            let selection = extract_selection(subset, topology)?;
            let fixed_trace_eligible = __NUMSIM_FIXED_TRACE__;
        let buffers = __NUMSIM_PREPARE_FUNCTION__(
            inputs,
            &physical,
            &allocation_ids,
            topology,
        )?;
        let execution_policy = execution_policy
            .with_setmaxnreg_calling_initial_count(__NUMSIM_SETMAXNREG_POLICY__)
            .with_tmem_column_capacity(__NUMSIM_TMEM_COLUMNS__);
        __NUMSIM_CLUSTER_BUFFERS_SETUP__
        run_synccheck_analysis_phase(
                py,
                __NUMSIM_INDEX___usize,
                __NUMSIM_NAME__,
                physical,
                inputs,
                &allocation_ids,
                selection,
                max_workers,
                execution_policy,
                fixed_trace_eligible,
                max_warp_preemptions,
                max_completion_schedule_deviations,
                max_schedules,
                max_backtrack_nodes,
                max_events_per_run,
                max_total_events,
                max_loop_steps,
                max_wall_time_ms,
                max_diagnostic_bytes,
                max_polls,
                max_transitions,
                |warp, physical, services| {
                    __NUMSIM_CLUSTER_BUFFERS_SELECT__
                    NumSimModuleFuture(__NUMSIM_ANALYSIS_WARP_FUNCTION__(
                        warp,
                        physical,
                        Arc::clone(&buffers),
                        services,
                    ))
                },
            )
        }"#;

const RACE_PHASE_TEMPLATE: &str = r#"        __NUMSIM_INDEX___usize => {
            let topology = LaunchTopology::new(
                __NUMSIM_CLUSTERS__,
                __NUMSIM_CTAS_PER_CLUSTER__,
                __NUMSIM_WARPS_PER_CTA__,
            )
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
            let physical = PhysicalMemory::with_global(topology, global.clone());
            let selection = extract_selection(subset, topology)?;
        let buffers = __NUMSIM_PREPARE_FUNCTION__(
            inputs,
            &physical,
            &allocation_ids,
            topology,
        )?;
        let execution_policy = execution_policy
            .with_setmaxnreg_calling_initial_count(__NUMSIM_SETMAXNREG_POLICY__)
            .with_tmem_column_capacity(__NUMSIM_TMEM_COLUMNS__);
        __NUMSIM_CLUSTER_BUFFERS_SETUP__
        let global_write_allocations = __NUMSIM_RACECHECK_WRITE_SEED__;
        let global_write_allocations = if full_global_write_seed {
            allocation_ids.to_vec()
        } else {
            global_write_allocations
        };
        run_racecheck_analysis_phase(
                py,
                __NUMSIM_INDEX___usize,
                __NUMSIM_NAME__,
                physical,
                global_write_allocations,
                selection,
                max_workers,
                execution_policy,
                inspect_accesses,
                __NUMSIM_GLOBAL_MEMORY_MODEL__,
                max_polls,
                max_transitions,
                |warp, physical, services| {
                    __NUMSIM_CLUSTER_BUFFERS_SELECT__
                    NumSimModuleFuture(__NUMSIM_ANALYSIS_WARP_FUNCTION__(
                        warp,
                        physical,
                        Arc::clone(&buffers),
                        services,
                    ))
                },
            )
        }"#;

fn remove_marked_region(generated: &str, marker: &str) -> String {
    let begin_marker = format!("// {marker}:begin");
    let end_marker = format!("// {marker}:end");
    let begin = generated.find(&begin_marker).expect("region begin");
    let end = generated[begin..].find(&end_marker).expect("region end") + begin + end_marker.len();
    format!("{}{}", &generated[..begin], &generated[end..])
}

const IMPORTS_BEGIN: &str = "// numsim-engine-imports:begin";
const IMPORTS_END: &str = "// numsim-engine-imports:end";

fn import_block() -> String {
    format!("{IMPORTS_BEGIN}\nuse numsim_engine::artifact_support::*;\nuse numsim_engine::abi::v2;\n{IMPORTS_END}")
}

fn render_engine_boundary(source: &str) -> AResult<String> {
    if source.matches(IMPORTS_BEGIN).count() != 1 || source.matches(IMPORTS_END).count() != 1 {
        return Err(super::super::analyze::util::Failure::Ffi(
            super::super::analyze::util::ffi_error(
                "generated Rust must contain exactly one engine import block",
            ),
        ));
    }
    let start = source.find(IMPORTS_BEGIN).expect("imports");
    let end = source[start..].find(IMPORTS_END).expect("imports end") + start + IMPORTS_END.len();
    if source[start..end] != import_block() {
        return Err(super::super::analyze::util::Failure::Ffi(
            super::super::analyze::util::ffi_error(
                "generated Rust engine import block is not canonical",
            ),
        ));
    }
    let body = format!("{}{}", &source[..start], &source[end..]);
    if body.contains("numsim_engine::") {
        return Err(super::super::analyze::util::Failure::Ffi(
            super::super::analyze::util::ffi_error(
                "generated Rust references the engine outside its public ABI import",
            ),
        ));
    }
    Ok(source.to_owned())
}

fn module_exports(request: &ModuleRequest) -> String {
    let names: &[&str] = if !request.analysis_capable {
        &["metadata", "run", "advance_phase"]
    } else if request.analysis_checker.as_deref() == Some("racecheck") {
        &["metadata", "native_racecheck_phase"]
    } else {
        &["metadata", "native_synccheck_phase"]
    };
    names
        .iter()
        .map(|name| format!("    module.add_function(wrap_pyfunction!({name}, module)?)?;"))
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn emit_rust_module(
    ctx: &Ctx,
    plans: &[KernelPlan],
    request: &ModuleRequest,
) -> AResult<String> {
    let options = EmitOptions {
        collect_errors: false,
        analysis_capable: request.analysis_capable,
        split_thresholds: request.split_thresholds.clone(),
    };
    let count = plans.len();
    let mut emitted: Vec<KernelItem> = Vec::new();
    for (index, plan) in plans.iter().enumerate() {
        let mut emitter = Emitter::new(ctx, plan, index as i64, count as i64, &options)?;
        let body = emitter.emit()?;
        emitted.push(render_kernel(&mut emitter, index, count, body, request)?);
    }

    let kernel_structs: Vec<String> = emitted
        .iter()
        .map(|item| {
            let parts: Vec<String> = [
                item.private_metadata_declarations.clone(),
                format!(
                    "#[derive(Clone)]\nstruct Kernel{}Buffers {{\n{}\n}}",
                    item.index, item.fields
                ),
                local_shared_handles_method(item),
            ]
            .into_iter()
            .filter(|part| !part.is_empty())
            .collect();
            parts.join("\n\n")
        })
        .collect();
    let kernel_functions: Vec<String> = emitted
        .iter()
        .map(|item| {
            let index = item.index.to_string();
            let split_prefix = if item.split_helpers.is_empty() {
                String::new()
            } else {
                format!("{}\n\n", item.split_helpers)
            };
            let mut parts = vec![
                fill(
                    PREPARE_TEMPLATE,
                    &[
                        ("split_prefix", &split_prefix),
                        ("prepare_function", &item.prepare_function),
                        ("index", &index),
                        ("setup", &item.setup),
                        ("readonly_proxy", if item.readonly_proxy { "true" } else { "false" }),
                        ("initializers", &item.initializers),
                    ],
                ),
                fill(
                    WARP_TEMPLATE,
                    &[
                        (
                            "warp_optimization",
                            if request.analysis_capable {
                                "#[optimize(size)]"
                            } else {
                                ""
                            },
                        ),
                        ("warp_function", &item.warp_function),
                        ("warp_engine_type", &item.warp_engine_type),
                        ("index", &index),
                        ("body", &item.body),
                    ],
                ),
            ];
            if !request.analysis_capable {
                parts.push(fill(
                    EXECUTE_TEMPLATE,
                    &[
                        ("execute_function", &item.execute_function),
                        ("index", &index),
                        ("setmaxnreg_policy", &setmaxnreg_policy_expression(item)),
                        ("cluster_buffers_setup", cluster_buffers_setup(item)),
                        ("cluster_buffers_select", cluster_buffers_select(item)),
                        ("warp_function", &item.warp_function),
                    ],
                ));
            }
            parts.join("\n\n")
        })
        .collect();
    let kernel_names: Vec<String> = emitted.iter().map(|item| json_string(&item.name)).collect();
    let topology_dimensions: Vec<String> = emitted
        .iter()
        .map(|item| {
            format!(
                "({}_usize, {}_usize, {}_usize)",
                item.clusters, item.ctas_per_cluster, item.warps_per_cta
            )
        })
        .collect();
    let count_text = count.to_string();
    let phase_pairs = |item: &KernelItem| -> Vec<(String, String)> {
        vec![
            ("index".to_owned(), item.index.to_string()),
            ("clusters".to_owned(), item.clusters.to_string()),
            (
                "ctas_per_cluster".to_owned(),
                item.ctas_per_cluster.to_string(),
            ),
            ("warps_per_cta".to_owned(), item.warps_per_cta.to_string()),
            ("kernel_count".to_owned(), count_text.clone()),
            ("prepare_function".to_owned(), item.prepare_function.clone()),
            ("tmem_columns".to_owned(), item.tmem_columns.to_string()),
            ("name".to_owned(), json_string(&item.name)),
            ("execute_function".to_owned(), item.execute_function.clone()),
            (
                "analysis_warp_function".to_owned(),
                item.analysis_warp_function.clone(),
            ),
            (
                "fixed_trace".to_owned(),
                fixed_trace_fast_path_expression(item),
            ),
            (
                "setmaxnreg_policy".to_owned(),
                setmaxnreg_policy_expression(item),
            ),
            (
                "global_memory_model".to_owned(),
                item.global_memory_model_enabled.to_string(),
            ),
            ("racecheck_write_seed".to_owned(), item.racecheck_write_seed.clone()),
            (
                "cluster_buffers_setup".to_owned(),
                cluster_buffers_setup(item).to_owned(),
            ),
            (
                "cluster_buffers_select".to_owned(),
                cluster_buffers_select(item).to_owned(),
            ),
        ]
    };
    let render_phases = |template: &str| -> String {
        emitted
            .iter()
            .map(|item| {
                let pairs = phase_pairs(item);
                let borrowed: Vec<(&str, &str)> = pairs
                    .iter()
                    .map(|(k, v)| (k.as_str(), v.as_str()))
                    .collect();
                fill(template, &borrowed)
            })
            .collect::<Vec<_>>()
            .join("\n")
    };
    let run_phases = render_phases(RUN_PHASE_TEMPLATE);
    let native_sync_check_phases = render_phases(SYNC_PHASE_TEMPLATE);
    let native_race_check_phases = render_phases(RACE_PHASE_TEMPLATE);
    let mut source = fill(
        MODULE_TEMPLATE,
        &[
            ("engine_abi_imports", &import_block()),
            ("exports", &module_exports(request)),
            ("abi_version", &request.abi_version.to_string()),
            (
                "memory_helpers",
                &if !request.analysis_capable {
                    super::memory_support::local_memory_helpers()
                } else {
                    String::new()
                },
            ),
            ("kernel_structs", &kernel_structs.join("\n\n")),
            ("kernel_functions", &kernel_functions.join("\n\n")),
            ("kernel_names", &kernel_names.join(", ")),
            ("topology_dimensions", &topology_dimensions.join(", ")),
            ("kernel_count", &count_text),
            ("run_phases", &run_phases),
            ("native_sync_check_phases", &native_sync_check_phases),
            ("native_race_check_phases", &native_race_check_phases),
        ],
    );
    if !request.analysis_capable {
        source = remove_marked_region(&source, "numsim-analysis-entrypoints");
    } else {
        source = remove_marked_region(&source, "numsim-numeric-entrypoints");
        source = source.replace("ctx.with_active_mask(ctx.active_mask())", "ctx");
        source = remove_marked_region(
            &source,
            if request.analysis_checker.as_deref() == Some("racecheck") {
                "numsim-synccheck-entrypoint"
            } else {
                "numsim-racecheck-entrypoint"
            },
        );
    }
    source = source.replace("WarpValue::from_fn(", "WarpValue::from_fn_copy(");
    render_engine_boundary(&source)
}
