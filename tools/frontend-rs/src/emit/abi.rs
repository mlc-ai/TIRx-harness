//! Mechanical rendering of exact engine ABI calls.

use crate::tables::json_string;

pub const WARP: &str = "warp";

pub fn call(
    function: &str,
    arguments: &[String],
    generics: &[String],
    propagate: bool,
    await_result: bool,
) -> String {
    let generic_suffix = if generics.is_empty() {
        String::new()
    } else {
        format!("::<{}>", generics.join(", "))
    };
    let mut suffix = if await_result { ".await" } else { "" }.to_owned();
    if propagate {
        suffix.push('?');
    }
    format!(
        "v2::{function}{generic_suffix}({}){suffix}",
        arguments.join(", ")
    )
}

pub fn site_expr(expression: &str) -> String {
    format!("v2::SiteId::new({expression})")
}

pub fn site(source_op_id: u64) -> String {
    site_expr(&format!("{source_op_id}_u64"))
}

pub fn lane_mask(mask: &str) -> String {
    format!("v2::LaneMask::from_bits(({mask}).bits())")
}

pub fn context(context: &str) -> String {
    format!("v2_context({context})")
}

/// `(warp, ctx, site)` call.
pub fn warp_call(
    function: &str,
    site: &str,
    arguments: &[String],
    variant: Option<&str>,
    context_code: Option<&str>,
    await_result: bool,
    propagate: bool,
) -> String {
    let mut all = vec![
        WARP.to_owned(),
        context_code.map_or_else(|| context("ctx"), str::to_owned),
        site.to_owned(),
    ];
    all.extend(arguments.iter().cloned());
    let generics: Vec<String> = variant.map(str::to_owned).into_iter().collect();
    call(function, &all, &generics, propagate, await_result)
}

pub fn buffer_address(
    space: &str,
    buffer_ref: &str,
    index: &str,
    itemsize: i64,
    buffer_name: &str,
) -> String {
    format!(
        "v2_buffer_address::<{space}>(&{buffer_ref}, &{index}, {itemsize}_usize, {})",
        json_string(buffer_name)
    )
}

/// `(ctx, site)` call: the register family takes no warp.
pub fn lane_call(
    function: &str,
    site: &str,
    arguments: &[String],
    variant: Option<&str>,
) -> String {
    let mut all = vec![context("ctx"), site.to_owned()];
    all.extend(arguments.iter().cloned());
    let generics: Vec<String> = variant.map(str::to_owned).into_iter().collect();
    call(function, &all, &generics, true, false)
}

/// `lane_call` with a caller-supplied context.
pub fn lane_call_context(
    function: &str,
    site: &str,
    arguments: &[String],
    variant: Option<&str>,
    context_code: Option<&str>,
) -> String {
    let mut all = vec![
        context_code.map_or_else(|| context("ctx"), str::to_owned),
        site.to_owned(),
    ];
    all.extend(arguments.iter().cloned());
    let generics: Vec<String> = variant.map(str::to_owned).into_iter().collect();
    call(function, &all, &generics, true, false)
}

pub fn cloned(expression: &str) -> String {
    format!("({expression}).clone()")
}

pub fn address(space: &str, pointer: &str, logical_buffer: Option<&str>) -> String {
    match logical_buffer {
        Some(name) => format!(
            "v2_named_address::<{space}>({pointer}, {})",
            json_string(name)
        ),
        None => format!("v2_address::<{space}>({pointer})"),
    }
}

/// Render a warp-register operand: every site already cloned the value in.
pub fn register(expression: &str) -> String {
    format!("v2_register({})", cloned(expression))
}

/// Render a register built by evaluating `body` once per lane.
pub fn per_lane(body: &str) -> String {
    format!("v2::R::from_fn(|lane| {body})")
}

pub fn splat(value: &str) -> String {
    format!("v2::R::splat({value})")
}

pub fn physical_ptr(base: &str, index: &str, itemsize: i64) -> String {
    format!(
        "PhysicalPtr::new({}, {}, {itemsize}_usize)",
        cloned(base),
        cloned(index)
    )
}

pub(super) fn element_ref(space: &str, constructor: &str, arguments: &[String]) -> String {
    call(
        &format!("ElementRef::<{space}>::{constructor}"),
        arguments,
        &[],
        false,
        false,
    )
}

pub(super) fn named_buffer(space: &str, buffer_ref: &str, buffer_name: &str) -> String {
    format!(
        "v2_named_buffer::<{space}>({}, {})",
        cloned(buffer_ref),
        json_string(buffer_name)
    )
}

pub(super) const FN_MAP_STATE: &str = "()";

fn dims(dimensions: &[i64]) -> String {
    dimensions
        .iter()
        .map(|dimension| format!("{dimension}_usize"))
        .collect::<Vec<_>>()
        .join(", ")
}

pub(super) fn table_state(dimensions: &[i64], table: &str) -> String {
    format!("(vec![{}], {table})", dims(dimensions))
}

pub(super) fn empty_table_state(dimensions: &[i64], space: &str) -> String {
    format!(
        "(vec![{}], Vec::<v2::ElementRef<{space}>>::new())",
        dims(dimensions)
    )
}

pub(super) fn mapped_view(
    name: &str,
    space: &str,
    buffer_ref: &str,
    buffer_name: &str,
    mapper: &str,
    state: &str,
) -> String {
    format!(
        "let {name} = v2::MappedView::new({}, {mapper}, {state});",
        named_buffer(space, buffer_ref, buffer_name)
    )
}
