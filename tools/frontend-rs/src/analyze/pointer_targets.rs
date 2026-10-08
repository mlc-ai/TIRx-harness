//! Flow-insensitive candidate roots for the global write seed. Opaque writes
//! keep the full seed. Runtime validation catches raw address arithmetic that
//! escapes a candidate allocation and repeats with the complete seed.

use tvm::tirx::TensorMapTypeObj;

use std::collections::BTreeSet;
use tvm::ir::TensorRegionObj;
use tvm::ir::{CallObj, PointerTypeObj, TensorLoadObj};
use tvm::prim::{AddObj, CastObj, LetObj, SelectObj, SubObj};
use tvm::tirx::{BindObj, BufferStoreObj, BufferVar};
use tvm::tvm_ffi::{object::ObjectRef, ObjectRefCore};

use super::frontend::KernelPlan;
use super::util::{
    as_buffer, as_var, dtype_of, ffi_text, int_imm, oref, same, AResult, IdMap, IdSet,
};
use super::Ctx;
use crate::decode::pointers::{
    address_operand, pointer_call, register_pointer_sources, PointerCall,
};
use crate::decode::{projected_buffer, ptx::PtxDecode};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Target {
    Backing(usize),
    Pointer(usize),
}

#[derive(Clone, PartialEq, Eq)]
enum Fact {
    Bottom,
    Number,
    Targets(BTreeSet<Target>),
    Unknown,
}

impl Fact {
    fn target(target: Target) -> Self {
        Self::Targets([target].into())
    }

    fn join(self, other: Self) -> Self {
        match (self, other) {
            (Self::Bottom, fact) | (fact, Self::Bottom) => fact,
            (Self::Number, Self::Number) => Self::Number,
            (Self::Targets(mut a), Self::Targets(b)) => {
                a.extend(b);
                Self::Targets(a)
            }
            _ => Self::Unknown,
        }
    }

    fn offset(self, other: Self, subtract: bool) -> Self {
        match (self, other) {
            (Self::Unknown, _) | (_, Self::Unknown) => Self::Unknown,
            (Self::Bottom, _) | (_, Self::Bottom) => Self::Bottom,
            (target @ Self::Targets(_), Self::Number) => target,
            (Self::Number, target @ Self::Targets(_)) if !subtract => target,
            (Self::Number, Self::Number) => Self::Number,
            _ => Self::Unknown,
        }
    }

    fn invalid(self) -> Self {
        match self {
            Self::Targets(_) | Self::Unknown => Self::Unknown,
            _ => Self::Number,
        }
    }
}

pub struct PointerTargets<'a> {
    plan: &'a KernelPlan,
    bindings: IdMap<ObjectRef>,
    facts: IdMap<Fact>,
}

impl<'a> PointerTargets<'a> {
    pub fn new(ctx: &Ctx, plan: &'a KernelPlan) -> AResult<Self> {
        let mut result = Self {
            plan,
            bindings: IdMap::default(),
            facts: IdMap::default(),
        };
        let mut writes: IdMap<Vec<Vec<ObjectRef>>> = IdMap::default();
        let mut escaped = IdSet::default();
        for buffer in &plan.memory_plan.as_ref().expect("memory plan").buffers {
            let key = plan.buffer_bindings.storage_key(&buffer.buffer)?;
            let initial = if buffer.is_parameter {
                Fact::Number
            } else {
                Fact::Bottom
            };
            let previous = result.facts.get(&key).cloned().unwrap_or(Fact::Bottom);
            result.facts.set(key.clone(), previous.join(initial));
            writes.insert(key, Vec::new());
        }
        for node in &plan.nodes {
            if let Some(bind) = node.as_node::<BindObj>() {
                result
                    .bindings
                    .set(oref(bind.var.clone()), oref(bind.value.clone()));
            }
            if let Some(store) = node.as_node::<BufferStoreObj>() {
                let key = plan.buffer_bindings.storage_key(&store.buffer)?;
                if let Some(defs) = writes.get_mut(&key) {
                    defs.push(vec![oref(store.value.clone())]);
                }
            }
            // Address-exposed register storage can be changed by indirect stores
            // or helpers. Do not infer its contents from direct stores alone.
            if let Some(buffer) = projected_buffer(node)? {
                escaped.add(plan.buffer_bindings.storage_key(&buffer)?);
            }
            // Tile destinations can overwrite storage without a BufferStore
            // or explicit PTX register destination in the source IR.
            if let Some(region) = node.as_node::<TensorRegionObj>() {
                escaped.add(
                    plan.buffer_bindings
                        .storage_key(&BufferVar::try_from(region.source.clone())?)?,
                );
            }
            if let Some(call) = node.as_node::<CallObj>() {
                if let Some(value) = address_operand(call)? {
                    if let Some(load) = value.as_node::<TensorLoadObj>() {
                        if let Some(buffer) = as_buffer(&oref(load.source.clone())) {
                            escaped.add(plan.buffer_bindings.storage_key(&buffer)?);
                        }
                    }
                }
            }
            let Some(PtxDecode::Decoded(decoded)) = ctx.decoding.ptx_decode(node)? else {
                continue;
            };
            let sources = register_pointer_sources(&decoded)?;
            for slot in &decoded.operands {
                if slot.kind != "reg" || !slot.rw.contains('w') {
                    continue;
                }
                for destination in slot.values.iter().flatten() {
                    let Some(load) = destination.as_node::<TensorLoadObj>() else {
                        continue;
                    };
                    let Some(buffer) = as_buffer(&oref(load.source.clone())) else {
                        continue;
                    };
                    let key = plan.buffer_bindings.storage_key(&buffer)?;
                    if let Some(defs) = writes.get_mut(&key) {
                        defs.push(sources.clone());
                    }
                }
            }
        }
        for key in escaped.keys() {
            if result.facts.get(&key) == Some(&Fact::Bottom) {
                result.facts.set(key, Fact::Unknown);
            }
        }
        // Finite target sets grow monotonically, including loop-carried and
        // predicated definitions. No control-flow path is discarded.
        loop {
            let mut next = result.facts.clone();
            for key in writes.keys() {
                let mut fact = result.facts.get(&key).expect("storage fact").clone();
                for sources in writes.get(&key).expect("storage writes") {
                    let mut value = if sources.is_empty() {
                        Fact::Number
                    } else {
                        Fact::Bottom
                    };
                    for source in sources {
                        if int_imm(source) != Some(0) {
                            value = value.join(result.expr(source, &mut IdSet::default())?);
                        }
                    }
                    fact = fact.join(value);
                }
                next.set(key, fact);
            }
            if next.equals(&result.facts) {
                break;
            }
            result.facts = next;
        }
        Ok(result)
    }

    fn buffer(&self, buffer: &BufferVar, seen: &mut IdSet) -> AResult<Fact> {
        let plan = self
            .plan
            .memory_plan
            .as_ref()
            .expect("memory plan")
            .resolve(buffer)?;
        Ok(if let Some(data) = &plan.dynamic_data_var {
            self.expr(&oref(data.clone()), seen)?
        } else if let Some(index) = plan.backing_index {
            Fact::target(Target::Backing(index))
        } else {
            Fact::Unknown
        })
    }

    fn expr(&self, node: &ObjectRef, seen: &mut IdSet) -> AResult<Fact> {
        if !seen.add(node.clone()) {
            return Ok(Fact::Unknown);
        }
        let value = self.expr_inner(node, seen);
        seen.remove(node);
        value
    }

    fn expr_inner(&self, node: &ObjectRef, seen: &mut IdSet) -> AResult<Fact> {
        if let Some(buffer) = projected_buffer(node)? {
            return self.buffer(&buffer, seen);
        }
        if let Some(load) = node.as_node::<TensorLoadObj>() {
            if !matches!(dtype_of(node)?.as_str(), "handle" | "uint64" | "uint32") {
                return Ok(Fact::Number);
            }
            let Some(buffer) = as_buffer(&oref(load.source.clone())) else {
                return Ok(Fact::Unknown);
            };
            let key = self.plan.buffer_bindings.storage_key(&buffer)?;
            return Ok(self.facts.get(&key).cloned().unwrap_or(Fact::Unknown));
        }
        if let Some(var) = as_var(node) {
            if let Some(value) = self.bindings.get(node) {
                return self.expr(value, seen);
            }
            if let Some(index) = self.plan.func.params.iter().position(|p| same(&p, &var)) {
                if let Some(pointer) = var.ty.as_node::<PointerTypeObj>() {
                    if pointer.element_type.as_node::<TensorMapTypeObj>().is_some() {
                        return Ok(Fact::Unknown);
                    }
                    if ffi_text(&pointer.storage_scope) == "global" {
                        return Ok(Fact::target(Target::Pointer(index)));
                    }
                }
            }
            return Ok(if matches!(dtype_of(node)?.as_str(), "handle" | "uint64") {
                Fact::Unknown
            } else {
                Fact::Number
            });
        }
        if let Some(add) = node.as_node::<AddObj>() {
            return Ok(self
                .expr(&oref(add.a.clone()), seen)?
                .offset(self.expr(&oref(add.b.clone()), seen)?, false));
        }
        if let Some(sub) = node.as_node::<SubObj>() {
            return Ok(self
                .expr(&oref(sub.a.clone()), seen)?
                .offset(self.expr(&oref(sub.b.clone()), seen)?, true));
        }
        if let Some(cast) = node.as_node::<CastObj>() {
            return Ok(self.expr(&oref(cast.value.clone()), seen)?.invalid());
        }
        if let Some(select) = node.as_node::<SelectObj>() {
            return Ok(self
                .expr(&oref(select.true_value.clone()), seen)?
                .join(self.expr(&oref(select.false_value.clone()), seen)?));
        }
        if node.as_node::<LetObj>().is_some() {
            // Uncommon expression-local bindings are conservatively unresolved.
            return Ok(Fact::Unknown);
        }
        if let Some(call) = node.as_node::<CallObj>() {
            match pointer_call(call)? {
                PointerCall::Address(value) => {
                    if let Some(load) = value.as_node::<TensorLoadObj>() {
                        if let Some(buffer) = as_buffer(&oref(load.source.clone())) {
                            return self.buffer(&buffer, seen);
                        }
                    }
                    return self.expr(&value, seen);
                }
                PointerCall::Forward(value) => return self.expr(&value, seen),
                PointerCall::Select(a, b) => {
                    return Ok(self.expr(&a, seen)?.join(self.expr(&b, seen)?));
                }
                PointerCall::Opaque => {}
            }
            // Opaque calls may return arbitrary pointer bits. Their arguments
            // can also include non-expression metadata such as symbol names.
            return Ok(if matches!(dtype_of(node)?.as_str(), "handle" | "uint64") {
                Fact::Unknown
            } else {
                Fact::Number
            });
        }
        if let Some((a, b)) = super::topology_operands(node) {
            return Ok(self
                .expr(&a, seen)?
                .invalid()
                .join(self.expr(&b, seen)?.invalid()));
        }
        Ok(if dtype_of(node)? == "handle" {
            Fact::Unknown
        } else {
            Fact::Number
        })
    }

    pub fn resolve(&self, node: &ObjectRef) -> AResult<Option<BTreeSet<Target>>> {
        Ok(match self.expr(node, &mut IdSet::default())? {
            Fact::Targets(targets) => Some(targets),
            _ => None,
        })
    }
}
