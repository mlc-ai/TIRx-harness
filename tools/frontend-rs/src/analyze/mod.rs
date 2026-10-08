//! Native TIRx frontend analysis.

use std::cell::RefCell;

use tvm::analysis::Analyzer;
use tvm::ir::{Expr, Var};
use tvm::tirx::BufferVar;
use tvm::tvm_ffi::Map;

pub mod buffers;
pub mod cuda_arch;
pub mod expression_bindings;
pub mod frontend;
pub mod layout;
pub mod memory;
pub mod pointer_targets;
pub mod shapes;
pub mod tile_forms;
pub mod topology;
pub mod util;
pub mod vector;

use self::layout::LayoutInfo;
use self::util::{buffer_ref, AResult, IdMap};
use crate::schema::Schema;

/// Per-analysis services: the exported tables, one TVM analyzer, the
/// per-buffer layout inspection cache for one transpile and the
/// decoded call payload cache.
pub struct Ctx<'a> {
    pub schema: &'a Schema,
    pub analyzer: Analyzer,
    layouts: RefCell<IdMap<LayoutInfo>>,
    pub decoding: crate::decode::Cache,
    /// Successful tcgen05.ld/st structural proofs.
    pub tcgen_ldst_validations: RefCell<Vec<tile_forms::TcgenLdstCacheEntry>>,
}

impl<'a> Ctx<'a> {
    pub fn new(schema: &'a Schema) -> AResult<Self> {
        Ok(Self {
            schema,
            analyzer: Analyzer::new()?,
            layouts: RefCell::new(IdMap::default()),
            decoding: crate::decode::Cache::default(),
            tcgen_ldst_validations: RefCell::new(Vec::new()),
        })
    }

    pub fn inspect_layout(
        &self,
        buffer: &BufferVar,
        bindings: &Map<Var, Expr>,
    ) -> AResult<LayoutInfo> {
        let key = buffer_ref(buffer);
        if let Some(cached) = self.layouts.borrow().get(&key) {
            return Ok(cached.clone());
        }
        let info = layout::inspect_buffer_layout(self.schema, &self.analyzer, buffer, bindings)?;
        self.layouts.borrow_mut().insert(key, info.clone());
        Ok(info)
    }
}

/// Operands of a bound binary expression node.
pub fn topology_operands(
    node: &tvm::tvm_ffi::object::ObjectRef,
) -> Option<(
    tvm::tvm_ffi::object::ObjectRef,
    tvm::tvm_ffi::object::ObjectRef,
)> {
    use tvm::prim::{
        AddObj, AndObj, DivObj, EQObj, FloorDivObj, FloorModObj, GEObj, GTObj, LEObj, LTObj,
        MaxObj, MinObj, ModObj, MulObj, NEObj, OrObj, SubObj,
    };
    use tvm::tvm_ffi::ObjectRefCore;
    macro_rules! binary {
        ($($obj:ty),*) => {
            $(if let Some(binary) = node.as_node::<$obj>() {
                return Some((util::oref(binary.a.clone()), util::oref(binary.b.clone())));
            })*
        };
    }
    binary!(
        AddObj,
        SubObj,
        MulObj,
        DivObj,
        ModObj,
        FloorDivObj,
        FloorModObj,
        MinObj,
        MaxObj,
        LTObj,
        LEObj,
        GTObj,
        GEObj,
        EQObj,
        NEObj,
        AndObj,
        OrObj
    );
    None
}
