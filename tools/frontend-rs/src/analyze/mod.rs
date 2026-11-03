//! Native TIRx frontend analysis.

use std::cell::RefCell;

use tvm_tirx::sym::Analyzer;
use tvm_tirx::ir::{Expr, Var};

use tvm_tirx::tvm_ffi::Map;

pub mod buffers;
pub mod cuda_arch;
pub mod expression_bindings;
pub mod frontend;
pub mod layout;
pub mod memory;
pub mod pointer_targets;
pub mod shapes;
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
    /// TMEM column addresses, from the bindings that produce each tensor.
    tmem_addresses: RefCell<IdMap<Expr>>,
    /// Byte offsets of declared views from their storage roots.
    view_byte_offsets: RefCell<IdMap<Expr>>,
    pub decoding: crate::decode::Cache,
}

impl<'a> Ctx<'a> {
    pub fn new(schema: &'a Schema) -> AResult<Self> {
        Ok(Self {
            schema,
            analyzer: Analyzer::new(),
            layouts: RefCell::new(IdMap::default()),
            tmem_addresses: RefCell::new(IdMap::default()),
            view_byte_offsets: RefCell::new(IdMap::default()),
            decoding: crate::decode::Cache::default(),
        })
    }

    pub fn inspect_layout(
        &self,
        buffer: &Var,
        bindings: &Map<Var, Expr>,
    ) -> AResult<LayoutInfo> {
        let key = buffer_ref(buffer);
        if let Some(cached) = self.layouts.borrow().get(&key) {
            return Ok(cached.clone());
        }
        let tmem_address = self.tmem_addresses.borrow().get(&key).cloned();
        let byte_offset = self.view_byte_offsets.borrow().get(&key).cloned();
        let info = layout::inspect_buffer_layout(
            self.schema,
            &self.analyzer,
            buffer,
            bindings,
            tmem_address,
            byte_offset,
        )?;
        self.layouts.borrow_mut().insert(key, info.clone());
        Ok(info)
    }

    /// The TMEM column address the declaration of `tensor` gives it.
    pub fn tmem_address(&self, tensor: &Var) -> Option<Expr> {
        self.tmem_addresses.borrow().get(&buffer_ref(tensor)).cloned()
    }

    /// Record the TMEM column addresses and view byte offsets `bindings` declare,
    /// for layout inspection.
    pub fn record_tensor_decls(&self, bindings: &buffers::BufferBindings) -> AResult<()> {
        let mut addresses = self.tmem_addresses.borrow_mut();
        for (tensor, address) in &bindings.tmem_addresses {
            addresses.insert(buffer_ref(tensor), address.clone());
        }
        let mut offsets = self.view_byte_offsets.borrow_mut();
        for (view, _) in &bindings.declarations {
            if let Some(offset) = bindings.root_byte_offset(view)? {
                offsets.insert(buffer_ref(view), offset);
            }
        }
        Ok(())
    }
}

/// Operands of a bound binary expression node.
pub fn topology_operands(
    node: &tvm_tirx::tvm_ffi::object::ObjectRef,
) -> Option<(
    tvm_tirx::tvm_ffi::object::ObjectRef,
    tvm_tirx::tvm_ffi::object::ObjectRef,
)> {
    use tvm_tirx::ir::prim::{
        AddObj, AndObj, DivObj, EQObj, FloorDivObj, FloorModObj, GEObj, GTObj, LEObj, LTObj, MaxObj,
        MinObj, ModObj, MulObj, NEObj, OrObj, SubObj,
    };
    use tvm_tirx::tvm_ffi::ObjectRefCore;
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
