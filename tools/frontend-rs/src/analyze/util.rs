//! Shared node access, text, identity and JSON helpers for the native analysis.

use std::collections::HashMap;
use std::sync::Arc;
use tvm::ir::TensorRegionObj;

use crate::tvm_compat::int_value;
use tvm::analysis::Analyzer;
use tvm::ir::StringImmObj;
use tvm::ir::{
    CallObj, ExprObj, FloatImmObj, GlobalVarObj, IntImmObj, OpaqueExprObj, PointerTypeObj,
    PrimExpr, PrimTypeObj, SequentialSpanObj, SpanObj, TensorLoadObj, TupleGetItemObj, TupleObj,
    Type, Var, VarObj,
};
use tvm::prim::{
    AddObj, AndObj, BitwiseAndObj, BitwiseNotObj, BitwiseOrObj, BitwiseXorObj, BroadcastObj,
    CastObj, DivObj, EQObj, FloorDivObj, FloorModObj, GEObj, GTObj, LEObj, LShiftObj, LTObj,
    LetObj, MaxObj, MinObj, ModObj, MulObj, NEObj, NotObj, OrObj, RShiftObj, RampObj, SelectObj,
    ShuffleObj, SubObj,
};
use tvm::tirx::{
    AllocBufferObj, AssertStmtObj, AttrStmtObj, BindObj, BreakObj, BufferStoreObj, BufferTypeObj,
    BufferVar, ContinueObj, DeclBufferObj, EvaluateObj, ForObj, IfThenElseObj, ReturnObj,
    ScopeIdDefStmtObj, SeqStmtObj, StmtObj, TilePrimitiveCallObj, WhileObj,
};
use tvm::tvm_ffi::object::ObjectRef;
use tvm::tvm_ffi::{
    self, Any, AnyView, DLDataType, DLDataTypeExt, Error, ObjectIdentity, ObjectRefCast,
    ObjectRefCore, String as FfiString,
};

/// Failure of one native analysis.
#[derive(Debug)]
pub enum Failure {
    /// An emission failure already attached to its source node.
    Recorded,
    /// An input NumSim does not support; raised as `UnsupportedTIRxError`.
    Unsupported {
        message: String,
        unsupported: Vec<String>,
    },
    /// `UnmodeledTIRxFormError`: a valid form NumSim intentionally does not model.
    Unmodeled {
        target: String,
        message: String,
        /// The rejected node's source span, attached by the frontend loop.
        span: Option<Json>,
    },
    /// A call, node or form the frontend has no rule for, or an inconsistency
    /// between its own analysis and emission; raised as `NumSimBuildError`.
    NotCovered(String),
    /// A TVM FFI failure.
    Ffi(Error),
}

impl From<Error> for Failure {
    fn from(error: Error) -> Self {
        Failure::Ffi(error)
    }
}

impl From<std::convert::Infallible> for Failure {
    fn from(never: std::convert::Infallible) -> Self {
        match never {}
    }
}

pub type AResult<T> = std::result::Result<T, Failure>;

pub fn unsupported<T>(message: impl Into<String>) -> AResult<T> {
    Err(Failure::Unsupported {
        message: message.into(),
        unsupported: Vec::new(),
    })
}

pub fn unmodeled<T>(target: impl Into<String>, message: impl Into<String>) -> AResult<T> {
    Err(Failure::Unmodeled {
        target: target.into(),
        message: message.into(),
        span: None,
    })
}

pub fn not_covered<T>(message: impl Into<String>) -> AResult<T> {
    Err(Failure::NotCovered(message.into()))
}

pub fn ffi_error(message: &str) -> Error {
    Error::new(tvm_ffi::VALUE_ERROR, message, "")
}

/// Upcast any object handle to a plain `ObjectRef`.
pub fn oref<T: ObjectRefCast>(value: T) -> ObjectRef {
    value.try_cast::<ObjectRef>().expect("object handle upcast")
}

pub fn ident<T: ObjectRefCore>(value: &T) -> ObjectIdentity {
    ObjectIdentity::of(value)
}

pub fn same<A: ObjectRefCore, B: ObjectRefCore>(lhs: &A, rhs: &B) -> bool {
    ObjectIdentity::of(lhs) == ObjectIdentity::of(rhs)
}

/// An insertion-ordered map keyed by object identity.
/// Scope and partition snapshots share storage until a binding changes.
#[derive(Clone)]
pub struct IdMap<V> {
    index: Arc<HashMap<ObjectIdentity, usize>>,
    entries: Arc<Vec<(ObjectRef, Arc<V>)>>,
}

impl<V> Default for IdMap<V> {
    fn default() -> Self {
        Self {
            index: Arc::default(),
            entries: Arc::default(),
        }
    }
}

impl<V> IdMap<V> {
    pub fn get(&self, key: &ObjectRef) -> Option<&V> {
        self.index
            .get(&ident(key))
            .map(|at| self.entries[*at].1.as_ref())
    }

    pub fn get_mut(&mut self, key: &ObjectRef) -> Option<&mut V>
    where
        V: Clone,
    {
        match self.index.get(&ident(key)) {
            Some(at) => Some(Arc::make_mut(&mut Arc::make_mut(&mut self.entries)[*at].1)),
            None => None,
        }
    }

    pub fn contains(&self, key: &ObjectRef) -> bool {
        self.index.contains_key(&ident(key))
    }

    pub fn insert(&mut self, key: ObjectRef, value: V) -> bool {
        let identity = ident(&key);
        if self.index.contains_key(&identity) {
            return false;
        }
        Arc::make_mut(&mut self.index).insert(identity, self.entries.len());
        Arc::make_mut(&mut self.entries).push((key, Arc::new(value)));
        true
    }

    /// Replace a value without changing its traversal position.
    pub fn set(&mut self, key: ObjectRef, value: V) {
        if let Some(&at) = self.index.get(&ident(&key)) {
            Arc::make_mut(&mut self.entries)[at].1 = Arc::new(value);
        } else {
            self.insert(key, value);
        }
    }

    pub fn keys(&self) -> Vec<ObjectRef> {
        self.entries.iter().map(|(key, _)| key.clone()).collect()
    }

    pub fn equals(&self, other: &Self) -> bool
    where
        V: PartialEq,
    {
        self.entries.len() == other.entries.len()
            && self
                .entries
                .iter()
                .all(|(key, value)| other.get(key) == Some(value.as_ref()))
    }

    pub fn remove(&mut self, key: &ObjectRef) -> Option<V>
    where
        V: Clone,
    {
        let at = Arc::make_mut(&mut self.index).remove(&ident(key))?;
        let (_, value) = Arc::make_mut(&mut self.entries).remove(at);
        self.index = Arc::new(
            self.entries
                .iter()
                .enumerate()
                .map(|(position, (entry, _))| (ident(entry), position))
                .collect(),
        );
        Some(Arc::unwrap_or_clone(value))
    }
}

pub type IdSet = IdMap<()>;

impl IdSet {
    pub fn add(&mut self, key: ObjectRef) -> bool {
        self.insert(key, ())
    }
}

/// `type(node).__name__` of the Python wrapper for a bound TVM node.
pub fn kind(node: &ObjectRef) -> Option<&'static str> {
    macro_rules! kinds {
        ($($obj:ty => $name:literal),* $(,)?) => {
            $(if node.as_node::<$obj>().is_some() { return Some($name); })*
        };
    }
    kinds!(
        IntImmObj => "IntImm",
        FloatImmObj => "FloatImm",
        VarObj => "Var",
        CallObj => "Call",
        TensorLoadObj => "TensorLoad",
        StringImmObj => "StringImm",
        BitwiseAndObj => "BitwiseAnd",
        BitwiseOrObj => "BitwiseOr",
        BitwiseXorObj => "BitwiseXor",
        BitwiseNotObj => "BitwiseNot",
        LShiftObj => "LShift",
        RShiftObj => "RShift",
        AddObj => "Add",
        SubObj => "Sub",
        MulObj => "Mul",
        DivObj => "Div",
        ModObj => "Mod",
        FloorDivObj => "FloorDiv",
        FloorModObj => "FloorMod",
        MinObj => "Min",
        MaxObj => "Max",
        EQObj => "EQ",
        NEObj => "NE",
        LTObj => "LT",
        LEObj => "LE",
        GTObj => "GT",
        GEObj => "GE",
        AndObj => "And",
        OrObj => "Or",
        NotObj => "Not",
        SelectObj => "Select",
        CastObj => "Cast",
        LetObj => "Let",
        RampObj => "Ramp",
        BroadcastObj => "Broadcast",
        ShuffleObj => "Shuffle",
        TupleObj => "Tuple",
        TupleGetItemObj => "TupleGetItem",
        GlobalVarObj => "GlobalVar",
        OpaqueExprObj => "OpaqueExpr",
        SeqStmtObj => "SeqStmt",
        AttrStmtObj => "AttrStmt",
        BindObj => "Bind",
        AllocBufferObj => "AllocBuffer",
        DeclBufferObj => "DeclBuffer",
        ScopeIdDefStmtObj => "ScopeIdDefStmt",
        TilePrimitiveCallObj => "TilePrimitiveCall",
        EvaluateObj => "Evaluate",
        IfThenElseObj => "IfThenElse",
        AssertStmtObj => "AssertStmt",
        ForObj => "For",
        WhileObj => "While",
        BreakObj => "Break",
        ContinueObj => "Continue",
        ReturnObj => "Return",
        BufferStoreObj => "BufferStore",
        TensorRegionObj => "TensorRegion",
    );
    None
}

/// Operands of TVM's primitive bitwise nodes, in evaluation order.
pub fn bitwise_expr(node: &ObjectRef) -> Option<(&'static str, Vec<ObjectRef>)> {
    macro_rules! binary {
        ($ty:ty, $name:literal) => {
            if let Some(value) = node.as_node::<$ty>() {
                return Some(($name, vec![oref(value.a.clone()), oref(value.b.clone())]));
            }
        };
    }
    binary!(BitwiseAndObj, "bitwise_and");
    binary!(BitwiseOrObj, "bitwise_or");
    binary!(BitwiseXorObj, "bitwise_xor");
    binary!(LShiftObj, "shift_left");
    binary!(RShiftObj, "shift_right");
    node.as_node::<BitwiseNotObj>()
        .map(|value| ("bitwise_not", vec![oref(value.a.clone())]))
}

pub fn kind_or_bail(node: &ObjectRef) -> AResult<&'static str> {
    match kind(node) {
        Some(name) => Ok(name),
        None => not_covered(format!(
            "node kind without a native binding: {}",
            runtime_kind(node)?
        )),
    }
}

/// The leaf name of the node's canonical runtime type key.
pub fn runtime_kind(node: &ObjectRef) -> AResult<String> {
    let type_index = AnyView::from(node).type_index();
    let info = unsafe { tvm_ffi::tvm_ffi_sys::TVMFFIGetTypeInfo(type_index) };
    if info.is_null() {
        return not_covered(format!(
            "node runtime type index {type_index} has no registered type info"
        ));
    }
    let type_key = unsafe { (*info).type_key.as_str() };
    Ok(type_key.rsplit('.').next().unwrap_or(type_key).to_owned())
}

pub fn dtype_text(dtype: DLDataType) -> String {
    DLDataTypeExt::to_string(&dtype).as_str().to_owned()
}

pub fn ffi_text(value: &FfiString) -> String {
    value.as_str().to_owned()
}

/// `str(node)` for a TVM object: the registered repr printer.
pub fn repr_text(node: &ObjectRef) -> AResult<String> {
    let printed: Any =
        tvm_ffi::cached_global_func!("ffi.ReprPrint").call_tuple((Any::from(node.clone()),))?;
    Ok(ffi_text(&FfiString::try_from(printed)?))
}

pub fn repr_of<T: ObjectRefCast + Clone>(value: &T) -> AResult<String> {
    repr_text(&oref(value.clone()))
}

pub fn expr_type(node: &ObjectRef) -> Option<Type> {
    node.as_node::<ExprObj>().map(|expr| expr.ty.clone())
}

pub fn is_pointer_type(ty: &Type) -> bool {
    ty.as_node::<PointerTypeObj>().is_some()
}

pub fn prim_dtype(ty: &Type) -> Option<DLDataType> {
    ty.as_node::<PrimTypeObj>().map(|prim| prim.dtype)
}

/// `"handle"` for pointers, else `str(value.ty.dtype)`.
pub fn dtype_of(node: &ObjectRef) -> AResult<String> {
    // String immediates are metadata operands.  TVM now gives them the
    // shared ir.StringType; NumSim's call signatures have always represented
    // that non-numeric operand with the empty dtype.
    if node.as_node::<StringImmObj>().is_some() {
        return Ok(String::new());
    }
    let Some(ty) = expr_type(node) else {
        return not_covered("dtype_of on a value without a type");
    };
    if is_pointer_type(&ty) {
        return Ok("handle".to_owned());
    }
    if let Some(dtype) = prim_dtype(&ty) {
        return Ok(dtype_text(dtype));
    }
    if let Some(buffer) = ty.as_node::<BufferTypeObj>() {
        // Python renders the PrimType object itself here.
        return repr_of(&buffer.dtype);
    }
    not_covered("dtype_of on an expression with an opaque type")
}

pub fn as_var(node: &ObjectRef) -> Option<Var> {
    node.clone().try_cast::<Var>().ok()
}

pub fn as_buffer(node: &ObjectRef) -> Option<BufferVar> {
    let var = as_var(node)?;
    BufferVar::try_from(&var).ok()
}

pub fn buffer_name(buffer: &BufferVar) -> String {
    ffi_text(&buffer.as_var().name)
}

pub fn buffer_dtype(buffer: &BufferVar) -> String {
    dtype_text(buffer.buffer_type().dtype.dtype)
}

pub fn buffer_scope(buffer: &BufferVar) -> String {
    ffi_text(&buffer.buffer_type().storage_scope)
}

pub fn buffer_ref(buffer: &BufferVar) -> ObjectRef {
    oref(buffer.as_var().clone())
}

pub fn prim(node: &ObjectRef) -> AResult<PrimExpr> {
    Ok(PrimExpr::try_from(Any::from(node.clone()))?)
}

pub fn int_imm(node: &ObjectRef) -> Option<i64> {
    node.as_node::<IntImmObj>()
        .and_then(|imm| int_value(imm).ok())
}

pub fn int_imm_expr(expr: &PrimExpr) -> Option<i64> {
    expr.as_node::<IntImmObj>()
        .and_then(|imm| int_value(imm).ok())
}

pub fn simplify(analyzer: &Analyzer, expr: &PrimExpr) -> AResult<PrimExpr> {
    Ok(analyzer.simplify(expr)?)
}

pub fn static_int(
    analyzer: &Analyzer,
    expr: &PrimExpr,
    field: &str,
    requirement: &str,
) -> AResult<i64> {
    let simplified = simplify(analyzer, expr)?;
    match int_imm_expr(&simplified) {
        Some(value) => Ok(value),
        None => unsupported(format!("{field} {requirement}, got {}", repr_of(expr)?)),
    }
}

pub fn static_string(value: &ObjectRef, field: &str) -> AResult<String> {
    match value.as_node::<StringImmObj>() {
        Some(imm) => Ok(ffi_text(&imm.value)),
        None => unsupported(format!(
            "{field} must be a static string, got {}",
            repr_text(value)?
        )),
    }
}

/// Source span JSON as `source_map.source_span_from_tvm` serializes it.
pub fn span_json(span: Option<&tvm::ir::Span>) -> Option<Json> {
    let span = span?;
    if let Some(sequential) = span.as_node::<SequentialSpanObj>() {
        let mut children = Vec::new();
        for child in sequential.spans.iter() {
            if let Some(child_json) = span_json(Some(&child)) {
                children.push(child_json);
            }
        }
        if children.is_empty() {
            return None;
        }
        return Some(json_object(vec![
            ("kind", Json::from("sequential")),
            ("spans", Json::Array(children)),
        ]));
    }
    let leaf = span.as_node::<SpanObj>()?;
    let source_name = ffi_text(&leaf.source_name.as_ref()?.name);
    if source_name.is_empty() {
        return None;
    }
    let coordinates = [leaf.line, leaf.column, leaf.end_line, leaf.end_column];
    if coordinates.iter().any(|value| *value < 1) {
        return None;
    }
    if leaf.end_line < leaf.line || (leaf.end_line == leaf.line && leaf.end_column < leaf.column) {
        return None;
    }
    Some(json_object(vec![
        ("kind", Json::from("span")),
        ("source_name", Json::String(source_name)),
        ("line", Json::from(leaf.line as i64)),
        ("column", Json::from(leaf.column as i64)),
        ("end_line", Json::from(leaf.end_line as i64)),
        ("end_column", Json::from(leaf.end_column as i64)),
    ]))
}

pub fn node_span(node: &ObjectRef) -> Option<tvm::ir::Span> {
    if let Some(expr) = node.as_node::<ExprObj>() {
        return expr.span.clone();
    }
    node.as_node::<StmtObj>().and_then(|stmt| stmt.span.clone())
}

/// Python `str.capitalize()`.
pub fn capitalize(text: &str) -> String {
    let mut characters = text.chars();
    match characters.next() {
        Some(first) => first
            .to_uppercase()
            .chain(characters.flat_map(char::to_lowercase))
            .collect(),
        None => String::new(),
    }
}

/// `text` with its first character upper-cased and the rest unchanged.
pub fn upper_first(text: &str) -> String {
    let mut characters = text.chars();
    match characters.next() {
        Some(first) => first.to_uppercase().chain(characters).collect(),
        None => String::new(),
    }
}

pub use serde_json::Value as Json;

pub fn json_object(fields: Vec<(&str, Json)>) -> Json {
    Json::Object(
        fields
            .into_iter()
            .map(|(key, value)| (key.to_owned(), value))
            .collect(),
    )
}

pub fn json_strings(values: impl IntoIterator<Item = String>) -> Json {
    Json::Array(values.into_iter().map(Json::String).collect())
}

/// Sorted, de-duplicated string collection (Python `sorted(set(...))`).
pub fn sorted_unique<I: IntoIterator<Item = String>>(values: I) -> Vec<String> {
    let mut items: Vec<String> = values.into_iter().collect();
    items.sort();
    items.dedup();
    items
}
