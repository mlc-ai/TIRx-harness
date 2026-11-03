//! Shared node access, text, identity and JSON helpers for the native analysis.

use std::collections::HashMap;
use std::sync::Arc;
use tvm_tirx::ir::TensorRegionObj;

use crate::tvm_compat::int_value;
use tvm_tirx::sym::Analyzer;
use tvm_tirx::ir::StringImmObj;
use tvm_tirx::ir::{
    CallObj, CallSiteLocObj, Expr, ExprObj, FloatImm, FloatImmObj, GlobalVarObj, IntImm, IntImmObj,
    Location, OpaqueExprObj, PrimType, PrimTypeObj, PtrTypeObj, Range, SourceLocObj, TensorLoadObj,
    TupleGetItemObj, TupleObj, Type, Var, VarObj,
};
use tvm_tirx::ir::prim::{
    AddObj, AndObj, BitwiseAndObj, BitwiseNotObj, BitwiseOrObj, BitwiseXorObj, BroadcastObj,
    CastObj, DivObj, EQObj, FloorDivObj, FloorModObj, GEObj, GTObj, LEObj, LShiftObj, LTObj, LetObj,
    MaxObj, MinObj, ModObj, MulObj, NEObj, NotObj, OrObj, RShiftObj, RampObj, SelectObj, ShuffleObj,
    SubObj,
};
use tvm_tirx::ir::{
    AssertStmtObj, BindObj, BreakObj, ContinueObj, EvaluateObj, ForObj, IfObj, ReturnObj,
    SeqStmtObj, Stmt, StmtObj, TensorStoreObj, WhileObj,
};
use tvm_tirx::tirx::Function;
use tvm_tirx::tirx::{
    AllocBufferObj,
    AttrStmtObj,
    DeclBufferObj,
    ScopeIdDefStmtObj,
    TensorTypeObj,
    TilePrimitiveCallObj,
};
use tvm_tirx::tvm_ffi::object::ObjectRef;
use tvm_tirx::tvm_ffi::{
    self, Any, AnyView, DLDataType, DLDataTypeExt, Error, FieldGetter, ObjectCore, ObjectIdentity,
    ObjectRefCast, ObjectRefCore, String as FfiString,
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
        IfObj => "IfThenElse",
        AssertStmtObj => "AssertStmt",
        ForObj => "For",
        WhileObj => "While",
        BreakObj => "Break",
        ContinueObj => "Continue",
        ReturnObj => "Return",
        TensorStoreObj => "BufferStore",
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
    ty.as_node::<PtrTypeObj>().is_some()
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
    if let Some(buffer) = ty.as_node::<TensorTypeObj>() {
        // Python renders the PrimType object itself here.
        return repr_of(&buffer.dtype);
    }
    not_covered("dtype_of on an expression with an opaque type")
}

pub fn as_var(node: &ObjectRef) -> Option<Var> {
    node.clone().try_cast::<Var>().ok()
}

/// Tensor accessors of a `Var` whose type is a `TensorType` (C++ `tirx::TensorVar`).
pub trait TensorVarExt {
    /// The tensor type; panics on a variable of another type.
    fn tensor_type(&self) -> &TensorTypeObj;
}

impl TensorVarExt for Var {
    fn tensor_type(&self) -> &TensorTypeObj {
        self.ty
            .as_node::<TensorTypeObj>()
            .expect("tensor variable type invariant was violated")
    }
}

/// Whether `var` is a tensor variable: a `Var` whose type is a `TensorType`.
pub fn is_tensor_var(var: &Var) -> bool {
    var.ty.as_node::<TensorTypeObj>().is_some()
}

/// The body of a function definition. A declaration has none, so it cannot be analyzed.
pub fn function_body(func: &Function) -> AResult<Stmt> {
    match &func.body {
        Some(body) => Ok(body.clone().into()),
        None => unsupported("a function declaration has no body to analyze"),
    }
}

pub fn as_buffer(node: &ObjectRef) -> Option<Var> {
    as_var(node).filter(is_tensor_var)
}

/// `node` as a tensor variable, or a type error.
pub fn tensor_var(node: &ObjectRef) -> AResult<Var> {
    as_buffer(node).ok_or_else(|| {
        Failure::Ffi(Error::new(
            tvm_ffi::TYPE_ERROR,
            "expected a Var whose type is a TensorType",
            "",
        ))
    })
}

pub fn buffer_name(buffer: &Var) -> String {
    ffi_text(&buffer.name)
}

pub fn buffer_dtype(buffer: &Var) -> String {
    dtype_text(buffer.tensor_type().dtype.dtype)
}

pub fn buffer_scope(buffer: &Var) -> String {
    enum_name(&*buffer.tensor_type().scope)
}

/// The name of an `ffi.Enum` value, such as a memory scope (`shared`) or a layout axis
/// (`TCol`); Python's `.name`.
pub fn enum_name<T: ObjectCore>(value: &T) -> String {
    let name: FfiString = FieldGetter::new(T::type_index(), "_str_index")
        .and_then(|getter| getter.get(value))
        .expect("ffi.Enum values are named");
    ffi_text(&name)
}

pub fn buffer_ref(buffer: &Var) -> ObjectRef {
    oref(buffer.clone())
}

pub fn prim(node: &ObjectRef) -> AResult<Expr> {
    prim_expr(Any::from(node.clone()))
}

/// An integer constant, checked as the old `IntImm::new(dtype, value)` was.
pub fn int_const(dtype: &str, value: i64) -> tvm_ffi::Result<IntImm> {
    IntImm::from_dtype(DLDataType::try_from_str(dtype)?, value, Location::unknown())
}

/// A floating-point constant, checked as the old `FloatImm::new(dtype, value)` was.
pub fn float_const(dtype: &str, value: f64) -> tvm_ffi::Result<FloatImm> {
    FloatImm::from_dtype(DLDataType::try_from_str(dtype)?, value, Location::unknown())
}

/// The primitive type of `dtype`, checked as the old `PrimType::new(dtype)` was.
pub fn prim_type(dtype: &str) -> tvm_ffi::Result<PrimType> {
    PrimType::from_dtype(DLDataType::try_from_str(dtype)?)
}

/// `value` as a primitive expression: an `Expr` whose type is a `PrimType`.
pub fn prim_expr(value: impl Into<Any>) -> AResult<Expr> {
    let expr = Expr::try_from(value.into())?;
    if prim_dtype(&expr.ty).is_none() {
        return Err(Failure::Ffi(Error::new(
            tvm_ffi::TYPE_ERROR,
            "expected an expression whose type is a PrimType",
            "",
        )));
    }
    Ok(expr)
}

/// The dtype of a primitive expression; panics on another type, which `prim_expr` rules out.
pub trait PrimExprExt {
    fn dtype(&self) -> DLDataType;
}

impl PrimExprExt for Expr {
    fn dtype(&self) -> DLDataType {
        prim_dtype(&self.ty).expect("primitive expression type invariant was violated")
    }
}

pub fn int_imm(node: &ObjectRef) -> Option<i64> {
    node.as_node::<IntImmObj>()
        .and_then(|imm| int_value(imm).ok())
}

pub fn int_imm_expr(expr: &Expr) -> Option<i64> {
    expr.as_node::<IntImmObj>()
        .and_then(|imm| int_value(imm).ok())
}

/// Constrain each variable to its range in `analyzer`, as the old `Analyzer::bind` did.
///
/// tvm-tirx takes the ranges as one constraint context, which replaces the facts of any
/// context synchronized before.
pub fn bind_ranges(analyzer: &Analyzer, ranges: impl IntoIterator<Item = (Var, Range)>) -> AResult<()> {
    let context: ObjectRef = tvm_ffi::cached_global_func!("sym.ConstraintContext")
        .call_tuple((None::<ObjectRef>, 0_i64))?
        .try_into()?;
    for (var, range) in ranges {
        let constraint: ObjectRef =
            tvm_ffi::cached_global_func!("sym.RangeConstraint").call_tuple((var, range))?.try_into()?;
        tvm_ffi::cached_global_func!("sym.ConstraintContextAppend")
            .call_packed(&[AnyView::from(&context), AnyView::from(&constraint)])?;
    }
    analyzer.sync_constraint_context(&context)?;
    Ok(())
}

pub fn simplify(analyzer: &Analyzer, expr: &Expr) -> AResult<Expr> {
    Ok(analyzer.simplify(expr, 2)?)
}

pub fn static_int(
    analyzer: &Analyzer,
    expr: &Expr,
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
pub fn span_json(span: Option<&Location>) -> Option<Json> {
    let span = span?;
    if span.as_node::<CallSiteLocObj>().is_some() {
        // A CallSiteLoc chain flattens to its source locations, outermost call site first.
        let mut leaves = Vec::new();
        call_site_leaves(span, &mut leaves);
        if leaves.is_empty() {
            return None;
        }
        return Some(json_object(vec![
            ("kind", Json::from("sequential")),
            ("spans", Json::Array(leaves)),
        ]));
    }
    let leaf = span.as_node::<SourceLocObj>()?;
    let source_name = ffi_text(&leaf.source_name.name);
    if source_name.is_empty() {
        return None;
    }
    let coordinates = [leaf.start_line, leaf.start_column, leaf.end_line, leaf.end_column];
    if coordinates.iter().any(|value| *value < 1) {
        return None;
    }
    if leaf.end_line < leaf.start_line
        || (leaf.end_line == leaf.start_line && leaf.end_column < leaf.start_column)
    {
        return None;
    }
    Some(json_object(vec![
        ("kind", Json::from("span")),
        ("source_name", Json::String(source_name)),
        ("line", Json::from(leaf.start_line as i64)),
        ("column", Json::from(leaf.start_column as i64)),
        ("end_line", Json::from(leaf.end_line as i64)),
        ("end_column", Json::from(leaf.end_column as i64)),
    ]))
}

fn call_site_leaves(location: &Location, leaves: &mut Vec<Json>) {
    if let Some(call_site) = location.as_node::<CallSiteLocObj>() {
        call_site_leaves(&call_site.caller, leaves);
        call_site_leaves(&call_site.callee, leaves);
    } else if let Some(leaf) = span_json(Some(location)) {
        leaves.push(leaf);
    }
}

pub fn node_span(node: &ObjectRef) -> Option<Location> {
    if let Some(expr) = node.as_node::<ExprObj>() {
        return Some(expr.loc.clone());
    }
    node.as_node::<StmtObj>().map(|stmt| stmt.loc.clone())
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
