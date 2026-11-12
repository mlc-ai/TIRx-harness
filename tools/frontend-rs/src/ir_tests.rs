//! Identity and lexical traversal contracts, without public test-only services.

use crate::decode::{data_source, DECL_TENSOR, TENSOR_DATA_PTR};
use crate::analyze::buffers::{BufferBindings};
use crate::analyze::util::{int_const, oref, prim_type, same};
use tvm_tirx::ir::prim::{Add, AddObj};
use tvm_tirx::ir::{
    Bind, Call, Evaluate, Expr, For, ForKind, If, Location, Op, PrimType, PtrType, RegionStmt,
    SeqStmt, SourceLoc, SourceName, Stmt, Type, Var, While,
};
use tvm_tirx::runtime::MemoryScope;
use tvm_tirx::tirx::TensorType;
use tvm_tirx::tvm_ffi::{Any, Array, Function, Map, ObjectRefCore, String as FfiString};

fn save(value: Any) -> FfiString {
    Function::get_global("ffi.ToJSONGraphString")
        .unwrap()
        .call_tuple((value, Option::<Map<FfiString, Any>>::None))
        .unwrap()
        .try_into()
        .unwrap()
}

/// The registered memory scope `name`, through its JSON spelling.
fn memory_scope(name: &str) -> MemoryScope {
    let graph = format!(
        r#"{{"root_index": 0, "nodes": [{{"type": "runtime.MemoryScope", "data": "{name}"}}]}}"#
    );
    Function::get_global("ffi.FromJSONGraphString")
        .unwrap()
        .call_tuple((FfiString::from(graph),))
        .unwrap()
        .try_into()
        .unwrap()
}

fn tensor_type() -> TensorType {
    TensorType::new(
        [int_const("int64", 4).unwrap()],
        prim_type("int32").unwrap(),
        memory_scope("global"),
        0,
        None,
        tvm_tirx::ir::UNKNOWN_LOC,
    )
    .unwrap()
}

fn call(ty: impl Into<Type>, op: &str, args: Vec<Expr>, ty_args: Vec<Type>) -> Expr {
    Call::new(Some(&ty.into()), Op::get(op).unwrap(), args, None, ty_args, tvm_tirx::ir::UNKNOWN_LOC)
        .unwrap()
        .into()
}

/// `Bind(view, decl_tensor(tensor_data_ptr(source)))`.
fn declare_view(view: &Var, source: &Var) -> Stmt {
    let ty = tensor_type();
    let pointer = PtrType::new(prim_type("int32").unwrap(), &memory_scope("global")).unwrap();
    let data = call(pointer, TENSOR_DATA_PTR, vec![source.clone().into()], Vec::new());
    let decl = call(ty.clone(), DECL_TENSOR, vec![data], vec![ty.into()]);
    Bind::new(view.clone(), decl, tvm_tirx::ir::UNKNOWN_LOC).unwrap().into()
}

fn tensor(name: &str) -> Var {
    Var::new(name, tensor_type(), tvm_tirx::ir::UNKNOWN_LOC)
}

#[test]
fn buffer_bindings_preserve_identities_and_last_declaration() {
    let first = tensor("same");
    let second = tensor("same");
    let alias = tensor("alias");
    let scalar = Var::new("same", prim_type("int32").unwrap(), tvm_tirx::ir::UNKNOWN_LOC);
    let declarations = vec![declare_view(&alias, &first), declare_view(&alias, &second)];
    let func = tvm_tirx::tirx::Function::new(
        vec![first.clone(), scalar, second.clone()],
        &SeqStmt::new(declarations.clone(), tvm_tirx::ir::UNKNOWN_LOC),
        Type::from(PrimType::void()),
        tvm_tirx::ir::DictAttrs::new(Map::<FfiString, Any>::new()),
        tvm_tirx::ir::UNKNOWN_LOC,
    );
    let before = save(func.clone().into());
    let parameters = crate::buffer_parameters(func.clone()).unwrap();
    assert_eq!(parameters.len(), 2);
    assert!(same(&parameters.get(0).unwrap(), &first));
    assert!(same(&parameters.get(1).unwrap(), &second));
    let bindings = BufferBindings::build(&declarations).unwrap();
    assert_eq!(bindings.declarations.len(), 2);
    for ((buffer, data), source) in bindings.declarations.iter().zip([&first, &second]) {
        assert!(same(buffer, &alias));
        let (root, offsets) = data_source(data).unwrap().unwrap();
        assert!(same(&root, source) && offsets.is_empty());
    }
    assert!(same(&bindings.storage_key(&alias).unwrap(), &second));
    assert!(!bindings.same_storage(&first, &alias).unwrap());
    assert_eq!(save(func.into()), before);
}

#[test]
fn cyclic_buffer_bindings_keep_each_starting_identity() {
    let first = tensor("first");
    let second = tensor("second");
    let bindings =
        BufferBindings::build(&[declare_view(&first, &second), declare_view(&second, &first)])
            .unwrap();
    for buffer in [&first, &second, &first] {
        assert!(same(&bindings.storage_key(buffer).unwrap(), buffer));
    }
}

#[test]
fn substitution_keeps_same_named_variables_distinct() {
    let first = Var::new("same", PrimType::int(32), tvm_tirx::ir::UNKNOWN_LOC);
    let second = Var::new("same", PrimType::int(32), tvm_tirx::ir::UNKNOWN_LOC);
    let replacement = Var::new("replacement", PrimType::int(32), tvm_tirx::ir::UNKNOWN_LOC);
    let expression = Add::new(crate::analyze::util::primitive(&first).unwrap(), crate::analyze::util::primitive(&second).unwrap(), tvm_tirx::ir::UNKNOWN_LOC).unwrap();
    let actual = crate::substitute(
        oref(expression.clone()),
        [(first.clone(), Expr::from(replacement.clone()))]
            .into_iter()
            .collect(),
    )
    .unwrap();
    let actual = actual.downcast_ref::<AddObj>().unwrap();
    assert!(same(&actual.a, &replacement));
    assert!(same(&actual.b, &second));
    assert!(same(&expression.a, &first));
    assert!(same(&expression.b, &second));
}

#[test]
fn statement_walk_preserves_occurrences_order_identity_and_spans() {
    let span: Location =
        SourceLoc::new(SourceName::get("statement-order.py"), 7, 1, 7, 9).into();
    let shared: Stmt = Evaluate::new(int_const("int32", 1).unwrap(), span.clone()).into();
    let sequence: Stmt = SeqStmt::new(vec![shared.clone(), shared.clone()], tvm_tirx::ir::UNKNOWN_LOC).into();
    let condition = int_const("bool", 1).unwrap();
    let while_loop: Stmt = While::new(&condition, shared.clone(), tvm_tirx::ir::UNKNOWN_LOC)
        .unwrap()
        .into();
    let branch: Stmt = If::new(&condition, sequence.clone(), Some(&while_loop), tvm_tirx::ir::UNKNOWN_LOC)
        .unwrap()
        .into();
    let counted: Stmt = For::new(
        tvm_tirx::ir::PrimVar::new("i", prim_type("int32").unwrap(), tvm_tirx::ir::UNKNOWN_LOC),
        int_const("int32", 0).unwrap(),
        int_const("int32", 2).unwrap(),
        int_const("int32", 1).unwrap(),
        ForKind::Default,
        branch.clone(),
        None,
        tvm_tirx::ir::UNKNOWN_LOC,
    )
    .unwrap()
    .into();
    let init: Stmt = Evaluate::new(int_const("int32", 2).unwrap(), tvm_tirx::ir::UNKNOWN_LOC).into();
    // Launch operands and attributes are not statements; only the region body is walked.
    let fields = Array::from_vec(vec![FfiString::from("grid.x")]);
    let attrs: Map<FfiString, Any> = [(FfiString::from("cuda.launch_fields"), Any::from(fields))]
        .into_iter()
        .collect();
    let region: Stmt = RegionStmt::new(
        Op::get("tirx.device_entry").unwrap(),
        vec![Expr::from(int_const("int32", 1).unwrap())],
        Vec::<Var>::new(),
        tvm_tirx::ir::DictAttrs::new(attrs),
        SeqStmt::new(vec![init.clone(), counted.clone()], tvm_tirx::ir::UNKNOWN_LOC),
        Vec::<Var>::new(),
        tvm_tirx::ir::UNKNOWN_LOC,
    )
    .unwrap()
    .into();
    // Every loop and branch body is a SeqStmt.
    let body = |stmt: &Stmt| -> Stmt {
        if let Some(region) = stmt.downcast_ref::<tvm_tirx::ir::RegionStmtObj>() {
            return region.body.clone().into();
        }
        if let Some(loop_stmt) = stmt.downcast_ref::<tvm_tirx::ir::ForObj>() {
            return loop_stmt.body.clone().into();
        }
        if let Some(loop_stmt) = stmt.downcast_ref::<tvm_tirx::ir::WhileObj>() {
            return loop_stmt.body.clone().into();
        }
        let branch = stmt.downcast_ref::<tvm_tirx::ir::IfObj>().unwrap();
        branch.else_case.clone().unwrap().into()
    };
    assert!(same(
        &branch.downcast_ref::<tvm_tirx::ir::IfObj>().unwrap().then_case,
        &sequence
    ));
    let expected = vec![
        region.clone(),
        body(&region),
        init,
        counted.clone(),
        body(&counted),
        branch.clone(),
        sequence,
        shared.clone(),
        shared.clone(),
        body(&branch),
        while_loop.clone(),
        body(&while_loop),
        shared,
    ];
    let before = save(region.clone().into());
    let actual = crate::walk_statements(region.clone()).unwrap();
    assert_eq!(actual.len(), expected.len());
    for (left, right) in actual.iter().zip(expected) {
        assert!(same(left, &right));
    }
    for index in [7, 8, 12] {
        assert!(same(&actual.get(index).unwrap().loc, &span));
    }
    assert_eq!(save(region.into()), before);
}
