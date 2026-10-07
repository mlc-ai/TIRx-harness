//! Identity and lexical traversal contracts, without public test-only services.

use crate::analyze::buffers::BufferBindings;
use crate::analyze::util::{oref, same};
use tvm::ir::{Expr, IntImm, SourceName, Span, Var};
use tvm::prim::{Add, AddObj};
use tvm::tirx::{
    AttrStmt, BufferType, DeclBuffer, Evaluate, For, IfThenElse, PrimFunc, SeqStmt, Stmt, While,
};
use tvm::tvm_ffi::{Any, AnyView, Array, Function, Map, ObjectRefCore, String as FfiString};

fn initialize() {
    tvm::libinfo::load_compiler().expect("TVM compiler for native IR tests");
}

fn save(value: Any) -> FfiString {
    Function::get_global("ffi.ToJSONGraphString")
        .unwrap()
        .call_tuple((value, Option::<Map<FfiString, Any>>::None))
        .unwrap()
        .try_into()
        .unwrap()
}

#[test]
fn buffer_bindings_preserve_identities_and_last_declaration() {
    initialize();
    let ty = BufferType::new(
        "global",
        "int32",
        vec![Expr::from(IntImm::new("int64", 4).unwrap())],
    )
    .unwrap();
    let first = ty.new_var("same");
    let second = ty.new_var("same");
    let alias = ty.new_var("alias");
    let scalar = Var::new("same", "int32").unwrap();
    let first_data = first.data().unwrap();
    let second_data = second.data().unwrap();
    let declarations = vec![
        Stmt::from(DeclBuffer::new(&alias, first_data.clone()).unwrap()),
        Stmt::from(DeclBuffer::new(&alias, second_data.clone()).unwrap()),
    ];
    let func = PrimFunc::new(
        vec![first.as_var().clone(), scalar, second.as_var().clone()],
        SeqStmt::new(declarations.clone()).unwrap(),
    )
    .unwrap();
    let before = save(func.clone().into());
    let parameters = crate::buffer_parameters(func.clone()).unwrap();
    assert_eq!(parameters.len(), 2);
    assert!(same(&parameters.get(0).unwrap(), first.as_var()));
    assert!(same(&parameters.get(1).unwrap(), second.as_var()));
    let bindings = BufferBindings::build(&declarations).unwrap();
    assert_eq!(bindings.declarations.len(), 2);
    for ((buffer, data), expected) in bindings.declarations.iter().zip([first_data, second_data]) {
        assert!(same(buffer.as_var(), alias.as_var()));
        assert!(same(data, &expected));
    }
    assert!(same(
        &bindings.storage_key(&alias).unwrap(),
        second.as_var()
    ));
    assert!(!bindings.same_storage(&first, &alias).unwrap());
    assert_eq!(save(func.into()), before);
}

#[test]
fn cyclic_buffer_bindings_keep_each_starting_identity() {
    initialize();
    let ty = BufferType::new(
        "global",
        "int32",
        vec![Expr::from(IntImm::new("int64", 4).unwrap())],
    )
    .unwrap();
    let first = ty.new_var("first");
    let second = ty.new_var("second");
    let bindings = BufferBindings::build(&[
        DeclBuffer::new(&first, second.data().unwrap())
            .unwrap()
            .into(),
        DeclBuffer::new(&second, first.data().unwrap())
            .unwrap()
            .into(),
    ])
    .unwrap();
    for buffer in [&first, &second, &first] {
        assert!(same(
            &bindings.storage_key(buffer).unwrap(),
            buffer.as_var()
        ));
    }
}

#[test]
fn substitution_keeps_same_named_variables_distinct() {
    initialize();
    let first = Var::new("same", "int32").unwrap();
    let second = Var::new("same", "int32").unwrap();
    let replacement = Var::new("replacement", "int32").unwrap();
    let expression = Add::new(&first, &second).unwrap();
    let actual = crate::substitute(
        oref(expression.clone()),
        [(first.clone(), Expr::from(replacement.clone()))]
            .into_iter()
            .collect(),
    )
    .unwrap();
    let actual = actual.as_node::<AddObj>().unwrap();
    assert!(same(&actual.a, &replacement));
    assert!(same(&actual.b, &second));
    assert!(same(&expression.a, &first));
    assert!(same(&expression.b, &second));
}

#[test]
fn statement_walk_preserves_occurrences_order_identity_and_spans() {
    initialize();
    let span = Span::new(&SourceName::get("statement-order.py").unwrap(), 7, 7, 1, 9).unwrap();
    let shared: Stmt = Evaluate::with_span(IntImm::new("int32", 1).unwrap(), Some(&span))
        .unwrap()
        .into();
    let sequence: Stmt = SeqStmt::new(vec![shared.clone(), shared.clone()])
        .unwrap()
        .into();
    let condition = IntImm::new("bool", 1).unwrap();
    let while_loop: Stmt = While::new(&condition, shared.clone()).unwrap().into();
    let branch: Stmt =
        IfThenElse::with_span(&condition, sequence.clone(), Some(while_loop.clone()), None)
            .unwrap()
            .into();
    let counted: Stmt = For::new(
        Var::new("i", "int32").unwrap(),
        IntImm::new("int32", 0).unwrap(),
        IntImm::new("int32", 2).unwrap(),
        branch.clone(),
    )
    .unwrap()
    .into();
    let annotation_only = Evaluate::from_i64(99).unwrap();
    let attribute: Stmt = AttrStmt::new(
        annotation_only,
        "annotation",
        IntImm::new("int32", 0).unwrap(),
        counted.clone(),
    )
    .unwrap()
    .into();
    let init: Stmt = Evaluate::from_i64(2).unwrap().into();
    let empty = Array::<Any>::new(Vec::new());
    let annotations: Map<FfiString, Any> = [(FfiString::from("ignored"), sequence.clone().into())]
        .into_iter()
        .collect();
    // SBlock is intentionally outside tvm-rust's typed pass surface. The
    // reflected traversal still visits its executed init/body fields.
    let block: Stmt = Function::get_global("s_tir.SBlock")
        .unwrap()
        .call_packed(&[
            AnyView::from(&empty),
            AnyView::from(&empty),
            AnyView::from(&empty),
            AnyView::from(&FfiString::from("block")),
            AnyView::from(&attribute),
            AnyView::from(&init),
            AnyView::from(&empty),
            AnyView::from(&empty),
            AnyView::from(&annotations),
            AnyView::from(&Option::<Span>::None),
        ])
        .unwrap()
        .try_into()
        .unwrap();
    let root: Stmt = Function::get_global("s_tir.SBlockRealize")
        .unwrap()
        .call_tuple((empty, condition, block.clone(), Option::<Span>::None))
        .unwrap()
        .try_into()
        .unwrap();
    let expected = vec![
        root.clone(),
        block,
        init,
        attribute,
        counted,
        branch,
        sequence,
        shared.clone(),
        shared.clone(),
        while_loop,
        shared,
    ];
    let before = save(root.clone().into());
    let actual = crate::walk_statements(root.clone()).unwrap();
    assert_eq!(actual.len(), expected.len());
    for (left, right) in actual.iter().zip(expected) {
        assert!(same(&left, &right));
    }
    for index in [7, 8, 10] {
        assert!(same(
            actual.get(index).unwrap().span.as_ref().unwrap(),
            &span
        ));
    }
    assert_eq!(save(root.into()), before);
}
