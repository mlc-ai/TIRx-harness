//! Source boundaries that keep instruction rules out of shared infrastructure.

use proc_macro2::{Delimiter, TokenStream, TokenTree};
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

/// Rust's lexer handles comments, raw strings, escapes, and macro arguments.
/// Documentation attributes are comments, not executable instruction references.
fn string_literals(source: &str) -> Vec<(usize, String)> {
    fn collect(tokens: TokenStream, found: &mut Vec<(usize, String)>) {
        let mut attribute = false;
        for token in tokens {
            match &token {
                TokenTree::Group(group) => {
                    let is_doc = attribute
                        && group.delimiter() == Delimiter::Bracket
                        && matches!(group.stream().into_iter().next(),
                            Some(TokenTree::Ident(name)) if name == "doc");
                    if !is_doc {
                        collect(group.stream(), found);
                    }
                }
                TokenTree::Literal(literal) => {
                    if let Ok(value) = syn::parse_str::<syn::LitStr>(&literal.to_string()) {
                        found.push((literal.span().start().line, value.value()));
                    }
                }
                _ => {}
            }
            // Preserve # across the ! in an inner documentation attribute.
            attribute = matches!(&token, TokenTree::Punct(p) if p.as_char() == '#')
                || (attribute && matches!(&token, TokenTree::Punct(p) if p.as_char() == '!'));
        }
    }
    let mut found = Vec::new();
    collect(source.parse().expect("valid Rust tokens"), &mut found);
    found
}

fn rust_files(directory: &Path) -> Vec<PathBuf> {
    let mut paths: Vec<_> = fs::read_dir(directory)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "rs"))
        .collect();
    paths.sort();
    paths
}

/// Recognize calls, including formatted names and nested generic arguments.
/// Type references within another function's generics are not calls.
fn direct_abi_calls(source: &str) -> Vec<(usize, String)> {
    let mut calls = Vec::new();
    for (start, _) in source.match_indices("v2::") {
        let mut angles = 0;
        let mut braces = 0;
        for (offset, character) in source[start + 4..].char_indices() {
            match character {
                '{' => braces += 1,
                '}' if braces > 0 => braces -= 1,
                _ if braces > 0 => {}
                '<' => angles += 1,
                '>' if angles > 0 => angles -= 1,
                _ if angles > 0 => {}
                '(' => {
                    let line = source[..start]
                        .bytes()
                        .filter(|byte| *byte == b'\n')
                        .count()
                        + 1;
                    calls.push((line, source[start..start + 4 + offset].to_owned()));
                    break;
                }
                c if c.is_ascii_alphanumeric() || c == '_' || c == ':' || c.is_whitespace() => {}
                _ => break,
            }
        }
    }
    calls
}

fn instruction_names(source: &str) -> BTreeSet<String> {
    string_literals(source)
        .into_iter()
        .map(|(_, value)| value)
        .filter(|name| name.starts_with("tirx.") || name.starts_with("prim."))
        .filter(|name| !name.ends_with('.'))
        .filter(|name| {
            name.chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.')
        })
        .collect()
}

fn instruction_references(source: &str, names: &BTreeSet<String>) -> Vec<(usize, String)> {
    let mut found = Vec::new();
    for (line, value) in string_literals(source) {
        for word in value.split(|c: char| !c.is_ascii_alphanumeric() && c != '_' && c != '.') {
            // prim._Op* names construct IR via FFI; they are not IR instruction
            // names. Keep this distinction narrow: all other prim.* spellings,
            // including future/formatted names, remain forbidden.
            let instruction_prefix = word.starts_with("tirx.cuda.")
                || word.starts_with("tirx.ptx.")
                || (word.starts_with("prim.") && !word.starts_with("prim._Op"));
            if names.contains(word) || instruction_prefix {
                found.push((line, word.to_owned()));
            }
        }
    }
    found
}

#[test]
fn only_abi_renders_direct_engine_calls() {
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut violations = Vec::new();
    let mut pending = vec![source.join("emit")];
    let mut files = Vec::new();
    while let Some(directory) = pending.pop() {
        files.extend(rust_files(&directory));
        pending.extend(
            fs::read_dir(directory)
                .unwrap()
                .map(|entry| entry.unwrap().path())
                .filter(|path| path.is_dir()),
        );
    }
    for path in files {
        if path == source.join("emit/abi.rs") {
            continue;
        }
        for (line, call) in direct_abi_calls(&fs::read_to_string(&path).unwrap()) {
            violations.push(format!("{}:{line}: {call}", path.display()));
        }
    }
    assert!(violations.is_empty(), "{}", violations.join("\n"));
}

#[test]
fn infrastructure_has_no_instruction_names() {
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut names = instruction_names(&fs::read_to_string(source.join("registry.rs")).unwrap());
    names.extend(instruction_names(
        &fs::read_to_string(source.join("decode.rs")).unwrap(),
    ));
    for path in rust_files(&source.join("decode")) {
        names.extend(instruction_names(&fs::read_to_string(path).unwrap()));
    }
    assert!(names.len() > 600);

    // Tile lowering is a separate, intentionally unchanged instruction layer.
    let mut files = rust_files(&source.join("analyze"));
    files.extend(rust_files(&source).into_iter().filter(|path| {
        !matches!(
            path.file_name().unwrap().to_str().unwrap(),
            "registry.rs" | "decode.rs"
        )
    }));
    files.extend(
        [
            "calls.rs",
            "diagnostics.rs",
            "expr.rs",
            "mod.rs",
            "module.rs",
            "scaffold.rs",
            "shared_blocks.rs",
            "stmt.rs",
        ]
        .map(|name| source.join("emit").join(name)),
    );
    let mut violations = Vec::new();
    for path in files {
        for (line, name) in instruction_references(&fs::read_to_string(&path).unwrap(), &names) {
            violations.push(format!("{}:{line}: {name}", path.display()));
        }
    }
    assert!(violations.is_empty(), "{}", violations.join("\n"));
}

#[test]
fn abi_guard_distinguishes_calls_from_type_arguments() {
    assert_eq!(
        direct_abi_calls("v2::reg::add::<Tag<Nested<A, B>>>(args)").len(),
        1
    );
    assert_eq!(
        direct_abi_calls("v2::{family}::{operation}({arguments})").len(),
        1
    );
    assert_eq!(direct_abi_calls("v2::SiteId::new(0)").len(), 1);
    assert!(direct_abi_calls("helper::<v2::Local>(args)").is_empty());
    assert!(direct_abi_calls("v2::Tag<(v2::A, v2::B)>").is_empty());
}

#[test]
fn instruction_guard_detects_registered_and_future_names() {
    let names = instruction_names("OpRow::new(\"tirx.exp\", \"family\", emit)");
    assert!(!instruction_references("name == \"tirx.exp\"", &names).is_empty());
    assert!(!instruction_references("name == \"tirx.ptx.future_op\"", &names).is_empty());
    assert!(instruction_references("attribute == \"tirx.dyn_smem_bytes\"", &names).is_empty());
}

#[test]
fn instruction_guard_ignores_fields_comments_and_ffi_constructors() {
    let names = BTreeSet::new();
    for source in [
        "prim.dtype",
        "// prim.if_then_else\n/* tirx.ptx.mov */ prim.dtype",
        "/// prim.if_then_else\nfn helper() {}",
        "//! tirx.ptx.mov\nfn helper() {}",
        r#"format!("prim._Op{operator}")"#,
        r#"get_global("prim._OpSub")"#,
    ] {
        assert!(
            instruction_references(source, &names).is_empty(),
            "{source}"
        );
    }
}

#[test]
fn instruction_guard_detects_raw_escaped_and_formatted_instruction_literals() {
    let names = instruction_names(r#"row("tirx.address_of")"#);
    for source in [
        r#"op == "tirx.address_of""#,
        r##"op == r#"tirx.ptx.mov"#"##,
        r#"op == "prim.\x69f_then_else""#,
        r#"format!("tirx.ptx.{operation}")"#,
        r#"format!("prim.{operation}")"#,
        r#"op == "prim.future_instruction""#,
        r#"panic!("unexpected tirx.ptx.future_op")"#,
    ] {
        assert!(
            !instruction_references(source, &names).is_empty(),
            "{source}"
        );
    }
    assert_eq!(
        instruction_references("\n\n\"tirx.ptx.mov\"", &names)[0].0,
        3
    );
}
