//! `shared_blocks.render_shared_splits`: factor repeated private split bodies
//! without changing instruction order.
//!
//! Lexical matching, shape grouping, constant specialization and call
//! rewriting follow `shared_blocks.rs`; unsupported binding and string forms
//! remain unshared.

use std::collections::HashMap;

use super::super::analyze::util::{ffi_error, AResult, Failure};
use super::module::render_split_future_calls;
use super::SplitHelper;

#[cfg(test)]
mod tests;

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Id,
    Num,
    Str,
    Lifetime,
    Other,
}

#[derive(Clone)]
struct Tok {
    text: String,
    before: String,
    kind: Kind,
    start: usize,
    end: usize,
}

/// Python's `str` predicates behind `re`'s Unicode `\w`, `\s` and `\d`.
fn property(c: char) -> (u8, u8) {
    if c.is_ascii() {
        let b = c as u8;
        return (
            (u8::from(b.is_ascii_alphanumeric() || b == b'_'))
                | (u8::from(b.is_ascii_whitespace() || (28..=31).contains(&b)) << 1)
                | (u8::from(b.is_ascii_digit()) << 2)
                | (u8::from(b.is_ascii_uppercase()) << 3),
            if b.is_ascii_digit() { b - b'0' } else { 0 },
        );
    }
    (
        u8::from(c.is_alphanumeric())
            | (u8::from(c.is_whitespace()) << 1)
            | (u8::from(c.is_numeric()) << 2)
            | (u8::from(c.is_uppercase()) << 3),
        0,
    )
}

fn word(c: char) -> bool {
    property(c).0 & 1 != 0
}

fn space(c: char) -> bool {
    property(c).0 & 2 != 0
}

fn digit(c: char) -> bool {
    property(c).0 & 4 != 0
}

fn upper(c: char) -> bool {
    property(c).0 & 8 != 0
}

fn ident(c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_'
}

fn integer(s: &str) -> Option<(&str, &str)> {
    let (n, t) = s.split_once('_')?;
    if n.is_empty() || !n.chars().all(digit) {
        return None;
    }
    if matches!(
        t,
        "i8" | "u8" | "i16" | "u16" | "i32" | "u32" | "i64" | "u64" | "isize" | "usize"
    ) {
        Some((n, t))
    } else {
        None
    }
}

fn integer_value(s: &str) -> Option<u128> {
    s.chars().try_fold(0u128, |n, c| {
        n.checked_mul(10)?.checked_add(property(c).1 as u128)
    })
}

fn tokens(s: &str) -> Vec<Tok> {
    let mut i = 0;
    let mut before = String::new();
    let mut out = Vec::new();
    while i < s.len() {
        let start = i;
        let c = s[i..].chars().next().unwrap();
        if space(c) {
            while i < s.len() {
                let c = s[i..].chars().next().unwrap();
                if !space(c) {
                    break;
                }
                i += c.len_utf8()
            }
            before.push_str(&s[start..i]);
            continue;
        }
        if s[i..].starts_with("//") {
            i = s[i..].find('\n').map(|n| i + n).unwrap_or(s.len());
            before.push(' ');
            continue;
        }
        if s[i..].starts_with("/*") {
            if let Some(n) = s[i + 2..].find("*/") {
                i += n + 4;
                before.push(' ');
                continue;
            }
        }
        let mut kind = Kind::Other;
        if c == '"' {
            let mut j = i + 1;
            let mut closed = false;
            while j < s.len() {
                let c = s[j..].chars().next().unwrap();
                j += c.len_utf8();
                if c == '\\' {
                    if j < s.len() {
                        j += s[j..].chars().next().unwrap().len_utf8()
                    }
                    continue;
                }
                if c == '"' {
                    closed = true;
                    break;
                }
            }
            if closed {
                i = j;
                kind = Kind::Str
            } else {
                i += 1
            }
        } else if c == '\'' && s[i + 1..].chars().next().is_some_and(ident) {
            i += 2;
            while i < s.len() {
                let c = s[i..].chars().next().unwrap();
                if !word(c) {
                    break;
                }
                i += c.len_utf8()
            }
            kind = Kind::Lifetime
        } else if ident(c) {
            i += c.len_utf8();
            while i < s.len() {
                let c = s[i..].chars().next().unwrap();
                if !word(c) {
                    break;
                }
                i += c.len_utf8()
            }
            kind = Kind::Id
        } else if digit(c) && s[..i].chars().next_back().map_or(true, |c| !word(c)) {
            let mut j = i + c.len_utf8();
            while j < s.len() {
                let c = s[j..].chars().next().unwrap();
                if !digit(c) {
                    break;
                }
                j += c.len_utf8()
            }
            let mut end = j;
            if s[j..].starts_with('_') {
                let mut k = j + 1;
                while k < s.len() {
                    let c = s[k..].chars().next().unwrap();
                    if !word(c) {
                        break;
                    }
                    k += c.len_utf8()
                }
                if integer(&s[i..k]).is_some() {
                    end = k
                }
            }
            if s[end..].chars().next().map_or(true, |c| !word(c)) {
                i = end;
                kind = Kind::Num
            } else {
                i += c.len_utf8()
            }
        } else {
            let width = ["..=", "::", "->", "=>", ".."]
                .iter()
                .find(|p| s[i..].starts_with(**p))
                .map(|p| p.len())
                .unwrap_or(c.len_utf8());
            i += width
        }
        let text = s[start..i].to_owned();
        out.push(Tok {
            text,
            before: std::mem::take(&mut before),
            kind,
            start,
            end: i,
        });
    }
    out
}

struct Helper {
    name: String,
    asynchronous: bool,
    inline_never: bool,
    args: Vec<(String, String)>,
    implicit_parameters: Vec<String>,
    body: String,
}

struct Pool {
    kind: String,
    ty: String,
    values: Vec<String>,
}

struct Body {
    helper: usize,
    tokens: Vec<Tok>,
    pools: Vec<Pool>,
    names: Vec<String>,
}

fn pool_capture(pools: &mut Vec<Pool>, kind: &str, ty: &str, value: &str) -> String {
    let p = if let Some(i) = pools.iter().position(|p| p.kind == kind && p.ty == ty) {
        i
    } else {
        pools.push(Pool {
            kind: kind.into(),
            ty: ty.into(),
            values: Vec::new(),
        });
        pools.len() - 1
    };
    let values = &mut pools[p].values;
    let index = if let Some(i) = values.iter().position(|v| v == value) {
        i
    } else {
        values.push(value.into());
        values.len() - 1
    };
    format!("numsim_block_{kind}[{index}]")
}

/// The emitted `v2_named_address::<v2::Space>(pointer, label)` argument; format
/// strings, patterns and other literal contexts stay untouched.
fn named_address_label(toks: &[Tok], index: usize) -> bool {
    if index < 10 || toks.get(index + 1).map_or(true, |t| t.text != ")") {
        return false;
    }
    let call = &toks[index - 10..index];
    call[0].text == "v2_named_address"
        && call[1].text == "::"
        && call[2].text == "<"
        && call[3].text == "v2"
        && call[4].text == "::"
        && call[5].kind == Kind::Id
        && call[6].text == ">"
        && call[7].text == "("
        && call[8].kind == Kind::Id
        && call[9].text == ","
}

fn implicit_capture(s: &str, symbols: &HashMap<String, String>) -> bool {
    for (i, c) in s.char_indices() {
        if c != '{' || s[..i].ends_with('{') || !s[i + 1..].chars().next().is_some_and(ident) {
            continue;
        }
        let mut j = i + 2;
        while j < s.len() {
            let c = s[j..].chars().next().unwrap();
            if !word(c) {
                break;
            }
            j += c.len_utf8()
        }
        if s[j..]
            .chars()
            .next()
            .is_some_and(|c| matches!(c, '}' | ':' | '!'))
            && symbols.contains_key(&s[i + 1..j])
        {
            return true;
        }
    }
    false
}

/// Parameterize only scalar values: patterns and type-level lengths remain
/// constants. Binding names are alpha-renamed within the emitter's unique names.
fn capture(h: &Helper, index: usize, toks: Vec<Tok>) -> AResult<Option<Body>> {
    let mut symbols: HashMap<String, String> = HashMap::new();
    let mut names = Vec::new();
    for (i, (name, _)) in h.args.iter().enumerate() {
        let value = format!("numsim_local_{i}");
        symbols.insert(name.clone(), value.clone());
        names.push(value)
    }
    let mut depth = 0;
    for (i, t) in toks.iter().enumerate() {
        let s = t.text.as_str();
        let prev = if i == 0 { "" } else { &toks[i - 1].text };
        if matches!(
            s,
            "match"
                | "fn"
                | "struct"
                | "enum"
                | "impl"
                | "type"
                | "union"
                | "trait"
                | "const"
                | "static"
                | "use"
                | "mod"
        ) {
            return Ok(None);
        }
        if s == "<" && (depth > 0 || prev == "::" || prev.chars().next().is_some_and(upper)) {
            depth += 1
        } else if s == ">" && depth > 0 {
            depth -= 1
        } else if (depth > 0 || prev == ";") && integer(s).is_some() {
            return Ok(None);
        }
        if t.kind == Kind::Id
            && (s.starts_with("numsim_block_")
                || s.strip_prefix("numsim_local_")
                    .is_some_and(|v| !v.is_empty() && v.chars().all(digit)))
        {
            return Ok(None);
        }
        if matches!(s, "r" | "br")
            && i + 1 < toks.len()
            && (toks[i + 1].text == "#" || toks[i + 1].kind == Kind::Str)
        {
            return Ok(None);
        }
        if !matches!(s, "let" | "for") {
            continue;
        }
        let mut b = i + 1;
        let Some(binding) = toks.get(b) else {
            return Ok(None);
        };
        if binding.text == "mut" {
            b += 1
        }
        let Some(binding) = toks.get(b) else {
            return Ok(None);
        };
        if binding.kind != Kind::Id {
            return Ok(None);
        }
        let Some(next) = toks.get(b + 1) else {
            return Ok(None);
        };
        let next = next.text.as_str();
        if (s == "let" && !matches!(next, "=" | ":")) || (s == "for" && next != "in") {
            return Ok(None);
        }
        // The renderer owns the signature. Its implicit parameters remain
        // bound even when a nested scope shadows them inside the body.
        if toks[b].text != "_" && !h.implicit_parameters.contains(&toks[b].text) {
            let name = format!("numsim_local_{}", symbols.len());
            symbols.entry(toks[b].text.clone()).or_insert(name);
        }
    }
    for t in &toks {
        if t.kind == Kind::Str && implicit_capture(&t.text, &symbols) {
            return Ok(None);
        }
    }
    let mut pools = Vec::new();
    let mut rewritten = Vec::with_capacity(toks.len());
    for (i, t) in toks.iter().enumerate() {
        let prev = if i == 0 { "" } else { &toks[i - 1].text };
        let next = toks.get(i + 1).map(|t| t.text.as_str()).unwrap_or("");
        let mut t = t.clone();
        match t.kind {
            Kind::Num => {
                if let Some((n, ty)) = integer(&t.text) {
                    let signed_too_large = ty.starts_with('i')
                        && ty != "isize"
                        && integer_value(n)
                            .map(|n| n >= 1u128 << (ty[1..].parse::<u32>().unwrap() - 1))
                            .unwrap_or(true);
                    if !signed_too_large {
                        t.text = pool_capture(&mut pools, ty, ty, &t.text)
                    }
                } else if i >= 4
                    && toks[i - 4..i]
                        .iter()
                        .map(|t| t.text.as_str())
                        .eq(["buffers", ".", "buffers", "["])
                {
                    t.text =
                        pool_capture(&mut pools, "buffers", "usize", &(t.text.clone() + "_usize"))
                }
            }
            Kind::Str => {
                if t.text
                    .strip_prefix("\"anonymous_buffer_")
                    .and_then(|s| s.strip_suffix('"'))
                    .is_some_and(|s| !s.is_empty() && s.chars().all(digit))
                    || named_address_label(&toks, i)
                {
                    t.text = pool_capture(&mut pools, "labels", "&'static str", &t.text)
                }
            }
            Kind::Id => {
                if prev != "." && prev != "::" && next != "::" {
                    if let Some(s) = symbols.get(&t.text) {
                        t.text = s.clone()
                    }
                }
            }
            Kind::Other | Kind::Lifetime => {}
        }
        rewritten.push(t);
    }
    Ok(Some(Body {
        helper: index,
        tokens: rewritten,
        pools,
        names,
    }))
}

#[derive(Hash, PartialEq, Eq)]
struct Shape {
    asynchronous: bool,
    types: Vec<String>,
    pools: Vec<(String, String, usize)>,
    tokens: Vec<String>,
}

fn shape(h: &Helper, b: &Body) -> Shape {
    Shape {
        asynchronous: h.asynchronous,
        types: h.args.iter().map(|a| a.1.clone()).collect(),
        pools: b
            .pools
            .iter()
            .map(|p| (p.kind.clone(), p.ty.clone(), p.values.len()))
            .collect(),
        tokens: b.tokens.iter().map(|t| t.text.clone()).collect(),
    }
}

/// Restore constants that agree across every member so Rust can still fold them.
fn specialize(members: &mut [Body]) {
    let mut replacement = HashMap::new();
    let keys: Vec<_> = members[0]
        .pools
        .iter()
        .map(|p| (p.kind.clone(), p.ty.clone()))
        .collect();
    for (kind, ty) in keys {
        let p = members[0]
            .pools
            .iter()
            .position(|p| p.kind == kind && p.ty == ty)
            .unwrap();
        let mut varying = Vec::new();
        for i in 0..members[0].pools[p].values.len() {
            let first = &members[0].pools[p].values[i];
            let value = if members.iter().all(|m| &m.pools[p].values[i] == first) {
                first.clone()
            } else {
                let s = format!("numsim_block_{kind}[{}]", varying.len());
                varying.push(i);
                s
            };
            replacement.insert(format!("numsim_block_{kind}[{i}]"), value);
        }
        for m in members.iter_mut() {
            if varying.is_empty() {
                m.pools.remove(p);
            } else {
                m.pools[p].values = varying
                    .iter()
                    .map(|i| m.pools[p].values[*i].clone())
                    .collect()
            }
        }
    }
    for m in members {
        for t in &mut m.tokens {
            if let Some(s) = replacement.get(&t.text) {
                t.text = s.clone()
            }
        }
    }
}

fn render(b: &Body) -> String {
    let mut s = String::new();
    for t in &b.tokens {
        s.push_str(&t.before);
        s.push_str(&t.text);
    }
    s
}

type Replacements = HashMap<String, (String, Vec<String>)>;

/// Edit original byte ranges: comments, strings, and whitespace remain intact.
fn rewrite(s: &str, replacements: &Replacements) -> AResult<String> {
    let ts = tokens(s);
    let mut stack = Vec::new();
    let mut edits: Vec<(usize, usize, String)> = Vec::new();
    for (i, t) in ts.iter().enumerate() {
        if t.text == "(" {
            let prev = i.checked_sub(1).map(|i| &ts[i]);
            let replacement = prev
                .filter(|p| p.kind == Kind::Id)
                .and_then(|p| replacements.get(&p.text));
            stack.push(replacement);
            if let Some(r) = replacement {
                let p = prev.unwrap();
                edits.push((p.start, p.end, r.0.clone()))
            }
        } else if t.text == ")" {
            let Some(entry) = stack.pop() else {
                return failed("unbalanced generated parentheses");
            };
            if let Some(r) = entry {
                if !r.1.is_empty() {
                    let sep = if matches!(ts[i - 1].text.as_str(), "(" | ",") {
                        ""
                    } else {
                        ","
                    };
                    edits.push((t.start, t.start, format!("{sep} {}", r.1.join(", "))))
                }
            }
        }
    }
    if !stack.is_empty() {
        return failed("unbalanced generated parentheses");
    }
    edits.sort_by_key(|e| (e.0, e.1));
    let mut out = String::new();
    let mut prev = 0;
    for (start, end, text) in edits {
        out.push_str(&s[prev..start]);
        out.push_str(&text);
        prev = end
    }
    out.push_str(&s[prev..]);
    Ok(out)
}

fn failed<T>(message: &str) -> AResult<T> {
    Err(Failure::Ffi(ffi_error(&format!(
        "NumSim shared-block pass failed:\n{message}"
    ))))
}

/// Render split modules while sharing repeated pure text shapes; the root
/// body's calls are rewritten to the shared helpers.
pub fn render_shared_splits(
    split_helpers: &[SplitHelper],
    root: &str,
    warp_engine_type: &str,
) -> AResult<(String, String)> {
    if split_helpers.iter().filter(|helper| helper.inline_never).count() < 2 {
        let modules: Vec<String> = split_helpers
            .iter()
            .map(|helper| helper.render_module(warp_engine_type, None, None, None, false))
            .collect();
        return Ok((modules.join("\n\n"), render_split_future_calls(root)));
    }
    let helpers: Vec<Helper> = split_helpers
        .iter()
        .map(|helper| Helper {
            name: helper.name.clone(),
            asynchronous: helper.is_async,
            inline_never: helper.inline_never,
            args: helper
                .arguments
                .iter()
                .map(|argument| (argument.name.clone(), argument.rust_type.clone()))
                .collect(),
            implicit_parameters: vec!["ctx".to_owned(), "warp".to_owned()],
            body: helper.body.join("\n"),
        })
        .collect();
    let mut groups: Vec<Vec<Body>> = Vec::new();
    let mut shapes: HashMap<Shape, usize> = HashMap::new();
    for (i, h) in helpers.iter().enumerate() {
        if h.inline_never {
            let ts = tokens(&h.body);
            if let Some(body) = capture(&helpers[i], i, ts)? {
                let sh = shape(&helpers[i], &body);
                let len = groups.len();
                let g = *shapes.entry(sh).or_insert(len);
                if g == len {
                    groups.push(Vec::new())
                }
                groups[g].push(body)
            }
        }
    }
    let mut replacements = Replacements::new();
    let mut shared = Vec::new();
    for mut members in groups {
        if members.len() < 2 {
            continue;
        }
        specialize(&mut members);
        let first = &members[0];
        let name = helpers[first.helper].name.clone() + "_shared";
        for m in &members {
            replacements.insert(
                helpers[m.helper].name.clone(),
                (
                    name.clone(),
                    m.pools
                        .iter()
                        .map(|p| format!("&[{}]", p.values.join(", ")))
                        .collect(),
                ),
            );
        }
        let mut params: Vec<(String, String)> = first
            .names
            .iter()
            .zip(&helpers[first.helper].args)
            .map(|(n, (_, t))| (n.clone(), t.clone()))
            .collect();
        params.extend(first.pools.iter().map(|p| {
            (
                format!("numsim_block_{}", p.kind),
                format!("&[{}; {}]", p.ty, p.values.len()),
            )
        }));
        shared.push((first.helper, name, params, render(first)));
    }
    let mut modules = Vec::new();
    for (i, h) in helpers.iter().enumerate() {
        if replacements.contains_key(&h.name) {
            continue;
        }
        let body = if replacements.is_empty() {
            h.body.clone()
        } else {
            rewrite(&h.body, &replacements)?
        };
        modules.push(split_helpers[i].render_module(
            warp_engine_type,
            None,
            None,
            Some(&body),
            false,
        ));
    }
    for (i, name, params, body) in shared {
        let body = rewrite(&body, &replacements)?;
        modules.push(split_helpers[i].render_module(
            warp_engine_type,
            Some(&name),
            Some(&params),
            Some(&body),
            false,
        ));
    }
    let root = if replacements.is_empty() {
        root.to_owned()
    } else {
        rewrite(root, &replacements)?
    };
    Ok((modules.join("\n\n"), render_split_future_calls(&root)))
}
