//! Compile and execute the shared-block pass's result without a public FFI hook.

use super::render_shared_splits;
use crate::emit::module_template::MODULE_TEMPLATE;
use crate::emit::{SplitArgument, SplitHelper};
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "numsim-shared-block-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn run(command: &mut Command) {
    let output = command.output().unwrap();
    assert!(
        output.status.success(),
        "{command:?}: {}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn check_shared_blocks(is_async: bool) {
    let mut helpers = Vec::new();
    let mut calls = Vec::new();
    let mut expected = 0;
    for (index, (argument, constant, label, label_offset)) in [
        (13, 3, "input α", 101),
        (17, 5, "input \"β\"", 103),
        (19, 7, "output γ", 107),
    ]
    .into_iter()
    .enumerate()
    {
        let name = format!("helper_{index}");
        let variable = format!("value_β{index}");
        let local = format!("local_γ{index}");
        let mut body = Vec::new();
        if is_async {
            body.push("YieldOnce(false).await;".to_owned());
        }
        body.extend([
            format!("let {local} = {variable} * 2_i64 + {constant}_i64;"),
            format!("let label_value = v2_named_address::<v2::Global>({variable}, {label:?});"),
            format!("ctx += {local} + label_value;"),
            "{ let mut ctx = ctx + 100_i64; ctx += 1_i64; assert!(ctx > 0);".to_owned(),
            "  let warp = &mut *warp; *warp = (); }".to_owned(),
            "ctx += 0_i64; let _ = &mut *warp;".to_owned(),
            "ctx += format!(\"label:{}\", label_value).len() as i64;".to_owned(),
        ]);
        helpers.push(SplitHelper {
            name: name.clone(),
            is_async,
            inline_never: true,
            arguments: vec![SplitArgument {
                name: variable.clone(),
                rust_type: "i64".to_owned(),
                call_code: variable,
                snapshot_before_control: false,
            }],
            body,
        });
        let call = format!("{name}(ctx, &mut warp, {argument}_i64)");
        calls.push(if is_async {
            format!("ctx = NumSimModuleFuture({call}).await?;")
        } else {
            format!("ctx = {call}.unwrap();")
        });
        expected += argument * 3 + constant + label_offset + 9;
    }
    let (modules, root) = render_shared_splits(&helpers, &calls.join("\n"), "WarpEngine").unwrap();
    // Verify this exercises factoring as well as compiling equivalent helpers.
    assert!(modules.contains("fn helper_0_shared("));
    assert!(!modules.contains("fn helper_1("));
    let prelude = if is_async {
        let start = MODULE_TEMPLATE
            .find("macro_rules! numsim_local_future {")
            .unwrap();
        let end = MODULE_TEMPLATE.find("__NUMSIM_MEMORY_HELPERS__").unwrap();
        format!("{}\n{ASYNC_DRIVER}", &MODULE_TEMPLATE[start..end])
    } else {
        String::new()
    };
    let entry = if is_async {
        "async fn run() -> Result<(), EngineError>"
    } else {
        "fn main()"
    };
    let result = if is_async { "Ok(())" } else { "" };
    let source = format!(
        "{COMMON_PRELUDE}\n{prelude}\n{modules}\n\
         {entry} {{ let mut ctx = 0_i64; let mut warp = ();\n\
         {root}\nassert_eq!(ctx, {expected}); {result} }}\n"
    );
    let scratch = Scratch::new();
    let source_path = scratch.0.join("semantic.rs");
    let executable = scratch.0.join("semantic");
    fs::write(&source_path, source).unwrap();
    let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    run(Command::new(rustc)
        .arg("--edition=2021")
        .arg(&source_path)
        .arg("-o")
        .arg(&executable));
    run(&mut Command::new(executable));
}

#[test]
fn shared_blocks_preserve_numeric_results_and_labels() {
    check_shared_blocks(false);
}

#[test]
fn shared_blocks_preserve_numeric_results_labels_and_resumptions() {
    check_shared_blocks(true);
}

const COMMON_PRELUDE: &str = r#"
type WarpContext = i64; type WarpEngine = (); type EngineError = ();
mod v2 { pub struct Global; }
fn v2_named_address<S>(pointer: i64, label: &'static str) -> i64 {
    pointer + match label {
        "input α" => 101,
        "input \"β\"" => 103,
        "output γ" => 107,
        _ => panic!("unexpected address label: {}", label),
    }
}
"#;

const ASYNC_DRIVER: &str = r#"
struct YieldOnce(bool);
impl std::future::Future for YieldOnce {
    type Output = ();
    fn poll(mut self: std::pin::Pin<&mut Self>, _: &mut std::task::Context<'_>)
        -> std::task::Poll<()> {
        if self.0 { std::task::Poll::Ready(()) }
        else { self.0 = true; std::task::Poll::Pending }
    }
}
fn main() {
    let mut future = std::pin::pin!(run());
    let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
    let mut pending = 0;
    loop {
        match std::future::Future::poll(future.as_mut(), &mut cx) {
            std::task::Poll::Ready(result) => {
                result.unwrap();
                assert_eq!(pending, 3);
                break;
            }
            std::task::Poll::Pending => {
                pending += 1;
                assert!(pending <= 3, "shared future did not resume");
            }
        }
    }
}
"#;
