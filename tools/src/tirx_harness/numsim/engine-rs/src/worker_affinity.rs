//! Worker-thread placement.
//!
//! A multi-worker launch confines its worker threads to one logical CPU per
//! idle physical core, on one NUMA node when that node has enough idle cores,
//! with a few spare cores so the scheduler can still move a worker whose core
//! another process starts using. Workers otherwise migrate across sockets and
//! share cores with each other, which inflates the shared analysis state's
//! cache misses by tens of percent; pinning each worker to exactly one core
//! instead stalls every worker whenever one core is shared with a foreign
//! thread, because completions are pumped by whichever worker holds the turn.
//!
//! The mask is applied to the launching thread (workers inherit it) and
//! restored when the launch ends. `NUMSIM_WORKER_AFFINITY` selects the
//! policy: unset or `auto` chooses idle cores as above (no confinement when
//! the host has fewer idle cores than workers, exposes no topology, or the
//! launch has a single worker); an explicit CPU list such as `0,2,4-7` is
//! used as the mask verbatim; `off` leaves the scheduler alone. On the
//! 2-socket 224-thread benchmark host, t64 MegaMoE at 16 workers runs 7 %
//! faster confined on a lightly loaded host and 9 % faster with 80 foreign
//! spinning processes, because the spare cores let confined workers dodge
//! tenants that land on their cores. The test suite sets `off` because many
//! engine processes launch there at once and would all pick the same idle
//! cores.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use numsim_fp_env::{current_thread_allowed_cpus, set_current_thread_allowed_cpus};

pub(crate) const WORKER_AFFINITY_ENV: &str = "NUMSIM_WORKER_AFFINITY";
/// Thread-name prefix of the worker pool; the monitor finds workers by it.
pub(crate) const WORKER_THREAD_PREFIX: &str = "numsim-worker-";

/// A core counts as idle when none of its hardware threads was busier than
/// this fraction over the sampling window.
const IDLE_BUSY_LIMIT: f64 = 0.5;
/// `/proc/stat` advances in 10 ms jiffies; shorter windows carry no signal.
const MIN_SAMPLE_WINDOW: Duration = Duration::from_millis(50);
/// Share of the workers' runnable time spent waiting for a CPU above which
/// the confinement is released: foreign threads hold enough of the chosen
/// cores that the spare ones no longer absorb them.
const RELEASE_WAIT_SHARE: f64 = 0.10;
const MONITOR_INTERVAL: Duration = Duration::from_millis(500);
const MONITOR_STOP_POLL: Duration = Duration::from_millis(25);

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum WorkerAffinityPolicy {
    Off,
    Auto,
    Explicit(Vec<usize>),
}

impl WorkerAffinityPolicy {
    pub(crate) fn from_env() -> Self {
        Self::parse(std::env::var(WORKER_AFFINITY_ENV).ok().as_deref())
    }

    fn parse(value: Option<&str>) -> Self {
        match value.map(str::trim) {
            None | Some("") | Some("auto") => Self::Auto,
            Some("off") | Some("0") | Some("none") => Self::Off,
            Some(list) => match parse_cpu_list(list) {
                Some(cpus) if !cpus.is_empty() => Self::Explicit(cpus),
                _ => Self::Off,
            },
        }
    }
}

/// CPUs a launch's worker threads may use; `None` leaves them to the scheduler.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct WorkerAffinityPlan {
    cpus: Option<Vec<usize>>,
}

impl WorkerAffinityPlan {
    pub(crate) fn for_workers(policy: &WorkerAffinityPolicy, worker_count: usize) -> Self {
        let cpus = match policy {
            WorkerAffinityPolicy::Off => None,
            WorkerAffinityPolicy::Explicit(cpus) => {
                let mut cpus = cpus.clone();
                cpus.sort_unstable();
                cpus.dedup();
                Some(cpus)
            }
            WorkerAffinityPolicy::Auto => (worker_count >= 2)
                .then(host_cpus)
                .flatten()
                .map(|cpus| choose_worker_cpus(&cpus, worker_count, process_memory_node()))
                .filter(|chosen| chosen.len() >= worker_count),
        };
        Self { cpus }
    }

    pub(crate) fn cpus(&self) -> Option<&[usize]> {
        self.cpus.as_deref()
    }
}

/// Confines the calling thread to the plan's CPUs until dropped; threads
/// spawned meanwhile inherit the mask. A monitor thread spawned before the
/// confinement (so it keeps the inherited mask) watches the workers' runqueue
/// wait and releases them to the inherited mask when foreign threads crowd
/// the chosen cores. Best effort: a host that refuses the mask keeps the
/// inherited one.
#[must_use]
pub(crate) struct WorkerAffinityGuard {
    restore: Vec<usize>,
    release: Arc<WorkerAffinityRelease>,
    stop: Arc<AtomicBool>,
    monitor: Option<JoinHandle<()>>,
}

impl WorkerAffinityGuard {
    pub(crate) fn apply(plan: &WorkerAffinityPlan) -> Option<Self> {
        let cpus = plan.cpus()?;
        let restore = current_thread_allowed_cpus().ok()?;
        let _ = UNCONFINED_CPUS.set(restore.clone());
        let release = Arc::new(WorkerAffinityRelease {
            released: AtomicBool::new(false),
            inherited: restore.clone(),
        });
        let stop = Arc::new(AtomicBool::new(false));
        let monitor = std::thread::Builder::new()
            .name("numsim-affinity".to_string())
            .spawn({
                let release = Arc::clone(&release);
                let stop = Arc::clone(&stop);
                move || monitor_runqueue_wait(&release, &stop)
            })
            .ok();
        if set_current_thread_allowed_cpus(cpus).is_err() {
            stop.store(true, Ordering::Relaxed);
            if let Some(monitor) = monitor {
                let _ = monitor.join();
            }
            return None;
        }
        Some(Self {
            restore,
            release,
            stop,
            monitor,
        })
    }

    /// Handle the workers poll to learn that the confinement was released.
    pub(crate) fn release(&self) -> Arc<WorkerAffinityRelease> {
        Arc::clone(&self.release)
    }
}

impl Drop for WorkerAffinityGuard {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(monitor) = self.monitor.take() {
            let _ = monitor.join();
        }
        let _ = set_current_thread_allowed_cpus(&self.restore);
    }
}

/// The mask the workers were confined from, for a helper thread a confined
/// worker spawns (it inherits the worker's one core otherwise).
static UNCONFINED_CPUS: OnceLock<Vec<usize>> = OnceLock::new();

/// Widens the calling thread to the mask the workers were confined from;
/// nothing to do when no confinement was applied.
pub(crate) fn unconfine_current_thread() {
    if let Some(cpus) = UNCONFINED_CPUS.get() {
        let _ = set_current_thread_allowed_cpus(cpus);
    }
}

/// Shared between the monitor, which flips `released`, and the workers, which
/// widen their own masks when they see it.
pub(crate) struct WorkerAffinityRelease {
    released: AtomicBool,
    inherited: Vec<usize>,
}

impl WorkerAffinityRelease {
    /// Widens the calling worker's mask once the monitor released the
    /// confinement; `widened` remembers that it already happened.
    pub(crate) fn widen_current_thread(&self, widened: &mut bool) {
        if !*widened && self.released.load(Ordering::Relaxed) {
            *widened = true;
            let _ = set_current_thread_allowed_cpus(&self.inherited);
        }
    }

    #[cfg(test)]
    fn is_released(&self) -> bool {
        self.released.load(Ordering::Relaxed)
    }
}

/// Runs on the unconfined monitor thread until the guard drops or the
/// workers' runqueue wait over one interval exceeds `RELEASE_WAIT_SHARE`.
fn monitor_runqueue_wait(release: &WorkerAffinityRelease, stop: &AtomicBool) {
    let mut previous = None;
    while !stop.load(Ordering::Relaxed) {
        let deadline = Instant::now() + MONITOR_INTERVAL;
        while Instant::now() < deadline {
            if stop.load(Ordering::Relaxed) {
                return;
            }
            std::thread::sleep(MONITOR_STOP_POLL);
        }
        let Some(now) = worker_schedstat_totals() else {
            return;
        };
        if let Some(earlier) = previous {
            if should_release(earlier, now) {
                release.released.store(true, Ordering::Relaxed);
                return;
            }
        }
        previous = Some(now);
    }
}

/// `(on-cpu ns, runqueue-wait ns)` summed over the worker threads.
fn worker_schedstat_totals() -> Option<(u64, u64)> {
    let mut totals = (0u64, 0u64);
    let mut seen = false;
    for entry in std::fs::read_dir("/proc/self/task").ok()?.flatten() {
        let path = entry.path();
        let Ok(comm) = std::fs::read_to_string(path.join("comm")) else {
            continue;
        };
        if !comm.starts_with(WORKER_THREAD_PREFIX) {
            continue;
        }
        let Some((run, wait)) = std::fs::read_to_string(path.join("schedstat"))
            .ok()
            .as_deref()
            .and_then(parse_schedstat)
        else {
            continue;
        };
        totals.0 = totals.0.saturating_add(run);
        totals.1 = totals.1.saturating_add(wait);
        seen = true;
    }
    seen.then_some(totals)
}

/// `/proc/<pid>/task/<tid>/schedstat`: on-cpu ns, runqueue-wait ns, timeslices.
fn parse_schedstat(text: &str) -> Option<(u64, u64)> {
    let mut fields = text.split_whitespace().map(str::parse::<u64>);
    match (fields.next(), fields.next()) {
        (Some(Ok(run)), Some(Ok(wait))) => Some((run, wait)),
        _ => None,
    }
}

fn should_release(earlier: (u64, u64), now: (u64, u64)) -> bool {
    let run = now.0.saturating_sub(earlier.0);
    let wait = now.1.saturating_sub(earlier.1);
    let runnable = run.saturating_add(wait);
    runnable > 0 && wait as f64 / runnable as f64 > RELEASE_WAIT_SHARE
}

/// Spare cores beyond one per worker, so a worker whose core a foreign thread
/// starts using can move instead of dragging the launch.
fn mask_size(worker_count: usize) -> usize {
    worker_count + (worker_count / 4).max(2)
}

#[derive(Clone, Debug, PartialEq)]
struct HostCpu {
    cpu: usize,
    node: usize,
    /// Physical core identity, unique across packages.
    core: (usize, usize),
    /// Fraction of the sampling window this hardware thread was busy.
    busy: f64,
}

/// Pick one logical CPU per idle physical core, up to `mask_size`, preferring
/// the NUMA node holding most of the process's memory when it has an idle
/// core per worker (the kernel's global buffers were first-touched there),
/// otherwise the node with the most idle cores, and the less busy hardware
/// thread of each core. Returns fewer than `worker_count` CPUs when the host
/// cannot offer that many idle cores.
fn choose_worker_cpus(
    cpus: &[HostCpu],
    worker_count: usize,
    memory_node: Option<usize>,
) -> Vec<usize> {
    let mut cores: BTreeMap<(usize, usize), Vec<&HostCpu>> = BTreeMap::new();
    for cpu in cpus {
        cores.entry(cpu.core).or_default().push(cpu);
    }
    let mut idle_cores: Vec<(usize, usize, usize)> = cores
        .values()
        .filter(|threads| threads.iter().all(|thread| thread.busy <= IDLE_BUSY_LIMIT))
        .map(|threads| {
            let best = threads
                .iter()
                .min_by(|a, b| {
                    a.busy
                        .partial_cmp(&b.busy)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(a.cpu.cmp(&b.cpu))
                })
                .expect("a core has at least one hardware thread");
            (best.node, best.cpu, threads[0].core.0)
        })
        .collect();
    let mut per_node: BTreeMap<usize, usize> = BTreeMap::new();
    for (node, _, _) in &idle_cores {
        *per_node.entry(*node).or_default() += 1;
    }
    let most_idle = per_node
        .iter()
        .max_by_key(|(node, count)| (**count, std::cmp::Reverse(**node)))
        .map(|(node, _)| *node);
    let preferred = match memory_node {
        Some(node) if per_node.get(&node).is_some_and(|count| *count >= worker_count) => node,
        _ => match most_idle {
            Some(node) => node,
            None => return Vec::new(),
        },
    };
    idle_cores.sort_by_key(|(node, cpu, _)| (*node != preferred, *node, *cpu));
    let mut chosen: Vec<usize> = idle_cores
        .into_iter()
        .take(mask_size(worker_count))
        .map(|(_, cpu, _)| cpu)
        .collect();
    chosen.sort_unstable();
    chosen
}

fn host_cpus() -> Option<Vec<HostCpu>> {
    let allowed = current_thread_allowed_cpus().ok()?;
    if allowed.is_empty() {
        return None;
    }
    let nodes = read_numa_nodes();
    let busy = sample_busy_fractions();
    let mut cpus = Vec::with_capacity(allowed.len());
    for cpu in allowed {
        let core = read_core_identity(cpu)?;
        cpus.push(HostCpu {
            cpu,
            node: nodes.get(&cpu).copied().unwrap_or(0),
            core,
            busy: busy.get(cpu).copied().unwrap_or(0.0),
        });
    }
    Some(cpus)
}

/// NUMA node holding the largest share of this process's resident memory,
/// from `/proc/self/numa_maps` (`N<node>=<pages>` per mapping, weighted by
/// the mapping's page size).
fn process_memory_node() -> Option<usize> {
    let text = std::fs::read_to_string("/proc/self/numa_maps").ok()?;
    memory_node_from_numa_maps(&text)
}

fn memory_node_from_numa_maps(text: &str) -> Option<usize> {
    let mut per_node: BTreeMap<usize, u64> = BTreeMap::new();
    for line in text.lines() {
        let mut page_kb = 4u64;
        let mut pages = Vec::new();
        for field in line.split_whitespace() {
            if let Some(kb) = field.strip_prefix("kernelpagesize_kB=") {
                page_kb = kb.parse().unwrap_or(4);
            } else if let Some((node, count)) = field
                .strip_prefix('N')
                .and_then(|rest| rest.split_once('='))
            {
                if let (Ok(node), Ok(count)) = (node.parse::<usize>(), count.parse::<u64>()) {
                    pages.push((node, count));
                }
            }
        }
        for (node, count) in pages {
            *per_node.entry(node).or_default() += count.saturating_mul(page_kb);
        }
    }
    per_node
        .iter()
        .max_by_key(|(node, kb)| (**kb, std::cmp::Reverse(**node)))
        .map(|(node, _)| *node)
}

fn read_core_identity(cpu: usize) -> Option<(usize, usize)> {
    let topology = format!("/sys/devices/system/cpu/cpu{cpu}/topology");
    let package = read_usize(&format!("{topology}/physical_package_id"))?;
    let core = read_usize(&format!("{topology}/core_id"))?;
    Some((package, core))
}

fn read_usize(path: &str) -> Option<usize> {
    std::fs::read_to_string(path).ok()?.trim().parse().ok()
}

fn read_numa_nodes() -> BTreeMap<usize, usize> {
    let mut nodes = BTreeMap::new();
    let Ok(entries) = std::fs::read_dir("/sys/devices/system/node") else {
        return nodes;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(node) = name
            .to_str()
            .and_then(|name| name.strip_prefix("node"))
            .and_then(|id| id.parse::<usize>().ok())
        else {
            continue;
        };
        let Ok(list) = std::fs::read_to_string(entry.path().join("cpulist")) else {
            continue;
        };
        for cpu in parse_cpu_list(list.trim()).unwrap_or_default() {
            nodes.insert(cpu, node);
        }
    }
    nodes
}

/// `0,2,4-7` → `[0, 2, 4, 5, 6, 7]`; `None` on any malformed part.
fn parse_cpu_list(list: &str) -> Option<Vec<usize>> {
    let mut cpus = Vec::new();
    for part in list.split(',').map(str::trim).filter(|part| !part.is_empty()) {
        match part.split_once('-') {
            Some((lo, hi)) => {
                let lo: usize = lo.trim().parse().ok()?;
                let hi: usize = hi.trim().parse().ok()?;
                if hi < lo {
                    return None;
                }
                cpus.extend(lo..=hi);
            }
            None => cpus.push(part.parse().ok()?),
        }
    }
    Some(cpus)
}

/// Per-CPU `(busy, total)` jiffies since boot.
#[derive(Clone, Debug)]
struct StatSample {
    at: Instant,
    jiffies: Vec<(u64, u64)>,
}

fn read_stat_sample() -> Option<StatSample> {
    let text = std::fs::read_to_string("/proc/stat").ok()?;
    Some(StatSample {
        at: Instant::now(),
        jiffies: parse_stat_jiffies(&text),
    })
}

/// Per-CPU `(busy, total)` jiffies from `/proc/stat` text. The aggregate
/// `cpu ` line and CPUs beyond the affinity mask width are skipped.
fn parse_stat_jiffies(text: &str) -> Vec<(u64, u64)> {
    let mut jiffies = Vec::new();
    for line in text.lines() {
        let Some((label, rest)) = line.split_once(' ') else {
            continue;
        };
        let Some(index) = label
            .strip_prefix("cpu")
            .filter(|id| !id.is_empty())
            .and_then(|id| id.parse::<usize>().ok())
            .filter(|index| *index < numsim_fp_env::MAX_AFFINITY_CPUS)
        else {
            continue;
        };
        let values: Vec<u64> = rest
            .split_whitespace()
            .filter_map(|field| field.parse().ok())
            .collect();
        if values.len() < 5 {
            continue;
        }
        let total: u64 = values.iter().sum();
        let idle = values[3] + values[4];
        if jiffies.len() <= index {
            jiffies.resize(index + 1, (0, 0));
        }
        jiffies[index] = (total - idle, total);
    }
    jiffies
}

fn busy_between(earlier: &StatSample, later: &StatSample) -> Vec<f64> {
    later
        .jiffies
        .iter()
        .enumerate()
        .map(|(cpu, &(busy, total))| {
            let (busy0, total0) = earlier.jiffies.get(cpu).copied().unwrap_or((0, 0));
            let elapsed = total.saturating_sub(total0);
            if elapsed == 0 {
                0.0
            } else {
                busy.saturating_sub(busy0) as f64 / elapsed as f64
            }
        })
        .collect()
}

static BUSY_CACHE: Mutex<Option<(StatSample, Vec<f64>)>> = Mutex::new(None);

/// Takes the first `/proc/stat` sample so the first launch can measure the
/// window since import instead of sleeping for one. Called from the artifact
/// module's import hook; harmless to call more than once.
pub fn prime_host_load_sample() {
    let mut cache = BUSY_CACHE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if cache.is_none() {
        if let Some(sample) = read_stat_sample() {
            *cache = Some((sample, Vec::new()));
        }
    }
}

/// Busy fraction per CPU over the window since the previous sample (at least
/// `MIN_SAMPLE_WINDOW`; a launch closer than that to the previous sample
/// reuses the previous fractions, or sleeps out the window when there are
/// none yet).
fn sample_busy_fractions() -> Vec<f64> {
    let mut cache = BUSY_CACHE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let Some(mut now) = read_stat_sample() else {
        return Vec::new();
    };
    let (previous, fractions) = cache.get_or_insert_with(|| (now.clone(), Vec::new()));
    let elapsed = now.at.duration_since(previous.at);
    if elapsed < MIN_SAMPLE_WINDOW {
        if !fractions.is_empty() {
            return fractions.clone();
        }
        std::thread::sleep(MIN_SAMPLE_WINDOW - elapsed);
        let Some(later) = read_stat_sample() else {
            return Vec::new();
        };
        now = later;
    }
    *fractions = busy_between(previous, &now);
    *previous = now;
    fractions.clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu(cpu: usize, node: usize, core: usize, busy: f64) -> HostCpu {
        HostCpu {
            cpu,
            node,
            core: (node, core),
            busy,
        }
    }

    /// Two nodes with two cores each; every core has two hardware threads
    /// (`cpu` and `cpu + 8`).
    fn host(busy: impl Fn(usize) -> f64) -> Vec<HostCpu> {
        (0..4)
            .flat_map(|core| {
                let node = core / 2;
                [cpu(core, node, core, busy(core)), cpu(core + 8, node, core, busy(core + 8))]
            })
            .collect()
    }

    #[test]
    fn policy_parses_auto_off_and_explicit_lists() {
        assert_eq!(WorkerAffinityPolicy::parse(None), WorkerAffinityPolicy::Auto);
        assert_eq!(WorkerAffinityPolicy::parse(Some("")), WorkerAffinityPolicy::Auto);
        assert_eq!(WorkerAffinityPolicy::parse(Some(" auto ")), WorkerAffinityPolicy::Auto);
        assert_eq!(WorkerAffinityPolicy::parse(Some("off")), WorkerAffinityPolicy::Off);
        assert_eq!(WorkerAffinityPolicy::parse(Some("0")), WorkerAffinityPolicy::Off);
        assert_eq!(
            WorkerAffinityPolicy::parse(Some("3,1,4-6")),
            WorkerAffinityPolicy::Explicit(vec![3, 1, 4, 5, 6])
        );
        assert_eq!(WorkerAffinityPolicy::parse(Some("4-2")), WorkerAffinityPolicy::Off);
        assert_eq!(WorkerAffinityPolicy::parse(Some("x")), WorkerAffinityPolicy::Off);
    }

    #[test]
    fn explicit_lists_become_the_mask_and_off_confines_nothing() {
        let plan =
            WorkerAffinityPlan::for_workers(&WorkerAffinityPolicy::Explicit(vec![7, 5, 7]), 3);
        assert_eq!(plan.cpus(), Some(&[5, 7][..]));
        let plan = WorkerAffinityPlan::for_workers(&WorkerAffinityPolicy::Off, 3);
        assert_eq!(plan.cpus(), None);
    }

    #[test]
    fn auto_prefers_the_node_with_more_idle_cores_and_keeps_spare_cores() {
        // Core 0 (node 0) is busy on one hardware thread; node 1 has two idle cores.
        let cpus = host(|cpu| if cpu == 8 { 0.9 } else { 0.0 });
        // Two workers may use up to four cores: node 1 first, then node 0's idle core.
        assert_eq!(choose_worker_cpus(&cpus, 2, None), vec![1, 2, 3]);
        // Nothing beyond the three idle cores: four workers stay unconfined.
        assert_eq!(choose_worker_cpus(&cpus, 4, None).len(), 3);
        // The memory node wins when it has an idle core per worker …
        assert_eq!(choose_worker_cpus(&cpus, 1, Some(0)), vec![1, 2, 3]);
        // … and is ignored when it does not.
        assert_eq!(choose_worker_cpus(&cpus, 2, Some(0)), vec![1, 2, 3]);
        assert_eq!(mask_size(16), 20);
        assert_eq!(mask_size(2), 4);
    }

    #[test]
    fn auto_uses_the_less_busy_hardware_thread_of_a_core() {
        let cpus = host(|cpu| if cpu < 8 { 0.3 } else { 0.1 });
        assert_eq!(choose_worker_cpus(&cpus, 4, None), vec![8, 9, 10, 11]);
    }

    #[test]
    fn auto_ties_pick_the_lower_node_first() {
        let cpus = host(|_| 0.0);
        assert_eq!(choose_worker_cpus(&cpus, 1, None), vec![0, 1, 2]);
        assert_eq!(choose_worker_cpus(&cpus, 1, Some(1)), vec![0, 2, 3]);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn auto_plan_on_this_host_is_well_formed() {
        let started = Instant::now();
        let plan = WorkerAffinityPlan::for_workers(&WorkerAffinityPolicy::Auto, 16);
        let elapsed = started.elapsed();
        eprintln!("auto plan for 16 workers on this host ({elapsed:?}): {:?}", plan.cpus());
        // Two /proc/stat samples one window apart plus sysfs reads; anything
        // slower means the plan is doing work proportional to jiffy counts.
        assert!(elapsed < Duration::from_secs(2), "plan took {elapsed:?}");
        if let Some(cpus) = plan.cpus() {
            assert!(cpus.len() >= 16 && cpus.len() <= mask_size(16));
            assert!(cpus.windows(2).all(|pair| pair[0] < pair[1]));
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn guard_restores_the_inherited_mask() {
        let Ok(inherited) = current_thread_allowed_cpus() else {
            return;
        };
        let plan = WorkerAffinityPlan {
            cpus: Some(vec![inherited[0]]),
        };
        let guard = WorkerAffinityGuard::apply(&plan).expect("own cpu is allowed");
        assert_eq!(current_thread_allowed_cpus().unwrap(), vec![inherited[0]]);
        drop(guard);
        assert_eq!(current_thread_allowed_cpus().unwrap(), inherited);
    }

    #[test]
    fn stat_parser_skips_the_aggregate_line_and_reads_per_cpu_jiffies() {
        let text = "cpu  420576696 10 20 30 40 5 6 0 0 0\n\
                    cpu0 100 0 20 800 80 0 0 0 0 0\n\
                    cpu2 5 0 5 90 0 0 0 0 0 0\n\
                    intr 1 2 3\n\
                    ctxt 99\n";
        assert_eq!(
            parse_stat_jiffies(text),
            vec![(120, 1000), (0, 0), (10, 100)]
        );
    }

    #[test]
    fn stat_busy_fractions_ignore_cpus_without_progress() {
        let earlier = StatSample {
            at: Instant::now(),
            jiffies: vec![(10, 100), (50, 100)],
        };
        let later = StatSample {
            at: Instant::now(),
            jiffies: vec![(20, 120), (50, 100), (5, 10)],
        };
        assert_eq!(busy_between(&earlier, &later), vec![0.5, 0.0, 0.5]);
    }

    #[test]
    fn release_triggers_on_runqueue_wait_share() {
        assert!(!should_release((0, 0), (1_000, 50)));
        assert!(should_release((0, 0), (1_000, 200)));
        assert!(!should_release((0, 0), (0, 0)));
        // Threads that exited between samples cannot produce a spurious release.
        assert!(!should_release((5_000, 5_000), (1_000, 1_000)));
        assert_eq!(parse_schedstat("913458 12 1\n"), Some((913458, 12)));
        assert_eq!(parse_schedstat("x"), None);
    }

    #[test]
    fn workers_widen_once_after_a_release() {
        let release = WorkerAffinityRelease {
            released: AtomicBool::new(false),
            inherited: current_thread_allowed_cpus().unwrap_or_default(),
        };
        let mut widened = false;
        release.widen_current_thread(&mut widened);
        assert!(!widened && !release.is_released());
        release.released.store(true, Ordering::Relaxed);
        release.widen_current_thread(&mut widened);
        assert!(widened);
    }

    #[test]
    fn memory_node_is_the_one_with_most_resident_kilobytes() {
        let text = "7f00 default anon=10 dirty=10 N0=8 N1=2 kernelpagesize_kB=4\n\
                    7f10 default file=/x mapped=3 N0=3 kernelpagesize_kB=4\n\
                    7f20 default anon=1 dirty=1 N1=1 kernelpagesize_kB=2048\n";
        assert_eq!(memory_node_from_numa_maps(text), Some(1));
        assert_eq!(memory_node_from_numa_maps("7f00 default\n"), None);
    }

    #[test]
    fn cpu_lists_reject_malformed_parts() {
        assert_eq!(parse_cpu_list("0-1, 4"), Some(vec![0, 1, 4]));
        assert_eq!(parse_cpu_list(""), Some(Vec::new()));
        assert_eq!(parse_cpu_list("1-"), None);
        assert_eq!(parse_cpu_list("a"), None);
    }
}
