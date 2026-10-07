use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::task::{Wake, Waker};

/// Scheduler class attached to one wake call.
///
/// The hint is thread-local to the call stack, not to the task: a semantic-
/// progress watch may fire on another worker, wraps the destination waker, and
/// establishes this class only while forwarding that wake. Unrelated barrier
/// or completion wakes remain normal and can promote the same warp.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum WakePriority {
    #[default]
    Normal,
    PollRecheck,
}

/// Stable order within one scheduler priority class.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReadyOrder {
    LowestId,
    Fifo,
}

/// Shared runnable-ID queue used by concrete and controlled executors.
///
/// Normal wakes always run before semantic-progress poll rechecks. A normal
/// wake promotes an ID that is already queued for a poll recheck; a later
/// poll-recheck wake never demotes normal work. Retired IDs cannot be woken.
pub(crate) struct PriorityReadyQueue {
    state: Mutex<PriorityReadyState>,
}

struct PriorityReadyState {
    order: ReadyOrder,
    normal: ReadyBucket,
    poll_recheck: ReadyBucket,
    queued: BTreeMap<usize, WakePriority>,
    active: BTreeSet<usize>,
}

#[derive(Default)]
struct ReadyBucket {
    ids: BTreeSet<usize>,
    fifo: VecDeque<usize>,
}

impl ReadyBucket {
    fn insert(&mut self, id: usize, order: ReadyOrder, front: bool) {
        assert!(self.ids.insert(id), "ready ID must not be queued twice");
        if order == ReadyOrder::Fifo {
            if front {
                self.fifo.push_front(id);
            } else {
                self.fifo.push_back(id);
            }
        }
    }

    fn remove(&mut self, id: usize) {
        // FIFO entries are removed lazily by `pop`; keeping promotion and
        // retirement O(log n) avoids scanning a polling wave on every wake.
        self.ids.remove(&id);
    }

    fn pop(&mut self, order: ReadyOrder) -> Option<usize> {
        match order {
            ReadyOrder::LowestId => {
                let id = self.ids.iter().next().copied()?;
                self.ids.remove(&id);
                Some(id)
            }
            ReadyOrder::Fifo => loop {
                let id = self.fifo.pop_front()?;
                if self.ids.remove(&id) {
                    return Some(id);
                }
            },
        }
    }

    fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

impl PriorityReadyQueue {
    pub(crate) fn new(ids: impl IntoIterator<Item = usize>, order: ReadyOrder) -> Self {
        let active = ids.into_iter().collect::<BTreeSet<_>>();
        let mut normal = ReadyBucket::default();
        for id in active.iter().copied() {
            normal.insert(id, order, false);
        }
        Self {
            state: Mutex::new(PriorityReadyState {
                order,
                normal,
                poll_recheck: ReadyBucket::default(),
                queued: active
                    .iter()
                    .copied()
                    .map(|id| (id, WakePriority::Normal))
                    .collect(),
                active,
            }),
        }
    }

    /// Queue one active ID and return the effective newly queued priority.
    ///
    /// `Some(Normal)` also denotes promotion from `PollRecheck`; `None` means
    /// that the ID was retired or was already queued at equal/higher priority.
    pub(crate) fn schedule(&self, id: usize, priority: WakePriority) -> Option<WakePriority> {
        let mut state = self.state.lock().expect("priority ready queue poisoned");
        if !state.active.contains(&id) {
            return None;
        }
        match state.queued.get(&id).copied() {
            None => {
                let order = state.order;
                match priority {
                    WakePriority::Normal => state.normal.insert(id, order, false),
                    WakePriority::PollRecheck => state.poll_recheck.insert(id, order, false),
                }
                state.queued.insert(id, priority);
                Some(priority)
            }
            Some(WakePriority::PollRecheck) if priority == WakePriority::Normal => {
                state.poll_recheck.remove(id);
                let order = state.order;
                state.normal.insert(id, order, false);
                state.queued.insert(id, WakePriority::Normal);
                Some(WakePriority::Normal)
            }
            Some(_) => None,
        }
    }

    /// Activate a newly claimed scheduling domain ahead of existing normal
    /// FIFO work. Lowest-ID queues ignore the placement hint by definition.
    pub(crate) fn activate_normal_front(&self, id: usize) {
        let mut state = self.state.lock().expect("priority ready queue poisoned");
        assert!(state.active.insert(id), "new ready ID must be inactive");
        assert!(
            state.queued.insert(id, WakePriority::Normal).is_none(),
            "new ready ID must not already be queued"
        );
        let order = state.order;
        state.normal.insert(id, order, true);
    }

    pub(crate) fn pop(&self) -> Option<usize> {
        self.pop_with_priority().map(|(id, _)| id)
    }

    pub(crate) fn pop_with_priority(&self) -> Option<(usize, WakePriority)> {
        let mut state = self.state.lock().expect("priority ready queue poisoned");
        let order = state.order;
        let (id, priority) = if let Some(id) = state.normal.pop(order) {
            (id, WakePriority::Normal)
        } else {
            (state.poll_recheck.pop(order)?, WakePriority::PollRecheck)
        };
        state.queued.remove(&id);
        Some((id, priority))
    }

    pub(crate) fn contains(&self, id: usize) -> bool {
        self.state
            .lock()
            .expect("priority ready queue poisoned")
            .queued
            .contains_key(&id)
    }

    pub(crate) fn retire(&self, id: usize) {
        let mut state = self.state.lock().expect("priority ready queue poisoned");
        state.active.remove(&id);
        if let Some(priority) = state.queued.remove(&id) {
            match priority {
                WakePriority::Normal => state.normal.remove(id),
                WakePriority::PollRecheck => state.poll_recheck.remove(id),
            }
        }
    }

    pub(crate) fn highest_priority(&self) -> Option<WakePriority> {
        let state = self.state.lock().expect("priority ready queue poisoned");
        if !state.normal.is_empty() {
            Some(WakePriority::Normal)
        } else if !state.poll_recheck.is_empty() {
            Some(WakePriority::PollRecheck)
        } else {
            None
        }
    }

    pub(crate) fn has_ready(&self) -> bool {
        self.highest_priority().is_some()
    }
}

thread_local! {
    static WAKE_PRIORITY: Cell<WakePriority> = const { Cell::new(WakePriority::Normal) };
}

pub(crate) fn current_wake_priority() -> WakePriority {
    WAKE_PRIORITY.get()
}

fn wake_by_ref_with_priority(waker: &Waker, priority: WakePriority) {
    let previous = WAKE_PRIORITY.replace(priority);
    struct RestoreWakePriority(WakePriority);

    impl Drop for RestoreWakePriority {
        fn drop(&mut self) {
            WAKE_PRIORITY.set(self.0);
        }
    }

    let _restore = RestoreWakePriority(previous);
    waker.wake_by_ref();
}

struct PollRecheckWake {
    inner: Waker,
}

impl PollRecheckWake {
    fn forward(&self) {
        wake_by_ref_with_priority(&self.inner, WakePriority::PollRecheck);
    }
}

impl Wake for PollRecheckWake {
    fn wake(self: Arc<Self>) {
        self.forward();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.forward();
    }
}

/// Wrap a waiter waker so semantic-progress notifications enter the executor's
/// low-priority poll-recheck queue.
pub(crate) fn poll_recheck_waker(waker: &Waker) -> Waker {
    Waker::from(Arc::new(PollRecheckWake {
        inner: waker.clone(),
    }))
}

/// Cooperatively place the current warp behind its already-runnable peers.
///
/// The first poll wakes the current task and returns [`Poll::Pending`]. The
/// cluster executor's waker appends that warp to its cluster-local FIFO. The
/// next poll completes, resuming native Rust control flow with locals intact.
#[must_use = "futures do nothing unless awaited or polled"]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Reschedule {
    scheduled: bool,
    priority: WakePriority,
}

/// Create one cooperative scheduling checkpoint for the current warp.
pub const fn reschedule() -> Reschedule {
    Reschedule {
        scheduled: false,
        priority: WakePriority::Normal,
    }
}

/// Create a low-priority checkpoint for a native polling-loop recheck.
pub(crate) const fn poll_recheck_reschedule() -> Reschedule {
    Reschedule {
        scheduled: false,
        priority: WakePriority::PollRecheck,
    }
}

impl Future for Reschedule {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        if self.scheduled {
            Poll::Ready(())
        } else {
            self.scheduled = true;
            wake_by_ref_with_priority(context.waker(), self.priority);
            Poll::Pending
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::task::{Wake, Waker};

    #[derive(Default)]
    struct WakeCounter(AtomicUsize);

    impl Wake for WakeCounter {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }

        fn wake_by_ref(self: &Arc<Self>) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    struct PriorityRecorder(AtomicU8);

    impl Wake for PriorityRecorder {
        fn wake(self: Arc<Self>) {
            self.wake_by_ref();
        }

        fn wake_by_ref(self: &Arc<Self>) {
            let priority = match current_wake_priority() {
                WakePriority::Normal => 1,
                WakePriority::PollRecheck => 2,
            };
            self.0.store(priority, Ordering::SeqCst);
        }
    }

    #[test]
    fn reschedule_self_wakes_once_then_completes() {
        let wake_count = Arc::new(WakeCounter::default());
        let waker = Waker::from(Arc::clone(&wake_count));
        let mut context = Context::from_waker(&waker);
        let mut checkpoint = Box::pin(reschedule());

        assert_eq!(checkpoint.as_mut().poll(&mut context), Poll::Pending);
        assert_eq!(wake_count.0.load(Ordering::SeqCst), 1);
        assert_eq!(checkpoint.as_mut().poll(&mut context), Poll::Ready(()));
        assert_eq!(wake_count.0.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn poll_recheck_reschedule_self_wakes_at_low_priority() {
        let recorded = Arc::new(PriorityRecorder(AtomicU8::new(0)));
        let waker = Waker::from(Arc::clone(&recorded));
        let mut context = Context::from_waker(&waker);
        let mut checkpoint = Box::pin(poll_recheck_reschedule());

        assert_eq!(checkpoint.as_mut().poll(&mut context), Poll::Pending);
        assert_eq!(recorded.0.load(Ordering::SeqCst), 2);
        assert_eq!(current_wake_priority(), WakePriority::Normal);
        assert_eq!(checkpoint.as_mut().poll(&mut context), Poll::Ready(()));
    }

    #[test]
    fn poll_recheck_waker_tags_only_the_forwarded_wake() {
        let recorded = Arc::new(PriorityRecorder(AtomicU8::new(0)));
        let normal = Waker::from(Arc::clone(&recorded));
        let poll_recheck = poll_recheck_waker(&normal);

        poll_recheck.wake_by_ref();
        assert_eq!(recorded.0.load(Ordering::SeqCst), 2);
        assert_eq!(current_wake_priority(), WakePriority::Normal);

        normal.wake_by_ref();
        assert_eq!(recorded.0.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn normal_work_runs_before_an_older_poll_recheck() {
        let ready = PriorityReadyQueue::new([0, 1], ReadyOrder::Fifo);
        assert_eq!(ready.pop(), Some(0));
        assert_eq!(ready.pop(), Some(1));

        assert_eq!(
            ready.schedule(0, WakePriority::PollRecheck),
            Some(WakePriority::PollRecheck)
        );
        assert_eq!(
            ready.schedule(1, WakePriority::Normal),
            Some(WakePriority::Normal)
        );
        assert_eq!(ready.pop_with_priority(), Some((1, WakePriority::Normal)));
        assert_eq!(
            ready.pop_with_priority(),
            Some((0, WakePriority::PollRecheck))
        );
    }

    #[test]
    fn normal_wake_promotes_but_poll_recheck_never_demotes() {
        let ready = PriorityReadyQueue::new([0], ReadyOrder::Fifo);
        assert_eq!(ready.pop(), Some(0));
        assert_eq!(
            ready.schedule(0, WakePriority::PollRecheck),
            Some(WakePriority::PollRecheck)
        );
        assert_eq!(
            ready.schedule(0, WakePriority::Normal),
            Some(WakePriority::Normal)
        );
        assert_eq!(ready.schedule(0, WakePriority::PollRecheck), None);
        assert_eq!(ready.pop_with_priority(), Some((0, WakePriority::Normal)));
    }
}
