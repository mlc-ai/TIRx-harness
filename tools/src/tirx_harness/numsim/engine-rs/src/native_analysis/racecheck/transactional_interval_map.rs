use std::collections::BTreeMap;
use std::fmt;

const EMPTY_CELL: u32 = u32::MAX;
const EMPTY_NARROW_CELL: u16 = u16::MAX;
const FREE_ENTRY_START: usize = usize::MAX;
const PAGE_SHIFT: usize = 12;
const PAGE_BYTES: usize = 1 << PAGE_SHIFT;
const PAGE_MASK: usize = PAGE_BYTES - 1;
const MIN_INDEXED_BYTES: usize = 1024 * 1024;
const MAX_INDEXED_BYTES: usize = 4 * 1024 * 1024;
const MAX_BYTES_PER_ENTRY: usize = 512;

#[derive(Clone)]
struct IndexedEntry<T> {
    start: usize,
    value: T,
}

impl<T> IndexedEntry<T> {
    #[inline(always)]
    fn is_live(&self) -> bool {
        self.start != FREE_ENTRY_START
    }
}

#[derive(Clone)]
struct IndexedPage {
    base: u32,
    cells: Box<[u16; PAGE_BYTES]>,
}

impl IndexedPage {
    fn empty(base: u32) -> Self {
        debug_assert_ne!(base, EMPTY_CELL);
        Self {
            base: if base < u32::from(EMPTY_NARROW_CELL) {
                0
            } else {
                base
            },
            cells: Box::new([EMPTY_NARROW_CELL; PAGE_BYTES]),
        }
    }

    #[inline(always)]
    fn entry_index(&self, offset: usize) -> Option<usize> {
        let value = self.cells[offset];
        if value == EMPTY_NARROW_CELL {
            None
        } else {
            Some((self.base + u32::from(value)) as usize)
        }
    }

    fn first_occupied(&self, start: usize, end: usize) -> Option<usize> {
        self.cells[start..end]
            .iter()
            .position(|cell| *cell != EMPTY_NARROW_CELL)
    }

    #[inline(always)]
    fn fill(&mut self, start: usize, end: usize, value: u32) -> Option<Box<[u32; PAGE_BYTES]>> {
        if value == EMPTY_CELL {
            self.cells[start..end].fill(EMPTY_NARROW_CELL);
            return None;
        }
        if value >= self.base {
            let delta = value - self.base;
            if delta < u32::from(EMPTY_NARROW_CELL) {
                self.cells[start..end].fill(delta as u16);
                return None;
            }
        }
        self.rebase_or_widen(start, end, value)
    }

    #[cold]
    #[inline(never)]
    fn rebase_or_widen(
        &mut self,
        start: usize,
        end: usize,
        value: u32,
    ) -> Option<Box<[u32; PAGE_BYTES]>> {
        debug_assert_ne!(value, EMPTY_CELL);
        let old_base = self.base;
        let mut minimum = value;
        let mut maximum = value;
        for cell in self.cells.iter().copied() {
            if cell == EMPTY_NARROW_CELL {
                continue;
            }
            let index = old_base + u32::from(cell);
            minimum = minimum.min(index);
            maximum = maximum.max(index);
        }
        if maximum - minimum < u32::from(EMPTY_NARROW_CELL) {
            for cell in self.cells.iter_mut() {
                if *cell != EMPTY_NARROW_CELL {
                    let index = old_base + u32::from(*cell);
                    *cell = (index - minimum) as u16;
                }
            }
            self.base = minimum;
            self.cells[start..end].fill((value - minimum) as u16);
            return None;
        }

        let mut wide = Box::new([EMPTY_CELL; PAGE_BYTES]);
        for (destination, source) in wide.iter_mut().zip(self.cells.iter()) {
            if *source != EMPTY_NARROW_CELL {
                *destination = old_base + u32::from(*source);
            }
        }
        wide[start..end].fill(value);
        Some(wide)
    }

    fn any_occupied(&self, start: usize, end: usize) -> bool {
        self.cells[start..end]
            .iter()
            .any(|cell| *cell != EMPTY_NARROW_CELL)
    }
}

#[derive(Clone)]
pub(crate) struct IndexedEntries<T> {
    base_page: usize,
    pages: Vec<Option<IndexedPage>>,
    wide_pages: BTreeMap<usize, Box<[u32; PAGE_BYTES]>>,
    entries: Vec<IndexedEntry<T>>,
    free_entries: Vec<u32>,
    live_len: usize,
}

impl<T> Default for IndexedEntries<T> {
    fn default() -> Self {
        Self {
            base_page: 0,
            pages: Vec::new(),
            wide_pages: BTreeMap::new(),
            entries: Vec::new(),
            free_entries: Vec::new(),
            live_len: 0,
        }
    }
}

impl<T> IndexedEntries<T> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.live_len
    }

    #[inline(always)]
    fn entry_index_at(&self, byte_offset: usize) -> Option<usize> {
        let page = byte_offset >> PAGE_SHIFT;
        let page_index = page.checked_sub(self.base_page)?;
        if let Some(page) = self.pages.get(page_index)?.as_ref() {
            return page.entry_index(byte_offset & PAGE_MASK);
        }
        self.wide_entry_index_at(page, byte_offset & PAGE_MASK)
    }

    #[cold]
    #[inline(never)]
    fn wide_entry_index_at(&self, page: usize, offset: usize) -> Option<usize> {
        let value = self.wide_pages.get(&page)?[offset];
        (value != EMPTY_CELL).then_some(value as usize)
    }

    #[inline(always)]
    fn get(&self, start: usize) -> Option<&T> {
        let entry = self.entries.get(self.entry_index_at(start)?)?;
        (entry.is_live() && entry.start == start).then_some(&entry.value)
    }

    #[inline(always)]
    fn get_mut(&mut self, start: usize) -> Option<&mut T> {
        let index = self.entry_index_at(start)?;
        let entry = self.entries.get_mut(index)?;
        (entry.is_live() && entry.start == start).then_some(&mut entry.value)
    }

    /// Ensure the complete byte range is indexable.
    fn prepare_span(&mut self, start: usize, end: usize) -> bool {
        self.ensure_range(start, end)
    }

    fn insert(&mut self, start: usize, end: usize, value: T) -> Result<Option<T>, T> {
        if let Some(existing) = self.get_mut(start) {
            return Ok(Some(std::mem::replace(existing, value)));
        }
        if (self.entries.len() >= EMPTY_CELL as usize && self.free_entries.is_empty())
            || !self.ensure_range(start, end)
        {
            return Err(value);
        }
        Ok(self.insert_after_prepare(start, end, value))
    }

    fn insert_after_prepare(&mut self, start: usize, end: usize, value: T) -> Option<T> {
        if let Some(existing) = self.get_mut(start) {
            return Some(std::mem::replace(existing, value));
        }
        debug_assert!(self.entries.len() < EMPTY_CELL as usize || !self.free_entries.is_empty());
        debug_assert!(self.range_is_indexable(start, end));
        let entry_index = if let Some(entry_index) = self.free_entries.pop() {
            self.entries[entry_index as usize] = IndexedEntry { start, value };
            entry_index
        } else {
            let entry_index = self.entries.len() as u32;
            self.entries.push(IndexedEntry { start, value });
            entry_index
        };
        self.live_len += 1;
        self.fill_range(start, end, entry_index);
        None
    }

    fn replace_one<FStart, FEnd, FSplit>(
        &mut self,
        start: usize,
        end: usize,
        value: T,
        start_of: FStart,
        end_of: FEnd,
        split: FSplit,
    ) -> Result<(), T>
    where
        FStart: Fn(&T) -> usize,
        FEnd: Fn(&T) -> usize,
        FSplit: Fn(&T, usize, usize) -> T,
    {
        let Some(index) = self.entry_index_at(start) else {
            return Err(value);
        };
        let entry = &self.entries[index];
        let entry_end = end_of(&entry.value);
        if !entry.is_live() || entry.start > start || end > entry_end {
            return Err(value);
        }
        let entry_start = entry.start;
        let left = (entry_start < start).then(|| split(&entry.value, entry_start, start));
        let right = (end < entry_end).then(|| split(&entry.value, end, entry_end));
        debug_assert_eq!(start_of(&value), start);
        debug_assert_eq!(end_of(&value), end);

        match (left, right) {
            (None, None) => {
                self.entries[index].value = value;
            }
            (Some(left), None) => {
                let entry = &mut self.entries[index];
                entry.value = left;
                self.insert_overwriting_range(start, end, value);
            }
            (None, Some(right)) => {
                let entry = &mut self.entries[index];
                entry.start = end;
                entry.value = right;
                self.insert_overwriting_range(start, end, value);
            }
            (Some(left), Some(right)) => {
                if start - entry_start >= entry_end - end {
                    let entry = &mut self.entries[index];
                    entry.value = left;
                    self.insert_overwriting_range(end, entry_end, right);
                } else {
                    let entry = &mut self.entries[index];
                    entry.start = end;
                    entry.value = right;
                    self.insert_overwriting_range(entry_start, start, left);
                }
                self.insert_overwriting_range(start, end, value);
            }
        }
        Ok(())
    }

    fn replace_range<FStart, FEnd, FSplit>(
        &mut self,
        start: usize,
        end: usize,
        mut replacement: Vec<T>,
        start_of: FStart,
        end_of: FEnd,
        split: FSplit,
    ) -> Result<(), Vec<T>>
    where
        FStart: Fn(&T) -> usize,
        FEnd: Fn(&T) -> usize,
        FSplit: Fn(&T, usize, usize) -> T,
    {
        if !self.ensure_range(start, end) {
            return Err(replacement);
        }
        if replacement.len() == 1 {
            let value = replacement.pop().expect("a single replacement is available");
            match self.replace_one(start, end, value, &start_of, &end_of, &split) {
                Ok(()) => return Ok(()),
                Err(value) => replacement.push(value),
            }
        }
        let mut overlapping = Vec::new();
        let mut cursor = start;
        while cursor < end {
            if let Some(index) = self.entry_index_at(cursor) {
                overlapping.push(index as u32);
                cursor = end_of(&self.entries[index].value).min(end);
            } else {
                cursor += 1;
            }
        }

        let left = overlapping.first().and_then(|index| {
            let entry = &self.entries[*index as usize];
            (entry.start < start).then(|| split(&entry.value, entry.start, start))
        });
        let right = overlapping.last().and_then(|index| {
            let entry = &self.entries[*index as usize];
            let entry_end = end_of(&entry.value);
            (entry_end > end).then(|| split(&entry.value, end, entry_end))
        });

        for index in overlapping {
            let (entry_start, entry_end) = {
                let entry = &mut self.entries[index as usize];
                debug_assert!(entry.is_live());
                let range = (entry.start, end_of(&entry.value));
                // A valid half-open interval can never start at usize::MAX, so
                // the impossible start is an unambiguous tombstone. The end
                // already lives in T and need not be duplicated in every hot
                // indexed entry.
                entry.start = FREE_ENTRY_START;
                range
            };
            self.live_len -= 1;
            self.fill_range(entry_start, entry_end, EMPTY_CELL);
            self.free_entries.push(index);
        }
        if let Some(value) = left {
            self.insert_known_range(start_of(&value), end_of(&value), value);
        }
        for value in replacement {
            self.insert_known_range(start_of(&value), end_of(&value), value);
        }
        if let Some(value) = right {
            self.insert_known_range(start_of(&value), end_of(&value), value);
        }
        Ok(())
    }

    fn insert_known_range(&mut self, start: usize, end: usize, value: T) {
        debug_assert!(start < end);
        debug_assert!(self.range_is_empty(start, end));
        self.insert_overwriting_range(start, end, value);
    }

    fn insert_overwriting_range(&mut self, start: usize, end: usize, value: T) {
        debug_assert!(start < end);
        let index = if let Some(index) = self.free_entries.pop() {
            self.entries[index as usize] = IndexedEntry { start, value };
            index
        } else {
            let index = u32::try_from(self.entries.len())
                .expect("indexed transaction promotes before exhausting u32 entries");
            self.entries.push(IndexedEntry { start, value });
            index
        };
        self.live_len += 1;
        self.fill_range(start, end, index);
    }

    #[inline(always)]
    fn state_until<FEnd>(
        &self,
        byte_offset: usize,
        byte_end: usize,
        end_of: FEnd,
    ) -> (Option<&T>, usize)
    where
        FEnd: Fn(&T) -> usize,
    {
        if let Some(entry_index) = self.entry_index_at(byte_offset) {
            let entry = &self.entries[entry_index];
            debug_assert!(entry.is_live());
            return (Some(&entry.value), end_of(&entry.value).min(byte_end));
        }
        (None, self.empty_state_until(byte_offset, byte_end))
    }

    #[inline(never)]
    fn empty_state_until(&self, byte_offset: usize, byte_end: usize) -> usize {
        if self.pages.is_empty() {
            return byte_end;
        }
        let table_start = self.base_page.saturating_mul(PAGE_BYTES);
        let table_end = self
            .base_page
            .saturating_add(self.pages.len())
            .saturating_mul(PAGE_BYTES);
        if byte_offset < table_start {
            return table_start.min(byte_end);
        }
        if byte_offset >= table_end {
            return byte_end;
        }

        let scan_end = byte_end.min(table_end);
        let mut cursor = byte_offset;
        while cursor < scan_end {
            let page = cursor >> PAGE_SHIFT;
            let page_start = page.saturating_mul(PAGE_BYTES);
            let chunk_end = scan_end.min(page_start.saturating_add(PAGE_BYTES));
            let page_index = page - self.base_page;
            let local_start = cursor - page_start;
            let local_end = chunk_end - page_start;
            let occupied = self.pages[page_index]
                .as_ref()
                .and_then(|page| page.first_occupied(local_start, local_end))
                .or_else(|| {
                    self.wide_pages.get(&page).and_then(|cells| {
                        cells[local_start..local_end]
                            .iter()
                            .position(|cell| *cell != EMPTY_CELL)
                    })
                });
            if let Some(delta) = occupied {
                return cursor + delta;
            }
            cursor = chunk_end;
        }
        byte_end
    }

    fn try_for_each_exact_range_mut<FEnd, FNeedsVisit, F>(
        &mut self,
        start: usize,
        end: usize,
        end_of: FEnd,
        mut needs_visit: FNeedsVisit,
        mut visit: F,
    ) -> bool
    where
        FEnd: Fn(&T) -> usize,
        FNeedsVisit: FnMut(&T) -> bool,
        F: FnMut(&mut T),
    {
        let mut cursor = start;
        let mut any_visit = false;
        while cursor < end {
            let Some(index) = self.entry_index_at(cursor) else {
                return false;
            };
            let entry = &self.entries[index];
            let entry_end = end_of(&entry.value);
            if !entry.is_live() || entry.start != cursor || entry_end <= cursor || entry_end > end {
                return false;
            }
            any_visit |= needs_visit(&entry.value);
            cursor = entry_end;
        }
        if !any_visit {
            return true;
        }

        let mut cursor = start;
        while cursor < end {
            let index = self
                .entry_index_at(cursor)
                .expect("an exact indexed cover remains present during in-place visitation");
            let entry = &mut self.entries[index];
            let entry_end = end_of(&entry.value);
            if needs_visit(&entry.value) {
                visit(&mut entry.value);
            }
            cursor = entry_end;
        }
        true
    }

    fn ensure_range(&mut self, start: usize, end: usize) -> bool {
        debug_assert!(start < end);
        let start_page = start >> PAGE_SHIFT;
        let end_page = ((end - 1) >> PAGE_SHIFT).saturating_add(1);
        let density_limit_bytes = self
            .live_len
            .saturating_add(1)
            .saturating_mul(MAX_BYTES_PER_ENTRY)
            .max(MIN_INDEXED_BYTES)
            .min(MAX_INDEXED_BYTES);
        let page_limit = density_limit_bytes.saturating_add(PAGE_BYTES - 1) / PAGE_BYTES;

        if self.pages.is_empty() {
            let page_count = end_page - start_page;
            if page_count > page_limit {
                return false;
            }
            self.base_page = start_page;
            self.pages
                .extend(std::iter::repeat_with(|| None).take(page_count));
            return true;
        }

        let current_end_page = self.base_page.saturating_add(self.pages.len());
        if start_page >= self.base_page && end_page <= current_end_page {
            return true;
        }
        let new_base_page = self.base_page.min(start_page);
        let new_end_page = current_end_page.max(end_page);
        let Some(new_page_count) = new_end_page.checked_sub(new_base_page) else {
            return false;
        };
        if new_page_count > page_limit {
            return false;
        }

        if new_base_page == self.base_page {
            self.pages.resize_with(new_page_count, || None);
            return true;
        }

        let destination = self.base_page - new_base_page;
        let mut pages = Vec::with_capacity(new_page_count);
        pages.extend(std::iter::repeat_with(|| None).take(new_page_count));
        for (index, page) in std::mem::take(&mut self.pages).into_iter().enumerate() {
            pages[destination + index] = page;
        }
        self.base_page = new_base_page;
        self.pages = pages;
        true
    }

    fn range_is_indexable(&self, start: usize, end: usize) -> bool {
        if self.pages.is_empty() {
            return false;
        }
        let start_page = start >> PAGE_SHIFT;
        let end_page = ((end - 1) >> PAGE_SHIFT).saturating_add(1);
        start_page >= self.base_page && end_page <= self.base_page.saturating_add(self.pages.len())
    }

    #[inline(always)]
    fn fill_range(&mut self, start: usize, end: usize, value: u32) {
        debug_assert!(start < end);
        let start_page = start >> PAGE_SHIFT;
        if start_page == (end - 1) >> PAGE_SHIFT {
            debug_assert!(start_page >= self.base_page);
            let page_index = start_page - self.base_page;
            let local_start = start & PAGE_MASK;
            let local_end = local_start + (end - start);
            self.fill_page_range(start_page, page_index, local_start, local_end, value);
            return;
        }
        self.fill_range_across_pages(start, end, value);
    }

    fn fill_page_range(
        &mut self,
        page: usize,
        page_index: usize,
        local_start: usize,
        local_end: usize,
        value: u32,
    ) {
        if let Some(cells) = self.wide_pages.get_mut(&page) {
            cells[local_start..local_end].fill(value);
            return;
        }
        let page_slot = self
            .pages
            .get_mut(page_index)
            .expect("prepared indexed span lies inside the page table");
        if value == EMPTY_CELL {
            if let Some(page) = page_slot.as_mut() {
                debug_assert!(page.fill(local_start, local_end, value).is_none());
            }
            return;
        }
        let compact = page_slot.get_or_insert_with(|| IndexedPage::empty(value));
        if let Some(wide) = compact.fill(local_start, local_end, value) {
            *page_slot = None;
            self.wide_pages.insert(page, wide);
        }
    }

    #[inline(never)]
    fn fill_range_across_pages(&mut self, start: usize, end: usize, value: u32) {
        let mut cursor = start;
        while cursor < end {
            let page = cursor >> PAGE_SHIFT;
            let page_start = page.saturating_mul(PAGE_BYTES);
            let chunk_end = end.min(page_start.saturating_add(PAGE_BYTES));
            let page_index = page
                .checked_sub(self.base_page)
                .expect("prepared indexed span starts inside the page table");
            let local_start = cursor - page_start;
            let local_end = chunk_end - page_start;
            self.fill_page_range(page, page_index, local_start, local_end, value);
            cursor = chunk_end;
        }
    }

    fn range_is_empty(&self, start: usize, end: usize) -> bool {
        debug_assert!(start < end);
        let mut cursor = start;
        while cursor < end {
            let page = cursor >> PAGE_SHIFT;
            let page_start = page.saturating_mul(PAGE_BYTES);
            let chunk_end = end.min(page_start.saturating_add(PAGE_BYTES));
            let Some(page_index) = page.checked_sub(self.base_page) else {
                cursor = chunk_end;
                continue;
            };
            let local_start = cursor - page_start;
            let local_end = chunk_end - page_start;
            let compact_occupied = self
                .pages
                .get(page_index)
                .and_then(Option::as_ref)
                .is_some_and(|page| page.any_occupied(local_start, local_end));
            let wide_occupied = self.wide_pages.get(&page).is_some_and(|cells| {
                cells[local_start..local_end]
                    .iter()
                    .any(|cell| *cell != EMPTY_CELL)
            });
            if compact_occupied || wide_occupied {
                return false;
            }
            cursor = chunk_end;
        }
        true
    }

    fn into_entries(self) -> impl Iterator<Item = (usize, T)> {
        self.entries
            .into_iter()
            .filter(IndexedEntry::is_live)
            .map(|entry| (entry.start, entry.value))
    }

    fn iter(&self) -> impl Iterator<Item = (&usize, &T)> {
        self.entries
            .iter()
            .filter(|entry| entry.is_live())
            .map(|entry| (&entry.start, &entry.value))
    }

    fn for_each_value_mut(&mut self, mut visit: impl FnMut(&mut T)) {
        for entry in &mut self.entries {
            if entry.is_live() {
                visit(&mut entry.value);
            }
        }
    }
}

/// Sparse transactional intervals with a direct byte-occupancy index for the
/// common exact-or-disjoint case.
///
/// Mixed access widths and partial overlaps remain indexed while the address
/// range is dense enough. A pathologically sparse range promotes losslessly
/// to the general interval tree before validation.
#[derive(Clone)]
pub(crate) enum TransactionalIntervalMap<T> {
    Indexed(IndexedEntries<T>),
    General(BTreeMap<usize, T>),
}

impl<T> Default for TransactionalIntervalMap<T> {
    fn default() -> Self {
        Self::Indexed(IndexedEntries::default())
    }
}

impl<T: fmt::Debug> fmt::Debug for TransactionalIntervalMap<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_map()
            .entries(self.iter().map(|(key, value)| (key, value)))
            .finish()
    }
}

impl<T: PartialEq> PartialEq for TransactionalIntervalMap<T> {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len()
            && self
                .iter()
                .all(|(key, value)| other.get(*key).is_some_and(|other| other == value))
    }
}

impl<T: Eq> Eq for TransactionalIntervalMap<T> {}

impl<T> TransactionalIntervalMap<T> {
    #[inline(always)]
    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Indexed(entries) => entries.len(),
            Self::General(entries) => entries.len(),
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline(always)]
    pub(crate) fn get(&self, start: usize) -> Option<&T> {
        match self {
            Self::Indexed(entries) => entries.get(start),
            Self::General(entries) => entries.get(&start),
        }
    }

    #[inline(always)]
    pub(crate) fn get_mut(&mut self, start: usize) -> Option<&mut T> {
        match self {
            Self::Indexed(entries) => entries.get_mut(start),
            Self::General(entries) => entries.get_mut(&start),
        }
    }

    pub(crate) fn prepare_span(&mut self, start: usize, end: usize) {
        debug_assert!(start < end);
        let compatible = match self {
            Self::Indexed(entries) => entries.prepare_span(start, end),
            Self::General(_) => true,
        };
        if !compatible {
            self.promote();
        }
    }

    pub(crate) fn insert_prepared(&mut self, start: usize, end: usize, value: T) -> Option<T> {
        self.prepare_span(start, end);
        let fallback = match self {
            Self::Indexed(entries) => match entries.insert(start, end, value) {
                Ok(previous) => return previous,
                Err(value) => value,
            },
            Self::General(entries) => return entries.insert(start, value),
        };
        self.promote();
        match self {
            Self::General(entries) => entries.insert(start, fallback),
            Self::Indexed(_) => unreachable!("promotion produces a general interval map"),
        }
    }

    pub(crate) fn insert_after_prepare(&mut self, start: usize, end: usize, value: T) -> Option<T> {
        match self {
            Self::Indexed(entries) => entries.insert_after_prepare(start, end, value),
            Self::General(entries) => entries.insert(start, value),
        }
    }

    #[inline(always)]
    pub(crate) fn state_until<F>(
        &self,
        byte_offset: usize,
        byte_end: usize,
        end_of: F,
    ) -> (Option<&T>, usize)
    where
        F: Fn(&T) -> usize,
    {
        match self {
            Self::Indexed(entries) => entries.state_until(byte_offset, byte_end, end_of),
            Self::General(entries) => {
                if let Some(value) = entries
                    .range(..=byte_offset)
                    .next_back()
                    .map(|(_, value)| value)
                    .filter(|value| end_of(value) > byte_offset)
                {
                    return (Some(value), end_of(value).min(byte_end));
                }
                let next = entries
                    .range((
                        std::ops::Bound::Excluded(byte_offset),
                        std::ops::Bound::Unbounded,
                    ))
                    .next()
                    .map_or(byte_end, |(start, _)| (*start).min(byte_end));
                (None, next)
            }
        }
    }

    /// Mutate an interval range in place when existing entries cover it
    /// exactly, with no gaps and no boundary splits.
    ///
    /// The read-only preflight keeps a failed attempt transactional: callers
    /// may fall back to their general split/replace path without observing any
    /// partial mutation.
    pub(crate) fn try_for_each_exact_range_mut<FEnd, FNeedsVisit, F>(
        &mut self,
        start: usize,
        end: usize,
        end_of: FEnd,
        mut needs_visit: FNeedsVisit,
        mut visit: F,
    ) -> bool
    where
        FEnd: Fn(&T) -> usize,
        FNeedsVisit: FnMut(&T) -> bool,
        F: FnMut(&mut T),
    {
        match self {
            Self::Indexed(entries) => {
                entries.try_for_each_exact_range_mut(start, end, end_of, needs_visit, visit)
            }
            Self::General(entries) => {
                let mut cursor = start;
                let mut any_visit = false;
                while cursor < end {
                    let Some(value) = entries.get(&cursor) else {
                        return false;
                    };
                    let entry_end = end_of(value);
                    if entry_end <= cursor || entry_end > end {
                        return false;
                    }
                    any_visit |= needs_visit(value);
                    cursor = entry_end;
                }
                if !any_visit {
                    return true;
                }

                let mut cursor = start;
                while cursor < end {
                    let value = entries
                        .get_mut(&cursor)
                        .expect("an exact tree cover remains present during in-place visitation");
                    let entry_end = end_of(value);
                    if needs_visit(value) {
                        visit(value);
                    }
                    cursor = entry_end;
                }
                true
            }
        }
    }

    /// Replace a subrange of one indexed entry without allocating a container.
    /// A miss returns the value without changing the map, for general fallback.
    pub(crate) fn try_replace_one<FStart, FEnd, FSplit>(
        &mut self,
        start: usize,
        end: usize,
        replacement: T,
        start_of: FStart,
        end_of: FEnd,
        split: FSplit,
    ) -> Result<(), T>
    where
        FStart: Fn(&T) -> usize,
        FEnd: Fn(&T) -> usize,
        FSplit: Fn(&T, usize, usize) -> T,
    {
        match self {
            Self::Indexed(entries) => {
                entries.replace_one(start, end, replacement, start_of, end_of, split)
            }
            Self::General(_) => Err(replacement),
        }
    }

    pub(crate) fn try_replace_range<FStart, FEnd, FSplit>(
        &mut self,
        start: usize,
        end: usize,
        replacement: Vec<T>,
        start_of: FStart,
        end_of: FEnd,
        split: FSplit,
    ) -> Result<(), Vec<T>>
    where
        FStart: Fn(&T) -> usize,
        FEnd: Fn(&T) -> usize,
        FSplit: Fn(&T, usize, usize) -> T,
    {
        match self {
            Self::Indexed(entries) => {
                entries.replace_range(start, end, replacement, start_of, end_of, split)
            }
            Self::General(_) => Err(replacement),
        }
    }

    pub(crate) fn general_mut(&mut self) -> &mut BTreeMap<usize, T> {
        self.promote();
        match self {
            Self::General(entries) => entries,
            Self::Indexed(_) => unreachable!("promotion produces a general interval map"),
        }
    }

    pub(crate) fn into_sorted_values(self) -> Vec<T> {
        match self {
            Self::Indexed(entries) => {
                let mut entries = entries.into_entries().collect::<Vec<_>>();
                entries.sort_unstable_by_key(|(start, _)| *start);
                entries.into_iter().map(|(_, value)| value).collect()
            }
            Self::General(entries) => entries.into_values().collect(),
        }
    }

    pub(crate) fn into_values(self) -> Vec<T> {
        match self {
            Self::Indexed(entries) => entries.into_entries().map(|(_, value)| value).collect(),
            Self::General(entries) => entries.into_values().collect(),
        }
    }

    fn promote(&mut self) {
        let previous = std::mem::replace(self, Self::General(BTreeMap::new()));
        *self = match previous {
            Self::Indexed(entries) => Self::General(entries.into_entries().collect()),
            general @ Self::General(_) => general,
        };
    }

    /// Visit every live entry's value mutably, in no particular order.
    pub(crate) fn for_each_value_mut(&mut self, mut visit: impl FnMut(&mut T)) {
        match self {
            Self::Indexed(entries) => entries.for_each_value_mut(visit),
            Self::General(map) => map.values_mut().for_each(|value| visit(value)),
        }
    }

    pub(crate) fn iter(&self) -> Box<dyn Iterator<Item = (&usize, &T)> + '_> {
        match self {
            Self::Indexed(entries) => Box::new(entries.iter()),
            Self::General(entries) => Box::new(entries.iter()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{IndexedEntry, TransactionalIntervalMap, PAGE_BYTES};

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct Segment {
        end: usize,
        value: usize,
    }

    #[test]
    fn indexed_entry_omits_the_value_owned_end_without_bool_padding() {
        assert_eq!(
            std::mem::size_of::<IndexedEntry<[u8; 56]>>(),
            std::mem::size_of::<usize>() + 56,
        );
    }

    #[test]
    fn mixed_disjoint_geometry_stays_indexed() {
        let mut map = TransactionalIntervalMap::default();
        map.insert_prepared(12, 16, Segment { end: 16, value: 1 });
        map.insert_prepared(4, 8, Segment { end: 8, value: 2 });
        map.insert_prepared(16, 24, Segment { end: 24, value: 3 });

        assert!(matches!(map, TransactionalIntervalMap::Indexed(_)));
        assert_eq!(
            map.state_until(13, 16, |segment| segment.end),
            (Some(&Segment { end: 16, value: 1 }), 16)
        );
        assert_eq!(map.state_until(8, 12, |segment| segment.end), (None, 12));
    }

    #[test]
    fn partial_overlap_remains_queryable_before_replacement() {
        let mut map = TransactionalIntervalMap::default();
        map.insert_prepared(0, 4, Segment { end: 4, value: 1 });
        map.prepare_span(2, 6);

        assert!(matches!(map, TransactionalIntervalMap::Indexed(_)));
        assert_eq!(map.get(0), Some(&Segment { end: 4, value: 1 }));
        assert_eq!(
            map.state_until(2, 6, |segment| segment.end),
            (Some(&Segment { end: 4, value: 1 }), 4)
        );
    }

    #[test]
    fn exact_mixed_width_update_stays_indexed() {
        let mut map = TransactionalIntervalMap::default();
        map.insert_prepared(0, 4, Segment { end: 4, value: 1 });
        map.insert_prepared(8, 16, Segment { end: 16, value: 2 });
        map.prepare_span(8, 16);

        assert!(matches!(map, TransactionalIntervalMap::Indexed(_)));
        assert_eq!(map.get(8), Some(&Segment { end: 16, value: 2 }));
    }

    #[test]
    fn exact_range_visits_existing_geometry_in_place() {
        let mut map = TransactionalIntervalMap::default();
        map.insert_prepared(0, 4, Segment { end: 4, value: 1 });
        map.insert_prepared(4, 8, Segment { end: 8, value: 2 });
        map.insert_prepared(8, 16, Segment { end: 16, value: 3 });

        assert!(map.try_for_each_exact_range_mut(
            0,
            16,
            |segment| segment.end,
            |_| true,
            |segment| segment.value += 10,
        ));
        assert_eq!(map.get(0), Some(&Segment { end: 4, value: 11 }));
        assert_eq!(map.get(4), Some(&Segment { end: 8, value: 12 }));
        assert_eq!(map.get(8), Some(&Segment { end: 16, value: 13 }));
    }

    #[test]
    fn alternating_geometry_reuses_indexed_entry_slots() {
        let mut map = TransactionalIntervalMap::default();
        map.insert_prepared(0, 32, Segment { end: 32, value: 0 });

        for value in 1..100 {
            let narrow = (0..32)
                .map(|start| Segment {
                    end: start + 1,
                    value,
                })
                .collect();
            assert!(map
                .try_replace_range(
                    0,
                    32,
                    narrow,
                    |segment| segment.end - 1,
                    |segment| segment.end,
                    |segment, _, split_end| Segment {
                        end: split_end,
                        value: segment.value,
                    },
                )
                .is_ok());
            assert!(map
                .try_replace_range(
                    0,
                    32,
                    vec![Segment { end: 32, value }],
                    |_| 0,
                    |segment| segment.end,
                    |segment, _, split_end| Segment {
                        end: split_end,
                        value: segment.value,
                    },
                )
                .is_ok());
        }

        let TransactionalIntervalMap::Indexed(entries) = map else {
            panic!("dense alternating geometry remains indexed");
        };
        assert_eq!(entries.live_len, 1);
        assert_eq!(entries.entries.len(), 32);
        assert_eq!(entries.free_entries.len(), 31);
    }

    #[test]
    fn exhausting_global_entry_indices_keeps_page_relative_indices_narrow() {
        let mut map = TransactionalIntervalMap::default();
        for start in 0..usize::from(u16::MAX) {
            map.insert_prepared(
                start,
                start + 1,
                Segment {
                    end: start + 1,
                    value: start,
                },
            );
        }
        map.insert_prepared(
            usize::from(u16::MAX),
            usize::from(u16::MAX) + 1,
            Segment {
                end: usize::from(u16::MAX) + 1,
                value: usize::from(u16::MAX),
            },
        );

        let TransactionalIntervalMap::Indexed(entries) = &map else {
            panic!("entry-index exhaustion remains indexed");
        };
        assert!(entries.wide_pages.is_empty());
        for start in [0, 1, 32_767, usize::from(u16::MAX)] {
            assert_eq!(
                map.get(start),
                Some(&Segment {
                    end: start + 1,
                    value: start,
                })
            );
        }
    }

    #[test]
    fn compact_page_overflow_uses_lossless_wide_side_table() {
        let mut map = TransactionalIntervalMap::default();
        map.insert_prepared(0, 1, Segment { end: 1, value: 0 });
        for start in PAGE_BYTES..PAGE_BYTES + usize::from(u16::MAX) {
            map.insert_prepared(
                start,
                start + 1,
                Segment {
                    end: start + 1,
                    value: start,
                },
            );
        }
        map.insert_prepared(1, 2, Segment { end: 2, value: 1 });

        let TransactionalIntervalMap::Indexed(entries) = &map else {
            panic!("dense page overflow remains indexed");
        };
        assert!(entries.wide_pages.contains_key(&0));
        assert_eq!(map.get(0), Some(&Segment { end: 1, value: 0 }));
        assert_eq!(map.get(1), Some(&Segment { end: 2, value: 1 }));
    }

    #[test]
    fn nested_single_range_replacements_preserve_untouched_subranges() {
        #[derive(Clone, Debug, PartialEq, Eq)]
        struct BoundedSegment {
            start: usize,
            end: usize,
            value: usize,
        }

        let mut map = TransactionalIntervalMap::default();
        map.insert_prepared(
            0,
            64,
            BoundedSegment {
                start: 0,
                end: 64,
                value: 0,
            },
        );
        for (start, end, value) in [(24, 32, 1), (8, 12, 2), (48, 60, 3)] {
            assert!(map
                .try_replace_range(
                    start,
                    end,
                    vec![BoundedSegment { start, end, value }],
                    |segment| segment.start,
                    |segment| segment.end,
                    |segment, split_start, split_end| BoundedSegment {
                        start: split_start,
                        end: split_end,
                        value: segment.value,
                    },
                )
                .is_ok());
        }

        let expected = [
            (0, 8, 0),
            (8, 12, 2),
            (12, 24, 0),
            (24, 32, 1),
            (32, 48, 0),
            (48, 60, 3),
            (60, 64, 0),
        ];
        for (start, end, value) in expected {
            assert_eq!(
                map.state_until(start, end, |segment| segment.end),
                (Some(&BoundedSegment { start, end, value }), end)
            );
        }
        assert!(matches!(map, TransactionalIntervalMap::Indexed(_)));
    }

    #[test]
    fn single_replacements_match_dense_bytes_and_failed_attempts_do_not_mutate() {
        #[derive(Clone, Debug, PartialEq, Eq)]
        struct BoundedSegment {
            start: usize,
            end: usize,
            value: usize,
        }
        let start_of = |segment: &BoundedSegment| segment.start;
        let end_of = |segment: &BoundedSegment| segment.end;
        let split = |segment: &BoundedSegment, start, end| BoundedSegment {
            start,
            end,
            value: segment.value,
        };
        let values = |map: &TransactionalIntervalMap<BoundedSegment>| {
            (0..128)
                .map(|byte| map.state_until(byte, byte + 1, end_of).0.map(|s| s.value))
                .collect::<Vec<_>>()
        };
        let mut map = TransactionalIntervalMap::default();
        let mut oracle = vec![None; 128];
        let mut random = 0x71_70_66_2_u64;
        let mut hits = 0;
        let mut misses = 0;
        for step in 1..=1024 {
            random = random.wrapping_mul(6364136223846793005).wrapping_add(1);
            let (start, end) = match step % 8 {
                0 => (0, 128),
                1 => (0, 1),
                2 => (127, 128),
                3 => (32, 96),
                _ => {
                    let start = (random >> 32) as usize % 128;
                    (start, start + 1)
                }
            };
            let before = map.clone();
            let replacement = BoundedSegment {
                start,
                end,
                value: step,
            };
            match map.try_replace_one(start, end, replacement.clone(), start_of, end_of, split) {
                Ok(()) => hits += 1,
                Err(returned) => {
                    misses += 1;
                    assert_eq!(returned, replacement);
                    assert_eq!(
                        map.iter().collect::<Vec<_>>(),
                        before.iter().collect::<Vec<_>>()
                    );
                    map.try_replace_range(start, end, vec![returned], start_of, end_of, split)
                        .unwrap();
                }
            }
            assert_eq!(values(&before), oracle, "clone changed at step {step}");
            oracle[start..end].fill(Some(step));
            assert_eq!(values(&map), oracle, "byte states differ at step {step}");
        }
        assert!(hits > 0 && misses > 0);

        map.general_mut();
        let before = map.clone();
        let replacement = BoundedSegment {
            start: 3,
            end: 4,
            value: 2048,
        };
        assert_eq!(
            map.try_replace_one(3, 4, replacement.clone(), start_of, end_of, split),
            Err(replacement),
        );
        assert_eq!(
            map.iter().collect::<Vec<_>>(),
            before.iter().collect::<Vec<_>>()
        );
        assert_eq!(values(&map), oracle);
    }

    #[test]
    fn failed_exact_range_preflight_does_not_mutate() {
        let mut map = TransactionalIntervalMap::default();
        map.insert_prepared(0, 4, Segment { end: 4, value: 1 });
        map.insert_prepared(8, 16, Segment { end: 16, value: 2 });

        assert!(!map.try_for_each_exact_range_mut(
            0,
            16,
            |segment| segment.end,
            |_| true,
            |segment| segment.value += 10,
        ));
        assert_eq!(map.get(0), Some(&Segment { end: 4, value: 1 }));
        assert_eq!(map.get(8), Some(&Segment { end: 16, value: 2 }));
    }
}
