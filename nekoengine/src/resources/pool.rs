use std::marker::PhantomData;
use std::fmt;
use std::hash::{Hash, Hasher};

pub struct Handle<T> {
    index: u32,
    generation: u32,
    _m: PhantomData<fn() -> T>,
}

enum State<T> {
    Free,
    Reserved,
    Occupied(T),
}

pub struct Pool<T> {
    slots: Vec<Slot<T>>,
    free: Vec<u32>,
    count: usize, // occupied slots only
}

pub struct Slot<T> {
    generation: u32,
    state: State<T>,
}

impl<T> Handle<T> {
    fn new(index: u32, generation: u32) -> Self {
        Self {
            index,
            generation,
            _m: PhantomData,
        }
    }

    pub fn index(&self) -> u32 {
        self.index
    }
}
impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for Handle<T> {}
impl<T> PartialEq for Handle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index && self.generation == other.generation
    }
}
impl<T> Eq for Handle<T> {}
impl<T> Hash for Handle<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.index.hash(state);
        self.generation.hash(state);
    }
}
impl<T> fmt::Debug for Handle<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Handle<{}>({}v{})",
            std::any::type_name::<T>(),
            self.index,
            self.generation
        )
    }
}

impl<T> Default for Pool<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Pool<T> {
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            free: Vec::new(),
            count: 0,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            slots: Vec::with_capacity(capacity),
            free: Vec::new(),
            count: 0,
        }
    }

    /// Number of slots holding a value (reserved slots are not counted).
    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Takes a free slot (or pushes a new one) and marks it with `state`.
    fn acquire(&mut self, state: State<T>) -> Handle<T> {
        if let Some(index) = self.free.pop() {
            let slot = &mut self.slots[index as usize];
            debug_assert!(matches!(slot.state, State::Free));
            slot.state = state;
            Handle::new(index, slot.generation)
        } else {
            let index = u32::try_from(self.slots.len()).expect("pool exceeded u32::MAX slots");
            self.slots.push(Slot {
                generation: 0,
                state,
            });
            Handle::new(index, 0)
        }
    }

    pub fn insert(&mut self, value: T) -> Handle<T> {
        self.count += 1;
        self.acquire(State::Occupied(value))
    }

    /// Hands out a handle now, value comes later via `fill`.
    /// `get` returns `None` until then.
    pub fn reserve(&mut self) -> Handle<T> {
        self.acquire(State::Reserved)
    }

    /// Fills a reserved slot. Returns the value back as `Err` if the handle is
    /// stale (e.g. unloaded while loading) or the slot is not reserved, so the
    /// caller can clean up GPU resources instead of leaking them.
    pub fn fill(&mut self, handle: Handle<T>, value: T) -> Result<(), T> {
        match self.slots.get_mut(handle.index as usize) {
            Some(slot)
                if slot.generation == handle.generation
                    && matches!(slot.state, State::Reserved) =>
            {
                slot.state = State::Occupied(value);
                self.count += 1;
                Ok(())
            }
            _ => Err(value),
        }
    }

    pub fn get(&self, handle: Handle<T>) -> Option<&T> {
        match self.slots.get(handle.index as usize)? {
            Slot {
                generation,
                state: State::Occupied(v),
            } if *generation == handle.generation => Some(v),
            _ => None,
        }
    }

    pub fn get_mut(&mut self, handle: Handle<T>) -> Option<&mut T> {
        match self.slots.get_mut(handle.index as usize)? {
            Slot {
                generation,
                state: State::Occupied(v),
            } if *generation == handle.generation => Some(v),
            _ => None,
        }
    }

    /// True if the handle is valid and has a value.
    pub fn contains(&self, handle: Handle<T>) -> bool {
        self.get(handle).is_some()
    }

    /// True if the handle is valid but still waiting for `fill`.
    pub fn is_pending(&self, handle: Handle<T>) -> bool {
        matches!(
            self.slots.get(handle.index as usize),
            Some(Slot { generation, state: State::Reserved }) if *generation == handle.generation
        )
    }

    /// Removes the value and invalidates every copy of the handle.
    /// Also cancels a reservation (returns `None` in that case).
    /// If `T` owns GPU resources, push the returned value onto your deferred
    /// deletion queue instead of dropping it right away.
    pub fn remove(&mut self, handle: Handle<T>) -> Option<T> {
        let slot = self.slots.get_mut(handle.index as usize)?;
        if slot.generation != handle.generation {
            return None;
        }
        match std::mem::replace(&mut slot.state, State::Free) {
            State::Free => None,
            State::Reserved => {
                slot.generation = slot.generation.wrapping_add(1);
                self.free.push(handle.index);
                None
            }
            State::Occupied(v) => {
                slot.generation = slot.generation.wrapping_add(1);
                self.free.push(handle.index);
                self.count -= 1;
                Some(v)
            }
        }
    }

    /// Replaces the value behind a valid handle, keeping the handle valid
    /// (hot reload). Returns the old value.
    pub fn replace(&mut self, handle: Handle<T>, value: T) -> Option<T> {
        self.get_mut(handle)
            .map(|slot| std::mem::replace(slot, value))
    }

    /// Mutable access to two different entries at once.
    /// Returns `None` if either handle is invalid or they point to the same slot.
    pub fn get2_mut(&mut self, a: Handle<T>, b: Handle<T>) -> Option<(&mut T, &mut T)> {
        if a.index == b.index || !self.contains(a) || !self.contains(b) {
            return None;
        }
        let (lo, hi, swapped) = if a.index < b.index {
            (a.index, b.index, false)
        } else {
            (b.index, a.index, true)
        };
        let (left, right) = self.slots.split_at_mut(hi as usize);
        let (x, y) = (&mut left[lo as usize], &mut right[0]);
        match (&mut x.state, &mut y.state) {
            (State::Occupied(x), State::Occupied(y)) => Some(if swapped { (y, x) } else { (x, y) }),
            _ => None,
        }
    }

    pub fn get_at_index(&self, index: usize) -> Option<Handle<T>> {
        self.slots.get(index).map(|slot| Handle::new(index as u32, slot.generation))
    }

    pub fn iter(&self) -> impl Iterator<Item = (Handle<T>, &T)> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(i, slot)| match &slot.state {
                State::Occupied(v) => Some((Handle::new(i as u32, slot.generation), v)),
                _ => None,
            })
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (Handle<T>, &mut T)> {
        self.slots
            .iter_mut()
            .enumerate()
            .filter_map(|(i, slot)| match &mut slot.state {
                State::Occupied(v) => Some((Handle::new(i as u32, slot.generation), v)),
                _ => None,
            })
    }

    /// Removes everything and invalidates all handles. Returns the values so
    /// you can queue their GPU resources for deferred destruction.
    pub fn drain(&mut self) -> Vec<T> {
        let mut out = Vec::with_capacity(self.count);
        self.free.clear();
        for (i, slot) in self.slots.iter_mut().enumerate() {
            if let State::Occupied(v) = std::mem::replace(&mut slot.state, State::Free) {
                out.push(v);
            }
            slot.generation = slot.generation.wrapping_add(1);
            self.free.push(i as u32);
        }
        self.count = 0;
        out
    }
}

impl<T> std::ops::Index<Handle<T>> for Pool<T> {
    type Output = T;
    /// Panics on a stale handle. Prefer `get` for anything that can be unloaded.
    fn index(&self, handle: Handle<T>) -> &T {
        self.get(handle).expect("invalid or stale pool handle")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
 
    #[test]
    fn insert_get_remove() {
        let mut p = Pool::new();
        let a = p.insert("a");
        let b = p.insert("b");
        assert_eq!(p.len(), 2);
        assert_eq!(p.get(a), Some(&"a"));
        assert_eq!(p.remove(a), Some("a"));
        assert_eq!(p.get(a), None);
        assert_eq!(p.remove(a), None);
        assert_eq!(p.get(b), Some(&"b"));
        assert_eq!(p.len(), 1);
    }
 
    #[test]
    fn stale_handle_after_slot_reuse() {
        let mut p = Pool::new();
        let a = p.insert(1);
        p.remove(a);
        let b = p.insert(2); // reuses the slot
        assert_eq!(a.index(), b.index());
        assert_ne!(a, b);
        assert_eq!(p.get(a), None);
        assert_eq!(p.get(b), Some(&2));
    }
 
    #[test]
    fn reserve_and_fill() {
        let mut p: Pool<i32> = Pool::new();
        let h = p.reserve();
        assert!(p.is_pending(h));
        assert_eq!(p.get(h), None);
        assert_eq!(p.len(), 0);
        assert!(p.fill(h, 7).is_ok());
        assert_eq!(p.get(h), Some(&7));
        assert!(!p.is_pending(h));
        assert_eq!(p.fill(h, 8), Err(8)); // not reserved anymore
    }
 
    #[test]
    fn cancel_reservation_rejects_late_fill() {
        let mut p: Pool<i32> = Pool::new();
        let h = p.reserve();
        assert_eq!(p.remove(h), None); // unloaded while loading
        assert_eq!(p.fill(h, 1), Err(1)); // caller gets the value back
        let h2 = p.insert(5);
        assert_eq!(h.index(), h2.index());
        assert_eq!(p.get(h2), Some(&5));
    }
 
    #[test]
    fn iteration_skips_free_and_reserved() {
        let mut p = Pool::new();
        let a = p.insert(1);
        let _r = p.reserve();
        let c = p.insert(3);
        p.remove(a);
        let items: Vec<_> = p.iter().map(|(h, v)| (h, *v)).collect();
        assert_eq!(items, vec![(c, 3)]);
        for (_, v) in p.iter_mut() {
            *v += 1;
        }
        assert_eq!(p.get(c), Some(&4));
    }
 
    #[test]
    fn get2_mut_works_in_both_orders() {
        let mut p = Pool::new();
        let a = p.insert(1);
        let b = p.insert(2);
        {
            let (x, y) = p.get2_mut(a, b).unwrap();
            assert_eq!((*x, *y), (1, 2));
        }
        {
            let (x, y) = p.get2_mut(b, a).unwrap();
            assert_eq!((*x, *y), (2, 1));
            std::mem::swap(x, y);
        }
        assert_eq!((p[a], p[b]), (2, 1));
        assert!(p.get2_mut(a, a).is_none());
    }
 
    #[test]
    fn replace_keeps_handle_valid() {
        let mut p = Pool::new();
        let a = p.insert(1);
        assert_eq!(p.replace(a, 2), Some(1));
        assert_eq!(p.get(a), Some(&2));
    }
 
    #[test]
    fn drain_invalidates_everything() {
        let mut p = Pool::new();
        let a = p.insert(1);
        let b = p.insert(2);
        let mut v = p.drain();
        v.sort();
        assert_eq!(v, vec![1, 2]);
        assert!(p.get(a).is_none() && p.get(b).is_none());
        assert!(p.is_empty());
        let c = p.insert(3);
        assert_eq!(p.get(c), Some(&3));
        assert!(p.get(a).is_none());
    }
 
    #[test]
    fn handle_is_copy_for_non_copy_t() {
        struct NotCopy;
        let mut p = Pool::new();
        let h = p.insert(NotCopy);
        let h2 = h; // Copy
        assert_eq!(h, h2);
    }
}