//! # Lazy Clone Collections
//!
//! The `lazy-clone-collections` crate provides standard collections with
//! support for efficient cloning.  When a lazy-clone collection is cloned, the
//! original and the cloned collections share their internal representation.  As
//! the original and/or the clone are updated, their internal representations
//! increasingly diverge.  A lazy-clone collection clones its internal state
//! on-demand, or lazily, to implement updates.
//!
//! Externally, the lazy-clone collections provide standard destructive update
//! semantics. Where applicable, each lazy-clone collections attempts to match
//! the interface of the corresponding collection type from
//! [`std::collections`]. Internally, the lazy-clone collections use data
//! structures that behavior like those from a functional language.  Upon an
//! update, a collection builds and switches to a new representation, but the
//! pre-update representation may continue to exist and be used by other
//! collections.  The new and original structures may partially overlap.
//!
//! There are many names for this type of behavior.  The internal structures
//! might be called immutable, persistent, applicative, or be said to have value
//! semantics.  (As an optimization, the structures destructively update nodes
//! with a reference count of one, but Rust's ownership semantics ensures this
//! is transparent to clients.)
//!
//! The lazy-clone collections are designed to support uses cases where you need
//! to create and keep many clones of your collections.  The standard
//! collections are more efficient most of the time.  However, when aggressive
//! cloning is necessary, the standard collections will quickly explode in
//! memory usage leading to severe declines in performance.  An example where
//! lazy-clone collections shine is in representing symbolic state in a symbolic
//! execution engine that performing a breadth-first exploration.
//!
//! Lazy-clone collections require that their elements implement [`Clone`].
//!
//! The crate provides the following collections:
//!
//! * [`AvlMap`] provides a map that matches the (stable) interface of
//!   [`std::collections::BTreeMap`].  It is implemented using the venerable
//!   [AVL tree](https://en.wikipedia.org/wiki/AVL_tree) data structure. It is
//!   best in cases where cloning map elements is expensive, for example, if the
//!   keys or mapped values are strings or standard collections.
//!
//! * [`BTreeMap`] also provides a map that matches the interface of
//!   [`std::collections::BTreeMap`] and uses
//!   [B-trees](https://en.wikipedia.org/wiki/B-tree) in its implementation.
//!   Given an [`AvlMap`] and a [`BTreeMap`] holding the same elements, the
//!   [`BTreeMap`] holds more elements in each node and is shallower.  Lookup
//!   operations are likely faster in the [`BTreeMap`].  On an update, the
//!   [`BTreeMap`] will clone fewer nodes, but is likely to clone more elements.
//!   The relative performance of the two structures will depend on the mix of
//!   operations and the expense of cloning operations.
//!
//! * [`AvlSet`] keeps a set of values and matches the
//!   [`std::collections::BTreeSet`] interface.  It is a thin wrapper over an
//!   [`AvlMap<T, ()>`] and shares its properties.
//!
//! * TODO: `BTreeSet`
//!
//! * [`Stack`] provides a Last-In First-Out (LIFO) stack.  It can also be used
//!   as a singly-linked list. A lazy-clone stack may be useful for modeling and
//!   recording the evolution of a population of individuals where new
//!   individuals are derived from old. Each individual can be associated with a
//!   stack that records that individual's history.  Because of internal
//!   sharing, a set of stacks may form an "ancestral tree" with each stack
//!   corresponding to a path from the root to a leaf.

#[warn(missing_docs)]
mod stack;
pub use stack::Stack;

mod avl;
pub use avl::avl_set::AvlSet;
pub use avl::AvlMap;

pub mod btree;
pub type BTreeMap<K, V> = btree::BTreeMap<K, V, 5>;
pub type BTreeSet<T> = btree::btree_set::BTreeSet<T, 5>;

struct SortedMergeIter<I: Iterator> {
    lhs: std::iter::Peekable<I>,
    rhs: std::iter::Peekable<I>,
}

impl<I: Iterator> SortedMergeIter<I> {
    fn new(lhs: I, rhs: I) -> Self {
        Self {
            lhs: lhs.peekable(),
            rhs: rhs.peekable(),
        }
    }
}

impl<T: Ord, I: Iterator<Item = T>> Iterator for SortedMergeIter<I> {
    type Item = (Option<T>, Option<T>);

    fn next(&mut self) -> Option<Self::Item> {
        use std::cmp::Ordering::*;

        match (self.lhs.peek(), self.rhs.peek()) {
            (None, None) => None,
            (None, Some(_)) => Some((None, self.rhs.next())),
            (Some(_), None) => Some((self.lhs.next(), None)),

            (Some(lhs), Some(rhs)) => match lhs.cmp(rhs) {
                Less => Some((self.lhs.next(), None)),
                Equal => Some((self.lhs.next(), self.rhs.next())),
                Greater => Some((None, self.rhs.next())),
            },
        }
    }
}

impl<T: Ord, I: Iterator<Item = T>> std::iter::FusedIterator
    for SortedMergeIter<I>
{
}

enum KeepFlags {
    LeftSolo = 0b0100,
    Common = 0b0010,
    RightSolo = 0b0001,
}

struct SetOpIter<I: Iterator, const P: u32>(SortedMergeIter<I>);

impl<I: Iterator, const P: u32> SetOpIter<I, P> {
    fn new(lhs: I, rhs: I) -> Self {
        Self(SortedMergeIter::new(lhs, rhs))
    }
}

impl<T, I, const P: u32> Iterator for SetOpIter<I, P>
where
    T: Ord,
    I: Iterator<Item = T>,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        use KeepFlags::*;
        loop {
            let (a, b) = self.0.next()?;
            if a.is_none() && P & RightSolo as u32 != 0 {
                return b;
            } else if b.is_none() && P & LeftSolo as u32 != 0 {
                return a;
            } else if a.is_some() && b.is_some() && P & Common as u32 != 0 {
                return b;
            }
        }
    }
}

impl<T: Ord, I: Iterator<Item = T>, const P: u32> std::iter::FusedIterator
    for SetOpIter<I, P>
{
}

macro_rules! make_set_op_iter {
    ( $name:ident, $iter:ty, $policy:literal $(, $N:ident)*) => {
        pub struct $name<'a, T: 'a $(, const $N: usize)*> {
            iter: crate::SetOpIter<$iter, $policy>,
        }

        impl<'a, T $(, const $N: usize)*> $name<'a, T $(, $N)*> {
            fn new(lhs: $iter, rhs: $iter) -> Self {
                Self {
                    iter: crate::SetOpIter::new(lhs, rhs),
                }
            }
        }

        impl<'a, T: 'a + Ord $(, const $N: usize)*> Iterator
        for $name<'a, T $(, $N)*> {
            type Item = &'a T;

            fn next(&mut self) -> Option<Self::Item> {
                self.iter.next()
            }
        }

        impl<'a, T: 'a + Ord$(, const $N: usize)*> std::iter::FusedIterator
        for $name<'a, T $(, $N)*> {}
    };
}

use make_set_op_iter;

#[allow(dead_code)]
enum StitchErr<H, E> {
    TooFewElems(Option<H>), // H is partial result, if possible
    Other(E),               // E is an err from another module
}

pub trait Map {
    type Key;
    type Value;

    // A Half is "half" of a map.  It contains an internal map node and a
    // key-value pair where the key is greater than everything in the node.
    type Half;

    // MAX_HALF_LEN is usually the maximum number of elements in an internal map
    // node plus one for the extra key-value item.
    const MAX_HALF_LEN: usize;

    fn contains_key_<Q>(&mut self, key: &Q) -> bool
    where
        Self::Key: std::borrow::Borrow<Q>,
        Q: Ord + ?Sized;

    fn get_mut_<Q>(&mut self, key: &Q) -> Option<&mut Self::Value>
    where
        Self::Key: std::borrow::Borrow<Q> + Clone,
        Self::Value: Clone,
        Q: Ord + ?Sized;

    fn insert_(
        &mut self,
        key: Self::Key,
        val: Self::Value,
    ) -> Option<Self::Value>
    where
        Self::Key: Clone + Ord,
        Self::Value: Clone;

    fn new_() -> Self;

    // Make a Half map from the given elements, which should be non-empty but
    // may contain one more than fits in a leaf node.  elems is emptied.
    fn make_half(elems: &mut Vec<(Self::Key, Self::Value)>) -> Self::Half;

    // convert a Half into a standard Map
    fn make_whole(h: Self::Half, len: usize) -> Self
    where
        Self::Key: Clone + Ord,
        Self::Value: Clone;

    // combine a lower Half and an upper Half into "half" of a yet larger map
    fn stitch(lf: Self::Half, rt: Self::Half) -> Self::Half
    where
        Self::Key: Clone + Ord,
        Self::Value: Clone;
}

#[derive(Debug)]
pub struct Entry<'a, M: Map> {
    map: &'a mut M,
    key: M::Key,
}

impl<'a, M: Map> Entry<'a, M> {
    pub fn and_modify<F>(self, f: F) -> Self
    where
        F: FnOnce(&mut M::Value),
        M::Key: Clone + Ord,
        M::Value: Clone,
    {
        if let Some(v) = self.map.get_mut_(&self.key) {
            f(v)
        }

        self
    }

    pub fn key(&self) -> &M::Key {
        &self.key
    }

    fn or_aux<F>(self, f: F) -> &'a mut M::Value
    where
        F: FnOnce() -> M::Value,
        M::Key: Clone + Ord,
        M::Value: Clone,
    {
        if !self.map.contains_key_(&self.key) {
            self.map.insert_(self.key.clone(), f());
        }
        self.map.get_mut_(&self.key).unwrap()
    }

    pub fn or_default(self) -> &'a mut M::Value
    where
        M::Key: Clone + Ord,
        M::Value: Clone + Default,
    {
        self.or_aux(M::Value::default)
    }

    pub fn or_insert(self, default: M::Value) -> &'a mut M::Value
    where
        M::Key: Clone + Ord,
        M::Value: Clone,
    {
        self.or_aux(|| default)
    }

    pub fn or_insert_with<F: FnOnce() -> M::Value>(
        self,
        default: F,
    ) -> &'a mut M::Value
    where
        M::Key: Clone + Ord,
        M::Value: Clone,
    {
        self.or_aux(default)
    }

    pub fn or_insert_with_key<F: FnOnce(&M::Key) -> M::Value>(
        self,
        default: F,
    ) -> &'a mut M::Value
    where
        M::Key: Clone + Ord,
        M::Value: Clone,
    {
        if !self.map.contains_key_(&self.key) {
            let v = default(&self.key);
            self.map.insert_(self.key.clone(), v);
        }
        self.map.get_mut_(&self.key).unwrap()
    }
}

pub(crate) trait Set {
    type Value;

    fn insert_(&mut self, value: Self::Value) -> bool
    where
        Self::Value: Clone + Ord;
}

#[cfg(feature = "serde")]
mod serde {
    use super::{Map, Set, StitchErr};
    use serde::de::{Deserialize, MapAccess, SeqAccess, Visitor};
    use std::{fmt, marker::PhantomData};

    pub(crate) struct MapVisitor<M: Map> {
        pub desc: String,
        pub marker: PhantomData<fn() -> M>,
    }

    impl<'de, MAP: Map> Visitor<'de> for MapVisitor<MAP>
    where
        MAP::Key: Clone + Deserialize<'de> + Ord,
        MAP::Value: Clone + Deserialize<'de>,
    {
        type Value = MAP;

        // Format a message stating what data this Visitor expects to receive.
        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str(&self.desc)
        }

        fn visit_map<M>(self, mut access: M) -> Result<Self::Value, M::Error>
        where
            M: MapAccess<'de>,
        {
            fn build_map_by_halves<'de, MAP: Map, M>(
                access: &mut M,
                len: usize,
                scratch: &mut Vec<(MAP::Key, MAP::Value)>, // used for leaves
            ) -> Result<MAP::Half, StitchErr<MAP::Half, M::Error>>
            where
                M: MapAccess<'de>,
                MAP::Key: Clone + Deserialize<'de> + Ord,
                MAP::Value: Clone + Deserialize<'de>,
            {
                assert!(len > 0);

                if len <= MAP::MAX_HALF_LEN {
                    assert!(scratch.is_empty());

                    for _ in 0..len {
                        let kv = match access.next_entry() {
                            Ok(Some(kv)) => kv,

                            Ok(None) => {
                                let partial = if scratch.is_empty() {
                                    None
                                } else {
                                    Some(MAP::make_half(scratch))
                                };
                                return Err(StitchErr::TooFewElems(partial));
                            }

                            Err(e) => return Err(StitchErr::Other(e)),
                        };

                        scratch.push(kv);
                    }

                    Ok(MAP::make_half(scratch))
                } else {
                    let lf_len = (len + 1) / 2; // left gets big half for mid pt
                    let rt_len = len - lf_len;
                    let lhs =
                        build_map_by_halves::<MAP, M>(access, lf_len, scratch)?;
                    let rhs =
                        build_map_by_halves::<MAP, M>(access, rt_len, scratch);

                    match rhs {
                        Ok(rhs) => Ok(MAP::stitch(lhs, rhs)),

                        Err(StitchErr::TooFewElems(None)) => {
                            Err(StitchErr::TooFewElems(Some(lhs)))
                        }

                        Err(StitchErr::TooFewElems(Some(rhs))) => Err(
                            StitchErr::TooFewElems(Some(MAP::stitch(lhs, rhs))),
                        ),

                        e @ Err(StitchErr::Other(_)) => e,
                    }
                }
            }

            let mut map = MAP::new_();

            if let Some(len) = access.size_hint() {
                if len > 0 {
                    // we allocate and pass a buffer for temporarily holding
                    // leaf entries to avoid de/allocations & simplify errors
                    let mut scratch = Vec::new();
                    let res = build_map_by_halves::<MAP, M>(
                        &mut access,
                        len,
                        &mut scratch,
                    );

                    match res {
                        // if size_hint was small, there may be more elems
                        Ok(h) => map = MAP::make_whole(h, len),

                        Err(StitchErr::TooFewElems(Some(h))) => {
                            return Ok(MAP::make_whole(h, len))
                        }

                        Err(StitchErr::TooFewElems(None)) => {
                            return Ok(MAP::new_())
                        }

                        Err(StitchErr::Other(e)) => return Err(e),
                    }
                }
            }

            while let Some((key, value)) = access.next_entry()? {
                map.insert_(key, value);
            }

            Ok(map)
        }
    }

    pub(crate) struct SetVisitor<S: Set> {
        pub set: Box<S>,
        pub desc: String,
    }

    impl<'de, S: Set> Visitor<'de> for SetVisitor<S>
    where
        S::Value: Clone + Deserialize<'de> + Ord,
    {
        type Value = S;

        // Format a message stating what data this Visitor expects to receive.
        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str(&self.desc)
        }

        fn visit_seq<M>(
            mut self,
            mut access: M,
        ) -> Result<Self::Value, M::Error>
        where
            M: SeqAccess<'de>,
        {
            while let Some(elem) = access.next_element()? {
                self.set.insert_(elem);
            }

            Ok(*self.set)
        }
    }
}
