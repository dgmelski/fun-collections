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

mod btree;
pub type BTreeMap<K, V> = btree::BTreeMap<K, V, 8>;

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

#[allow(clippy::enum_variant_names)]
enum SetOpFlag {
    KeepLeftOnly = 0b0100,
    KeepCommon = 0b0010,
    KeepRightOnly = 0b0001,
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
        use SetOpFlag::*;
        loop {
            let (a, b) = self.0.next()?;
            if a.is_none() && P & KeepRightOnly as u32 != 0 {
                return b;
            } else if b.is_none() && P & KeepLeftOnly as u32 != 0 {
                return a;
            } else if a.is_some() && b.is_some() && P & KeepCommon as u32 != 0 {
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
    ( $name:ident, $iter:ty, $policy:literal ) => {
        pub struct $name<'a, T: 'a>(crate::SetOpIter<$iter, $policy>);

        impl<'a, T> $name<'a, T> {
            fn new(lhs: $iter, rhs: $iter) -> Self {
                Self(crate::SetOpIter::new(lhs, rhs))
            }
        }

        impl<'a, T: 'a + Ord> Iterator for $name<'a, T> {
            type Item = &'a T;

            fn next(&mut self) -> Option<Self::Item> {
                self.0.next()
            }
        }

        impl<'a, T: 'a + Ord> std::iter::FusedIterator for $name<'a, T> {}
    };
}

use make_set_op_iter;
