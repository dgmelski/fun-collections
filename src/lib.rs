//! # "Functional" collections that provide memory-efficient cloning
//!
//! `fun-collections` is a set of "functional" collections.  The collections use
//! persistent data structures, which means that a clone `s.clone()` shares its
//! internal representation with `s`.  The representations of a collection and
//! its clones gradually diverge as they are updated.  `fun-collections`
//! provides a subset of the functionality found in the `im` crate, which is way
//! more mature.  You probably should use the im crate instead of this one.

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
