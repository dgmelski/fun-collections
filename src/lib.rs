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
pub use avl::AvlMap;
pub use avl::AvlSet;

mod btree;
pub type BTreeMap<K, V> = btree::BTreeMap<K, V, 8>;

struct SortedMergeIter<'a, T: 'a, I: Iterator<Item = &'a T>> {
    lhs: std::iter::Peekable<I>,
    rhs: std::iter::Peekable<I>,
}

impl<'a, T: 'a, I: Iterator<Item = &'a T>> SortedMergeIter<'a, T, I> {
    fn new(lhs: I, rhs: I) -> Self {
        Self {
            lhs: lhs.peekable(),
            rhs: rhs.peekable(),
        }
    }
}

impl<'a, T: 'a + Ord, I: Iterator<Item = &'a T>> Iterator
    for SortedMergeIter<'a, T, I>
{
    type Item = (Option<&'a T>, Option<&'a T>);

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

impl<'a, T, I> std::iter::FusedIterator for SortedMergeIter<'a, T, I>
where
    T: 'a + Ord,
    I: Iterator<Item = &'a T>,
{
}

enum SetOpFlag {
    KeepLeftOnly = 0b0100,
    KeepCommon = 0b0010,
    KeepRightOnly = 0b0001,
}

struct SetOpIter<'a, T: 'a, I: Iterator<Item = &'a T>, const P: u32>(
    SortedMergeIter<'a, T, I>,
);

impl<'a, T, I, const P: u32> SetOpIter<'a, T, I, P>
where
    T: 'a,
    I: Iterator<Item = &'a T>,
{
    fn new(lhs: I, rhs: I) -> Self {
        Self(SortedMergeIter::new(lhs, rhs))
    }
}

impl<'a, T, I, const P: u32> Iterator for SetOpIter<'a, T, I, P>
where
    T: 'a + Ord,
    I: Iterator<Item = &'a T>,
{
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        use SetOpFlag::*;
        loop {
            match self.0.next()? {
                (None, None) => {
                    panic!("merge should give None, never (None, None)")
                }

                (None, rt @ Some(_)) => {
                    if P & KeepRightOnly as u32 != 0 {
                        return rt;
                    }
                }

                (lf @ Some(_), None) => {
                    if P & KeepLeftOnly as u32 != 0 {
                        return lf;
                    }
                }

                (Some(_), rt @ Some(_)) => {
                    if P & KeepCommon as u32 != 0 {
                        return rt;
                    }
                }
            }
        }
    }
}

impl<'a, T, I, const P: u32> std::iter::FusedIterator for SetOpIter<'a, T, I, P>
where
    T: 'a + Ord,
    I: Iterator<Item = &'a T>,
{
}

macro_rules! make_set_op_iter {
    ( $name:ident, $iter:ty, $policy:literal ) => {
        pub struct $name<'a, T: 'a>(crate::SetOpIter<'a, T, $iter, $policy>);

        impl<'a, T: 'a> $name<'a, T> {
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
