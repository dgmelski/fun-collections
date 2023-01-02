#![allow(dead_code, unused_imports, unused_macros)]
use proptest::prelude::*;
use std::cell::RefCell;
use std::ops::{Bound, Range};
use std::rc::Rc;

macro_rules! assert_eq_all {
    ( $x:expr  $(, $y:expr )* ) => {
        let seed = $x;
        $(
            assert_eq!(seed, $y);
        )*
    };
}

pub(super) use assert_eq_all;

pub(super) fn assert_eq_iters<I: Iterator, J: Iterator<Item = I::Item>>(
    mut i: I,
    mut j: J,
) where
    I::Item: std::fmt::Debug + Eq, // same inferred for J::Item
{
    loop {
        match (i.next(), j.next()) {
            (None, None) => return,
            (a, b) => assert_eq!(a, b),
        }
    }
}

pub(super) fn assert_eq_iters_back<I, J>(mut i: I, mut j: J)
where
    I: DoubleEndedIterator,
    J: DoubleEndedIterator<Item = I::Item>,
    I::Item: std::fmt::Debug + Eq, // same inferred for J::Item
{
    loop {
        match (i.next_back(), j.next_back()) {
            (None, None) => return,
            (a, b) => assert_eq!(a, b),
        }
    }
}

pub(super) struct CloneCounter {
    cntr: Rc<RefCell<usize>>,
}

impl CloneCounter {
    pub fn new(cntr: Rc<RefCell<usize>>) -> Self {
        Self { cntr }
    }
}

impl Clone for CloneCounter {
    fn clone(&self) -> Self {
        *self.cntr.borrow_mut() += 1;
        Self {
            cntr: self.cntr.clone(),
        }
    }
}

impl Eq for CloneCounter {}

impl std::fmt::Debug for CloneCounter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("CloneCounter: {}", self.cntr.borrow()))
    }
}

impl PartialEq for CloneCounter {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

pub(super) type U16Pairs = Vec<(u16, u16)>;

pub(super) fn u16_pairs(
    elem_range: Range<u16>,
    len_range: Range<usize>,
) -> impl Strategy<Value = U16Pairs> {
    prop::collection::vec((elem_range.clone(), elem_range), len_range)
}

pub(super) fn tiny_int_pairs() -> impl Strategy<Value = U16Pairs> {
    u16_pairs(0..64, 0..48)
}

pub(super) fn small_int_pairs() -> impl Strategy<Value = U16Pairs> {
    u16_pairs(0..1024, 0..512)
}

pub(super) type U16Seq = Vec<u16>;

pub(super) fn u16_seq(
    ub: u16,
    max_len: usize,
) -> impl Strategy<Value = U16Seq> {
    prop::collection::vec(0..ub, 0..max_len)
}

pub(super) fn tiny_int_seq() -> impl Strategy<Value = U16Seq> {
    u16_seq(64, 48)
}

pub(super) fn small_int_seq() -> impl Strategy<Value = U16Seq> {
    u16_seq(1024, 512)
}

#[allow(dead_code)]
pub(super) fn string_u16_pairs() -> impl Strategy<Value = Vec<(String, u16)>> {
    prop::collection::vec(("[a-z]{0,2}", 0u16..1024u16), 0..512)
}

pub(super) fn range_bounds_1k(
) -> impl Strategy<Value = (Bound<u16>, Bound<u16>)> {
    use Bound::*;

    (1u16..1023)
        .prop_flat_map(|n| {
            (
                prop_oneof![
                    Just(Bound::Unbounded),
                    (0u16..=n).prop_map(Bound::Excluded),
                    (0u16..=n).prop_map(Bound::Included),
                ],
                prop_oneof![
                    Just(Bound::Unbounded),
                    (n..1024).prop_map(Bound::Excluded),
                    (n..1024).prop_map(Bound::Included),
                ],
            )
        })
        .prop_map(|(lb, ub)| match (lb, ub) {
            (Excluded(x), Excluded(y)) if x == y => {
                // convert the panic case to a non-panic case (friendlier than
                // filtering for proptest?)
                (Included(x), Excluded(y))
            }

            xy => xy,
        })
}
