use proptest::prelude::*;
use std::ops::Bound;

#[allow(dead_code)]
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

pub(super) type SmallIntPairs = Vec<(u16, u16)>;

pub(super) fn small_int_pairs() -> impl Strategy<Value = SmallIntPairs> {
    prop::collection::vec((0u16..1024u16, 0u16..1024u16), 0..512)
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
