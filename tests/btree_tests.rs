use proptest::prelude::*;
use std::cmp::Ord;
use std::collections::BTreeMap as StdMap;
use std::ops::Bound;

// We use low branching to encourage deeper trees and more corner cases, even
// though our default setting may use higher branching.
type BTreeMap<K, V> = lazy_clone_collections::btree::BTreeMap<K, V, 2>;

fn small_int_pairs() -> impl Strategy<Value = Vec<(u16, u16)>> {
    prop::collection::vec((0u16..1024u16, 0u16..1024u16), 0..512)
}

fn string_u16_pairs() -> impl Strategy<Value = Vec<(String, u16)>> {
    prop::collection::vec(("[a-z]{0,2}", 0u16..1024u16), 0..512)
}

fn make_matching_maps<K, V>(v: Vec<(K, V)>) -> (BTreeMap<K, V>, StdMap<K, V>)
where
    K: Clone + Ord,
    V: Clone,
{
    let m1 = BTreeMap::from_iter(v.clone().into_iter());
    let m2 = StdMap::from_iter(v.into_iter());
    (m1, m2)
}

#[allow(clippy::type_complexity)]
fn make_matching_maps_with_sharing<K, V>(
    mut u: Vec<(K, V)>,
) -> (BTreeMap<K, V>, StdMap<K, V>, BTreeMap<K, V>, StdMap<K, V>)
where
    K: Clone + Ord,
    V: Clone,
{
    let v = u.split_off(u.len() / 2);
    let (a, b) = make_matching_maps(u);
    let (mut x, mut y) = (a.clone(), b.clone());
    x.extend(v.clone());
    y.extend(v);

    (a, b, x, y)
}

fn check_into_iter(v: Vec<(String, u16)>) {
    let (m1, m2) = make_matching_maps(v);
    assert_eq!(m1.len(), m2.len());
    let mut cnt = m1.len();
    for (x, y) in m1.into_iter().zip(m2.into_iter()) {
        assert_eq!(x, y);
        cnt -= 1;
    }
    assert_eq!(cnt, 0);
}

#[test]
#[should_panic]
fn test_range_ex_ex_panic() {
    let m = BTreeMap::from([(0u8, 0u8), (1, 1), (2, 2)]);
    m.range((Bound::Excluded(1), Bound::Excluded(1)));
}

#[test]
#[should_panic]
fn test_inverted_range_panic() {
    let m = BTreeMap::from([(0u8, 0u8), (1, 1), (2, 2)]);
    m.range((Bound::Included(2), Bound::Excluded(1)));
}

fn range_bounds_1k() -> impl Strategy<Value = (Bound<u16>, Bound<u16>)> {
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

fn check_range(u: Vec<(u16, u16)>, r: (Bound<u16>, Bound<u16>)) {
    let (a, b, x, y) = make_matching_maps_with_sharing(u);

    assert!(a.range(r).cmp(b.range(r)).is_eq());
    assert!(x.range(r).cmp(y.range(r)).is_eq());
}

fn check_range_mut(u: Vec<(u16, u16)>, r: (Bound<u16>, Bound<u16>)) {
    let (a, b, mut x, mut y) = make_matching_maps_with_sharing(u);

    for (k, v) in x.range_mut(r) {
        *v += k + 1024;
    }

    for (k, v) in y.range_mut(r) {
        *v += k + 1024;
    }

    assert!(a.iter().cmp(b.iter()).is_eq());
    assert!(x.iter().cmp(y.iter()).is_eq());
}

proptest! {
    #[test]
    fn test_into_iter(v in string_u16_pairs()) {
        check_into_iter(v);
    }

    #[test]
    fn test_range(v in small_int_pairs(), r in range_bounds_1k()) {
        check_range(v, r);
    }

    #[test]
    fn test_range_mut(v in small_int_pairs(), r in range_bounds_1k()) {
        check_range_mut(v, r);
    }
}
