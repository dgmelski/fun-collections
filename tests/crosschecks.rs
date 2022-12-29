use lazy_clone_collections::{AvlMap, BTreeMap};
use proptest::prelude::*;
use std::collections::BTreeMap as StdMap;
use std::ops::Bound;
use Bound::*;

mod common;
use common::*;

type NarrowMap<K, V> = lazy_clone_collections::btree::BTreeMap<K, V, 1>;

#[derive(Clone)]
struct Maps<K, V> {
    avl_map: AvlMap<K, V>,       // AvlMap
    btree_map: BTreeMap<K, V>,   // BTreeMap with recommended MIN_OCCUPANCY
    narrow_map: NarrowMap<K, V>, // BTreeMap with smallest possible MIN_OCCUPANCY
    std_map: StdMap<K, V>,       // std::collections::BTreeMap
}

impl<K, V> Maps<K, V> {
    fn new(v: Vec<(K, V)>) -> Maps<K, V>
    where
        K: Clone + Ord,
        V: Clone,
    {
        Maps {
            avl_map: AvlMap::from_iter(v.clone()),
            btree_map: BTreeMap::from_iter(v.clone()),
            narrow_map: NarrowMap::from_iter(v.clone()),
            std_map: StdMap::from_iter(v),
        }
    }

    fn new_overlapping(
        u: Vec<(K, V)>,
        v: Vec<(K, V)>,
    ) -> (Maps<K, V>, Maps<K, V>)
    where
        K: Clone + Ord,
        V: Clone,
    {
        let m1 = Self::new(u);

        let mut m2 = m1.clone();
        m2.avl_map.extend(v.clone());
        m2.btree_map.extend(v.clone());
        m2.narrow_map.extend(v.clone());
        m2.std_map.extend(v);

        (m1, m2)
    }
}

fn check_range(v: SmallIntPairs, r: (Bound<u16>, Bound<u16>)) {
    let maps = Maps::new(v);
    assert_eq_iters(maps.avl_map.range(r), maps.std_map.range(r));
    assert_eq_iters(maps.btree_map.range(r), maps.std_map.range(r));
    assert_eq_iters(maps.narrow_map.range(r), maps.std_map.range(r));
}

#[test]
fn range_regr1() {
    check_range(vec![(248, 0), (249, 0), (0, 0)], (Unbounded, Excluded(248)));
}

proptest! {
    #[test]
    fn test_range(v in small_int_pairs(), r in range_bounds_1k()) {
        check_range(v, r);
    }
}
