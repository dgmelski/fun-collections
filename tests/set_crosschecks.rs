use lazy_clone_collections::{AvlSet, BTreeSet};
use proptest::prelude::*;
use std::collections::BTreeSet as StdSet;
use std::ops::Bound;

mod common;
use common::*;

type NarrowSet<T> = lazy_clone_collections::btree::btree_set::BTreeSet<T, 1>;

#[derive(Clone)]
struct Sets<T> {
    avl_set: AvlSet<T>,       // AvlSet
    btree_set: BTreeSet<T>,   // BTreeSet with recommended MIN_OCCUPANCY
    narrow_set: NarrowSet<T>, // BTreeSet with smallest possible MIN_OCCUPANCY
    std_set: StdSet<T>,       // std::collections::BTreeSet
}

impl<T> Sets<T>
where
    T: Clone + Ord,
{
    fn new(v: Vec<T>) -> Sets<T> {
        Sets {
            avl_set: AvlSet::from_iter(v.clone()),
            btree_set: BTreeSet::from_iter(v.clone()),
            narrow_set: NarrowSet::from_iter(v.clone()),
            std_set: StdSet::from_iter(v),
        }
    }

    // fn new_overlapping(u: Vec<T>, v: Vec<T>) -> (Sets<T>, Sets<T>) {
    //     let m1 = Self::new(u);

    //     let mut m2 = m1.clone();
    //     m2.avl_set.extend(v.clone());
    //     m2.btree_set.extend(v.clone());
    //     m2.narrow_set.extend(v.clone());
    //     m2.std_set.extend(v);

    //     (m1, m2)
    // }

    fn chk(&self)
    where
        T: Eq + std::fmt::Debug,
    {
        assert_eq_all!(
            self.avl_set.len(),
            self.btree_set.len(),
            self.narrow_set.len(),
            self.std_set.len()
        );

        assert_eq_iters(self.avl_set.iter(), self.std_set.iter());
        assert_eq_iters(self.btree_set.iter(), self.std_set.iter());
        assert_eq_iters(self.narrow_set.iter(), self.std_set.iter());
    }
}

fn check_append(u: U16Seq, v: U16Seq) {
    let mut s1 = Sets::new(u);
    let mut s2 = Sets::new(v);

    s1.avl_set.append(&mut s2.avl_set);
    s1.btree_set.append(&mut s2.btree_set);
    s1.narrow_set.append(&mut s2.narrow_set);
    s1.std_set.append(&mut s2.std_set);

    s1.chk();
    s2.chk();
}

fn check_contains(u: U16Seq) {
    let sets = Sets::new(u);

    for i in 0..64 {
        assert_eq_all!(
            sets.avl_set.contains(&i),
            sets.btree_set.contains(&i),
            sets.narrow_set.contains(&i),
            sets.std_set.contains(&i)
        );
    }
}

#[test]
fn test_first_and_last() {
    let mut sets = Sets::new(vec![(4, 0), (2, 0), (3, 0)]);

    assert_eq_all!(
        sets.avl_set.first(),
        sets.btree_set.first(),
        sets.narrow_set.first(),
        sets.std_set.first()
    );

    assert_eq_all!(
        sets.avl_set.last(),
        sets.btree_set.last(),
        sets.narrow_set.last(),
        sets.std_set.last()
    );

    assert_eq_all!(
        sets.avl_set.pop_first(),
        sets.btree_set.pop_first(),
        sets.narrow_set.pop_first(),
        sets.std_set.pop_first()
    );

    assert_eq_all!(
        sets.avl_set.pop_last(),
        sets.btree_set.pop_last(),
        sets.narrow_set.pop_last(),
        sets.std_set.pop_last()
    );

    sets.avl_set.clear();
    sets.btree_set.clear();
    sets.narrow_set.clear();
    sets.std_set.clear();
    sets.chk();

    assert_eq_all!(
        sets.avl_set.first(),
        sets.btree_set.first(),
        sets.narrow_set.first(),
        sets.std_set.first()
    );

    assert_eq_all!(
        sets.avl_set.last(),
        sets.btree_set.last(),
        sets.narrow_set.last(),
        sets.std_set.last()
    );

    assert_eq_all!(
        sets.avl_set.pop_first(),
        sets.btree_set.pop_first(),
        sets.narrow_set.pop_first(),
        sets.std_set.pop_first()
    );

    assert_eq_all!(
        sets.avl_set.pop_last(),
        sets.btree_set.pop_last(),
        sets.narrow_set.pop_last(),
        sets.std_set.pop_last()
    );
}

fn check_remove(v: U16Seq, w: Vec<u16>) {
    let mut m = Sets::new(v);

    for i in w {
        assert_eq_all!(
            m.avl_set.remove(&i),
            m.btree_set.remove(&i),
            m.narrow_set.remove(&i),
            m.std_set.remove(&i)
        );
    }
}

fn check_retain(v: U16Seq) {
    let mut m = Sets::new(v);

    fn f(k: &u16) -> bool {
        k % 2 == 0
    }

    m.avl_set.retain(f);
    m.btree_set.retain(f);
    m.narrow_set.retain(f);
    m.std_set.retain(f);
    m.chk();
}

fn check_split_off(v: U16Seq, p: u16) {
    let mut m1 = Sets::new(v);
    let m2 = Sets {
        avl_set: m1.avl_set.split_off(&p),
        btree_set: m1.btree_set.split_off(&p),
        narrow_set: m1.narrow_set.split_off(&p),
        std_set: m1.std_set.split_off(&p),
    };

    m1.chk();
    m2.chk();
}

#[test]
fn test_split_off_regr1() {
    check_split_off(vec![0], 1);
}

fn check_range(v: U16Seq, r: (Bound<u16>, Bound<u16>)) {
    let sets = Sets::new(v);
    assert_eq_iters(sets.avl_set.range(r), sets.std_set.range(r));
    assert_eq_iters(sets.btree_set.range(r), sets.std_set.range(r));
    assert_eq_iters(sets.narrow_set.range(r), sets.std_set.range(r));
}

fn check_range_back(v: U16Seq, r: (Bound<u16>, Bound<u16>)) {
    let sets = Sets::new(v);

    assert_eq_iters_back(sets.avl_set.range(r), sets.std_set.range(r));
    assert_eq_iters_back(sets.btree_set.range(r), sets.std_set.range(r));
    assert_eq_iters_back(sets.narrow_set.range(r), sets.std_set.range(r));
}

proptest! {
    #[test]
    fn test_range(v in small_int_seq(), r in range_bounds_1k()) {
        check_range(v, r);
    }

    #[test]
    fn test_range_back(v in small_int_seq(), r in range_bounds_1k()) {
        check_range_back(v, r);
    }

    #[test]
    fn test_append(u in small_int_seq(), v in small_int_seq()){
        check_append(u, v);
    }

    #[test]
    fn test_contains(u in u16_seq(64,48)) {
        check_contains(u);
    }

    #[test]
    fn test_remove(
        v in u16_seq(64, 48),
        w in prop::collection::vec(0u16..64, 48))
    {
        check_remove(v, w);
    }

    #[test]
    fn test_retain(v in small_int_seq()) {
        check_retain(v);
    }

    #[test]
    fn test_split_off(v in u16_seq(64, 48), w in 0_u16..64) {
        check_split_off(v, w);
    }
}
