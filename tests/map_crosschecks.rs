use lazy_clone_collections::{AvlMap, BTreeMap};
use proptest::prelude::*;
use std::cell::RefCell;
use std::collections::BTreeMap as StdMap;
use std::ops::Bound;
use std::rc::Rc;
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

impl<K, V> Maps<K, V>
where
    K: Clone + Ord,
    V: Clone,
{
    fn new(v: Vec<(K, V)>) -> Maps<K, V> {
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
    ) -> (Maps<K, V>, Maps<K, V>) {
        let m1 = Self::new(u);

        let mut m2 = m1.clone();
        m2.avl_map.extend(v.clone());
        m2.btree_map.extend(v.clone());
        m2.narrow_map.extend(v.clone());
        m2.std_map.extend(v);

        (m1, m2)
    }

    fn chk(&self)
    where
        K: Eq + std::fmt::Debug,
        V: Eq + std::fmt::Debug,
    {
        assert_eq_all!(
            self.avl_map.len(),
            self.btree_map.len(),
            self.narrow_map.len(),
            self.std_map.len()
        );

        assert_eq_iters(self.avl_map.iter(), self.std_map.iter());
        assert_eq_iters(self.btree_map.iter(), self.std_map.iter());
        assert_eq_iters(self.narrow_map.iter(), self.std_map.iter());
    }
}

fn check_append(u: U16Pairs, v: U16Pairs) {
    let mut m1 = Maps::new(u);
    let mut m2 = Maps::new(v);

    m1.avl_map.append(&mut m2.avl_map);
    m1.btree_map.append(&mut m2.btree_map);
    m1.narrow_map.append(&mut m2.narrow_map);
    m1.std_map.append(&mut m2.std_map);

    m1.chk();
    m2.chk();
}

fn check_contains_key(u: U16Pairs) {
    let maps = Maps::new(u);

    for i in 0..64 {
        assert_eq_all!(
            maps.avl_map.contains_key(&i),
            maps.btree_map.contains_key(&i),
            maps.narrow_map.contains_key(&i),
            maps.std_map.contains_key(&i)
        );
    }
}

fn check_and_modify(v: U16Pairs, i: u16) {
    let mut maps = Maps::new(v);

    let k = *maps.std_map.entry(i).and_modify(|v| *v = 10101).key();

    assert_eq_all!(
        &k,
        maps.avl_map.entry(i).and_modify(|v| *v = 10101).key(),
        maps.btree_map.entry(i).and_modify(|v| *v = 10101).key(),
        maps.narrow_map.entry(i).and_modify(|v| *v = 10101).key()
    );

    maps.chk();
}

fn check_or_default(v: U16Pairs, i: u16) {
    let mut maps = Maps::new(v);

    assert_eq_all!(
        maps.std_map.entry(i).or_default(),
        maps.avl_map.entry(i).or_default(),
        maps.btree_map.entry(i).or_default(),
        maps.narrow_map.entry(i).or_default()
    );
}

fn check_or_insert(v: U16Pairs, i: u16) {
    let mut maps = Maps::new(v);

    assert_eq_all!(
        maps.std_map.entry(i).or_insert(3200),
        maps.avl_map.entry(i).or_insert(3200),
        maps.btree_map.entry(i).or_insert(3200),
        maps.narrow_map.entry(i).or_insert(3200)
    );
}

fn check_or_insert_with(v: U16Pairs, i: u16) {
    let mut maps = Maps::new(v);

    assert_eq_all!(
        maps.std_map.entry(i).or_insert_with(|| 64000),
        maps.avl_map.entry(i).or_insert_with(|| 64000),
        maps.btree_map.entry(i).or_insert_with(|| 64000),
        maps.narrow_map.entry(i).or_insert_with(|| 64000)
    );
}

fn check_or_insert_with_key(v: U16Pairs, i: u16) {
    let mut maps = Maps::new(v);

    assert_eq_all!(
        maps.std_map.entry(i).or_insert_with_key(|k| k ^ 0xFFFF),
        maps.avl_map.entry(i).or_insert_with_key(|k| k ^ 0xFFFF),
        maps.btree_map.entry(i).or_insert_with_key(|k| k ^ 0xFFFF),
        maps.narrow_map.entry(i).or_insert_with_key(|k| k ^ 0xFFFF)
    );
}

#[test]
fn test_first_and_last() {
    let mut maps = Maps::new(vec![(4, 0), (2, 0), (3, 0)]);

    assert_eq_all!(
        maps.avl_map.first_entry().map(|e| *e.key()),
        maps.btree_map.first_entry().map(|e| *e.key()),
        maps.narrow_map.first_entry().map(|e| *e.key()),
        maps.std_map.first_entry().map(|e| *e.key())
    );

    assert_eq_all!(
        maps.avl_map.last_entry().map(|e| *e.key()),
        maps.btree_map.last_entry().map(|e| *e.key()),
        maps.narrow_map.last_entry().map(|e| *e.key()),
        maps.std_map.last_entry().map(|e| *e.key())
    );

    assert_eq_all!(
        maps.avl_map.first_key_value(),
        maps.btree_map.first_key_value(),
        maps.narrow_map.first_key_value(),
        maps.std_map.first_key_value()
    );

    assert_eq_all!(
        maps.avl_map.last_key_value(),
        maps.btree_map.last_key_value(),
        maps.narrow_map.last_key_value(),
        maps.std_map.last_key_value()
    );

    assert_eq_all!(
        maps.avl_map.pop_first(),
        maps.btree_map.pop_first(),
        maps.narrow_map.pop_first(),
        maps.std_map.pop_first()
    );

    assert_eq_all!(
        maps.avl_map.pop_last(),
        maps.btree_map.pop_last(),
        maps.narrow_map.pop_last(),
        maps.std_map.pop_last()
    );

    maps.avl_map.clear();
    maps.btree_map.clear();
    maps.narrow_map.clear();
    maps.std_map.clear();
    maps.chk();

    assert_eq_all!(
        maps.avl_map.first_entry().map(|e| *e.key()),
        maps.btree_map.first_entry().map(|e| *e.key()),
        maps.narrow_map.first_entry().map(|e| *e.key()),
        maps.std_map.first_entry().map(|e| *e.key())
    );

    assert_eq_all!(
        maps.avl_map.last_entry().map(|e| *e.key()),
        maps.btree_map.last_entry().map(|e| *e.key()),
        maps.narrow_map.last_entry().map(|e| *e.key()),
        maps.std_map.last_entry().map(|e| *e.key())
    );

    assert_eq_all!(
        maps.avl_map.first_key_value(),
        maps.btree_map.first_key_value(),
        maps.narrow_map.first_key_value(),
        maps.std_map.first_key_value()
    );

    assert_eq_all!(
        maps.avl_map.last_key_value(),
        maps.btree_map.last_key_value(),
        maps.narrow_map.last_key_value(),
        maps.std_map.last_key_value()
    );

    assert_eq_all!(
        maps.avl_map.pop_first(),
        maps.btree_map.pop_first(),
        maps.narrow_map.pop_first(),
        maps.std_map.pop_first()
    );

    assert_eq_all!(
        maps.avl_map.pop_last(),
        maps.btree_map.pop_last(),
        maps.narrow_map.pop_last(),
        maps.std_map.pop_last()
    );
}

#[test]
fn count_into_iter_clones() {
    let avl_cntr = Rc::new(RefCell::new(0));
    let btree_cntr = Rc::new(RefCell::new(0));
    let narrow_cntr = Rc::new(RefCell::new(0));
    let std_cntr = Rc::new(RefCell::new(0));

    let m1 = Maps {
        avl_map: AvlMap::from_iter(vec![
            (5, CloneCounter::new(avl_cntr.clone())),
            (3, CloneCounter::new(avl_cntr.clone())),
            (1, CloneCounter::new(avl_cntr.clone())),
            (2, CloneCounter::new(avl_cntr.clone())),
            (4, CloneCounter::new(avl_cntr.clone())),
        ]),

        btree_map: BTreeMap::from_iter(vec![
            (5, CloneCounter::new(btree_cntr.clone())),
            (3, CloneCounter::new(btree_cntr.clone())),
            (1, CloneCounter::new(btree_cntr.clone())),
            (2, CloneCounter::new(btree_cntr.clone())),
            (4, CloneCounter::new(btree_cntr.clone())),
        ]),

        narrow_map: NarrowMap::from_iter(vec![
            (5, CloneCounter::new(narrow_cntr.clone())),
            (3, CloneCounter::new(narrow_cntr.clone())),
            (1, CloneCounter::new(narrow_cntr.clone())),
            (2, CloneCounter::new(narrow_cntr.clone())),
            (4, CloneCounter::new(narrow_cntr.clone())),
        ]),

        std_map: StdMap::from_iter(vec![
            (5, CloneCounter::new(std_cntr.clone())),
            (3, CloneCounter::new(std_cntr.clone())),
            (1, CloneCounter::new(std_cntr.clone())),
            (2, CloneCounter::new(std_cntr.clone())),
            (4, CloneCounter::new(std_cntr.clone())),
        ]),
    };

    assert_eq_all!(
        0,
        *avl_cntr.borrow(),
        *btree_cntr.borrow(),
        *narrow_cntr.borrow(),
        *std_cntr.borrow()
    );

    // when we clone m1, only the elements of std_map should increment
    let m2 = m1.clone();
    assert_eq_all!(
        0,
        *avl_cntr.borrow(),
        *btree_cntr.borrow(),
        *narrow_cntr.borrow()
    );
    assert_eq!(5, *std_cntr.borrow());

    fn cmp_iters(m: Maps<usize, CloneCounter>) {
        let mut x = m.avl_map.into_iter();
        let mut y = m.btree_map.into_iter();
        let mut z = m.narrow_map.into_iter();
        let mut w = m.std_map.into_iter();
        loop {
            match (x.next(), y.next(), z.next(), w.next()) {
                (None, None, None, None) => break,

                (a, b, c, d) => {
                    assert_eq_all!(a, b, c, d);
                }
            }
        }
    }

    // The maps are shared and the into_iter's for the lazy-clone collectors
    // will have to clone and their counters will catch up.
    cmp_iters(m1);
    assert_eq_all!(
        5,
        *avl_cntr.borrow(),
        *btree_cntr.borrow(),
        *narrow_cntr.borrow(),
        *std_cntr.borrow()
    );

    // m1 was consumed, so m2 received sole ownership.  Consuming it with the into
    // iter will not cause any additional clones.
    cmp_iters(m2);
    assert_eq_all!(
        5,
        *avl_cntr.borrow(),
        *btree_cntr.borrow(),
        *narrow_cntr.borrow(),
        *std_cntr.borrow()
    );
}

fn check_into_keys(v: U16Pairs) {
    let m = Maps::new(v);

    assert_eq_iters(m.avl_map.into_keys(), m.btree_map.into_keys());
    assert_eq_iters(m.narrow_map.into_keys(), m.std_map.into_keys());
}

fn check_into_values(v: U16Pairs) {
    let m = Maps::new(v);

    assert_eq_iters(m.avl_map.into_values(), m.btree_map.into_values());
    assert_eq_iters(m.narrow_map.into_values(), m.std_map.into_values());
}

fn check_keys(v: U16Pairs) {
    let m = Maps::new(v);

    assert_eq_iters(m.avl_map.keys(), m.btree_map.keys());
    assert_eq_iters(m.narrow_map.keys(), m.std_map.keys());
}

fn check_values(v: U16Pairs) {
    let m = Maps::new(v);

    assert_eq_iters(m.avl_map.values(), m.btree_map.values());
    assert_eq_iters(m.narrow_map.values(), m.std_map.values());
}

fn check_remove(v: U16Pairs, w: Vec<u16>) {
    let mut m = Maps::new(v);

    for i in w {
        assert_eq_all!(
            m.avl_map.remove(&i),
            m.btree_map.remove(&i),
            m.narrow_map.remove(&i),
            m.std_map.remove(&i)
        );
    }
}

fn check_retain(v: U16Pairs) {
    let mut m = Maps::new(v);

    fn f(k: &u16, v: &mut u16) -> bool {
        *v ^= 0xff;
        k % 2 == 0
    }

    m.avl_map.retain(f);
    m.btree_map.retain(f);
    m.narrow_map.retain(f);
    m.std_map.retain(f);
    m.chk();
}

fn check_split_off(v: U16Pairs, p: u16) {
    let mut m1 = Maps::new(v);
    let m2 = Maps {
        avl_map: m1.avl_map.split_off(&p),
        btree_map: m1.btree_map.split_off(&p),
        narrow_map: m1.narrow_map.split_off(&p),
        std_map: m1.std_map.split_off(&p),
    };

    m1.chk();
    m2.chk();
}

#[test]
fn test_split_off_regr1() {
    check_split_off(vec![(0, 0)], 1);
}

fn check_range(v: U16Pairs, r: (Bound<u16>, Bound<u16>)) {
    let maps = Maps::new(v);
    assert_eq_iters(maps.avl_map.range(r), maps.std_map.range(r));
    assert_eq_iters(maps.btree_map.range(r), maps.std_map.range(r));
    assert_eq_iters(maps.narrow_map.range(r), maps.std_map.range(r));
}

fn check_range_back(v: U16Pairs, r: (Bound<u16>, Bound<u16>)) {
    let maps = Maps::new(v);

    assert_eq_iters_back(maps.avl_map.range(r), maps.std_map.range(r));
    assert_eq_iters_back(maps.btree_map.range(r), maps.std_map.range(r));
    assert_eq_iters_back(maps.narrow_map.range(r), maps.std_map.range(r));
}

fn check_range_mut(u: U16Pairs, v: U16Pairs, r: (Bound<u16>, Bound<u16>)) {
    let (m1, mut m2) = Maps::new_overlapping(u, v);

    m2.avl_map.range_mut(r).for_each(|(_, v)| *v = 10101);
    m2.btree_map.range_mut(r).for_each(|(_, v)| *v = 10101);
    m2.narrow_map.range_mut(r).for_each(|(_, v)| *v = 10101);
    m2.std_map.range_mut(r).for_each(|(_, v)| *v = 10101);

    assert_eq_iters(m1.avl_map.iter(), m1.std_map.iter());
    assert_eq_iters(m1.btree_map.iter(), m1.std_map.iter());
    assert_eq_iters(m1.narrow_map.iter(), m1.std_map.iter());

    assert_eq_iters(m2.avl_map.iter(), m2.std_map.iter());
    assert_eq_iters(m2.btree_map.iter(), m2.std_map.iter());
    assert_eq_iters(m2.narrow_map.iter(), m2.std_map.iter());
}

#[test]
fn range_regr1() {
    check_range(vec![(248, 0), (249, 0), (0, 0)], (Unbounded, Excluded(248)));
}

#[cfg(feature = "serde")]
mod serde {
    #![allow(unused_imports)]
    use super::*;
    use crate::common::*;
    use proptest::prelude::*;
    use serde_test::{assert_de_tokens, assert_tokens, Token};

    fn check_serde(v: U16Pairs) {
        let m = Maps::new(v);

        let mut ts = vec![Token::Map {
            len: Some(m.std_map.len()),
        }];
        for (&k, &v) in m.std_map.iter() {
            ts.push(Token::U16(k));
            ts.push(Token::U16(v));
        }
        ts.push(Token::MapEnd);

        assert_tokens(&m.avl_map, &ts);
        assert_tokens(&m.btree_map, &ts);
        assert_tokens(&m.narrow_map, &ts);
    }

    fn check_de_short_hint(v: U16Pairs, hint: usize) {
        let m = Maps::new(v);

        let hint = if m.std_map.len() <= 1 {
            0
        } else {
            // normalize the hint to something shorter than the map length
            hint % (m.std_map.len() - 1)
        };

        let mut ts = vec![Token::Map { len: Some(hint) }];
        for (&k, &v) in m.std_map.iter() {
            ts.push(Token::U16(k));
            ts.push(Token::U16(v));
        }
        ts.push(Token::MapEnd);

        assert_de_tokens(&m.avl_map, &ts);
        assert_de_tokens(&m.btree_map, &ts);
        assert_de_tokens(&m.narrow_map, &ts);
    }

    #[test]
    fn test_de_short_hint_regr1() {
        check_de_short_hint(vec![(0, 0)], 1);
    }

    proptest! {
        #[test]
        fn test_serde(v in small_int_pairs()) {
            check_serde(v);
        }

        #[test]
        fn test_de_short_hint(
            (v, hint) in
            small_int_pairs().prop_flat_map(|v| {
                let len = v.len();
                (Just(v), (0..=len))
            }))
        {
            check_de_short_hint(v, hint);
        }
    }
}

proptest! {
    #[test]
    fn test_range(v in small_int_pairs(), r in range_bounds_1k()) {
        check_range(v, r);
    }

    #[test]
    fn test_range_back(v in small_int_pairs(), r in range_bounds_1k()) {
        check_range_back(v, r);
    }

    #[test]
    fn test_range_mut(
        u in small_int_pairs(),
        v in small_int_pairs(),
        r in range_bounds_1k()
    ) {
        check_range_mut(u, v, r);
    }

    #[test]
    fn test_append(u in small_int_pairs(), v in small_int_pairs()){
        check_append(u, v);
    }

    #[test]
    fn test_contains_key(u in u16_pairs(64,48)) {
        check_contains_key(u);
    }

    #[test]
    fn test_and_modify(v in u16_pairs(64, 48), i in 0u16..64) {
        check_and_modify(v, i);
    }

    #[test]
    fn test_or_default(v in u16_pairs(64, 48), i in 0u16..64) {
        check_or_default(v, i);
    }

    #[test]
    fn test_or_insert(v in u16_pairs(64, 48), i in 0u16..64) {
        check_or_insert(v, i);
    }

    #[test]
    fn test_or_insert_with(v in u16_pairs(64, 48), i in 0u16..64) {
        check_or_insert_with(v, i);
    }

    #[test]
    fn test_or_insert_with_key(v in u16_pairs(64, 48), i in 0u16..64) {
        check_or_insert_with_key(v, i);
    }

    #[test]
    fn test_into_keys(u in small_int_pairs()) {
        check_into_keys(u);
    }

    #[test]
    fn test_into_values(u in small_int_pairs()) {
        check_into_values(u);
    }

    #[test]
    fn test_keys(u in small_int_pairs()) {
        check_keys(u);
    }

    #[test]
    fn test_values(u in small_int_pairs()) {
        check_values(u);
    }

    #[test]
    fn test_remove(
        v in u16_pairs(64, 48),
        w in prop::collection::vec(0u16..64, 48))
    {
        check_remove(v, w);
    }

    #[test]
    fn test_retain(v in small_int_pairs()) {
        check_retain(v);
    }

    #[test]
    fn test_split_off(v in u16_pairs(64, 48), w in 0_u16..64) {
        check_split_off(v, w);
    }
}
