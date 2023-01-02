use lazy_clone_collections::{AvlSet, BTreeSet};
use proptest::prelude::*;
use std::collections::BTreeSet as StdSet;
use std::ops::Bound;

mod common;
use common::*;

type NarrowSet<T> = lazy_clone_collections::btree::btree_set::BTreeSet<T, 1>;

#[derive(Clone, Debug)]
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

    fn new_overlapping(u: Vec<T>, v: Vec<T>) -> (Sets<T>, Sets<T>) {
        let m1 = Self::new(u);

        let mut m2 = m1.clone();
        m2.avl_set.extend(v.clone());
        m2.btree_set.extend(v.clone());
        m2.narrow_set.extend(v.clone());
        m2.std_set.extend(v);

        (m1, m2)
    }

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

        use lazy_clone_collections::Set;
        self.avl_set.check().unwrap();
        self.btree_set.check().unwrap();
        self.narrow_set.check().unwrap();
    }
}

macro_rules! make_setop_tests {
    ($iter:ident, $cons:ident, $op:tt) => {
        mod $iter {
            use super::*;
            // use proptest::prelude::*;

            fn check_iter(u: U16Seq, v: U16Seq) {
                let s = Sets::new(u);
                let t = Sets::new(v);

                let si = s.std_set.$iter(&t.std_set);
                assert_eq_iters(s.avl_set.$iter(&t.avl_set), si.clone());
                assert_eq_iters(s.btree_set.$iter(&t.btree_set), si.clone());
                assert_eq_iters(s.narrow_set.$iter(&t.narrow_set), si);

                s.chk();
                t.chk();
            }

            fn check_new(mut vs: Vec<U16Seq>) {
                let s = Sets::new_overlapping(vs.pop().unwrap(), vs.pop().unwrap());
                let t = Sets::new_overlapping(vs.pop().unwrap(), vs.pop().unwrap());

                let x_std =
                    StdSet::from_iter(s.0.std_set.$iter(&t.0.std_set).copied());

                let x = Sets {
                    avl_set: AvlSet::$cons(s.0.avl_set, t.0.avl_set),
                    btree_set: BTreeSet::$cons(s.0.btree_set, t.0.btree_set),
                    narrow_set: NarrowSet::$cons(s.0.narrow_set, t.0.narrow_set),
                    std_set: x_std,
                };

                x.chk();
                s.1.chk();
                t.1.chk();

                let x_std =
                    StdSet::from_iter(s.1.std_set.$iter(&t.1.std_set).copied());

                let x = Sets {
                    avl_set: AvlSet::$cons(s.1.avl_set, t.1.avl_set),
                    btree_set: BTreeSet::$cons(s.1.btree_set, t.1.btree_set),
                    narrow_set: NarrowSet::$cons(s.1.narrow_set, t.1.narrow_set),
                    std_set: x_std,
                };

                x.chk();
            }

            fn check_bitop(u: U16Seq, v: U16Seq) {
                let s = Sets::new(u);
                let t = Sets::new(v);

                let x = Sets {
                    avl_set: &s.avl_set $op &t.avl_set,
                    btree_set: &s.btree_set $op &t.btree_set,
                    narrow_set: &s.narrow_set $op &t.narrow_set,
                    std_set: &s.std_set $op &t.std_set,
                };

                x.chk();
            }

            proptest! {
                #[test]
                fn test_iter(u in small_int_seq(), v in small_int_seq()) {
                    check_iter(u, v);
                }

                #[test]
                fn test_new(vs in [
                    small_int_seq(),
                    small_int_seq(),
                    small_int_seq(),
                    small_int_seq(),
                ]) {
                    check_new(vs.to_vec());
                }

                #[test]
                fn test_bitop(u in tiny_int_seq(), v in tiny_int_seq()) {
                    check_bitop(u, v);
                }
            }
        }
    };
}

make_setop_tests!(difference, new_diff, -);
make_setop_tests!(intersection, new_intersection, &);
make_setop_tests!(union, new_union, |);
make_setop_tests!(symmetric_difference, new_sym_diff, ^);

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

fn check_borrowed_key(u: Vec<String>, k: &str) {
    let mut s = Sets::new(u);

    assert_eq_all!(
        s.avl_set.contains(k),
        s.btree_set.contains(k),
        s.narrow_set.contains(k),
        s.std_set.contains(k)
    );

    assert_eq_all!(
        s.avl_set.get(k),
        s.btree_set.get(k),
        s.narrow_set.get(k),
        s.std_set.get(k)
    );

    use Bound::*;
    type Q = str;
    type R<'a> = (Bound<&'a str>, Bound<&'a str>);
    let r = (Included(k), Unbounded::<&str>);

    assert_eq_iters(
        s.avl_set.range::<Q, R<'_>>(r),
        s.std_set.range::<Q, R<'_>>(r),
    );

    assert_eq_iters(
        s.btree_set.range::<Q, R<'_>>(r),
        s.std_set.range::<Q, R<'_>>(r),
    );

    assert_eq_iters(
        s.narrow_set.range::<Q, R<'_>>(r),
        s.std_set.range::<Q, R<'_>>(r),
    );

    let mut t = Sets {
        avl_set: s.avl_set.split_off(k),
        btree_set: s.btree_set.split_off(k),
        narrow_set: s.narrow_set.split_off(k),
        std_set: s.std_set.split_off(k),
    };

    s.chk();
    t.chk();

    assert_eq_all!(
        None,
        s.avl_set.take(k),
        s.btree_set.take(k),
        s.narrow_set.take(k),
        s.std_set.take(k)
    );

    assert_eq_all!(
        t.avl_set.take(k),
        t.btree_set.take(k),
        t.narrow_set.take(k),
        t.std_set.take(k)
    );

    // tests signature more than functionality
    assert_eq_all!(
        false,
        s.avl_set.remove(k),
        s.btree_set.remove(k),
        s.narrow_set.remove(k),
        s.std_set.remove(k),
        t.avl_set.remove(k),
        t.btree_set.remove(k),
        t.narrow_set.remove(k),
        t.std_set.remove(k)
    );
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

fn check_is_disjoint(u: U16Seq, v: U16Seq) {
    let s = Sets::new(u);
    let t = Sets::new(v);

    assert_eq_all!(
        s.std_set.is_disjoint(&t.std_set),
        s.avl_set.is_disjoint(&t.avl_set),
        s.btree_set.is_disjoint(&t.btree_set),
        s.narrow_set.is_disjoint(&t.narrow_set)
    );
}

fn check_is_subset_superset(u: U16Seq, v: U16Seq) {
    let s = Sets::new(u);
    let t = Sets::new(v);

    assert_eq_all!(
        s.std_set.is_subset(&t.std_set),
        s.avl_set.is_subset(&t.avl_set),
        s.btree_set.is_subset(&t.btree_set),
        s.narrow_set.is_subset(&t.narrow_set)
    );

    assert_eq_all!(
        s.std_set.is_superset(&t.std_set),
        s.avl_set.is_superset(&t.avl_set),
        s.btree_set.is_superset(&t.btree_set),
        s.narrow_set.is_superset(&t.narrow_set)
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

#[derive(Clone, Debug, Eq)]
struct Ignores2nd(String, u16);

impl From<&Ignores2nd> for (String, u16) {
    fn from(n: &Ignores2nd) -> Self {
        (n.0.clone(), n.1)
    }
}

impl From<Ignores2nd> for (String, u16) {
    fn from(n: Ignores2nd) -> Self {
        (n.0, n.1)
    }
}

impl PartialEq for Ignores2nd {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl PartialOrd for Ignores2nd {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl Ord for Ignores2nd {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

fn check_replace(u: Vec<(String, u16)>, k: String) {
    let u = Vec::from_iter(u.into_iter().map(|(x, y)| Ignores2nd(x, y)));
    let mut s = Sets::new(u);
    let kv = Ignores2nd(k.clone(), 2048);

    let orig: Option<(String, u16)> = s.std_set.get(&kv).map(|e| e.into());
    assert_eq_all!(
        orig,
        s.avl_set.replace(kv.clone()).map(|e| e.into()),
        s.btree_set.replace(kv.clone()).map(|e| e.into()),
        s.narrow_set.replace(kv.clone()).map(|e| e.into()),
        s.std_set.replace(kv).map(|e| e.into())
    );

    s.chk();

    let kv2 = Ignores2nd(k.clone(), 0);
    if s.std_set.contains(&kv2) {
        assert_eq_all!(
            Some((k, 2048)),
            s.avl_set.get(&kv2).map(|e| e.into()),
            s.btree_set.get(&kv2).map(|e| e.into()),
            s.narrow_set.get(&kv2).map(|e| e.into()),
            s.std_set.get(&kv2).map(|e| e.into())
        );
    }
}

#[test]
fn test_replace_regr1() {
    check_replace(vec![], "".to_string())
}

#[test]
fn test_replace_regr2() {
    check_replace(vec![("".into(), 0), ("".into(), 1)], "".to_string())
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

fn check_take(u: Vec<(String, u16)>, k: String) {
    let u = Vec::from_iter(u.into_iter().map(|(x, y)| Ignores2nd(x, y)));
    let mut s = Sets::new(u);
    let kv = Ignores2nd(k, 2048);
    let orig: Option<(String, u16)> = s.std_set.get(&kv).map(|e| e.into());

    assert_eq_all!(
        orig,
        s.avl_set.take(&kv).map(|e| e.into()),
        s.btree_set.take(&kv).map(|e| e.into()),
        s.narrow_set.take(&kv).map(|e| e.into()),
        s.std_set.take(&kv).map(|e| e.into())
    );

    s.chk();
}

#[cfg(feature = "serde")]
mod serde {
    #![allow(unused_imports)]
    use super::*;
    use crate::common::*;
    use proptest::prelude::*;
    use serde_test::{assert_de_tokens, assert_tokens, Token};

    fn make_tokens(m: &StdSet<u16>) -> Vec<Token> {
        let mut ts = vec![
            Token::Struct {
                name: "AvlSet",
                len: 1,
            },
            Token::Str("map"),
            Token::Map { len: Some(m.len()) },
        ];

        for &k in m.iter() {
            ts.push(Token::U16(k));
            ts.push(Token::Unit);
        }
        ts.push(Token::MapEnd);
        ts.push(Token::StructEnd);
        ts
    }

    fn check_serde(v: U16Seq) {
        let m = Sets::new(v);

        let mut ts = make_tokens(&m.std_set);
        assert_tokens(&m.avl_set, &ts);

        ts[0] = Token::Struct {
            name: "BTreeSet",
            len: 1,
        };
        assert_tokens(&m.btree_set, &ts);
        assert_tokens(&m.narrow_set, &ts);
    }

    proptest! {
        #[test]
        fn test_serde(v in small_int_seq()) {
            check_serde(v);
        }
    }
}

#[test]
fn test_set_send() {
    use std::sync::mpsc;
    use std::thread;

    let (tx, rx) = mpsc::channel();

    thread::spawn(move || {
        let v = Vec::from_iter(0u16..1024);
        let s = Sets::new(v);
        tx.send(s).unwrap();
    });

    let received = rx.recv().unwrap();
    received.chk();
}

fn check_concurrent_inserts(u: Vec<u16>, mut v: Vec<u16>) {
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    let s = Sets::new(u);
    let mut handles = Vec::new();
    let (tx, rx) = mpsc::channel();

    let x = v.len() / 4;
    for _ in 0..4 {
        let mut s = s.clone();
        let v = v.split_off(v.len().saturating_sub(x));
        let tx = tx.clone();

        let h = thread::spawn(move || {
            for i in v {
                s.avl_set.insert(i);
                s.btree_set.insert(i);
                s.narrow_set.insert(i);
                s.std_set.insert(i);
                thread::sleep(Duration::from_millis(1));
            }

            s.chk();

            tx.send(s).unwrap();
        });

        handles.push(h);
    }

    while let Some(h) = handles.pop() {
        h.join().unwrap();
    }

    s.chk();

    for _ in 0..4 {
        let t = rx.recv().unwrap();
        t.chk();
    }

    s.chk();
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
    fn test_borrowed_key(
        u in prop::collection::vec("[a-z]{0,2}", 0..384),
        k in "[a-z]{0,2}")
    {
        check_borrowed_key(u.clone(), &k);

        // "ma" is roughly in the middle of the range; more importantly, it is
        // the borrowed form of String, so we test the type signature.
        check_borrowed_key(u, "ma");
    }

    #[test]
    fn test_contains(u in u16_seq(64,48)) {
        check_contains(u);
    }

    #[test]
    fn test_is_disjoint(u in small_int_seq(), v in small_int_seq()) {
        check_is_disjoint(u, v);
    }

    #[test]
    fn test_is_subset_superset(u in small_int_seq(), v in small_int_seq()) {
        check_is_subset_superset(u, v);
    }

    #[test]
    fn test_remove(
        v in u16_seq(64, 48),
        w in prop::collection::vec(0u16..64, 48))
    {
        check_remove(v, w);
    }

    #[test]
    fn test_replace(u in string_u16_pairs(), k in "[a-z]{0,2}") {
        check_replace(u, k);
    }

    #[test]
    fn test_retain(v in small_int_seq()) {
        check_retain(v);
    }

    #[test]
    fn test_split_off(v in u16_seq(64, 48), w in 0_u16..64) {
        check_split_off(v, w);
    }

    #[test]
    fn test_take(u in string_u16_pairs(), k in "[a-z]{0,2}") {
        check_take(u, k);
    }

    #[test]
    fn test_concurrent_inserts(
        u in small_int_seq(),
        v in prop::collection::vec(0u16..1024, 64)
    ) {
        check_concurrent_inserts(u, v)
    }
}
