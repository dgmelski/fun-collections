use super::core::*;
use std::borrow::Borrow;
use std::iter::FusedIterator;
use std::sync::Arc;

#[derive(Clone, Default)]
pub struct BTreeMap<K, V> {
    root: OptNodePtr<K, V>,
    len: usize,
}

impl<K, V> BTreeMap<K, V> {
    pub fn new() -> Self {
        Self { root: None, len: 0 }
    }

    pub fn clear(&mut self) {
        self.root = None;
        self.len = 0;
    }

    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q> + Clone,
        V: Clone,
        Q: Ord + ?Sized,
    {
        self.root.as_ref().and_then(|n| n.get(key)).map(|kv| kv.1)
    }

    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q> + Clone,
        V: Clone,
        Q: Ord + ?Sized,
    {
        self.root
            .as_mut()
            .and_then(|n| Arc::make_mut(n).get_mut(key))
    }

    pub fn insert(&mut self, key: K, val: V) -> Option<V>
    where
        K: Clone + Ord,
        V: Clone,
    {
        use InsertResult::*;

        if let Some(arc) = self.root.as_mut() {
            let n = Arc::make_mut(arc);
            match n.insert((key, val)) {
                Absorbed => {
                    self.len += 1;
                    None
                }

                Replaced(_, val) => Some(val),

                Split(lt_k, kv) => {
                    let gt_k = self.root.take().unwrap();
                    let b = Node::new_branch(lt_k.unwrap(), kv, gt_k);
                    self.root = Some(Arc::new(b));
                    self.len += 1;
                    None
                }
            }
        } else {
            let arc = Arc::new(Node::new_leaf((key, val)));
            self.root = Some(arc);
            self.len = 1;
            None
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn iter(
        &self,
    ) -> impl DoubleEndedIterator<Item = (&K, &V)> + ExactSizeIterator + FusedIterator
    {
        Iter::new(self.root.as_ref(), self.len)
    }

    pub fn iter_mut(
        &mut self,
    ) -> impl DoubleEndedIterator<Item = (&K, &mut V)>
           + ExactSizeIterator
           + FusedIterator
    where
        K: Clone,
        V: Clone,
    {
        IterMut::new(self.root.as_mut(), self.len)
    }

    pub fn len(&self) -> usize {
        self.len
    }

    fn rm_and_rebal<RM>(&mut self, remover: RM) -> Option<(K, V)>
    where
        K: Clone,
        V: Clone,
        RM: FnOnce(&mut Node<K, V>) -> Option<((K, V), IsUnderPop)>,
    {
        let rc = self.root.as_mut()?;
        let n = Arc::make_mut(rc);
        let (old_kv, is_under_pop) = remover(n)?;

        self.len -= 1;

        if is_under_pop.0 && n.is_empty() {
            self.root = n.pop_child();
        }

        Some(old_kv)
    }

    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q> + Clone,
        V: Clone,
        Q: Ord + ?Sized,
    {
        // TODO: avoid unnecessary cloning
        // if !self.contains_key(key) {
        //     return None;
        // }

        self.rm_and_rebal(|n| n.remove(key)).map(|e| e.1)
    }

    #[cfg(test)]
    fn chk(&self)
    where
        K: Clone + Ord,
    {
        if let Some(root) = self.root.as_ref() {
            assert_eq!(root.chk(), self.len);
        } else {
            assert_eq!(self.len, 0);
        }
    }
}

impl<K: Clone, V: Clone> IntoIterator for BTreeMap<K, V> {
    type Item = (K, V);
    type IntoIter = IntoIter<K, V>;

    fn into_iter(self) -> Self::IntoIter {
        new_into_iter(self.root, self.len)
    }
}

impl<K, V> Extend<(K, V)> for BTreeMap<K, V>
where
    K: Clone + Ord,
    V: Clone,
{
    fn extend<T: IntoIterator<Item = (K, V)>>(&mut self, iter: T) {
        for (k, v) in iter {
            self.insert(k, v);
        }
    }
}

impl<K, V> FromIterator<(K, V)> for BTreeMap<K, V>
where
    K: Clone + Ord,
    V: Clone,
{
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
        let mut fmap = BTreeMap::new();
        fmap.extend(iter);
        fmap
    }
}

#[cfg(test)]
mod life_tracker {
    use std::cell::RefCell;

    thread_local! {
        static CNT_LIVE: RefCell<isize> = RefCell::new(0);
    }

    #[derive(Debug)]
    pub struct Counted<T> {
        // private interior so nothing external (to mod) can create it
        v: T,
    }

    impl<T> Counted<T> {
        pub fn new(v: T) -> Self {
            CNT_LIVE.with(|x| *x.borrow_mut() += 1);
            Self { v }
        }

        pub fn as_mut(&mut self) -> &mut T {
            &mut self.v
        }

        pub fn as_ref(&self) -> &T {
            &self.v
        }
    }

    impl<T> std::borrow::Borrow<T> for Counted<T> {
        fn borrow(&self) -> &T {
            &self.v
        }
    }

    impl<T: Clone> Clone for Counted<T> {
        fn clone(&self) -> Self {
            Counted::new(self.v.clone())
        }
    }

    impl<T> Drop for Counted<T> {
        fn drop(&mut self) {
            CNT_LIVE.with(|x| *x.borrow_mut() -= 1)
        }
    }

    pub fn get_cnt_live() -> isize {
        CNT_LIVE.with(|x| *x.borrow())
    }
}

#[cfg(test)]
mod test {
    extern crate quickcheck;
    use super::*;
    use crate::btree::core::test::btree_strat;
    use life_tracker::*;
    use proptest::prelude::*;
    use quickcheck::quickcheck;

    #[test]
    fn test_life_tracker() {
        // fails with "field 'marker' of struct life_tracker::Counted is private"
        // let c = Counted {
        //     marker: std::marker::PhantomData,
        // };

        assert_eq!(get_cnt_live(), 0);

        {
            let c = Counted::new(0);
            assert_eq!(get_cnt_live(), 1);

            let d = c.clone();
            assert_eq!(get_cnt_live(), 2);

            let mut m = BTreeMap::new();
            m.insert(1, c);
            assert_eq!(get_cnt_live(), 2);

            let mut n = m.clone();
            assert_eq!(get_cnt_live(), 2);

            n.insert(2, d);
            assert_eq!(get_cnt_live(), 3);
        }

        assert_eq!(get_cnt_live(), 0);
    }

    fn check_insert(elems: Vec<(u16, u8)>) {
        let mut m = BTreeMap::new();
        let mut n = std::collections::BTreeMap::new();

        for (k, v) in elems {
            m.insert(k, Counted::new(v));
            n.insert(k, v);
            assert_eq!(get_cnt_live(), n.len() as isize);
        }

        for (k, v) in n.iter() {
            assert_eq!(Some(v), m.get(k).map(|x| x.borrow()));
        }

        m.clear();
        assert_eq!(get_cnt_live(), 0);
    }

    #[test]
    fn test_insert_regr1() {
        check_insert(vec![(0, 126)]);
    }

    fn check_remove1(keys: Vec<u8>, tgts: Vec<u8>) {
        let mut m = BTreeMap::new();
        let mut n = std::collections::BTreeMap::new();
        for k in keys {
            m.insert(k, Counted::new(k));
            m.chk();
            n.insert(k, k);
            assert_eq!(get_cnt_live(), n.len() as isize);
        }

        for t in tgts {
            assert_eq!(m.remove(&t).map(|v| *v.borrow()), n.remove(&t));
            m.chk();
            assert_eq!(get_cnt_live(), n.len() as isize);
        }
    }

    #[test]
    fn test_remove_regr1() {
        check_remove1(vec![0, 2, 3, 4, 7, 5, 8, 6, 9, 10, 1, 11], vec![0, 1]);
    }

    #[test]
    fn test_into_iter_drops() {
        {
            let mut m = BTreeMap::new();
            for i in 0..30 {
                m.insert(i, Counted::new(i));
            }

            assert_eq!(get_cnt_live(), 30);

            {
                // create an IntoIter, but only partially consume it
                let mut iter = m.into_iter();
                for i in 0..15 {
                    assert_eq!(iter.next().map(|kv| kv.0), Some(i));
                    assert_eq!(get_cnt_live(), 29 - i);
                }
            }

            // check that the drop of the partially consumed IntoIter resulted
            // in all elements being dropped
            assert_eq!(get_cnt_live(), 0);
        }
    }

    fn check_overlapping_iter_mut(u: Vec<u8>, v: Vec<u8>) {
        let mut m = BTreeMap::new();
        for i in u {
            m.insert(i, Counted::new(i ^ 0x0f));
        }
        let len_m = m.len() as isize;

        let mut n = m.clone();
        for i in v {
            n.insert(i, Counted::new(i ^ 0x0f));
        }
        let len_n = n.len() as isize;

        assert!(get_cnt_live() >= len_m);

        for (_, v) in m.iter_mut() {
            *v.as_mut() ^= 0xf0;
        }

        assert_eq!(get_cnt_live(), len_m + len_n);

        for (k, v) in m.iter() {
            assert_eq!(&(k ^ 0xff), v.as_ref());
        }

        for (k, v) in n.iter() {
            assert_eq!(&(k ^ 0x0f), v.as_ref());
        }

        m.clear();
        assert_eq!(get_cnt_live(), len_n);

        n.clear();
        assert_eq!(get_cnt_live(), 0);
    }

    #[test]
    fn test_overlapping_iter_mut_regr1() {
        check_overlapping_iter_mut(vec![0], vec![])
    }

    #[test]
    fn test_overlapping_iter_mut_regr2() {
        check_overlapping_iter_mut(
            vec![
                51, 70, 52, 2, 1, 53, 0, 3, 71, 4, 12, 72, 13, 54, 14, 15, 55,
                73, 74, 7, 16, 56, 17, 58, 75,
            ],
            vec![
                32, 33, 59, 60, 45, 18, 61, 46, 34, 35, 5, 76, 77, 62, 19, 36,
                63, 47, 78, 37, 48, 8, 57, 64, 20, 6, 49, 38, 21, 65, 50, 39,
                66, 9, 67, 40, 22, 10, 23, 41, 24, 25, 26, 79, 27, 28, 42, 29,
                43, 30, 80, 44, 11, 31, 68, 81, 69, 82,
            ],
        );
    }

    fn check_overlapping_into_iter(u: Vec<u8>, v: Vec<u8>) {
        let mut m = BTreeMap::new();
        for i in u {
            m.insert(i, Counted::new(i ^ 0x80));
        }
        let len_m = m.len() as isize;

        let mut n = m.clone();
        for i in v {
            n.insert(i, Counted::new(i ^ 0x7f));
        }
        let len_n = n.len() as isize;

        assert!(get_cnt_live() >= len_m);

        for (k, v) in m {
            assert_eq!(&(k ^ 0x80), v.borrow());
        }

        assert_eq!(get_cnt_live(), len_n);

        n.into_iter().next();

        assert_eq!(get_cnt_live(), 0);
    }

    #[test]
    fn test_overlapping_into_iter_regr1() {
        check_overlapping_into_iter(vec![0], vec![])
    }

    quickcheck! {
        #[test]
        fn qc_test_insert(elems: Vec<(u16, u8)>) -> () {
            check_insert(elems);
        }

        #[test]
        fn qc_test_remove(keys: Vec<u8>, tgts: Vec<u8>) -> () {
            check_remove1(keys, tgts);
        }

        #[test]
        fn qc_test_overlapping_iter_mut(u: Vec<u8>, v: Vec<u8>) -> () {
            check_overlapping_iter_mut(u, v);
        }

        #[test]
        fn qc_test_overlapping_into_iter(u: Vec<u8>, v: Vec<u8>) -> () {
            check_overlapping_into_iter(u, v);
        }
    }

    fn check_get(m: BTreeMap<u32, u32>, tgts: Vec<u32>) {
        m.chk();

        let mut n = std::collections::BTreeMap::new();
        for (k, v) in m.iter() {
            n.insert(*k, *v);
        }

        for t in tgts.iter() {
            assert_eq!(m.get(t), n.get(t));
        }
    }

    fn check_insert2(mut m: BTreeMap<u32, u32>, tgts: Vec<u32>) {
        m.chk();

        let mut n = std::collections::BTreeMap::new();
        for (k, v) in m.iter() {
            n.insert(*k, *v);
        }

        for t in tgts.iter().copied() {
            assert_eq!(m.insert(t, t), n.insert(t, t));
            m.chk();
        }
    }

    fn check_remove2(mut m: BTreeMap<u32, u32>, tgts: Vec<u32>) {
        m.chk();

        let mut n = std::collections::BTreeMap::new();
        for (k, v) in m.iter() {
            n.insert(*k, *v);
        }

        for t in tgts.iter() {
            assert_eq!(m.remove(t), n.remove(t));
            m.chk();
        }
    }

    fn check_strat((n, len): (NodePtr<u32, u32>, usize)) {
        let m = BTreeMap { root: Some(n), len };
        assert!(m.get(&4).is_some());
    }

    proptest! {
        #[test]
        fn test_strat(x in btree_strat(3)) {
            check_strat(x);
        }

        #[test]
        fn test_get(
            (root, len) in btree_strat(3),
            t in prop::collection::vec(0u32..1024, 1..256))
        {
            check_get(BTreeMap{ root: Some(root), len }, t);
        }

        #[test]
        fn test_insert2(
            (root, len) in btree_strat(3),
            t in prop::collection::vec(0u32..1024, 1..256))
        {
            check_insert2(BTreeMap{ root: Some(root), len }, t);
        }

        #[test]
        fn test_remove2(
            (root, len) in btree_strat(3),
            t in prop::collection::vec(0u32..1024, 1..256))
        {
            check_remove2(BTreeMap{ root: Some(root), len }, t);
        }
    }
}
