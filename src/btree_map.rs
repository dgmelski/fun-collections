#![allow(dead_code)] // FIXME

use std::borrow::Borrow;
use std::cmp::Ordering::*;
use std::mem::replace;
use std::rc::Rc;

type NodePtr<K, V, const N: usize> = Option<Rc<Node<K, V, N>>>;
type ChildAndElem<K, V, const N: usize> = (NodePtr<K, V, N>, K, V);

struct Node<K, V, const N: usize> {
    elems: Vec<ChildAndElem<K, V, N>>,
    right: NodePtr<K, V, N>,
}

impl<K: Clone, V: Clone, const N: usize> Clone for Node<K, V, N> {
    fn clone(&self) -> Self {
        Self {
            elems: self.elems.clone(),
            right: self.right.clone(),
        }
    }
}

enum InsertResult<K, V, const N: usize> {
    Replaced(V),
    Split(ChildAndElem<K, V, N>),
    Absorbed,
}

fn unwrap_or_clone<K, V, const N: usize>(n: NodePtr<K, V, N>) -> Node<K, V, N>
where
    K: Clone,
    V: Clone,
{
    let mut n = n.unwrap();
    Rc::make_mut(&mut n);
    match Rc::try_unwrap(n) {
        Ok(n) => n,
        Err(_) => panic!("Rc::try_unwrap should succeed after Rc::make_mut"),
    }
}

struct NeedsRebal(bool);

impl<K, V, const N: usize> Node<K, V, N> {
    // minimum and maximum element counts for non-root nodes
    const MIN_OCCUPANCY: usize = N;
    const MAX_OCCUPANCY: usize = 2 * N;

    fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Ord,
    {
        for (left, k, v) in self.elems.iter() {
            match key.cmp(k.borrow()) {
                Less => return left.as_ref().and_then(|n| n.get(key)),
                Equal => return Some(v),
                Greater => (),
            }
        }

        self.right.as_ref().and_then(|n| n.get(key))
    }

    fn insert(&mut self, key: K, val: V) -> InsertResult<K, V, N>
    where
        K: Clone + Ord,
        V: Clone,
    {
        use InsertResult::*;

        let mut ub_x = 0;
        for (_, k, v) in self.elems.iter_mut() {
            match key.cmp(k) {
                Less => break,
                Equal => return Replaced(std::mem::replace(v, val)),
                Greater => (),
            }

            ub_x += 1;
        }

        // Recurse to the appropriate child if it exists (we're not a leaf).
        // If we are a leaf, pretend that we visited a child and it resulted in
        // needing to insert a new separator at this level.
        let res = if ub_x < self.elems.len() {
            match self.elems[ub_x].0.as_mut() {
                Some(n) => Rc::make_mut(n).insert(key, val),
                None => Split((None, key, val)),
            }
        } else {
            match self.right.as_mut() {
                Some(n) => Rc::make_mut(n).insert(key, val),
                None => Split((None, key, val)),
            }
        };

        // update for a node split at the next level down
        if let Split(s) = res {
            // TODO: split before insert to reduce memmove
            self.elems.insert(ub_x, s);
            if self.elems.len() <= Self::MAX_OCCUPANCY {
                return Absorbed;
            }

            let mut other_half = self.elems.split_off(Self::MIN_OCCUPANCY + 1);
            std::mem::swap(&mut self.elems, &mut other_half);
            let (lf_kid, k, v) = other_half.pop().unwrap();
            let lefts = Some(Rc::new(Node {
                elems: other_half,
                right: lf_kid,
            }));

            return Split((lefts, k, v));
        } else {
            // res is Replaced(v) or Absorbed
            return res;
        }
    }

    fn is_leaf(&self) -> bool {
        self.right.is_none()
    }

    fn rot_lf(&mut self, idx: usize)
    where
        K: Clone,
        V: Clone,
    {
        assert!(self.right.is_some(), "cannot rotate a leaf's children");

        // extract the new separator (k2 and v2) from child on right of idx
        let right = if idx < self.elems.len() - 1 {
            &mut self.elems[idx + 1].0
        } else {
            &mut self.right
        };

        let right = right.as_mut().unwrap();
        assert!(
            right.elems.len() > Self::MIN_OCCUPANCY,
            "rot_lf from an impovershed child"
        );

        let n = Rc::make_mut(right);

        let (k1_to_k2, k2, v2) = n.elems.remove(0);

        // replace (and take) the old separator (k1 and v1)
        let pivot = &mut self.elems[idx];
        let k1 = std::mem::replace(&mut pivot.1, k2);
        let v1 = std::mem::replace(&mut pivot.2, v2);

        // push the old separator to the end of left
        let mut piv_child = pivot.0.as_mut().unwrap();
        assert!(
            piv_child.elems.len() < Self::MIN_OCCUPANCY,
            "rot_lf into a rich child"
        );

        let piv_child = Rc::make_mut(&mut piv_child);
        let lt_k1 = std::mem::replace(&mut piv_child.right, k1_to_k2);
        piv_child.elems.push((lt_k1, k1, v1));
    }

    fn rot_rt(&mut self, idx: usize)
    where
        K: Clone,
        V: Clone,
    {
        assert!(self.right.is_some(), "cannot rotate a leaf's children");

        // idx holds the current separator, k1.  Get the pieces that will rotate
        // in to replace k1 & v1.
        let mut left = self.elems[idx].0.as_mut().unwrap();
        assert!(
            left.elems.len() > N / 2 - 1,
            "rot_rt from impoverished child"
        );

        let left = Rc::make_mut(&mut left);
        let (lt_k0, k0, v0) = left.elems.pop().unwrap();
        let k0_to_k1 = std::mem::replace(&mut left.right, lt_k0);

        // move k0 and v0 into the pivot position
        let k1 = std::mem::replace(&mut self.elems[idx].1, k0);
        let v1 = std::mem::replace(&mut self.elems[idx].2, v0);

        // move k1 and v1 down and to the right of the pivot
        let right = if idx < self.elems.len() - 1 {
            &mut self.elems[idx + 1].0
        } else {
            &mut self.right
        };
        let mut right = right.as_mut().unwrap();
        let right = Rc::make_mut(&mut right);

        right.elems.insert(0, (k0_to_k1, k1, v1));
    }

    // merge the subtree self.index[at].0 and the one to its right
    fn merge_kids(&mut self, at: usize)
    where
        K: Clone,
        V: Clone,
    {
        let (lhs, k, v) = self.elems.remove(at);
        let mut lhs = unwrap_or_clone(lhs);

        let rhs = if at < self.elems.len() {
            self.elems[at].0.take()
        } else {
            self.right.take()
        };
        let mut rhs = unwrap_or_clone(rhs);

        let lt_k = replace(&mut lhs.right, rhs.right.take());

        lhs.elems.push((lt_k, k, v));
        lhs.elems.extend(rhs.elems);

        if at < self.elems.len() {
            self.elems[at].0 = Some(Rc::new(lhs));
        } else {
            self.right = Some(Rc::new(lhs));
        }
    }

    // rebalances when the self.elems[at] is underpopulated
    fn rebal(&mut self, at: usize) -> NeedsRebal
    where
        K: Clone,
        V: Clone,
    {
        if at > 0 {
            let sz = self.elems[at - 1].0.as_ref().unwrap().elems.len();
            if sz > Self::MIN_OCCUPANCY {
                self.rot_rt(at - 1);
            } else {
                self.merge_kids(at - 1);
            }
        } else if self.elems.len() > 1 {
            let sz = self.elems[at + 1].0.as_ref().unwrap().elems.len();
            if sz > Self::MIN_OCCUPANCY {
                self.rot_lf(at);
            } else {
                self.merge_kids(at);
            }
        } else {
            // we must be the root
            let sz = self.right.as_ref().unwrap().elems.len();
            if sz > Self::MIN_OCCUPANCY {
                self.rot_lf(at);
            } else {
                self.merge_kids(at);
            }
        }

        NeedsRebal(self.elems.len() < Self::MIN_OCCUPANCY)
    }

    fn rm_greatest(&mut self) -> (K, V, NeedsRebal)
    where
        K: Clone,
        V: Clone,
    {
        if let Some(rt) = self.right.as_mut() {
            // self is a branch; recurse to the rightmost child
            let rt = Rc::make_mut(rt);
            let ret = rt.rm_greatest();
            if let &NeedsRebal(true) = &ret.2 {
                (ret.0, ret.1, self.rebal(self.elems.len()))
            } else {
                ret
            }
        } else {
            // self is a leaf
            let (_, k, v) = self.elems.pop().unwrap();
            (k, v, NeedsRebal(self.elems.len() < Self::MIN_OCCUPANCY))
        }
    }

    fn remove<Q>(&mut self, key: &Q) -> (Option<V>, NeedsRebal)
    where
        K: Borrow<Q> + Clone,
        V: Clone,
        Q: Ord,
    {
        for i in 0..(self.elems.len()) {
            match key.cmp(self.elems[i].1.borrow()) {
                Less => {
                    if self.is_leaf() {
                        return (None, NeedsRebal(false));
                    }

                    let mut lt_k = self.elems[i].0.as_mut().unwrap();
                    let lt_k = Rc::make_mut(&mut lt_k);
                    let ret = lt_k.remove(key);
                    if let &(_, NeedsRebal(true)) = &ret {
                        return (ret.0, self.rebal(i));
                    } else {
                        return ret;
                    }
                }

                Equal => {
                    if self.is_leaf() {
                        let old_v = self.elems.remove(i).2;
                        return (
                            Some(old_v),
                            NeedsRebal(self.elems.len() < Self::MIN_OCCUPANCY),
                        );
                    }

                    let mut lt_k = self.elems[i].0.as_mut().unwrap();
                    let lt_k = Rc::make_mut(&mut lt_k);
                    let (k, v, needs_rebal) = lt_k.rm_greatest();
                    self.elems[i].1 = k;
                    let old_v = replace(&mut self.elems[i].2, v);
                    if let NeedsRebal(true) = needs_rebal {
                        return (Some(old_v), self.rebal(i));
                    } else {
                        return (Some(old_v), NeedsRebal(false));
                    }
                }

                Greater => (),
            }
        }

        if self.is_leaf() {
            return (None, NeedsRebal(false));
        }

        let mut gt_k = self.right.as_mut().unwrap();
        let gt_k = Rc::make_mut(&mut gt_k);
        let ret = gt_k.remove(key);
        if let &(_, NeedsRebal(true)) = &ret {
            return (ret.0, self.rebal(self.elems.len()));
        } else {
            return ret;
        }
    }
}

pub struct BTreeMap<K, V, const N: usize = 2> {
    len: usize,
    root: NodePtr<K, V, N>,
}

impl<K, V, const N: usize> BTreeMap<K, V, N> {
    /// Retrieves the value associated with the given key, if it is in the map.
    ///
    /// # Examples
    /// ```
    /// type BTreeMap<K, V> = fun_collections::BTreeMap<K, V, 2>;
    ///
    /// let mut m = BTreeMap::new();
    /// m.insert(0, 1);
    /// assert_eq!(m.get(&0), Some(&1));
    /// assert_eq!(m.get(&1), None);
    /// ```
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Ord,
    {
        self.root.as_ref().and_then(|n| n.get(key))
    }

    /// Associates 'val' with 'key' and returns the value previously associated
    /// with 'key', if it exists.
    ///
    /// # Examples
    /// ```
    /// type BTreeMap<K, V> = fun_collections::BTreeMap<K, V, 2>;
    ///
    /// let mut m = BTreeMap::new();
    /// assert_eq!(m.insert(0, 1), None);
    /// assert_eq!(m.insert(0, 0), Some(1));
    /// ```
    pub fn insert(&mut self, key: K, val: V) -> Option<V>
    where
        K: Clone + Ord,
        V: Clone,
    {
        if let Some(r) = self.root.as_mut() {
            match Rc::make_mut(r).insert(key, val) {
                InsertResult::Replaced(v) => Some(v),

                InsertResult::Split(s) => {
                    self.len += 1;
                    self.root = Some(Rc::new(Node {
                        elems: vec![s],
                        right: self.root.take(),
                    }));
                    None
                }

                InsertResult::Absorbed => {
                    self.len += 1;
                    None
                }
            }
        } else {
            self.len = 1;
            self.root = Some(Rc::new(Node {
                elems: vec![(None, key, val)],
                right: None,
            }));
            None
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn new() -> Self {
        Self { len: 0, root: None }
    }

    /// Removes and returns the value associated with key, if it exists.
    ///
    /// # Examples
    /// ```
    /// type BTreeMap<K, V> = fun_collections::BTreeMap<K, V, 2>;
    ///
    /// let mut m = BTreeMap::new();
    /// m.insert(0, 'a');
    /// m.insert(1, 'b');
    /// assert_eq!(m.remove(&0), Some('a'));
    /// assert_eq!(m.remove(&1), Some('b'));
    /// assert_eq!(m.remove(&1), None);
    /// assert_eq!(m.len(), 0);
    /// ```
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q> + Clone,
        V: Clone,
        Q: Ord,
    {
        if let Some(rc) = self.root.as_mut() {
            let n = Rc::make_mut(rc);
            let (old_v, needs_rebal) = n.remove(key);

            if old_v.is_some() {
                self.len -= 1;
            }

            if needs_rebal.0 && n.elems.len() == 0 {
                self.root = n.right.take();
            }

            old_v
        } else {
            return None;
        }
    }
}

#[cfg(test)]
fn chk_node_ptr<'a, K: Ord, V, const N: usize>(
    n: &'a NodePtr<K, V, N>,
    prev: Option<&'a K>,
) -> (usize, Option<&'a K>) {
    match n {
        Some(n) => {
            assert!(n.elems.len() >= N, "minimum occupancy violated");
            assert!(n.elems.len() <= 2 * N, "maximum occupancy violated");
            n.chk(prev)
        }

        None => (0, prev),
    }
}

#[cfg(test)]
impl<K: Ord, V, const N: usize> Node<K, V, N> {
    fn chk(&self, prev: Option<&K>) -> (usize, Option<&K>) {
        assert!(self.elems.len() > 0, "no entries");

        let (ht, prev) = chk_node_ptr(&self.elems[0].0, prev);
        prev.map(|k| assert!(k < &self.elems[0].1, "order violation"));
        let mut prev = Some(&self.elems[0].1);

        for entry in self.elems.iter().skip(1) {
            let curr = chk_node_ptr(&entry.0, prev);
            assert_eq!(ht, curr.0, "uneven branches");
            assert!(prev.unwrap() < &entry.1, "order violation");
            prev = Some(&entry.1);
        }

        let curr = chk_node_ptr(&self.right, prev);
        assert_eq!(ht, curr.0, "uneven branches");
        (ht + 1, curr.1)
    }
}

#[cfg(test)]
impl<K: Ord, V, const N: usize> BTreeMap<K, V, N> {
    fn chk(&self) {
        self.root.as_ref().map(|n| n.chk(None));
    }
}

#[cfg(not(test))]
impl<K: Ord, V, const N: usize> Node<K, V, N> {
    fn chk(&self, _: Option<&K>) {}
}

#[cfg(not(test))]
impl<K: Ord, V, const N: usize> BTreeMap<K, V, N> {
    fn chk(&self) {}
}

#[cfg(test)]
mod test {
    extern crate quickcheck;
    use quickcheck::quickcheck;

    type BTreeMap<K, V> = super::BTreeMap<K, V, 2>;
    type TestElems = Vec<(u8, u16)>;

    #[test]
    fn remove_test1() {
        let mut m = BTreeMap::new();
        m.insert(0, 'a');
        m.insert(1, 'b');
        assert_eq!(m.remove(&0), Some('a'));
        assert_eq!(m.remove(&1), Some('b'));
        assert_eq!(m.remove(&1), None);
        assert_eq!(m.len(), 0);
    }

    fn test_insert(elems: TestElems) -> () {
        let mut m1 = BTreeMap::new();
        let mut m2 = std::collections::HashMap::new();
        for (k, v) in elems {
            assert_eq!(m1.insert(k, v), m2.insert(k, v));
            assert_eq!(m1.len(), m2.len());
            m1.chk();
        }

        for (k, v) in m2.iter() {
            assert_eq!(m1.get(k), Some(v));
        }
    }

    fn test_remove(elems: TestElems) -> () {
        let mut m1 = BTreeMap::new();
        let mut m2 = std::collections::HashMap::new();
        for (k, v) in elems {
            if k < 128 {
                assert_eq!(m1.insert(k, v), m2.insert(k, v));
            } else {
                let k = k - 128;
                assert_eq!(m1.remove(&k), m2.remove(&k));
            }
            assert_eq!(m1.len(), m2.len());
            m1.chk();
        }

        for (k, v) in m2.iter() {
            assert_eq!(m1.get(k), Some(v));
        }
    }

    #[test]
    fn insert_regr1() {
        test_insert(vec![(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]);
    }

    #[test]
    fn remove_regr1() {
        let elems = vec![
            (82, 0),
            (83, 0),
            (0, 0),
            (5, 0),
            (84, 0),
            (1, 0),
            (6, 0),
            (86, 0),
            (87, 0),
            (7, 0),
            (8, 0),
            (88, 0),
            (2, 0),
            (9, 0),
            (85, 0),
            (81, 0),
            (3, 0),
            (4, 0),
            (209, 0),
        ];
        test_remove(elems);
    }

    quickcheck! {
        fn qc_insert(elems: TestElems) -> () {
            test_insert(elems);
        }

        fn qc_remove(elems: TestElems) -> () {
            test_remove(elems);
        }
    }
}
