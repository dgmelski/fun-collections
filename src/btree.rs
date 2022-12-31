use std::borrow::Borrow;
use std::cmp::Ordering::*;
use std::collections::VecDeque;
use std::iter::{ExactSizeIterator, FusedIterator};
use std::mem::replace;
use std::ops::{Bound, RangeBounds};
use std::sync::Arc;

use crate::{Entry, Map};

pub mod btree_set;

type NodePtr<K, V, const N: usize> = Arc<Node<K, V, N>>;

#[derive(Clone)]
struct Node<K, V, const N: usize> {
    elems: Vec<(K, V)>,
    kids: Vec<NodePtr<K, V, N>>,
}

impl<K, V, const N: usize> std::fmt::Debug for Node<K, V, N>
where
    K: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "Node({:?}..{:?})",
            self.elems.first().unwrap().0,
            self.elems.last().unwrap().0
        ))
    }
}

enum InsertResult<K, V, const N: usize> {
    Replaced(V),
    Split(Option<NodePtr<K, V, N>>, K, V),
    Absorbed,
}

struct IsUnderPop(bool);

const fn max_occupancy(min_occupancy: usize) -> usize {
    2 * min_occupancy + 1
}

impl<K, V, const N: usize> Node<K, V, N> {
    // minimum and maximum element counts for non-root nodes
    const MIN_OCCUPANCY: usize = N;
    const MAX_OCCUPANCY: usize = max_occupancy(N);

    fn child(&self, i: usize) -> Option<&Arc<Self>> {
        self.kids.get(i)
    }

    fn child_mut(&mut self, i: usize) -> Option<&mut Arc<Self>> {
        self.kids.get_mut(i)
    }

    fn key(&self, i: usize) -> &K {
        &self.elems[i].0
    }

    fn key_mut(&mut self, i: usize) -> &mut K {
        &mut self.elems[i].0
    }

    fn val(&self, i: usize) -> &V {
        &self.elems[i].1
    }

    fn val_mut(&mut self, i: usize) -> &mut V {
        &mut self.elems[i].1
    }

    fn len(&self) -> usize {
        self.elems.len()
    }

    fn get_key_value<Q>(&self, key: &Q) -> Option<&(K, V)>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        for i in 0..self.len() {
            match key.cmp(self.key(i).borrow()) {
                Less => return self.child(i)?.get_key_value(key),
                Equal => return Some(&self.elems[i]),
                Greater => (),
            }
        }

        self.kids.last()?.get_key_value(key)
    }

    fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut (K, V)>
    where
        K: Borrow<Q> + Clone,
        V: Clone,
        Q: Ord + ?Sized,
    {
        for i in 0..self.len() {
            match key.cmp(self.key(i).borrow()) {
                Less => {
                    let rc = self.child_mut(i)?;
                    let n = Arc::make_mut(rc);
                    return n.get_mut(key);
                }

                Equal => return Some(&mut self.elems[i]),
                Greater => (),
            }
        }

        let rc = self.kids.last_mut()?;
        let n = Arc::make_mut(rc);
        n.get_mut(key)
    }

    fn insert(&mut self, key: K, val: V) -> InsertResult<K, V, N>
    where
        K: Clone + Ord,
        V: Clone,
    {
        use InsertResult::*;

        let mut ub_x = 0;
        while ub_x < self.len() {
            match key.cmp(self.key(ub_x)) {
                Less => break,
                Equal => return Replaced(replace(self.val_mut(ub_x), val)),
                Greater => (),
            }

            ub_x += 1;
        }

        // Recurse to the appropriate child if it exists (ie, we're not a leaf).
        // If we are a leaf, pretend that we visited a child and it resulted in
        // needing to insert a new separator at this level.
        let res = match self.child_mut(ub_x) {
            Some(n) => Arc::make_mut(n).insert(key, val),
            None => Split(None, key, val),
        };

        // update for a node split at the next level down
        if let Split(child, k, v) = res {
            // TODO: split before insert to reduce memmove

            self.elems.insert(ub_x, (k, v));
            if let Some(rc) = child {
                self.kids.insert(ub_x, rc);
            }

            if self.elems.len() <= Self::MAX_OCCUPANCY {
                return Absorbed;
            }

            self.split()
        } else {
            // res is Replaced(v) or Absorbed
            res
        }
    }

    // split this overcrowded node
    fn split(&mut self) -> InsertResult<K, V, N> {
        assert!(self.len() > Self::MAX_OCCUPANCY);

        // take the top half from the existing node
        let mut other_elems = self.elems.split_off(Self::MIN_OCCUPANCY + 1);
        let mut other_kids = if self.kids.is_empty() {
            Vec::new()
        } else {
            self.kids.split_off(Self::MIN_OCCUPANCY + 1)
        };

        // swap the top half into the existing node
        std::mem::swap(&mut self.elems, &mut other_elems);
        std::mem::swap(&mut self.kids, &mut other_kids);

        // take the separator between the divided sides
        let (mid_k, mid_v) = other_elems.pop().unwrap();

        // make a node for the lhs
        let lhs = Some(Arc::new(Node {
            elems: other_elems,
            kids: other_kids,
        }));

        InsertResult::Split(lhs, mid_k, mid_v)
    }

    fn is_branch(&self) -> bool {
        !self.kids.is_empty()
    }

    fn is_leaf(&self) -> bool {
        self.kids.is_empty()
    }

    fn stitch(
        lf: InnerHalf<K, V, N>,
        rt: InnerHalf<K, V, N>,
    ) -> InnerHalf<K, V, N> {
        match (lf.0, rt.0) {
            (None, None) => (
                Some((
                    Arc::new(Self {
                        elems: vec![(lf.1, lf.2)],
                        kids: vec![],
                    }),
                    1,
                )),
                rt.1,
                rt.2,
            ),

            (Some((mut rc, mut ht)), None) => {
                assert!(rc.is_leaf());

                let n = Arc::get_mut(&mut rc).unwrap();
                n.elems.push((lf.1, lf.2));

                let rc = if n.len() > Self::MAX_OCCUPANCY {
                    let InsertResult::Split(lf, k, v) = n.split() else {
                        panic!("split should never fail");
                    };

                    ht += 1;

                    Arc::new(Self {
                        elems: vec![(k, v)],
                        kids: vec![lf.unwrap(), rc],
                    })
                } else {
                    rc
                };

                (Some((rc, ht)), rt.1, rt.2)
            }

            (None, Some((mut rc, mut ht))) => {
                assert!(rc.is_leaf());

                let n = Arc::get_mut(&mut rc).unwrap();
                n.elems.insert(0, (lf.1, lf.2));

                let rc = if n.len() > Self::MAX_OCCUPANCY {
                    let InsertResult::Split(lf, k, v) = n.split() else {
                        panic!("split should never fail");
                    };

                    ht += 1;

                    Arc::new(Self {
                        elems: vec![(k, v)],
                        kids: vec![lf.unwrap(), rc],
                    })
                } else {
                    rc
                };

                (Some((rc, ht)), rt.1, rt.2)
            }

            (Some((mut lf_rc, lf_ht)), Some((mut rt_rc, rt_ht))) => {
                assert!(rt_ht > 0);
                assert!(lf_ht >= rt_ht, "left should split before right");

                if lf_ht > rt_ht {
                    assert_eq!(lf_ht, rt_ht + 1, "left & right out-of-sync");

                    let lf_n = Arc::get_mut(&mut lf_rc).unwrap();
                    lf_n.elems.push((lf.1, lf.2));
                    lf_n.kids.push(rt_rc);

                    (Some((lf_rc, lf_ht)), rt.1, rt.2)
                } else if lf_rc.len() + 1 + rt_rc.len() < Self::MAX_OCCUPANCY {
                    let lf_n = Arc::get_mut(&mut lf_rc).unwrap();
                    let rt_n = Arc::get_mut(&mut rt_rc).unwrap();
                    lf_n.elems.push((lf.1, lf.2));
                    lf_n.elems.append(&mut rt_n.elems);
                    lf_n.kids.append(&mut rt_n.kids);

                    (Some((lf_rc, lf_ht)), rt.1, rt.2)
                } else {
                    assert!(lf_rc.len() >= Self::MIN_OCCUPANCY);
                    assert!(rt_rc.len() >= Self::MIN_OCCUPANCY);

                    (
                        Some((
                            Arc::new(Self {
                                elems: vec![(lf.1, lf.2)],
                                kids: vec![lf_rc, rt_rc],
                            }),
                            lf_ht + 1,
                        )),
                        rt.1,
                        rt.2,
                    )
                }
            }
        }
    }

    fn rot_lf(&mut self, idx: usize)
    where
        K: Clone,
        V: Clone,
    {
        assert!(self.is_branch(), "cannot rotate a leaf's children");

        // extract the new separator (k2 and v2) from child on right of idx
        let right = self.child_mut(idx + 1).unwrap();

        assert!(
            right.elems.len() > Self::MIN_OCCUPANCY,
            "rot_lf from an impovershed child"
        );

        let n = Arc::make_mut(right);

        let (k2, v2) = n.elems.remove(0);
        let k1_to_k2 = if n.kids.is_empty() {
            None
        } else {
            Some(n.kids.remove(0))
        };

        // replace (and take) the old separator (k1 and v1)
        let k1 = std::mem::replace(self.key_mut(idx), k2);
        let v1 = std::mem::replace(self.val_mut(idx), v2);

        // push the old separator to the end of left
        let left = self.child_mut(idx).unwrap();
        assert!(
            left.elems.len() < Self::MIN_OCCUPANCY,
            "rot_lf into a rich child"
        );

        let left = Arc::make_mut(left);
        left.elems.push((k1, v1));
        if let Some(k1_to_k2) = k1_to_k2 {
            left.kids.push(k1_to_k2);
        }
    }

    fn rot_rt(&mut self, idx: usize)
    where
        K: Clone,
        V: Clone,
    {
        assert!(self.is_branch(), "cannot rotate a leaf's children");

        // idx holds the current separator, k1.  Get the pieces that will rotate
        // in to replace k1 & v1.
        let left = self.child_mut(idx).unwrap();
        assert!(
            left.elems.len() > Self::MIN_OCCUPANCY,
            "rot_rt from impoverished child"
        );

        let left = Arc::make_mut(left);
        let (k0, v0) = left.elems.pop().unwrap();
        let k0_to_k1 = left.kids.pop();

        // move k0 and v0 into the pivot position
        let k1 = replace(self.key_mut(idx), k0);
        let v1 = replace(self.val_mut(idx), v0);

        // move k1 and v1 down and to the right of the pivot
        let right = self.child_mut(idx + 1).unwrap();
        let right = Arc::make_mut(right);
        right.elems.insert(0, (k1, v1));
        if let Some(k0_to_k1) = k0_to_k1 {
            right.kids.insert(0, k0_to_k1);
        }
    }

    // merge the subtree self.index[at].0 and the one to its right
    fn merge_kids(&mut self, at: usize)
    where
        K: Clone,
        V: Clone,
    {
        // take the left child and the separator key & val
        let (mid_k, mid_v) = self.elems.remove(at);
        let lhs_rc: NodePtr<K, V, N> = self.kids.remove(at);
        let mut lhs_n = match Arc::try_unwrap(lhs_rc) {
            Ok(n) => n,
            Err(rc) => (*rc).clone(),
        };

        // put the separator key & val into the lhs
        lhs_n.elems.push((mid_k, mid_v));

        // get a private copy of the rhs
        let rhs_rc = self.child_mut(at).unwrap();
        let rhs_ref = Arc::make_mut(rhs_rc);

        // We own & can take from lhs_n, but we want rhs's elements at the end.
        // Swap lhs & rhs vecs so we can use a cheaper append for merging.
        std::mem::swap(&mut lhs_n.elems, &mut rhs_ref.elems);
        rhs_ref.elems.extend(lhs_n.elems);

        std::mem::swap(&mut lhs_n.kids, &mut rhs_ref.kids);
        rhs_ref.kids.extend(lhs_n.kids);
    }

    // rebalances when the self.elems[at] is underpopulated
    fn rebal(&mut self, at: usize) -> IsUnderPop
    where
        K: Clone,
        V: Clone,
    {
        assert!(self.is_branch(), "cannot rebalance a leaf");

        if at > 0 {
            if self.kids[at - 1].elems.len() > Self::MIN_OCCUPANCY {
                self.rot_rt(at - 1);
            } else {
                self.merge_kids(at - 1);
            }
        } else if self.kids[at + 1].elems.len() > Self::MIN_OCCUPANCY {
            self.rot_lf(at);
        } else {
            self.merge_kids(at);
        }

        IsUnderPop(self.elems.len() < Self::MIN_OCCUPANCY)
    }

    fn pop_first(&mut self) -> Option<((K, V), IsUnderPop)>
    where
        K: Clone,
        V: Clone,
    {
        if let Some(rt) = self.child_mut(0) {
            // self is a branch; recurse to the rightmost child
            let rt = Arc::make_mut(rt);
            let ret = rt.pop_first();
            if let Some((kv, IsUnderPop(true))) = ret {
                Some((kv, self.rebal(0)))
            } else {
                ret
            }
        } else {
            // self is a leaf
            let kv = self.elems.remove(0);
            Some((kv, IsUnderPop(self.elems.len() < Self::MIN_OCCUPANCY)))
        }
    }

    fn pop_last(&mut self) -> Option<((K, V), IsUnderPop)>
    where
        K: Clone,
        V: Clone,
    {
        if let Some(rt) = self.kids.last_mut() {
            // self is a branch; recurse to the rightmost child
            let rt = Arc::make_mut(rt);
            let ret = rt.pop_last();
            if let Some((kv, IsUnderPop(true))) = ret {
                Some((kv, self.rebal(self.elems.len())))
            } else {
                ret
            }
        } else {
            // self is a leaf
            let kv = self.elems.pop().unwrap();
            Some((kv, IsUnderPop(self.elems.len() < Self::MIN_OCCUPANCY)))
        }
    }

    fn remove<Q>(&mut self, key: &Q) -> Option<((K, V), IsUnderPop)>
    where
        K: Borrow<Q> + Clone,
        V: Clone,
        Q: Ord + ?Sized,
    {
        for i in 0..(self.elems.len()) {
            match key.cmp(self.key(i).borrow()) {
                Less => {
                    let lt_k = self.child_mut(i)?;
                    let lt_k = Arc::make_mut(lt_k);
                    let (kv, IsUnderPop(is_under_pop)) = lt_k.remove(key)?;
                    return Some((
                        kv,
                        IsUnderPop(is_under_pop && self.rebal(i).0),
                    ));
                }

                Equal => {
                    if self.is_leaf() {
                        let old_kv = self.elems.remove(i);
                        return Some((
                            old_kv,
                            IsUnderPop(self.elems.len() < Self::MIN_OCCUPANCY),
                        ));
                    }

                    let lt_k = self.child_mut(i).unwrap();
                    let lt_k = Arc::make_mut(lt_k);
                    let (kv, is_under_pop) = lt_k.pop_last().unwrap();
                    let old_kv = replace(&mut self.elems[i], kv);
                    if is_under_pop.0 {
                        return Some((old_kv, self.rebal(i)));
                    } else {
                        return Some((old_kv, IsUnderPop(false)));
                    }
                }

                Greater => (),
            }
        }

        // greater than all in this node; try descending rightmost child
        let gt_k = self.kids.last_mut()?;
        let gt_k = Arc::make_mut(gt_k);
        let (kv, IsUnderPop(is_under_pop)) = gt_k.remove(key)?;
        Some((
            kv,
            IsUnderPop(is_under_pop && self.rebal(self.elems.len()).0),
        ))
    }
}

#[derive(Clone)]
pub struct BTreeMap<K, V, const N: usize = 5> {
    len: usize,
    root: Option<NodePtr<K, V, N>>,
}

impl<K, V, const N: usize> BTreeMap<K, V, N> {
    pub fn append(&mut self, other: &mut Self)
    where
        K: Clone + Ord,
        V: Clone,
    {
        self.extend(std::mem::take(other));
    }

    pub fn clear(&mut self) {
        self.len = 0;
        self.root = None;
    }

    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let mut curr = match self.root.as_ref() {
            None => return false,
            Some(rc) => rc.as_ref(),
        };

        loop {
            let mut i = 0;
            while i < curr.len() {
                match key.cmp(curr.key(i).borrow()) {
                    Less => break,
                    Equal => return true,
                    Greater => i += 1,
                }
            }

            match curr.child(i) {
                None => return false,
                Some(rc) => curr = rc.as_ref(),
            }
        }
    }

    /// Returns an Entry that simplifies some update operations.
    pub fn entry(&mut self, key: K) -> Entry<'_, Self>
    where
        K: Clone + Ord,
        V: Clone,
    {
        Entry { map: self, key }
    }

    /// Return an Entry for the least key in the map.
    pub fn first_entry(&mut self) -> Option<Entry<'_, Self>>
    where
        K: Clone + Ord,
        V: Clone,
    {
        let key = self.first_key_value()?.0.clone();
        Some(Entry { map: self, key })
    }

    pub fn first_key_value(&self) -> Option<(&K, &V)> {
        let mut curr = self.root.as_ref()?;
        while let Some(next) = curr.child(0) {
            curr = next;
        }
        Some((curr.key(0), curr.val(0)))
    }

    /// Retrieves the value associated with the given key, if there is one.
    ///
    /// # Examples
    /// ```
    /// use lazy_clone_collections::BTreeMap;
    ///
    /// let mut m = BTreeMap::new();
    /// m.insert(0, 1);
    /// assert_eq!(m.get(&0), Some(&1));
    /// assert_eq!(m.get(&1), None);
    /// ```
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.get_key_value(key).map(|e| e.1)
    }

    /// Retrieves the entry associated with the given key, if there is one.
    ///
    /// # Examples
    /// ```
    /// use lazy_clone_collections::BTreeMap;
    ///
    /// let mut m = BTreeMap::new();
    /// m.insert(0, 1);
    /// assert_eq!(m.get_key_value(&0), Some((&0, &1)));
    /// assert_eq!(m.get(&1), None);
    /// ```
    pub fn get_key_value<Q>(&self, key: &Q) -> Option<(&K, &V)>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.root
            .as_ref()?
            .get_key_value(key)
            .map(|(ref k, ref v)| (k, v))
    }

    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q> + Clone,
        V: Clone,
        Q: Ord + ?Sized,
    {
        let rc = self.root.as_mut()?;
        let n = Arc::make_mut(rc);
        n.get_mut(key).map(|e| &mut e.1)
    }

    /// Associates 'val' with 'key' and returns the value previously associated
    /// with 'key', if it exists.
    ///
    /// # Examples
    /// ```
    /// use lazy_clone_collections::BTreeMap;
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
            match Arc::make_mut(r).insert(key, val) {
                InsertResult::Replaced(v) => Some(v),

                InsertResult::Split(lhs, k, v) => {
                    self.len += 1;
                    self.root = Some(Arc::new(Node {
                        elems: vec![(k, v)],
                        kids: vec![lhs.unwrap(), self.root.take().unwrap()],
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
            self.root = Some(Arc::new(Node {
                elems: vec![(key, val)],
                kids: Vec::new(),
            }));
            None
        }
    }

    pub fn into_keys(
        self,
    ) -> impl DoubleEndedIterator<Item = K> + ExactSizeIterator + FusedIterator
    where
        K: Clone,
        V: Clone, // needed to clone shared nodes
    {
        // TODO: needlessly clones values from owned nodes
        self.into_iter().map(|e| e.0)
    }

    pub fn into_values(
        self,
    ) -> impl DoubleEndedIterator<Item = V> + ExactSizeIterator + FusedIterator
    where
        K: Clone, // needed to clone shared nodes
        V: Clone,
    {
        // TODO: needlessly clones keys from owned nodes
        self.into_iter().map(|e| e.1)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn iter(&self) -> Iter<'_, K, V, N> {
        let make_erg = make_node_iter;
        let iter = if let Some(arc) = self.root.as_ref() {
            let erg = make_erg(arc);
            InnerIter {
                work: VecDeque::from([erg]),
                len: self.len,
                make_node_iter: make_erg,
            }
        } else {
            InnerIter {
                work: VecDeque::new(),
                len: 0,
                make_node_iter: make_erg,
            }
        };

        Iter { iter }
    }

    pub fn iter_mut(&mut self) -> IterMut<'_, K, V, N>
    where
        K: Clone,
        V: Clone,
    {
        let make_erg = make_node_iter_mut;
        let iter = if let Some(arc) = self.root.as_mut() {
            let erg = make_erg(arc);
            InnerIter {
                work: VecDeque::from([erg]),
                len: self.len,
                make_node_iter: make_erg,
            }
        } else {
            InnerIter {
                work: VecDeque::new(),
                len: 0,
                make_node_iter: make_erg,
            }
        };

        IterMut { iter }
    }

    pub fn keys(
        &self,
    ) -> impl DoubleEndedIterator<Item = &K> + ExactSizeIterator + FusedIterator
    {
        self.iter().map(|e| e.0)
    }

    /// Return an Entry for the least key in the map.
    pub fn last_entry(&mut self) -> Option<Entry<'_, Self>>
    where
        K: Clone + Ord,
        V: Clone,
    {
        let key = self.last_key_value()?.0.clone();
        Some(Entry { map: self, key })
    }

    pub fn last_key_value(&self) -> Option<(&K, &V)> {
        let mut curr = self.root.as_ref()?;
        while let Some(next) = curr.kids.last() {
            curr = next;
        }
        curr.elems.last().map(|(ref k, ref v)| (k, v))
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn new() -> Self {
        Self { len: 0, root: None }
    }

    fn rm_and_rebal<RM>(&mut self, remover: RM) -> Option<(K, V)>
    where
        K: Clone,
        V: Clone,
        RM: FnOnce(&mut Node<K, V, N>) -> Option<((K, V), IsUnderPop)>,
    {
        let rc = self.root.as_mut()?;
        let n = Arc::make_mut(rc);
        let (old_kv, is_under_pop) = remover(n)?;

        self.len -= 1;

        if is_under_pop.0 && n.elems.is_empty() {
            assert!(n.kids.len() <= 1);
            self.root = n.kids.pop();
        }

        Some(old_kv)
    }

    pub fn pop_first(&mut self) -> Option<(K, V)>
    where
        K: Clone,
        V: Clone,
    {
        self.rm_and_rebal(|n| n.pop_first())
    }

    pub fn pop_last(&mut self) -> Option<(K, V)>
    where
        K: Clone,
        V: Clone,
    {
        self.rm_and_rebal(|n| n.pop_last())
    }

    pub fn range<Q, R>(&self, range: R) -> Range<'_, K, V, N>
    where
        Q: Ord + ?Sized,
        K: Borrow<Q>,
        R: RangeBounds<Q>,
    {
        Range::new(self, range)
    }

    pub fn range_mut<Q, R>(&mut self, range: R) -> RangeMut<'_, K, V, N>
    where
        Q: Ord + ?Sized,
        K: Borrow<Q> + Clone,
        R: RangeBounds<Q>,
        V: Clone,
    {
        RangeMut::new(self, range)
    }

    /// Removes and returns the value associated with key, if it exists.
    ///
    /// # Examples
    /// ```
    /// use lazy_clone_collections::BTreeMap;
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
        Q: Ord + ?Sized,
    {
        // avoid unnecessary cloning
        if !self.contains_key(key) {
            return None;
        }

        self.rm_and_rebal(|n| n.remove(key)).map(|e| e.1)
    }

    pub fn retain<F>(&mut self, mut f: F)
    where
        K: Clone + Ord,
        V: Clone,
        F: FnMut(&K, &mut V) -> bool,
    {
        // TODO: this can probably be more efficient. For example, we know the
        // keys are sorted, so we can probably build the map more efficiently.
        // (We do not know how many keys there are, however.)
        *self = std::mem::take(self)
            .into_iter()
            .filter_map(
                |(k, mut v)| {
                    if f(&k, &mut v) {
                        Some((k, v))
                    } else {
                        None
                    }
                },
            )
            .collect();
    }

    pub fn split_off<Q>(&mut self, key: &Q) -> Self
    where
        Q: Ord + ?Sized,
        K: Borrow<Q> + Clone + Ord,
        V: Clone,
    {
        // TODO: this can probably be more efficient
        let (a, b) = std::mem::take(self)
            .into_iter()
            .partition(|(k, _)| key > k.borrow());

        *self = a;
        b
    }

    // TOOD: try_insert()

    pub fn values(
        &self,
    ) -> impl DoubleEndedIterator<Item = &V> + ExactSizeIterator + FusedIterator
    {
        self.iter().map(|e| e.1)
    }

    pub fn values_mut(
        &mut self,
    ) -> impl DoubleEndedIterator<Item = &mut V> + ExactSizeIterator + FusedIterator
    where
        K: Clone,
        V: Clone,
    {
        self.iter_mut().map(|e| e.1)
    }
}

impl<K, V, const N: usize> std::fmt::Debug for BTreeMap<K, V, N>
where
    K: std::fmt::Debug,
    V: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

// we implement our own default to avoid the Default constraints on K and V
impl<K, V, const N: usize> Default for BTreeMap<K, V, N> {
    fn default() -> Self {
        Self { len: 0, root: None }
    }
}

impl<K, V, const N: usize> Extend<(K, V)> for BTreeMap<K, V, N>
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

impl<K: Eq, V: Eq, const N: usize> Eq for BTreeMap<K, V, N> {}

impl<K, V, const N: usize, const M: usize> From<[(K, V); M]>
    for BTreeMap<K, V, N>
where
    K: Clone + Ord,
    V: Clone,
{
    fn from(vs: [(K, V); M]) -> Self {
        BTreeMap::from_iter(vs.into_iter())
    }
}

impl<K, V, const N: usize> FromIterator<(K, V)> for BTreeMap<K, V, N>
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

impl<K, V, const N: usize> IntoIterator for BTreeMap<K, V, N>
where
    K: Clone,
    V: Clone,
{
    type Item = (K, V);
    type IntoIter = IntoIter<K, V, N>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter::new(self)
    }
}

impl<'a, K, V, const N: usize> IntoIterator for &'a BTreeMap<K, V, N> {
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V, N>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, K, V, const N: usize> IntoIterator for &'a mut BTreeMap<K, V, N>
where
    K: Clone,
    V: Clone,
{
    type Item = (&'a K, &'a mut V);
    type IntoIter = IterMut<'a, K, V, N>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

type InnerHalf<K, V, const N: usize> =
    (Option<(Arc<Node<K, V, N>>, usize)>, K, V);

pub struct Half<K, V, const N: usize> {
    h: InnerHalf<K, V, N>,
}

impl<K, V, const N: usize> Map for BTreeMap<K, V, N> {
    type Key = K;
    type Value = V;
    type Half = Half<K, V, N>;

    fn contains_key_<Q>(&mut self, key: &Q) -> bool
    where
        Self::Key: std::borrow::Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.contains_key(key)
    }

    fn get_mut_<Q>(&mut self, k: &Q) -> Option<&mut V>
    where
        K: Borrow<Q> + Clone,
        V: Clone,
        Q: Ord + ?Sized,
    {
        self.get_mut(k)
    }

    fn insert_(&mut self, key: K, val: V) -> Option<V>
    where
        K: Clone + Ord,
        V: Clone,
    {
        self.insert(key, val)
    }

    fn make_half(key: Self::Key, value: Self::Value) -> Self::Half {
        Self::Half {
            h: (None, key, value),
        }
    }

    fn make_whole(h: Self::Half, mut len: usize) -> Self
    where
        K: Clone + Ord,
        V: Clone,
    {
        let (n, k, v) = h.h;
        let n = n.map(|x| x.0); // discard height

        // create a map without the final kv
        len = len.saturating_sub(1);
        let mut m = Self { len, root: n };

        #[cfg(test)]
        m.chk();

        m.insert(k, v);
        m
    }

    fn stitch(lf: Self::Half, rt: Self::Half) -> Self::Half
    where
        Self::Key: Clone + Ord,
        Self::Value: Clone,
    {
        assert!(lf.h.1 < rt.h.1);
        Self::Half {
            h: Node::stitch(lf.h, rt.h),
        }
    }
}

impl<K, V, const N: usize> PartialEq for BTreeMap<K, V, N>
where
    K: PartialEq,
    V: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len()
            && self.iter().zip(other.iter()).all(|(x, y)| x == y)
    }
}

impl<K, V, const N: usize> PartialOrd for BTreeMap<K, V, N>
where
    K: PartialOrd,
    V: PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.iter().partial_cmp(other.iter())
    }
}

impl<K, V, const N: usize> Ord for BTreeMap<K, V, N>
where
    K: Ord,
    V: Ord,
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.iter().cmp(other.iter())
    }
}

// *************
//   Iterators
// *************

// Our iteration strategy uses a VecDeque of node iterators.  The NodeIter trait
// describes what we need to iterate an individual node.
trait NodeIter {
    type ElemItem;
    type ChildItem;

    fn is_empty(&self) -> bool;

    fn next_elem(&mut self) -> Option<Self::ElemItem>;
    fn next_back_elem(&mut self) -> Option<Self::ElemItem>;

    fn next_kid(&mut self) -> Option<Self::ChildItem>;
    fn next_back_kid(&mut self) -> Option<Self::ChildItem>;

    fn needs_lf_des(&self) -> bool;
    fn needs_rt_des(&self) -> bool;
}

// BorrowedNodeIter is used to implement Iter and IterMut where we'll have
// borrowed refs and borrowed mutable refs, respectively, for the iterated node.
// 'I' and 'J' will be Iterators of the appropriate types over the elems and
// kids of a node.
#[derive(Debug)]
struct BorrowedNodeIter<I, J> {
    elems: I,
    kids: J,
    needs_lf_des: bool,
    needs_rt_des: bool,
}

impl<I, J> NodeIter for BorrowedNodeIter<I, J>
where
    I: DoubleEndedIterator + ExactSizeIterator,
    J: DoubleEndedIterator + ExactSizeIterator,
{
    type ElemItem = I::Item;
    type ChildItem = J::Item;

    fn is_empty(&self) -> bool {
        self.elems.len() == 0 && self.kids.len() == 0
    }

    fn next_elem(&mut self) -> Option<I::Item>
    where
        I: Iterator,
        J: ExactSizeIterator,
    {
        assert!(!self.needs_lf_des);
        self.needs_lf_des = self.kids.len() != 0;
        self.elems.next()
    }

    fn next_back_elem(&mut self) -> Option<I::Item>
    where
        I: DoubleEndedIterator,
        J: ExactSizeIterator,
    {
        assert!(!self.needs_rt_des);
        self.needs_rt_des = self.kids.len() != 0;
        self.elems.next_back()
    }

    fn next_kid(&mut self) -> Option<J::Item>
    where
        J: ExactSizeIterator,
    {
        assert!(self.kids.len() == 0 || self.needs_lf_des);
        self.needs_lf_des = false;
        self.kids.next()
    }

    fn next_back_kid(&mut self) -> Option<J::Item>
    where
        J: DoubleEndedIterator + ExactSizeIterator,
    {
        assert!(self.kids.len() == 0 || self.needs_rt_des);
        self.needs_rt_des = false;
        self.kids.next_back()
    }

    fn needs_lf_des(&self) -> bool {
        self.needs_lf_des
    }

    fn needs_rt_des(&self) -> bool {
        self.needs_rt_des
    }
}

impl<I, J> BorrowedNodeIter<I, J> {
    fn new(elems: I, kids: J) -> Self
    where
        J: ExactSizeIterator,
    {
        let needs_descent = kids.len() != 0;
        BorrowedNodeIter {
            elems,
            kids,
            needs_lf_des: needs_descent,
            needs_rt_des: needs_descent,
        }
    }
}

// InnerIter is the shared implementation for /all/ of the iterators.  They are
// all wrappers around InnerIter, which they specialize in two ways: first, by
// providing different implementations of NodeIter and second by providing a
// function to create NodeIter's.
#[derive(Debug)]
struct InnerIter<I: NodeIter> {
    work: std::collections::VecDeque<I>,
    len: usize,
    make_node_iter: fn(I::ChildItem) -> I,
}

impl<I: NodeIter> Iterator for InnerIter<I> {
    type Item = I::ElemItem;

    fn next(&mut self) -> Option<Self::Item> {
        // 'erg' as in a unit of work, that being the node to iterate
        let mut erg = self.work.front_mut()?;

        if erg.needs_lf_des() {
            while let Some(kid) = erg.next_kid() {
                if erg.is_empty() {
                    self.work.pop_front();
                }
                self.work.push_front((self.make_node_iter)(kid));
                erg = self.work.front_mut().unwrap();
            }
        }

        let ret = erg.next_elem();

        assert!(ret.is_some());
        self.len -= 1;

        if erg.is_empty() {
            self.work.pop_front();
        }

        ret
    }
}

impl<I: NodeIter> DoubleEndedIterator for InnerIter<I> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let mut erg = self.work.back_mut()?;

        if erg.needs_rt_des() {
            while let Some(kid) = erg.next_back_kid() {
                if erg.is_empty() {
                    self.work.pop_back();
                }
                self.work.push_back((self.make_node_iter)(kid));
                erg = self.work.back_mut().unwrap();
            }
        }

        let ret = erg.next_back_elem();

        assert!(ret.is_some());
        self.len -= 1;

        if erg.is_empty() {
            self.work.pop_back();
        }

        ret
    }
}

use std::slice::Iter as SliceIter;

type InnerNodeIter<'a, K, V, const N: usize> =
    BorrowedNodeIter<SliceIter<'a, (K, V)>, SliceIter<'a, Arc<Node<K, V, N>>>>;

#[derive(Debug)]
pub struct Iter<'a, K, V, const N: usize> {
    iter: InnerIter<InnerNodeIter<'a, K, V, N>>,
}

fn make_node_iter<K, V, const N: usize>(
    n: &Arc<Node<K, V, N>>,
) -> InnerNodeIter<'_, K, V, N> {
    InnerNodeIter::new(n.elems.iter(), n.kids.iter())
}

impl<'a, K, V, const N: usize> ExactSizeIterator for Iter<'a, K, V, N> {
    fn len(&self) -> usize {
        self.iter.len
    }
}

impl<'a, K, V, const N: usize> FusedIterator for Iter<'a, K, V, N> {}

impl<'a, K, V, const N: usize> Iterator for Iter<'a, K, V, N> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(ref k, ref v)| (k, v))
    }
}

impl<'a, K, V, const N: usize> DoubleEndedIterator for Iter<'a, K, V, N> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back().map(|(ref k, ref v)| (k, v))
    }
}

use std::slice::IterMut as SliceIterMut;

#[derive(Debug)]
pub struct IterMut<'a, K, V, const N: usize>
where
    K: Clone,
    V: Clone,
{
    #[allow(clippy::type_complexity)]
    iter: InnerIter<
        BorrowedNodeIter<
            SliceIterMut<'a, (K, V)>,
            SliceIterMut<'a, Arc<Node<K, V, N>>>,
        >,
    >,
}

type NodeIterMut<'a, K, V, const N: usize> = BorrowedNodeIter<
    SliceIterMut<'a, (K, V)>,
    SliceIterMut<'a, Arc<Node<K, V, N>>>,
>;

fn make_node_iter_mut<K, V, const N: usize>(
    n: &mut Arc<Node<K, V, N>>,
) -> NodeIterMut<'_, K, V, N>
where
    K: Clone,
    V: Clone,
{
    let n = Arc::make_mut(n);
    NodeIterMut::new(n.elems.iter_mut(), n.kids.iter_mut())
}

impl<'a, K, V, const N: usize> ExactSizeIterator for IterMut<'a, K, V, N>
where
    K: Clone,
    V: Clone,
{
    fn len(&self) -> usize {
        self.iter.len
    }
}

impl<'a, K, V, const N: usize> FusedIterator for IterMut<'a, K, V, N>
where
    K: Clone,
    V: Clone,
{
}

impl<'a, K, V, const N: usize> Iterator for IterMut<'a, K, V, N>
where
    K: Clone,
    V: Clone,
{
    type Item = (&'a K, &'a mut V);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(ref k, ref mut v)| (k, v))
    }
}

impl<'a, K, V, const N: usize> DoubleEndedIterator for IterMut<'a, K, V, N>
where
    K: Clone,
    V: Clone,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back().map(|(ref k, ref mut v)| (k, v))
    }
}

// *** IntoIter ***

#[derive(Debug)]
struct ArcNodeIntoIter<K, V, const N: usize> {
    // It would be nice to use std:slice::Iter's for elems & kids, but it is
    // challenging to establish a reasonable lifetime for the iterator.  As we
    // destruct the owned portions of the tree, we drop our references to the
    // shared tree portions.  If a parallel thread were to drop the last Arc,
    // the shared node would disappear.  To prevent this, the iterator keeps its
    // own Arc to the node.  To use slice Iters, we'd need to promise they don't
    // outlive the Arc...
    n: Arc<Node<K, V, N>>,
    lb_elems: usize,
    ub_elems: usize,
    lb_kids: usize,
    ub_kids: usize,
    needs_lf_des: bool, // Needs Left Descent
    needs_rt_des: bool, // Needs Right Descent
}

#[derive(Debug)]
struct OwnedNodeIntoIter<K, V, const N: usize> {
    elems: std::vec::IntoIter<(K, V)>,
    kids: std::vec::IntoIter<Arc<Node<K, V, N>>>,
    needs_lf_des: bool, // Needs Left Descent
    needs_rt_des: bool, // Needs Right Descent
}

#[derive(Debug)]
enum IntoNodeIter<K, V, const N: usize> {
    Arcked(ArcNodeIntoIter<K, V, N>),
    Owned(OwnedNodeIntoIter<K, V, N>),
}

impl<K, V, const N: usize> NodeIter for IntoNodeIter<K, V, N>
where
    K: Clone,
    V: Clone,
{
    type ElemItem = (K, V);
    type ChildItem = Result<Node<K, V, N>, Arc<Node<K, V, N>>>;

    fn is_empty(&self) -> bool {
        match self {
            IntoNodeIter::Arcked(b) => {
                b.lb_elems >= b.ub_elems && b.lb_kids >= b.ub_kids
            }

            IntoNodeIter::Owned(n) => n.elems.len() == 0 && n.kids.len() == 0,
        }
    }

    fn needs_lf_des(&self) -> bool {
        match self {
            IntoNodeIter::Arcked(b) => b.needs_lf_des,
            IntoNodeIter::Owned(n) => n.needs_lf_des,
        }
    }

    fn needs_rt_des(&self) -> bool {
        match self {
            IntoNodeIter::Arcked(b) => b.needs_rt_des,
            IntoNodeIter::Owned(n) => n.needs_rt_des,
        }
    }

    fn next_elem(&mut self) -> Option<Self::ElemItem> {
        match self {
            IntoNodeIter::Arcked(b) => {
                assert!(!b.needs_lf_des);
                b.needs_lf_des = b.lb_kids < b.ub_kids;
                if b.lb_elems < b.ub_elems {
                    b.lb_elems += 1;
                    Some(b.n.elems[b.lb_elems - 1].clone())
                } else {
                    None
                }
            }

            IntoNodeIter::Owned(n) => {
                assert!(!n.needs_lf_des);
                n.needs_lf_des = n.kids.len() > 0;
                n.elems.next()
            }
        }
    }

    fn next_back_elem(&mut self) -> Option<Self::ElemItem> {
        match self {
            IntoNodeIter::Arcked(b) => {
                assert!(!b.needs_rt_des);
                b.needs_rt_des = b.lb_kids < b.ub_kids;
                if b.lb_elems < b.ub_elems {
                    b.ub_elems -= 1;
                    Some(b.n.elems[b.ub_elems].clone())
                } else {
                    None
                }
            }

            IntoNodeIter::Owned(n) => {
                assert!(!n.needs_rt_des);
                n.needs_rt_des = n.kids.len() > 0;
                n.elems.next_back()
            }
        }
    }

    fn next_kid(&mut self) -> Option<Self::ChildItem> {
        match self {
            IntoNodeIter::Arcked(b) => {
                assert!(b.needs_lf_des || b.lb_kids == b.ub_kids);
                b.needs_lf_des = false;
                if b.lb_kids < b.ub_kids {
                    b.lb_kids += 1;
                    Some(Err(b.n.kids[b.lb_kids - 1].clone()))
                } else {
                    None
                }
            }

            IntoNodeIter::Owned(n) => {
                assert!(n.needs_lf_des || n.kids.len() == 0);
                n.needs_lf_des = false;
                n.kids.next().map(Arc::try_unwrap)
            }
        }
    }

    fn next_back_kid(&mut self) -> Option<Self::ChildItem> {
        match self {
            IntoNodeIter::Arcked(b) => {
                assert!(b.needs_rt_des || b.lb_kids == b.ub_kids);
                b.needs_rt_des = false;
                if b.lb_kids < b.ub_kids {
                    b.ub_kids -= 1;
                    Some(Err(b.n.kids[b.ub_kids].clone()))
                } else {
                    None
                }
            }

            IntoNodeIter::Owned(n) => {
                assert!(n.needs_rt_des || n.kids.len() == 0);
                n.needs_rt_des = false;
                n.kids.next_back().map(Arc::try_unwrap)
            }
        }
    }
}

#[derive(Debug)]
pub struct IntoIter<K, V, const N: usize>
where
    K: Clone,
    V: Clone,
{
    iter: InnerIter<IntoNodeIter<K, V, N>>,
}

fn make_into_node_iter<K, V, const N: usize>(
    n: Result<Node<K, V, N>, Arc<Node<K, V, N>>>,
) -> IntoNodeIter<K, V, N>
where
    K: Clone,
    V: Clone,
{
    match n {
        Ok(n) => {
            let is_branch = n.is_branch();
            IntoNodeIter::Owned(OwnedNodeIntoIter {
                elems: n.elems.into_iter(),
                kids: n.kids.into_iter(),
                needs_lf_des: is_branch,
                needs_rt_des: is_branch,
            })
        }

        Err(n) => {
            let is_branch = n.is_branch();
            let cnt_elems = n.elems.len();
            let cnt_kids = n.kids.len();
            IntoNodeIter::Arcked(ArcNodeIntoIter {
                n,
                lb_elems: 0,
                ub_elems: cnt_elems,
                lb_kids: 0,
                ub_kids: cnt_kids,
                needs_lf_des: is_branch,
                needs_rt_des: is_branch,
            })
        }
    }
}

impl<K, V, const N: usize> IntoIter<K, V, N>
where
    K: Clone,
    V: Clone,
{
    fn new(m: BTreeMap<K, V, N>) -> Self {
        if let Some(n) = m.root {
            IntoIter {
                iter: InnerIter {
                    work: VecDeque::from([make_into_node_iter(
                        Arc::try_unwrap(n),
                    )]),
                    len: m.len,
                    make_node_iter: make_into_node_iter,
                },
            }
        } else {
            IntoIter {
                iter: InnerIter {
                    work: VecDeque::new(),
                    len: 0,
                    make_node_iter: make_into_node_iter,
                },
            }
        }
    }
}

impl<K, V, const N: usize> ExactSizeIterator for IntoIter<K, V, N>
where
    K: Clone,
    V: Clone,
{
    fn len(&self) -> usize {
        self.iter.len
    }
}

impl<K, V, const N: usize> FusedIterator for IntoIter<K, V, N>
where
    K: Clone,
    V: Clone,
{
}

impl<K, V, const N: usize> Iterator for IntoIter<K, V, N>
where
    K: Clone,
    V: Clone,
{
    type Item = (K, V);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

impl<K, V, const N: usize> DoubleEndedIterator for IntoIter<K, V, N>
where
    K: Clone,
    V: Clone,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back()
    }
}

// *** Range Iterator ***
struct IsIn(bool);

enum BdOrd {
    Unbounded,
    Less,
    Equal(IsIn),
    Greater,
}

fn cmp_bd<Q: Ord + ?Sized>(bd: Bound<&Q>, k: &Q) -> BdOrd {
    match bd {
        Bound::Unbounded => BdOrd::Unbounded,
        Bound::Excluded(bd_k) => match bd_k.cmp(k) {
            Less => BdOrd::Less,
            Equal => BdOrd::Equal(IsIn(false)),
            Greater => BdOrd::Greater,
        },
        Bound::Included(bd_k) => match bd_k.cmp(k) {
            Less => BdOrd::Less,
            Equal => BdOrd::Equal(IsIn(true)),
            Greater => BdOrd::Greater,
        },
    }
}

#[derive(Debug, Default)]
struct RangeEdges {
    lb_elems: usize,
    lb_kids: usize,
    left_needs_pruning: bool,
    ub_elems: usize,
    ub_kids: usize,
    right_needs_pruning: bool,
}

fn find_range_edges<K, V, const N: usize, Q>(
    n: &Arc<Node<K, V, N>>,
    lb: Bound<&Q>,
    ub: Bound<&Q>,
) -> RangeEdges
where
    Q: Ord + ?Sized,
    K: Borrow<Q>,
{
    // assume range is past n's greatest key
    let mut res = RangeEdges {
        lb_elems: n.elems.len(),
        lb_kids: n.elems.len(), // explore the last kid
        left_needs_pruning: true,
        ub_elems: n.elems.len(),
        ub_kids: n.elems.len() + 1, // use elems' len b/c kids might be empty
        right_needs_pruning: true,
    };

    // find the left edge
    let mut elems = n.elems.iter().enumerate().peekable();
    while let Some((i, (k, _))) = elems.peek() {
        match cmp_bd(lb, k.borrow()) {
            BdOrd::Unbounded => {
                res.lb_elems = 0;
                res.lb_kids = 0;
                res.left_needs_pruning = false;
                break;
            }

            BdOrd::Less => {
                res.lb_elems = *i;
                res.lb_kids = *i;
                res.left_needs_pruning = true;
                break;
            }

            BdOrd::Equal(IsIn(is_in)) => {
                res.lb_elems = i + (!is_in as usize);
                res.lb_kids = i + 1;
                res.left_needs_pruning = false;
                break;
            }

            BdOrd::Greater => {
                elems.next();
            }
        }
    }

    // find the right edge
    for (i, (k, _)) in elems {
        match cmp_bd(ub, k.borrow()) {
            BdOrd::Unbounded => {
                res.ub_elems = n.elems.len();
                res.ub_kids = n.elems.len() + 1;
                res.right_needs_pruning = false;
                break;
            }

            BdOrd::Less => {
                res.ub_elems = i;
                res.ub_kids = i + 1;
                res.right_needs_pruning = true;
                break;
            }

            BdOrd::Equal(IsIn(is_in)) => {
                res.ub_elems = i + (is_in as usize);
                res.ub_kids = i + 1;
                res.right_needs_pruning = false;
                break;
            }

            BdOrd::Greater => (),
        }
    }

    res
}

fn make_worklist<'a, K, V, const N: usize, Q>(
    n: Option<&'a Arc<Node<K, V, N>>>,
    work: &mut VecDeque<InnerNodeIter<'a, K, V, N>>,
    mut lb: Bound<&Q>,
    ub: Bound<&Q>,
) where
    Q: Ord + ?Sized,
    K: Borrow<Q>,
{
    let Some(n) = n else { return };
    let edges = find_range_edges(n, lb, ub);
    let elems = n.elems[edges.lb_elems..edges.ub_elems].iter();
    let mut kids = n
        .kids
        .get(edges.lb_kids..edges.ub_kids)
        .unwrap_or_default()
        .iter();

    if edges.left_needs_pruning {
        // We only need to use lb & ub together when there was only a
        // single viable child in n.  Otherwise, when we prune the left, it's
        // more efficient to use an ub of Unbounded and vice-versa on the right.
        let ub = if kids.len() == 1 && edges.right_needs_pruning {
            // assert!(edges.right_needs_pruning);
            assert_eq!(elems.len(), 0);
            ub
        } else {
            Bound::Unbounded
        };

        // uses our sub-scoped version of ub
        make_worklist(kids.next(), work, lb, ub);

        lb = Bound::Unbounded;
    }

    let to_prune = if edges.right_needs_pruning {
        kids.next_back()
    } else {
        None
    };

    if elems.len() > 0 || kids.len() > 0 {
        work.push_back(InnerNodeIter {
            elems,
            kids,
            needs_lf_des: !edges.left_needs_pruning
                && edges.lb_elems == edges.lb_kids,
            needs_rt_des: !edges.right_needs_pruning
                && edges.ub_elems < edges.ub_kids,
        });
    }

    make_worklist(to_prune, work, lb, ub);
}

fn make_mut_worklist<'a, K, V, const N: usize, Q>(
    n: Option<&'a mut Arc<Node<K, V, N>>>,
    work: &mut VecDeque<NodeIterMut<'a, K, V, N>>,
    mut lb: Bound<&Q>,
    ub: Bound<&Q>,
) where
    Q: Ord + ?Sized,
    K: Borrow<Q> + Clone,
    V: Clone,
{
    let Some(n) = n else { return };
    let edges = find_range_edges(n, lb, ub); // 1
    let n = Arc::make_mut(n); // 2
    let elems = n.elems[edges.lb_elems..edges.ub_elems].iter_mut(); // 3
    let mut kids = n // 4
        .kids
        .get_mut(edges.lb_kids..edges.ub_kids)
        .unwrap_or_default()
        .iter_mut();

    if edges.left_needs_pruning {
        // We only need to keep use lb & ub together when there was only a
        // single viable child in n.  Otherwise, when we prune the left, it's
        // more efficient to use an ub of Unbounded and vice-versa on the right.
        let ub = if kids.len() == 1 && edges.right_needs_pruning {
            // assert!(edges.right_needs_pruning);
            assert_eq!(elems.len(), 0);
            ub
        } else {
            Bound::Unbounded
        };

        // uses our sub-scoped version of ub
        make_mut_worklist(kids.next(), work, lb, ub);

        lb = Bound::Unbounded;
    }

    let to_prune = if edges.right_needs_pruning {
        kids.next_back()
    } else {
        None
    };

    if elems.len() > 0 || kids.len() > 0 {
        work.push_back(NodeIterMut {
            elems,
            kids,
            needs_lf_des: !edges.left_needs_pruning
                && edges.lb_elems == edges.lb_kids,
            needs_rt_des: !edges.right_needs_pruning
                && edges.ub_elems < edges.ub_kids,
        });
    }

    make_mut_worklist(to_prune, work, lb, ub);
}

pub struct Range<'a, K, V, const N: usize> {
    iter: InnerIter<InnerNodeIter<'a, K, V, N>>,
}

impl<'a, K, V, const N: usize> Range<'a, K, V, N> {
    fn new<Q, R>(m: &'a BTreeMap<K, V, N>, range: R) -> Self
    where
        Q: Ord + ?Sized,
        K: Borrow<Q>,
        R: RangeBounds<Q>,
    {
        use Bound::*;
        match (range.start_bound(), range.end_bound()) {
            (Excluded(k1) | Included(k1), Excluded(k2) | Included(k2))
                if k2 < k1 =>
            {
                panic!("bad range")
            }

            _ => (),
        }

        let mut work = VecDeque::new();
        make_worklist(
            m.root.as_ref(),
            &mut work,
            range.start_bound(),
            range.end_bound(),
        );

        Self {
            iter: InnerIter {
                work,
                len: m.len(), // only an estimate...
                make_node_iter,
            },
        }
    }
}

impl<'a, K, V, const N: usize> std::fmt::Debug for Range<'a, K, V, N>
where
    K: std::fmt::Debug,
    V: std::fmt::Debug,
{
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> Result<(), std::fmt::Error> {
        f.debug_struct("lazy_clone_collections::btree::Range")
            .field("work", &self.iter.work)
            .finish()
    }
}

impl<'a, K, V, const N: usize> FusedIterator for Range<'a, K, V, N> {}

impl<'a, K, V, const N: usize> Iterator for Range<'a, K, V, N> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(ref k, ref v)| (k, v))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let lb = self
            .iter
            .work
            .iter()
            .map(|erg| erg.elems.len() + N * erg.kids.len())
            .sum();
        (lb, Some(self.iter.len))
    }
}

impl<'a, K, V, const N: usize> DoubleEndedIterator for Range<'a, K, V, N> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back().map(|(ref k, ref v)| (k, v))
    }
}

pub struct RangeMut<'a, K, V, const N: usize> {
    iter: InnerIter<NodeIterMut<'a, K, V, N>>,
}

impl<'a, K, V, const N: usize> RangeMut<'a, K, V, N> {
    fn new<Q, R>(m: &'a mut BTreeMap<K, V, N>, range: R) -> Self
    where
        Q: Ord + ?Sized,
        K: Borrow<Q> + Clone,
        R: RangeBounds<Q>,
        V: Clone,
    {
        use Bound::*;
        match (range.start_bound(), range.end_bound()) {
            (Excluded(k1) | Included(k1), Excluded(k2) | Included(k2))
                if k2 < k1 =>
            {
                panic!("bad range")
            }

            _ => (),
        }

        let len = m.len();
        let mut work = VecDeque::new();
        make_mut_worklist(
            m.root.as_mut(),
            &mut work,
            range.start_bound(),
            range.end_bound(),
        );

        Self {
            iter: InnerIter {
                work,
                len, // Only an estimate ...
                make_node_iter: make_node_iter_mut,
            },
        }
    }
}

impl<'a, K, V, const N: usize> std::fmt::Debug for RangeMut<'a, K, V, N>
where
    K: std::fmt::Debug,
    V: std::fmt::Debug,
{
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> Result<(), std::fmt::Error> {
        f.debug_struct("lazy_clone_collections::btree::Range")
            .field("work", &self.iter.work)
            .finish()
    }
}

impl<'a, K, V, const N: usize> FusedIterator for RangeMut<'a, K, V, N> {}

impl<'a, K, V, const N: usize> Iterator for RangeMut<'a, K, V, N> {
    type Item = (&'a K, &'a mut V);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(ref k, ref mut v)| (k, v))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let lb = self
            .iter
            .work
            .iter()
            .map(|erg| erg.elems.len() + N * erg.kids.len())
            .sum();
        (lb, Some(self.iter.len))
    }
}

impl<'a, K, V, const N: usize> DoubleEndedIterator for RangeMut<'a, K, V, N> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back().map(|(ref k, ref mut v)| (k, v))
    }
}

#[cfg(test)]
fn chk_node_ptr<'a, K: Ord, V, const N: usize>(
    n: Option<&'a Arc<Node<K, V, N>>>,
    prev: Option<&'a K>,
) -> (usize, Option<&'a K>) {
    match n {
        Some(n) => {
            assert!(n.elems.len() >= N, "minimum occupancy violated");
            assert!(
                n.elems.len() <= max_occupancy(N),
                "maximum occupancy violated"
            );
            n.chk(prev)
        }

        None => (0, prev),
    }
}

#[cfg(test)]
impl<K: Ord, V, const N: usize> Node<K, V, N> {
    fn chk(&self, prev: Option<&K>) -> (usize, Option<&K>) {
        assert!(!self.elems.is_empty(), "no entries");

        let (ht, prev) = chk_node_ptr(self.child(0), prev);
        if let Some(k) = prev {
            assert!(k < self.key(0), "order violation");
        }
        let mut prev = Some(self.key(0));

        for i in 1..self.len() {
            let curr = chk_node_ptr(self.child(i), prev);
            assert_eq!(ht, curr.0, "uneven branches");
            assert!(prev.unwrap() < self.key(i), "order violation");
            prev = Some(self.key(i));
        }

        let curr = chk_node_ptr(self.kids.last(), prev);
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

#[cfg(feature = "serde")]
impl<K, V, const N: usize> serde::ser::Serialize for BTreeMap<K, V, N>
where
    K: serde::ser::Serialize,
    V: serde::ser::Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::ser::Serializer,
    {
        use serde::ser::SerializeMap;
        let mut map = serializer.serialize_map(Some(self.len()))?;
        for (k, v) in self {
            map.serialize_entry(k, v)?;
        }
        map.end()
    }
}

#[cfg(feature = "serde")]
impl<'de, K, V, const N: usize> serde::de::Deserialize<'de>
    for BTreeMap<K, V, N>
where
    K: Clone + serde::de::Deserialize<'de> + Ord,
    V: Clone + serde::de::Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::de::Deserializer<'de>,
    {
        let map_visitor = super::serde::MapVisitor {
            map: Box::new(BTreeMap::new()),
            desc: "lazy_clone_collections::AvlMap".to_string(),
        };
        deserializer.deserialize_map(map_visitor)
    }
}

impl<K: Clone, V: Clone, const N: usize> From<BTreeMap<K, V, N>>
    for Vec<(K, V)>
{
    fn from(value: BTreeMap<K, V, N>) -> Self {
        let mut res = Self::new();
        for (k, v) in value {
            res.push((k, v));
        }
        res
    }
}

#[cfg(test)]
mod test {
    extern crate quickcheck;
    use quickcheck::quickcheck;

    // use the smallest possible MIN_OCCUPATION to stress rebalances, splits,
    // rotations, etc.
    type BTreeMap<K, V> = super::BTreeMap<K, V, 1>;
    type StdBTreeMap<K, V> = std::collections::BTreeMap<K, V>;
    type TestElems = Vec<(u8, u16)>;

    fn make_maps(v: TestElems) -> (BTreeMap<u8, u16>, StdBTreeMap<u8, u16>) {
        (BTreeMap::from_iter(v.clone()), StdBTreeMap::from_iter(v))
    }

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

    fn test_insert(elems: TestElems) {
        let mut m1 = BTreeMap::new();
        let mut m2 = std::collections::BTreeMap::new();
        for (k, v) in elems {
            assert_eq!(m1.insert(k, v), m2.insert(k, v));
            assert_eq!(m1.len(), m2.len());
            assert!(m1.contains_key(&k));
            m1.chk();
        }

        for (k, v) in m2.iter() {
            assert_eq!(m1.get(k), Some(v));
        }

        assert!(m1.iter().cmp(m2.iter()).is_eq());
    }

    fn test_remove(elems: TestElems) {
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

    fn into_iter_test(u: Vec<u8>, v: Vec<u8>) {
        // unshared
        let m1: BTreeMap<u8, ()> = u.iter().map(|x| (*x, ())).collect();
        let n1: std::collections::BTreeMap<u8, ()> =
            u.iter().map(|x| (*x, ())).collect();
        assert!(m1.into_iter().cmp(n1.into_iter()).is_eq());

        // shared
        let m1: BTreeMap<u8, ()> = u.iter().map(|x| (*x, ())).collect();
        let m2 = m1;
        let n1: std::collections::BTreeMap<u8, ()> =
            u.iter().map(|x| (*x, ())).collect();
        assert!(m2.into_iter().cmp(n1.into_iter()).is_eq());

        // partly shared
        let m1: BTreeMap<u8, ()> = u.iter().map(|x| (*x, ())).collect();
        let n1: std::collections::BTreeMap<u8, ()> =
            u.iter().map(|x| (*x, ())).collect();

        let mut m2 = m1.clone();
        m2.extend(v.iter().map(|x| (*x, ())));

        let mut n2 = n1.clone();
        n2.extend(v.iter().map(|x| (*x, ())));

        assert!(m2.into_iter().cmp(n2.into_iter()).is_eq());
        assert!(m1.into_iter().cmp(n1.into_iter()).is_eq());
    }

    fn into_iter_rev(u: Vec<u8>) {
        let m1: BTreeMap<u8, ()> = u.iter().map(|x| (*x, ())).collect();
        let n1: std::collections::BTreeMap<u8, ()> =
            u.iter().map(|x| (*x, ())).collect();
        assert!(m1.into_iter().rev().cmp(n1.into_iter().rev()).is_eq());
    }

    #[test]
    fn into_iter_rev_regr1() {
        into_iter_rev(vec![0, 1, 2])
    }

    fn iter_mut_test(v1: Vec<u8>, v2: Vec<u8>) {
        let mut m1 = BTreeMap::new();
        let mut n1 = std::collections::BTreeMap::new();
        for i in v1 {
            m1.insert(i, 0);
            n1.insert(i, 0);
        }

        let mut m2 = m1.clone();
        let mut n2 = n1.clone();
        for i in v2 {
            m2.insert(i, 1);
            n2.insert(i, 1);
        }

        m1.iter_mut().for_each(|(k, v)| *v = (*k).wrapping_mul(2));
        n1.iter_mut().for_each(|(k, v)| *v = (*k).wrapping_mul(2));

        m2.iter_mut().for_each(|(k, v)| *v = (*k).wrapping_mul(3));
        n2.iter_mut().for_each(|(k, v)| *v = (*k).wrapping_mul(3));

        assert!(m1.iter().cmp(n1.iter()).is_eq());
        assert!(m2.iter().cmp(n2.iter()).is_eq());
    }

    fn iter_mut_rev_test(v: Vec<u8>) {
        let v: Vec<(u8, u8)> = v.into_iter().map(|k| (k, 0)).collect();
        let mut m1: BTreeMap<_, _> = v.clone().into_iter().collect();
        let mut n1: std::collections::BTreeMap<_, _> = v.into_iter().collect();

        let mut i = m1.iter_mut();
        let mut j = n1.iter_mut();
        while let Some((k1, v1)) = i.next_back() {
            let Some((k2, v2)) = j.next_back() else {
                panic!("early end");
            };

            assert_eq!(k1, k2);
            assert_eq!(v1, v2);
        }

        assert_eq!(j.next(), None);
    }

    #[test]
    fn iter_mut_rev_regr1() {
        iter_mut_rev_test(vec![0, 1, 2]);
    }

    fn iter_mut_alt_test(v: Vec<u8>, dirs: Vec<bool>) {
        let v: Vec<(u8, u8)> = v.into_iter().map(|k| (k, 0)).collect();
        let mut m1: BTreeMap<_, _> = v.clone().into_iter().collect();
        let n1: std::collections::BTreeMap<_, _> = v.into_iter().collect();

        // switch back to vec for better debugging output
        let mut v: Vec<_> = n1.into_iter().collect();

        let mut i = m1.iter_mut();
        for is_fwd in dirs {
            // println!("i: {:?}", i);
            if is_fwd {
                // println!("fwd {:?}", v);
                assert_eq!(i.next().map(|e| *e.0), v.first().map(|e| e.0));
                if !v.is_empty() {
                    v.remove(0);
                }
            } else {
                // println!("rev {:?}", v);
                assert_eq!(i.next_back().map(|e| *e.0), v.pop().map(|e| e.0));
            }
        }
    }

    fn into_iter_alt_test(v: Vec<u8>, dirs: Vec<bool>) {
        let v: Vec<(u8, u8)> = v.into_iter().map(|k| (k, 0)).collect();

        // create map with some sharing to test into_iter moving/cloning
        let mut v_lo = v.clone();
        let v_hi = v_lo.split_off(v.len() / 2);
        let m: BTreeMap<_, _> = v.clone().into_iter().collect();
        let mut m1 = m.clone();
        m1.extend(v_hi);
        assert!(m.len() <= m1.len());

        // switch back to vec for better debugging output
        let n1: std::collections::BTreeMap<_, _> = v.into_iter().collect();
        let mut v: Vec<_> = n1.into_iter().collect();

        let mut i = m1.iter_mut();
        for is_fwd in dirs {
            // println!("i: {:?}", i);
            if is_fwd {
                // println!("fwd {:?}", v);
                assert_eq!(i.next().map(|e| *e.0), v.first().map(|e| e.0));
                if !v.is_empty() {
                    v.remove(0);
                }
            } else {
                // println!("rev {:?}", v);
                assert_eq!(i.next_back().map(|e| *e.0), v.pop().map(|e| e.0));
            }
        }
    }

    #[test]
    fn iter_mut_alt_regr1() {
        iter_mut_alt_test(
            vec![0, 1, 2, 3, 4, 5],
            vec![true, true, true, false, true, true],
        );
    }

    #[test]
    fn iter_mut_alt_regr2() {
        iter_mut_alt_test(
            vec![3, 7, 8, 1, 9, 4, 0, 10, 2, 11, 5, 12, 13, 6, 14],
            vec![true, true, true, true, true, true, true, false, true, false],
        );
    }

    #[test]
    fn iter_mut_alt_regr3() {
        iter_mut_alt_test(
            vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            vec![
                true, true, true, true, true, true, true, false, true, true,
                true, true, true, true,
            ],
        );
    }

    #[test]
    fn iter_mut_alt_regr4() {
        iter_mut_alt_test(vec![0, 1], vec![false]);
    }

    #[test]
    fn into_iter_regr1() {
        into_iter_test(vec![], vec![0, 1, 2]);
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

    #[test]
    fn remove_regr2() {
        let elems =
            vec![(21, 0), (0, 0), (1, 0), (22, 0), (23, 0), (24, 0), (149, 0)];
        test_remove(elems);
    }

    #[test]
    fn remove_regr3() {
        let elems =
            vec![(127, 0), (0, 0), (1, 0), (2, 0), (4, 0), (3, 0), (255, 0)];
        test_remove(elems);
    }

    // I cannot see how to copy nor clone RangeBounds.  As a work around, pass
    // the same range twice.
    fn test_range<R>(v: TestElems, r1: R, r2: R)
    where
        R: std::fmt::Debug + std::ops::RangeBounds<u8>,
    {
        assert_eq!(r1.start_bound(), r2.start_bound());
        assert_eq!(r2.end_bound(), r2.end_bound());
        let (m, n) = make_maps(v);
        let mr = m.range(r1);
        assert!(mr.cmp(n.range(r2)).is_eq());
    }

    fn test_small_range<R>(r1: R, r2: R)
    where
        R: std::fmt::Debug + std::ops::RangeBounds<u8>,
    {
        test_range(vec![(1, 0), (3, 0), (5, 0)], r1, r2);
    }

    fn test_range_combos(v: TestElems, lb: u8, ub: u8) {
        let (lb, ub) = if ub < lb { (ub, lb) } else { (lb, ub) };

        test_range(v.clone(), lb..ub, lb..ub);
        test_range(v.clone(), lb..=ub, lb..=ub);
        test_range(v.clone(), lb.., lb..);
        test_range(v.clone(), ..ub, ..ub);
        test_range(v.clone(), ..=ub, ..=ub);

        use std::ops::Bound::*;
        test_range(
            v.clone(),
            (Excluded(lb), Included(ub)),
            (Excluded(lb), Included(ub)),
        );

        if lb < ub {
            test_range(
                v,
                (Excluded(lb), Excluded(ub)),
                (Excluded(lb), Excluded(ub)),
            );
        }
    }

    #[test]
    fn test_range_all() {
        test_small_range(.., ..);
    }

    #[test]
    fn test_range_left_inclusive() {
        for i in 0..=6 {
            test_small_range(i.., i..);
        }
    }

    #[test]
    fn test_range_left_exclusive() {
        use std::ops::Bound::*;
        for i in 0..=6 {
            test_small_range(
                (Excluded(i), Unbounded),
                (Excluded(i), Unbounded),
            );
        }
    }

    #[test]
    fn test_range_right_inclusive() {
        for i in 0..=6 {
            test_small_range(..=i, ..=i); // HERE
        }
    }

    #[test]
    fn test_range_right_exclusize() {
        for i in 0..=6 {
            test_small_range(..i, ..i);
        }
    }

    use std::ops::Bound::*;

    #[test]
    #[should_panic]
    fn test_range_panic() {
        let m: BTreeMap<_, _> = [(0, 0), (3, 3)].into_iter().collect();
        m.range((Excluded(0), Excluded(0)));
    }

    #[test]
    #[should_panic]
    fn test_range_panic2() {
        let m: BTreeMap<_, _> = [(0, 0), (3, 3)].into_iter().collect();
        #[allow(clippy::reversed_empty_ranges)]
        m.range(1..0);
    }

    #[test]
    fn test_range_regr1() {
        test_range(vec![(255, 0)], 255..255, 255..255);
        test_range(vec![(255, 0)], 255..=255, 255..=255);
        test_range(vec![(255, 0)], 255.., 255..);
    }

    #[test]
    fn test_range_regr2() {
        test_range_combos(vec![(0, 0), (1, 0), (2, 0)], 2, 0);
    }

    #[test]
    fn test_range_regr3() {
        test_range_combos(vec![(1, 0)], 0, 2);
    }

    quickcheck! {
        fn qc_insert(elems: TestElems) -> () {
            test_insert(elems);
        }

        fn qc_remove(elems: TestElems) -> () {
            test_remove(elems);
        }

        fn qc_into_iter(u: Vec<u8>, v: Vec<u8>) -> () {
            into_iter_test(u, v);
        }

        fn qc_into_iter_rev(u: Vec<u8>) -> () {
            into_iter_rev(u);
        }

        fn qc_iter_mut(u: Vec<u8>, v: Vec<u8>) -> () {
            iter_mut_test(u, v);
        }

        fn qc_iter_mut_rev(u: Vec<u8>) -> () {
            iter_mut_rev_test(u);
        }

        fn qc_iter_mut_alt(u: Vec<u8>, dirs: Vec<bool>) -> () {
            iter_mut_alt_test(u, dirs);
        }

        fn qc_into_iter_alt(v: Vec<u8>, dirs: Vec<bool>) -> () {
            into_iter_alt_test(v, dirs)
        }

        fn qc_range(v: TestElems, lb: u8, ub: u8) -> () {
            test_range_combos(v, lb, ub);
        }
    }
}
