use std::borrow::Borrow;
use std::cmp::Ordering::*;
use std::mem::replace;
use std::sync::Arc;

use crate::{Entry, Map, OccupiedEntry, VacantEntry};

pub mod btree_set;

type NodePtr<K, V, const N: usize> = Arc<Node<K, V, N>>;

#[derive(Clone)]
struct Node<K, V, const N: usize> {
    elems: Vec<(K, V)>,
    kids: Vec<NodePtr<K, V, N>>,
}

enum InsertResult<K, V, const N: usize> {
    Replaced(V),
    Split((Option<NodePtr<K, V, N>>, K, V)),
    Absorbed,
}

struct IsUnderPop(bool);

impl<K, V, const N: usize> Node<K, V, N> {
    // minimum and maximum element counts for non-root nodes
    const MIN_OCCUPANCY: usize = N;
    const MAX_OCCUPANCY: usize = 2 * N;

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

    fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
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

                Equal => return Some(self.val_mut(i)),
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
            None => Split((None, key, val)),
        };

        // update for a node split at the next level down
        if let Split((child, k, v)) = res {
            // TODO: split before insert to reduce memmove

            self.elems.insert(ub_x, (k, v));
            if let Some(rc) = child {
                self.kids.insert(ub_x, rc);
            }

            if self.elems.len() <= Self::MAX_OCCUPANCY {
                return Absorbed;
            }

            // split this overcrowded node

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

            Split((lhs, mid_k, mid_v))
        } else {
            // res is Replaced(v) or Absorbed
            res
        }
    }

    fn is_branch(&self) -> bool {
        !self.kids.is_empty()
    }

    fn is_leaf(&self) -> bool {
        self.kids.is_empty()
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
pub struct BTreeMap<K, V, const N: usize = 2> {
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
        // TODO: all the same, do we need the double lookup?
        if self.contains_key(&key) {
            let val = self.get_mut(&key).unwrap();
            Entry::Occupied(OccupiedEntry { key, val })
        } else {
            Entry::Vacant(VacantEntry { key, map: self })
        }
    }

    /// Return an Entry for the least key in the map.
    pub fn first_entry(&mut self, key: K) -> Entry<'_, Self>
    where
        K: Clone + Ord,
        V: Clone,
    {
        // avoid "double borrow" problem
        if self.is_empty() {
            return Entry::Vacant(VacantEntry { key, map: self });
        }

        let curr = self.root.as_mut().unwrap();
        let mut n = Arc::make_mut(curr);
        while !n.kids.is_empty() {
            n = Arc::make_mut(&mut n.kids[0]);
        }

        Entry::Occupied(OccupiedEntry {
            key,
            val: &mut n.elems[0].1,
        })
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
        n.get_mut(key)
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

                InsertResult::Split((lhs, k, v)) => {
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

    pub fn into_keys(self) -> impl Iterator<Item = K>
    where
        K: Clone,
        V: Clone, // needed to clone shared nodes
    {
        // TODO: needlessly clones values from owned nodes
        self.into_iter().map(|e| e.0)
    }

    pub fn into_values(self) -> impl Iterator<Item = V>
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
        let mut curr = self.root.as_ref();
        let mut w = Vec::new();
        while let Some(rc) = curr {
            w.push((rc.as_ref(), 0));
            curr = rc.child(0);
        }

        Iter { w, len: self.len() }
    }

    pub fn iter_mut(&mut self) -> IterMut<'_, K, V, N>
    where
        K: Clone,
        V: Clone,
    {
        let mut w = Vec::new();
        let mut curr = self.root.as_mut();
        while let Some(arc) = curr {
            let n = Arc::make_mut(arc);
            let elems = n.elems.iter_mut();
            let mut kids = n.kids.iter_mut();
            curr = kids.next();
            w.push((elems, kids));
        }

        IterMut {
            fwd: w,
            rev: Vec::new(),
            len: self.len,
        }
    }

    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.iter().map(|e| e.0)
    }

    /// Return an Entry for the least key in the map.
    pub fn last_entry(&mut self, key: K) -> Entry<'_, Self>
    where
        K: Clone + Ord,
        V: Clone,
    {
        // avoid "double borrow" problem
        if self.is_empty() {
            return Entry::Vacant(VacantEntry { key, map: self });
        }

        let curr = self.root.as_mut().unwrap();
        let mut n = Arc::make_mut(curr);
        while let Some(arc) = n.kids.last_mut() {
            n = Arc::make_mut(arc);
        }

        Entry::Occupied(OccupiedEntry {
            key,
            val: &mut n.elems.last_mut().unwrap().1,
        })
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

    // TODO: range
    // TODO: range_mut

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
            .partition(|(k, _)| key < k.borrow());

        *self = a;
        b
    }

    // TOOD: try_insert()

    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.iter().map(|e| e.1)
    }

    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut V>
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

impl<K, V, const N: usize> From<[(K, V); N]> for BTreeMap<K, V, N>
where
    K: Clone + Ord,
    V: Clone,
{
    fn from(vs: [(K, V); N]) -> Self {
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

impl<K, V, const N: usize> Map for BTreeMap<K, V, N> {
    type Key = K;
    type Value = V;

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

pub struct Iter<'a, K, V, const N: usize> {
    w: Vec<(&'a Node<K, V, N>, usize)>,
    len: usize,
}

impl<'a, K, V, const N: usize> std::fmt::Debug for Iter<'a, K, V, N>
where
    K: std::fmt::Debug,
    V: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("btree::Iter")
            .field("len", &self.len)
            .field("next", &self.w.last().and_then(|(n, i)| n.elems.get(*i)))
            .finish()
    }
}

impl<'a, K, V, const N: usize> Iterator for Iter<'a, K, V, N> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        let (n, i) = self.w.last_mut()?;
        let ret = (n.key(*i), n.val(*i));

        *i += 1;

        let mut curr = if *i < n.len() {
            n.child(*i)
        } else {
            let curr = n.kids.last();
            self.w.pop();
            curr
        };

        while let Some(rc) = curr {
            self.w.push((rc.as_ref(), 0));
            curr = rc.child(0);
        }

        self.len -= 1;

        Some(ret)
    }
}

type IterMutWorklist<'a, K, V, const N: usize> = Vec<(
    std::slice::IterMut<'a, (K, V)>,
    std::slice::IterMut<'a, Arc<Node<K, V, N>>>,
)>;

pub struct IterMut<'a, K, V, const N: usize> {
    fwd: IterMutWorklist<'a, K, V, N>,
    rev: IterMutWorklist<'a, K, V, N>,
    len: usize,
}

impl<'a, K, V, const N: usize> IterMut<'a, K, V, N> {
    fn try_rot_lf(&mut self) -> Option<&'a mut Arc<Node<K, V, N>>> {
        let (elems, kids) = self.rev.get_mut(0)?;

        if elems.len() == 0 && kids.len() == 0 {
            let _ = self.rev.remove(0);
            return self.try_rot_lf();
        }

        if kids.len() < elems.len() {
            return None;
        }

        let ret = kids.next();

        if elems.len() == 0 && kids.len() == 0 {
            let _ = self.rev.remove(0);
        }

        ret
    }

    fn try_rot_rt(&mut self) -> Option<&'a mut Arc<Node<K, V, N>>> {
        let (elems, kids) = self.fwd.get_mut(0)?;

        if elems.len() == 0 && kids.len() == 0 {
            let _ = self.fwd.remove(0);
            return self.try_rot_rt();
        }

        if kids.len() < elems.len() {
            return None;
        }

        let ret = kids.next_back();

        if elems.len() == 0 && kids.len() == 0 {
            let _ = self.fwd.remove(0);
        }

        ret
    }

    fn descend_left(&mut self, mut curr: Option<&'a mut Arc<Node<K, V, N>>>)
    where
        K: Clone,
        V: Clone,
    {
        while let Some(arc) = curr {
            let n = Arc::make_mut(arc);
            let elems = n.elems.iter_mut();
            let mut kids = n.kids.iter_mut();
            curr = kids.next();
            self.fwd.push((elems, kids));
        }
    }

    fn descend_right(&mut self, mut curr: Option<&'a mut Arc<Node<K, V, N>>>)
    where
        K: Clone,
        V: Clone,
    {
        while let Some(arc) = curr {
            let n = Arc::make_mut(arc);
            let elems = n.elems.iter_mut();
            let mut kids = n.kids.iter_mut();
            curr = kids.next_back();
            self.rev.push((elems, kids));
        }
    }

    fn advance_fwd(&mut self) -> Option<()>
    where
        K: Clone,
        V: Clone,
    {
        let mut curr;
        if let Some((elems, kids)) = self.fwd.last_mut() {
            curr = kids.next();

            if elems.len() == 0 {
                assert_eq!(kids.len(), 0);
                self.fwd.pop();
            }

            if curr.is_none() && self.fwd.is_empty() {
                curr = self.try_rot_lf();
            }
        } else {
            curr = self.try_rot_lf();
        }

        self.descend_left(curr);

        Some(())
    }

    fn advance_rev(&mut self) -> Option<()>
    where
        K: Clone,
        V: Clone,
    {
        let mut curr;
        if let Some((elems, kids)) = self.rev.last_mut() {
            curr = kids.next_back();

            if elems.len() == 0 {
                assert_eq!(kids.len(), 0);
                self.rev.pop();
            }

            if curr.is_none() && self.rev.is_empty() {
                curr = self.try_rot_rt();
            }
        } else {
            curr = self.try_rot_rt();
        }

        self.descend_right(curr);

        Some(())
    }
}

impl<'a, K, V, const N: usize> std::fmt::Debug for IterMut<'a, K, V, N>
where
    K: std::fmt::Debug,
    V: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("btree::IterMut")
            .field("len", &self.len)
            .field("node_elems", &self.fwd.last().map(|(elems, _)| elems))
            .finish()
    }
}

impl<'a, K, V, const N: usize> Iterator for IterMut<'a, K, V, N>
where
    K: Clone,
    V: Clone,
{
    type Item = (&'a K, &'a mut V);

    fn next(&mut self) -> Option<Self::Item> {
        let ret = self
            .fwd
            .last_mut()
            .and_then(|e| e.0.next())
            .or_else(|| self.rev.get_mut(0).and_then(|e| e.0.next()))?;

        self.advance_fwd();

        self.len -= 1;

        let (ref k, ref mut v) = ret;
        Some((k, v))
    }
}

impl<'a, K, V, const N: usize> DoubleEndedIterator for IterMut<'a, K, V, N>
where
    K: Clone,
    V: Clone,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        // do lazy initialization of rev iterator stack
        if self.rev.is_empty() {
            if let Some((elems, kids)) = self.fwd.get_mut(0) {
                assert!(elems.len() >= kids.len());
                if elems.len() == kids.len() {
                    let curr = kids.next_back();
                    self.descend_right(curr);
                }
            }
        }

        let ret = {
            let t = self.rev.last_mut();
            let t = t.and_then(|e| e.0.next_back());
            t.or_else(|| self.fwd.get_mut(0).and_then(|e| e.0.next_back()))?
        };

        self.advance_rev();

        self.len -= 1;

        let (ref k, ref mut v) = ret;
        Some((k, v))
    }
}

// "erg" as in "unit of work" that is put on the IntoIter's worklist
enum IntoIterErg<K, V, const N: usize> {
    Owned(
        std::vec::IntoIter<(K, V)>,
        std::vec::IntoIter<Arc<Node<K, V, N>>>,
    ),
    Borrowed(Arc<Node<K, V, N>>, usize),
}

pub struct IntoIter<K, V, const N: usize> {
    len: usize,
    work: Vec<IntoIterErg<K, V, N>>,
}

impl<K, V, const N: usize> IntoIter<K, V, N> {
    fn descend(&mut self) {
        use IntoIterErg::*;
        while let Some(curr) = self.work.last_mut() {
            match curr {
                Owned(elems, kids) => {
                    if let Some(arc) = kids.next() {
                        let next_erg = match Arc::try_unwrap(arc) {
                            Ok(n) => {
                                Owned(n.elems.into_iter(), n.kids.into_iter())
                            }
                            Err(arc) => Borrowed(arc, 0),
                        };

                        if kids.len() == 0 {
                            assert_eq!(elems.len(), 0);
                            self.work.pop();
                        }

                        self.work.push(next_erg);
                    } else {
                        if elems.len() == 0 {
                            self.work.pop();
                        }
                        return;
                    }
                }

                Borrowed(arc, i) => {
                    if *i < arc.kids.len() {
                        let next_kid = arc.kids[*i].clone();

                        if *i == arc.elems.len() {
                            self.work.pop();
                        }

                        self.work.push(Borrowed(next_kid, 0));
                    } else {
                        if *i >= arc.elems.len() {
                            self.work.pop();
                        }
                        return;
                    }
                }
            };
        }
    }

    fn new(m: BTreeMap<K, V, N>) -> Self {
        if m.is_empty() {
            return Self {
                len: 0,
                work: Vec::new(),
            };
        }

        use IntoIterErg::*;
        let state1 = match Arc::try_unwrap(m.root.unwrap()) {
            Ok(n) => Owned(n.elems.into_iter(), n.kids.into_iter()),
            Err(arc) => Borrowed(arc, 0),
        };

        let mut ii = Self {
            len: m.len,
            work: vec![state1],
        };
        ii.descend();

        ii
    }
}

impl<K, V, const N: usize> std::fmt::Debug for IntoIter<K, V, N>
where
    K: std::fmt::Debug,
    V: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name: &str;
        let desc: &dyn std::fmt::Debug;
        let some_elems: Option<&std::vec::IntoIter<(K, V)>>;
        let some_kv: Option<&(K, V)>;
        match self.work.last() {
            None => {
                name = "next";
                desc = &None::<(K, V)>;
            }

            Some(IntoIterErg::Owned(elems, _)) => {
                name = "node_elems";
                some_elems = Some(elems);
                desc = &some_elems;
            }

            Some(IntoIterErg::Borrowed(n, i)) => {
                name = "next";
                some_kv = n.elems.get(*i);
                desc = &some_kv;
            }
        };

        f.debug_struct("btree::IntoIter")
            .field("len", &self.len)
            .field(name, desc)
            .finish()
    }
}

impl<K, V, const N: usize> Iterator for IntoIter<K, V, N>
where
    K: Clone,
    V: Clone,
{
    type Item = (K, V);

    fn next(&mut self) -> Option<Self::Item> {
        use IntoIterErg::*;

        let ret = match self.work.last_mut()? {
            Owned(elems, _) => elems.next().unwrap(),

            Borrowed(arc, i) => {
                let ret = arc.elems[*i].clone();
                *i += 1;
                ret
            }
        };

        self.descend();

        self.len -= 1;

        Some(ret)
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
            assert!(n.elems.len() <= 2 * N, "maximum occupancy violated");
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

#[cfg(test)]
mod test {
    extern crate quickcheck;
    use quickcheck::quickcheck;

    // use the smallest possible MIN_OCCUPATION to stress rebalances, splits,
    // rotations, etc.
    type BTreeMap<K, V> = super::BTreeMap<K, V, 1>;
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

        fn qc_iter_mut(u: Vec<u8>, v: Vec<u8>) -> () {
            iter_mut_test(u, v);
        }

        fn qc_iter_mut_rev(u: Vec<u8>) -> () {
            iter_mut_rev_test(u);
        }

        fn qc_iter_mut_alt(u: Vec<u8>, dirs: Vec<bool>) -> () {
            iter_mut_alt_test(u, dirs);
        }
    }
}
