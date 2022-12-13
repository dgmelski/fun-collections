use std::borrow::Borrow;
use std::cmp::Ordering::*;
use std::mem::replace;
use std::rc::Rc;

type NodePtr<K, V, const N: usize> = Rc<Node<K, V, N>>;

struct Node<K, V, const N: usize> {
    elems: Vec<(K, V)>,
    kids: Vec<NodePtr<K, V, N>>,
}

impl<K: Clone, V: Clone, const N: usize> Clone for Node<K, V, N> {
    fn clone(&self) -> Self {
        Self {
            elems: self.elems.clone(),
            kids: self.kids.clone(),
        }
    }
}

enum InsertResult<K, V, const N: usize> {
    Replaced(V),
    Split((Option<NodePtr<K, V, N>>, K, V)),
    Absorbed,
}

struct NeedsRebal(bool);

impl<K, V, const N: usize> Node<K, V, N> {
    // minimum and maximum element counts for non-root nodes
    const MIN_OCCUPANCY: usize = N;
    const MAX_OCCUPANCY: usize = 2 * N;

    fn child(&self, i: usize) -> Option<&Rc<Self>> {
        self.kids.get(i)
    }

    fn child_mut(&mut self, i: usize) -> Option<&mut Rc<Self>> {
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

    fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Ord,
    {
        for i in 0..self.len() {
            match key.cmp(self.key(i).borrow()) {
                Less => return self.child(i).and_then(|n| n.get(key)),
                Equal => return Some(self.val(i)),
                Greater => (),
            }
        }

        self.kids.last().and_then(|n| n.get(key))
    }

    fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q> + Clone,
        V: Clone,
        Q: Ord,
    {
        for i in 0..self.len() {
            match key.cmp(self.key(i).borrow()) {
                Less => {
                    if let Some(rc) = self.child_mut(i) {
                        let n = Rc::make_mut(rc);
                        return n.get_mut(key);
                    } else {
                        return None;
                    }
                }

                Equal => return Some(self.val_mut(i)),
                Greater => (),
            }
        }

        if let Some(rc) = self.kids.last_mut() {
            let n = Rc::make_mut(rc);
            n.get_mut(key)
        } else {
            None
        }
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

        // Recurse to the appropriate child if it exists (we're not a leaf).
        // If we are a leaf, pretend that we visited a child and it resulted in
        // needing to insert a new separator at this level.
        let res = match self.child_mut(ub_x) {
            Some(n) => Rc::make_mut(n).insert(key, val),
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
            let lhs = Some(Rc::new(Node {
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

        let n = Rc::make_mut(right);

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

        let left = Rc::make_mut(left);
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

        let left = Rc::make_mut(left);
        let (k0, v0) = left.elems.pop().unwrap();
        let k0_to_k1 = left.kids.pop();

        // move k0 and v0 into the pivot position
        let k1 = replace(self.key_mut(idx), k0);
        let v1 = replace(self.val_mut(idx), v0);

        // move k1 and v1 down and to the right of the pivot
        let right = self.child_mut(idx + 1).unwrap();
        let right = Rc::make_mut(right);
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
        let mut lhs_n = match Rc::try_unwrap(lhs_rc) {
            Ok(n) => n,
            Err(rc) => (*rc).clone(),
        };

        // put the separator key & val into the lhs
        lhs_n.elems.push((mid_k, mid_v));

        // get a private copy of the rhs
        let rhs_rc = self.child_mut(at).unwrap();
        let rhs_ref = Rc::make_mut(rhs_rc);

        // We own & can take from lhs_n, but we want rhs's elements at the end.
        // Swap lhs & rhs vecs so we can use a cheaper append for merging.
        std::mem::swap(&mut lhs_n.elems, &mut rhs_ref.elems);
        rhs_ref.elems.extend(lhs_n.elems);

        std::mem::swap(&mut lhs_n.kids, &mut rhs_ref.kids);
        rhs_ref.kids.extend(lhs_n.kids);
    }

    // rebalances when the self.elems[at] is underpopulated
    fn rebal(&mut self, at: usize) -> NeedsRebal
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

        NeedsRebal(self.elems.len() < Self::MIN_OCCUPANCY)
    }

    fn rm_greatest(&mut self) -> (K, V, NeedsRebal)
    where
        K: Clone,
        V: Clone,
    {
        if let Some(rt) = self.kids.last_mut() {
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
            let (k, v) = self.elems.pop().unwrap();
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
            match key.cmp(self.key(i).borrow()) {
                Less => {
                    if self.is_leaf() {
                        return (None, NeedsRebal(false));
                    }

                    let lt_k = self.child_mut(i).unwrap();
                    let lt_k = Rc::make_mut(lt_k);
                    let ret = lt_k.remove(key);
                    if let &(_, NeedsRebal(true)) = &ret {
                        return (ret.0, self.rebal(i));
                    } else {
                        return ret;
                    }
                }

                Equal => {
                    if self.is_leaf() {
                        let old_v = self.elems.remove(i).1;
                        return (
                            Some(old_v),
                            NeedsRebal(self.elems.len() < Self::MIN_OCCUPANCY),
                        );
                    }

                    let lt_k = self.child_mut(i).unwrap();
                    let lt_k = Rc::make_mut(lt_k);
                    let (k, v, needs_rebal) = lt_k.rm_greatest();
                    *self.key_mut(i) = k;
                    let old_v = replace(self.val_mut(i), v);
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

        let gt_k = self.kids.last_mut().unwrap();
        let gt_k = Rc::make_mut(gt_k);
        let ret = gt_k.remove(key);
        if let &NeedsRebal(true) = &ret.1 {
            (ret.0, self.rebal(self.elems.len()))
        } else {
            ret
        }
    }
}

pub struct BTreeMap<K, V, const N: usize = 2> {
    len: usize,
    root: Option<NodePtr<K, V, N>>,
}

impl<K, V, const N: usize> Clone for BTreeMap<K, V, N>
where
    K: Clone,
    V: Clone,
{
    fn clone(&self) -> Self {
        Self {
            len: self.len,
            root: self.root.clone(),
        }
    }
}

impl<K, V, const N: usize> Default for BTreeMap<K, V, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V, const N: usize> BTreeMap<K, V, N> {
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Ord,
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

    /// Retrieves the value associated with the given key, if it is in the map.
    ///
    /// # Examples
    /// ```
    /// use fun_collections::BTreeMap;
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

    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q> + Clone,
        V: Clone,
        Q: Ord,
    {
        if let Some(rc) = self.root.as_mut() {
            let n = Rc::make_mut(rc);
            n.get_mut(key)
        } else {
            None
        }
    }

    /// Associates 'val' with 'key' and returns the value previously associated
    /// with 'key', if it exists.
    ///
    /// # Examples
    /// ```
    /// use fun_collections::BTreeMap;
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

                InsertResult::Split((lhs, k, v)) => {
                    self.len += 1;
                    self.root = Some(Rc::new(Node {
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
            self.root = Some(Rc::new(Node {
                elems: vec![(key, val)],
                kids: Vec::new(),
            }));
            None
        }
    }

    pub fn iter(&self) -> Iter<'_, K, V, N> {
        let mut curr = self.root.as_ref();
        let mut w = Vec::new();
        while let Some(rc) = curr {
            w.push((rc.as_ref(), 0));
            curr = rc.child(0);
        }

        Iter { w }
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
    /// use fun_collections::BTreeMap;
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

            if needs_rebal.0 && n.elems.is_empty() {
                self.root = n.kids.pop();
            }

            old_v
        } else {
            None
        }
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

pub struct Iter<'a, K, V, const N: usize> {
    w: Vec<(&'a Node<K, V, N>, usize)>,
}

impl<'a, K, V, const N: usize> Iterator for Iter<'a, K, V, N> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some((n, i)) = self.w.last_mut() {
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

            Some(ret)
        } else {
            None
        }
    }
}

#[cfg(test)]
fn chk_node_ptr<'a, K: Ord, V, const N: usize>(
    n: Option<&'a Rc<Node<K, V, N>>>,
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

        let (ht, prev) = chk_node_ptr(self.child(0), prev);
        prev.map(|k| assert!(k < self.key(0), "order violation"));
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

    fn test_insert(elems: TestElems) -> () {
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
    }
}
