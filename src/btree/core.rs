#![allow(unused_imports)]
#![allow(dead_code)]

use std::borrow::Borrow;
use std::cmp::Ordering::*;
use std::collections::VecDeque;
use std::iter::FusedIterator;
use std::marker::PhantomData;
use std::mem::{replace, MaybeUninit};
use std::ops::{Deref, DerefMut};
use std::ptr::{copy, copy_nonoverlapping};
use std::sync::Arc;

pub const fn max_occupancy(min_occ: usize) -> usize {
    2 * min_occ + 1
}

pub const MIN_OCCUPANCY: usize = 5;
pub const MAX_OCCUPANCY: usize = max_occupancy(MIN_OCCUPANCY);
pub const MAX_KIDS: usize = MAX_OCCUPANCY + 1;

type InlineVecLen = u32;

#[allow(unused_macros)]
macro_rules! node {
    ($elems:expr, $kids:expr) => {{
        let mut n = Node::new();

        let elems = $elems;
        for e in elems {
            n.elems.push(e);
        }

        let ks = $kids;
        if !ks.is_empty() {
            let mut kids = Kids::new();
            for c in ks {
                kids.push((Arc::new(c), ()));
            }
            n.kids = Some(Box::new(kids));
        }

        n
    }};
}

pub(crate) use node;

// A vector with a fixed capacity and known Size. Conceptually, it stores
// pairs of values, called 'keys' and 'vals' w/o further interpretation.
pub struct InlineVec<K, V, const N: usize> {
    keys: [MaybeUninit<K>; N], // keys stored contiguously for cache
    vals: [MaybeUninit<V>; N],
    len: InlineVecLen,
}

impl<K, V, const N: usize> InlineVec<K, V, N> {
    pub fn new() -> Self {
        Self {
            keys: unsafe { MaybeUninit::uninit().assume_init() },
            vals: unsafe { MaybeUninit::uninit().assume_init() },
            len: 0,
        }
    }

    fn clear(&mut self) {
        for i in 0..self.len {
            unsafe {
                self.keys[i as usize].assume_init_drop();
                self.vals[i as usize].assume_init_drop();
            }
        }
        self.len = 0;
    }

    fn insert(&mut self, idx: usize, kv: (K, V)) {
        assert!(idx <= self.len as usize);
        assert!((self.len as usize) < N);
        unsafe {
            let keys = self.keys.as_mut_ptr();
            copy(keys.add(idx), keys.add(idx + 1), self.len as usize - idx);

            let vals = self.vals.as_mut_ptr();
            copy(vals.add(idx), vals.add(idx + 1), self.len as usize - idx);
        }
        self.keys[idx].write(kv.0);
        self.vals[idx].write(kv.1);
        self.len += 1;
    }

    // get the keys as a slice
    fn keys(&self) -> &[K] {
        unsafe {
            let keys: *const K = self.keys[0].assume_init_ref();
            std::slice::from_raw_parts(keys, self.len as usize)
        }
    }

    // get the keys as a mutable slice
    fn keys_mut(&mut self) -> &mut [K] {
        unsafe {
            let keys: *mut K = self.keys[0].assume_init_mut();
            std::slice::from_raw_parts_mut(keys, self.len as usize)
        }
    }

    // get mutable slices for both keys and vals
    fn elems_mut(&mut self) -> (&mut [K], &mut [V]) {
        (
            unsafe {
                let keys: *mut K = self.keys[0].assume_init_mut();
                std::slice::from_raw_parts_mut(keys, self.len as usize)
            },
            unsafe {
                let vals: *mut V = self.vals[0].assume_init_mut();
                std::slice::from_raw_parts_mut(vals, self.len as usize)
            },
        )
    }

    fn len(&self) -> usize {
        self.len as usize
    }

    pub fn push(&mut self, kv: (K, V)) {
        assert!(self.len < N as InlineVecLen);
        self.keys[self.len as usize].write(kv.0);
        self.vals[self.len as usize].write(kv.1);
        self.len += 1;
    }

    pub fn pop(&mut self) -> (K, V) {
        assert!(self.len > 0);
        self.len -= 1;
        (
            unsafe { self.keys[self.len as usize].assume_init_read() },
            unsafe { self.vals[self.len as usize].assume_init_read() },
        )
    }

    // Removes & returns the element at idx
    fn remove(&mut self, idx: usize) -> (K, V) {
        assert!(idx < self.len as usize);
        assert!(self.len > 0);
        unsafe {
            let keys = self.keys.as_mut_ptr();
            let k = keys.add(idx).read().assume_init();
            copy(
                keys.add(idx + 1),
                keys.add(idx),
                (self.len as usize) - idx - 1,
            );

            let vals = self.vals.as_mut_ptr();
            let v = vals.add(idx).read().assume_init();
            copy(
                vals.add(idx + 1),
                vals.add(idx),
                (self.len as usize) - idx - 1,
            );

            self.len -= 1;

            (k, v)
        }
    }

    fn vals(&self) -> &[V] {
        unsafe {
            let vals: *const V = self.vals[0].assume_init_ref();
            std::slice::from_raw_parts(vals, self.len as usize)
        }
    }

    fn vals_mut(&mut self) -> &mut [V] {
        unsafe {
            let vals: *mut V = self.vals[0].assume_init_mut();
            std::slice::from_raw_parts_mut(vals, self.len as usize)
        }
    }
}

impl<K, V, const N: usize> Clone for InlineVec<K, V, N>
where
    K: Clone,
    V: Clone,
{
    fn clone(&self) -> Self {
        let mut ret = Self::new();
        for i in 0..(self.len as usize) {
            unsafe {
                ret.keys[i].write(self.keys[i].assume_init_ref().clone());
                ret.vals[i].write(self.vals[i].assume_init_ref().clone());
            }
        }
        ret.len = self.len;

        ret
    }
}

impl<K, V, const N: usize> Default for InlineVec<K, V, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V> std::fmt::Debug for Elems<K, V>
where
    K: std::fmt::Debug,
    V: std::fmt::Debug,
{
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> Result<(), std::fmt::Error> {
        f.debug_list()
            .entries(self.keys().iter().zip(self.vals().iter()))
            .finish()
    }
}

impl<K, V> std::fmt::Debug for Kids<K, V>
where
    K: std::fmt::Debug,
    V: std::fmt::Debug,
{
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> Result<(), std::fmt::Error> {
        f.debug_list().entries(self.keys().iter()).finish()
    }
}

impl<K, V, const N: usize> Drop for InlineVec<K, V, N> {
    fn drop(&mut self) {
        self.clear();
    }
}

pub struct IsUnderPop(pub bool);

pub type Elems<K, V> = InlineVec<K, V, MAX_OCCUPANCY>;

// Our storage of 'Kids' is a bit hacky: we reuse our existing fixed-capacity
// vector for storing key-value pairs, but set the value type to () and trust
// the compiler to optimize away manipulations of the zero-sized type.
pub type Kids<K, V> = InlineVec<NodePtr<K, V>, (), MAX_KIDS>;

// A node in the BTree.
//
// # Layout
//
// We use a single node type to represent both internal branches and leaves.
// Most of the nodes will be leaves.  [A tree of height 3 has greater than 80%
// leaf nodes.]  Leaving space for child pointers in leaves could waste a lot of
// space, so we take the pointer-chasing penalty and store them externally.
// (TODO: use dynamically-sized types in rust, though it seems painful.)
#[derive(Clone, Default)]
pub struct Node<K, V> {
    pub elems: Elems<K, V>,
    pub kids: Option<Box<Kids<K, V>>>,
}

impl<K, V> Node<K, V> {
    // create an empty leaf
    pub fn new() -> Self {
        Self {
            elems: InlineVec::new(),
            kids: None,
        }
    }

    pub fn new_leaf(kv: (K, V)) -> Self {
        let mut ret = Self::new();
        ret.elems.push(kv);
        ret
    }

    pub fn new_branch(
        lhs: NodePtr<K, V>,
        kv: (K, V),
        rhs: NodePtr<K, V>,
    ) -> Self {
        let mut ret = Self::new_leaf(kv);
        let mut kids = Box::new(Kids::new());
        kids.push((lhs, ()));
        kids.push((rhs, ()));
        ret.kids = Some(kids);
        ret
    }

    fn child(&self, i: usize) -> Option<&NodePtr<K, V>> {
        self.kids().and_then(|ks| ks.get(i))
    }

    fn child_mut(&mut self, i: usize) -> Option<&mut NodePtr<K, V>> {
        self.kids_mut().and_then(|ks| ks.get_mut(i))
    }

    fn last_child(&self) -> Option<&NodePtr<K, V>> {
        let kids = self.kids.as_ref()?;
        if kids.len > 0 {
            kids.keys().get((kids.len - 1) as usize)
        } else {
            None
        }
    }

    fn last_child_mut(&mut self) -> Option<&mut NodePtr<K, V>> {
        let kids = self.kids.as_mut()?;
        let klen = kids.len as usize;
        if klen > 0 {
            kids.keys_mut().get_mut(klen - 1)
        } else {
            None
        }
    }

    pub fn pop_child(&mut self) -> Option<NodePtr<K, V>> {
        self.kids.as_mut().map(|ks| ks.pop().0)
    }

    fn key(&self, i: usize) -> &K {
        &self.elems.keys()[i]
    }

    fn key_mut(&mut self, i: usize) -> &mut K {
        &mut self.elems.keys_mut()[i]
    }

    fn kids(&self) -> Option<&[NodePtr<K, V>]> {
        self.kids.as_ref().map(|ks| ks.keys())
    }

    fn kids_mut(&mut self) -> Option<&mut [NodePtr<K, V>]> {
        self.kids.as_mut().map(|ks| ks.keys_mut())
    }

    fn val(&self, i: usize) -> &V {
        &self.elems.vals()[i]
    }

    fn val_mut(&mut self, i: usize) -> &mut V {
        &mut self.elems.vals_mut()[i]
    }

    fn clear(&mut self) {
        self.elems.clear();
        self.kids.take();
    }

    fn insert_kv(&mut self, idx: usize, kv: (K, V)) {
        self.elems.insert(idx, kv);
    }

    fn push_kv(&mut self, kv: (K, V)) {
        self.elems.push(kv);
    }

    fn remove_kv(&mut self, idx: usize) -> (K, V) {
        self.elems.remove(idx)
    }

    fn pop_kv(&mut self) -> (K, V) {
        self.elems.pop()
    }

    pub fn is_empty(&self) -> bool {
        self.elems.len == 0
    }

    pub fn get<Q>(&self, key: &Q) -> Option<(&K, &V)>
    where
        K: Borrow<Q> + Clone,
        V: Clone,
        Q: Ord + ?Sized,
    {
        for (i, k) in self.elems.keys().iter().enumerate() {
            match key.cmp(k.borrow()) {
                Less => return self.child(i)?.get(key),
                Equal => return Some((self.key(i), self.val(i))),
                Greater => (),
            }
        }

        self.last_child()?.get(key)
    }

    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q> + Clone,
        V: Clone,
        Q: Ord + ?Sized,
    {
        for (i, k) in self.elems.keys().iter().enumerate() {
            match key.cmp(k.borrow()) {
                Less => {
                    return Arc::make_mut(self.child_mut(i)?).get_mut(key);
                }
                Equal => return Some(self.val_mut(i)),
                Greater => (),
            }
        }

        Arc::make_mut(self.last_child_mut()?).get_mut(key)
    }

    pub fn insert(&mut self, kv: (K, V)) -> InsertResult<K, V>
    where
        K: Clone + Ord,
        V: Clone,
    {
        use InsertResult::*;

        let mut ub_x = 0;
        // let len = self.elems.len();
        for k in self.elems.keys().iter() {
            match kv.0.cmp(k) {
                Less => break,
                Equal => {
                    let old_k = replace(self.key_mut(ub_x), kv.0);
                    let old_v = replace(self.val_mut(ub_x), kv.1);
                    return Replaced(old_k, old_v);
                }
                Greater => (),
            }

            ub_x += 1;
        }

        assert!(ub_x < MAX_KIDS);

        // Recurse to the appropriate child if it exists (ie, we're not a leaf).
        // If we are a leaf, pretend that we visited a child and it resulted in
        // needing to insert a new separator at this level.
        let res = if let Some(kid) = self.child_mut(ub_x) {
            Arc::make_mut(kid).insert(kv)
        } else {
            Split(None, kv)
        };

        let Split(kid, kv) = res else { return res; };

        if self.elems.len() < MAX_OCCUPANCY {
            self.ins_no_split(ub_x, kid, kv)
        } else {
            self.ins_split(ub_x, kid, kv)
        }
    }

    fn ins_no_split(
        &mut self,
        idx: usize,
        kid: OptNodePtr<K, V>,
        kv: (K, V),
    ) -> InsertResult<K, V> {
        assert!(self.elems.len() < MAX_OCCUPANCY);
        assert!(idx <= self.elems.len());
        assert_eq!(self.kids.is_some(), kid.is_some());

        self.elems.insert(idx, kv);
        if let Some(kid) = kid {
            self.kids.as_mut().unwrap().insert(idx, (kid, ()));
        }

        InsertResult::Absorbed
    }

    // (conceptually) inserts kv into this node, making it overcrowded.  Select
    // and remove a dividing value and move elems below the divider into a new
    // node.  Return the new node and the divider.
    //
    // We retain the higher elements so that a parent node can insert the new
    // node and divider immediately before us.
    fn ins_split(
        &mut self,
        idx: usize,
        kid: OptNodePtr<K, V>,
        kv: (K, V),
    ) -> InsertResult<K, V> {
        assert!(self.elems.len() == MAX_OCCUPANCY);
        assert_eq!(self.kids.is_some(), kid.is_some());

        // Assume MAX_OCCUPANCY is 11.  We contain 11 kids and are adding a
        // twelfth.  After the split, we will have a dividing element, a node
        // with 5 elems, and a node with 6 elems.  As a heuristic, we divide the
        // elems so that an insert immediately after the current insert will go
        // into as small a node as possible.

        let mut lhs_arc = Arc::new(Node::new());
        let lhs = Arc::get_mut(&mut lhs_arc).unwrap();
        if self.kids.is_some() {
            lhs.kids = Some(Box::new(Kids::new()));
        }

        let div_kv: (K, V);
        let lhs_len: usize;
        let mut splitter = Splitter::new(self, lhs);
        if idx < MIN_OCCUPANCY {
            splitter.xfer(idx, idx);
            splitter.push_back_dst(kv, kid);
            splitter.xfer(MIN_OCCUPANCY - idx - 1, MIN_OCCUPANCY - idx);
            div_kv = splitter.pop_front_src();
            lhs_len = splitter.dst_len();
            splitter.finalize_src();
        } else if idx == MIN_OCCUPANCY || idx == MIN_OCCUPANCY + 1 {
            splitter.xfer(idx, idx);
            splitter.push_back_kid_dst(kid);
            div_kv = kv;
            lhs_len = splitter.dst_len();
            splitter.finalize_src();
        } else {
            splitter.xfer(MIN_OCCUPANCY + 1, MIN_OCCUPANCY + 2);
            div_kv = splitter.pop_front_src();
            lhs_len = splitter.dst_len();

            // repurpose our splitter for compacting & inserting val on rhs
            splitter.dst_keys = splitter.src_keys;
            splitter.dst_vals = splitter.src_vals;
            splitter.dst_kids = splitter.src_kids;
            splitter.dst_idx = 0;
            splitter.dst_kids_idx = 0;

            splitter.xfer(idx - MIN_OCCUPANCY - 2, idx - MIN_OCCUPANCY - 2);
            splitter.push_back_dst(kv, kid);
            splitter.xfer(MAX_OCCUPANCY - idx, MAX_KIDS - idx);
        }

        lhs.elems.len = lhs_len as InlineVecLen;
        self.elems.len = (MAX_OCCUPANCY - lhs_len) as InlineVecLen;

        if let Some(kids) = lhs.kids.as_mut() {
            kids.len = lhs.elems.len + 1;
        }

        if let Some(kids) = self.kids.as_mut() {
            kids.len = self.elems.len + 1;
        }

        InsertResult::Split(Some(lhs_arc), div_kv)
    }

    fn rot_lf(&mut self, idx: usize)
    where
        K: Clone,
        V: Clone,
    {
        assert!(self.kids.is_some(), "cannot rotate a leaf's children");

        // extract the new separator (k2 and v2) from child on right of idx
        let right = self.child_mut(idx + 1).unwrap();

        assert!(
            right.elems.len() > MIN_OCCUPANCY,
            "rot_lf from an impovershed child"
        );

        let right = Arc::make_mut(right);

        let (k2, v2) = right.elems.remove(0);
        let k1_to_k2 = right.kids.as_mut().map(|b| b.remove(0));

        // replace (and take) the old separator (k1 and v1)
        let k1 = std::mem::replace(self.key_mut(idx), k2);
        let v1 = std::mem::replace(self.val_mut(idx), v2);

        // push the old separator to the end of left
        let left = self.child_mut(idx).unwrap();
        assert!(left.elems.len() < MIN_OCCUPANCY, "rot_lf into a rich child");

        let left = Arc::make_mut(left);
        left.elems.push((k1, v1));
        if let Some(k1_to_k2) = k1_to_k2 {
            left.kids.as_mut().unwrap().push(k1_to_k2);
        }
    }

    fn rot_rt(&mut self, idx: usize)
    where
        K: Clone,
        V: Clone,
    {
        assert!(self.kids.is_some(), "cannot rotate a leaf's children");

        // idx holds the current separator, k1.  Get the pieces that will rotate
        // in to replace k1 & v1.
        let left = self.child_mut(idx).unwrap();
        assert!(
            left.elems.len() > MIN_OCCUPANCY,
            "rot_rt from impoverished child"
        );

        let left = Arc::make_mut(left);
        let (k0, v0) = left.elems.pop();
        let k0_to_k1 = left.kids.as_mut().map(|b| b.pop());

        // move k0 and v0 into the pivot position
        let k1 = replace(self.key_mut(idx), k0);
        let v1 = replace(self.val_mut(idx), v0);

        // move k1 and v1 down and to the right of the pivot
        let right = self.child_mut(idx + 1).unwrap();
        let right = Arc::make_mut(right);
        right.elems.insert(0, (k1, v1));
        if let Some(k0_to_k1) = k0_to_k1 {
            right.kids.as_mut().unwrap().insert(0, k0_to_k1);
        }
    }

    // merge the subtree self.index[at].0 and the one to its right
    fn merge_kids(&mut self, at: usize)
    where
        K: Clone,
        V: Clone,
    {
        assert!(self.kids.is_some());
        let kids = self.kids.as_mut().unwrap();

        // we destruct the rhs, moving its pieces to the lhs
        let rhs_rc = kids.remove(at + 1).0;
        let mut rhs_n = match Arc::try_unwrap(rhs_rc) {
            Ok(n) => n,
            Err(rc) => (*rc).clone(),
        };

        let mid_kv = self.elems.remove(at);

        let lhs_rc = &mut kids.keys_mut()[at];
        let lhs_mut = Arc::make_mut(lhs_rc);

        lhs_mut.elems.push(mid_kv);

        let lhs_len = lhs_mut.elems.len();
        let rhs_len = rhs_n.elems.len();

        unsafe {
            let lhs_keys = lhs_mut.elems.keys_mut().as_mut_ptr();
            let rhs_keys = rhs_n.elems.keys_mut().as_mut_ptr();
            copy(rhs_keys, lhs_keys.add(lhs_len), rhs_len);

            let lhs_vals = lhs_mut.elems.vals_mut().as_mut_ptr();
            let rhs_vals = rhs_n.elems.vals_mut().as_mut_ptr();
            copy(rhs_vals, lhs_vals.add(lhs_len), rhs_len);

            lhs_mut.elems.len += rhs_len as InlineVecLen;
            rhs_n.elems.len = 0;

            assert!(lhs_mut.elems.len() <= MAX_OCCUPANCY);

            if let Some(lhs_kids) = lhs_mut.kids.as_mut() {
                let rhs_kids = rhs_n.kids.as_mut().unwrap();

                let lhs_len = lhs_kids.len();
                let rhs_len = rhs_kids.len();

                let lhs_ptr = lhs_kids.keys.as_mut_ptr();
                let rhs_ptr = rhs_kids.keys.as_mut_ptr();
                copy(rhs_ptr, lhs_ptr.add(lhs_len), rhs_len);

                lhs_kids.len += rhs_kids.len;
                rhs_kids.len = 0;

                assert!(lhs_kids.len() <= MAX_KIDS);
            }
        }
    }

    // rebalances when the self.elems[at] is underpopulated
    fn rebal(&mut self, at: usize) -> IsUnderPop
    where
        K: Clone,
        V: Clone,
    {
        assert!(self.kids.is_some(), "cannot rebalance a leaf");
        assert!(self.child(at).unwrap().elems.len() == MIN_OCCUPANCY - 1);

        // We cannot merge if lhs::divider::rhs is too big
        //   - len(neighbor) + len(child(at)) + 1 > MAX_OCCUPANCY
        //   - len(neighbor) + MIN_OCCUPANCY - 1 + 1> MAX_OCCUPANCY
        //   - len(neighbor) > MAX_OCCUPANCY - MIN_OCCUPANCY
        // Traditionally, that limit is equal to MIN_OCCUPANCY + 1, but let's
        // anticipate future tweaks to MIN/MAX occupancy relation.
        const MERGE_THRESHOLD: usize = MAX_OCCUPANCY - MIN_OCCUPANCY;
        if at > 0 {
            if self.child(at - 1).unwrap().elems.len() > MERGE_THRESHOLD {
                self.rot_rt(at - 1);
            } else {
                self.merge_kids(at - 1);
            }
        } else if self.child(at + 1).unwrap().elems.len() > MERGE_THRESHOLD {
            self.rot_lf(at);
        } else {
            self.merge_kids(at);
        }

        IsUnderPop(self.elems.len() < MIN_OCCUPANCY)
    }

    pub fn pop_greatest(&mut self) -> ((K, V), IsUnderPop)
    where
        K: Clone,
        V: Clone,
    {
        if let Some(rt) = self.last_child_mut() {
            // self is a branch; recurse to the rightmost child
            let rt = Arc::make_mut(rt);
            let ret = rt.pop_greatest();
            if let (kv, IsUnderPop(true)) = ret {
                (kv, self.rebal(self.elems.len()))
            } else {
                ret
            }
        } else {
            // self is a leaf
            let kv = self.elems.pop();
            (kv, IsUnderPop(self.elems.len() < MIN_OCCUPANCY))
        }
    }

    pub fn remove<Q>(&mut self, key: &Q) -> Option<((K, V), IsUnderPop)>
    where
        K: Borrow<Q> + Clone,
        V: Clone,
        Q: Ord + ?Sized,
    {
        for (i, k) in self.elems.keys().iter().enumerate() {
            match key.cmp(k.borrow()) {
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
                    if self.kids.is_none() {
                        let old_kv = self.elems.remove(i);
                        return Some((
                            old_kv,
                            IsUnderPop(self.elems.len() < MIN_OCCUPANCY),
                        ));
                    }

                    let lt_k = self.child_mut(i).unwrap();
                    let lt_k = Arc::make_mut(lt_k);
                    let (kv, is_under_pop) = lt_k.pop_greatest();
                    let old_kv = (
                        replace(self.key_mut(i), kv.0),
                        replace(self.val_mut(i), kv.1),
                    );
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
        let gt_k = self.last_child_mut()?;
        let gt_k = Arc::make_mut(gt_k);
        let (kv, IsUnderPop(is_under_pop)) = gt_k.remove(key)?;
        Some((
            kv,
            IsUnderPop(is_under_pop && self.rebal(self.elems.len()).0),
        ))
    }

    pub fn chk(&self) -> usize
    where
        K: Clone + Ord,
    {
        self.chk_aux(None, 1).2
    }

    fn chk_aux(
        &self,
        mut greatest: Option<K>,
        min_len: usize,
    ) -> (Option<K>, usize, usize)
    where
        K: Clone + Ord,
    {
        assert!(self.elems.len() >= min_len);
        assert!(self.elems.len() <= MAX_OCCUPANCY);
        assert!(
            self.kids.is_none()
                || self.kids.as_ref().unwrap().len() == self.elems.len() + 1
        );

        let mut ht = 0;
        let mut len = 0;

        if let Some(c) = self.child(0) {
            let (g, h, l) = c.chk_aux(greatest, MIN_OCCUPANCY);
            greatest = g;
            ht = h;
            len = l;
        }

        for i in 0..self.elems.len() {
            len += 1;

            let k = self.key(i);
            if let Some(g) = greatest.as_ref() {
                assert!(g < k)
            }

            greatest = Some(k.clone());
            if let Some(c) = self.child(i + 1) {
                let (g, h, l) = c.chk_aux(greatest, MIN_OCCUPANCY);
                assert_eq!(h, ht);
                len += l;
                greatest = g;
            }
        }

        (greatest, ht + 1, len)
    }
}

impl<K, V> std::fmt::Debug for Node<K, V>
where
    K: std::fmt::Debug,
    V: std::fmt::Debug,
{
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> Result<(), std::fmt::Error> {
        let kids = self.kids.as_ref().map(|ks| ks.keys()).unwrap_or(&[]);
        f.debug_struct("Node")
            .field("elems", &self.elems)
            .field("kids", &kids)
            .finish()
    }
}

pub type NodePtr<K, V> = Arc<Node<K, V>>;
pub type OptNodePtr<K, V> = Option<NodePtr<K, V>>;

pub enum InsertResult<K, V> {
    Absorbed,
    Replaced(K, V),
    Split(OptNodePtr<K, V>, (K, V)),
}

struct Splitter<'a, K, V> {
    src_keys: *mut MaybeUninit<K>,
    src_vals: *mut MaybeUninit<V>,
    src_idx: usize,

    dst_keys: *mut MaybeUninit<K>,
    dst_vals: *mut MaybeUninit<V>,
    dst_idx: usize,

    src_kids: Option<*mut MaybeUninit<NodePtr<K, V>>>,
    src_kids_idx: usize,

    dst_kids: Option<*mut MaybeUninit<NodePtr<K, V>>>,
    dst_kids_idx: usize,

    marker: std::marker::PhantomData<&'a mut (K, V)>,
}

impl<'a, K, V> Splitter<'a, K, V> {
    fn new(src: &'a mut Node<K, V>, dst: &'a mut Node<K, V>) -> Self {
        Self {
            src_keys: src.elems.keys.as_mut_ptr(),
            src_vals: src.elems.vals.as_mut_ptr(),
            src_idx: 0,

            dst_keys: dst.elems.keys.as_mut_ptr(),
            dst_vals: dst.elems.vals.as_mut_ptr(),
            dst_idx: 0,

            src_kids: src.kids.as_mut().map(|ks| ks.keys.as_mut_ptr()),
            src_kids_idx: 0,

            dst_kids: dst.kids.as_mut().map(|ks| ks.keys.as_mut_ptr()),
            dst_kids_idx: 0,

            marker: PhantomData,
        }
    }

    fn xfer(&mut self, cnt_elems: usize, cnt_kids: usize) {
        assert!(self.src_idx + cnt_elems <= MAX_OCCUPANCY);
        assert!(self.dst_idx + cnt_elems <= MAX_OCCUPANCY);
        assert!(self.src_kids_idx + cnt_kids <= MAX_KIDS);
        assert!(self.dst_kids_idx + cnt_kids <= MAX_KIDS);

        unsafe {
            copy(
                self.src_keys.add(self.src_idx),
                self.dst_keys.add(self.dst_idx),
                cnt_elems,
            );

            copy(
                self.src_vals.add(self.src_idx),
                self.dst_vals.add(self.dst_idx),
                cnt_elems,
            );

            if self.src_kids.is_some() {
                let src_kids = self.src_kids.unwrap();
                let dst_kids = self.dst_kids.unwrap();

                copy(
                    src_kids.add(self.src_kids_idx),
                    dst_kids.add(self.dst_kids_idx),
                    cnt_kids,
                );
            }
        }

        self.src_idx += cnt_elems;
        self.dst_idx += cnt_elems;
        self.src_kids_idx += cnt_kids;
        self.dst_kids_idx += cnt_kids;
    }

    fn push_back_dst(&mut self, kv: (K, V), kid: OptNodePtr<K, V>) {
        assert!(self.dst_idx < MAX_OCCUPANCY);
        unsafe {
            self.dst_keys
                .add(self.dst_idx)
                .write(MaybeUninit::new(kv.0));
            self.dst_vals
                .add(self.dst_idx)
                .write(MaybeUninit::new(kv.1));
        }
        self.dst_idx += 1;

        self.push_back_kid_dst(kid)
    }

    fn push_back_kid_dst(&mut self, kid: OptNodePtr<K, V>) {
        assert!(self.dst_idx < MAX_OCCUPANCY);
        if let Some(kid) = kid {
            unsafe {
                self.dst_kids
                    .unwrap()
                    .add(self.dst_kids_idx)
                    .write(MaybeUninit::new(kid));
            }
        }
        self.dst_kids_idx += 1;
    }

    fn pop_front_src(&mut self) -> (K, V) {
        assert!(self.src_idx < MAX_OCCUPANCY);
        let key;
        let val;
        unsafe {
            key = self.src_keys.add(self.src_idx).read().assume_init();
            val = self.src_vals.add(self.src_idx).read().assume_init();
        }
        self.src_idx += 1;
        (key, val)
    }

    fn finalize_src(&mut self) {
        unsafe {
            copy(
                self.src_keys.add(self.src_idx),
                self.src_keys,
                MAX_OCCUPANCY - self.src_idx,
            );

            copy(
                self.src_vals.add(self.src_idx),
                self.src_vals,
                MAX_OCCUPANCY - self.src_idx,
            );

            if self.src_kids.is_some() {
                copy(
                    self.src_kids.unwrap().add(self.src_kids_idx),
                    self.src_kids.unwrap(),
                    MAX_KIDS - self.src_kids_idx,
                );
            }
        }

        // catch further reads from src:
        self.src_idx = MAX_OCCUPANCY;
        self.src_kids_idx = MAX_KIDS;
    }

    fn dst_len(&self) -> usize {
        self.dst_idx
    }
}

pub enum IterAction<E, N> {
    Return(E),  // Return element E
    Descend(N), // Descend to node N
}

pub trait NodeIterator<E, N>:
    DoubleEndedIterator<Item = IterAction<E, N>>
{
    fn new(n: N) -> Self;
}

pub struct NodeIter<KI, VI, NI> {
    keys: KI,
    vals: VI,
    kids: NI,

    is_next_elem: bool,
    is_next_back_elem: bool,
}

impl<KI, VI, NI> Iterator for NodeIter<KI, VI, NI>
where
    KI: DoubleEndedIterator,
    VI: DoubleEndedIterator,
    NI: DoubleEndedIterator + ExactSizeIterator,
{
    type Item = IterAction<(KI::Item, VI::Item), NI::Item>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.is_next_elem {
            self.is_next_elem = self.kids.len() == 0;
            let k = self.keys.next()?;
            let v = self.vals.next()?;
            Some(IterAction::Return((k, v)))
        } else {
            self.is_next_elem = true;
            self.kids.next().map(IterAction::Descend)
        }
    }
}

impl<KI, VI, NI> DoubleEndedIterator for NodeIter<KI, VI, NI>
where
    KI: DoubleEndedIterator,
    VI: DoubleEndedIterator,
    NI: DoubleEndedIterator + ExactSizeIterator,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.is_next_back_elem {
            self.is_next_back_elem = self.kids.len() == 0;
            let k = self.keys.next_back()?;
            let v = self.vals.next_back()?;
            Some(IterAction::Return((k, v)))
        } else {
            self.is_next_back_elem = true;
            self.kids.next_back().map(IterAction::Descend)
        }
    }
}

type SliceIter<'a, T> = std::slice::Iter<'a, T>;
type SliceIterMut<'a, T> = std::slice::IterMut<'a, T>;

pub type NodeIterRef<'a, K, V> =
    NodeIter<SliceIter<'a, K>, SliceIter<'a, V>, SliceIter<'a, NodePtr<K, V>>>;

pub type NodeIterMut<'a, K, V> = NodeIter<
    SliceIter<'a, K>,
    SliceIterMut<'a, V>,
    SliceIterMut<'a, NodePtr<K, V>>,
>;

impl<'a, K, V> NodeIterator<(&'a K, &'a V), &'a NodePtr<K, V>>
    for NodeIterRef<'a, K, V>
{
    fn new(n: &'a NodePtr<K, V>) -> Self {
        let kids = if let Some(kids) = n.kids() {
            kids.iter()
        } else {
            [].iter()
        };

        let has_kids = kids.len() > 0;
        Self {
            keys: n.elems.keys().iter(),
            vals: n.elems.vals().iter(),
            kids,
            is_next_elem: !has_kids,
            is_next_back_elem: !has_kids,
        }
    }
}

impl<'a, K, V> NodeIterator<(&'a K, &'a mut V), &'a mut NodePtr<K, V>>
    for NodeIterMut<'a, K, V>
where
    K: Clone,
    V: Clone,
{
    fn new(n: &'a mut NodePtr<K, V>) -> Self {
        let n = Arc::make_mut(n);

        let kids = if let Some(kids) = n.kids.as_mut() {
            kids.keys_mut().iter_mut()
        } else {
            [].iter_mut()
        };
        let has_kids = kids.len() > 0;

        let (keys, vals) = n.elems.elems_mut();

        Self {
            keys: keys.iter(),
            vals: vals.iter_mut(),
            kids,
            is_next_elem: !has_kids,
            is_next_back_elem: !has_kids,
        }
    }
}

pub enum OwnedOrLeased<K, V> {
    Owned(Node<K, V>),
    Leased(NodePtr<K, V>),
}

pub struct NodeIntoIter<K, V> {
    n: OwnedOrLeased<K, V>,
    next_elem_x: usize,
    end_elems: usize,
    next_kid_x: usize,
    end_kids: usize,
}

impl<K, V> Iterator for NodeIntoIter<K, V>
where
    K: Clone,
    V: Clone,
{
    type Item = IterAction<(K, V), OwnedOrLeased<K, V>>;

    fn next(&mut self) -> Option<Self::Item> {
        use IterAction::*;
        use OwnedOrLeased::*;

        if self.next_kid_x < self.end_kids
            && self.next_kid_x <= self.next_elem_x
        {
            let kid = match &mut self.n {
                Owned(n) => unsafe {
                    n.kids.as_mut().unwrap().keys[self.next_kid_x]
                        .assume_init_read()
                },
                Leased(arc) => arc.child(self.next_kid_x).unwrap().clone(),
            };
            self.next_kid_x += 1;

            // Could we switch from a "Leased" node to an "Owned" node here?
            // No: if we're leased, we just cloned kid & try_unwrap will fail.
            match Arc::try_unwrap(kid) {
                Ok(n) => Some(Descend(Owned(n))),
                Err(arc) => Some(Descend(Leased(arc))),
            }
        } else if self.next_elem_x < self.end_elems {
            let kv = match &mut self.n {
                Owned(n) => unsafe {
                    let k = n.elems.keys[self.next_elem_x].assume_init_read();
                    let v = n.elems.vals[self.next_elem_x].assume_init_read();
                    (k, v)
                },
                Leased(arc) => (
                    arc.key(self.next_elem_x).clone(),
                    arc.val(self.next_elem_x).clone(),
                ),
            };
            self.next_elem_x += 1;

            Some(Return(kv))
        } else {
            None
        }
    }
}

impl<K, V> DoubleEndedIterator for NodeIntoIter<K, V>
where
    K: Clone,
    V: Clone,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        use IterAction::*;
        use OwnedOrLeased::*;

        if self.next_kid_x < self.end_kids && self.end_elems < self.end_kids {
            self.end_kids -= 1;
            let kid = match &mut self.n {
                Owned(n) => unsafe {
                    n.kids.as_mut().unwrap().keys[self.end_kids]
                        .assume_init_read()
                },
                Leased(arc) => arc.child(self.end_kids).unwrap().clone(),
            };

            // Could we switch from a "Leased" node to an "Owned" node here?
            // No: if we're leased, we just cloned kid & try_unwrap will fail.
            match Arc::try_unwrap(kid) {
                Ok(n) => Some(Descend(Owned(n))),
                Err(arc) => Some(Descend(Leased(arc))),
            }
        } else if self.next_elem_x < self.end_elems {
            self.next_elem_x -= 1;
            let kv = match &mut self.n {
                Owned(n) => unsafe {
                    let k = n.elems.keys[self.end_elems].assume_init_read();
                    let v = n.elems.vals[self.end_elems].assume_init_read();
                    (k, v)
                },
                Leased(arc) => (
                    arc.key(self.end_elems).clone(),
                    arc.val(self.end_elems).clone(),
                ),
            };

            Some(Return(kv))
        } else {
            None
        }
    }
}

impl<K, V> NodeIterator<(K, V), OwnedOrLeased<K, V>> for NodeIntoIter<K, V>
where
    K: Clone,
    V: Clone,
{
    fn new(n: OwnedOrLeased<K, V>) -> Self {
        use OwnedOrLeased::*;
        let (end_elems, end_kids) = match &n {
            Leased(n) => (n.elems.len, n.kids().map(|x| x.len()).unwrap_or(0)),
            Owned(n) => (n.elems.len, n.kids().map(|x| x.len()).unwrap_or(0)),
        };

        Self {
            n,
            next_elem_x: 0,
            end_elems: end_elems as usize,
            next_kid_x: 0,
            end_kids,
        }
    }
}

// NodeIntoIter may own a node /and/ have given away some of its held elements.
// We need to ensure that the remaining elements are dropped & just once.
impl<K, V> Drop for NodeIntoIter<K, V> {
    fn drop(&mut self) {
        // if we don't own the node, we don't need to clean it up
        let OwnedOrLeased::Owned(n) = &mut self.n else { return; };

        for i in self.next_elem_x..self.end_elems {
            unsafe {
                n.elems.keys[i].assume_init_drop();
                n.elems.vals[i].assume_init_drop();
            }
        }
        n.elems.len = 0;

        let Some(mut kids) = n.kids.take() else { return; };
        for i in self.next_kid_x..self.end_kids {
            unsafe {
                kids.keys[i].assume_init_drop();
            }
        }
        kids.len = 0;
    }
}

pub struct InnerIter<E, N, NI: NodeIterator<E, N>> {
    work: VecDeque<NI>,
    len: usize,
    marker: PhantomData<IterAction<E, N>>,
}

impl<E, N, NI: NodeIterator<E, N>> InnerIter<E, N, NI> {
    pub fn new(root: Option<N>, len: usize) -> Self {
        let mut work = VecDeque::new();
        if let Some(root) = root {
            work.push_back(NI::new(root));
        }

        Self {
            work,
            len,
            marker: PhantomData,
        }
    }
}

impl<E, N, NI: NodeIterator<E, N>> Iterator for InnerIter<E, N, NI> {
    type Item = E;

    fn next(&mut self) -> Option<Self::Item> {
        use IterAction::*;
        while let Some(w) = self.work.front_mut() {
            match w.next() {
                Some(Return(kv)) => {
                    self.len -= 1;
                    return Some(kv);
                }

                Some(Descend(n)) => self.work.push_front(NI::new(n)),

                None => {
                    self.work.pop_front();
                }
            }
        }

        None
    }
}

impl<E, N, NI: NodeIterator<E, N>> DoubleEndedIterator for InnerIter<E, N, NI> {
    fn next_back(&mut self) -> Option<Self::Item> {
        use IterAction::*;
        while let Some(w) = self.work.back_mut() {
            match w.next() {
                Some(Return(kv)) => {
                    self.len -= 1;
                    return Some(kv);
                }

                Some(Descend(n)) => self.work.push_back(NI::new(n)),

                None => {
                    self.work.pop_back();
                }
            }
        }

        None
    }
}

impl<E, N, NI: NodeIterator<E, N>> ExactSizeIterator for InnerIter<E, N, NI> {
    fn len(&self) -> usize {
        self.len
    }
}

impl<E, N, NI: NodeIterator<E, N>> FusedIterator for InnerIter<E, N, NI> {}

pub type Iter<'a, K, V> =
    InnerIter<(&'a K, &'a V), &'a NodePtr<K, V>, NodeIterRef<'a, K, V>>;

pub type IterMut<'a, K, V> =
    InnerIter<(&'a K, &'a mut V), &'a mut NodePtr<K, V>, NodeIterMut<'a, K, V>>;

pub type IntoIter<K, V> =
    InnerIter<(K, V), OwnedOrLeased<K, V>, NodeIntoIter<K, V>>;

pub fn new_into_iter<K: Clone, V: Clone>(
    root: OptNodePtr<K, V>,
    len: usize,
) -> IntoIter<K, V> {
    let root = root.map(|arc| match Arc::try_unwrap(arc) {
        Ok(n) => OwnedOrLeased::Owned(n),
        Err(arc) => OwnedOrLeased::Leased(arc),
    });

    IntoIter::new(root, len)
}

#[cfg(test)]
pub(crate) mod test {
    use super::*;

    use proptest::prelude::*;

    // Generates a tree of ht 'h' filled with (0, 0).  Because all the keys are
    // the same, it violates the sortedness invariant of the tree.
    fn btree_skel_strat(ht: usize) -> impl Strategy<Value = NodePtr<u32, u32>> {
        (MIN_OCCUPANCY..=MAX_OCCUPANCY)
            .prop_flat_map(move |len| {
                (
                    Just(len),
                    prop::collection::vec(
                        btree_skel_strat(ht - 1),
                        if ht > 1 { len + 1 } else { 0 },
                    ),
                )
            })
            .prop_map(|(len, ks)| {
                let mut n = Node::new();
                for _ in 0..len {
                    n.elems.push((0, 0));
                }

                if !ks.is_empty() {
                    assert!(ks.len() == len + 1);
                    let mut kids = Box::new(Kids::new());
                    for c in ks {
                        kids.push((c, ()));
                    }
                    n.kids = Some(kids);
                }

                Arc::new(n)
            })
            .boxed()
    }

    pub fn btree_strat(
        ht: usize,
    ) -> impl Strategy<Value = (NodePtr<u32, u32>, usize)> {
        fn assign_elems<I: Iterator<Item = u32>>(
            n: &mut NodePtr<u32, u32>,
            gen: &mut I,
        ) {
            let n = Arc::get_mut(n).unwrap();
            let len = n.elems.len as usize;
            for i in 0..len {
                if let Some(c) = n.child_mut(i) {
                    assign_elems(c, gen);
                }

                let x = gen.next().unwrap();
                *n.key_mut(i) = x * 2; // mul by const to create gaps
                *n.val_mut(i) = x;
            }

            if let Some(c) = n.last_child_mut() {
                assign_elems(c, gen);
            }
        }

        btree_skel_strat(ht).prop_map(|mut n| {
            let mut ns = std::iter::successors(Some(0), |n| Some(n + 1));
            assign_elems(&mut n, &mut ns);
            (n, ns.next().unwrap() as usize)
        })
    }
}
