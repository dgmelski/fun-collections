#![warn(missing_docs)]
use std::borrow::Borrow;
use std::cmp::Ordering::*;
use std::collections::VecDeque;
use std::fmt::{Debug, Formatter};
use std::iter::FusedIterator;
use std::marker::PhantomData;
use std::mem::replace;
use std::ops::{Bound, RangeBounds};
use std::sync::Arc;

use super::{Entry, Map};

pub mod avl_set;

type OptNode<K, V> = Option<Arc<Node<K, V>>>;
struct IsShorter(bool);
struct IsTaller(bool);

#[cfg(test)]
#[macro_export]
macro_rules! chk_node {
    ( $x:expr ) => {{
        let n = $x;
        chk(&n, None);
        n
    }};
}

#[cfg(test)]
#[macro_export]
macro_rules! chk_map {
    ( $x:expr ) => {{
        let n = $x;
        let chk_len = chk(&n.root, None).0;
        assert_eq!(chk_len, n.len);
        n
    }};
}

#[cfg(not(test))]
macro_rules! chk_node {
    ( $x:expr ) => {{
        $x
    }};
}

#[cfg(not(test))]
macro_rules! chk_map {
    ( $x: expr ) => {{
        $x
    }};
}

#[derive(Clone)]
struct Node<K, V> {
    key: K,
    val: V,
    left_ht: i8,
    right_ht: i8,
    left: OptNode<K, V>,
    right: OptNode<K, V>,
}

impl<K, V> Node<K, V> {
    fn new(key: K, val: V, left: OptNode<K, V>, right: OptNode<K, V>) -> Self {
        Node {
            key,
            val,
            left_ht: height(&left),
            right_ht: height(&right),
            left,
            right,
        }
    }

    fn opt_new(
        k: K,
        v: V,
        l: OptNode<K, V>,
        r: OptNode<K, V>,
    ) -> OptNode<K, V> {
        Some(Arc::new(Self::new(k, v, l, r)))
    }

    // Returns the "balance factor" of the node
    fn bal(&self) -> i8 {
        self.right_ht - self.left_ht
    }

    // Is the given node balanced, that is -1 <= self.bal() <= 1 ?
    fn is_bal(&self) -> bool {
        // single-branch range inclusion check; requires unsigned wrap around
        ((self.bal() + 1) as u8) <= 2
    }

    fn height(&self) -> i8 {
        self.left_ht.max(self.right_ht) + 1
    }

    fn set_left(&mut self, rt: OptNode<K, V>) {
        self.left_ht = height(&rt);
        self.left = rt;
    }

    fn set_right(&mut self, rt: OptNode<K, V>) {
        self.right_ht = height(&rt);
        self.right = rt;
    }

    fn for_each<F>(&self, g: &mut F)
    where
        F: FnMut((&K, &V)),
    {
        if let Some(rc) = self.left.as_ref() {
            rc.for_each(g);
        }

        g((&self.key, &self.val));

        if let Some(rc) = self.right.as_ref() {
            rc.for_each(g);
        }
    }

    fn for_each_mut<F>(&mut self, g: &mut F)
    where
        K: Clone,
        V: Clone,
        F: FnMut((&K, &mut V)),
    {
        if let Some(rc) = self.left.as_mut() {
            Arc::make_mut(rc).for_each_mut(g);
        }

        g((&self.key, &mut self.val));

        if let Some(rc) = self.right.as_mut() {
            Arc::make_mut(rc).for_each_mut(g);
        }
    }
}

impl<K: Ord, V> Node<K, V> {
    #[cfg(test)]
    fn chk(&self, greatest: Option<&K>) -> (usize, Option<&K>) {
        // is our node in order with left-side ancestors?
        assert!(greatest.iter().all(|&k| k < &self.key));

        // do we know the heights of our children?
        assert_eq!(height(&self.left), self.left_ht);
        assert_eq!(height(&self.right), self.right_ht);

        // are we balanced?
        assert!(self.is_bal());

        // are our left descendents okay?
        let (lf_len, greatest) = chk(&self.left, greatest);

        // are our left descendents all less than us?
        assert!(greatest.iter().all(|&k| k < &self.key));

        // are our right descendents okay?
        let (rt_len, greatest) = chk(&self.right, Some(&self.key));

        (lf_len + rt_len + 1, greatest)
    }

    #[allow(dead_code)]
    #[cfg(not(test))]
    fn chk(&self) {}
}

impl<K: Debug, V: Debug> Debug for Node<K, V> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "(ht: {} {{{:?}: {:?}}} ",
            self.height(),
            self.key,
            self.val
        ))?;

        match &self.left {
            None => f.write_str(".")?,
            Some(lf) => lf.fmt(f)?,
        }

        f.write_str(" ")?;

        match &self.right {
            None => f.write_str(".")?,
            Some(rt) => rt.fmt(f)?,
        }

        f.write_str(")")
    }
}

/// A map from keys to values sorted by key.
///
/// We aim for the API to be compatible with the (stable) API of
/// [`std::collections::BTreeMap`].
///
/// Internally, the map uses 'persistent' AVL trees.  [AVL
/// trees](https://en.wikipedia.org/wiki/AVL_tree) were the first self-balancing
/// binary tree.  The trees are persistent in that trees cloned from a common
/// ancestor will share nodes until they are updated.  When a map is updated, it
/// creates a new tree with the update, though the tree prior to the update may
/// continue to exist and be used by other maps.
///
/// AvlMaps shine when (1) you need many related maps that were cloned from
/// common ancestors and (2) cloning of keys and/or values is expensive.  When
/// you need cloned maps but cloning of entries is relatively cheap,
/// [`lazy_clone_collections::BTreeMap`s](crate.btree.BTreeMap) will often give
/// better performance. BTrees store more entries in each node, leading to
/// shallower trees.  Updates need to clone fewer nodes, but clone more entries
/// for each node cloned.
#[derive(Clone)]
pub struct AvlMap<K, V> {
    len: usize,
    root: OptNode<K, V>,
}

impl<K: Debug, V: Debug> Debug for AvlMap<K, V> {
    /// Format and AvlMap using "map" notation.
    ///
    /// # Examples
    /// ```
    /// use lazy_clone_collections::AvlMap;
    ///
    /// let m = AvlMap::from([(0, 'a'), (1, 'b')]);
    /// assert_eq!(format!("{:?}", m), r#"AvlMap({0: 'a', 1: 'b'})"#);
    /// ```
    fn fmt(&self, fmt: &mut Formatter<'_>) -> std::fmt::Result {
        fmt.write_str("AvlMap(")?;
        fmt.debug_map().entries(self.iter()).finish()?;
        fmt.write_str(")")
    }
}

impl<K, V> Default for AvlMap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V> PartialEq for AvlMap<K, V>
where
    K: PartialEq,
    V: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len()
            && self.iter().zip(other.iter()).all(|(x, y)| x == y)
    }
}

impl<K: Eq, V: Eq> Eq for AvlMap<K, V> {}

impl<K, V> PartialOrd for AvlMap<K, V>
where
    K: PartialOrd,
    V: PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.iter().partial_cmp(other.iter())
    }
}

impl<K: Ord, V: Ord> Ord for AvlMap<K, V> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.iter().cmp(other.iter())
    }
}

impl<K, V> std::hash::Hash for AvlMap<K, V>
where
    K: std::hash::Hash,
    V: std::hash::Hash,
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.len.hash(state); // increase entropy?
        self.for_each(|(k, v)| {
            k.hash(state);
            v.hash(state);
        });
    }
}

impl<K, Q, V> std::ops::Index<&Q> for AvlMap<K, V>
where
    K: Borrow<Q>,
    Q: Ord + ?Sized,
{
    type Output = V;

    fn index(&self, index: &Q) -> &Self::Output {
        match self.get(index) {
            Some(v) => v,
            None => panic!("Key not found in AvlMap"),
        }
    }
}

fn height<K, V>(opt_node: &OptNode<K, V>) -> i8 {
    opt_node.as_ref().map_or(0, |rc| rc.height())
}

fn len<K, V>(opt_node: &OptNode<K, V>) -> usize {
    opt_node
        .as_ref()
        .map_or(0, |r| len(&r.left) + 1 + len(&r.right))
}

#[cfg(test)]
fn chk<'a, K: Ord, V>(
    opt_node: &'a OptNode<K, V>,
    greatest: Option<&'a K>,
) -> (usize, Option<&'a K>) {
    match opt_node.as_ref() {
        None => (0, greatest),
        Some(n) => n.chk(greatest),
    }
}

// prerequisites:
//   - opt_node.is_some()
//   - opt_node.unwrap().get_mut().is_some() (the node is uniquely owned)
fn take_node<K: Clone, V: Clone>(opt_node: &mut OptNode<K, V>) -> Node<K, V> {
    let old_rc = opt_node.take().unwrap();
    match Arc::try_unwrap(old_rc) {
        Ok(n) => n,
        Err(_) => panic!("Attempt to take a shared node"),
    }
}

fn rot_lf<K: Clone, V: Clone>(root: &mut OptNode<K, V>) -> IsShorter {
    // We want the following transformation:
    //    a(x, b(y, z)))   =>   b(a(x, y), z)
    // x and z retain the same parents.

    let mut a_opt = root.take();
    let a_rc = a_opt.as_mut().unwrap();
    let a = Arc::make_mut(a_rc);

    let mut b_opt = a.right.take();
    let b_rc = b_opt.as_mut().unwrap();
    let b = Arc::make_mut(b_rc);

    // if b is balanced, the rotation will make a shorter tree
    let b_was_bal = b.bal() == 0;

    // move y from b to a
    a.right_ht = b.left_ht;
    a.right = b.left.take();

    // make a be b's left child
    b.left_ht = a.height();
    b.left = a_opt;

    // install b as the new root
    *root = b_opt;

    IsShorter(!b_was_bal)
}

fn rot_rt_lf<K: Clone, V: Clone>(root: &mut OptNode<K, V>) -> IsShorter {
    // We want the following transformation:
    //    a(x, b(c(y, z), w))   =>   c(a(x, y), b(z, w))
    // x and w retain the same parents.

    let mut a_opt = root.take();
    let a_rc = a_opt.as_mut().unwrap();
    let a = Arc::make_mut(a_rc);

    let mut b_opt = a.right.take();
    let b_rc = b_opt.as_mut().unwrap();
    let b = Arc::make_mut(b_rc);

    let mut c_opt = b.left.take();
    let c_rc = c_opt.as_mut().unwrap();
    let c = Arc::make_mut(c_rc);

    // We need to take care not to overwrite any links before taking them.
    // With the unlinks we've done, we have
    //   a(x, None)
    //   b(None, w)
    //   c(y, z)

    // move c's children to a and b
    a.right_ht = c.left_ht;
    a.right = c.left.take();

    b.left_ht = c.right_ht;
    b.left = c.right.take();

    // move a and b into c
    c.left_ht = a.height();
    c.left = a_opt.take();

    c.right_ht = b.height();
    c.right = b_opt.take();

    // install c as the new root
    *root = c_opt;

    // this rebalance always makes the tree shorter
    IsShorter(true)
}

fn rot_rt<K: Clone, V: Clone>(root: &mut OptNode<K, V>) -> IsShorter {
    // We want the following transformation:
    //    a(b(x, y), z)   =>   b(x, a(y, z))
    // x and z retain the same parents.

    let mut a_opt = root.take();
    let a_rc = a_opt.as_mut().unwrap();
    let a = Arc::make_mut(a_rc);

    let mut b_opt = a.left.take();
    let b_rc = b_opt.as_mut().unwrap();
    let b = Arc::make_mut(b_rc);

    let b_was_bal = b.bal() == 0;

    // We have
    //   a(None, z)
    //   b(x, y)

    // move y from b to a
    a.left_ht = b.right_ht;
    a.left = b.right.take();

    // move a into b
    b.right_ht = a.height();
    b.right = a_opt.take();

    // install b as the new root
    *root = b_opt;

    IsShorter(!b_was_bal)
}

fn rot_lf_rt<K: Clone, V: Clone>(root: &mut OptNode<K, V>) -> IsShorter {
    // We want the following transformation:
    //    a(b(x,c(y,z)),w)   =>   c(b(x,y),a(z,w))
    // x and w retain the same parents.

    let mut a_opt = root.take();
    let a_rc = a_opt.as_mut().unwrap();
    let a = Arc::make_mut(a_rc);

    let mut b_opt = a.left.take();
    let b_rc = b_opt.as_mut().unwrap();
    let b = Arc::make_mut(b_rc);

    let mut c_opt = b.right.take();
    let c_rc = c_opt.as_mut().unwrap();
    let c = Arc::make_mut(c_rc);

    // We have:
    //   a(None, w)
    //   b(x, None)
    //   c(y, z)

    b.right_ht = c.left_ht;
    b.right = c.left.take(); // => b(x, y), c(None, z)

    a.left_ht = c.right_ht;
    a.left = c.right.take(); // => a(z, w), c(None, None)

    c.left_ht = b.height();
    c.left = b_opt; // => c(b(x, y), None)

    c.right_ht = a.height();
    c.right = a_opt; // => c(b(x, y), a(z, w))

    *root = c_opt;

    IsShorter(true)
}

// rebalance by "shifting height" from left to right
fn rebal_lf_to_rt<K, V>(root: &mut OptNode<K, V>) -> IsShorter
where
    K: Clone,
    V: Clone,
{
    let n = Arc::get_mut(root.as_mut().unwrap()).unwrap();

    if n.left.as_ref().unwrap().bal() <= 0 {
        rot_rt(root)
    } else {
        rot_lf_rt(root)
    }
}

// rebalance by "shifting height" from right to left
fn rebal_rt_to_lf<K, V>(root: &mut OptNode<K, V>) -> IsShorter
where
    K: Clone,
    V: Clone,
{
    let n = Arc::get_mut(root.as_mut().unwrap()).unwrap();

    if n.right.as_ref().unwrap().bal() >= 0 {
        rot_lf(root)
    } else {
        rot_rt_lf(root)
    }
}

// Inserts (k,v) into the map rooted at root and returns the replaced value and
// whether the updated node is taller as a result of insertion.
fn ins<K, V>(root: &mut OptNode<K, V>, k: K, v: V) -> (Option<V>, IsTaller)
where
    K: Clone + Ord,
    V: Clone,
{
    let n = match root.as_mut() {
        None => {
            *root = Some(Arc::new(Node::new(k, v, None, None)));
            return (None, IsTaller(true)); // *** EARLY RETURN ***
        }

        Some(rc) => Arc::make_mut(rc),
    };

    match k.cmp(&n.key) {
        Equal => (Some(std::mem::replace(&mut n.val, v)), IsTaller(false)),

        Less => {
            let (old_v, is_taller) = ins(&mut n.left, k, v);
            n.left_ht += is_taller.0 as i8;
            if is_taller.0 && n.bal() < -1 {
                rebal_lf_to_rt(root);
                (old_v, IsTaller(false))
            } else {
                (old_v, IsTaller(is_taller.0 && n.bal() < 0))
            }
        }

        Greater => {
            let (old_v, is_taller) = ins(&mut n.right, k, v);
            n.right_ht += is_taller.0 as i8;
            if is_taller.0 && n.bal() > 1 {
                rebal_rt_to_lf(root);
                (old_v, IsTaller(false))
            } else {
                (old_v, IsTaller(is_taller.0 && n.bal() > 0))
            }
        }
    }
}

// helper function for remove that removes the leftmost node and returns both
// its key and value and whether or not the removal made the tree smaller.
fn rm_leftmost<K, V>(root: &mut OptNode<K, V>) -> (Option<(K, V)>, IsShorter)
where
    K: Clone,
    V: Clone,
{
    let n = match root.as_mut() {
        None => return (None, IsShorter(false)), // *** EARLY RETURN ***
        Some(rc) => Arc::make_mut(rc),
    };

    if n.left.is_some() {
        let (kv, is_shorter) = rm_leftmost(&mut n.left);
        n.left_ht -= is_shorter.0 as i8;
        if is_shorter.0 && n.bal() > 1 {
            (kv, rebal_rt_to_lf(root))
        } else {
            (kv, IsShorter(is_shorter.0 && n.bal() == 0))
        }
    } else {
        let old_n = take_node(root);
        *root = old_n.right;
        (Some((old_n.key, old_n.val)), IsShorter(true))
    }
}

// helper function for remove that removes the rightmost node and returns both
// its key and value and whether or not the removal made the tree smaller.
fn rm_rightmost<K, V>(root: &mut OptNode<K, V>) -> (Option<(K, V)>, IsShorter)
where
    K: Clone,
    V: Clone,
{
    let n = match root.as_mut() {
        None => return (None, IsShorter(false)), // *** EARLY RETURN ***
        Some(rc) => Arc::make_mut(rc),
    };

    if n.right.is_some() {
        let (kv, is_shorter) = rm_rightmost(&mut n.right);
        n.right_ht -= is_shorter.0 as i8;
        if is_shorter.0 && n.bal() < -1 {
            (kv, rebal_lf_to_rt(root))
        } else {
            (kv, IsShorter(is_shorter.0 && n.bal() == 0))
        }
    } else {
        let old_n = take_node(root);
        *root = old_n.left;
        (Some((old_n.key, old_n.val)), IsShorter(true))
    }
}

// removes k from the map and returns the associated value and whether the
// tree at root is shorter as a result of the deletion.
fn rm<K, V, Q>(root: &mut OptNode<K, V>, k: &Q) -> (Option<(K, V)>, IsShorter)
where
    K: Borrow<Q> + Clone + Ord,
    V: Clone,
    Q: Ord + ?Sized,
{
    let n = match root.as_mut() {
        None => return (None, IsShorter(false)), // *** EARLY RETURN ***
        Some(rc) => Arc::make_mut(rc),
    };

    match k.cmp(n.key.borrow()) {
        Less => {
            let (v, is_shorter) = rm(&mut n.left, k);
            n.left_ht -= is_shorter.0 as i8;
            if is_shorter.0 && n.bal() > 1 {
                (v, rebal_rt_to_lf(root))
            } else {
                (v, IsShorter(is_shorter.0 && n.bal() == 0))
            }
        }

        Greater => {
            let (v, is_shorter) = rm(&mut n.right, k);
            n.right_ht -= is_shorter.0 as i8;
            if is_shorter.0 && n.bal() < -1 {
                (v, rebal_lf_to_rt(root))
            } else {
                (v, IsShorter(is_shorter.0 && n.bal() == 0))
            }
        }

        Equal => match (&n.left, &n.right) {
            (None, None) => {
                let old_n = take_node(root);
                (Some((old_n.key, old_n.val)), IsShorter(true))
            }

            (None, Some(_)) => {
                let old_n = take_node(root);
                *root = old_n.right;
                (Some((old_n.key, old_n.val)), IsShorter(true))
            }

            (Some(_), None) => {
                let old_n = take_node(root);
                *root = old_n.left;
                (Some((old_n.key, old_n.val)), IsShorter(true))
            }

            _ => {
                // both children are populated
                let (succ, is_shorter) = rm_leftmost(&mut n.right);
                let (succ_key, succ_val) = succ.unwrap();
                let old_key = replace(&mut n.key, succ_key);
                let old_val = replace(&mut n.val, succ_val);
                let old_elt = (old_key, old_val);

                n.right_ht -= is_shorter.0 as i8;
                if is_shorter.0 && n.bal() < -1 {
                    // we were taller on left and lost height on right
                    (Some(old_elt), rebal_lf_to_rt(root))
                } else {
                    (Some(old_elt), IsShorter(is_shorter.0 && n.bal() == 0))
                }
            }
        },
    }
}

fn retain<K, V, F>(root: OptNode<K, V>, f: &mut F) -> (OptNode<K, V>, usize)
where
    K: Clone + Ord,
    V: Clone,
    F: FnMut(&K, &mut V) -> bool,
{
    let Some(root) = root else {
        return (None, 0);
    };

    let (left, left_len) = retain(root.left.clone(), f);
    let (right, right_len) = retain(root.right.clone(), f);
    let len = left_len + right_len;
    let mut v = root.val.clone();

    if f(&root.key, &mut v) {
        (join(left, root.key.clone(), v, right), len + 1)
    } else {
        (join2(left, None, right), len)
    }
}

fn join_rt<K: Clone + Ord, V: Clone>(
    left: Arc<Node<K, V>>,
    k: K,
    v: V,
    opt_right: OptNode<K, V>,
) -> OptNode<K, V> {
    assert!(left.height() > height(&opt_right) + 1);

    // ultimately, we return a clone of left with the right branch replaced
    let mut t2 = left;
    let t2n = Arc::make_mut(&mut t2);

    let c = t2n.right.take();

    if height(&c) <= height(&opt_right) + 1 {
        let opt_t1 = Node::opt_new(k, v, c, opt_right);
        t2n.set_right(opt_t1);

        if t2n.is_bal() {
            chk_node!(Some(t2))
        } else {
            if rot_rt(&mut t2n.right).0 {
                t2n.right_ht -= 1;
            }
            let mut opt_t2 = Some(t2);
            rot_lf(&mut opt_t2);
            chk_node!(opt_t2)
        }
    } else {
        let opt_t1 = join_rt(c.unwrap(), k, v, opt_right);
        t2n.set_right(opt_t1);
        let is_bal = t2n.is_bal();
        let mut opt_t2 = Some(t2);
        if !is_bal {
            rot_lf(&mut opt_t2);
        }
        chk_node!(opt_t2)
    }
}

fn join_lf<K: Clone + Ord, V: Clone>(
    opt_left: OptNode<K, V>,
    k: K,
    v: V,
    right: Arc<Node<K, V>>,
) -> OptNode<K, V> {
    assert!(right.height() > height(&opt_left) + 1);

    // ultimately, we return a clone of right with the left branch replaced
    let mut t2 = right;
    let t2n = Arc::make_mut(&mut t2);

    let c = t2n.left.take();

    if height(&c) <= height(&opt_left) + 1 {
        let opt_t1 = Node::opt_new(k, v, opt_left, c);
        t2n.set_left(opt_t1);

        if t2n.is_bal() {
            chk_node!(Some(t2))
        } else {
            if rot_lf(&mut t2n.left).0 {
                t2n.left_ht -= 1;
            }
            let mut opt_t2 = Some(t2);
            rot_rt(&mut opt_t2);
            assert!(opt_t2.as_ref().unwrap().is_bal());
            chk_node!(opt_t2)
        }
    } else {
        let opt_t1 = join_lf(opt_left, k, v, c.unwrap());
        t2n.set_left(opt_t1);
        let is_bal = t2n.is_bal();
        let mut opt_t2 = Some(t2);
        if !is_bal {
            rot_rt(&mut opt_t2);
        }
        chk_node!(opt_t2)
    }
}

// Creates a merge of disjoint trees and a key k that divides them.
// Prereq: left.last_key() < k < right.first_key()
fn join<K: Clone + Ord, V: Clone>(
    opt_left: OptNode<K, V>,
    k: K,
    v: V,
    opt_right: OptNode<K, V>,
) -> OptNode<K, V> {
    let bal = height(&opt_right) - height(&opt_left);
    if bal < -1 {
        join_rt(opt_left.unwrap(), k, v, opt_right)
    } else if bal > 1 {
        join_lf(opt_left, k, v, opt_right.unwrap())
    } else {
        chk_node!(Node::opt_new(k, v, opt_left, opt_right))
    }
}

fn join2<K: Clone + Ord, V: Clone>(
    opt_left: OptNode<K, V>,
    opt_kv: Option<(K, V)>,
    mut opt_right: OptNode<K, V>,
) -> OptNode<K, V> {
    if let Some(kv) = opt_kv {
        join(opt_left, kv.0, kv.1, opt_right)
    } else if opt_left.is_none() {
        opt_right
    } else if opt_right.is_none() {
        opt_left
    } else {
        let (k, v) = rm_leftmost(&mut opt_right).0.unwrap();
        join(opt_left, k, v, opt_right)
    }
}

#[derive(PartialEq, Eq)]
enum NoMatchMergePolicy {
    Discard,
    Keep,
}
struct Merger<K, V, F>
where
    F: FnMut((K, V), (K, V)) -> Option<(K, V)>,
{
    on_only_left: NoMatchMergePolicy,
    on_only_right: NoMatchMergePolicy,
    deconflict: F,
    entry_dummy: std::marker::PhantomData<(K, V)>,
}

impl<K, V, F> Merger<K, V, F>
where
    K: Clone + Ord,
    V: Clone,
    F: FnMut((K, V), (K, V)) -> Option<(K, V)>,
{
    pub fn merge(
        &mut self,
        opt_t1: OptNode<K, V>,
        opt_t2: OptNode<K, V>,
    ) -> OptNode<K, V> {
        use NoMatchMergePolicy::*;
        match (&opt_t1, &opt_t2, &self.on_only_left, &self.on_only_right) {
            (None, _, _, Keep) => return opt_t2,
            (None, _, _, Discard) => return None,
            (_, None, Keep, _) => return opt_t1,
            (_, None, Discard, _) => return None,
            _ => (),
        }

        let t1 = match Arc::try_unwrap(opt_t1.unwrap()) {
            Ok(n) => n,
            Err(rc) => (*rc).clone(),
        };

        let (t2_lt, t2_kv, t2_gt) = split(opt_t2, &t1.key);
        let lf_int = self.merge(t1.left, t2_lt);
        let rt_int = self.merge(t1.right, t2_gt);
        let kv = match (t2_kv, &self.on_only_left) {
            (None, Keep) => Some((t1.key, t1.val)),
            (None, Discard) => None,
            (Some(t2_kv), _) => (self.deconflict)((t1.key, t1.val), t2_kv),
        };
        join2(lf_int, kv, rt_int)
    }
}

fn intersect<K: Clone + Ord, V: Clone>(
    opt_t1: OptNode<K, V>,
    opt_t2: OptNode<K, V>,
) -> OptNode<K, V> {
    let mut merger = Merger {
        on_only_left: NoMatchMergePolicy::Discard,
        on_only_right: NoMatchMergePolicy::Discard,
        deconflict: &mut |lhs, _| Some(lhs),
        entry_dummy: std::marker::PhantomData,
    };

    merger.merge(opt_t1, opt_t2)
}

fn diff<K: Clone + Ord, V: Clone>(
    opt_t1: OptNode<K, V>,
    opt_t2: OptNode<K, V>,
) -> OptNode<K, V> {
    let mut merger = Merger {
        on_only_left: NoMatchMergePolicy::Keep,
        on_only_right: NoMatchMergePolicy::Discard,
        deconflict: &mut |_, _| None,
        entry_dummy: std::marker::PhantomData,
    };

    merger.merge(opt_t1, opt_t2)
}

#[allow(dead_code)]
fn sym_diff<K: Clone + Ord, V: Clone>(
    opt_t1: OptNode<K, V>,
    opt_t2: OptNode<K, V>,
) -> OptNode<K, V> {
    let mut merger = Merger {
        on_only_left: NoMatchMergePolicy::Keep,
        on_only_right: NoMatchMergePolicy::Keep,
        deconflict: &mut |_, _| None,
        entry_dummy: std::marker::PhantomData,
    };

    merger.merge(opt_t1, opt_t2)
}

fn union<K: Clone + Ord, V: Clone>(
    opt_t1: OptNode<K, V>,
    opt_t2: OptNode<K, V>,
) -> OptNode<K, V> {
    let mut merger = Merger {
        on_only_left: NoMatchMergePolicy::Keep,
        on_only_right: NoMatchMergePolicy::Keep,
        deconflict: &mut |_, rhs| Some(rhs),
        entry_dummy: std::marker::PhantomData,
    };

    merger.merge(opt_t1, opt_t2)
}

#[allow(clippy::type_complexity)]
fn split<K, V, Q>(
    opt_root: OptNode<K, V>,
    k: &Q,
) -> (OptNode<K, V>, Option<(K, V)>, OptNode<K, V>)
where
    K: Borrow<Q> + Clone + Ord,
    V: Clone,
    Q: Ord + ?Sized,
{
    match opt_root {
        None => (None, None, None),
        Some(rc) => {
            // To reuse the node, we'd have to pass it into join.  By moving
            // pieces out of the node, we might avoid some cloning & Arc updates.
            let n = match Arc::try_unwrap(rc) {
                Ok(n) => n,
                Err(rc) => (*rc).clone(),
            };

            match k.cmp(n.key.borrow()) {
                Equal => (n.left, Some((n.key, n.val)), n.right),

                Less => {
                    let (l1, orig_kv, r1) = split(n.left, k);
                    (l1, orig_kv, join(r1, n.key, n.val, n.right))
                }

                Greater => {
                    let (l1, orig_kv, r1) = split(n.right, k);
                    (join(n.left, n.key, n.val, l1), orig_kv, r1)
                }
            }
        }
    }
}

impl<K, V> AvlMap<K, V> {
    /// Move all the elements of other into self leaving other empty.
    ///
    /// When there are matching keys, prefer the entries from other.
    ///
    /// # Examples
    /// ```
    /// use lazy_clone_collections::AvlMap;
    ///
    /// let mut m1 = AvlMap::from([(0,'a'), (1,'a')]);
    /// let mut m2 = AvlMap::from([(1,'b'), (2,'b')]);
    /// m1.append(&mut m2);
    /// assert_eq!(m1.len(), 3);
    /// assert_eq!(m1.get(&0), Some(&'a'));
    /// assert_eq!(m1.get(&1), Some(&'b'));
    /// assert_eq!(m1.get(&2), Some(&'b'));
    /// assert!(m2.is_empty());
    pub fn append(&mut self, other: &mut Self)
    where
        K: Ord + Clone,
        V: Clone,
    {
        self.root = union(self.root.take(), other.root.take());
        self.len = len(&self.root);
        other.len = 0;
    }

    /// Drops all elements from the map.
    pub fn clear(&mut self) {
        self.len = 0;
        self.root = None;
    }

    /// Tests if the map contains a value for the given key.
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.get(key).is_some()
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

    /// Returns the key-value pair for the least key in the map
    ///
    /// # Examples
    /// ```
    /// use lazy_clone_collections::AvlMap;
    ///
    /// let fmap = AvlMap::from([(2,0), (1,0)]);
    /// assert_eq!(fmap.first_key_value(), Some((&1, &0)));
    /// ```
    pub fn first_key_value(&self) -> Option<(&K, &V)> {
        let mut curr = self.root.as_ref()?;
        while let Some(next) = curr.left.as_ref() {
            curr = next;
        }
        Some((&curr.key, &curr.val))
    }

    /// Returns a reference to the value associated with key.
    ///
    /// # Example
    /// ```
    /// use lazy_clone_collections::AvlMap;
    ///
    /// let mut fmap = AvlMap::new();
    /// fmap.insert(0, 100);
    ///
    /// assert_eq!(fmap.get(&0), Some(&100));
    /// ```
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.get_key_value(key).map(|e| e.1)
    }

    /// Returns the entry for the given key.
    ///
    /// # Example
    /// ```
    /// use lazy_clone_collections::AvlMap;
    ///
    /// let mut m = AvlMap::from([(0, 100), (12, 7)]);
    /// assert_eq!(m.get_key_value(&12), Some((&12, &7)));
    /// ```
    pub fn get_key_value<Q>(&self, k: &Q) -> Option<(&K, &V)>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let mut curr = &self.root;
        while let Some(n) = curr {
            match k.cmp(n.key.borrow()) {
                Less => curr = &n.left,
                Equal => return Some((&n.key, &n.val)),
                Greater => curr = &n.right,
            }
        }

        None
    }

    /// Returns a mutable reference to the value associated with k.
    ///
    /// # Example
    /// ```
    /// use lazy_clone_collections::AvlMap;
    ///
    /// let mut fmap = AvlMap::new();
    /// fmap.insert(1, 7);
    ///
    /// *fmap.get_mut(&1).unwrap() = 2;
    /// assert_eq!(fmap.get(&1), Some(&2));
    /// ```
    pub fn get_mut<Q>(&mut self, k: &Q) -> Option<&mut V>
    where
        K: Borrow<Q> + Clone,
        Q: Ord + ?Sized,
        V: Clone,
    {
        let mut curr = &mut self.root;
        while let Some(rc) = curr {
            let n = Arc::make_mut(rc);
            match k.cmp(n.key.borrow()) {
                Less => curr = &mut n.left,
                Equal => return Some(&mut n.val),
                Greater => curr = &mut n.right,
            }
        }

        None
    }

    /// Inserts a key-value pair in the map.
    ///
    /// # Examples
    /// ```
    /// use lazy_clone_collections::AvlMap;
    ///
    /// let mut fmap = AvlMap::new();
    /// fmap.insert(0, "a");
    /// assert_eq!(fmap.get(&0), Some(&"a"));
    /// ```
    pub fn insert(&mut self, key: K, val: V) -> Option<V>
    where
        K: Clone + Ord,
        V: Clone,
    {
        let (ret, _) = ins(&mut self.root, key, val);
        self.len += ret.is_none() as usize;
        ret
    }

    /// Converts the map into an iterator over its keys.
    ///
    /// Consumes the map.
    /// # Examples
    /// ```
    /// use lazy_clone_collections::AvlMap;
    ///
    /// let m = AvlMap::from([(1, 'a'), (2, 'b')]);
    /// let keys: Vec<_> = m.into_keys().collect();
    /// assert_eq!(keys, [1, 2]);
    /// ```
    pub fn into_keys(
        self,
    ) -> impl DoubleEndedIterator<Item = K> + ExactSizeIterator + FusedIterator
    where
        K: Clone,
        V: Clone,
    {
        use IterAction::*;

        let mut work =
            VecDeque::<IterAction<IntoIterNode<K, V, IntoKey>>>::new();
        if let Some(n) = opt_to_iter_node(self.root) {
            work.push_back(Descend(n));
        }

        InnerIter {
            work,
            len: self.len,
        }
    }

    /// Converts the map into an iterator over its values, ordered by their
    /// corresponding keys.
    ///
    /// Consumes the map.
    /// # Examples
    /// ```
    /// use lazy_clone_collections::AvlMap;
    ///
    /// let m = AvlMap::from([(100, 'a'), (2, 'z')]);
    /// let vals: Vec<_> = m.into_values().collect();
    /// assert_eq!(vals, ['z', 'a']);
    /// ```
    pub fn into_values(
        self,
    ) -> impl DoubleEndedIterator<Item = V> + ExactSizeIterator + FusedIterator
    where
        K: Clone,
        V: Clone,
    {
        use IterAction::*;

        let mut work =
            VecDeque::<IterAction<IntoIterNode<K, V, IntoValue>>>::new();
        if let Some(n) = opt_to_iter_node(self.root) {
            work.push_back(Descend(n));
        }

        InnerIter {
            work,
            len: self.len,
        }
    }

    /// Returns true if self contains no entries, false otherwise.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Creates an iterator over the map entries, sorted by key.
    ///
    /// It is more efficient to visit each entry using
    /// [`for_each`](#method.for_each).  (Because tree nodes may share children,
    /// the children cannot have ancestor links.  The iterator tracks its
    /// progress with a Vec.)  The primary advantage of `iter` is the
    /// functionality provided by the [`Iterator`] trait.
    ///
    /// # Examples
    /// ```
    /// use lazy_clone_collections::AvlMap;
    ///
    /// let m = AvlMap::from([(0,1), (1,2), (2, 3)]);
    /// for (i, (k, v)) in m.iter().enumerate() {
    ///     assert_eq!(&i, k);
    ///     assert_eq!(&(i+1), v);
    /// }
    /// ```
    pub fn iter(&self) -> Iter<K, V> {
        let mut work = VecDeque::new();
        if let Some(root) = self.root.as_ref() {
            work.push_front(IterAction::Descend(root));
        }

        Iter {
            iter: InnerIter {
                work,
                len: self.len,
            },
        }
    }

    /// Returns iterator of the map's entries, sorted by key, with a mutable
    /// refence to each value (and an immutable reference to each key).
    ///
    /// The iterator obtains sole ownership of each value it returns, cloning
    /// nodes if necessary.  The cloning occurs even if the calling code does
    /// not mutate the returned values.  When the entire iterator is consumed,
    /// all shared nodes in self are cloned.
    ///
    /// As with [`iter()`](#method.iter) and [`for_each`](#method.for_each), the
    /// method [`for_each_mut`](#method.for_each_mut) is expected to be more
    /// efficient but less ergonomic than [`iter_mut`](#method.iter_mut).
    ///
    /// # Examples
    /// ```
    /// use lazy_clone_collections::AvlMap;
    ///
    /// let mut m = AvlMap::from([(0,0), (1,1), (2,2)]);
    /// for (k, v) in m.iter_mut() {
    ///     *v += k;
    /// }
    /// assert_eq!(m.get(&0), Some(&0));
    /// assert_eq!(m.get(&1), Some(&2));
    /// assert_eq!(m.get(&2), Some(&4));
    /// ```
    pub fn iter_mut(&mut self) -> IterMut<'_, K, V>
    where
        K: Clone,
        V: Clone,
    {
        let mut work = VecDeque::<IterAction<IterMutNode<'_, K, V>>>::new();
        if let Some(rc) = self.root.as_mut() {
            let n = Arc::make_mut(rc);
            work.push_back(IterAction::Descend(n));
        }

        IterMut {
            iter: InnerIter {
                work,
                len: self.len,
            },
        }
    }

    /// Produces an iterator over the keys of the map, in sorted order.
    ///
    /// This is a simple projection from [`iter`](#method.iter).
    /// # Examples
    /// ```
    /// use lazy_clone_collections::AvlMap;
    ///
    /// let m = AvlMap::from([(0,0), (1,1), (2,2)]);
    /// let cnt_even_keys = m.keys().filter(|&k| k % &2 == 0).count();
    /// assert_eq!(cnt_even_keys, 2);
    /// ```
    pub fn keys(
        &self,
    ) -> impl DoubleEndedIterator<Item = &K> + ExactSizeIterator + FusedIterator
    {
        self.iter().map(|p| p.0)
    }

    /// Return an Entry for the greatest key in the map.
    pub fn last_entry(&mut self) -> Option<Entry<'_, Self>>
    where
        K: Clone + Ord,
        V: Clone,
    {
        let key = self.last_key_value()?.0.clone();
        Some(Entry { map: self, key })
    }

    /// Returns the key-value pair for the greatest key in the map
    ///
    /// # Examples
    /// ```
    /// use lazy_clone_collections::AvlMap;
    ///
    /// let fmap = AvlMap::from([(2,0), (1,0)]);
    /// assert_eq!(fmap.last_key_value(), Some((&2, &0)));
    /// ```
    pub fn last_key_value(&self) -> Option<(&K, &V)> {
        let mut prev = &None;
        let mut curr = &self.root;
        while let Some(rc) = curr.as_ref() {
            prev = curr;
            curr = &rc.right;
        }
        prev.as_ref().map(|rc| (&rc.key, &rc.val))
    }

    /// Returns the number of entries in self.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Creates a new, empty map.
    /// # Examples
    /// ```
    /// use lazy_clone_collections::AvlMap;
    /// let m: AvlMap<usize, usize> = AvlMap::new();
    /// assert!(m.is_empty());
    /// ```
    pub fn new() -> Self {
        AvlMap { len: 0, root: None }
    }

    /// Builds a map with entries from the LHS map with keys that are not in the
    /// RHS map.
    ///
    /// # Examples
    /// ```
    /// use lazy_clone_collections::AvlMap;
    ///
    /// let lhs = AvlMap::from([(0,1), (1, 2)]);
    /// let rhs = AvlMap::from([(1,5), (3,4)]);
    /// let d = AvlMap::new_diff(lhs, rhs);
    /// assert_eq!(d.get(&0), Some(&1));
    /// assert_eq!(d.get(&1), None);
    /// ```
    pub fn new_diff(lhs: Self, rhs: Self) -> Self
    where
        K: Clone + Ord,
        V: Clone,
    {
        let root = diff(lhs.root, rhs.root);
        let len = len(&root);
        Self { root, len }
    }

    // FIXME: favor RHS over LHS entries
    /// Creates a map with entries from the LHS that have keys in the RHS.
    ///
    /// # Examples
    /// ```
    /// use lazy_clone_collections::AvlMap;
    ///
    /// let lhs = AvlMap::from([(0,1), (1, 2)]);
    /// let rhs = AvlMap::from([(1,5), (3,4)]);
    /// let i = AvlMap::new_intersect(lhs, rhs);
    /// assert_eq!(i.get(&0), None);
    /// assert_eq!(i.get(&1), Some(&2));
    /// ```
    pub fn new_intersect(lhs: Self, rhs: Self) -> Self
    where
        K: Clone + Ord,
        V: Clone,
    {
        let root = intersect(lhs.root, rhs.root);
        let len = len(&root);
        Self { root, len }
    }

    /// Build a new map by joining two maps around a pivot key that divides the
    /// entries of the maps.
    ///
    /// The constructed map contains the entries from both maps and the pivot
    /// key and the value provided for the pivot key.
    ///
    /// Requires:
    ///    - greatest key of lhs is less than key
    ///    - key is less than the least key of rhs
    ///
    /// # Examples
    /// ```
    /// use lazy_clone_collections::AvlMap;
    ///
    /// let f1 = AvlMap::from([(0, 'a'), (1, 'b')]);
    /// let f2 = AvlMap::from([(3, 'd')]);
    /// let f3 = AvlMap::new_join(f1, 2, 'c', f2);
    /// assert_eq!(f3.get(&0), Some(&'a'));
    /// assert_eq!(f3.get(&1), Some(&'b'));
    /// assert_eq!(f3.get(&2), Some(&'c'));
    /// assert_eq!(f3.get(&3), Some(&'d'));
    /// ```
    pub fn new_join(lhs: Self, key: K, val: V, rhs: Self) -> Self
    where
        K: Ord + Clone,
        V: Clone,
    {
        debug_assert!(lhs.last_key_value().map_or(true, |(k, _)| *k < key));
        debug_assert!(rhs.first_key_value().map_or(true, |(k, _)| key < *k));

        Self {
            root: join(lhs.root, key, val, rhs.root),
            len: lhs.len + 1 + rhs.len,
        }
    }

    /// Builds a map with entries from the LHS and RHS maps that have keys that
    /// occur in only one of the maps and not the other.
    ///
    /// # Examples
    /// ```
    /// use lazy_clone_collections::AvlMap;
    ///
    /// let lhs = AvlMap::from([(0,1), (1, 2)]);
    /// let rhs = AvlMap::from([(1,5), (3,4)]);
    /// let d = AvlMap::new_sym_diff(lhs, rhs);
    /// assert_eq!(d.get(&0), Some(&1));
    /// assert_eq!(d.get(&1), None);
    /// assert_eq!(d.get(&3), Some(&4));
    /// ```
    pub fn new_sym_diff(lhs: Self, rhs: Self) -> Self
    where
        K: Clone + Ord,
        V: Clone,
    {
        let root = sym_diff(lhs.root, rhs.root);
        let len = len(&root);
        Self { root, len }
    }

    /// Builds a map with entries from both maps, with entries from the RHS
    /// taking precedence when a key appears in both maps.
    ///
    /// # Examples
    /// ```
    /// use lazy_clone_collections::AvlMap;
    ///
    /// let lhs = AvlMap::from([(0,'a'), (1, 'a')]);
    /// let rhs = AvlMap::from([(1,'b'), (2,'b')]);
    /// let joint = AvlMap::new_union(lhs, rhs);
    /// assert_eq!(joint.get(&0), Some(&'a'));
    /// assert_eq!(joint.get(&1), Some(&'b'));
    /// assert_eq!(joint.get(&2), Some(&'b'));
    /// ```
    pub fn new_union(lhs: Self, rhs: Self) -> Self
    where
        K: Clone + Ord,
        V: Clone,
    {
        let root = union(lhs.root, rhs.root);
        let len = len(&root);
        Self { root, len }
    }

    /// Removes the entry with the least key and returns it.
    ///
    /// # Examples
    /// ```
    /// use lazy_clone_collections::AvlMap;
    ///
    /// let mut m = AvlMap::from([(2, 2), (0, 0), (1, 1)]);
    /// assert_eq!(m.pop_first(), Some((0, 0)));
    /// assert_eq!(m.len(), 2);
    /// ```
    pub fn pop_first(&mut self) -> Option<(K, V)>
    where
        K: Clone,
        V: Clone,
    {
        let kv = rm_leftmost(&mut self.root).0?;
        self.len -= 1;
        Some(kv)
    }

    /// Removes the entry with the greatest key and returns it.
    ///
    /// # Examples
    /// ```
    /// use lazy_clone_collections::AvlMap;
    ///
    /// let mut m = AvlMap::from([(2, 2), (0, 0), (1, 1)]);
    /// assert_eq!(m.pop_last(), Some((2, 2)));
    /// assert_eq!(m.len(), 2);
    /// ```
    pub fn pop_last(&mut self) -> Option<(K, V)>
    where
        K: Clone,
        V: Clone,
    {
        let kv = rm_rightmost(&mut self.root).0?;
        self.len -= 1;
        Some(kv)
    }

    /// Returns an iterator over the elements with a key in the given range.
    pub fn range<Q, R>(&self, range: R) -> Range<'_, K, V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
        R: RangeBounds<Q>,
    {
        use Bound::*;
        use IterAction::*;

        // TODO: sanity checks on range

        // start with a worklist that covers the entire map (possibly empty)
        let mut work = VecDeque::<IterAction<NormIterNode<'_, K, V>>>::new();
        if let Some(root) = self.root.as_ref() {
            work.push_back(Descend(root));
        }

        let lb = range.start_bound();
        let ub = range.end_bound();

        // Prune the "left side" of our iteration space.  At some point, we may
        // push a "Descend" for a node that includes entries beyond the end of
        // the given range.  This should only happen once while finding the left
        // edge and the node will be at the back of the queue.
        while let Some(Descend(_)) = work.front() {
            let Some(Descend(n)) = work.pop_front() else {
                panic!("we just checked!");
            };

            if let Some(rt) = n.right.as_ref() {
                match ub {
                    Excluded(k) | Included(k) if k <= n.key.borrow() => (),
                    _ => work.push_front(Descend(rt)),
                }
            }

            if range.contains(n.key.borrow()) {
                work.push_front(Return((&n.key, &n.val)));
            }

            if let Some(lf) = n.left.as_ref() {
                match lb {
                    Excluded(k) | Included(k) if k >= n.key.borrow() => (),
                    _ => work.push_front(Descend(lf)),
                }
            }
        }

        // prune the right side of the iteration space.
        while let Some(Descend(_)) = work.back() {
            let Some(Descend(n)) = work.pop_back() else {
                panic!("we just checked!");
            };

            if let Some(lf) = n.left.as_ref() {
                match lb {
                    Excluded(k) | Included(k) if k >= n.key.borrow() => (),
                    _ => work.push_back(Descend(lf)),
                }
            }

            if range.contains(n.key.borrow()) {
                work.push_back(Return((&n.key, &n.val)));
            }

            if let Some(rt) = n.right.as_ref() {
                match ub {
                    Excluded(k) | Included(k) if k <= n.key.borrow() => (),
                    _ => work.push_back(Descend(rt)),
                }
            }
        }

        Range {
            iter: InnerIter {
                work,
                len: self.len(),
            },
        }
    }

    /// Returns a mutable iterator over the elements with a key in the given range.
    pub fn range_mut<T, R>(
        &mut self,
        range: R,
    ) -> impl DoubleEndedIterator<Item = (&K, &mut V)> + FusedIterator
    where
        T: Ord + ?Sized,
        K: Borrow<T> + Clone,
        R: RangeBounds<T>,
        V: Clone,
    {
        use Bound::*;
        use IterAction::*;

        // TODO: sanity checks on range

        let len = self.len();

        // start with a worklist that covers the entire map (possibly empty)
        let mut work = VecDeque::<IterAction<IterMutNode<'_, K, V>>>::new();
        if let Some(root) = self.root.as_mut() {
            let n = Arc::make_mut(root);
            work.push_back(Descend(n));
        }

        let lb = range.start_bound();
        let ub = range.end_bound();

        // Prune the "left side" of our iteration space.  At some point, we may
        // push a "Descend" for a node that includes entries beyond the end of
        // the given range.  This should only happen once while finding the left
        // edge and the node will be at the back of the queue.
        while let Some(Descend(_)) = work.front() {
            let Some(Descend(n)) = work.pop_front() else {
                panic!("we just checked!");
            };

            if let Some(rt) = n.right.as_mut() {
                match ub {
                    Excluded(k) | Included(k) if k <= n.key.borrow() => (),
                    _ => work.push_front(Descend(Arc::make_mut(rt))),
                }
            }

            if range.contains(n.key.borrow()) {
                work.push_front(Return((&n.key, &mut n.val)));
            }

            if let Some(lf) = n.left.as_mut() {
                match lb {
                    Excluded(k) | Included(k) if k >= n.key.borrow() => (),
                    _ => work.push_front(Descend(Arc::make_mut(lf))),
                }
            }
        }

        // prune the right side of the iteration space.
        while let Some(Descend(_)) = work.back() {
            let Some(Descend(n)) = work.pop_back() else {
                panic!("we just checked!");
            };

            if let Some(lf) = n.left.as_mut() {
                match lb {
                    Excluded(k) | Included(k) if k >= n.key.borrow() => (),
                    _ => work.push_back(Descend(Arc::make_mut(lf))),
                }
            }

            if range.contains(n.key.borrow()) {
                work.push_back(Return((&n.key, &mut n.val)));
            }

            if let Some(rt) = n.right.as_mut() {
                match ub {
                    Excluded(k) | Included(k) if k <= n.key.borrow() => (),
                    _ => work.push_back(Descend(Arc::make_mut(rt))),
                }
            }
        }

        InnerIter::<IterMutNode<'_, K, V>> { work, len }
    }

    /// Removes the entry for the given key and returns the unmapped value.
    ///
    /// # Examples
    /// ```
    /// use lazy_clone_collections::AvlMap;
    ///
    /// let mut fmap = AvlMap::new();
    /// fmap.insert(1, 2);
    /// fmap.insert(2, 3);
    /// assert_eq!(fmap.remove(&2), Some(3));
    /// assert_eq!(fmap.remove(&2), None);
    /// ```
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q> + Clone + Ord,
        Q: Ord + ?Sized,
        V: Clone,
    {
        self.remove_entry(key).map(|e| e.1)
    }

    /// Removes and returns the entry matching the given key.
    ///
    /// # Examples
    /// ```
    /// use lazy_clone_collections::AvlMap;
    ///
    /// let mut fmap = AvlMap::new();
    /// fmap.insert(1, 2);
    /// fmap.insert(2, 3);
    /// assert_eq!(fmap.remove(&2), Some(3));
    /// assert_eq!(fmap.remove(&2), None);
    /// ```
    pub fn remove_entry<Q>(&mut self, key: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q> + Clone + Ord,
        Q: Ord + ?Sized,
        V: Clone,
    {
        let kv = rm(&mut self.root, key).0?;
        self.len -= 1;
        chk_map!(&self);
        Some(kv)
    }

    /// Applies f to each map entry, discarding those for which f returns false.
    ///
    /// # Examples
    /// ```
    /// use lazy_clone_collections::AvlMap;
    ///
    /// let mut m = AvlMap::<i32, char>::from([(-1, 'a'), (0, 'a'), (1, 'a')]);
    /// m.retain(|k, v| {
    ///     *v = 'b';
    ///     k.is_positive()
    /// });
    /// assert_eq!(m.get(&0), None);
    /// assert_eq!(m.get(&1), Some(&'b'));
    /// assert_eq!(m.len(), 1);
    /// ```
    pub fn retain<F>(&mut self, mut f: F)
    where
        K: Clone + Ord,
        V: Clone,
        F: FnMut(&K, &mut V) -> bool,
    {
        let (root, len) = retain(self.root.take(), &mut f);
        self.root = root;
        self.len = len;
    }

    /// Moves all elements greater than or equal to the provided key into a new
    /// map and returns it.
    ///
    /// # Examples
    /// ```
    /// use lazy_clone_collections::AvlMap;
    ///
    /// let mut fmap: AvlMap<_, _> = (0..10).map(|i| (i, i * 2)).collect();
    /// let higher_fives = fmap.split_off(&5);
    /// assert_eq!(fmap.last_key_value(), Some((&4,&8)));
    /// assert_eq!(fmap.get(&6), None);
    /// assert_eq!(higher_fives.first_key_value(), Some((&5, &10)));
    /// assert_eq!(higher_fives.get(&6), Some(&12));
    /// ```
    pub fn split_off<Q>(&mut self, key: &Q) -> Self
    where
        K: Borrow<Q> + Clone + Ord,
        Q: Ord + ?Sized,
        V: Clone,
    {
        let (lhs, orig_kv, mut rhs) = split(self.root.take(), key);

        let len_lhs = len(&lhs);
        let len_rhs = self.len - len_lhs;

        self.len = len_lhs;
        self.root = lhs;

        if let Some((k, v)) = orig_kv {
            ins(&mut rhs, k, v);
        }

        AvlMap {
            len: len_rhs,
            root: rhs,
        }
    }

    /// Produces an iterator over the values of the map, ordered by their
    /// associated keys.
    ///
    /// The iterator is a simple projection from [`iter`](#method.iter).
    /// # Examples
    /// ```
    /// use lazy_clone_collections::AvlMap;
    ///
    /// let m = AvlMap::from([(0,0), (1,1), (2,2)]);
    /// let sum_values: u32 = m.values().sum();
    /// assert_eq!(sum_values, 3);
    /// ```
    pub fn values(
        &self,
    ) -> impl DoubleEndedIterator<Item = &V> + ExactSizeIterator + FusedIterator
    {
        self.iter().map(|p| p.1)
    }

    /// Returns an iterator of mutable references to the map's values, ordered
    /// by their associated keys.
    ///
    /// The iterator is a simple projection from the iterator returned by
    /// [`iter_mut`](#method.iter) and has the similar properties.
    ///
    /// # Examples
    /// ```
    /// use lazy_clone_collections::AvlMap;
    ///
    /// let mut m = AvlMap::from([(0,0), (1,1), (2,2)]);
    /// for v in m.values_mut() {
    ///     *v *= 17;
    /// };
    /// assert_eq!(m.get(&2), Some(&34));
    /// ```
    pub fn values_mut(
        &mut self,
    ) -> impl DoubleEndedIterator<Item = &mut V> + ExactSizeIterator + FusedIterator
    where
        K: Clone,
        V: Clone,
    {
        self.iter_mut().map(|p| p.1)
    }

    /// Applies f to each entry of the map in order of the keys.
    ///
    /// # Examples
    /// ```
    /// use lazy_clone_collections::AvlMap;
    ///
    /// let m = AvlMap::from([(0,-10), (1,0), (2,12)]);
    /// let mut cnt_keys_gt_vals = 0;
    /// m.for_each(|(k, v)| if k > v { cnt_keys_gt_vals += 1 });
    /// assert_eq!(cnt_keys_gt_vals, 2);
    /// ```
    pub fn for_each<F: FnMut((&K, &V))>(&self, mut f: F) {
        if let Some(rc) = self.root.as_ref() {
            rc.for_each(&mut f);
        }
    }

    /// Applies a function to every key-value pair in the map.
    ///
    /// The passed function must take a reference to the key type and a mutable
    /// reference to the value type.  Any shared nodes in the tree are cloned,
    /// regardless of whether the contained values are mutated.
    ///
    /// # Examples
    /// ```
    /// use lazy_clone_collections::AvlMap;
    ///
    /// let mut fmap = AvlMap::new();
    /// fmap.insert(0, "a");
    /// fmap.for_each_mut(|(_, v)| *v = "b");
    /// assert_eq!(fmap.get(&0), Some(&"b"));
    /// ```
    pub fn for_each_mut<F: FnMut((&K, &mut V))>(&mut self, mut f: F)
    where
        K: Clone,
        V: Clone,
    {
        if let Some(rc) = self.root.as_mut() {
            Arc::make_mut(rc).for_each_mut(&mut f);
        }
    }

    #[cfg(test)]
    fn chk(&self)
    where
        K: Ord,
    {
        assert_eq!(self.len, chk(&self.root, None).0);
    }
}

#[derive(Debug)]
enum IterAction<I: IterNode> {
    Descend(I::Node),
    Return(I::Item),
}

trait IterNode {
    type Item;
    type Node;

    fn destruct(
        n: Self::Node,
    ) -> (Option<Self::Node>, Self::Item, Option<Self::Node>);
}

struct InnerIter<Node: IterNode> {
    work: VecDeque<IterAction<Node>>,
    len: usize,
}

impl<Node: IterNode> Iterator for InnerIter<Node> {
    type Item = Node::Item;

    fn next(&mut self) -> Option<Self::Item> {
        use IterAction::*;

        let a = self.work.pop_front()?;
        match a {
            Return(x) => {
                self.len -= 1;
                Some(x)
            }

            Descend(n) => {
                let (lf, item, rt) = Node::destruct(n);

                if let Some(rt) = rt {
                    self.work.push_front(Descend(rt));
                }

                self.work.push_front(Return(item));

                if let Some(lf) = lf {
                    self.work.push_front(Descend(lf));
                }

                self.next()
            }
        }
    }
}

impl<Node: IterNode> DoubleEndedIterator for InnerIter<Node> {
    fn next_back(&mut self) -> Option<Self::Item> {
        use IterAction::*;

        let a = self.work.pop_back()?;
        match a {
            Return(x) => {
                self.len -= 1;
                Some(x)
            }

            Descend(n) => {
                let (lf, item, rt) = Node::destruct(n);

                if let Some(lf) = lf {
                    self.work.push_back(Descend(lf));
                }

                self.work.push_back(Return(item));

                if let Some(rt) = rt {
                    self.work.push_back(Descend(rt));
                }

                self.next_back()
            }
        }
    }
}

impl<Node: IterNode> ExactSizeIterator for InnerIter<Node> {
    fn len(&self) -> usize {
        self.len
    }
}

impl<Node: IterNode> FusedIterator for InnerIter<Node> {}

pub struct NormIterNode<'a, K, V> {
    marker: PhantomData<&'a Arc<Node<K, V>>>,
}

impl<'a, K, V> IterNode for NormIterNode<'a, K, V> {
    type Item = (&'a K, &'a V);
    type Node = &'a Arc<Node<K, V>>;

    fn destruct(
        n: Self::Node,
    ) -> (Option<Self::Node>, Self::Item, Option<Self::Node>) {
        (n.left.as_ref(), (&n.key, &n.val), n.right.as_ref())
    }
}

pub struct Iter<'a, K, V> {
    iter: InnerIter<NormIterNode<'a, K, V>>,
}

impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

impl<'a, K, V> DoubleEndedIterator for Iter<'a, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back()
    }
}

impl<'a, K, V> ExactSizeIterator for Iter<'a, K, V> {
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl<'a, K, V> FusedIterator for Iter<'a, K, V> {}

#[derive(Debug)]
pub struct IterMutNode<'a, K, V> {
    marker: PhantomData<&'a mut Arc<Node<K, V>>>,
}

impl<'a, K: Clone, V: Clone> IterNode for IterMutNode<'a, K, V> {
    type Item = (&'a K, &'a mut V);
    type Node = &'a mut Node<K, V>;

    fn destruct(
        n: Self::Node,
    ) -> (Option<Self::Node>, Self::Item, Option<Self::Node>) {
        (
            n.left.as_mut().map(Arc::make_mut),
            (&n.key, &mut n.val),
            n.right.as_mut().map(Arc::make_mut),
        )
    }
}

pub struct IterMut<'a, K: Clone, V: Clone> {
    iter: InnerIter<IterMutNode<'a, K, V>>,
}

impl<'a, K: Clone, V: Clone> Iterator for IterMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

impl<'a, K: Clone, V: Clone> DoubleEndedIterator for IterMut<'a, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back()
    }
}

impl<'a, K: Clone, V: Clone> ExactSizeIterator for IterMut<'a, K, V> {
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl<'a, K: Clone, V: Clone> FusedIterator for IterMut<'a, K, V> {}

pub struct Range<'a, K, V> {
    iter: InnerIter<NormIterNode<'a, K, V>>,
}

impl<'a, K, V> Iterator for Range<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.iter.len()))
    }
}

impl<'a, K, V> DoubleEndedIterator for Range<'a, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back()
    }
}

impl<'a, K, V> FusedIterator for Range<'a, K, V> {}

trait IntoItem<K, V> {
    type Item;

    fn from_arc(k: &K, v: &V) -> Self::Item;
    fn from_own(k: K, v: V) -> Self::Item;
}

struct IntoKey;
struct IntoValue;
struct IntoKeyValue;

impl<K: Clone, V> IntoItem<K, V> for IntoKey {
    type Item = K;

    fn from_arc(k: &K, _: &V) -> Self::Item {
        k.clone()
    }

    fn from_own(k: K, _: V) -> Self::Item {
        k
    }
}

impl<K, V: Clone> IntoItem<K, V> for IntoValue {
    type Item = V;

    fn from_arc(_: &K, v: &V) -> Self::Item {
        v.clone()
    }

    fn from_own(_: K, v: V) -> Self::Item {
        v
    }
}

impl<K: Clone, V: Clone> IntoItem<K, V> for IntoKeyValue {
    type Item = (K, V);

    fn from_arc(k: &K, v: &V) -> Self::Item {
        (k.clone(), v.clone())
    }

    fn from_own(k: K, v: V) -> Self::Item {
        (k, v)
    }
}

enum IntoIterNode<K, V, II> {
    Arcked(Arc<Node<K, V>>),
    Owned(Node<K, V>),
    #[allow(dead_code)]
    Marker(PhantomData<II>),
}

fn opt_to_iter_node<K, V, II>(
    n: OptNode<K, V>,
) -> Option<IntoIterNode<K, V, II>> {
    use IntoIterNode::*;
    match Arc::try_unwrap(n?) {
        Ok(n) => Some(Owned(n)),
        Err(arc) => Some(Arcked(arc)),
    }
}

impl<K: Clone, V: Clone, II: IntoItem<K, V>> IterNode
    for IntoIterNode<K, V, II>
{
    type Item = II::Item;
    type Node = Self;

    fn destruct(
        n: Self::Node,
    ) -> (Option<Self::Node>, Self::Item, Option<Self::Node>) {
        use IntoIterNode::*;

        match n {
            Arcked(arc) => (
                arc.left.clone().map(|rc| Arcked(rc)),
                II::from_arc(&arc.key, &arc.val),
                arc.right.clone().map(|rc| Arcked(rc)),
            ),

            IntoIterNode::Owned(n) => (
                opt_to_iter_node(n.left),
                II::from_own(n.key, n.val),
                opt_to_iter_node(n.right),
            ),

            IntoIterNode::Marker(_) => {
                panic!("IntoIterNode::Marker should not exist");
            }
        }
    }
}

pub struct IntoIter<K: Clone, V: Clone> {
    iter: InnerIter<IntoIterNode<K, V, IntoKeyValue>>,
}

impl<K: Clone, V: Clone> Iterator for IntoIter<K, V> {
    type Item = (K, V);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

impl<K: Clone, V: Clone> DoubleEndedIterator for IntoIter<K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back()
    }
}

impl<K: Clone, V: Clone> ExactSizeIterator for IntoIter<K, V> {
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl<K: Clone, V: Clone> FusedIterator for IntoIter<K, V> {}

impl<K: Clone, V: Clone> IntoIterator for AvlMap<K, V> {
    type Item = (K, V);
    type IntoIter = IntoIter<K, V>;

    fn into_iter(self) -> Self::IntoIter {
        use IterAction::*;

        let mut work =
            VecDeque::<IterAction<IntoIterNode<K, V, IntoKeyValue>>>::new();
        if let Some(n) = opt_to_iter_node(self.root) {
            work.push_back(Descend(n));
        }

        IntoIter {
            iter: InnerIter {
                work,
                len: self.len,
            },
        }
    }
}

impl<'a, K, V> IntoIterator for &'a AvlMap<K, V> {
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, K: Clone, V: Clone> IntoIterator for &'a mut AvlMap<K, V> {
    type Item = (&'a K, &'a mut V);
    type IntoIter = IterMut<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<'a, K: Clone + Ord, V: Clone> Extend<(&'a K, &'a V)> for AvlMap<K, V> {
    fn extend<I: IntoIterator<Item = (&'a K, &'a V)>>(&mut self, iter: I) {
        for (k, v) in iter {
            self.insert(k.clone(), v.clone());
        }
    }
}

impl<K: Clone + Ord, V: Clone> Extend<(K, V)> for AvlMap<K, V> {
    fn extend<T: IntoIterator<Item = (K, V)>>(&mut self, iter: T) {
        for (k, v) in iter {
            self.insert(k, v);
        }
    }
}

impl<K, V, const N: usize> From<[(K, V); N]> for AvlMap<K, V>
where
    K: Clone + Ord,
    V: Clone,
{
    fn from(vs: [(K, V); N]) -> Self {
        AvlMap::from_iter(vs.into_iter())
    }
}

impl<K: Clone + Ord, V: Clone> FromIterator<(K, V)> for AvlMap<K, V> {
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
        let mut fmap = AvlMap::new();
        fmap.extend(iter);
        fmap
    }
}

impl<K, V> Map for AvlMap<K, V> {
    type Key = K;
    type Value = V;

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
        Q: Ord + ?Sized,
        V: Clone,
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

#[cfg(feature = "serde")]
mod avl_serde {
    use super::AvlMap;
    use serde::de::{Deserialize, MapAccess, Visitor};
    use std::fmt;
    use std::marker::PhantomData;

    pub(super) struct AvlMapVisitor<K, V> {
        marker: PhantomData<fn() -> AvlMap<K, V>>,
    }

    impl<K, V> AvlMapVisitor<K, V> {
        pub fn new() -> Self {
            AvlMapVisitor {
                marker: PhantomData,
            }
        }
    }

    impl<'de, K, V> Visitor<'de> for AvlMapVisitor<K, V>
    where
        K: Clone + Deserialize<'de> + Ord,
        V: Clone + Deserialize<'de>,
    {
        type Value = AvlMap<K, V>;

        // Format a message stating what data this Visitor expects to receive.
        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("lazy_clone_collections::AvlMap")
        }

        fn visit_map<M>(self, mut access: M) -> Result<Self::Value, M::Error>
        where
            M: MapAccess<'de>,
        {
            let mut map = AvlMap::<K, V>::new();

            while let Some((key, value)) = access.next_entry()? {
                map.insert(key, value);
            }

            Ok(map)
        }
    }
}

#[cfg(feature = "serde")]
impl<K, V> serde::ser::Serialize for AvlMap<K, V>
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
impl<'de, K, V> serde::de::Deserialize<'de> for AvlMap<K, V>
where
    K: Clone + serde::de::Deserialize<'de> + Ord,
    V: Clone + serde::de::Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::de::Deserializer<'de>,
    {
        deserializer.deserialize_map(avl_serde::AvlMapVisitor::new())
    }
}

#[cfg(test)]
mod test {
    extern crate quickcheck;
    use super::*;
    use quickcheck::quickcheck;

    // this is a compile-time test
    fn _default_maps_for_no_default_entries() {
        struct Foo;
        let _ = AvlMap::<usize, Foo>::default();
    }

    // run with: `cargo test --features serde,serde_test`
    #[cfg(feature = "serde_test")]
    mod serde_test {
        use serde_test::{assert_tokens, Token};

        #[test]
        fn test_serde() {
            let mut s = crate::AvlMap::new();
            s.insert('a', 0_i32);
            s.insert('b', 1_i32);
            s.insert('c', 2_i32);

            assert_tokens(
                &s,
                &[
                    Token::Map { len: Some(3) },
                    Token::Char('a'),
                    Token::I32(0),
                    Token::Char('b'),
                    Token::I32(1),
                    Token::Char('c'),
                    Token::I32(2),
                    Token::MapEnd,
                ],
            );
        }
    }

    fn bal_test(vs: Vec<(u8, u32)>) {
        let mut fmap = AvlMap::new();
        for &(k, v) in vs.iter() {
            fmap.insert(k, v);
            println!("{:?}", fmap);
            fmap.chk();
        }
    }

    fn rm_test(vs: Vec<(i8, u32)>) {
        let mut fmap = AvlMap::new();
        let mut btree = std::collections::BTreeMap::new();

        for &(k, v) in vs.iter() {
            match k {
                1..=i8::MAX => {
                    let k = k % 32;
                    assert_eq!(fmap.insert(k, v), btree.insert(k, v));
                }

                0 | i8::MIN => (),

                _ => {
                    let k = -k % 32;
                    assert_eq!(fmap.remove(&k), btree.remove(&k));
                }
            }

            // println!("{:?}", fmap);
            assert!(fmap.iter().cmp(btree.iter()).is_eq());
            fmap.chk();
        }
    }

    fn split_test<K: Clone + Ord, V: Clone>(mut fmap: AvlMap<K, V>, k: &K) {
        let rhs = fmap.split_off(k);
        fmap.chk();
        rhs.chk();
        assert!(fmap.last_key_value().map_or(true, |(k2, _)| k2 < k));
        assert!(rhs.first_key_value().map_or(true, |(k2, _)| k <= k2));
    }

    // systematically try deleting each element of fmap
    fn chk_all_removes(fmap: AvlMap<u8, u8>) {
        for (k, v) in fmap.iter() {
            let mut fmap2 = fmap.clone();
            assert_eq!(fmap2.remove(k), Some(*v));
            fmap2.chk();
        }
    }

    #[test]
    fn rm_each_test() {
        // build map in order to encourage skewing
        let fmap: AvlMap<_, _> = (0..32).map(|x| (x, x + 100)).collect();
        chk_all_removes(fmap);

        // build map in reverse order to encourage opposite skewing
        let fmap: AvlMap<_, _> = (0..32).rev().map(|x| (x, x + 100)).collect();
        chk_all_removes(fmap);
    }

    #[test]
    fn iter_mut_test() {
        let mut m: AvlMap<_, _> = (0..8).map(|x| (x, 0)).collect();

        for (i, (k, v)) in m.iter_mut().enumerate() {
            assert_eq!(i, *k);
            assert_eq!(0, *v);
            *v = 1;
        }

        m.chk();

        for (i, (k, v)) in m.iter().enumerate() {
            assert_eq!(i, *k);
            assert_eq!(1, *v);
        }
    }

    #[test]
    fn bal_test_regr1() {
        bal_test(vec![(4, 0), (0, 0), (5, 0), (1, 0), (2, 0), (3, 0)]);
    }

    #[test]
    fn bal_test_regr2() {
        bal_test(vec![(3, 0), (0, 0), (1, 0), (2, 0), (4, 0)]);
    }

    #[test]
    fn bal_test_regr3() {
        bal_test(vec![
            (127, 0),
            (3, 0),
            (1, 0),
            (4, 0),
            (6, 0),
            (2, 0),
            (5, 0),
            (127, 0),
        ]);
    }

    #[test]
    fn rm_test_regr1() {
        rm_test(vec![(101, 0), (100, 0), (1, 0), (-100, 0)]);
    }

    #[test]
    fn rm_test_regr2() {
        rm_test(vec![
            (99, 0),
            (1, 0),
            (103, 0),
            (3, 0),
            (98, 0),
            (2, 0),
            (8, 0),
            (4, 0),
            (5, 0),
            (6, 0),
            (7, 0),
            (102, 0),
            (9, 0),
            (97, 0),
            (-102, 0),
            (10, 0),
            (-97, 0),
        ]);
    }

    #[test]
    fn rm_test_regr3() {
        rm_test(vec![
            (31, 0),
            (14, 0),
            (1, 0),
            (15, 0),
            (32, 0),
            (16, 0),
            (17, 0),
            (-14, 0),
            (-31, 0),
        ]);
    }

    #[test]
    fn iter_len_test() {
        let fmap: AvlMap<_, _> = (0..10).map(|i| (i, ())).collect();

        let mut iter = fmap.iter();
        let mut cnt = 10;
        while iter.next().is_some() {
            assert_eq!(iter.len(), cnt - 1);
            cnt -= 1;
        }
    }

    type TestEntries = Vec<(u8, u16)>;

    fn intersection_test(v1: TestEntries, v2: TestEntries) {
        let f1 = AvlMap::from_iter(v1.into_iter());
        let f2 = AvlMap::from_iter(v2.into_iter());
        let both = AvlMap::new_intersect(f1.clone(), f2.clone());

        for (k, v) in both.iter() {
            assert_eq!(f1.get(k), Some(v));
            assert!(f2.contains_key(k));
        }

        for (k, v) in f1.iter() {
            if f2.contains_key(k) {
                assert_eq!(both.get(k), Some(v));
            }
        }

        for (k, _) in f2.iter() {
            assert_eq!(f1.contains_key(k), both.contains_key(k));
        }
    }

    // fn union_test(v1: TestEntries, v2: TestEntries) -> () {
    //     let f1 = AvlMap::from_iter(v1.into_iter());
    //     let f2 = AvlMap::from_iter(v2.into_iter());
    //     let either = AvlMap::new_union(f1.clone(), f2.clone());

    //     assert!(either.iter().all(|(k, _)| f1.contains(k) || f2.contains(k)));
    //     f1.iter().for_each(|(k, _)| assert!(either.contains(k)));
    //     f2.iter()
    //         .for_each(|(k, v)| assert_eq!(Some(v), either.get(k)));
    // }

    fn diff_test(v1: TestEntries, v2: TestEntries) {
        let f1 = AvlMap::from_iter(v1.into_iter());
        let f2 = AvlMap::from_iter(v2.into_iter());
        let diff = AvlMap::new_diff(f1.clone(), f2.clone());

        for (k, v) in diff.iter() {
            assert_eq!(f1.get(k), Some(v));
            assert!(!f2.contains_key(k));
        }

        for (k, v) in f1.iter() {
            assert!(f2.contains_key(k) || diff.get(k) == Some(v));
        }

        assert!(f2.iter().all(|(k, _)| !diff.contains_key(k)));
    }

    fn sym_diff_test(v1: TestEntries, v2: TestEntries) {
        let f1 = AvlMap::from_iter(v1.into_iter());
        let f2 = AvlMap::from_iter(v2.into_iter());
        let sym_diff = AvlMap::new_sym_diff(f1.clone(), f2.clone());

        for (k, v) in sym_diff.iter() {
            if !f2.contains_key(k) {
                assert_eq!(f1.get(k), Some(v));
            } else {
                assert_eq!(f2.get(k), Some(v));
            }
        }

        for (k, v) in f1.iter() {
            assert!(f2.contains_key(k) || sym_diff.get(k) == Some(v));
        }

        for (k, v) in f2.iter() {
            assert!(f1.contains_key(k) || sym_diff.get(k) == Some(v));
        }
    }

    #[test]
    fn intersection_regr1() {
        let vs1 = vec![(5, 0), (6, 0)];
        let mut vs2 = vec![(4, 0), (0, 0), (12, 0), (1, 0), (13, 0), (7, 0)];
        vs2.extend([(8, 0), (2, 0), (3, 0), (15, 0), (5, 0), (16, 0), (14, 0)]);
        vs2.extend([(9, 0), (17, 0), (10, 0), (18, 0), (6, 0)]);
        vs2.extend([(11, 0), (19, 0)]);
        intersection_test(vs1, vs2)
    }

    #[test]
    fn intersection_regr2() {
        let vs1 = vec![(11, 0), (12, 0), (0, 0), (5, 0)];
        let mut vs2 = vec![(9, 0), (1, 0), (10, 0), (11, 0), (12, 0), (3, 0)];
        vs2.extend([(4, 0), (13, 0), (6, 0), (2, 0), (7, 0), (8, 0), (5, 0)]);
        intersection_test(vs1, vs2)
    }

    #[test]
    fn intersection_regr3() {
        intersection_test(
            vec![(1, 0), (255, 0), (0, 0)],
            vec![(0, 0), (255, 0)],
        );
    }

    fn into_iter_test(vs: Vec<u8>) {
        let f1: AvlMap<_, _> = vs.iter().map(|&k| (k, ())).collect();
        let m1: std::collections::BTreeMap<_, _> =
            vs.iter().map(|&k| (k, ())).collect();

        for ((x, ()), (&y, ())) in f1.into_iter().zip(m1.iter()) {
            assert_eq!(x, y);
        }
    }

    quickcheck! {
        fn qc_into_iter_test(vs: Vec<u8>) -> () {
            into_iter_test(vs);
        }

        fn qc_bal_test(vs: Vec<(u8, u32)>) -> () {
            bal_test(vs);
        }

        fn qc_rm_test(vs: Vec<(i8, u32)>) -> () {
            rm_test(vs);
        }

        fn qc_rm_test2(vs: Vec<(u8, u8)>) -> () {
            let fmap = vs.into_iter().collect();
            chk_all_removes(fmap);
        }

        fn qc_join_test(v1: Vec<u32>, v2: Vec<u32>) -> () {
            let mid = v1.len();
            let f1: AvlMap<_, _> = v1.into_iter().enumerate().collect();
            let f2: AvlMap<_, _> =
                v2.into_iter().enumerate().map(|(i,v)| (i+mid+1, v)).collect();
            let f3 = AvlMap::new_join(f1, mid, 0, f2);
            f3.chk();
        }

        fn qc_split_test(vs: Vec<(u8, u16)>) -> () {
            let f1: AvlMap<_, _> = vs.into_iter().collect();

            // try extremum splits
            split_test(f1.clone(), &u8::MIN);
            split_test(f1.clone(), &u8::MAX);

            if f1.is_empty() {
                return;
            }

            let &lb = f1.first_key_value().unwrap().0;
            let &ub = f1.last_key_value().unwrap().0;
            for k in lb..=ub {
                split_test(f1.clone(), &k);
            }

            // one final test on a map that has sole ownership of everything
            let mid = lb + (ub - lb) / 2;
            split_test(f1, &mid);
        }

        fn qc_intersection_test(v1: TestEntries, v2: TestEntries) -> () {
            intersection_test(v1, v2);
        }

        // fn qc_union_test(v1: TestEntries, v2: TestEntries) -> () {
        //     union_test(v1, v2);
        // }

        fn qc_diff_test(v1: TestEntries, v2: TestEntries) -> () {
            diff_test(v1, v2);
        }

        fn qc_sym_diff_test(v1: TestEntries, v2: TestEntries) -> () {
            sym_diff_test(v1, v2);
        }
    }
}
