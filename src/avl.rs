#![warn(missing_docs)]
use std::borrow::Borrow;
use std::cmp::Ordering::*;
use std::fmt::{Debug, Formatter};
use std::iter::FusedIterator;
use std::mem::replace;
use std::rc::Rc;

type OptNode<K, V> = Option<Rc<Node<K, V>>>;
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
        Some(Rc::new(Self::new(k, v, l, r)))
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
        K: Clone,
        V: Clone,
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
            Rc::make_mut(rc).for_each_mut(g);
        }

        g((&self.key, &mut self.val));

        if let Some(rc) = self.right.as_mut() {
            Rc::make_mut(rc).for_each_mut(g);
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

impl<K: Clone, V: Clone> Clone for Node<K, V> {
    fn clone(&self) -> Self {
        Node {
            key: self.key.clone(),
            val: self.val.clone(),
            left_ht: self.left_ht,
            right_ht: self.right_ht,
            left: self.left.clone(),
            right: self.right.clone(),
        }
    }
}

impl<K: Clone + Debug, V: Clone + Debug> Debug for Node<K, V> {
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

impl<K: Clone + Debug, V: Clone + Debug> Debug for AvlMap<K, V> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match &self.root {
            None => f.write_str("AvlMap(EMPTY)"),
            Some(rc) => {
                // use Node's Debug formatter
                f.write_fmt(format_args!("AvlMap(#{}, {:?}", self.len, rc))
            }
        }
    }
}

impl<K: Clone + Ord, V: Clone> PartialEq for AvlMap<K, V>
where
    K: PartialEq,
    V: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len()
            && self.iter().zip(other.iter()).all(|(x, y)| x == y)
    }
}

impl<K: Clone + Eq + Ord, V: Clone + Eq> Eq for AvlMap<K, V> {}

impl<K, V> PartialOrd for AvlMap<K, V>
where
    K: Clone + Ord,
    V: Clone + PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.iter().partial_cmp(other.iter())
    }
}

impl<K, V> Ord for AvlMap<K, V>
where
    K: Clone + Ord,
    V: Clone + Ord,
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.iter().cmp(other.iter())
    }
}

impl<K, V> std::hash::Hash for AvlMap<K, V>
where
    K: Clone + std::hash::Hash + Ord,
    V: Clone + std::hash::Hash,
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.for_each(|(k, v)| {
            k.hash(state);
            v.hash(state);
        });
    }
}

impl<K, Q, V> std::ops::Index<&Q> for AvlMap<K, V>
where
    K: Borrow<Q> + Clone + Ord,
    V: Clone,
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
    match Rc::try_unwrap(old_rc) {
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
    let a = Rc::make_mut(a_rc);

    let mut b_opt = a.right.take();
    let b_rc = b_opt.as_mut().unwrap();
    let b = Rc::make_mut(b_rc);

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
    let a = Rc::make_mut(a_rc);

    let mut b_opt = a.right.take();
    let b_rc = b_opt.as_mut().unwrap();
    let b = Rc::make_mut(b_rc);

    let mut c_opt = b.left.take();
    let c_rc = c_opt.as_mut().unwrap();
    let c = Rc::make_mut(c_rc);

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
    let a = Rc::make_mut(a_rc);

    let mut b_opt = a.left.take();
    let b_rc = b_opt.as_mut().unwrap();
    let b = Rc::make_mut(b_rc);

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
    let a = Rc::make_mut(a_rc);

    let mut b_opt = a.left.take();
    let b_rc = b_opt.as_mut().unwrap();
    let b = Rc::make_mut(b_rc);

    let mut c_opt = b.right.take();
    let c_rc = c_opt.as_mut().unwrap();
    let c = Rc::make_mut(c_rc);

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
    let n = Rc::get_mut(root.as_mut().unwrap()).unwrap();

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
    let n = Rc::get_mut(root.as_mut().unwrap()).unwrap();

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
            *root = Some(Rc::new(Node::new(k, v, None, None)));
            return (None, IsTaller(true)); // *** EARLY RETURN ***
        }

        Some(rc) => Rc::make_mut(rc),
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
    K: Clone + Ord,
    V: Clone,
{
    let n = match root.as_mut() {
        None => return (None, IsShorter(false)), // *** EARLY RETURN ***
        Some(rc) => Rc::make_mut(rc),
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
        Some(rc) => Rc::make_mut(rc),
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

fn join_rt<K: Clone + Ord, V: Clone>(
    left: Rc<Node<K, V>>,
    k: K,
    v: V,
    opt_right: OptNode<K, V>,
) -> OptNode<K, V> {
    assert!(left.height() > height(&opt_right) + 1);

    // ultimately, we return a clone of left with the right branch replaced
    let mut t2 = left;
    let t2n = Rc::make_mut(&mut t2);

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
    right: Rc<Node<K, V>>,
) -> OptNode<K, V> {
    assert!(right.height() > height(&opt_left) + 1);

    // ultimately, we return a clone of right with the left branch replaced
    let mut t2 = right;
    let t2n = Rc::make_mut(&mut t2);

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

        let t1 = match Rc::try_unwrap(opt_t1.unwrap()) {
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
        deconflict: &mut |lhs, _| Some(lhs),
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
            // pieces out of the node, we might avoid some cloning & Rc updates.
            let n = match Rc::try_unwrap(rc) {
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

impl<K: Clone + Ord, V: Clone> AvlMap<K, V> {
    /// Drops all elements from the map.
    pub fn clear(&mut self) {
        self.len = 0;
        self.root = None;
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
        let mut work = Vec::new();
        let mut curr = self.root.as_ref();
        while let Some(n) = curr {
            work.push(n);
            curr = n.left.as_ref();
        }

        Iter {
            work,
            len: self.len,
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
    pub fn iter_mut(&mut self) -> IterMut<K, V> {
        let work = match self.root.as_mut() {
            Some(rc) => vec![IterMutAction::Descend(rc)],
            None => Vec::new(),
        };

        IterMut {
            work,
            len: self.len,
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
    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.iter().map(|p| p.0)
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
    pub fn values(&self) -> impl Iterator<Item = &V> {
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
    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut V> {
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
    /// This method provides partial compensation for the absence of iter_mut().
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
    pub fn for_each_mut<F: FnMut((&K, &mut V))>(&mut self, mut f: F) {
        if let Some(rc) = self.root.as_mut() {
            Rc::make_mut(rc).for_each_mut(&mut f);
        }
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
    pub fn insert(&mut self, key: K, val: V) -> Option<V> {
        let (ret, _) = ins(&mut self.root, key, val);
        self.len += ret.is_none() as usize;
        ret
    }

    /// Removes a key from a map and returns the unmapped value.
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
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        if let (opt_v @ Some(_), _) = rm(&mut self.root, key) {
            self.len -= 1;
            chk_map!(&self);
            opt_v.map(|e| e.1)
        } else {
            None
        }
    }

    /// Join the RHS into this map.
    ///
    /// Requires:
    ///    self.last_key_value().map_or(true, |(x,_)| x < key);
    ///    rhs.first_key_value().map_or(true, |(y,_)| key < y);
    pub fn join_with(&mut self, key: K, val: V, mut rhs: AvlMap<K, V>) {
        assert!(self.last_key_value().map_or(true, |(k2, _)| *k2 < key));
        assert!(rhs.first_key_value().map_or(true, |(k2, _)| key < *k2));

        self.len += 1 + rhs.len();
        self.root = join(self.root.take(), key, val, rhs.root.take());
    }

    /// Build a new map by joining two maps around a pivot key that divides the
    /// entries of the maps.
    ///
    /// The constructed map contains the entries from both maps and the pivot
    /// key and the value provided for the pivot key.
    ///
    /// Requires:
    ///    lhs.last_key_value().map_or(true, |(m,_)| m < key);
    ///    rhs.first_key_value().map_or(true, |(n,_)| key < n);
    ///
    /// # Examples
    /// ```
    /// use lazy_clone_collections::AvlMap;
    ///
    /// let f1 = AvlMap::from([(0, 'a'), (1, 'b')]);
    /// let f2 = AvlMap::from([(3, 'd')]);
    /// let f3 = AvlMap::join(&f1, 2, 'c', &f2);
    /// assert_eq!(f3.get(&0), Some(&'a'));
    /// assert_eq!(f3.get(&2), Some(&'c'));
    /// assert_eq!(f3.get(&3), Some(&'d'));
    /// ```
    pub fn join(lhs: &Self, key: K, val: V, rhs: &Self) -> Self {
        assert!(lhs.last_key_value().map_or(true, |(k2, _)| *k2 < key));
        assert!(rhs.first_key_value().map_or(true, |(k2, _)| key < *k2));

        let mut lhs = lhs.clone();
        lhs.join_with(key, val, rhs.clone());
        lhs
    }

    // FIXME: this is not the semantics of BTreeMap.
    /// Moves all elements greater than a key into a new map and returns the
    /// original key-value pair (if present) and the new map.
    ///
    /// # Examples
    /// ```
    /// use lazy_clone_collections::AvlMap;
    ///
    /// let mut fmap: AvlMap<_, _> = (0..10).map(|i| (i, i * 2)).collect();
    /// let (orig_kv, higher_fives) = fmap.split_off(&5);
    /// assert_eq!(orig_kv, Some((5, 10)));
    /// assert_eq!(fmap.get(&4), Some(&8));
    /// assert_eq!(fmap.get(&6), None);
    /// assert_eq!(higher_fives.get(&6), Some(&12));
    /// ```
    pub fn split_off<Q>(&mut self, key: &Q) -> (Option<(K, V)>, Self)
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let (lhs, orig_kv, rhs) = split(self.root.take(), key);
        self.len = len(&lhs);
        self.root = lhs;

        let rhs_len = len(&rhs);
        let rhs = AvlMap {
            len: rhs_len,
            root: rhs,
        };

        (orig_kv, rhs)
    }

    /// Splits a map on a key returning one map with entries less than the key,
    /// one map with entries greater than the key, and the entry corresponding
    /// to the key.
    ///
    /// # Examples
    /// ```
    /// use lazy_clone_collections::AvlMap;
    ///
    /// let fmap = AvlMap::from([(0,1),(1,2),(2,3)]);
    /// let (lt, kv, gt) = AvlMap::split(&fmap, &1);
    /// assert_eq!(kv, Some((1, 2)));
    /// assert_eq!(lt.get(&0), Some(&1));
    /// assert_eq!(gt.get(&2), Some(&3));
    /// ```
    pub fn split<Q>(map: &Self, key: &Q) -> (Self, Option<(K, V)>, Self)
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let mut lhs = map.clone();
        let (orig_kv, rhs) = lhs.split_off(key);
        (lhs, orig_kv, rhs)
    }

    /// Removes entries with keys from the other map.
    ///
    /// # Examples
    /// ```
    /// use lazy_clone_collections::AvlMap;
    ///
    /// let mut lhs = AvlMap::from([(0,1), (1, 2)]);
    /// let rhs = AvlMap::from([(1,5), (3,4)]);
    /// lhs.diff_with(rhs);
    /// assert_eq!(lhs.get(&0), Some(&1));
    /// assert_eq!(lhs.get(&1), None);
    /// ```
    pub fn diff_with(&mut self, mut other: Self) {
        self.root = diff(self.root.take(), other.root.take());
        self.len = len(&self.root);
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
    /// let d = AvlMap::diff(&lhs, &rhs);
    /// assert_eq!(d.get(&0), Some(&1));
    /// assert_eq!(d.get(&1), None);
    /// ```
    pub fn diff(lhs: &Self, rhs: &Self) -> Self {
        let mut lhs = lhs.clone();
        lhs.diff_with(rhs.clone());
        lhs
    }

    /// Removes entries with keys from the other map and adds entries from the
    /// other map that have keys that are not in self.
    ///
    /// # Examples
    /// ```
    /// use lazy_clone_collections::AvlMap;
    ///
    /// let mut lhs = AvlMap::from([(0,1), (1, 2)]);
    /// let rhs = AvlMap::from([(1,5), (3,4)]);
    /// lhs.sym_diff_with(rhs);
    /// assert_eq!(lhs.get(&0), Some(&1));
    /// assert_eq!(lhs.get(&1), None);
    /// assert_eq!(lhs.get(&3), Some(&4));
    /// ```
    pub fn sym_diff_with(&mut self, mut other: Self) {
        self.root = sym_diff(self.root.take(), other.root.take());
        self.len = len(&self.root);
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
    /// let d = AvlMap::sym_diff(&lhs, &rhs);
    /// assert_eq!(d.get(&0), Some(&1));
    /// assert_eq!(d.get(&1), None);
    /// assert_eq!(d.get(&3), Some(&4));
    /// ```
    pub fn sym_diff(lhs: &Self, rhs: &Self) -> Self {
        let mut lhs = lhs.clone();
        lhs.sym_diff_with(rhs.clone());
        lhs
    }

    /// Discard entries that do not have a key from the other map.
    ///
    /// # Examples
    /// ```
    /// use lazy_clone_collections::AvlMap;
    ///
    /// let mut lhs = AvlMap::from([(0,1), (1, 2)]);
    /// let rhs = AvlMap::from([(1,5), (3,4)]);
    /// lhs.intersect_with(rhs);
    /// assert_eq!(lhs.get(&0), None);
    /// assert_eq!(lhs.get(&1), Some(&2));
    /// ```
    pub fn intersect_with(&mut self, mut other: Self) {
        self.root = intersect(self.root.take(), other.root.take());
        self.len = len(&self.root);
    }

    /// Creates a map with entries from the LHS that have keys in the RHS.
    ///
    /// # Examples
    /// ```
    /// use lazy_clone_collections::AvlMap;
    ///
    /// let lhs = AvlMap::from([(0,1), (1, 2)]);
    /// let rhs = AvlMap::from([(1,5), (3,4)]);
    /// let i = AvlMap::intersect(&lhs, &rhs);
    /// assert_eq!(i.get(&0), None);
    /// assert_eq!(i.get(&1), Some(&2));
    /// ```
    pub fn intersect(lhs: &Self, rhs: &Self) -> Self {
        let mut lhs = lhs.clone();
        lhs.intersect_with(rhs.clone());
        lhs
    }

    /// Adds the entries from other that don't have keys in this map.
    ///
    /// # Examples
    /// ```
    /// use lazy_clone_collections::AvlMap;
    ///
    /// let mut lhs = AvlMap::from([(0,1), (1, 2)]);
    /// let rhs = AvlMap::from([(1,5), (3,4)]);
    /// lhs.union_with(rhs);
    /// assert_eq!(lhs.get(&0), Some(&1));
    /// assert_eq!(lhs.get(&1), Some(&2));
    /// assert_eq!(lhs.get(&3), Some(&4));
    /// ```
    pub fn union_with(&mut self, mut other: Self) {
        self.root = union(self.root.take(), other.root.take());
        self.len = len(&self.root);
    }

    /// Builds a map with entries from both maps, with entries from the LHS
    /// taking precedence when a key appears in both maps.
    ///
    /// # Examples
    /// ```
    /// use lazy_clone_collections::AvlMap;
    ///
    /// let mut lhs = AvlMap::from([(0,1), (1, 2)]);
    /// let rhs = AvlMap::from([(1,5), (3,4)]);
    /// lhs.union_with(rhs);
    /// assert_eq!(lhs.get(&0), Some(&1));
    /// assert_eq!(lhs.get(&1), Some(&2));
    /// assert_eq!(lhs.get(&3), Some(&4));
    /// ```
    pub fn union(lhs: &Self, rhs: &Self) -> Self {
        let mut lhs = lhs.clone();
        lhs.union_with(rhs.clone());
        lhs
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
        let mut prev = &None;
        let mut curr = &self.root;
        while let Some(rc) = curr.as_ref() {
            prev = curr;
            curr = &rc.left;
        }
        prev.as_ref().map(|rc| (&rc.key, &rc.val))
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

    /// Tests if self contains an entry for the given key.
    pub fn contains<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        self.get(key).is_some()
    }

    /// Returns a reference to the value associated with k.
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
    pub fn get<Q>(&self, k: &Q) -> Option<&V>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        let mut curr = &self.root;
        while let Some(n) = curr {
            match k.cmp(n.key.borrow()) {
                Less => curr = &n.left,
                Equal => return Some(&n.val),
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
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        let mut curr = &mut self.root;
        while let Some(rc) = curr {
            let n = Rc::make_mut(rc);
            match k.cmp(n.key.borrow()) {
                Less => curr = &mut n.left,
                Equal => return Some(&mut n.val),
                Greater => curr = &mut n.right,
            }
        }

        None
    }

    /// Returns an Entry that simplifies some update operations.
    pub fn entry(&mut self, key: K) -> Entry<'_, K, V> {
        // TODO: frustrating that this traverses the tree twice
        if self.contains(&key) {
            let val = self.get_mut(&key).unwrap();
            Entry::Occupied(OccupiedEntry { key, val })
        } else {
            Entry::Vacant(VacantEntry { key, map: self })
        }
    }

    /// Returns true if self contains no entries, false otherwise.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the number of entries in self.
    pub fn len(&self) -> usize {
        self.len
    }

    #[cfg(test)]
    fn chk(&self) {
        assert_eq!(self.len, chk(&self.root, None).0);
    }
}

impl<K: Clone + Ord, V: Clone> std::ops::BitAnd for AvlMap<K, V> {
    type Output = AvlMap<K, V>;

    fn bitand(mut self, rhs: Self) -> Self::Output {
        self.intersect_with(rhs);
        self
    }
}

impl<K: Clone + Ord, V: Clone> std::ops::BitAnd for &AvlMap<K, V> {
    type Output = AvlMap<K, V>;

    fn bitand(self, rhs: Self) -> Self::Output {
        let mut res = self.clone();
        res.intersect_with(rhs.clone());
        res
    }
}

impl<K: Clone + Ord, V: Clone> Default for AvlMap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

pub struct Iter<'a, K, V> {
    // TODO: use FunStack for the spine. Vec will be more performant, but users
    // may expect our promise about "cheap cloning" to apply to the iterators.
    work: Vec<&'a Rc<Node<K, V>>>,
    len: usize,
}

impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        self.work.pop().map(|n| {
            self.len -= 1;
            let entry = (&n.key, &n.val);
            let mut curr = n.right.as_ref();
            while let Some(m) = curr {
                self.work.push(m);
                curr = m.left.as_ref();
            }
            entry
        })
    }
}

impl<'a, K, V> ExactSizeIterator for Iter<'a, K, V> {
    fn len(&self) -> usize {
        self.len
    }
}

impl<'a, K, V> FusedIterator for Iter<'a, K, V> {}

enum IterMutAction<'a, K, V> {
    Descend(&'a mut Rc<Node<K, V>>),
    Return((&'a K, &'a mut V)),
}

pub struct IterMut<'a, K, V> {
    work: Vec<IterMutAction<'a, K, V>>,
    len: usize,
}

impl<'a, K, V> Iterator for IterMut<'a, K, V>
where
    K: Clone,
    V: Clone,
{
    type Item = (&'a K, &'a mut V);

    fn next(&mut self) -> Option<Self::Item> {
        use IterMutAction::*;

        match self.work.pop() {
            Some(Descend(rc)) => {
                let mut n = Rc::make_mut(rc);

                loop {
                    if let Some(rt) = n.right.as_mut() {
                        self.work.push(Descend(rt));
                    }

                    if let Some(lf) = n.left.as_mut() {
                        self.work.push(Return((&n.key, &mut n.val)));
                        n = Rc::make_mut(lf);
                    } else {
                        break;
                    }
                }

                self.len -= 1;
                Some((&n.key, &mut n.val))
            }

            Some(IterMutAction::Return(ret)) => {
                self.len -= 1;
                Some(ret)
            }

            None => {
                assert_eq!(self.len, 0);
                None
            }
        }
    }
}

impl<'a, K, V> ExactSizeIterator for IterMut<'a, K, V>
where
    K: Clone,
    V: Clone,
{
    fn len(&self) -> usize {
        self.len
    }
}

impl<'a, K: Clone, V: Clone> FusedIterator for IterMut<'a, K, V> {}

enum MergeItem<'a, K, V> {
    LeftOnly((&'a K, &'a V)),
    Both((&'a K, &'a V)),
    RightOnly((&'a K, &'a V)),
}

struct MergeIter<'a, K, V> {
    lhs: Iter<'a, K, V>,
    rhs: Iter<'a, K, V>,
}

impl<'a, K: Ord, V> Iterator for MergeIter<'a, K, V> {
    type Item = MergeItem<'a, K, V>;

    fn next(&mut self) -> Option<Self::Item> {
        let Some(peek_lhs) = self.lhs.work.last() else {
            return self.rhs.next().map(|e| MergeItem::RightOnly(e));
        };

        let Some(peek_rhs) = self.rhs.work.last() else {
            return self.lhs.next().map(|e| MergeItem::LeftOnly(e));
        };

        match peek_lhs.key.cmp(&peek_rhs.key) {
            Less => self.lhs.next().map(|e| MergeItem::LeftOnly(e)),

            Equal => {
                self.lhs.next();
                self.rhs.next().map(|e| MergeItem::Both(e))
            }

            Greater => self.rhs.next().map(|e| MergeItem::RightOnly(e)),
        }
    }
}

impl<'a, K: Ord, V> FusedIterator for MergeIter<'a, K, V> {}

pub struct OccupiedEntry<'a, K, V> {
    key: K,
    val: &'a mut V,
}

impl<'a, K, V: Clone> OccupiedEntry<'a, K, V> {
    pub fn get(&self) -> &V {
        self.val
    }

    pub fn get_mut(&mut self) -> &mut V {
        self.val
    }

    pub fn insert(&mut self, new_val: V) -> V {
        std::mem::replace(self.val, new_val)
    }

    pub fn into_mut(self) -> &'a mut V {
        self.val
    }

    pub fn key(&self) -> &K {
        &self.key
    }

    pub fn remove(self) -> V {
        self.val.clone()
    }

    pub fn remove_entry(self) -> (K, V) {
        (self.key, self.val.clone())
    }
}

pub struct VacantEntry<'a, K, V> {
    key: K,
    map: &'a mut AvlMap<K, V>,
}

impl<'a, K: Clone + Ord, V: Clone> VacantEntry<'a, K, V> {
    pub fn insert(self, val: V) -> &'a mut V {
        // TODO: the clone() here is lamentable
        self.map.insert(self.key.clone(), val);
        self.map.get_mut(&self.key).unwrap()
    }

    pub fn into_key(self) -> K {
        self.key
    }

    pub fn key(&self) -> &K {
        &self.key
    }
}

pub enum Entry<'a, K, V> {
    Occupied(OccupiedEntry<'a, K, V>),
    Vacant(VacantEntry<'a, K, V>),
}

impl<'a, K, V: Clone> Entry<'a, K, V> {
    pub fn and_modify<F>(mut self, f: F) -> Self
    where
        F: FnOnce(&mut V),
    {
        if let Entry::Occupied(occ) = &mut self {
            f(occ.val);
        }

        self
    }

    pub fn key(&self) -> &K {
        match self {
            Entry::Occupied(x) => &x.key,
            Entry::Vacant(x) => &x.key,
        }
    }

    pub fn or_default(self) -> &'a mut V
    where
        K: Clone + Ord,
        V: Clone + Default,
    {
        match self {
            Entry::Occupied(x) => x.into_mut(),
            Entry::Vacant(x) => x.insert(V::default()),
        }
    }

    pub fn or_insert(self, default: V) -> &'a mut V
    where
        K: Clone + Ord,
    {
        match self {
            Entry::Occupied(x) => x.into_mut(),
            Entry::Vacant(x) => x.insert(default),
        }
    }

    pub fn or_insert_with<F: FnOnce() -> V>(self, default: F) -> &'a mut V
    where
        K: Clone + Ord,
    {
        match self {
            Entry::Occupied(x) => x.into_mut(),
            Entry::Vacant(x) => x.insert(default()),
        }
    }

    pub fn or_insert_with_key<F: FnOnce(&K) -> V>(self, default: F) -> &'a mut V
    where
        K: Clone + Ord,
    {
        match self {
            Entry::Occupied(x) => x.into_mut(),
            Entry::Vacant(x) => {
                let v = default(&x.key);
                x.insert(v)
            }
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

/// A sorted set of values.
///
/// The implementation is mostly a thin wrapper around [`AvlMap`].
#[derive(Clone)]
pub struct AvlSet<V>(AvlMap<V, ()>);

impl<V: Clone + Ord> AvlSet<V> {
    /// Removes all the entries from self.
    pub fn clear(&mut self) {
        self.0.clear();
    }

    /// Tests if self contains the given value.
    pub fn contains<Q>(&self, value: &Q) -> bool
    where
        V: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.0.contains(value)
    }

    /// Returns an iterator over elements in self and not in other
    pub fn difference<'a>(&'a self, other: &'a Self) -> Difference<'a, V> {
        Difference(MergeIter {
            lhs: self.0.iter(),
            rhs: other.0.iter(),
        })
    }

    /// Returns the least value in the set.
    pub fn first(&self) -> Option<&V> {
        self.0.first_key_value().map(|(k, _)| k)
    }

    /// Returns a reference to the element matching value, if it exists
    pub fn get<Q>(&self, value: &Q) -> Option<&V>
    where
        V: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let mut curr = &self.0.root;
        while let Some(n) = curr {
            match value.cmp(n.key.borrow()) {
                Less => curr = &n.left,
                Equal => return Some(&n.key),
                Greater => curr = &n.right,
            }
        }

        None
    }

    /// Inserts the given value and returns true if self did not already have
    /// the value and returns false otherwise.
    pub fn insert(&mut self, value: V) -> bool {
        self.0.insert(value, ()).is_none()
    }

    /// Returns an iterator of the values that are in both self and other.
    pub fn intersection<'a>(&'a self, other: &'a Self) -> Intersection<'a, V> {
        Intersection(MergeIter {
            lhs: self.0.iter(),
            rhs: other.0.iter(),
        })
    }

    /// Returns true if self and other have no common values and false otherwise
    pub fn is_disjoint(&self, other: &Self) -> bool {
        self.intersection(other).next().is_none()
    }

    /// Returns true if self is the empty set, false otherwise.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Tests if self is a subset of other.
    pub fn is_subset(&self, other: &Self) -> bool {
        if self.is_empty() {
            return true;
        } else if self.len() > other.len() {
            return false;
        }

        fn has_all<K, V>(
            lhs: &Rc<Node<K, V>>,
            w: &mut Vec<(&Rc<Node<K, V>>, bool)>,
        ) -> bool
        where
            K: Ord,
        {
            // Strategy: we do an inorder recursive traversal of lhs (the
            // suspected subset). Simultaneously, we keep an 'iterator' for the
            // rhs (the suspected superset).  We also update the iterator
            // 'in-order' but we look for opportunities to fast-forward.

            // first traverse to lhs's left child
            if let Some(lhs_left) = lhs.left.as_ref() {
                if !has_all(lhs_left, w) {
                    return false;
                }
            }

            // now, look for lhs.key in rhs using our iterator
            while let Some((rhs, is_left_done)) = w.last_mut() {
                // pointer check to short circuit further comparisons
                if Rc::ptr_eq(lhs, rhs) {
                    w.pop();
                    return true;
                }

                match lhs.key.cmp(&rhs.key) {
                    Less => {
                        // The only way we can find a lesser key than rhs.key
                        // is to traverse to its left.  If we already matched
                        // all of the keys there, it won't help to look again.
                        if *is_left_done {
                            return false;
                        }

                        *is_left_done = true;
                        let rhs_left = rhs.left.as_ref().unwrap();
                        w.push((rhs_left, rhs_left.left.is_none()));
                    }

                    Equal => {
                        // We visited all the lesser keys on the lhs and we
                        // do not need any remaining lesser keys on the rhs.
                        // Move on to greater keys.
                        let rhs = w.pop().unwrap().0;
                        if let Some(rhs_right) = rhs.right.as_ref() {
                            w.push((rhs_right, rhs_right.left.is_none()));
                        }

                        if let Some(lhs_right) = lhs.right.as_ref() {
                            return has_all(lhs_right, w);
                        } else {
                            return true;
                        }
                    }

                    Greater => {
                        // the greater nodes in the rhs are to the right or up
                        let rhs = w.pop().unwrap().0;
                        if let Some(rhs_right) = rhs.right.as_ref() {
                            w.push((rhs_right, rhs_right.left.is_none()));
                        }
                    }
                }
            }

            false
        }

        let lhs = self.0.root.as_ref().unwrap();
        let rhs = other.0.root.as_ref().unwrap();
        has_all(lhs, &mut vec![(rhs, rhs.left.is_none())])
    }

    /// tests if self is a superset of other.
    pub fn is_superset(&self, other: &Self) -> bool {
        other.is_subset(self)
    }

    /// Returns an iterator over self's values in sorted order.
    pub fn iter(&self) -> impl Iterator<Item = &V> {
        self.0.iter().map(|(k, _)| k)
    }

    /// Returns the greatest value in self.
    pub fn last(&self) -> Option<&V> {
        self.0.last_key_value().map(|e| e.0)
    }

    /// Returns the number of elements in self.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    // TODO: range

    /// Removes the given value from self returning true if the value was
    /// present and false otherwise.
    pub fn remove<Q>(&mut self, value: &Q) -> bool
    where
        V: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.0.remove(value).is_some()
    }

    /// Replace and return the matching value in the map.
    pub fn replace(&mut self, value: V) -> Option<V> {
        // TODO: adapt insert so we don't need multiple calls.
        let ret = self.take(&value);
        self.insert(value);
        ret
    }

    /// Retain values for which f returns true and discard others
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&V) -> bool,
    {
        fn dfs<V, F>(n: Node<V, ()>, f: &mut F, acc: &mut AvlMap<V, ()>)
        where
            F: FnMut(&V) -> bool,
            V: Clone + Ord,
        {
            if let Some(mut lf) = n.left {
                Rc::make_mut(&mut lf);
                if let Ok(x) = Rc::try_unwrap(lf) {
                    dfs(x, f, acc);
                }
            }

            if f(&n.key) {
                acc.insert(n.key, ());
            }

            if let Some(mut rt) = n.right {
                Rc::make_mut(&mut rt);
                if let Ok(x) = Rc::try_unwrap(rt) {
                    dfs(x, f, acc);
                }
            }
        }

        let Some(mut root) = self.0.root.take() else { return; };
        Rc::make_mut(&mut root);
        let Ok(root) = Rc::try_unwrap(root) else { panic!("try_unwrap fail?") };
        let mut acc = AvlMap::new();
        dfs(root, &mut f, &mut acc);
        self.0 = acc;
    }

    /// Removes all elements greater or equal to key and returns them.
    pub fn split_off<Q>(&mut self, key: &Q) -> Self
    where
        V: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let (old_elt, mut gt_map) = self.0.split_off(key);
        if let Some((k, v)) = old_elt {
            gt_map.insert(k, v);
        }
        Self(gt_map)
    }

    /// Returns an iterator over elements in self or other but not both.
    pub fn symmetric_difference<'a>(
        &'a self,
        other: &'a Self,
    ) -> SymmetricDifference<'a, V> {
        SymmetricDifference(MergeIter {
            lhs: self.0.iter(),
            rhs: other.0.iter(),
        })
    }

    /// Removes and returns the set member that matches value.
    pub fn take<Q>(&mut self, value: &Q) -> Option<V>
    where
        V: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        if let (opt_v @ Some(_), _) = rm(&mut self.0.root, value) {
            self.0.len -= 1;
            opt_v.map(|e| e.0)
        } else {
            None
        }
    }

    /// Returns an iterator over the elements of self and other, ordered by key.
    ///
    /// Common elements are only returned once.
    pub fn union<'a>(&'a self, other: &'a Self) -> Union<'a, V> {
        Union(MergeIter {
            lhs: self.0.iter(),
            rhs: other.0.iter(),
        })
    }

    /// Returns a new, empty set.
    pub fn new() -> Self {
        Self(AvlMap::new())
    }
}

impl<K: Clone + Ord> Default for AvlSet<K> {
    fn default() -> Self {
        Self::new()
    }
}

macro_rules! make_set_iter {
    ($name:ident, $rtn_left:expr, $rtn_both:expr, $rtn_right:expr) => {
        pub struct $name<'a, T>(MergeIter<'a, T, ()>);

        impl<'a, T: Ord> Iterator for $name<'a, T> {
            type Item = &'a T;

            fn next(&mut self) -> Option<Self::Item> {
                while let Some(mi) = self.0.next() {
                    match mi {
                        MergeItem::LeftOnly(e) => {
                            if $rtn_left {
                                return Some(e.0);
                            }
                        }

                        MergeItem::Both(e) => {
                            if $rtn_both {
                                return Some(e.0);
                            }
                        }

                        MergeItem::RightOnly(e) => {
                            if $rtn_right {
                                return Some(e.0);
                            }
                        }
                    }
                }

                None
            }
        }

        impl<'a, T: Ord> FusedIterator for $name<'a, T> {}
    };
}

make_set_iter!(Difference, true, false, false);
make_set_iter!(Intersection, false, true, false);
make_set_iter!(SymmetricDifference, true, false, true);
make_set_iter!(Union, true, true, true);

impl<T: Clone + Ord> Extend<T> for AvlSet<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for x in iter {
            self.insert(x);
        }
    }
}

impl<T: Clone + Ord> FromIterator<T> for AvlSet<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut s = AvlSet::new();
        s.extend(iter);
        s
    }
}

#[cfg(test)]
mod test {
    extern crate quickcheck;
    use super::*;
    use quickcheck::quickcheck;

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
        let (kv, rhs) = fmap.split_off(&k);
        fmap.chk();
        rhs.chk();
        assert!(kv.map_or(true, |(k2, _)| k2 == *k));
        assert!(fmap.last_key_value().map_or(true, |(k2, _)| k2 < k));
        assert!(rhs.first_key_value().map_or(true, |(k2, _)| k < k2));
    }

    // systematically try deleting each element of fmap
    fn chk_all_removes(fmap: AvlMap<u8, u8>) {
        for (k, v) in fmap.clone().iter() {
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

    #[test]
    fn intersect_test() {
        let mut lhs = AvlMap::from([(0, 1), (1, 2)]);
        let rhs = AvlMap::from([(1, 5), (3, 4)]);
        println!("{:?}", lhs);
        println!("{:?}", rhs);
        lhs.intersect_with(rhs);
        assert_eq!(lhs.get(&0), None);
        assert_eq!(lhs.get(&1), Some(&2));
    }

    type TestEntries = Vec<(u8, u16)>;

    fn intersection_test(v1: TestEntries, v2: TestEntries) -> () {
        let f1 = AvlMap::from_iter(v1.into_iter());
        let f2 = AvlMap::from_iter(v2.into_iter());
        let both = AvlMap::intersect(&f1, &f2);

        for (k, v) in both.iter() {
            assert_eq!(f1.get(k), Some(v));
            assert!(f2.contains(k));
        }

        for (k, v) in f1.iter() {
            if f2.contains(k) {
                assert_eq!(both.get(k), Some(v));
            }
        }

        for (k, _) in f2.iter() {
            assert_eq!(f1.contains(k), both.contains(k));
        }
    }

    fn union_test(v1: TestEntries, v2: TestEntries) -> () {
        let f1 = AvlMap::from_iter(v1.into_iter());
        let f2 = AvlMap::from_iter(v2.into_iter());
        let either = AvlMap::union(&f1, &f2);

        assert!(either.iter().all(|(k, _)| f1.contains(k) || f2.contains(k)));
        f1.iter()
            .for_each(|(k, v)| assert_eq!(Some(v), either.get(k)));
        f2.iter().for_each(|(k, _)| assert!(either.contains(k)));
    }

    fn diff_test(v1: TestEntries, v2: TestEntries) -> () {
        let f1 = AvlMap::from_iter(v1.into_iter());
        let f2 = AvlMap::from_iter(v2.into_iter());
        let diff = AvlMap::diff(&f1, &f2);

        for (k, v) in diff.iter() {
            assert_eq!(f1.get(k), Some(v));
            assert!(!f2.contains(k));
        }

        for (k, v) in f1.iter() {
            assert!(f2.contains(k) || diff.get(k) == Some(v));
        }

        assert!(f2.iter().all(|(k, _)| !diff.contains(k)));
    }

    fn sym_diff_test(v1: TestEntries, v2: TestEntries) -> () {
        let f1 = AvlMap::from_iter(v1.into_iter());
        let f2 = AvlMap::from_iter(v2.into_iter());
        let sym_diff = AvlMap::sym_diff(&f1, &f2);

        for (k, v) in sym_diff.iter() {
            if !f2.contains(k) {
                assert_eq!(f1.get(k), Some(v));
            } else {
                assert_eq!(f2.get(k), Some(v));
            }
        }

        for (k, v) in f1.iter() {
            assert!(f2.contains(k) || sym_diff.get(k) == Some(v));
        }

        for (k, v) in f2.iter() {
            assert!(f1.contains(k) || sym_diff.get(k) == Some(v));
        }
    }

    fn set_checks(
        s1: &AvlSet<u8>,
        s2: &AvlSet<u8>,
        t1: &std::collections::BTreeSet<u8>,
        t2: &std::collections::BTreeSet<u8>,
    ) {
        assert!(s1.intersection(&s2).cmp(t1.intersection(&t2)).is_eq());
        assert_eq!(s1.is_disjoint(&s2), t1.is_disjoint(&t2));
        assert_eq!(s1.is_subset(&s2), t1.is_subset(&t2));
        assert_eq!(s1.is_superset(&s2), t1.is_superset(&t2));
        assert!(s1.union(&s2).cmp(t1.union(&t2)).is_eq());
        assert!(s1
            .symmetric_difference(&s2)
            .cmp(t1.symmetric_difference(&t2))
            .is_eq());
    }

    fn set_test(v1: Vec<u8>, v2: Vec<u8>) {
        let s1: AvlSet<_> = v1.clone().into_iter().collect();
        let s2: AvlSet<_> = v2.clone().into_iter().collect();

        type OtherSet = std::collections::BTreeSet<u8>;
        let t1: OtherSet = v1.into_iter().collect();
        let t2: OtherSet = v2.into_iter().collect();

        set_checks(&s1, &s2, &t1, &t2);
        set_checks(&s2, &s1, &t2, &t1);
    }

    fn set_test2(v: Vec<u8>) {
        let mut s = vec![AvlSet::new()];
        let mut t = vec![std::collections::BTreeSet::new()];

        for (i, x) in v.into_iter().enumerate() {
            for j in 0..=i {
                let mut s1 = s[j].clone();
                s1.insert(x);
                s.push(s1);

                let mut t1 = t[j].clone();
                t1.insert(x);
                t.push(t1);
            }
        }

        for (s1, t1) in s.iter().zip(t.iter()) {
            for (s2, t2) in s.iter().zip(t.iter()) {
                set_checks(s1, s2, t1, t2);
                set_checks(s2, s1, t2, t1);
            }
        }
    }

    #[test]
    fn set_test_regr1() {
        set_test(vec![], vec![0]);
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

    quickcheck! {
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
            let f3 = AvlMap::join(&f1, mid, 0, &f2);
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

        fn qc_union_test(v1: TestEntries, v2: TestEntries) -> () {
            union_test(v1, v2);
        }

        fn qc_diff_test(v1: TestEntries, v2: TestEntries) -> () {
            diff_test(v1, v2);
        }

        fn qc_sym_diff_test(v1: TestEntries, v2: TestEntries) -> () {
            sym_diff_test(v1, v2);
        }

        fn qc_set_tests(v1: Vec<u8>, v2: Vec<u8>) -> () {
            set_test(v1, v2);
        }

        fn qc_set_tests2(v1: (u8,u8,u8,u8,u8,u8)) -> () {
            set_test2(vec![v1.0, v1.1, v1.2, v1.3, v1.4, v1.5]);
        }

    }
}
