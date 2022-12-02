use std::borrow::Borrow;
use std::cmp::Ordering::*;
use std::fmt::{Debug, Formatter};
use std::iter::FusedIterator;
use std::mem::replace;
use std::rc::Rc;

type OptNode<K, V> = Option<Rc<Node<K, V>>>;
type IsShorter = bool;
type IsTaller = bool;

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
        ((self.bal() + 1) as u8) < 2
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

    fn for_each_mut<F>(&mut self, g: &mut F) -> ()
    where
        K: Clone,
        V: Clone,
        F: FnMut(&K, &mut V),
    {
        self.left
            .as_mut()
            .map(|rc| Rc::make_mut(rc).for_each_mut(g));
        g(&self.key, &mut self.val);
        self.right
            .as_mut()
            .map(|rc| Rc::make_mut(rc).for_each_mut(g));
    }
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

#[derive(Clone)]
pub struct FunMap<K: Clone, V: Clone> {
    len: usize,
    root: OptNode<K, V>,
}

impl<K: Clone + Debug, V: Clone + Debug> Debug for FunMap<K, V> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match &self.root {
            None => f.write_str("FunMap(EMPTY)"),
            Some(rc) => {
                // use Node's Debug formatter
                f.write_fmt(format_args!("FunMap(#{}, {:?}", self.len, rc))
            }
        }
    }
}

fn height<K, V>(opt_node: &OptNode<K, V>) -> i8 {
    opt_node.as_ref().map_or(0, |rc| rc.height())
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

    !b_was_bal
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
    true
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

    !b_was_bal
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

    true
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
            return (None, true); // *** EARLY RETURN ***
        }

        Some(rc) => Rc::make_mut(rc),
    };

    match k.cmp(&n.key) {
        Equal => (Some(std::mem::replace(&mut n.val, v)), false),

        Less => {
            let (old_v, is_taller) = ins(&mut n.left, k, v);
            n.left_ht += is_taller as i8;
            if is_taller && n.bal() < -1 {
                rebal_lf_to_rt(root);
                (old_v, false)
            } else {
                (old_v, is_taller && n.bal() < 0)
            }
        }

        Greater => {
            let (old_v, is_taller) = ins(&mut n.right, k, v);
            n.right_ht += is_taller as i8;
            if is_taller && n.bal() > 1 {
                rebal_rt_to_lf(root);
                (old_v, false)
            } else {
                (old_v, is_taller && n.bal() > 0)
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
        None => return (None, false), // *** EARLY RETURN ***
        Some(rc) => Rc::make_mut(rc),
    };

    if n.left.is_some() {
        let (kv, is_shorter) = rm_leftmost(&mut n.left);
        n.left_ht -= is_shorter as i8;
        if is_shorter && n.bal() > 1 {
            (kv, rebal_rt_to_lf(root))
        } else {
            (kv, is_shorter && n.bal() == 0)
        }
    } else {
        let old_n = take_node(root);
        *root = old_n.right;
        (Some((old_n.key, old_n.val)), true)
    }
}

// removes k from the map and returns the associated value and whether the
// tree at root is shorter as a result of the deletion.
fn rm<K, V, Q>(root: &mut OptNode<K, V>, k: &Q) -> (Option<V>, IsShorter)
where
    K: Borrow<Q> + Clone + Ord,
    V: Clone,
    Q: Ord + ?Sized,
{
    let n = match root.as_mut() {
        None => return (None, false), // *** EARLY RETURN ***
        Some(rc) => Rc::make_mut(rc),
    };

    match k.cmp(n.key.borrow()) {
        Less => {
            let (v, is_shorter) = rm(&mut n.left, k);
            n.left_ht -= is_shorter as i8;
            if is_shorter && n.bal() > 1 {
                (v, rebal_rt_to_lf(root))
            } else {
                (v, is_shorter && n.bal() == 0)
            }
        }

        Greater => {
            let (v, is_shorter) = rm(&mut n.right, k);
            n.right_ht -= is_shorter as i8;
            if is_shorter && n.bal() < -1 {
                (v, rebal_lf_to_rt(root))
            } else {
                (v, is_shorter && n.bal() == 0)
            }
        }

        Equal => match (&n.left, &n.right) {
            (None, None) => {
                let old_n = take_node(root);
                (Some(old_n.val), true)
            }

            (None, Some(_)) => {
                let old_n = take_node(root);
                *root = old_n.right;
                (Some(old_n.val), true)
            }

            (Some(_), None) => {
                let old_n = take_node(root);
                *root = old_n.left;
                (Some(old_n.val), true)
            }

            _ => {
                // both children are populated
                let (succ, is_shorter) = rm_leftmost(&mut n.right);
                let (succ_key, succ_val) = succ.unwrap();
                n.key = succ_key;
                let old_val = replace(&mut n.val, succ_val);

                n.right_ht -= is_shorter as i8;
                if is_shorter && n.bal() < -1 {
                    // we were taller on left and lost height on right
                    (Some(old_val), rebal_lf_to_rt(root))
                } else {
                    (Some(old_val), is_shorter && n.bal() == 0)
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
        let opt_t1 = Node::opt_new(k, v, c, opt_right.clone());
        t2n.set_right(opt_t1);

        if t2n.is_bal() {
            Some(t2)
        } else {
            if rot_rt(&mut t2n.right) {
                t2n.right_ht -= 1;
            }
            let mut opt_t2 = Some(t2);
            rot_lf(&mut opt_t2);
            assert!(opt_t2.as_ref().unwrap().is_bal());
            opt_t2
        }
    } else {
        let opt_t1 = join_rt(c.unwrap(), k, v, opt_right);
        t2n.set_right(opt_t1);
        let is_bal = t2n.is_bal();
        let mut opt_t2 = Some(t2);
        if !is_bal {
            rot_lf(&mut opt_t2);
        }

        opt_t2
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
        let opt_t1 = Node::opt_new(k, v, opt_left.clone(), c);
        t2n.set_left(opt_t1);

        if t2n.is_bal() {
            Some(t2)
        } else {
            if rot_lf(&mut t2n.left) {
                t2n.left_ht -= 1;
            }
            let mut opt_t2 = Some(t2);
            rot_rt(&mut opt_t2);
            assert!(opt_t2.as_ref().unwrap().is_bal());
            opt_t2
        }
    } else {
        let opt_t1 = join_lf(opt_left, k, v, c.unwrap());
        t2n.set_left(opt_t1);
        let is_bal = t2n.is_bal();
        let mut opt_t2 = Some(t2);
        if !is_bal {
            rot_rt(&mut opt_t2);
        }

        opt_t2
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
        Node::opt_new(k, v, opt_left.clone(), opt_right.clone())
    }
}

impl<K: Clone + Ord, V: Clone> FunMap<K, V> {
    pub fn new() -> Self {
        FunMap { len: 0, root: None }
    }

    pub fn iter(&self) -> Iter<K, V> {
        let mut spine = Vec::new();
        let mut curr = self.root.as_ref();
        while let Some(n) = curr {
            spine.push(n);
            curr = n.left.as_ref();
        }

        Iter {
            spine,
            len: self.len,
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
    /// use fun_collections::FunMap;
    ///
    /// let mut fmap = FunMap::new();
    /// fmap.insert(0, "a");
    /// fmap.for_each_mut(|_, v| *v = "b");
    /// assert_eq!(fmap.get(&0), Some(&"b"));
    /// ```
    pub fn for_each_mut<F: FnMut(&K, &mut V)>(&mut self, mut f: F) -> () {
        self.root
            .as_mut()
            .map(|rc| Rc::make_mut(rc).for_each_mut(&mut f));
    }

    /// Inserts a key-value pair in the map.
    ///
    /// # Examples
    /// ```
    /// use fun_collections::FunMap;
    ///
    /// let mut fmap = FunMap::new();
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
    /// use fun_collections::FunMap;
    ///
    /// let mut fmap = FunMap::new();
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
            opt_v
        } else {
            None
        }
    }

    /// Build a map by joining with another map around a "pivot" key that is
    /// between our key values and the other maps key values.
    ///
    /// The constructed map contains all our entries, all the other maps entries,
    /// and the pivot key and the value provided for the pivot key.
    ///
    /// requires:
    ///   self.last_key_value().map_or(true, |(m,_)| m < key);
    ///   rhs.first_key_value().map_or(true, |(n,_)| key < n);
    ///
    /// # Examples
    /// ```
    /// use fun_collections::FunMap;
    ///
    /// let f1 = FunMap::from_iter([(0, 'a'), (1, 'b')]);
    /// let f2 = FunMap::from_iter([(3, 'd')]);
    /// let f3 = f1.make_join(2, 'c', &f2);
    /// assert_eq!(f3.get(&0), Some(&'a'));
    /// assert_eq!(f3.get(&2), Some(&'c'));
    /// assert_eq!(f3.get(&3), Some(&'d'));
    /// ```
    pub fn make_join(&self, key: K, val: V, rhs: &FunMap<K, V>) -> Self {
        assert!(self.last_key_value().map_or(true, |(k2, _)| *k2 < key));
        assert!(rhs.first_key_value().map_or(true, |(k2, _)| key < *k2));

        let mut lhs = self.clone();
        lhs.join_with(key, val, rhs.clone());
        lhs
    }

    pub fn join_with(&mut self, key: K, val: V, mut rhs: FunMap<K, V>) -> () {
        assert!(self.last_key_value().map_or(true, |(k2, _)| *k2 < key));
        assert!(rhs.first_key_value().map_or(true, |(k2, _)| key < *k2));

        self.len += 1 + rhs.len();
        self.root = join(self.root.take(), key, val, rhs.root.take());
    }

    /// Returns the key-value pair for the least key in the map
    ///
    /// # Examples
    /// ```
    /// use fun_collections::FunMap;
    ///
    /// let fmap = FunMap::from_iter([(2,0), (1,0)].into_iter());
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
    /// use fun_collections::FunMap;
    ///
    /// let fmap = FunMap::from_iter([(2,0), (1,0)].into_iter());
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

    /// Returns a reference to the value associated with k.
    ///
    /// # Example
    /// ```
    /// use fun_collections::FunMap;
    ///
    /// let mut fmap = FunMap::new();
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

    /// Returns a reference to the value associated with k.
    ///
    /// # Example
    /// ```
    /// use fun_collections::FunMap;
    ///
    /// let mut fmap = FunMap::new();
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

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn len(&self) -> usize {
        self.len
    }
}

impl<K: Clone + Ord, V: Clone> Default for FunMap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

pub struct Iter<'a, K, V> {
    // TODO: use FunStack for the spine. Vec will be more performant, but users
    // may expect our promise about "cheap cloning" to apply to the iterators.
    spine: Vec<&'a Rc<Node<K, V>>>,
    len: usize,
}

impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        self.spine.pop().map(|n| {
            self.len -= 1;
            let entry = (&n.key, &n.val);
            let mut curr = n.right.as_ref();
            while let Some(m) = curr {
                self.spine.push(m);
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

impl<K: Clone + Ord, V: Clone> Extend<(K, V)> for FunMap<K, V> {
    fn extend<T: IntoIterator<Item = (K, V)>>(&mut self, iter: T) {
        for (k, v) in iter {
            self.insert(k, v);
        }
    }
}

impl<K: Clone + Ord, V: Clone> FromIterator<(K, V)> for FunMap<K, V> {
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
        let mut fmap = FunMap::new();
        fmap.extend(iter);
        fmap
    }
}

impl<'a, K, V> FusedIterator for Iter<'a, K, V> {}

#[cfg(test)]
mod test {
    extern crate quickcheck;
    use super::*;
    use quickcheck::quickcheck;

    fn chk_bal<K: Clone, V: Clone>(root: &OptNode<K, V>) -> i8 {
        match root {
            None => 0,
            Some(rc) => {
                assert_eq!(rc.right_ht, chk_bal(&rc.right));
                assert_eq!(rc.left_ht, chk_bal(&rc.left));
                rc.height()
            }
        }
    }

    fn chk_sort<K: Clone + Ord, V: Clone>(fmap: &FunMap<K, V>) {
        fmap.iter().reduce(|(k1, _), n @ (k2, _)| {
            assert!(k1 < k2);
            n
        });

        assert_eq!(fmap.iter().count(), fmap.len());
    }

    fn bal_test(vs: Vec<(u8, u32)>) {
        let mut fmap = FunMap::new();
        for &(k, v) in vs.iter() {
            fmap.insert(k, v);
            println!("{:?}", fmap);
            chk_bal(&fmap.root);
            chk_sort(&fmap);
        }
    }

    fn rm_test(vs: Vec<(i8, u32)>) {
        let mut fmap = FunMap::new();
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
            chk_bal(&fmap.root);
        }
    }

    // systematically try deleting each element of fmap
    fn chk_all_removes(fmap: FunMap<u8, u8>) {
        for (k, v) in fmap.clone().iter() {
            let mut fmap2 = fmap.clone();
            assert_eq!(fmap2.remove(k), Some(*v));
            chk_bal(&fmap2.root);
            chk_sort(&fmap2);
        }
    }

    #[test]
    fn rm_each_test() {
        // build map in order to encourage skewing
        let fmap: FunMap<_, _> = (0..32).map(|x| (x, x + 100)).collect();
        chk_all_removes(fmap);

        // build map in reverse order to encourage opposite skewing
        let fmap: FunMap<_, _> = (0..32).rev().map(|x| (x, x + 100)).collect();
        chk_all_removes(fmap);
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
        let fmap: FunMap<_, _> = (0..10).map(|i| (i, ())).collect();

        let mut iter = fmap.iter();
        let mut cnt = 10;
        while iter.next().is_some() {
            assert_eq!(iter.len(), cnt - 1);
            cnt -= 1;
        }
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
            let f1: FunMap<_, _> = v1.into_iter().enumerate().collect();
            let f2: FunMap<_, _> =
                v2.into_iter().enumerate().map(|(i,v)| (i+mid+1, v)).collect();
            let f3 = f1.make_join(mid, 0, &f2);
            chk_bal(&f3.root);
            chk_sort(&f3);
        }
    }
}
