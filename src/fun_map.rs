use std::cmp::Ordering::*;
use std::fmt::{Debug, Formatter};
use std::mem::replace;
use std::rc::Rc;

struct Node<K, V> {
    key: K,
    val: V,
    bal: i8, // "balance factor" = height(right) - height(left)
    left: OptNode<K, V>,
    right: OptNode<K, V>,
}

type OptNode<K, V> = Option<Rc<Node<K, V>>>;

impl<K, V> Node<K, V> {
    fn new(key: K, val: V) -> Self {
        Node {
            key,
            val,
            bal: 0,
            left: None,
            right: None,
        }
    }
}

impl<K: Clone, V: Clone> Clone for Node<K, V> {
    fn clone(&self) -> Self {
        Node {
            key: self.key.clone(),
            val: self.val.clone(),
            bal: self.bal,
            left: self.left.clone(),
            right: self.right.clone(),
        }
    }
}

impl<K: Clone + Debug, V: Clone + Debug> Debug for Node<K, V> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("({{{:?}: {:?}}} ", self.key, self.val))?;

        if self.bal == 0 {
            f.write_str("=0 ")?;
        } else {
            f.write_fmt(format_args!("{:+} ", self.bal))?;
        }

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

fn rot_lf<K: Clone, V: Clone>(root: &mut OptNode<K, V>) {
    // We want the following transformation:
    //    a(x, b(y, z)))   =>   b(a(x, y), z)
    // x and z retain the same parents.

    let mut a_opt = root.take();
    let a_rc = a_opt.as_mut().unwrap();
    let a = Rc::make_mut(a_rc);

    let mut b_opt = a.right.take();
    let b_rc = b_opt.as_mut().unwrap();
    let b = Rc::make_mut(b_rc);

    // update the bal fields (while we have mutable access)
    if b.bal == 0 {
        a.bal = 1;
        b.bal = -1;
    } else {
        // b.bal == 1  -> ht(y) < ht(z)
        a.bal = 0;
        b.bal = 0;
    }

    // move y from b to a
    a.right = b.left.take();

    // make a be b's left child
    b.left = a_opt;

    // install b as the new root
    *root = b_opt;
}

fn rot_rt_lf<K: Clone, V: Clone>(root: &mut OptNode<K, V>) {
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

    // a right-left rotation is called for when:
    //   * b is two taller than x
    //   * c is one taller than w
    // After the rotation, the final balances will depend on the heights
    // of y and z.

    // assert_eq!(
    //     b.bal, -1,
    //     "right-left rotation when .right.left child is not taller"
    // );

    // Update the bal fields while we have mutable access
    match c.bal.cmp(&0) {
        Equal => {
            a.bal = 0;
            b.bal = 0;
        }

        Greater => {
            a.bal = -1;
            b.bal = 0;
        }

        Less => {
            a.bal = 0;
            b.bal = 1;
        }
    }
    c.bal = 0;

    // We need to take care not to overwrite any links before taking them.
    // With the unlinks we've done, we have
    //   a(x, None)
    //   b(None, w)
    //   c(y, z)

    // move c's children to a and b
    a.right = c.left.take();
    b.left = c.right.take();

    // move a and b into c
    c.left = a_opt.take();
    c.right = b_opt.take();

    // install c as the new root
    *root = c_opt;
}

fn rot_rt<K: Clone, V: Clone>(root: &mut OptNode<K, V>) {
    // We want the following transformation:
    //    a(b(x, y), z)   =>   b(x, a(y, z))
    // x and z retain the same parents.

    let mut a_opt = root.take();
    let a_rc = a_opt.as_mut().unwrap();
    let a = Rc::make_mut(a_rc);

    let mut b_opt = a.left.take();
    let b_rc = b_opt.as_mut().unwrap();
    let b = Rc::make_mut(b_rc);

    // update the balances while we have mutable access
    // we are called when a.bal = -2, which means
    //    ht(b) = ht(z) + 2
    //    max(ht(x), ht(y)) + 1 = ht(z) + 2
    //    max(ht(x), ht(y)) = ht(z) + 1
    if b.bal == 0 {
        // ht(x) = ht(y) = ht(z) + 1
        a.bal = -1; // because ht(y) > ht(z)
        b.bal = 1; // because ht(a) = (ht(y) + 1) > ht(x) = ht(y)
    } else {
        // assert_eq!(b.bal, -1,);
        // ht(x) = ht(y) + 1 = ht(z) + 1
        // ht(y) = ht(z) < ht(x)
        a.bal = 0;
        b.bal = 0;
    }

    // We have
    //   a(None, z)
    //   b(x, y)

    // move y from b to a
    a.left = b.right.take();

    // move a into b
    b.right = a_opt.take();

    // install b as the new root
    *root = b_opt
}

fn rot_lf_rt<K: Clone, V: Clone>(root: &mut OptNode<K, V>) {
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

    // assert_eq!(b.bal, 1)
    match c.bal.cmp(&0) {
        Equal => {
            a.bal = 0;
            b.bal = 0;
        }

        Less => {
            // ht(y) = ht(z) + 1
            // ht(y) > ht(z)
            // ht(x) = ht(y) = ht(w)
            a.bal = 1;
            b.bal = 0;
        }

        Greater => {
            // ht(y) + 1 = ht(z)
            // ht(y) < ht(z)
            // ht(x) = ht(z) = ht(w)
            a.bal = 0;
            b.bal = -1;
        }
    }
    c.bal = 0;

    // We have:
    //   a(None, w)
    //   b(x, None)
    //   c(y, z)

    b.right = c.left.take(); // => b(x, y), c(None, z)
    a.left = c.right.take(); // => a(z, w), c(None, None)

    c.left = b_opt; // => c(b(x, y), None)
    c.right = a_opt; // => c(b(x, y), a(z, w))

    *root = c_opt;
}

fn rebal_lf_to_rt<K, V>(root: &mut OptNode<K, V>) -> i8
where
    K: Clone,
    V: Clone,
{
    let n = Rc::get_mut(root.as_mut().unwrap()).unwrap();

    if n.left.as_ref().unwrap().bal <= 0 {
        rot_rt(root);
    } else {
        rot_lf_rt(root);
    }

    // +1 bal at root => same ht   1 -> 0
    //  0 bal at root => -1 ht     0 -> -1
    // -1 bal at root is not possible from selected rotations
    let root_bal = root.as_ref().unwrap().bal;
    assert!(root_bal == 0 || root_bal == 1, "Bad root_bal = {root_bal}.");
    root_bal - 1
}

fn rebal_rt_to_lf<K, V>(root: &mut OptNode<K, V>) -> i8
where
    K: Clone,
    V: Clone,
{
    let n = Rc::get_mut(root.as_mut().unwrap()).unwrap();

    if n.right.as_ref().unwrap().bal >= 0 {
        rot_lf(root);
    } else {
        rot_rt_lf(root);
    }

    // -1 bal at root => same ht   -1 -> 0
    //  0 bal at root => -1 ht      0 -> -1
    // +1 bal at root is not possible from selected rotations
    let root_bal = root.as_ref().unwrap().bal;
    assert!(
        root_bal == -1 || root_bal == 0,
        "Bad root_bal = {root_bal}."
    );
    -1 - root_bal
}

// Inserts (k,v) into the map rooted at root and returns the replaced value and
// whether the updated node is taller as a result of insertion.
fn ins<K, V>(root: &mut OptNode<K, V>, k: K, v: V) -> (Option<V>, bool)
where
    K: Clone + Ord,
    V: Clone,
{
    let n = match root.as_mut() {
        None => {
            *root = Some(Rc::new(Node::new(k, v)));
            return (None, true); // *** EARLY RETURN ***
        }

        Some(rc) => Rc::make_mut(rc),
    };

    match k.cmp(&n.key) {
        Equal => (Some(std::mem::replace(&mut n.val, v)), false),

        Less => {
            let (old_v, is_taller) = ins(&mut n.left, k, v);
            if is_taller && n.bal < 0 {
                rebal_lf_to_rt(root);
                (old_v, false)
            } else {
                n.bal -= is_taller as i8;
                (old_v, is_taller && n.bal < 0)
            }
        }

        Greater => {
            let (old_v, is_taller) = ins(&mut n.right, k, v);
            if is_taller && n.bal > 0 {
                rebal_rt_to_lf(root);
                (old_v, false)
            } else {
                n.bal += is_taller as i8;
                (old_v, is_taller && n.bal > 0)
            }
        }
    }
}

fn rebal_lf_rm<K, V, R>(
    root: &mut OptNode<K, V>,
    old_v: R,
    ht_delta: i8,
    root_bal: i8,
) -> (R, i8)
where
    K: Clone + Ord,
    V: Clone,
{
    if ht_delta == 0 {
        (old_v, 0)
    } else if root_bal == 2 {
        assert_eq!(ht_delta, -1);
        // The delete reduced the short side of the node, which did not
        // shrink the overall height.  The total change depends on rebal.
        (old_v, rebal_rt_to_lf(root))
    } else {
        // if root_bal went to 0, we lost height.     0 -> -1
        // if root_bal went to 1, height unchanged.   1 -> 0
        (old_v, root_bal - 1)
    }
}

fn rebal_rt_rm<K, V, R>(
    root: &mut OptNode<K, V>,
    old_v: R,
    ht_delta: i8,
    root_bal: i8,
) -> (R, i8)
where
    K: Clone + Ord,
    V: Clone,
{
    if ht_delta == 0 {
        (old_v, 0)
    } else if root_bal == -2 {
        // The delete reduced the short side of the node, which did not
        // shrink the overall height.  The total change depends on rebal.
        (old_v, rebal_lf_to_rt(root))
    } else {
        // if root_bal =  0, the taller (rt) side shrank -> -1 ht change
        // if root_bal = -1, the rt shrank, but lf is the same -> 0 ht change
        (old_v, -1 - root_bal)
    }
}

fn rm_leftmost<K, V>(root: &mut OptNode<K, V>) -> (Option<(K, V)>, i8)
where
    K: Clone + Ord,
    V: Clone,
{
    let n = match root.as_mut() {
        None => return (None, 0), // *** EARLY RETURN ***
        Some(rc) => Rc::make_mut(rc),
    };

    if n.left.is_some() {
        let (v, ht_delta) = rm_leftmost(&mut n.left);
        n.bal -= ht_delta; // subtraction because its on the left
        let root_bal = n.bal;
        rebal_lf_rm(root, v, ht_delta, root_bal)
    } else {
        let old_n = take_node(root);
        *root = old_n.right;
        (Some((old_n.key, old_n.val)), -1)
    }
}

fn rm<K, V>(root: &mut OptNode<K, V>, k: &K) -> (Option<V>, i8)
where
    K: Clone + Ord,
    V: Clone,
{
    let n = match root.as_mut() {
        None => return (None, 0), // *** EARLY RETURN ***
        Some(rc) => Rc::make_mut(rc),
    };

    match k.cmp(&n.key) {
        Less => {
            let (v, ht_delta) = rm(&mut n.left, k);
            n.bal -= ht_delta;
            let root_bal = n.bal;
            rebal_lf_rm(root, v, ht_delta, root_bal)
        }

        Greater => {
            let (v, ht_delta) = rm(&mut n.right, k);
            n.bal += ht_delta;
            let root_bal = n.bal;
            rebal_rt_rm(root, v, ht_delta, root_bal)
        }

        Equal => match (&n.left, &n.right) {
            (None, None) => {
                let old_n = take_node(root);
                (Some(old_n.val), -1)
            }

            (None, Some(_)) => {
                let old_n = take_node(root);
                *root = old_n.right;
                (Some(old_n.val), -1)
            }

            (Some(_), None) => {
                let old_n = take_node(root);
                *root = old_n.left;
                (Some(old_n.val), -1)
            }

            _ => {
                // both children are populated
                let (succ, ht_delta) = rm_leftmost(&mut n.right);
                let (succ_key, succ_val) = succ.unwrap();
                n.key = succ_key;
                let old_val = replace(&mut n.val, succ_val);

                n.bal += ht_delta;
                let root_bal = n.bal;
                rebal_rt_rm(root, Some(old_val), ht_delta, root_bal)
            }
        },
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

        Iter { spine }
    }

    pub fn insert(&mut self, k: K, v: V) -> Option<V> {
        let (ret, _) = ins(&mut self.root, k, v);
        self.len += ret.is_none() as usize;
        ret
    }

    /// Removes a key from a map and returns the mapped value, if present.
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
    pub fn remove(&mut self, k: &K) -> Option<V> {
        if let (opt_v @ Some(_), _) = rm(&mut self.root, k) {
            self.len -= 1;
            opt_v
        } else {
            None
        }
    }

    // TODO: generalize ala
    // https://doc.rust-lang.org/std/collections/struct.BTreeMap.html#method.get,
    // specifically, using type Q for the key.
    pub fn get(&self, k: &K) -> Option<&V> {
        let mut curr = &self.root;
        while let Some(n) = curr {
            match k.cmp(&n.key) {
                Less => curr = &n.left,
                Equal => return Some(&n.val),
                Greater => curr = &n.right,
            }
        }

        None
    }

    // Generalize for other key types, ala get()?
    pub fn get_mut(&mut self, k: &K) -> Option<&mut V> {
        let mut curr = &mut self.root;
        while let Some(rc) = curr {
            let n = Rc::make_mut(rc);
            match k.cmp(&n.key) {
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

pub struct Iter<'a, K: Clone, V: Clone> {
    // TODO: use FunStack for the spine. Vec will be more performant, but users
    // may expect our promise about "cheap cloning" to apply to the iterators.
    spine: Vec<&'a Rc<Node<K, V>>>,
}

impl<'a, K: Clone, V: Clone> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        self.spine.pop().map(|n| {
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

#[cfg(test)]
mod test {
    extern crate quickcheck;
    use super::*;
    use quickcheck::quickcheck;

    fn chk_bal<K: Clone, V: Clone>(root: &OptNode<K, V>) -> i8 {
        match root {
            None => 0,
            Some(rc) => {
                let rt_ht = chk_bal(&rc.right);
                let lf_ht = chk_bal(&rc.left);
                assert_eq!(rt_ht - lf_ht, rc.bal);
                rt_ht.max(lf_ht) + 1
            }
        }
    }

    fn chk_sort<K: Clone + Ord, V: Clone>(fmap: &FunMap<K, V>) {
        fmap.iter().reduce(|(k1, _), n @ (k2, _)| {
            assert!(k1 < k2);
            n
        });
    }

    fn bal_test(vs: Vec<(u8, u32)>) {
        let mut fmap = FunMap::new();
        for &(k, v) in vs.iter() {
            fmap.insert(k, v);
            // println!("{:?}", fmap);
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
    }
}
