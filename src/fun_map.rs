#![allow(dead_code, unused)] // FIXME

use std::cmp::Ordering::*;
// use std::fmt;
use std::fmt::{Debug, Formatter};
use std::rc::Rc;

#[derive(Clone)]
struct Node<K: Clone, V: Clone> {
    key: K,
    val: V,
    bal: i8, // "balance factor" = height(right) - height(left)
    left: OptNode<K, V>,
    right: OptNode<K, V>,
}

type OptNode<K, V> = Option<Rc<Node<K, V>>>;

impl<K: Clone + Debug, V: Clone + Debug> Debug for Node<K, V> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("(")?;
        match &self.left {
            None => f.write_str(".")?,
            Some(lf) => lf.fmt(f)?,
        }

        f.write_fmt(format_args!(" {{{:?}: {:?}}}", self.key, self.val))?;

        match self.bal.cmp(&0) {
            Less => f.write_str(" > ")?,
            Equal => f.write_str(" = ")?,
            Greater => f.write_str(" < ")?,
        }

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

    fn rot_lf(root: &mut OptNode<K, V>) {
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

    fn rot_rt_lf(root: &mut OptNode<K, V>) {
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

    fn rot_rt(root: &mut OptNode<K, V>) {
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

    fn rot_lf_rt(root: &mut OptNode<K, V>) {
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

    // Inserts (k,v) into the map rooted at r and returns the replaced value and
    // whether the tree is taller after the insertion.
    fn ins(root: &mut OptNode<K, V>, k: K, v: V) -> (Option<V>, bool) {
        match root.as_mut() {
            Some(rc) => {
                let n = Rc::make_mut(rc);
                match k.cmp(&n.key) {
                    Equal => (Some(std::mem::replace(&mut n.val, v)), false),

                    Less => {
                        let (old_v, is_taller) = Self::ins(&mut n.left, k, v);

                        if is_taller {
                            match n.bal.cmp(&0) {
                                Less => {
                                    // we were already left heavy
                                    if n.left.as_ref().unwrap().bal > 0 {
                                        Self::rot_lf_rt(root)
                                    } else {
                                        Self::rot_rt(root)
                                    }

                                    (old_v, false)
                                }

                                Equal => {
                                    // we were balanced
                                    n.bal = -1;
                                    (old_v, true)
                                }

                                Greater => {
                                    // we were right heavy
                                    n.bal = 0;
                                    (old_v, false)
                                }
                            }
                        } else {
                            (old_v, false)
                        }
                    }

                    Greater => {
                        let (old_v, is_taller) = Self::ins(&mut n.right, k, v);

                        if is_taller {
                            match n.bal.cmp(&0) {
                                Less => {
                                    // we were left heavy
                                    n.bal = 0;
                                    (old_v, false)
                                }

                                Equal => {
                                    // we were balanced
                                    n.bal = 1;
                                    (old_v, true)
                                }

                                Greater => {
                                    // we were already right heavy
                                    if n.right.as_ref().unwrap().bal < 0 {
                                        Self::rot_rt_lf(root)
                                    } else {
                                        Self::rot_lf(root)
                                    }

                                    (old_v, false)
                                }
                            }
                        } else {
                            (old_v, false)
                        }
                    }
                }
            }

            None => {
                *root = Some(Rc::new(Node {
                    key: k,
                    val: v,
                    bal: 0,
                    left: None,
                    right: None,
                }));

                (None, true)
            }
        }
    }

    pub fn insert(&mut self, k: K, v: V) -> Option<V> {
        let (ret, _) = Self::ins(&mut self.root, k, v);
        self.len += ret.is_none() as usize;
        ret
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

    #[test]
    fn bal_test_regr1() {
        bal_test(vec![(4, 0), (0, 0), (5, 0), (1, 0), (2, 0), (3, 0)]);
    }

    quickcheck! {
        fn qc_bal_test(vs: Vec<(u8, u32)>) -> () {
            bal_test(vs);
        }
    }
}
