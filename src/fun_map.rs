#![allow(dead_code, unused)] // FIXME

use std::cmp::Ordering::*;
// use std::fmt;
// use std::fmt::{Debug, Formatter};
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

#[derive(Clone)]
pub struct FunMap<K: Clone, V: Clone> {
    len: usize,
    root: OptNode<K, V>,
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
                            n.bal -= 1;
                            (old_v, n.bal < 0)
                        } else {
                            (old_v, false)
                        }
                    }

                    Greater => {
                        let (old_v, is_taller) = Self::ins(&mut n.right, k, v);
                        if is_taller {
                            n.bal += 1;
                            (old_v, n.bal > 0)
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
