use super::{AvlMap, Iter, Node};
use std::borrow::Borrow;
use std::cmp::Ordering::*;
use std::rc::Rc;

/// A sorted set of values.
///
/// The implementation is mostly a thin wrapper around [`AvlMap`].
#[derive(Clone, Default)]
pub struct AvlSet<V> {
    map: AvlMap<V, ()>,
}

impl<V> AvlSet<V> {
    /// Moves all elements from other into self and leaves other empty.
    pub fn append(&mut self, other: &mut Self)
    where
        V: Ord + Clone,
    {
        self.map.append(&mut other.map);
    }

    /// Removes all the entries from self.
    pub fn clear(&mut self) {
        self.map.clear();
    }

    /// Tests if self contains the given value.
    pub fn contains<Q>(&self, value: &Q) -> bool
    where
        V: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.map.contains_key(value)
    }

    /// Returns an iterator over elements in self and not in other
    pub fn difference<'a>(&'a self, other: &'a Self) -> Difference<'a, V>
    where
        V: Ord,
    {
        Difference::new(self.iter(), other.iter())
    }

    /// Returns the least value in the set.
    pub fn first(&self) -> Option<&V> {
        self.map.first_key_value().map(|(k, _)| k)
    }

    /// Returns a reference to the element matching value, if it exists
    pub fn get<Q>(&self, value: &Q) -> Option<&V>
    where
        V: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let mut curr = &self.map.root;
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
    pub fn insert(&mut self, value: V) -> bool
    where
        V: Clone + Ord,
    {
        self.map.insert(value, ()).is_none()
    }

    /// Returns an iterator of the values that are in both self and other.
    pub fn intersection<'a>(&'a self, other: &'a Self) -> Intersection<'a, V>
    where
        V: Ord,
    {
        Intersection::new(self.iter(), other.iter())
    }

    /// Returns true if self and other have no common values and false otherwise
    pub fn is_disjoint(&self, other: &Self) -> bool
    where
        V: Ord,
    {
        self.intersection(other).next().is_none()
    }

    /// Returns true if self is the empty set, false otherwise.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Tests if self is a subset of other.
    pub fn is_subset(&self, other: &Self) -> bool
    where
        V: Ord,
    {
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

        let lhs = self.map.root.as_ref().unwrap();
        let rhs = other.map.root.as_ref().unwrap();
        has_all(lhs, &mut vec![(rhs, rhs.left.is_none())])
    }

    /// tests if self is a superset of other.
    pub fn is_superset(&self, other: &Self) -> bool
    where
        V: Ord,
    {
        other.is_subset(self)
    }

    /// Returns an iterator over self's values in sorted order.
    pub fn iter(&self) -> SetIter<V> {
        SetIter {
            iter: self.map.iter(),
        }
    }

    /// Returns the greatest value in self.
    pub fn last(&self) -> Option<&V> {
        self.map.last_key_value().map(|e| e.0)
    }

    /// Returns the number of elements in self.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    // TODO: range

    /// Removes the given value from self returning true if the value was
    /// present and false otherwise.
    pub fn remove<Q>(&mut self, value: &Q) -> bool
    where
        V: Borrow<Q> + Clone + Ord,
        Q: Ord + ?Sized,
    {
        self.map.remove(value).is_some()
    }

    /// Replace and return the matching value in the map.
    pub fn replace(&mut self, value: V) -> Option<V>
    where
        V: Clone + Ord,
    {
        // TODO: adapt insert so we don't need multiple calls.
        let ret = self.take(&value);
        self.insert(value);
        ret
    }

    /// Retain values for which f returns true and discard others
    pub fn retain<F>(&mut self, mut f: F)
    where
        V: Clone + Ord,
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

        let Some(mut root) = self.map.root.take() else { return; };
        Rc::make_mut(&mut root);
        let Ok(root) = Rc::try_unwrap(root) else { panic!("try_unwrap fail?") };
        let mut acc = AvlMap::new();
        dfs(root, &mut f, &mut acc);
        self.map = acc;
    }

    /// Removes all elements greater or equal to key and returns them.
    pub fn split_off<Q>(&mut self, key: &Q) -> Self
    where
        V: Borrow<Q> + Clone + Ord,
        Q: Ord + ?Sized,
    {
        Self {
            map: self.map.split_off(key),
        }
    }

    /// Returns an iterator over elements in self or other but not both.
    pub fn symmetric_difference<'a>(
        &'a self,
        other: &'a Self,
    ) -> SymmetricDifference<'a, V> {
        SymmetricDifference::new(self.iter(), other.iter())
    }

    /// Removes and returns the set member that matches value.
    pub fn take<Q>(&mut self, value: &Q) -> Option<V>
    where
        V: Borrow<Q> + Clone + Ord,
        Q: Ord + ?Sized,
    {
        if let (opt_v @ Some(_), _) = super::rm(&mut self.map.root, value) {
            self.map.len -= 1;
            opt_v.map(|e| e.0)
        } else {
            None
        }
    }

    /// Returns an iterator over the elements of self and other, ordered by key.
    ///
    /// Common elements are only returned once.
    pub fn union<'a>(&'a self, other: &'a Self) -> Union<'a, V> {
        Union::new(self.iter(), other.iter())
    }

    /// Returns a new, empty set.
    pub fn new() -> Self {
        Self { map: AvlMap::new() }
    }

    /// Creates a new set with the elements of lhs that are not in rhs.
    pub fn new_diff(lhs: Self, rhs: Self) -> Self
    where
        V: Clone + Ord,
    {
        let root = super::diff(lhs.map.root, rhs.map.root);
        Self {
            map: AvlMap {
                len: super::len(&root),
                root,
            },
        }
    }

    /// Creates a new set with the elements of lhs that are also in rhs.
    pub fn new_intersection(lhs: Self, rhs: Self) -> Self
    where
        V: Clone + Ord,
    {
        let root = super::intersect(lhs.map.root, rhs.map.root);
        Self {
            map: AvlMap {
                len: super::len(&root),
                root,
            },
        }
    }

    /// Creates a new set with the elements of both lhs and rhs.
    pub fn new_union(lhs: Self, rhs: Self) -> Self
    where
        V: Clone + Ord,
    {
        let root = super::union(lhs.map.root, rhs.map.root);
        Self {
            map: AvlMap {
                len: super::len(&root),
                root,
            },
        }
    }

    /// Creates a new set with the elements that are in lhs or rhs but not both.
    pub fn new_sym_diff(lhs: Self, rhs: Self) -> Self
    where
        V: Clone + Ord,
    {
        let root = super::sym_diff(lhs.map.root, rhs.map.root);
        Self {
            map: AvlMap {
                len: super::len(&root),
                root,
            },
        }
    }
}

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

pub struct SetIter<'a, T> {
    iter: Iter<'a, T, ()>,
}

impl<'a, T> Iterator for SetIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|e| e.0)
    }
}

crate::make_set_op_iter!(Difference, SetIter<'a, T>, 0b100);
crate::make_set_op_iter!(Intersection, SetIter<'a, T>, 0b010);
crate::make_set_op_iter!(Union, SetIter<'a, T>, 0b111);
crate::make_set_op_iter!(SymmetricDifference, SetIter<'a, T>, 0b101);

impl<K: Clone + Ord> std::ops::BitAnd for &AvlSet<K> {
    type Output = AvlSet<K>;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self::Output::new_intersection(self.clone(), rhs.clone())
    }
}

impl<K: Clone + Ord> std::ops::BitOr for &AvlSet<K> {
    type Output = AvlSet<K>;

    fn bitor(self, rhs: Self) -> Self::Output {
        AvlSet::new_union(self.clone(), rhs.clone())
    }
}

impl<K: Clone + Ord> std::ops::Sub for &AvlSet<K> {
    type Output = AvlSet<K>;

    fn sub(self, rhs: Self) -> Self::Output {
        AvlSet::new_diff(self.clone(), rhs.clone())
    }
}

impl<K: Clone + Ord> std::ops::BitXor for &AvlSet<K> {
    type Output = AvlSet<K>;

    fn bitxor(self, rhs: Self) -> Self::Output {
        AvlSet::new_sym_diff(self.clone(), rhs.clone())
    }
}

#[cfg(test)]
mod test {
    extern crate quickcheck;
    use super::*;
    use quickcheck::quickcheck;

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

    quickcheck! {
        fn qc_set_tests(v1: Vec<u8>, v2: Vec<u8>) -> () {
            set_test(v1, v2);
        }

        fn qc_set_tests2(v1: (u8,u8,u8,u8,u8,u8)) -> () {
            set_test2(vec![v1.0, v1.1, v1.2, v1.3, v1.4, v1.5]);
        }
    }
}
