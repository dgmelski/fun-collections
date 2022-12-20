use super::*;

#[derive(Eq)]
pub struct BTreeSet<T, const N: usize> {
    map: BTreeMap<T, (), N>,
}

impl<T, const N: usize> BTreeSet<T, N> {
    /// Moves all elements from other into self and leaves other empty.
    pub fn append(&mut self, other: &mut Self)
    where
        T: Ord + Clone,
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
        T: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.map.contains_key(value)
    }

    // /// Returns an iterator over elements in self and not in other
    // pub fn difference<'a>(&'a self, other: &'a Self) -> Difference<'a, T>
    // where
    //     T: Ord,
    // {
    //     Difference::new(self.iter(), other.iter())
    // }

    // TODO: drain_filter? (Part of unstable API)

    /// Returns the least value in the set.
    pub fn first(&self) -> Option<&T> {
        self.map.first_key_value().map(|(k, _)| k)
    }

    /// Returns a reference to the element matching value, if it exists
    pub fn get<Q>(&self, value: &Q) -> Option<&T>
    where
        T: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.map.get_key_value(value).map(|e| e.0)
    }

    /// Inserts the given value and returns true if self did not already have
    /// the value and returns false otherwise.
    pub fn insert(&mut self, value: T) -> bool
    where
        T: Clone + Ord,
    {
        self.map.insert(value, ()).is_none()
    }

    // /// Returns an iterator of the values that are in both self and other.
    // pub fn intersection<'a>(&'a self, other: &'a Self) -> Intersection<'a, T>
    // where
    //     T: Ord,
    // {
    //     Intersection::new(self.iter(), other.iter())
    // }

    // /// Returns true if self and other have no common values and false otherwise
    // pub fn is_disjoint(&self, other: &Self) -> bool
    // where
    //     T: Ord,
    // {
    //     self.intersection(other).next().is_none()
    // }

    /// Returns true if self is the empty set, false otherwise.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    // /// Tests if self is a subset of other.
    // pub fn is_subset(&self, other: &Self) -> bool
    // where
    //     T: Ord,
    // {
    //     if self.is_empty() {
    //         return true;
    //     } else if self.len() > other.len() {
    //         return false;
    //     }

    //     fn has_all<K, V>(
    //         lhs: &Arc<Node<K, V>>,
    //         w: &mut Vec<(&Arc<Node<K, V>>, bool)>,
    //     ) -> bool
    //     where
    //         K: Ord,
    //     {
    //         // Strategy: we do an inorder recursive traversal of lhs (the
    //         // suspected subset). Simultaneously, we keep an 'iterator' for the
    //         // rhs (the suspected superset).  We also update the iterator
    //         // 'in-order' but we look for opportunities to fast-forward.

    //         // first traverse to lhs's left child
    //         if let Some(lhs_left) = lhs.left.as_ref() {
    //             if !has_all(lhs_left, w) {
    //                 return false;
    //             }
    //         }

    //         // now, look for lhs.key in rhs using our iterator
    //         while let Some((rhs, is_left_done)) = w.last_mut() {
    //             // pointer check to short circuit further comparisons
    //             if Arc::ptr_eq(lhs, rhs) {
    //                 w.pop();
    //                 return true;
    //             }

    //             match lhs.key.cmp(&rhs.key) {
    //                 Less => {
    //                     // The only way we can find a lesser key than rhs.key
    //                     // is to traverse to its left.  If we already matched
    //                     // all of the keys there, it won't help to look again.
    //                     if *is_left_done {
    //                         return false;
    //                     }

    //                     *is_left_done = true;
    //                     let rhs_left = rhs.left.as_ref().unwrap();
    //                     w.push((rhs_left, rhs_left.left.is_none()));
    //                 }

    //                 Equal => {
    //                     // We visited all the lesser keys on the lhs and we
    //                     // do not need any remaining lesser keys on the rhs.
    //                     // Move on to greater keys.
    //                     let rhs = w.pop().unwrap().0;
    //                     if let Some(rhs_right) = rhs.right.as_ref() {
    //                         w.push((rhs_right, rhs_right.left.is_none()));
    //                     }

    //                     if let Some(lhs_right) = lhs.right.as_ref() {
    //                         return has_all(lhs_right, w);
    //                     } else {
    //                         return true;
    //                     }
    //                 }

    //                 Greater => {
    //                     // the greater nodes in the rhs are to the right or up
    //                     // TODO: we may be able to pop multiple nodes from rhs
    //                     let rhs = w.pop().unwrap().0;
    //                     if let Some(rhs_right) = rhs.right.as_ref() {
    //                         w.push((rhs_right, rhs_right.left.is_none()));
    //                     }
    //                 }
    //             }
    //         }

    //         false
    //     }

    //     let lhs = self.map.root.as_ref().unwrap();
    //     let rhs = other.map.root.as_ref().unwrap();
    //     has_all(lhs, &mut vec![(rhs, rhs.left.is_none())])
    // }

    // /// tests if self is a superset of other.
    // pub fn is_superset(&self, other: &Self) -> bool
    // where
    //     T: Ord,
    // {
    //     other.is_subset(self)
    // }

    /// Returns an iterator over self's values in sorted order.
    pub fn iter(&self) -> Iter<T, N> {
        Iter {
            iter: self.map.iter(),
        }
    }

    /// Returns the greatest value in self.
    pub fn last(&self) -> Option<&T> {
        self.map.last_key_value().map(|e| e.0)
    }

    /// Returns the number of elements in self.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Returns a new, empty set.
    pub fn new() -> Self {
        Self {
            map: BTreeMap::new(),
        }
    }

    // /// Creates a new set with the elements of lhs that are not in rhs.
    // pub fn new_diff(lhs: Self, rhs: Self) -> Self
    // where
    //     T: Clone + Ord,
    // {
    //     let root = super::diff(lhs.map.root, rhs.map.root);
    //     Self {
    //         map: BTreeMap {
    //             len: super::len(&root),
    //             root,
    //         },
    //     }
    // }

    // /// Creates a new set with the elements of lhs that are also in rhs.
    // pub fn new_intersection(lhs: Self, rhs: Self) -> Self
    // where
    //     T: Clone + Ord,
    // {
    //     let root = super::intersect(lhs.map.root, rhs.map.root);
    //     Self {
    //         map: BTreeMap {
    //             len: super::len(&root),
    //             root,
    //         },
    //     }
    // }

    // /// Creates a new set with the elements of both lhs and rhs.
    // pub fn new_union(lhs: Self, rhs: Self) -> Self
    // where
    //     T: Clone + Ord,
    // {
    //     let root = super::union(lhs.map.root, rhs.map.root);
    //     Self {
    //         map: BTreeMap {
    //             len: super::len(&root),
    //             root,
    //         },
    //     }
    // }

    // /// Creates a new set with the elements that are in lhs or rhs but not both.
    // pub fn new_sym_diff(lhs: Self, rhs: Self) -> Self
    // where
    //     T: Clone + Ord,
    // {
    //     let root = super::sym_diff(lhs.map.root, rhs.map.root);
    //     Self {
    //         map: BTreeMap {
    //             len: super::len(&root),
    //             root,
    //         },
    //     }
    // }

    /// Removes the first element from the set and returns it.
    pub fn pop_first(&mut self) -> Option<T>
    where
        T: Clone,
    {
        self.map.pop_first().map(|e| e.0)
    }

    /// Removes the last element from the set and returns it.
    pub fn pop_last(&mut self) -> Option<T>
    where
        T: Clone,
    {
        self.map.pop_last().map(|e| e.0)
    }

    // TODO: range

    /// Removes the given value from self returning true if the value was
    /// present and false otherwise.
    pub fn remove<Q>(&mut self, value: &Q) -> bool
    where
        T: Borrow<Q> + Clone + Ord,
        Q: Ord + ?Sized,
    {
        self.map.remove(value).is_some()
    }

    // /// Replace and return the matching value in the map.
    // pub fn replace(&mut self, value: T) -> Option<T>
    // where
    //     T: Clone + Ord,
    // {
    //     // TODO: adapt insert so we don't need multiple calls.
    //     let ret = self.take(&value);
    //     self.insert(value);
    //     ret
    // }

    // /// Retain values for which f returns true and discard others
    // pub fn retain<F>(&mut self, mut f: F)
    // where
    //     T: Clone + Ord,
    //     F: FnMut(&T) -> bool,
    // {
    //     fn dfs<V, F>(n: Node<V, ()>, f: &mut F, acc: &mut BTreeMap<V, ()>)
    //     where
    //         F: FnMut(&V) -> bool,
    //         V: Clone + Ord,
    //     {
    //         if let Some(mut lf) = n.left {
    //             Arc::make_mut(&mut lf);
    //             if let Ok(x) = Arc::try_unwrap(lf) {
    //                 dfs(x, f, acc);
    //             }
    //         }

    //         if f(&n.key) {
    //             acc.insert(n.key, ());
    //         }

    //         if let Some(mut rt) = n.right {
    //             Arc::make_mut(&mut rt);
    //             if let Ok(x) = Arc::try_unwrap(rt) {
    //                 dfs(x, f, acc);
    //             }
    //         }
    //     }

    //     let Some(mut root) = self.map.root.take() else { return; };
    //     Arc::make_mut(&mut root);
    //     let Ok(root) = Arc::try_unwrap(root) else { panic!("try_unwrap fail?") };
    //     let mut acc = BTreeMap::new();
    //     dfs(root, &mut f, &mut acc);
    //     self.map = acc;
    // }

    /// Removes all elements greater or equal to key and returns them.
    pub fn split_off<Q>(&mut self, key: &Q) -> Self
    where
        T: Borrow<Q> + Clone + Ord,
        Q: Ord + ?Sized,
    {
        Self {
            map: self.map.split_off(key),
        }
    }

    // /// Returns an iterator over elements in self or other but not both.
    // pub fn symmetric_difference<'a>(
    //     &'a self,
    //     other: &'a Self,
    // ) -> SymmetricDifference<'a, T> {
    //     SymmetricDifference::new(self.iter(), other.iter())
    // }

    // /// Removes and returns the set member that matches value.
    // ///
    // /// # Examples
    // /// ```
    // /// use lazy_clone_collections::AvlSet;
    // ///
    // /// let mut s = AvlSet::new();
    // /// s.insert("abc".to_string());
    // /// s.insert("def".to_string());
    // /// assert_eq!(s.take("abc"), Some(String::from("abc")));
    // /// assert_eq!(s.len(), 1);
    // /// ```
    // pub fn take<Q>(&mut self, value: &Q) -> Option<T>
    // where
    //     T: Borrow<Q> + Clone + Ord,
    //     Q: Ord + ?Sized,
    // {
    //     if let Some((k, _)) = super::rm(&mut self.map.root, value).0 {
    //         self.map.len -= 1;
    //         Some(k)
    //     } else {
    //         None
    //     }
    // }

    // /// Returns an iterator over the elements of self and other, ordered by key.
    // ///
    // /// Common elements are only returned once.
    // pub fn union<'a>(&'a self, other: &'a Self) -> Union<'a, T> {
    //     Union::new(self.iter(), other.iter())
    // }
}

pub struct Iter<'a, T, const N: usize> {
    iter: crate::btree::Iter<'a, T, (), N>,
}

impl<'a, T, const N: usize> Iterator for Iter<'a, T, N> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|e| e.0)
    }
}

// crate::make_set_op_iter!(Difference, Iter<'a, T, N>, 0b100);
// crate::make_set_op_iter!(Intersection, Iter<'a, T, N>, 0b010);
// crate::make_set_op_iter!(Union, Iter<'a, T, N>, 0b111);
// crate::make_set_op_iter!(SymmetricDifference, Iter<'a, T, N>, 0b101);

pub struct IntoIter<T, const N: usize> {
    iter: crate::btree::IntoIter<T, (), N>,
}

impl<T: Clone, const N: usize> Iterator for IntoIter<T, N> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|e| e.0)
    }
}

// impl<K: Clone + Ord, const N: usize> std::ops::BitAnd for &BTreeSet<K, N> {
//     type Output = BTreeSet<K, N>;

//     fn bitand(self, rhs: Self) -> Self::Output {
//         Self::Output::new_intersection(self.clone(), rhs.clone())
//     }
// }

// impl<K: Clone + Ord, const N: usize> std::ops::BitOr for &BTreeSet<K, N> {
//     type Output = BTreeSet<K, N>;

//     fn bitor(self, rhs: Self) -> Self::Output {
//         BTreeSet::new_union(self.clone(), rhs.clone())
//     }
// }

// impl<K: Clone + Ord, const N: usize> std::ops::BitXor for &BTreeSet<K, N> {
//     type Output = BTreeSet<K, N>;

//     fn bitxor(self, rhs: Self) -> Self::Output {
//         BTreeSet::new_sym_diff(self.clone(), rhs.clone())
//     }
// }

impl<T: std::fmt::Debug, const N: usize> std::fmt::Debug for BTreeSet<T, N> {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fmt.write_str("BTreeSet(")?;
        fmt.debug_set().entries(self.iter()).finish()?;
        fmt.write_str(")")
    }
}

impl<T, const N: usize> Default for BTreeSet<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, T: Clone + Ord, const N: usize> Extend<&'a T> for BTreeSet<T, N> {
    fn extend<I: IntoIterator<Item = &'a T>>(&mut self, iter: I) {
        for x in iter {
            self.insert(x.clone());
        }
    }
}

impl<T: Clone + Ord, const N: usize> Extend<T> for BTreeSet<T, N> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for x in iter {
            self.insert(x);
        }
    }
}

impl<T: Clone + Ord, const N: usize> From<[T; N]> for BTreeSet<T, N> {
    fn from(value: [T; N]) -> Self {
        Self::from_iter(value.into_iter())
    }
}

impl<T: Clone + Ord, const N: usize> FromIterator<T> for BTreeSet<T, N> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut s = BTreeSet::new();
        s.extend(iter);
        s
    }
}

impl<T: std::hash::Hash, const N: usize> std::hash::Hash for BTreeSet<T, N> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.map.len.hash(state);
        for v in self.iter() {
            v.hash(state);
        }
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a BTreeSet<T, N> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T, N>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<T: Clone, const N: usize> IntoIterator for BTreeSet<T, N> {
    type Item = T;
    type IntoIter = IntoIter<T, N>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            iter: self.map.into_iter(),
        }
    }
}

impl<T: Ord, const N: usize> Ord for BTreeSet<T, N> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.map.cmp(&other.map)
    }
}

impl<T: PartialEq, const N: usize> PartialEq for BTreeSet<T, N> {
    fn eq(&self, other: &BTreeSet<T, N>) -> bool {
        self.map.eq(&other.map)
    }
}

impl<T: PartialOrd, const N: usize> PartialOrd for BTreeSet<T, N> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.map.partial_cmp(&other.map)
    }
}

// impl<K: Clone + Ord, const N: usize> std::ops::Sub for &BTreeSet<K, N> {
//     type Output = BTreeSet<K, N>;

//     fn sub(self, rhs: Self) -> Self::Output {
//         BTreeSet::new_diff(self.clone(), rhs.clone())
//     }
// }
