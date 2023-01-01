use super::*;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[derive(Clone, Eq)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(bound(deserialize = "T: Clone + Ord + Deserialize<'de>"))
)]
pub struct BTreeSet<T, const N: usize = 5> {
    map: BTreeMap<T, (), N>,
}

impl<T, const N: usize> BTreeSet<T, N> {
    /// Moves all elements from other into self and leaves other empty.
    ///
    /// # Examples
    /// ```
    /// use lazy_clone_collections::BTreeSet;
    ///
    /// let mut m1 = BTreeSet::from([0]);
    /// let mut m2 = BTreeSet::from([1]);
    /// m1.append(&mut m2);
    /// assert!(m1.contains(&1));
    /// assert_eq!(m2.len(), 0);
    /// ```
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

    /// Returns an iterator over elements in self and not in other
    pub fn difference<'a>(&'a self, other: &'a Self) -> Difference<'a, T, N>
    where
        T: Ord,
    {
        Difference::new(self.iter(), other.iter())
    }

    // TODO: drain_filter? (Part of unstable API)

    /// Returns the least value in the set.
    ///
    /// # Examples
    /// ```
    /// use lazy_clone_collections::BTreeSet;
    ///
    /// let s = BTreeSet::from([100, 0, 35, 104]);
    /// assert_eq!(s.first(), Some(&0));
    /// ```
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

    /// Returns an iterator of the values that are in both self and other.
    pub fn intersection<'a>(&'a self, other: &'a Self) -> Intersection<'a, T, N>
    where
        T: Ord,
    {
        Intersection::new(self.iter(), other.iter())
    }

    /// Returns true if self and other have no common values and false otherwise
    ///
    /// # Examples
    /// ```
    /// use lazy_clone_collections::BTreeSet;
    ///
    /// let s1 = BTreeSet::from([0, 1, 2]);
    /// let s2 = BTreeSet::from([2, 1, 4]);
    /// assert!(!s1.is_disjoint(&s2));
    /// assert!(s1.is_disjoint(&BTreeSet::new()))
    /// ```
    pub fn is_disjoint(&self, other: &Self) -> bool
    where
        T: Ord,
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
        T: Ord,
    {
        self.is_empty()
            || (self.len() < other.len()
                && self.difference(other).next().is_none())
    }

    /// tests if self is a superset of other.
    pub fn is_superset(&self, other: &Self) -> bool
    where
        T: Ord,
    {
        other.is_subset(self)
    }

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

    /// Creates a new set with the elements of lhs that are not in rhs.
    pub fn new_diff(mut lhs: Self, rhs: Self) -> Self
    where
        T: Clone + Ord,
    {
        if lhs.len() * 4 < rhs.len() {
            lhs.into_iter().filter(|k| !rhs.contains(k)).collect()
        } else {
            for k in rhs {
                lhs.remove(&k);
            }
            lhs
        }
    }

    /// Creates a new set with the elements of lhs that are also in rhs.
    pub fn new_intersection(lhs: Self, rhs: Self) -> Self
    where
        T: Clone + Ord,
    {
        if lhs.len() < rhs.len() {
            lhs.into_iter().filter(|k| rhs.contains(k)).collect()
        } else {
            rhs.into_iter().filter(|k| lhs.contains(k)).collect()
        }
    }

    /// Creates a new set with the elements of both lhs and rhs.
    pub fn new_union(mut lhs: Self, mut rhs: Self) -> Self
    where
        T: Clone + Ord,
    {
        if rhs.len() >= lhs.len() * 4 {
            // rhs is much bigger.  Move lhs to rhs, but give rhs precedence
            for k in lhs {
                rhs.map.entry(k).or_default();
            }
            rhs
        } else {
            lhs.extend(rhs);
            lhs
        }
    }

    /// Creates a new set with the elements that are in lhs or rhs but not both.
    pub fn new_sym_diff(lhs: Self, rhs: Self) -> Self
    where
        T: Clone + Ord,
    {
        let (mut lhs, rhs) = if lhs.len() >= rhs.len() {
            (lhs, rhs)
        } else {
            (rhs, lhs)
        };

        for k in rhs {
            if !lhs.remove(&k) {
                lhs.insert(k);
            }
        }

        lhs
    }

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

    pub fn range<Q, R>(
        &self,
        range: R,
    ) -> impl DoubleEndedIterator<Item = &T> + FusedIterator
    where
        Q: Ord + ?Sized,
        T: Borrow<Q>,
        R: RangeBounds<Q>,
    {
        self.map.range(range).map(|e| e.0)
    }

    /// Removes the given value from self returning true if the value was
    /// present and false otherwise.
    pub fn remove<Q>(&mut self, value: &Q) -> bool
    where
        T: Borrow<Q> + Clone + Ord,
        Q: Ord + ?Sized,
    {
        self.map.remove(value).is_some()
    }

    /// Replace and return the matching value in the map.
    ///
    /// # Examples
    /// ```
    /// use lazy_clone_collections::BTreeSet;
    ///
    /// #[derive(Clone, Copy, Debug)]
    /// struct X(usize); // all X's are equal
    /// impl std::cmp::PartialEq for X {
    ///     fn eq(&self, _: &Self) -> bool { true }
    /// }
    /// impl std::cmp::Eq for X { }
    /// impl std::cmp::PartialOrd for X {
    ///     fn partial_cmp(&self, _: &Self) -> Option<std::cmp::Ordering> {
    ///         Some(std::cmp::Ordering::Equal)
    ///     }
    /// }
    /// impl std::cmp::Ord for X {
    ///     fn cmp(&self, _: &Self) -> std::cmp::Ordering {
    ///         std::cmp::Ordering::Equal
    ///     }
    /// }
    ///
    /// let mut s = BTreeSet::from([(0, X(0))]);
    /// assert_eq!(s.replace((0, X(331))), Some((0, X(0))));
    /// assert_eq!(s.replace((10, X(5))), None);
    /// assert_eq!(s.len(), 1);
    /// assert_eq!(s.iter().next(), Some(&(0, X(331))));
    /// ```
    pub fn replace(&mut self, value: T) -> Option<T>
    where
        T: Clone + Ord,
    {
        if self.contains(&value) {
            let rc = self.map.root.as_mut()?;
            let n = Arc::make_mut(rc);
            let kv = n.get_mut(&value).unwrap();
            Some(replace(&mut kv.0, value))
        } else {
            None
        }
    }

    /// Retain values for which f returns true and discard others
    pub fn retain<F>(&mut self, f: F)
    where
        T: Clone + Ord,
        F: FnMut(&T) -> bool,
    {
        let s = std::mem::take(self);
        self.extend(s.into_iter().filter(f));
    }

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

    /// Returns an iterator over elements in self or other but not both.
    pub fn symmetric_difference<'a>(
        &'a self,
        other: &'a Self,
    ) -> SymmetricDifference<'a, T, N> {
        SymmetricDifference::new(self.iter(), other.iter())
    }

    /// Removes and returns the set member that matches value.
    ///
    /// # Examples
    /// ```
    /// use lazy_clone_collections::BTreeSet;
    ///
    /// let mut s = BTreeSet::new();
    /// s.insert("abc".to_string());
    /// s.insert("def".to_string());
    /// assert_eq!(s.take("abc"), Some(String::from("abc")));
    /// assert_eq!(s.len(), 1);
    /// ```
    pub fn take<Q>(&mut self, value: &Q) -> Option<T>
    where
        T: Borrow<Q> + Clone + Ord,
        Q: Ord + ?Sized,
    {
        // avoid unnecessary cloning
        if !self.contains(value) {
            return None;
        }

        self.map.rm_and_rebal(|n| n.remove(value)).map(|e| e.0)
    }

    /// Returns an iterator over the elements of self and other, ordered by key.
    ///
    /// Common elements are only returned once.
    pub fn union<'a>(&'a self, other: &'a Self) -> Union<'a, T, N> {
        Union::new(self.iter(), other.iter())
    }
}

impl<T, const N: usize> crate::Set for BTreeSet<T, N> {
    type Value = T;

    fn insert_(&mut self, value: T) -> bool
    where
        T: Clone + Ord,
    {
        self.insert(value)
    }
}

pub struct Iter<'a, T, const N: usize> {
    iter: crate::btree::Iter<'a, T, (), N>,
}

impl<'a, T, const N: usize> ExactSizeIterator for Iter<'a, T, N> {
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl<'a, T, const N: usize> FusedIterator for Iter<'a, T, N> {}

impl<'a, T, const N: usize> Iterator for Iter<'a, T, N> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|e| e.0)
    }
}

impl<'a, T, const N: usize> DoubleEndedIterator for Iter<'a, T, N> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back().map(|e| e.0)
    }
}

crate::make_set_op_iter!(Difference, Iter<'a, T, N>, 0b100, N);
crate::make_set_op_iter!(Intersection, Iter<'a, T, N>, 0b010, N);
crate::make_set_op_iter!(Union, Iter<'a, T, N>, 0b111, N);
crate::make_set_op_iter!(SymmetricDifference, Iter<'a, T, N>, 0b101, N);

pub struct IntoIter<T, const N: usize>
where
    T: Clone,
{
    iter: crate::btree::IntoIter<T, (), N>,
}

impl<T: Clone, const N: usize> ExactSizeIterator for IntoIter<T, N> {
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl<T: Clone, const N: usize> FusedIterator for IntoIter<T, N> {}

impl<T: Clone, const N: usize> Iterator for IntoIter<T, N> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|e| e.0)
    }
}

impl<T: Clone, const N: usize> DoubleEndedIterator for IntoIter<T, N> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back().map(|e| e.0)
    }
}

impl<K: Clone + Ord, const N: usize> std::ops::BitAnd for &BTreeSet<K, N> {
    type Output = BTreeSet<K, N>;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self::Output::new_intersection(self.clone(), rhs.clone())
    }
}

impl<K: Clone + Ord, const N: usize> std::ops::BitOr for &BTreeSet<K, N> {
    type Output = BTreeSet<K, N>;

    fn bitor(self, rhs: Self) -> Self::Output {
        BTreeSet::new_union(self.clone(), rhs.clone())
    }
}

impl<K: Clone + Ord, const N: usize> std::ops::BitXor for &BTreeSet<K, N> {
    type Output = BTreeSet<K, N>;

    fn bitxor(self, rhs: Self) -> Self::Output {
        BTreeSet::new_sym_diff(self.clone(), rhs.clone())
    }
}

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

impl<T: Clone + Ord, const N: usize, const M: usize> From<[T; M]>
    for BTreeSet<T, N>
{
    fn from(value: [T; M]) -> Self {
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

impl<K: Clone + Ord, const N: usize> std::ops::Sub for &BTreeSet<K, N> {
    type Output = BTreeSet<K, N>;

    fn sub(self, rhs: Self) -> Self::Output {
        BTreeSet::new_diff(self.clone(), rhs.clone())
    }
}
