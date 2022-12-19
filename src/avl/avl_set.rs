use super::{AvlMap, Node};
use std::borrow::Borrow;
use std::cmp::Ordering::*;
use std::sync::Arc;

/// A sorted set of values.
///
/// The implementation is a wrapper around [`AvlMap<T,()>`].
#[derive(Clone, Eq)]
pub struct AvlSet<T> {
    map: AvlMap<T, ()>,
}

impl<T> AvlSet<T> {
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

    /// Returns an iterator over elements in self and not in other
    pub fn difference<'a>(&'a self, other: &'a Self) -> Difference<'a, T>
    where
        T: Ord,
    {
        Difference::new(self.iter(), other.iter())
    }

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
    pub fn insert(&mut self, value: T) -> bool
    where
        T: Clone + Ord,
    {
        self.map.insert(value, ()).is_none()
    }

    /// Returns an iterator of the values that are in both self and other.
    pub fn intersection<'a>(&'a self, other: &'a Self) -> Intersection<'a, T>
    where
        T: Ord,
    {
        Intersection::new(self.iter(), other.iter())
    }

    /// Returns true if self and other have no common values and false otherwise
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
        if self.is_empty() {
            return true;
        } else if self.len() > other.len() {
            return false;
        }

        fn has_all<K, V>(
            lhs: &Arc<Node<K, V>>,
            w: &mut Vec<(&Arc<Node<K, V>>, bool)>,
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
                if Arc::ptr_eq(lhs, rhs) {
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
                        // TODO: we may be able to pop multiple nodes from rhs
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
        T: Ord,
    {
        other.is_subset(self)
    }

    /// Returns an iterator over self's values in sorted order.
    pub fn iter(&self) -> Iter<T> {
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
        Self { map: AvlMap::new() }
    }

    /// Creates a new set with the elements of lhs that are not in rhs.
    pub fn new_diff(lhs: Self, rhs: Self) -> Self
    where
        T: Clone + Ord,
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
        T: Clone + Ord,
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
        T: Clone + Ord,
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
        T: Clone + Ord,
    {
        let root = super::sym_diff(lhs.map.root, rhs.map.root);
        Self {
            map: AvlMap {
                len: super::len(&root),
                root,
            },
        }
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

    /// Replace and return the matching value in the map.
    pub fn replace(&mut self, value: T) -> Option<T>
    where
        T: Clone + Ord,
    {
        // TODO: adapt insert so we don't need multiple calls.
        let ret = self.take(&value);
        self.insert(value);
        ret
    }

    /// Retain values for which f returns true and discard others
    pub fn retain<F>(&mut self, mut f: F)
    where
        T: Clone + Ord,
        F: FnMut(&T) -> bool,
    {
        fn dfs<V, F>(n: Node<V, ()>, f: &mut F, acc: &mut AvlMap<V, ()>)
        where
            F: FnMut(&V) -> bool,
            V: Clone + Ord,
        {
            if let Some(mut lf) = n.left {
                Arc::make_mut(&mut lf);
                if let Ok(x) = Arc::try_unwrap(lf) {
                    dfs(x, f, acc);
                }
            }

            if f(&n.key) {
                acc.insert(n.key, ());
            }

            if let Some(mut rt) = n.right {
                Arc::make_mut(&mut rt);
                if let Ok(x) = Arc::try_unwrap(rt) {
                    dfs(x, f, acc);
                }
            }
        }

        let Some(mut root) = self.map.root.take() else { return; };
        Arc::make_mut(&mut root);
        let Ok(root) = Arc::try_unwrap(root) else { panic!("try_unwrap fail?") };
        let mut acc = AvlMap::new();
        dfs(root, &mut f, &mut acc);
        self.map = acc;
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
    ) -> SymmetricDifference<'a, T> {
        SymmetricDifference::new(self.iter(), other.iter())
    }

    /// Removes and returns the set member that matches value.
    ///
    /// # Examples
    /// ```
    /// use lazy_clone_collections::AvlSet;
    ///
    /// let mut s = AvlSet::new();
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
        if let Some((k, _)) = super::rm(&mut self.map.root, value).0 {
            self.map.len -= 1;
            Some(k)
        } else {
            None
        }
    }

    /// Returns an iterator over the elements of self and other, ordered by key.
    ///
    /// Common elements are only returned once.
    pub fn union<'a>(&'a self, other: &'a Self) -> Union<'a, T> {
        Union::new(self.iter(), other.iter())
    }
}

pub struct Iter<'a, T> {
    iter: crate::avl::Iter<'a, T, ()>,
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|e| e.0)
    }
}

crate::make_set_op_iter!(Difference, Iter<'a, T>, 0b100);
crate::make_set_op_iter!(Intersection, Iter<'a, T>, 0b010);
crate::make_set_op_iter!(Union, Iter<'a, T>, 0b111);
crate::make_set_op_iter!(SymmetricDifference, Iter<'a, T>, 0b101);

pub struct IntoIter<T> {
    iter: crate::avl::IntoIter<T, ()>,
}

impl<T: Clone> Iterator for IntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|e| e.0)
    }
}

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

impl<K: Clone + Ord> std::ops::BitXor for &AvlSet<K> {
    type Output = AvlSet<K>;

    fn bitxor(self, rhs: Self) -> Self::Output {
        AvlSet::new_sym_diff(self.clone(), rhs.clone())
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for AvlSet<T> {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fmt.write_str("AvlSet(")?;
        fmt.debug_set().entries(self.iter()).finish()?;
        fmt.write_str(")")
    }
}

impl<T> Default for AvlSet<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, T: Clone + Ord> Extend<&'a T> for AvlSet<T> {
    fn extend<I: IntoIterator<Item = &'a T>>(&mut self, iter: I) {
        for x in iter {
            self.insert(x.clone());
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

impl<T: Clone + Ord, const N: usize> From<[T; N]> for AvlSet<T> {
    fn from(value: [T; N]) -> Self {
        Self::from_iter(value.into_iter())
    }
}

impl<T: Clone + Ord> FromIterator<T> for AvlSet<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut s = AvlSet::new();
        s.extend(iter);
        s
    }
}

impl<T: std::hash::Hash> std::hash::Hash for AvlSet<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.map.len.hash(state);
        for v in self.iter() {
            v.hash(state);
        }
    }
}

impl<'a, T> IntoIterator for &'a AvlSet<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<T: Clone> IntoIterator for AvlSet<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            iter: self.map.into_iter(),
        }
    }
}

impl<T: Ord> Ord for AvlSet<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.map.cmp(&other.map)
    }
}

impl<T: PartialEq> PartialEq for AvlSet<T> {
    fn eq(&self, other: &AvlSet<T>) -> bool {
        self.map.eq(&other.map)
    }
}

impl<T: PartialOrd> PartialOrd for AvlSet<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.map.partial_cmp(&other.map)
    }
}

impl<K: Clone + Ord> std::ops::Sub for &AvlSet<K> {
    type Output = AvlSet<K>;

    fn sub(self, rhs: Self) -> Self::Output {
        AvlSet::new_diff(self.clone(), rhs.clone())
    }
}

#[cfg(feature = "serde")]
mod avl_serde {
    use super::AvlSet;
    use serde::de::{Deserialize, SeqAccess, Visitor};
    use std::fmt;
    use std::marker::PhantomData;

    pub(super) struct AvlSetVisitor<T> {
        marker: PhantomData<fn() -> AvlSet<T>>,
    }

    impl<T> AvlSetVisitor<T> {
        pub fn new() -> Self {
            AvlSetVisitor {
                marker: PhantomData,
            }
        }
    }

    impl<'de, T> Visitor<'de> for AvlSetVisitor<T>
    where
        T: Clone + Deserialize<'de> + Ord,
    {
        type Value = AvlSet<T>;

        // Format a message stating what data this Visitor expects to receive.
        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("lazy_clone_collections::AvlSet")
        }

        fn visit_seq<M>(self, mut access: M) -> Result<Self::Value, M::Error>
        where
            M: SeqAccess<'de>,
        {
            let mut set = AvlSet::<T>::new();

            while let Some(elem) = access.next_element()? {
                set.insert(elem);
            }

            Ok(set)
        }
    }
}

#[cfg(feature = "serde")]
impl<T> serde::ser::Serialize for AvlSet<T>
where
    T: serde::ser::Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::ser::Serializer,
    {
        use serde::ser::SerializeSeq;

        let mut seq = serializer.serialize_seq(Some(self.len()))?;
        for k in self {
            seq.serialize_element(k)?;
        }
        seq.end()
    }
}

#[cfg(feature = "serde")]
impl<'de, T> serde::de::Deserialize<'de> for AvlSet<T>
where
    T: Clone + serde::de::Deserialize<'de> + Ord,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::de::Deserializer<'de>,
    {
        deserializer.deserialize_seq(self::avl_serde::AvlSetVisitor::new())
    }
}

#[cfg(test)]
mod test {
    extern crate quickcheck;
    use super::*;
    use quickcheck::quickcheck;

    // this is a compile-time test
    fn _default_sets_for_no_default_entries() {
        struct Foo;
        let _ = AvlSet::<Foo>::default();
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
    fn dbg_fmt_test() {
        let m = AvlSet::from(['a', 'b']);
        assert_eq!(format!("{:?}", m), r#"AvlSet({'a', 'b'})"#);
    }

    // run with: `cargo test --features serde,serde_test`
    #[cfg(feature = "serde_test")]
    mod serde_test {
        use crate::AvlSet;
        use serde_test::{assert_tokens, Token};

        #[test]
        fn test_serde() {
            let mut s = AvlSet::new();
            s.insert('a');
            s.insert('b');
            s.insert('c');

            assert_tokens(
                &s,
                &[
                    Token::Seq { len: Some(3) },
                    Token::Char('a'),
                    Token::Char('b'),
                    Token::Char('c'),
                    Token::SeqEnd,
                ],
            );
        }
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
