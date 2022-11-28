use std::cmp::Ordering::*;
use std::fmt;
use std::fmt::{Debug, Formatter};
use std::rc::Rc;

#[derive(Clone)]
struct List<T> {
    val: T,
    next: OptList<T>,
}

type OptList<T> = Option<Rc<List<T>>>;

#[derive(Clone)]
/// Implements a stack with constant time `push`, `pop`, and `clone` operations.
///
/// The stored elements must implement `Clone`.  During `pop` operations, popped
/// items are moved if they are not shared among different stacks and cloned if
/// they are.
///
/// # Examples
/// In this example, we create a `FunStack<Box<i32>>`.  We use `T = Box<>`
/// to illustrate `FunStack`'s handling of elements that implement `Clone` but
/// not `Copy`.
/// ```
/// use fun_collections::FunStack;
///
/// let mut s = FunStack::new();
/// s.push(Box::new(0));
/// let mut t = s.clone(); // s and t share internal representations
/// s.push(Box::new(1));
/// t.push(Box::new(2));
///
/// assert_eq!(s.pop(), Some(Box::new(1))); // pop moves its Box(1) (unshared)
/// assert_eq!(s.pop(), Some(Box::new(0))); // pop clones Box(0) (shared with t)
/// assert_eq!(t.pop(), Some(Box::new(2))); // pop moves Box(2) (unshared)
/// assert_eq!(t.pop(), Some(Box::new(0))); // pop moves Box(0) (now unshared)
/// ```
pub struct FunStack<T> {
    sz: usize,
    list: OptList<T>,
}

// We could implement the iterator to clone the Rc links and the iterated items.
// This would allow the iterator to outlive the iterated stack or for the stack
// to be updated during iteration.  However, you can achieve the same effect
// by iterating a clone of the stack.  We use references because it should be
// more efficient than cloning Rc's.
pub struct FunStackIter<'a, T> {
    next: &'a OptList<T>,
}

impl<'a, T> Iterator for FunStackIter<'a, T>
where
    T: 'a,
{
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next {
            Some(rc) => {
                let ret = &rc.val;
                self.next = &rc.next;
                Some(ret)
            }
            None => None,
        }
    }
}

pub struct FunStackIterMut<'a, T> {
    // NB: originally, I tried to use `&'a mut Option<Rc<List<T>>>` as the
    // type for next, but I could not get it to work.  When I tried to get
    // next.as_mut() to get at the `&mut Rc...`, the reference had the
    // lifetime of the function, rather than the 'a.
    next: Option<&'a mut Rc<List<T>>>,
}

impl<'a, T: Clone> Iterator for FunStackIterMut<'a, T>
where
    T: 'a,
{
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        self.next.take().map(|rc| {
            let list = Rc::make_mut(rc);
            self.next = list.next.as_mut();
            &mut list.val
        })
    }
}

pub struct FunStackIntoIter<T> {
    stk: FunStack<T>,
}

impl<T: Clone> Iterator for FunStackIntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.stk.pop()
    }
}

impl<T: Clone> FunStack<T> {
    pub fn new() -> Self {
        FunStack { sz: 0, list: None }
    }

    /// Creates an iterator from the top to the bottom elements of the stack.
    ///
    /// # Example
    /// As with most collections, you cannot modify a `FunStack` while you are
    /// iterating over it.  However, because cloning a collection is so cheap,
    /// it is sometimes viable to iterate over a clone while you modify the
    /// original.  In this example, we iterate over a clone while moving the odd
    /// elements in the original stack to its top.
    /// ```
    /// use fun_collections::FunStack;
    /// let mut s: FunStack<_> = (0..4).collect();
    /// for (i, n) in s.clone().iter().enumerate() {
    ///     if n % 2 == 1 {
    ///         let x: i32 = s.remove(i);
    ///         assert_eq!(x, *n);
    ///         s.push(x);
    ///     }
    /// }
    /// assert_eq!(s.pop(), Some(1));
    /// assert_eq!(s.pop(), Some(3));
    /// assert_eq!(s.pop(), Some(2));
    /// assert_eq!(s.pop(), Some(0));
    /// assert_eq!(s.pop(), None);
    /// ```
    pub fn iter(&self) -> FunStackIter<T> {
        FunStackIter { next: &self.list }
    }

    /// Returns an iterator with mutable references.
    ///
    /// Before iterating to a shared node, the iterator will clone it.
    ///
    /// # Examples
    ///
    /// ```
    /// use fun_collections::FunStack;
    ///
    /// let mut s = FunStack::new();
    ///
    /// s.push(0);
    /// let mut t = s.clone();
    /// s.push(1);
    /// t.push(2);
    ///
    /// for x in s.iter_mut() {
    ///     *x += 100;
    /// }
    ///
    /// assert_eq!(s.pop(), Some(101));
    /// assert_eq!(s.pop(), Some(100));
    /// assert_eq!(s.pop(), None);
    ///
    /// assert_eq!(t.pop(), Some(2));
    /// assert_eq!(t.pop(), Some(0));
    /// assert_eq!(s.pop(), None);
    /// ```
    pub fn iter_mut(&mut self) -> FunStackIterMut<T> {
        FunStackIterMut {
            next: self.list.as_mut(),
        }
    }

    pub fn push(&mut self, val: T) {
        self.list = Some(Rc::new(List {
            val,
            next: self.list.take(),
        }));
        self.sz += 1;
    }

    pub fn top(&self) -> Option<&T> {
        self.list.as_ref().map(|n| &n.val)
    }

    // TODO: test, doc
    pub fn top_mut(&mut self) -> Option<&mut T> {
        self.list.as_mut().map(|rc| &mut Rc::make_mut(rc).val)
    }

    pub fn pop(&mut self) -> Option<T> {
        let opt_list = self.list.take();
        match opt_list {
            None => None,
            Some(rc) => {
                self.sz -= 1;

                match Rc::try_unwrap(rc) {
                    Ok(list) => {
                        self.list = list.next;
                        Some(list.val)
                    }

                    Err(rc) => {
                        self.list = rc.next.clone();
                        Some(rc.val.clone())
                    }
                }
            }
        }
    }

    /// Returns the number of elements in the stack.
    ///
    /// # Examples
    /// ```
    /// use fun_collections::FunStack;
    ///
    /// let mut s: FunStack<_> = (0..5).collect();
    /// assert_eq!(s.len(), 5);
    /// ```
    pub fn len(&self) -> usize {
        self.sz
    }

    /// Removes all elements from the stack.
    ///
    /// # Examples
    /// ```
    /// use fun_collections::FunStack;
    ///
    /// let mut s: FunStack<_> = (0..3).collect();
    /// assert_eq!(s.len(), 3);
    /// s.clear();
    /// assert!(s.is_empty());
    /// ```
    pub fn clear(&mut self) {
        self.list.take();
        self.sz = 0;
    }

    /// Tests if the element x occurs in the stack.
    ///
    /// # Examples
    /// ```
    /// use fun_collections::FunStack;
    ///
    /// let s: FunStack<_> = (3..8).collect();
    /// assert!(s.contains(&4));
    /// assert!(!s.contains(&0));
    /// ```
    pub fn contains(&self, x: &T) -> bool
    where
        T: PartialEq<T>,
    {
        self.iter().any(|y| x == y)
    }

    /// Tests if there are any elements in the stack.
    ///
    /// # Examples
    /// ```
    /// use fun_collections::FunStack;
    ///
    /// let mut s = FunStack::new();
    /// assert!(s.is_empty());
    /// s.push(47);
    /// assert!(!s.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.sz == 0
    }

    /// Removes the element at the given index and returns it.
    ///
    /// Any shared nodes on the way to the dropped element are cloned.
    ///
    /// # Panics
    /// Panics if the index is greater than or equal to the length of the stack.
    ///
    /// # Example
    /// ```
    /// use fun_collections::FunStack;
    /// let mut s = FunStack::from(vec!['a','b','c']);
    /// s.remove(1);
    /// assert_eq!(s.pop(), Some('c'));
    /// assert_eq!(s.pop(), Some('a'));
    /// assert_eq!(s.pop(), None);
    /// ```
    pub fn remove(&mut self, mut at: usize) -> T {
        if at >= self.sz {
            panic!("Asked to remove item # {at}, but only {} items.", self.sz)
        }
        self.sz -= 1;

        // find the link we need to update
        let mut curr = &mut self.list;
        while at > 0 {
            // we must clone up to the node we remove so we can route the last
            // link around the node we're dropping
            let n = Rc::make_mut(curr.as_mut().unwrap());
            curr = &mut n.next;
            at -= 1;
        }

        // Route around the node we're dropping.  If the node is shared, we need
        // to clone its contents, but not the node itself.
        match Rc::try_unwrap(curr.take().unwrap()) {
            Ok(n) => {
                *curr = n.next;
                n.val
            }

            Err(n) => {
                *curr = n.next.clone();
                n.val.clone()
            }
        }
    }

    // TODO: "Splits the list into two at the given index. Returns everything
    // after the given index, including the index. This operation should compute
    // in O(n) time."
    pub fn split_off(&mut self, at: usize) -> Self {
        if at >= self.sz {
            panic!("Asked to split off {at} items but only {} items.", self.sz)
        }
        unimplemented!();
    }
}

impl<T: Clone + Debug> Debug for FunStack<T> {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> Result<(), fmt::Error> {
        fmt.write_str("FunStack (TOP -> BOT): ")?;
        fmt.debug_list().entries(self.iter()).finish()
    }
}

impl<T: Clone> Default for FunStack<T> {
    fn default() -> Self {
        FunStack::new()
    }
}

impl<T> Drop for FunStack<T> {
    // avoid deep recursion when dropping a large stack
    fn drop(&mut self) {
        let mut hd_opt = self.list.as_mut().take();
        while let Some(rc) = hd_opt {
            if let Some(hd) = Rc::get_mut(rc) {
                // As sole owner of the top of the stack, we can break its link
                // to the rest of the stack so it will be freed w/o recursing.
                hd_opt = hd.next.as_mut().take();
            } else {
                // There are other owners for the rest of the stack; it won't
                // be dropped at this time.
                break;
            }
        }
    }
}

impl<T: Clone> IntoIterator for FunStack<T> {
    type Item = T;
    type IntoIter = FunStackIntoIter<Self::Item>;

    /// Converts the `FunStack<T>` into an `Iterator<T>`.
    ///
    /// # Example
    /// Demonstrate an implict call to into_iter when a `FunStack` is used as
    /// the iterated collection in a `for` loop.
    /// ```
    /// use fun_collections::FunStack;
    /// let s = FunStack::from(vec![0,1,2]);
    /// let mut expected = 2;
    /// for x in s { // equivalent to `for x in s.into_iter()`
    ///     assert_eq!(x, expected);
    ///     expected -= 1;
    /// }
    /// ```
    fn into_iter(self) -> Self::IntoIter {
        FunStackIntoIter { stk: self }
    }
}

impl<T: Clone + PartialEq> PartialEq for FunStack<T> {
    // TODO: test
    fn eq(&self, rhs: &Self) -> bool {
        self.sz == rhs.sz && self.iter().zip(rhs.iter()).all(|(a, b)| a == b)
    }
}

impl<T: Clone + Eq> Eq for FunStack<T> {}

impl<T: Clone + PartialOrd> PartialOrd for FunStack<T> {
    // tODO: test
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        let mut xs = self.iter();
        let mut ys = other.iter();

        loop {
            match (xs.next(), ys.next()) {
                (Some(a), Some(b)) => {
                    match a.partial_cmp(b) {
                        Some(Equal) => (), // continue
                        res => {
                            return res;
                        }
                    }
                }

                (None, Some(_)) => {
                    return Some(Less);
                }

                (Some(_), None) => {
                    return Some(Greater);
                }

                (None, None) => {
                    return Some(Equal);
                }
            }
        }
    }
}

impl<T: Clone + Ord> Ord for FunStack<T> {
    fn cmp(&self, rhs: &Self) -> std::cmp::Ordering {
        self.iter().cmp(rhs.iter())
    }
}

impl<T: Clone> Extend<T> for FunStack<T> {
    /// Pushes elements from an iterator onto the stack.
    ///
    /// Elements are pushed in order of the iteration, which means they will
    /// be popped in the reverse order.
    ///
    /// # Examples
    /// ```
    /// use fun_collections::FunStack;
    ///
    /// let mut s = FunStack::new();
    /// s.extend((0..10));
    ///
    /// assert_eq!(s.top(), Some(&9));
    /// assert_eq!(s.len(), 10);
    /// assert!(s.into_iter().cmp((0..10).rev()).is_eq());
    /// ```
    fn extend<Iter: IntoIterator<Item = T>>(&mut self, iter: Iter) {
        for x in iter {
            self.push(x);
        }
    }
}

impl<T: Clone> From<Vec<T>> for FunStack<T> {
    fn from(v: Vec<T>) -> Self {
        FunStack::from_iter(v.into_iter())
    }
}

impl<T: Clone> FromIterator<T> for FunStack<T> {
    fn from_iter<Iter: IntoIterator<Item = T>>(iter: Iter) -> Self {
        let mut s: FunStack<T> = FunStack::new();
        s.extend(iter);
        s
    }
}

#[cfg(test)]
mod tests {
    extern crate quickcheck;
    use super::*;
    use quickcheck::quickcheck;
    use std::cell::RefCell;

    #[test]
    fn collect_test() {
        let s: FunStack<_> = vec![0, 1].into_iter().collect();
        assert_eq!(s.iter().cmp(vec![1, 0].iter()), Equal);
    }

    #[test]
    fn diff_hd_shared_tl() {
        let mut s = FunStack::new();
        s.push(1);
        s.push(2);

        let mut s2 = s.clone();
        s.push(3);
        s2.push(4);

        assert_eq!(s.pop(), Some(3));
        assert_eq!(s.pop(), Some(2));
        assert_eq!(s.pop(), Some(1));
        assert_eq!(s2.pop(), Some(4));
        assert_eq!(s2.pop(), Some(2));
        assert_eq!(s2.pop(), Some(1));
    }

    #[test]
    fn iter_lifetime() {
        let mut s = FunStack::new();
        for i in vec![0, 1, 2] {
            s.push(i);
        }

        let mut iter = s.iter();
        let a2 = iter.next();
        let a1 = iter.next();
        let a0 = iter.next();
        drop(iter);

        // The references from the iterator are borrowed from 's' and live even
        // after drop(iter).  The compiler complains if we drop(s) here.

        assert_eq!(*a2.unwrap(), 2);
        assert_eq!(*a1.unwrap(), 1);
        assert_eq!(*a0.unwrap(), 0);
    }

    struct CloneCounter {
        pub counter: Rc<RefCell<usize>>,
    }

    impl CloneCounter {
        fn new(counter: Rc<RefCell<usize>>) -> Self {
            CloneCounter { counter }
        }
    }

    impl Clone for CloneCounter {
        fn clone(&self) -> Self {
            let mut v = (*self.counter).borrow_mut();
            *v += 1;
            CloneCounter::new(self.counter.clone())
        }
    }

    #[test]
    fn cloned_pop() {
        let mut s = FunStack::new();
        let cntr = Rc::new(RefCell::new(0_usize));
        s.push(CloneCounter::new(cntr.clone()));

        // creates sharing of stack contents
        let mut t = s.clone();
        assert_eq!(*(*cntr).borrow(), 0);

        // will require clone since stack is shared
        s.pop();
        assert_eq!(*(*cntr).borrow(), 1);

        // should move the result, instead of cloning
        t.pop();
        assert_eq!(*(*cntr).borrow(), 1);
    }

    #[test]
    fn cloned_ref_pop() {
        let cntr = CloneCounter::new(Rc::new(RefCell::new(0_usize)));
        let mut s = FunStack::new();

        s.push(&cntr);

        // creates sharing of stack contents
        let mut t = s.clone();
        assert_eq!(*(*cntr.counter).borrow(), 0);

        // Since the top of the stacks are shared, s.pop() ultimately calls
        // rc.val.clone().  'val' should have type &CloneCounter, however
        // rc.val.clone() does not dereference to CloneCounter::clone().
        // TODO: why not?
        s.pop();
        assert_eq!(*(*cntr.counter).borrow(), 0);

        // should move the result, instead of cloning
        t.pop();
        assert_eq!(*(*cntr.counter).borrow(), 0);
    }

    #[test]
    fn stack_of_stacks() {
        let mut vss = vec![vec![0, 1, 2], vec![3, 4], vec![5, 6, 7]];

        let mut fss = FunStack::new();
        for row in vss.iter() {
            let mut new_row = FunStack::new();
            for c in row.iter() {
                new_row.push(*c);
            }
            fss.push(new_row);
        }

        while let Some(vs) = vss.pop() {
            let fs = fss.pop().unwrap();
            for (v, f) in vs.iter().rev().zip(fs.iter()) {
                assert_eq!(*v, *f)
            }
        }
    }

    quickcheck! {
        fn qc_cmp_with_vec(xs: Vec<i32>) -> bool {
            let mut fun_stk = FunStack::new();
            let mut vec_stk = Vec::new();

            for &i in xs.iter() {
                if i < 0 {
                    assert_eq!(fun_stk.pop(), vec_stk.pop());
                } else {
                    fun_stk.push(i);
                    vec_stk.push(i);
                }
                assert_eq!(fun_stk.len(), vec_stk.len());
            }

            fun_stk.iter().cmp(vec_stk.iter().rev()).is_eq()
        }

        fn qc_sharing_test(xs: Vec<i32>) -> () {
            let mut fun_stks = vec![FunStack::new()];
            let mut vec_stks = vec![Vec::new()];

            for &i in xs.iter() {
                // every so often, clone the lead stack
                if i % 3 == 0 {
                    fun_stks.push(fun_stks[0].clone());  // NB: creates sharing
                    vec_stks.push(vec_stks[0].clone());  // WARNING: n^2
                }

                // update the lead stack by pushing or popping
                if i < 0 {
                    fun_stks[0].pop();
                    vec_stks[0].pop();
                } else {
                    fun_stks[0].push(i);
                    vec_stks[0].push(i);
                }
            }

            // Are the stacks equal?  Even as dropping FunStacks reduces
            // sharing?
            while let Some(s1) = fun_stks.pop() {
                let s2 = vec_stks.pop().unwrap();
                assert_eq!(s1.len(), s2.len());
                assert!(s1.iter().cmp(s2.iter().rev()).is_eq());
            }
        }
    }
}
