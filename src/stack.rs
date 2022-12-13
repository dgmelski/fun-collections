use std::cmp::Ordering::*;
use std::fmt;
use std::fmt::{Debug, Formatter};
use std::rc::Rc;

#[derive(Clone)]
struct List<T> {
    elem: T,
    rest: OptList<T>,
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
/// In this example, we create a `Stack` of `Box`es to illustrate
/// `Stack`'s use of clone on stored elements.
/// ```
/// use lazy_clone_collections::Stack;
///
/// let mut s = Stack::new();
/// s.push(Box::new(0));
/// let mut t = s.clone();
///
/// // At this point, s and t share their internal storage.
///
/// s.push(Box::new(1));
/// t.push(Box::new(2));
///
/// // s and t still have common storage of the Box(0) element, but also
/// // their own storage for Box(1) and Box(2), respectively.
///
/// assert_eq!(s.pop(), Some(Box::new(1))); // pop moves its Box(1) (unshared)
/// assert_eq!(s.pop(), Some(Box::new(0))); // pop clones Box(0) (shared with t)
/// assert_eq!(t.pop(), Some(Box::new(2))); // pop moves Box(2) (unshared)
/// assert_eq!(t.pop(), Some(Box::new(0))); // pop moves Box(0) (now unshared)
/// ```
pub struct Stack<T> {
    len: usize,
    elems: OptList<T>,
}

// We could implement the iterator to clone the Rc links and the iterated items.
// This would allow the iterator to outlive the iterated stack or for the stack
// to be updated during iteration.  However, you can achieve the same effect
// by iterating a clone of the stack.  We use references because it should be
// more efficient than cloning Rc's.
pub struct Iter<'a, T> {
    next: &'a OptList<T>,
}

impl<'a, T> Iterator for Iter<'a, T>
where
    T: 'a,
{
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next {
            Some(rc) => {
                let ret = &rc.elem;
                self.next = &rc.rest;
                Some(ret)
            }
            None => None,
        }
    }
}

pub struct IterMut<'a, T> {
    // NB: originally, I tried to use `&'a mut Option<Rc<List<T>>>` as the
    // type for next, but I could not get it to work.  When I tried to get
    // next.as_mut() to get at the `&mut Rc...`, the reference had the
    // lifetime of the function, rather than the 'a.
    next: Option<&'a mut Rc<List<T>>>,
}

impl<'a, T: Clone> Iterator for IterMut<'a, T>
where
    T: 'a,
{
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        self.next.take().map(|rc| {
            let list = Rc::make_mut(rc);
            self.next = list.rest.as_mut();
            &mut list.elem
        })
    }
}

pub struct IntoIter<T> {
    stk: Stack<T>,
}

impl<T: Clone> Iterator for IntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.stk.pop()
    }
}

impl<T: Clone> Stack<T> {
    /// Creates an empty stack.
    ///
    /// # Examples
    /// ```
    /// use lazy_clone_collections::Stack;
    ///
    /// let stack: Stack<i32> = Stack::new();
    /// ```
    pub fn new() -> Self {
        Stack {
            len: 0,
            elems: None,
        }
    }

    /// Creates an iterator from the top to the bottom elements of the stack.
    ///
    /// # Example
    /// As with most collections, you cannot modify a `Stack` while you are
    /// iterating over it.  However, because cloning a collection is so cheap,
    /// it is sometimes viable to iterate over a clone while you modify the
    /// original.  In this example, we iterate over a clone while moving the odd
    /// elements in the original stack to its top.
    /// ```
    /// use lazy_clone_collections::Stack;
    ///
    /// let mut s: Stack<_> = (0..6).collect();
    ///
    /// for (i, n) in s.clone().iter().enumerate() {
    ///     if n % 2 == 1 {
    ///         // remove indexes bottom to top, while iter goes top to bottom
    ///         let x: i32 = s.remove(5 - i);
    ///         assert_eq!(x, *n);
    ///         s.push(x);
    ///     }
    /// }
    ///
    /// assert_eq!(s.pop(), Some(1));
    /// assert_eq!(s.pop(), Some(3));
    /// assert_eq!(s.pop(), Some(5));
    /// assert_eq!(s.pop(), Some(4));
    /// assert_eq!(s.pop(), Some(2));
    /// assert_eq!(s.pop(), Some(0));
    /// assert_eq!(s.pop(), None);
    /// ```
    pub fn iter(&self) -> Iter<T> {
        Iter { next: &self.elems }
    }

    /// Returns an iterator with mutable references.
    ///
    /// Before iterating to a shared node, the iterator will clone it.
    ///
    /// # Examples
    ///
    /// ```
    /// use lazy_clone_collections::Stack;
    ///
    /// let mut s = Stack::new();
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
    pub fn iter_mut(&mut self) -> IterMut<T> {
        IterMut {
            next: self.elems.as_mut(),
        }
    }

    /// Pushes an element on top of the stack.
    ///
    /// # Examples
    /// ```
    /// use lazy_clone_collections::Stack;
    ///
    /// let mut s = Stack::new();
    ///
    /// s.push("hello");
    /// assert_eq!(s.top(), Some(&"hello"));
    /// ```
    pub fn push(&mut self, val: T) {
        self.elems = Some(Rc::new(List {
            elem: val,
            rest: self.elems.take(),
        }));
        self.len += 1;
    }

    /// Returns a reference to the top of the stack, or `None` if the stack is
    /// empty.
    ///
    /// # Examples
    /// ```
    /// use lazy_clone_collections::Stack;
    ///
    /// let mut s: Stack<_> = (0..3).collect();
    /// assert_eq!(s.top(), Some(&2));
    ///
    /// s.clear();
    /// assert_eq!(s.top(), None);
    /// ```
    pub fn top(&self) -> Option<&T> {
        self.elems.as_ref().map(|n| &n.elem)
    }

    /// Returns a mutable ref to the top of the stack or `None` if empty.
    ///
    /// Clones the top node if it is shared.
    ///
    /// # Examples
    /// ```
    ///
    /// use lazy_clone_collections::Stack;
    ///
    /// let mut s: Stack<_> = (0..=3).collect();
    ///
    /// assert_eq!(s.top_mut(), Some(&mut 3));
    ///
    /// *s.top_mut().unwrap() += 5;
    ///
    /// assert_eq!(s.pop(), Some(8));
    /// assert_eq!(s.pop(), Some(2));
    /// ```
    pub fn top_mut(&mut self) -> Option<&mut T> {
        self.elems.as_mut().map(|rc| &mut Rc::make_mut(rc).elem)
    }

    /// Returns the top of the stack or `None` if empty.
    pub fn pop(&mut self) -> Option<T> {
        let opt_list = self.elems.take();
        match opt_list {
            None => None,
            Some(rc) => {
                self.len -= 1;

                match Rc::try_unwrap(rc) {
                    Ok(list) => {
                        self.elems = list.rest;
                        Some(list.elem)
                    }

                    Err(rc) => {
                        self.elems = rc.rest.clone();
                        Some(rc.elem.clone())
                    }
                }
            }
        }
    }

    /// Returns the number of elements in the stack.
    ///
    /// # Examples
    /// ```
    /// use lazy_clone_collections::Stack;
    ///
    /// let mut s: Stack<_> = (0..5).collect();
    /// assert_eq!(s.len(), 5);
    /// ```
    pub fn len(&self) -> usize {
        self.len
    }

    /// Removes all elements from the stack.
    ///
    /// # Examples
    /// ```
    /// use lazy_clone_collections::Stack;
    ///
    /// let mut s: Stack<_> = (0..3).collect();
    /// assert_eq!(s.len(), 3);
    /// s.clear();
    /// assert!(s.is_empty());
    /// ```
    pub fn clear(&mut self) {
        self.elems.take();
        self.len = 0;
    }

    /// Tests if the element x occurs in the stack.
    ///
    /// # Examples
    /// ```
    /// use lazy_clone_collections::Stack;
    ///
    /// let s: Stack<_> = (3..8).collect();
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
    /// use lazy_clone_collections::Stack;
    ///
    /// let mut s = Stack::new();
    /// assert!(s.is_empty());
    /// s.push(47);
    /// assert!(!s.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Removes and returns the element at the given index.
    ///
    /// The bottom element of the stack has index 0 and the top has index
    /// `self.len() - 1`. Any shared nodes above the dropped element are cloned.
    ///
    /// # Panics
    /// Panics if the index is greater than or equal to the length of the stack.
    ///
    /// # Example
    /// ```
    /// use lazy_clone_collections::Stack;
    /// let mut s = Stack::from(vec!['a','b','c', 'd']);
    /// assert_eq!(s.remove(1), 'b');
    /// assert_eq!(s.len(), 3);
    /// assert_eq!(s.pop(), Some('d'));
    /// assert_eq!(s.pop(), Some('c'));
    /// assert_eq!(s.pop(), Some('a'));
    /// assert_eq!(s.pop(), None);
    /// ```
    pub fn remove(&mut self, at: usize) -> T {
        if at >= self.len {
            panic!("Asked to remove item # {at}, but only {} items.", self.len)
        }

        // switch to counting elements from the top
        let mut at = self.len - 1 - at;

        // find the link we need to update
        let mut curr = &mut self.elems;
        while at > 0 {
            // we must ensure we have sole ownership of all the nodes up to the
            // one we're dropping so we can route around it
            let n = Rc::make_mut(curr.as_mut().unwrap());
            curr = &mut n.rest;
            at -= 1;
        }

        self.len -= 1;

        // Route around the node we're dropping.  If the node is shared, we need
        // to clone its contents, but not the node itself.
        match Rc::try_unwrap(curr.take().unwrap()) {
            Ok(n) => {
                *curr = n.rest;
                n.elem
            }

            Err(n) => {
                *curr = n.rest.clone();
                n.elem.clone()
            }
        }
    }

    /// Splits the stack at the given index, retaining the bottom elements from
    /// [0..at] and returning a new stack with the top elements from [at..].
    ///
    /// The bottom element has index 0 and the top has index `self.len()-1`. Any
    /// shared nodes in the returned top portion of the stack are cloned.
    ///
    /// # Panics
    /// Panics if `at > self.len()`.
    ///
    /// # Examples
    /// ```
    /// use lazy_clone_collections::Stack;
    ///
    /// let mut s: Stack<_> = (0..21).collect();
    /// let t = s.split_off(7);
    ///
    /// assert_eq!(s.len(), 7);
    /// assert_eq!(t.len(), 14);
    /// assert!(s.into_iter().cmp((0..7).rev()).is_eq());
    /// assert!(t.into_iter().cmp((7..21).rev()).is_eq());
    /// ```
    pub fn split_off(&mut self, mut at: usize) -> Self {
        if at > self.len {
            panic!("Asked to split off {at} items but only {} avail.", self.len)
        }

        if at == 0 {
            return Stack {
                len: std::mem::take(&mut self.len),
                elems: self.elems.take(),
            };
        }

        if at == self.len {
            return Stack::new();
        }

        // for the remainder, count from top instead of from bottom
        at = self.len() - at;

        // save the sizes of the split stacks
        let sz_top_part = at;
        let sz_bot_part = self.len - at;

        // find the link we need to sever
        let mut curr = &mut self.elems;
        while at > 0 {
            // we must ensure we own all the nodes up to the split point so we
            // can break the final link
            let n = Rc::make_mut(curr.as_mut().unwrap());
            curr = &mut n.rest;
            at -= 1;
        }

        // execute the split
        let bot_elems = curr.take();
        let top_elems = self.elems.take();

        self.len = sz_bot_part;
        self.elems = bot_elems;

        Stack {
            len: sz_top_part,
            elems: top_elems,
        }
    }
}

impl<T: Clone + Debug> Debug for Stack<T> {
    /// Prints the `Stack` to the supplied `Formatter`.
    ///
    /// # Examples
    /// ```
    /// use lazy_clone_collections::Stack;
    ///
    /// let s: Stack<_> = (0..3).collect();
    ///
    /// assert_eq!(&format!("{:?}", s),
    ///     "Stack { len: 3, elems: TOP[2, 1, 0]BOT }");
    /// ```
    fn fmt(&self, fmt: &mut Formatter<'_>) -> Result<(), fmt::Error> {
        fmt.write_fmt(format_args!("Stack {{ len: {}, elems: TOP", self.len))?;
        fmt.debug_list().entries(self.iter()).finish()?;
        fmt.write_str("BOT }")
    }
}

impl<T: Clone> Default for Stack<T> {
    fn default() -> Self {
        Stack::new()
    }
}

impl<T> Drop for Stack<T> {
    // avoid deep recursion when dropping a large stack
    fn drop(&mut self) {
        let mut hd_opt = self.elems.as_mut().take();
        while let Some(rc) = hd_opt {
            if let Some(hd) = Rc::get_mut(rc) {
                // As sole owner of the top of the stack, we can break its link
                // to the rest of the stack so it will be freed w/o recursing.
                hd_opt = hd.rest.as_mut().take();
            } else {
                // There are other owners for the rest of the stack; it won't
                // be dropped at this time.
                break;
            }
        }
    }
}

impl<T: Clone> IntoIterator for Stack<T> {
    type Item = T;
    type IntoIter = IntoIter<Self::Item>;

    /// Converts the `Stack<T>` into an `Iterator<T>`.
    ///
    /// # Example
    /// Demonstrate an implict call to into_iter when a `Stack` is used as
    /// the iterated collection in a `for` loop.
    /// ```
    /// use lazy_clone_collections::Stack;
    /// let s = Stack::from(vec![0,1,2]);
    /// let mut expected = 2;
    /// for x in s { // equivalent to `for x in s.into_iter()`
    ///     assert_eq!(x, expected);
    ///     expected -= 1;
    /// }
    /// ```
    fn into_iter(self) -> Self::IntoIter {
        IntoIter { stk: self }
    }
}

impl<T: Clone + PartialEq> PartialEq for Stack<T> {
    // TODO: test
    fn eq(&self, rhs: &Self) -> bool {
        self.len == rhs.len && self.iter().zip(rhs.iter()).all(|(a, b)| a == b)
    }
}

impl<T: Clone + Eq> Eq for Stack<T> {}

impl<T: Clone + PartialOrd> PartialOrd for Stack<T> {
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

impl<T: Clone + Ord> Ord for Stack<T> {
    fn cmp(&self, rhs: &Self) -> std::cmp::Ordering {
        self.iter().cmp(rhs.iter())
    }
}

impl<T: Clone> Extend<T> for Stack<T> {
    /// Pushes elements from an iterator onto the stack.
    ///
    /// Elements are pushed in order of the iteration, which means they will
    /// be popped in the reverse order.
    ///
    /// # Examples
    /// ```
    /// use lazy_clone_collections::Stack;
    ///
    /// let mut s = Stack::new();
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

impl<T: Clone> From<Vec<T>> for Stack<T> {
    fn from(v: Vec<T>) -> Self {
        Stack::from_iter(v.into_iter())
    }
}

impl<T: Clone> FromIterator<T> for Stack<T> {
    fn from_iter<Iter: IntoIterator<Item = T>>(iter: Iter) -> Self {
        let mut s: Stack<T> = Stack::new();
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
        let s: Stack<_> = vec![0, 1].into_iter().collect();
        assert_eq!(s.iter().cmp(vec![1, 0].iter()), Equal);
    }

    #[test]
    fn diff_hd_shared_tl() {
        let mut s = Stack::new();
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
        let mut s = Stack::new();
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
        let mut s = Stack::new();
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
        let mut s = Stack::new();

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

        let mut fss = Stack::new();
        for row in vss.iter() {
            let mut new_row = Stack::new();
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

    #[test]
    fn split_at_test() {
        let mut s: Stack<_> = (0..21).collect();
        let t = s.split_off(7);

        assert_eq!(s.len(), 7);
        assert_eq!(t.len(), 14);
        assert!(s.into_iter().cmp((0..7).rev()).is_eq());
        assert!(t.into_iter().cmp((7..21).rev()).is_eq());

        s = (0..3).collect();
        let t = s.split_off(0);
        assert!(s.is_empty());
        assert!(t.into_iter().cmp((0..3).rev()).is_eq());

        s = (0..3).collect();
        let t = s.split_off(3);
        assert!(s.into_iter().cmp((0..3).rev()).is_eq());
        assert!(t.is_empty());
    }

    quickcheck! {
        fn qc_sharing_test(xs: Vec<i32>) -> () {
            let mut fun_stks = vec![Stack::new()];
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

            // Are the stacks equal?  Even as dropping Stacks reduces
            // sharing?
            while let Some(s1) = fun_stks.pop() {
                let s2 = vec_stks.pop().unwrap();
                assert_eq!(s1.len(), s2.len());
                assert!(s1.iter().cmp(s2.iter().rev()).is_eq());
            }
        }
    }
}
