use std::fmt;
use std::fmt::{Debug, Formatter};
use std::rc::Rc;

struct List<T> {
    val: T,
    next: Option<Rc<List<T>>>,
}

type OptList<T> = Option<Rc<List<T>>>;

#[derive(Clone)]
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
        match &self.next {
            Some(rc) => {
                let ret = &rc.val;
                self.next = &rc.next;
                Some(ret)
            }
            None => None,
        }
    }
}

impl<T: Clone> FunStack<T> {
    pub fn new() -> Self {
        FunStack { sz: 0, list: None }
    }

    pub fn iter<'a>(&'a self) -> FunStackIter<'a, T> {
        FunStackIter { next: &self.list }
    }

    pub fn push(&mut self, val: T) -> () {
        self.list = Some(Rc::new(List {
            val,
            next: self.list.take(),
        }));
        self.sz += 1;
    }

    pub fn top(&self) -> Option<&T> {
        self.list.as_ref().map(|n| &n.val)
    }

    pub fn pop(&mut self) -> Option<T> {
        let opt_list = self.list.take();
        match opt_list {
            None => None,
            Some(rc) => {
                self.sz -= 1;

                match Rc::try_unwrap(rc) {
                    Ok(mut list) => {
                        self.list = list.next.take();
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

    pub fn len(&self) -> usize {
        self.sz
    }
}

impl<T: Clone + Debug> Debug for FunStack<T> {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> Result<(), fmt::Error> {
        fmt.write_str("FunStack (TOP -> BOT): ")?;
        fmt.debug_list().entries(self.iter()).finish()
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

#[cfg(test)]
mod tests {
    extern crate quickcheck;
    use super::*;
    use quickcheck::quickcheck;
    use std::cell::RefCell;
    use std::cmp::Ordering::*;

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
    fn iter_test() {
        let mut s = FunStack::new();
        s.push('a');
        s.push('b');
        s.push('c');
        for (&c, d) in s.iter().zip(vec!['c', 'b', 'a']) {
            assert_eq!(c, d);
        }
    }

    #[test]
    fn len_test() {
        let mut s = FunStack::new();
        assert_eq!(s.len(), 0);
        s.push(&"abc");
        assert_eq!(s.len(), 1);
        assert_eq!(s.pop(), Some(&"abc"));
        assert_eq!(s.len(), 0);
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
            for (v,f) in vs.iter().rev().zip(fs.iter()) {
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

            match fun_stk.iter().cmp(vec_stk.iter().rev()) {
                Equal => true,
                _ => false,
            }
        }

        fn qc_sharing_test(xs: Vec<i32>) -> () {
            let mut fun_stks = vec![FunStack::new()];
            let mut vec_stks = vec![Vec::new()];

            for &i in xs.iter() {
                // clone the lead stacks
                fun_stks.push(fun_stks[0].clone());  // NB: creates sharing
                vec_stks.push(vec_stks[0].clone());  // WARNING: n^2

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
