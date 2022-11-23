use std::rc::Rc;

enum List {
    Cons(i32, Rc<List>),
    Nil,
}

use List::{Cons, Nil};

pub struct FunStack {
    sz: usize,
    list: Rc<List>,
}

impl FunStack {
    pub fn new() -> Self {
        FunStack {
            sz: 0,
            list: Rc::new(Nil),
        }
    }

    pub fn clone(&self) -> Self {
        FunStack {
            sz: self.sz,
            list: self.list.clone(),
        }
    }

    pub fn push(&mut self, v: i32) -> () {
        self.list = Rc::new(List::Cons(v, self.list.clone()));
        self.sz += 1;
    }

    pub fn top(&self) -> Option<i32> {
        match *self.list {
            Cons(v, _) => Some(v),
            Nil => None,
        }
    }

    pub fn pop(&mut self) -> Option<i32> {
        match self.list.as_ref() {
            Cons(v, next) => {
                let ret = Some(*v);
                self.list = next.clone();
                self.sz -= 1;
                ret
            }

            Nil => None,
        }
    }

    pub fn len(&self) -> usize {
        self.sz
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
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
}
