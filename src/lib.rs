use std::rc::Rc;

struct List(
    (i32, Option<Rc<List>>)
);

type OptList = Option<Rc<List>>;

pub struct FunStack {
    sz: usize,
    list: OptList,
}

impl FunStack {
    pub fn new() -> Self {
        FunStack {
            sz: 0,
            list: None,
        }
    }

    pub fn clone(&self) -> Self {
        FunStack {
            sz: self.sz,
            list: self.list.clone(),
        }
    }

    pub fn push(&mut self, v: i32) -> () {
        self.list = Some(Rc::new(List((v, self.list.take()))));
        self.sz += 1;
    }

    pub fn top(&self) -> Option<i32> {
        self.list.as_ref().map(|n| n.0.0)
    }

    pub fn pop(&mut self) -> Option<i32> {
        match self.list.as_mut() {
            Some(rc) => {
                let ret = Some(rc.0.0);
                self.list = rc.0.1.clone();  // .take() ?
                self.sz -= 1;
                ret
            }

            None => None,
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
