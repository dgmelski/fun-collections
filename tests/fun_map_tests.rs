extern crate quickcheck;
use fun_collections::FunMap;
use quickcheck::quickcheck;

#[test]
fn rot_rt_regr() {
    let mut fmap = FunMap::new();
    fmap.insert(2, 0);
    fmap.insert(0, 0);
    fmap.insert(1, 0);

    assert_eq!(fmap.len(), 3);
    let mut iter = fmap.iter();
    assert_eq!(iter.next(), Some((&0, &0)));
    assert_eq!(iter.next(), Some((&1, &0)));
    assert_eq!(iter.next(), Some((&2, &0)));
    assert_eq!(iter.next(), None);
}

quickcheck! {
    fn qc_cmp_with_btree(xs: Vec<(u8, u32)>) -> () {
        let mut btree = std::collections::BTreeMap::new();
        let mut fmap = FunMap::new();

        for (k, v) in xs.iter() {
            assert_eq!(btree.len(), fmap.len());
            assert_eq!(btree.insert(*k, *v), fmap.insert(*k, *v));
            assert!(btree.iter().cmp(fmap.iter()).is_eq());
        }
    }
}
