extern crate quickcheck;
use lazy_clone_collections::AvlMap;
use quickcheck::quickcheck;

#[test]
fn rot_rt_regr() {
    let mut fmap = AvlMap::new();
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
        let mut fmap = AvlMap::new();

        for (k, v) in xs.iter() {
            assert_eq!(btree.len(), fmap.len());
            assert_eq!(btree.insert(*k, *v), fmap.insert(*k, *v));
            assert!(btree.iter().cmp(fmap.iter()).is_eq());
        }

        for k in 0..=u8::MAX {
            assert_eq!(fmap.get(&k), btree.get(&k));
        }
    }
}
