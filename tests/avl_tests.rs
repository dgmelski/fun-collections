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

#[test]
fn entry_test() {
    let mut m = AvlMap::from([(0, 0), (1, 1), (2, 2)]);
    m.entry(0).and_modify(|v| *v = 7);
    assert_eq!(m.entry(3).or_default(), &0);
    assert_eq!(m.entry(4).or_insert(4), &4);

    assert_eq!(m.get(&0), Some(&7));
    assert_eq!(m.get(&3), Some(&0));
    assert_eq!(m.get(&4), Some(&4));
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
