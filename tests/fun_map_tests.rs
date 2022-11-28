extern crate quickcheck;
use fun_collections::FunMap;
use quickcheck::quickcheck;

quickcheck! {
    fn qc_cmp_with_btree(xs: Vec<(u8,u8)>) -> () {
        let mut btree = std::collections::BTreeMap::new();
        let mut fmap = FunMap::new();

        for (k, v) in xs.iter() {
            assert_eq!(btree.len(), fmap.len());
            assert_eq!(btree.insert(*k, *v), fmap.insert(*k, *v));
            assert!(btree.iter().cmp(fmap.iter()).is_eq());
        }
    }
}
