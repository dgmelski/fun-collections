//! Microbenchmarks of AvlMap against BTreeMap and HashMap.
//!
//! Invoke with
//! ```
//!     cargo +nightly bench [partial_test_name] --test benchmarks \
//!         --features bench
//! ```
//!
//! The "enable_bench" feature is a feature we introduced in the cargo.toml to
//! use as a gate that controls when the benchmark code is built.  This was
//! necessary because `#[bench]` requires `#![feature(test)]` which requires
//! nightly, but GitHub doesn't provide nightly (at least by default).
//!
//! If "partial_test_name" is excluded, all benchmarks are run.  If given, any
//! test name that contains partial_test_name will run.
#![cfg(feature = "bench")]
#![feature(test)]

extern crate test;

// An xmacro that takes the name of another macro and invokes it once for each
// of the map types we are testing: BTreeMap, AvlMap, and HashMap.  The passed
// argument should produce a benchmark function.
//
// We sidestep the issue that rust does not (easily) support token concatenation
// by creating submodules.  We reuse (abuse) the name of the macro parameter to
// create a separate namespace to separate the other instantiations.  Similarly,
// we create namespaces for each map type to separate the function instances,
// which typically have an short name like 'f'.
macro_rules! for_each_map_type {
    ( $macro_name:ident ) => {
        mod $macro_name {
            mod btreemap {
                use std::collections::BTreeMap;
                use test::Bencher;

                $macro_name!(BTreeMap);
            }

            mod hashmap {
                use std::collections::HashMap;
                use test::Bencher;

                $macro_name!(HashMap);
            }

            mod shared_avlmap {
                use lazy_clone_collections::AvlMap;
                use test::Bencher;

                $macro_name!(AvlMap);
            }

            mod shared_btree_minpop_0001 {
                use test::Bencher;

                type BTreeMap<K, V> =
                    lazy_clone_collections::btree::BTreeMap<K, V, 1>;

                $macro_name!(BTreeMap);
            }

            mod shared_btree_minpop_0002 {
                use test::Bencher;

                type BTreeMap<K, V> =
                    lazy_clone_collections::btree::BTreeMap<K, V, 2>;

                $macro_name!(BTreeMap);
            }

            mod shared_btree_minpop_0003 {
                use test::Bencher;

                type BTreeMap<K, V> =
                    lazy_clone_collections::btree::BTreeMap<K, V, 3>;

                $macro_name!(BTreeMap);
            }

            mod shared_btree_minpop_0007 {
                use test::Bencher;

                type BTreeMap<K, V> =
                    lazy_clone_collections::btree::BTreeMap<K, V, 7>;

                $macro_name!(BTreeMap);
            }

            mod shared_btree_minpop_0015 {
                use test::Bencher;

                type BTreeMap<K, V> =
                    lazy_clone_collections::btree::BTreeMap<K, V, 15>;

                $macro_name!(BTreeMap);
            }

            mod shared_btree_minpop_0031 {
                use test::Bencher;

                type BTreeMap<K, V> =
                    lazy_clone_collections::btree::BTreeMap<K, V, 31>;

                $macro_name!(BTreeMap);
            }

            mod shared_btree_minpop_1023 {
                use test::Bencher;

                type BTreeMap<K, V> =
                    lazy_clone_collections::btree::BTreeMap<K, V, 31>;

                $macro_name!(BTreeMap);
            }
        }
    };
}

macro_rules! build_500_elems {
    ( $map_t:ident ) => {
        #[bench]
        fn f(b: &mut Bencher) {
            b.iter(|| {
                let mut m = $map_t::new();
                for i in 0..500 {
                    m.insert(i, i);
                }
                m
            });
        }
    };
}

for_each_map_type!(build_500_elems);

macro_rules! clone_1000_elems {
    ( $map_t:ident ) => {
        #[bench]
        fn f(b: &mut Bencher) {
            let m: $map_t<_, _> = (0..1000).map(|x| (x, x)).collect();
            b.iter(|| m.clone());
        }
    };
}

for_each_map_type!(clone_1000_elems);

macro_rules! get_500_elems {
    ( $map_t:ident ) => {
        const MAP_LEN: usize = 500;

        #[bench]
        fn f(b: &mut Bencher) {
            let m: $map_t<_, _> = (0..MAP_LEN).map(|x| (x, x)).collect();
            b.iter(|| {
                let mut s = 0;
                for i in 0..MAP_LEN {
                    s += m.get(&i).unwrap();
                }
                s
            });
        }
    };
}

for_each_map_type!(get_500_elems);

macro_rules! iter_500_elems {
    ( $map_t:ident ) => {
        const MAP_LEN: usize = 500;

        #[bench]
        fn f(b: &mut Bencher) {
            let m: $map_t<_, _> = (0..MAP_LEN).map(|x| (x, x)).collect();
            b.iter(|| {
                let mut s = 0;
                for (k, v) in m.iter() {
                    s += k * v;
                }
                s
            });
        }
    };
}

for_each_map_type!(iter_500_elems);

macro_rules! remove_1000_elems {
    ( $map_t: ident ) => {
        const MAP_LEN: usize = 1000;

        #[bench]
        fn f(b: &mut Bencher) {
            let m: $map_t<_, _> = (0..MAP_LEN).map(|x| (x, x)).collect();
            b.iter(|| {
                let mut s = 0;
                let mut m = m.clone();
                for i in 0..MAP_LEN {
                    s += m.remove(&i).unwrap();
                }
                (s, m)
            });
        }
    };
}

for_each_map_type!(remove_1000_elems);

macro_rules! clone_and_removes_string_map {
    ( $map_t:ident ) => {
        const MAP_LEN: usize = 500;
        const CNT_CLONES: usize = 20;

        #[bench]
        fn f(b: &mut Bencher) {
            let m: $map_t<_, _> =
                (0..MAP_LEN).map(|x| (x, format!("{x}"))).collect();

            b.iter(|| {
                let mut s = 0;
                let mut v = vec![m.clone(); CNT_CLONES];
                for i in 0..(MAP_LEN / CNT_CLONES) {
                    for j in 1..CNT_CLONES {
                        s += v[j].remove(&(i * CNT_CLONES + j)).unwrap().len();
                    }
                }
                s
            });
        }
    };
}

for_each_map_type!(clone_and_removes_string_map);

macro_rules! clone_and_updates_rc_map {
    ( $map_t:ident ) => {
        use std::rc::Rc;

        const MAP_LEN: usize = 1000;
        const CNT_CLONES: usize = 100;
        const CNT_UPDATES: usize = 20;

        #[bench]
        fn f(b: &mut Bencher) {
            let m: $map_t<_, _> =
                (0..MAP_LEN).map(|x| (x, Rc::new(x))).collect();
            b.iter(|| {
                let mut s = 0;
                let mut vs = vec![m.clone(); CNT_CLONES];
                for v in vs.iter_mut() {
                    for idx in 0..CNT_UPDATES {
                        s += *v.insert(idx, Rc::new(idx)).unwrap();
                    }
                }
                (s, vs)
            });
        }
    };
}

for_each_map_type!(clone_and_updates_rc_map);

macro_rules! symex_sim {
    ( $map_t: ident ) => {
        use std::collections::VecDeque;
        use std::rc::Rc;

        const MAX_WORK: usize = 1000;
        const MAP_LEN: usize = 1000;

        #[bench]
        fn f(b: &mut Bencher) {
            // In symbolic execution, we would have a map from memory locations
            // to symbolic expressions.  The symbolic expressions would be a
            // reference counted data structure (with shared representation of
            // common subexpressions).  Here, we use a reference-counted usize
            // as a proxy for the reference-counted symbolic expression.  Each
            // clone of the map has to maintain the reference counts of the
            // "expressions."
            let m: $map_t<_, _> =
                (0..MAP_LEN).map(|x| (x, Rc::new(x))).collect();

            b.iter(|| {
                // Our symbolic execution engine will use a deque to do a
                // breadth-first search of reachable symbolic states.
                let mut w = VecDeque::new();
                w.push_back(m.clone());

                let mut addr = 0;
                while w.len() < MAX_WORK {
                    // get the current "symbolic (memory) state"
                    let mut m = w.pop_front().unwrap();

                    // update the state for the current basic block; modeled as
                    // updates of some symbolic locations
                    for a in addr..(addr + 8) {
                        let x = m.get_mut(&(a % m.len())).unwrap();
                        *x = Rc::new(**x + 1);
                    }
                    addr = (addr + 8) % m.len();

                    // Split the state for the branch at the end of the block.
                    // Normally, the states would get complementary constraints,
                    // but that's separate from the symbolic memory we're
                    // modeling.
                    let m1 = m.clone();
                    w.push_back(m);
                    w.push_back(m1);
                }
                w
            });
        }
    };
}

for_each_map_type!(symex_sim);

// macro_rules! symex_btree_width {
//     ( $mod:ident, $w:literal ) => {
//         mod $mod {
//             use test::Bencher;
//             type BTreeMap<K, V> = lazy_clone_collections::btree::BTreeMap<K, V, $w>;

//             symex_sim!(BTreeMap);
//         }
//     }
// }

// symex_btree_width!(symex_sim_btree_minpop_0001, 1);
// symex_btree_width!(symex_sim_btree_minpop_0002, 2);
// symex_btree_width!(symex_sim_btree_minpop_0003, 3);
// symex_btree_width!(symex_sim_btree_minpop_0004, 4);
// symex_btree_width!(symex_sim_btree_minpop_0005, 5);
// symex_btree_width!(symex_sim_btree_minpop_0006, 6);
// symex_btree_width!(symex_sim_btree_minpop_0007, 7);
// symex_btree_width!(symex_sim_btree_minpop_0008, 8);
// symex_btree_width!(symex_sim_btree_minpop_0012, 12);
// symex_btree_width!(symex_sim_btree_minpop_0015, 15);
// symex_btree_width!(symex_sim_btree_minpop_0016, 16);
// symex_btree_width!(symex_sim_btree_minpop_0031, 31);
// symex_btree_width!(symex_sim_btree_minpop_0032, 32);
// symex_btree_width!(symex_sim_btree_minpop_0063, 63);
// symex_btree_width!(symex_sim_btree_minpop_0127, 127);
// symex_btree_width!(symex_sim_btree_minpop_0255, 255);
// symex_btree_width!(symex_sim_btree_minpop_0511, 511);
// symex_btree_width!(symex_sim_btree_minpop_1023, 1023);

#[bench]
fn avl_into_iter(b: &mut test::Bencher) {
    let vs: Vec<_> = (0..1000).map(|i| (i, ())).collect();
    b.iter(|| {
        let m: lazy_clone_collections::AvlMap<_, _> =
            vs.iter().copied().collect();
        m.into_iter().map(|e| e.0).sum::<usize>();
    });
}
