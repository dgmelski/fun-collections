//! Microbenchmarks of FunMap against BTreeMap and HashMap.
//!
//! Invoke with
//! ```
//!     cargo +nightly bench [partial_test_name] --test fun_map_benchmarks \
//!         --features enable_bench
//! ```
//!
//! The "enable_bench" feature is a feature we introduced in the cargo.toml to
//! use as a gate that controls when the benchmark code is built.  This was
//! necessary because `#[bench]` requires `#![feature(test)]` which requires
//! nightly, but GitHub doesn't provide nightly (at least by default).
//!
//! If "partial_test_name" is excluded, all benchmarks are run.  If given, any
//! test name that contains partial_test_name will run.
#![cfg(feature = "enable_bench")]
#![feature(test)]

extern crate test;

use fun_collections::FunMap;
use std::collections::BTreeMap;
use std::collections::HashMap;
use test::Bencher;

const N: usize = 1000;

#[bench]
fn time_build_btreemap(b: &mut Bencher) {
    b.iter(|| {
        let mut m = BTreeMap::new();
        for i in 0..N {
            m.insert(i, i);
        }
        m.len()
    });
}

#[bench]
fn time_build_funmap(b: &mut Bencher) {
    b.iter(|| {
        let mut m = FunMap::new();
        for i in 0..N {
            m.insert(i, i);
        }
        m.len()
    });
}

#[bench]
fn time_build_hashmap(b: &mut Bencher) {
    b.iter(|| {
        let mut m = HashMap::new();
        for i in 0..N {
            m.insert(i, i);
        }
        m.len()
    });
}

#[bench]
fn time_clone_btreemap(b: &mut Bencher) {
    let m: BTreeMap<_, _> = (0..N).map(|x| (x, x)).collect();
    b.iter(|| {
        let m = m.clone();
        m.len()
    });
}

#[bench]
fn time_clone_funmap(b: &mut Bencher) {
    let m: FunMap<_, _> = (0..N).map(|x| (x, x)).collect();
    b.iter(|| {
        let m = m.clone();
        m.len()
    });
}

#[bench]
fn time_clone_hashmap(b: &mut Bencher) {
    let m: HashMap<_, _> = (0..N).map(|x| (x, x)).collect();
    b.iter(|| {
        let m = m.clone();
        m.len()
    });
}

#[bench]
fn time_get_btreemap(b: &mut Bencher) {
    let m: BTreeMap<_, _> = (0..N).map(|x| (x, x)).collect();
    b.iter(|| {
        let mut s = 0;
        for i in 0..N {
            s += m.get(&i).unwrap();
        }
        s
    });
}

#[bench]
fn time_get_funmap(b: &mut Bencher) {
    let m: FunMap<_, _> = (0..N).map(|x| (x, x)).collect();
    b.iter(|| {
        let mut s = 0;
        for i in 0..N {
            s += m.get(&i).unwrap();
        }
        s
    });
}

#[bench]
fn time_get_hashmap(b: &mut Bencher) {
    let m: HashMap<_, _> = (0..N).map(|x| (x, x)).collect();
    b.iter(|| {
        let mut s = 0;
        for i in 0..N {
            s += m.get(&i).unwrap();
        }
        s
    });
}

#[bench]
fn time_remove_btreemap(b: &mut Bencher) {
    let m: BTreeMap<_, _> = (0..N).map(|x| (x, x)).collect();
    b.iter(|| {
        let mut s = 0;
        let mut m = m.clone();
        for i in 0..N {
            s += m.remove(&i).unwrap();
        }
        s
    });
}

#[bench]
fn time_remove_funmap(b: &mut Bencher) {
    let m: FunMap<_, _> = (0..N).map(|x| (x, x)).collect();
    b.iter(|| {
        let mut s = 0;
        let mut m = m.clone();
        for i in 0..N {
            s += m.remove(&i).unwrap();
        }
        s
    });
}

#[bench]
fn time_remove_hashmap(b: &mut Bencher) {
    let m: HashMap<_, _> = (0..N).map(|x| (x, x)).collect();
    b.iter(|| {
        let mut s = 0;
        let mut m = m.clone();
        for i in 0..N {
            s += m.remove(&i).unwrap();
        }
        s
    });
}

const V_LEN: usize = 20;

#[bench]
fn time_clone10_update_funmap(b: &mut Bencher) {
    let m: FunMap<_, _> = (0..N).map(|x| (x, format!("{x}"))).collect();
    b.iter(|| {
        let mut s = 0;
        let mut v = vec![m.clone(); V_LEN];
        for i in 0..(N / V_LEN) {
            for j in 1..V_LEN {
                s += v[j].remove(&(i * V_LEN + j)).unwrap().len();
            }
        }
        s
    });
}

#[bench]
fn time_clone10_update_btreemap(b: &mut Bencher) {
    let m: BTreeMap<_, _> = (0..N).map(|x| (x, format!("{x}"))).collect();
    b.iter(|| {
        let mut s = 0;
        let mut v = vec![m.clone(); V_LEN];
        for i in 0..(N / V_LEN) {
            for j in 1..V_LEN {
                s += v[j].remove(&(i * V_LEN + j)).unwrap().len();
            }
        }
        s
    });
}

#[bench]
fn time_clone10_update_hashmap(b: &mut Bencher) {
    let m: HashMap<_, _> = (0..N).map(|x| (x, format!("{x}"))).collect();
    b.iter(|| {
        let mut s = 0;
        let mut v = vec![m.clone(); V_LEN];
        for i in 0..(N / V_LEN) {
            for j in 1..V_LEN {
                s += v[j].remove(&(i * V_LEN + j)).unwrap().len();
            }
        }
        s
    });
}
