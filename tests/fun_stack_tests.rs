extern crate quickcheck;
use fun_collections::FunStack;
use quickcheck::quickcheck;

#[test]
fn hello() {
    let s: FunStack<_> = (0..3).collect();
    assert_eq!(s.len(), 3);
}

quickcheck! {
    fn qc_cmp_with_vec(xs: Vec<i32>) -> bool {
        let mut fs = vec![FunStack::new()];
        let mut vs = vec![Vec::new()];

        let mut i = 0;

        for &x in xs.iter() {
            // Use the bottom bits to select a mutation operation.  Use the rest
            // as an operand, if required.
            let op = x % 16;
            let val = x / 16;
            let curr = i % fs.len();
            let has_smtg = !vs[curr].is_empty();
            match op {
                0 if has_smtg => {
                    // creates sharing
                    fs.push(fs[curr].clone());
                    vs.push(vs[curr].clone());
                }

                1 if has_smtg => {
                    let j = val as usize % fs[curr].len();
                    assert_eq!(fs[curr].remove(j), vs[curr].remove(j));
                }

                2 | 3 if has_smtg => {
                    assert_eq!(fs[curr].pop(), vs[curr].pop());
                }

                _ => {
                    fs[curr].push(val);
                    vs[curr].push(val);

                    // NB: doesn't test top_mut() on shared nodes
                    assert_eq!(fs[curr].top_mut(), vs[curr].last_mut());
                }
            }

            assert_eq!(fs[curr].top(), vs[curr].last());

            for j in 0..fs.len() {
                for k in vs[j][0..(2.min(vs[j].len()))].iter() {
                    assert!(fs[j].contains(&k));
                }
                assert_eq!(fs[j].is_empty(), vs[j].is_empty());
                assert_eq!(fs[j].len(), vs[j].len());
                assert!(fs[j].iter().cmp(vs[j].iter().rev()).is_eq());
            }

            i += 1;
        }

        true
    }
}
