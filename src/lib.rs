//! # "Functional" collections that provide memory-efficient cloning
//!
//! `fun-collections` is a set of "functional" collections.  The collections use
//! persistent data structures, which means that a clone `s.clone()` shares its
//! internal representation with `s`.  The representations of a collection and
//! its clones gradually diverge as they are updated.  `fun-collections`
//! provides a subset of the functionality found in the `im` crate, which is way
//! more mature.  You probably should use the im crate instead of this one.

mod fun_stack;
pub use fun_stack::*;
