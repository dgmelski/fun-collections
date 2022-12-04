This crate provides rust collections that are 'functional' (or persistent or
applicative or shared or immutable) under the hood.  Clone operations are cheap
because a clone of a collection shares its represenation with the original
collection, at first. As the clone or the original are updated, their internal
representations will diverge, although I attempt to retain sharing and minimize
new allocations.

The "immutablibity" is in the internal data structures.  I attempted to match the interface to the API of the standard rust collections, including by providing mutating operations.  When one of our collections updates, it clones its internal state as necessary, namely when that state is shared with other collections.  Thus, the state of the collection prior to an update may persist (in another collection), but the updated collection may use retain only part or none of that original state.

While persistent data structures are the norm in functional languages, their use
in a language like rust is likely to be niche.  For most uses, they will be much
slower than the standard libraries.  Their best use case is when you need to
keep many clones of the collections.  For example, this may be the case when
building a symbolic execution engine that explores many different execution
paths in a breadth-first manner.  Each explored path needs a *symbolic memory*.
By building the symbolic memory using persistent maps, the symbolic memories for
paths with a common prefix may share much of their representation.  This can
result in dramatic savings it space and time.
