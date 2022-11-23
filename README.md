This crate will provide Rust collections that are 'functional' (or applicative
or shared) under the hood.  Clone operations will be very cheap because the
clone will share the memory represenation with the originator. Updates to a
collection will minimize the allocation of new memory and continue to share as
much memory as possible with other clones of the collection.  We will use Chris
Okasaki's book "Purely Functional Data Structures" as a reference.
