# Ownership and Lifetime

Everything is simple until there are too many things to handle at once. So we
scope. A scope is a namespace (e.g., a function). We can also use multi-level
scopes. Scopes break the namespace into smaller ones to help us manage the
cognitive context.

It sounds all good, but causes a new problem: what do we do when an object
outlives the scope in which it was created?

That is why we need ownership. When an object is owned, it stays alive. When
its last owner releases it, it gets destroyed.

## Lack of Ownership

When nobody owns an object, nothing is responsible for destroying it: it may
leak or get destroyed on a guess. Let us see how ownership works by observing
the problem introduced by the lack of ownership.

A [raw pointer](https://en.cppreference.com/cpp/language/pointer) records
an address and nothing else, so it does not carry any information about
ownership. It does not say whether the pointed-to object is alive, and it
does not say who is responsible for destroying it. Code written against raw
pointers therefore encodes that responsibility nowhere but in the reader's
memory:

```cpp
Data * data = alloc_function(); // allocates and returns
op_function(data); // may or may not delete, depending on the code inside
delete data; // correct or a double free; the caller here cannot tell
```

Because the raw pointer carries no ownership information, there is no way for
the third line to know whether `data` was already deleted in `op_function()`.

```{note}
solvcon follows a convention for raw pointers: when a function receives a raw
pointer that is not null, someone else owns the object. If it is null, there
is no object.

A [reference](https://en.cppreference.com/cpp/language/reference)
(`Type &`) is a similar case, but it cannot be null: it must refer to an
existing object owned by someone else.
```

## Unique Pointer for Simple Ownership

Raw pointers and references are not sufficient to manage [object
lifetime](https://en.cppreference.com/cpp/language/lifetime).

If we consider a simple scenario that allows at most one owner for an object,
[`std::unique_ptr`](https://en.cppreference.com/cpp/memory/unique_ptr) can be
used to manage the ownership. The smart pointer makes it straight-forward to
manage the object lifetime using
[RAII](https://en.cppreference.com/cpp/language/raii) (resource acquisition is
initialization), tying the lifetime of a resource to the lifetime of an object.

The overhead of
[`std::unique_ptr`](https://en.cppreference.com/cpp/memory/unique_ptr) is
usually zero. If an object is not shared among multiple owners, or you only
want to make sure it gets release at the end of a scope, `std::unique_ptr` is
usually the right tool.

C++ carries such guarantees with
[RAII](https://en.cppreference.com/cpp/language/raii) (resource acquisition
is initialization), tying the lifetime of a resource to the lifetime of an
object, and spells the ownership out with smart pointers:
[`std::unique_ptr`](https://en.cppreference.com/cpp/memory/unique_ptr)
where there is exactly one owner,
[`std::shared_ptr`](https://en.cppreference.com/cpp/memory/shared_ptr)
where there are several.

## Shared Ownership

If an object should be shared among multiple owners,
[`std::shared_ptr`](https://en.cppreference.com/cpp/memory/shared_ptr) should
be used. It offers shared ownership with a reference count. The object is
destroyed when the last owner releases it.

Conceptually, shared ownership is a superset of unique ownership, but we must
not be fooled into using a `std::shared_ptr` when the code does not require
shared ownership. It costs much more overhead than a raw pointer or a unique
pointer.

To make shared ownership possible, a `std::shared_ptr` needs to actually manage
two objects: the referenced object and a control block for the reference
counts. That takes two memory allocations instead of one. The additional memory
allocation is a cost, and sometimes a significant one.

Always think twice before using a shared pointer. If you create a lot of small
objects, it is not the right scenario for shared pointers, because each object
will need to pay for the overhead. Shared pointers should be used to manage
objects that are large and have long lifetimes.

A memory buffer is in exactly that position.

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
