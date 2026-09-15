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

## Exclusively Use Shared Pointers

There are a lot of caveats in using a shared pointer. The first one is the
issue with the `std::shared_ptr` constructor.

The `std::shared_ptr` constructor accepts a raw pointer and takes ownership of
the object it points to. But the raw pointer stays with the caller, and nothing
stops it from being given to another shared pointer.

(shared-pointer-double-free)=
### Shared Pointer Double Free

To see the problem, first new an object (`Data`) and hand the pointer to the
constructor of the shared pointer:

```cpp
Data * raw_pointer = new Data;
std::shared_ptr<Data> sptr1(raw_pointer);
std::cout << "sptr1.use_count(): " << sptr1.use_count() << std::endl;
```

Then the object is held in the shared pointer:

```text
Data @0x5e9b5b1012b0 is constructed
sptr1.use_count(): 1
```

Then we give the same raw pointer to a second shared pointer:

```cpp
std::shared_ptr<Data> sptr2(raw_pointer);
std::cout << "sptr2.use_count(): " << sptr2.use_count() << std::endl;
```

The reference count is 1 rather than 2, because the second shared pointer
creates its own reference counter and does not know about the first one:

```text
sptr2.use_count(): 1
```

Release the first shared pointer:

```cpp
sptr1.reset();
std::cout << "sptr1.use_count() after sptr1.reset(): " << sptr1.use_count() << std::endl;
```

Its reference count drops to zero, so the `Data` object is destructed, although
`sptr2` still points to it:

```text
Data @0x5e9b5b1012b0 is destructed
sptr1.use_count() after sptr1.reset(): 0
```

Now release the second shared pointer:

```cpp
std::cout << "sptr2.use_count(): " << sptr2.use_count() << std::endl;
sptr2.reset();  // This line crashes with double free.
// This line never gets reached since the above line causes double free and
// crash.
std::cout << "sptr2.use_count() after sptr2.reset(): "
          << sptr2.use_count() << std::endl;
```

It can never reach the last line, since releasing the pointer destructs the
object again and results in double free:

```text
sptr2.use_count(): 1
Data @0x5e9b5b1012b0 is destructed
double free or corruption (!prev)
Aborted (core dumped)
```

To avoid the problem, we do not want anyone to directly call the `Data`
constructor:

```cpp
// We want to forbid it.
Data * raw_pointer = new Data;
```

Then it will be reasonably hard to write code that constructs two shared
pointers from the same raw pointer.

### Wrong Approach: Private Constructors

You could think that private constructors may solve the issue. Here I explain
how private constructors work with shared pointers, and then why it is not the
solution.

Here is a class with a private constructor for shared pointers:

```cpp
class Data
{
private:
    // A private constructor.
    Data() {}
public:
    static std::shared_ptr<Data> make()
    {
        std::shared_ptr<Data> ret(new Data);
        return ret;
    }
};
```

By making the constructor private, it is only possible to construct the `Data`
object from a member function. Then we implement the static member function
`make()` to call the constructor and return the shared pointer:

```cpp
class Data
{
public:
    static std::shared_ptr<Data> make()
    {
        std::shared_ptr<Data> ret(new Data);
        return ret;
    }
};
```

The static member function `make()` is a factory function to create
`std::shared_ptr<Data>`:

```cpp
std::shared_ptr<Data> data = Data::make();
```

Because the constructor of `Data` is private, the following code fails to
compile:

```cpp
std::shared_ptr<Data> data(new Data);
```

You will see messages like:

```text
private_ctor.cpp:18:36: error: 'Data::Data()' is private within this context
   18 |     std::shared_ptr<Data> data(new Data);
      |                                    ^~~~
private_ctor.cpp:7:5: note: declared private here
    7 |     Data() {}
      |     ^~~~
```

It works nicely, except it prevents us from using the function template
`std::make_shared<Data>()`. The inability to use the function template is why
using private constructors is a wrong approach.

Try writing this line:

```cpp
std::shared_ptr<Data> ret = std::make_shared<Data>();
```

The error messages will be like (in g++):

```text
bits/stl_construct.h: In instantiation of 'void std::_Construct(Data*)':
bits/alloc_traits.h:661       required from 'std::allocator_traits::construct'
bits/shared_ptr_base.h:604    required from 'std::_Sp_counted_ptr_inplace'
bits/shared_ptr_base.h:971    required from 'std::__shared_count'
bits/shared_ptr_base.h:1712   required from 'std::__shared_ptr'
bits/shared_ptr.h:464         required from 'std::shared_ptr'
bits/shared_ptr.h:1009        required from 'std::make_shared<Data>()'
private_make.cpp:18           required from here
bits/stl_construct.h:119: error: 'Data::Data()' is private within this context
  119 |       ::new((void*)__p) _Tp(std::forward<_Args>(__args)...);
      |       ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
private_make.cpp:7: note: declared private here
    7 |     Data() {}
      |     ^~~~
```

It is because the function template `std::make_shared` is not inside class
`Data`, and cannot access the private constructor! Using `friend` sometimes
works, but it depends on how `std::make_shared` is implemented. The template
does a lot of things behind the scenes. Simply making friends with that
function template may or may not work.

The function template `std::make_shared` is desired because it allocates the
`Data` object along with its reference counter. The reference counter of a
shared pointer must be dynamically allocated because it is shared among all
shared pointer instances. The `Data` object also needs to be dynamically
allocated. Without `std::make_shared`, two dynamic allocations will be used
instead of one, and it is a lot of overhead when we have many `Data` objects.

### Good Approach: Passkey Pattern

A sound approach is to use the passkey pattern:

```cpp
class Data
{
private:
    class ctor_passkey
    {
        ctor_passkey() {}
        friend class Data;
    };
public:
    static std::shared_ptr<Data> make()
    {
        std::shared_ptr<Data> ret = std::make_shared<Data>(ctor_passkey());
        return ret;
    }
    Data() = delete;
    Data(ctor_passkey const &) {}
    // TODO: Copyability and moveability should be considered, but we leave
    // them for now.
};
```

The default constructor should be deleted to prevent the following from
working:

```cpp
data = std::shared_ptr<Data>(new Data);
```

The compiler error messages:

```text
01_fully.cpp:91:38: error: call to deleted constructor of 'Data'
    data = std::shared_ptr<Data>(new Data);
                                     ^
01_fully.cpp:22:5: note: 'Data' has been explicitly marked deleted here
    Data() = delete;
    ^
```

The use of the function template `std::make_shared`:

```cpp
data = std::make_shared<Data>();
```

is forbidden for the same reason:

```text
bits/stl_construct.h: In instantiation of 'void std::_Construct(Data*)':
bits/alloc_traits.h:661       required from 'std::allocator_traits::construct'
bits/shared_ptr_base.h:604    required from 'std::_Sp_counted_ptr_inplace'
bits/shared_ptr_base.h:971    required from 'std::__shared_count'
bits/shared_ptr_base.h:1712   required from 'std::__shared_ptr'
bits/shared_ptr.h:464         required from 'std::shared_ptr'
bits/shared_ptr.h:1009        required from 'std::make_shared<Data>()'
01_fully.cpp:95               required from here
bits/stl_construct.h:119: error: use of deleted function 'Data::Data()'
  119 |       ::new((void*)__p) _Tp(std::forward<_Args>(__args)...);
      |       ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
01_fully.cpp:22: note: declared here
   22 |     Data() = delete;
      |     ^~~~
```

Then you only have the constructor that takes the passkey argument:
`Data(ctor_passkey const &)`. The internal passkey class has a private
constructor that can be accessed from the `Data` class by the added friendship:

```cpp
    class ctor_passkey
    {
        ctor_passkey() {}
        friend class Data;
    };
```

As such, the factory function `Data::make()` can call the passkeyed constructor
through the function template `std::make_shared()`:

```cpp
    static std::shared_ptr<Data> make()
    {
        std::shared_ptr<Data> ret = std::make_shared<Data>(ctor_passkey());
        return ret;
    }
```

This completely manages the object by using shared pointers. You always need to
construct `Data` by calling the factory function, and always get a shared
pointer `std::shared_ptr<Data>` rather than a raw pointer `Data *`.

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
