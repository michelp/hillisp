# hillisp

[CUDA](https://en.wikipedia.org/wiki/CUDA) parallel lisp inspired by
[Connection
Machines](https://en.wikipedia.org/wiki/Connection_Machine).

hillisp CUDA arrays are called "xectors" and can be operated on in
[CUDA
SIMT](https://en.wikipedia.org/wiki/Single_instruction,_multiple_threads)
fashion using a parallel lisp syntax.  Variants of this syntax were
described in [The Connection Machine (link to book on
Amazon)](http://www.amazon.com/The-Connection-Machine-Artificial-Intelligence/dp/0262580977)
by [Daniel Hillis](https://en.wikipedia.org/wiki/Danny_Hillis) and the
paper [Connection Machine Lisp: fine-grained parallel symbolic
processing](http://dl.acm.org/citation.cfm?id=319870) by Hillis and
[Guy L. Steele, Jr.](https://en.wikipedia.org/wiki/Guy_L._Steele,_Jr.)

## lisp

hillisp is an extremely tiny Lisp implementation written in CUDA C++.
It's primary purpose is to drive the GPU as efficiently as possible.
The language itself is not designed to be especially performant or
featureful, as any computational density your program needs should be
done on the CUDA device and should be appropriate for CUDA workloads.

To that end, the interpreter is very simple, has few "general purpose"
programming features, and is designed to undertake it's interpretation
duties (ie, scheduling, garbage collection) asynchronously while the
GPU is running CUDA kernels.  In this way it attempts to be as "zero
time" as possible.

hillisp is not a general purpose programming language, but a language
for exploring parallel algorithms using extremely powerful, modern GPU
hardware.  It would be inappropraite, for example, to write deployment
scripts with it.

## xectors

A xector is constructed using bracket syntax.  Currently only integer
xectors are supported.  Lisp functions operate on traditional
arguments like numbers, but can also operate on xectors entirely in
the GPU.  For example, the '+' function can add two integers together
(this is done on the CUDA "host") or it can add two xectors together
(this is done on the CUDA "device"):

    ? (+ 3 4)  # this happens on the host
    : 7
    ? (+ (fill 3 1000000) (fill 4 1000000))  # this happens on the device
    : [7 7 7 7 7 7 7 7 7 7 7 7 7 7 7  ... 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7]
    ?

The 'fill' function takes a value and a size and creates a xector of
the specified size filled with that value.  Thus, The second
expression above creates two xectors of one million integers each,
fills them with the values 3 and 7, respectively, then adds them
together, yielding a xector containing one million "10" values.

This is conceptually very similar to the following numpy code:

    >>> 3 + 4
    7
    >>> a = np.empty(1000000)
    >>> b = np.empty(1000000)
    >>> a.fill(3)
    >>> b.fill(4)
    >>> a + b
    array([ 7.,  7.,  7., ...,  7.,  7.,  7.])
    >>>

## CUDA kernels

Internally, '+' and 'fill' cause CUDA kernels to be queued for launch
asynchronously into a CUDA stream.  First two 'fill' kernels, then a
'+' kernel.  Since the 'fill' kernels don't depend on each other, they
can be dispatched in parallel.  While the first two kernels complete
the interpreter does garbage collection, and queues up the next kernel
to run, the '+' kernel which waits until the 'fill' kernels complete
before adding the two xectors together, yielding a third xector
containing the result.

Xectors are allocated using CUDA [Unified
Memory](http://devblogs.nvidia.com/parallelforall/unified-memory-in-cuda-6/).
Pointers to xector data can be accessed by both the host and the
device.  CUDA manages the unified memory so that only the minimal
amount of copying to and from the device to the host is required.

## TODO

  - Multi-device support.

  - Currently only 64 bit integer xectors are supported, but code is
    in place to support all the main CUDA numeric types and nested
    xectors.

  - Data-loading functions to fill xectors from data in files.

  - Implement loadable modules and wrap libraries like cub, cublas,
    cufft, cusparse, etc.  Make CUDA library reuse as trivial as
    possible.

  - Inlining CUDA C kernels for hand-tuning performance.

  - "Xappings": cuda distributed hash tables that can be indexed by a
    key as well as position.

  - Native graph types.

## Alpha, Dot, and Beta

The book and paper cited above expressed parallelism using a Lisp
macro-like parallel expression syntax with three operators, alpha,
dot, and beta.  Implementing these operators in hillisp is certainly a
goal, but I'm not positive it can be done efficiently yet without a
new feature in CUDA called dynamic parallelism, which requires a
greater compute capability than any devices I have available to me at
the moment.  Feel free to send me a dual-maxwell system and I'll get
it done. :)
