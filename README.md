# hillisp

CUDA parallel lisp inspired by [The Connection
Machine](https://en.wikipedia.org/wiki/Connection_Machine).

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
programming features and is designed to undertake it's interpretation
duties (ie, scheduling, garbage collection) asynchronously while the
GPU is running CUDA kernels.  In this way it attempts to be as "zero
time" as possible.

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

Internally, '+' and 'fill' cause CUDA kernels to be launched
asynchronously into a CUDA stream.  First two 'fill' kernels then a
'+' kernel.  While the kernels are running asynchronously the
interpreter advances forward to run evaluate the next expression.

## TODO

  - Currently only 64 bit integer xectors are supported, but code is
    in place to support all the main CUDA numeric types.

  - Data-loading functions to fill xectors from data in files.

  - Implement loadable modules and wrap libraries like cub, cublas,
    cufft, etc.  Make CUDA library reuse as trivial as possible.

  - "Xappings": cuda distributed hash tables that can be indexed by a
    key as well as position.

## Alpha, Dot, and Beta

The book and paper cited above expressed parallelism using a Lisp
macro-like parallel expression syntax with three operators, alpha,
dot, and beta.  Implementing these operators in hillisp is a goal, but
I'm not certain it can be done efficiently yet without a new feature
in CUDA called dynamic parallelism, which requires a greater compute
capability than any devices I have available to me at the moment.
Feel free to send me a dual-maxwell system and I'll get it done. :)