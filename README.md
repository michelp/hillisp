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
the GPU.  For example, the '+' operator can add two integers together
(this is done on the CUDA "host") or it can add two xectors together
(this is done on the CUDA "device"):

    ? (+ 3 4)  # this happens on the host
    : 7
    ? (+ (fill 3 1000000) (fill 4 1000000))  # this happens on the device
    : [7 7 7 7 7 7 7 7 7 7 7 7 7 7 7  ... 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7]
    ? 

The 'fill' function takes a value and a size and creates a xector of
the specified size filled with that value.  