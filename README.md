# lispu
Reminiscence of Connection Machine Lisp on CUDA

This project is an attempt to implement the spirt of Connection
Machine Lisp on NVIDIA CUDA GPUs.

# Background

In XXX W. Daniel Hillis described a new kind of "massively parallel"
computer architecture called The Connection Machine (CM).  The CM was
intended to process large amounts of data in parallel, much like the
human brain works.  In XXX he founded the company Thinking Machines to
implement the design.  He co-wrote a paper with Guy Steele and also
wrote the book The Connection Machine. The company went bankrupt in
1996.

As it turns out, at the time no one wanted or needed computers good at
parallel symbolic processing, but rather wanted machines that could do
dense floating point numeric computation in parallel for large scale
problems like weather modeling, genetics, and seismic analysis.  In
this regard the CM was an expensive custom design and a poor performer
compared to large numbers of commercial off-the-shelf (COTS) systems.

In the book and paper, Hillis describes three operators that extend
the Common Lisp language to allow programmers to write expressions
that are executed in parallel on the Connection Machine.  This book
was profoundly influential to me when I was young, so when I first
started working with CUDA, I was immediately struck by the
similarities modern "general purpose" GPU computing had to Hillis'
original design.

lispu is my attempt to implement the spirit of CM Lisp described by
Hillis and Steele using a CUDA capable GPU as the "Connection
Machine".  I have, however, take some modern liberties:

  - lispu is modeled primarily after PicoLisp, not Common Lisp.

  - other language features like types and modules are inspired by
    Python.

  - The original alpha, dot, and beta characters from CM Lisp don't
    exist on most keyboards, so they have been substituded with #, ~
    and &.

