# hillisp

CUDA parallel lisp inspired by [The Connection
Machine](https://en.wikipedia.org/wiki/Connection_Machine).

## xectors

Xectors are CUDA arrays that can be operated on in [CUDA
SIMT](https://en.wikipedia.org/wiki/Single_instruction,_multiple_threads)
using a parallel lisp syntax.  Variants of this syntax were described
in [The Connection Machine (link to book on
Amazon)](http://www.amazon.com/The-Connection-Machine-Artificial-Intelligence/dp/0262580977)
book by [Daniel Hillis](https://en.wikipedia.org/wiki/Danny_Hillis)
and the paper [Connection Machine Lisp: fine-grained parallel symbolic
processing](http://dl.acm.org/citation.cfm?id=319870) by Hillis and
[Guy L. Steele, Jr.](https://en.wikipedia.org/wiki/Guy_L._Steele,_Jr.)