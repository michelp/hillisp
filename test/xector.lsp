(set cnt 2000000)

(for i 0 10
     (assert (all (== (+= (fill 1 cnt) (fill 1 cnt)) (+= (fill 1 cnt) (fill 1 cnt)))))
)

(for i 0 10
     (assert (any (== (+= (fill 1 cnt) (fill 1 cnt)) (+= (fill 1 cnt) (fill 1 cnt)))))
)

(for i 0 10
     (assert (all (== (-= (fill 1 cnt) (fill 1 cnt)) (-= (fill 1 cnt) (fill 1 cnt)))))
)

(for i 0 10
     (assert (any (== (-= (fill 1 cnt) (fill 1 cnt)) (-= (fill 1 cnt) (fill 1 cnt)))))
)

(for i 0 10
     (assert (all (== (*= (fill 1 cnt) (fill 1 cnt)) (*= (fill 1 cnt) (fill 1 cnt)))))
)

(for i 0 10
     (assert (any (== (*= (fill 1 cnt) (fill 1 cnt)) (*= (fill 1 cnt) (fill 1 cnt)))))
)

(for i 0 10
     (assert (all (== (/= (fill 1 cnt) (fill 1 cnt)) (/= (fill 1 cnt) (fill 1 cnt)))))
)

(for i 0 10
     (assert (any (== (/= (fill 1 cnt) (fill 1 cnt)) (/= (fill 1 cnt) (fill 1 cnt)))))
)

(for i 0 10
     (assert (all (== (fma= (fill 1 cnt) (fill 1 cnt) (fill 1 cnt)) (fma= (fill 1 cnt) (fill 1 cnt) (fill 1 cnt)))))
)

(for i 0 10
     (assert (any (== (fma= (fill 1 cnt) (fill 1 cnt) (fill 1 cnt)) (fma= (fill 1 cnt) (fill 1 cnt) (fill 1 cnt)))))
)

(def ff (x) 
  (if x 
    (
     (+= (fill (car x) cnt) (ff (cdr x)))
     )
    ((fill 0 cnt))
  )
)

(assert (all (== (ff (range 0 10 1)) (fill 45 cnt))))
