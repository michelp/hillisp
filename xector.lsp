
(assert (all (== (+ (fill 1 1000000) (fill 1 1000000)) (+ (fill 1 1000000) (fill 1 1000000)))))
(assert (all (== (+ (fill 1 1000000) (fill 1 1000000)) (+ (fill 1 1000000) (fill 1 1000000)))))
(assert (all (== (+ (fill 1 1000000) (fill 1 1000000)) (+ (fill 1 1000000) (fill 1 1000000)))))
(assert (any (== (+ (fill 1 1000000) (fill 1 1000000)) (+ (fill 1 1000000) (fill 1 1000000)))))
(assert (any (== (+ (fill 1 1000000) (fill 1 1000000)) (+ (fill 1 1000000) (fill 1 1000000)))))
(assert (any (== (+ (fill 1 1000000) (fill 1 1000000)) (+ (fill 1 1000000) (fill 1 1000000)))))

(assert 
 (all
  (== 
   (+ (fill 1 1000000) 
      (+ (fill 1 1000000) 
         (+ (fill 1 1000000) 
            (+ (fill 1 1000000)
               (+ (fill 1 1000000)
                  (+ (fill 1 1000000)
                     (+ (fill 1 1000000)
                        (+ (fill 1 1000000)
                           (+ (fill 1 1000000)
                              (fill 1 1000000))))))))))
   (fill 10 1000000)
   )
  )
 )

(assert (all (== (fma (fill 1 1000000) (fill 1 1000000) (fill 1 1000000)) (fma (fill 1 1000000) (fill 1 1000000) (fill 1 1000000)))))
(assert (all (== (fma (fill 1 1000000) (fill 1 1000000) (fill 1 1000000)) (fma (fill 1 1000000) (fill 1 1000000) (fill 1 1000000)))))
(assert (all (== (fma (fill 1 1000000) (fill 1 1000000) (fill 1 1000000)) (fma (fill 1 1000000) (fill 1 1000000) (fill 1 1000000)))))
(assert (any (== (fma (fill 1 1000000) (fill 1 1000000) (fill 1 1000000)) (fma (fill 1 1000000) (fill 1 1000000) (fill 1 1000000)))))
(assert (any (== (fma (fill 1 1000000) (fill 1 1000000) (fill 1 1000000)) (fma (fill 1 1000000) (fill 1 1000000) (fill 1 1000000)))))
(assert (any (== (fma (fill 1 1000000) (fill 1 1000000) (fill 1 1000000)) (fma (fill 1 1000000) (fill 1 1000000) (fill 1 1000000)))))

(println passed)
