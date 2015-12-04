
(assert (all (== (+ (fill 1 30000) (fill 1 30000)) (+ (fill 1 30000) (fill 1 30000)))))
(assert (all (== (+ (fill 1 30000) (fill 1 30000)) (+ (fill 1 30000) (fill 1 30000)))))
(assert (all (== (+ (fill 1 30000) (fill 1 30000)) (+ (fill 1 30000) (fill 1 30000)))))
(assert (all (== (+ (fill 1 30000) (fill 1 30000)) (+ (fill 1 30000) (fill 1 30000)))))
(assert (all (== (+ (fill 1 30000) (fill 1 30000)) (+ (fill 1 30000) (fill 1 30000)))))
(assert (all (== (+ (fill 1 30000) (fill 1 30000)) (+ (fill 1 30000) (fill 1 30000)))))
(assert (all (== (+ (fill 1 30000) (fill 1 30000)) (+ (fill 1 30000) (fill 1 30000)))))
(assert (all (== (+ (fill 1 30000) (fill 1 30000)) (+ (fill 1 30000) (fill 1 30000)))))
(assert (all (== (+ (fill 1 30000) (fill 1 30000)) (+ (fill 1 30000) (fill 1 30000)))))
(assert (all (== (+ (fill 1 30000) (fill 1 30000)) (+ (fill 1 30000) (fill 1 30000)))))
(assert (all (== (+ (fill 1 30000) (fill 1 30000)) (+ (fill 1 30000) (fill 1 30000)))))
(assert (all (== (+ (fill 1 30000) (fill 1 30000)) (+ (fill 1 30000) (fill 1 30000)))))

(assert 
 (all
  (== 
   (+ (fill 1 30000) 
      (+ (fill 1 30000) 
         (+ (fill 1 30000) 
            (+ (fill 1 30000)
               (+ (fill 1 30000)
                  (+ (fill 1 30000)
                     (+ (fill 1 30000)
                        (+ (fill 1 30000)
                           (+ (fill 1 30000)
                              (fill 1 30000))))))))))
   (fill 10 30000)
   )
  )
 )

(println passed)
