; test comment

(assert (== 3 3))
(assert (== -3 -3))
(assert (== (1 2) (1 2)))
(assert (== (1 . 2) (1 . 2)))
(assert (not (== (1 . 2) (1 . 3))))
(assert (!= (1 . 2) (1 . 3)))
(assert (!= (1 1) (1 2)))
(assert (> 10 7))
(assert (< 1 4))
(assert (> 10 -4))
(assert (< -1 40))
(assert (== foo foo))
(assert (!= foo bar))
(assert (> foo bar))
(assert (< bar foo))

; comments?

(assert (is print (quote print))) ; comment ish
(assert (is (type print) fn1))
(assert (is (type 3) int))
(assert (is (type foo) symbol))
(assert (not (is (1 2) (1 2))))
(assert (isinstance 3 int))
(assert (isinstance 3 symbol))
(assert (isinstance foo symbol))
(assert (== (3 . (4 . (5 . nil))) (3 4 5)))
(assert (== (1 2 3) (1 2 3)))
(assert (not (== (1 2 3) (1 3 5))))
(assert (not (!= (1 2) (1 2))))
(assert (!= (1 2) (3 4)))
(assert (!= (1 2) (1 2 3)))
(assert (== (apply cons (3 4)) (3 . 4)))
(assert (== (eval (cons 3 4)) (3 . 4)))

(assert (is (if ()) nil))
(assert (== (if ((== 3 4) (cons 1 2))) nil))
(assert (== (if ((== 4 4) (cons 1 2))) (1 . 2)))

