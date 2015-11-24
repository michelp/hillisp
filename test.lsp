# test comment

(assert (== 3 3))
(assert (== -3 -3))
(assert (> 10 7))
(assert (< 1 4))
(assert (> 10 -4))
(assert (< -1 40))

# comments?

(assert (is print (quote print)))  # comment ish
(assert (is (type print) fn1))
(assert (is (type 3) int))
(assert (is (type foo) symbol))
(assert (isinstance 3 int))
(assert (isinstance 3 symbol))
(assert (isinstance foo symbol))
