fib = 1:1:(map (\(x,y) -> x + y) (zip fib (tail fib))) where s (x,y) = x + y

bound = 4000000

answer = sum([x | x <- takeWhile (<bound) fib, x `mod` 2 == 0])

main = print answer
