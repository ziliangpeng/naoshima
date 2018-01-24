import Prime

answer = sum $ takeWhile (<2000000) primes

main = print answer
