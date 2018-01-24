-- What is the largest prime factor of the number 600851475143 ?

import Prime

-- program below -- 

n = 600851475143

primeFactors1 = filter ((==0).(n `mod`)) (takeWhile ((<n).(^2)) primes)

allPrimeFactors = filter isPrime (primeFactors1 ++ map (n `div`) primeFactors1)

answer = maximum allPrimeFactors

main = print answer
