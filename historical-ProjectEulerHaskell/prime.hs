module Prime (isPrime, primes) where

isPrime x = null $ filter ((==0).(x `mod`)) (takeWhile ((<=x).(^2)) primes)

primes = 2:filter isPrime [3..]
