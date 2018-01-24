answer = [a * b * (1000 - a - b) | a <- [1..1000], b <- [a..1000], let c = 1000 - a - b in a ^ 2 + b ^ 2 == c ^ 2]

main = print $ sum answer
