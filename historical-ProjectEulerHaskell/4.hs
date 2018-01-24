
answer = maximum [x * y | x <- [100..999], y <- [100..x], let pd = show (x*y) in pd == reverse pd]

main = print answer
