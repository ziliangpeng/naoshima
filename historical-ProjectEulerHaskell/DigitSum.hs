module DigitSum (digitSum) where

digitSum :: Integer -> Integer
digitSum x = sum $ map ((+0).read.(\y->[y])) (show x)
