import sys
import rsa
import aes
from time import time

if len(sys.argv) > 1:
    repeat = map(int, sys.argv[1:])
else:
    repeat = [1]

for r in repeat:
    start_1 = time()
    for i in range(r):
        rsa.run()
    end_1 = time()
    elapsed_1 = end_1 - start_1

    start_2 = time()
    for i in range(r):
        aes.run()
    end_2 = time()
    elapsed_2 = end_2 - start_2

    print("Repeat:", r)
    print('RSA', elapsed_1)
    print('AES', elapsed_2)
