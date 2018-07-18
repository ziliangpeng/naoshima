import sys
import rsa
import aes
from timeit import default_timer as timer

if len(sys.argv) > 1:
    repeat = int(sys.argv[1])
else:
    repeat = 1

start_1 = timer()
for i in range(repeat):
    rsa.run()
end_1 = timer()
elapsed_1 = end_1 - start_1

start_2 = timer()
for i in range(repeat):
    aes.run()
end_2 = timer()
elapsed_2 = end_2 - start_2

print("Repeat:", repeat)
print('RSA', elapsed_1)
print('AES', elapsed_2)
