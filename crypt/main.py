import rsa
import aes
from timeit import default_timer as timer

start_1 = timer()
for i in range(100):
    rsa.run()
end_1 = timer()
elapsed_1 = end_1 - start_1

start_2 = timer()
for i in range(100):
    aes.run()
end_2 = timer()
elapsed_2 = end_2 - start_2


print('RSA', elapsed_1)
print('AES', elapsed_2)
