import tinygrad
from tinygrad.tensor import Tensor
from tinygrad.helpers import Context

import os
import time
import importlib
importlib.reload(tinygrad)
# importlib.reload(tinygrad)
# importlib.reload(tinygrad)

@Context(DEBUG=5)
def main():
    # os.environ['CPU'] = '1'
    # print(ContextVar._cache)
    s = time.time()
    n = 9
    m = 49
    k = 25
    x = Tensor.rand(n, m, requires_grad=True)
    y = Tensor.rand(m, k, requires_grad=True)
    
    z = x.matmul(y).sum()
    # z.backward()
    print(z.numpy())
    
    # print(x.grad.numpy())  # dz/dx
    # print(y.grad.numpy())  # dz/dy
    
    e = time.time()
    print(e-s)

main()

