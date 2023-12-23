import tinygrad
from tinygrad.tensor import Tensor
from tinygrad.helpers import Context

import os
import time
import importlib
importlib.reload(tinygrad)
# importlib.reload(tinygrad)
# importlib.reload(tinygrad)

@Context(DEBUG=7)
def main():
    # os.environ['CPU'] = '1'
    # print(ContextVar._cache)
    s = time.time()
    n = 3
    m = 7
    k = 5
    x = Tensor.rand(n, m, requires_grad=False)
    # x = Tensor.normal(n, m, requires_grad=False)
    y = Tensor.rand(m, k, requires_grad=False)
    
    z = x.matmul(y).sum()
    # z.backward()
    print(z.numpy())
    
    # print(x.grad.numpy())  # dz/dx
    # print(y.grad.numpy())  # dz/dy
    
    e = time.time()
    print(e-s)

main()
