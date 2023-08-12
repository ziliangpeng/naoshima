from autograd import grad
import autograd.numpy as np

def f(x):
    return np.sin(x)

df = grad(f)

x = 1.0
x = 3.14/2
y = f(x)
dy_dx = df(x)

print(f'y = {y:.4f}')
print(f'dy/dx = {dy_dx:.4f}')