from micrograd.engine import Value
import math

x = Value(2.0)
# y = x ** 3 + 5 * x + 3

y = math.sin(x)

print(f'{y.data:.4f}')

y.backward()
print(f'{x.grad:.4f}')