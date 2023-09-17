from jax import jit
import jax.numpy as jnp
								
# define the cube function
def cube(x):
	return x * x * x

# generate data
x = jnp.ones((10000, 10000))
print(x[0])
y = jnp.ones((10000, 1))

z = x @ y
print(x.shape)
print(y.shape)
print(z.shape)

jit_cube = jit(cube)
