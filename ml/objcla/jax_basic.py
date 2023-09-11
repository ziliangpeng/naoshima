import jax.numpy as jnp

# Create a 2D matrix A with shape (m, n)
A = jnp.array([[1, 2], [3, 4]])

# Create a 3D matrix B with shape (k, m, n)
B = jnp.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])


print(A.shape)
print(B.shape)

# Perform element-wise multiplication
result = A * B
print(result.shape)

print("Result:")
print(result)
