try: import pycuda
except ImportError: pass

import time
import numpy as np


def cuda2(a): # matmul
    try:
        import pycuda.autoinit
        import pycuda.driver as cuda
        import pycuda.gpuarray as gpuarray
        from pycuda.compiler import SourceModule
        import pycuda.driver as cuda
        import pycuda.autoinit  # Automatically initializes CUDA

        # Choose the device
        device = cuda.Device(0)  # Assuming you want to use the first device

        # Get device properties
        device_properties = device.get_attributes()
        print(device_properties)
        # Access the maximum threads per block
        max_threads_per_block = device_properties[cuda.device_attribute.MAX_THREADS_PER_BLOCK]

        print("Maximum threads per block:", max_threads_per_block)
    except ImportError:
        # print("pycuda not installed")
        return None

    # CUDA kernel for element-wise multiplication
    kernel_code = """
    __global__ void ElementWiseMul(float *a, float *c, int size, int edge) 
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        int idy = threadIdx.y + blockIdx.y * blockDim.y;
        if(idx < edge && idy < edge) {
            float acc = 0.;
            for(int i = 0; i < edge; i++){
                acc = acc + a[idx * edge + i] * a[i * edge + idy];
            }
            c[idx * edge + idy] = acc;
        }
    }
    """

    # Initialize matrices
    # size = 1000 * 1000
    assert a.shape[0] == a.shape[1]
    edge = a.shape[0]
    size = a.shape[0] * a.shape[1]
    a = a # np.random.rand(size).astype(np.float32)
    a1d = a.reshape(-1)
    c = np.empty_like(a)

    # Allocate memory on GPU
    a_gpu = cuda.mem_alloc(a.nbytes)
    c_gpu = cuda.mem_alloc(c.nbytes)

    # Transfer data to GPU
    cuda.memcpy_htod(a_gpu, a)
    # cuda.memcpy_htod(b_gpu, b)

    # Compile and get the kernel function
    mod = SourceModule(kernel_code)
    elementwise_mul = mod.get_function("ElementWiseMul")

    # Launch the kernel
    block_size = 32
    grid_size = int(np.ceil(edge / block_size))
    elementwise_mul(a_gpu, c_gpu, np.int32(size), np.int32(edge), block=(block_size, block_size, 1), grid=(grid_size, grid_size))

    # Copy the result back to CPU
    cuda.memcpy_dtoh(c, c_gpu)

    # Cleanup
    a_gpu.free()
    # b_gpu.free()
    c_gpu.free()

    # c now contains the result
    return c.reshape(a.shape)

def cuda(a):
    try:
        import pycuda.autoinit
        import pycuda.driver as cuda
        import pycuda.gpuarray as gpuarray
        from pycuda.compiler import SourceModule
    except ImportError:
        # print("pycuda not installed")
        return None

    # CUDA kernel for element-wise multiplication
    kernel_code = """
    __global__ void ElementWiseMul(float *a, float *c, int size)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < size)
            c[idx] = a[idx] * a[idx];
    }
    """

    # Initialize matrices
    # size = 1000 * 1000
    size = a.shape[0] * a.shape[1]
    a = a # np.random.rand(size).astype(np.float32)
    a1d = a.reshape(-1)
    c = np.empty_like(a)

    # Allocate memory on GPU
    a_gpu = cuda.mem_alloc(a.nbytes)
    c_gpu = cuda.mem_alloc(c.nbytes)

    # Transfer data to GPU
    cuda.memcpy_htod(a_gpu, a)
    # cuda.memcpy_htod(b_gpu, b)

    # Compile and get the kernel function
    mod = SourceModule(kernel_code)
    elementwise_mul = mod.get_function("ElementWiseMul")

    # Launch the kernel
    block_size = 256
    grid_size = int(np.ceil(size / block_size))
    elementwise_mul(a_gpu, c_gpu, np.int32(size), block=(block_size, 1, 1), grid=(grid_size, 1))

    # Copy the result back to CPU
    cuda.memcpy_dtoh(c, c_gpu)

    # Cleanup
    a_gpu.free()
    # b_gpu.free()
    c_gpu.free()

    # c now contains the result
    return c.reshape(a.shape)


def mul(a):
    b = np.zeros_like(a)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            for k in range(a.shape[1]):
                b[i][k] += a[i][j] * a[j][k]
    return b


def main():
    N = 1000
    for i in range(9):
        N = 2 ** i
        # a = np.ndarray([1000, 1000])
        a = np.random.rand(N, N).astype(np.float32)
        if i == 0: print(a.dtype)
        # print(a)
        start = time.time()
        # ret1 = cuda(a)
        ret1 = cuda2(a)
        end = time.time()
        print(f"N: {N}, cuda time: {end - start:.4f}")

        start = time.time()
        # ret2 = a * a
        ret2 = mul(a)
        end = time.time()
        print(f"N: {N}, numpy time: {end - start:.4f}")

        if ret1 is not None:
            close = np.testing.assert_allclose(ret1, ret2, rtol=1e-5, atol=1e-5)
        else:
            close = False
        print(f"close: {close}")


main()