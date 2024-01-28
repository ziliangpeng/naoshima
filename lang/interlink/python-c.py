import ctypes

# Load the C library
my_lib = ctypes.CDLL("./clib.so")

# Define the function prototype
cfunc = my_lib.cfunc
cfunc.restype = ctypes.c_int

cfunc_i = my_lib.cfunc_i
cfunc_i.argtypes = [ctypes.c_int]
cfunc_i.restype = ctypes.c_int

# Call the C function
result = cfunc()
print(f"result from cfunc: {result}")

result = cfunc_i(42)
print(f"result from cfunc_i: {result}")
