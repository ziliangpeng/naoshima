import gflags
import numpy as np
import sys

gflags.DEFINE_integer("A", 42, "A in y = Ax + B")
gflags.DEFINE_integer("B", 27, "B in y = Ax + B")

argv = gflags.FLAGS(sys.argv)


def generate_data():
    FLAGS = gflags.FLAGS

    # Generate some random data
    A = FLAGS.A
    B = FLAGS.B
    print(f"y = {A}x + {B}")
    np.random.seed(42)
    X = np.random.randn(1000, 1)
    y = A * X + B + np.random.randn(1000, 1) / 1000
    return X, y
