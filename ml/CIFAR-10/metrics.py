import tensorflow as tf
import os
import datetime


# Create the TensorBoard callback
def make_tb(name):
    log_dir = os.path.join(
        "logs", name + "-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    return tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1, update_freq="epoch"
    )
