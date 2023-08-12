import tensorflow as tf

x = tf.Variable(3.14/2)

with tf.GradientTape() as tape:
    y = tf.math.sin(x)

dy_dx = tape.gradient(y, x)

print(y.numpy())
print(dy_dx.numpy())