import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
sess = tf.Session()


a = tf.constant(6.5, name='constant_a')
b = tf.constant(3.4, name='constant_b')
c = tf.constant(3.0, name='constant_c')
d = tf.constant(100.2, name='constant_d')

# rank of a:
sess.run(tf.rank(a)) # 0
oneD = tf.constant(["how", "are"])
sess.run(tf.rank(oneD)) # 1


square = tf.square(a, name="square_a")
power = tf.pow(b, c, name="pow_b_c")
sqrt = tf.sqrt(d, name="sqrt_d")

final_sum = tf.add_n([square, power, sqrt], name="final_sum")

print ("Square of a:  ", sess.run(square))
print ("Power of b, c:  ", sess.run(power))
print ("finalsum:  ", sess.run(final_sum))

writer = tf.summary.FileWriter('./m2_example1', sess.graph)
#tensorboard --logdir="m2_example1"




writer.close()
sess.close()