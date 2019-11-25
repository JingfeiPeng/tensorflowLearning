import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

sess = tf.Session()

zeroD = np.array(30, dtype=np.int32)
sess.run(tf.rank(zeroD))
sess.run(tf.shape(zeroD))

oneD = np.array([5.6,6.3, 8.9, 9.0], dtype = np.float32)
sess.run(tf.shape(oneD)) # array([4], dtype=int32)

sess.close()