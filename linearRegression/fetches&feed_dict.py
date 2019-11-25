import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

W = tf.constant([10, 100], name="const_W")

x = tf.placeholder(tf.int32, name="x")
b = tf.placeholder(tf.int32, name="b")

# X has to be the same rank and shape as W
Wx = tf.multiply(W, x, name="Wx")
y = tf.add(Wx, b, name="y")

y_ = tf.subtract(x, b, name="y_")

# put all codes that require sess in this block
with tf.Session() as sess:
    # Wx is fetches, feed_dict is input to placeholder
    print ("intermediate result: Wx = ", sess.run(Wx, feed_dict={x:[3, 33]}))
    print ("Final result: Wx + b = ", sess.run(y, feed_dict={
        x: [5, 50],
        b: [7, 9]
        }))
    # directly specify the value for Wx, without doing the calculation for Wx
    print ("Intermediate specified: Wx + b = ", 
        sess.run(fetches = y, feed_dict = {Wx:[100,1000], b:[7,9]}))
    # gets y is [  57, 5009] y_ is [-2, 41]
    print ("Two result: [Wx+b, x - b]",
        sess.run(fetches=[y, y_], feed_dict={
            x: [5, 50],
            b: [7, 9]
        }))
writer = tf.summary.FileWriter('./linearReg', sess.graph)
writer.close()
