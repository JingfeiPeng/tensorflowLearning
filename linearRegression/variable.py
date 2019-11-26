import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# y = Wx+b
W = tf.Variable([2.5, 4.0], tf.float32, name="var_w")

x = tf.placeholder(tf.float32, name="x")
b = tf.Variable([5.0, 10.0], tf.float32, name="var_b")

y = W * x + b

# sess initial variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print ("Final result: ", sess.run(y, feed_dict={
        x: [10, 100]
    }))