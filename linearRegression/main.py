import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

sess = tf.Session()

# placeholders, x and y will hold values when running the graph
x = tf.placeholder(tf.int32 , shape=[3], name='x')
y = tf.placeholder(tf.int32, shape=[3], name='y')

sum_x = tf.reduce_sum(x, name="sum_x")
prod_y = tf.reduce_prod(y, name="prod_y")

final_mean = tf.reduce_mean([sum_x, prod_y], name="final_mean")

final_div = tf.div(sum_x, prod_y, name="final_div")

# feed_dict feeds in the value
print ("sum(x): ", sess.run(sum_x, feed_dict={x: [100, 200, 300]}))
print ("sum(x): ", sess.run(prod_y, feed_dict={y: [1, 2, 3]}))
print ("sum(x) / prod(y): ", sess.run(final_div, feed_dict={
    x: [ 10, 20, 30],
    y: [1, 2, 3]
}))

writer = tf.summary.FileWriter('./linearReg', sess.graph)


writer.close()
sess.close()