import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
sess = tf.Session()

x = tf.constant([100, 200, 300], name="x")
y = tf.constant([1, 2, 3 ], name="y")

sum_x = tf.reduce_sum(x, name="sum_x") # sums up all elements in x
prod_y = tf.reduce_prod(y, name="prod_y")

final_div = tf.div(sum_x, prod_y, name="final_div")
final_mean = tf.reduce_mean([sum_x, prod_y], name="final_mean")

print ("x: ", sess.run(x))
print ("y: ", sess.run(y))
print ("sum_x: ", sess.run(sum_x))
print ("prod_y: ", sess.run(prod_y))
print ("final_div: ", sess.run(final_div))

sess.close()