import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

a = tf.constant(6, name='constant_a')
b = tf.constant(3, name='constant_b')
c = tf.constant(10, name='constant_c')
d = tf.constant(5, name='constant_d')

mul = tf.multiply(a,b, name="mul") # computation nodes
div = tf.div(c,d , name="div")
addn = tf.add_n([mul, div], name="addn") # output of mul and div

# shows the node properties but not the value
print addn
print a

sess = tf.Session() # supervise the execution of tensorflow graphs
sess.run(addn)
sess.run(div)
sess.run(mul)

writer = tf.summary.FileWriter('./m2_example1', sess.graph)

#close the writer and sess
writer.close()
sess.close()

# visualization tool : tensorboard --logdir="m2_example1"