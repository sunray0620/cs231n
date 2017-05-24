import tensorflow as tf

a = tf.constant(2)
b = tf.constant(3)

c = a + b

with tf.Session() as sess:
	c_val = sess.run(c)
	print (type(c_val))
	print (c_val)
