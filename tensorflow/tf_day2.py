import tensorflow as tf

z = tf.Variable(tf.zeros_like([1,1,2,3]))
# z = tf.zeros_like([1,1,2,3])
op1 = z.assign([100,100,100,100])
op2 = z.assign_add()
op3 = z.assign_sub()
with tf.Session() as sess:
	sess.run(z.initializer)
	sess.run(op1)
	print(z.eval())
