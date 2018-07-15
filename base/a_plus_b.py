import tensorflow as tf
sess = tf.Session()
a = tf.placeholder(tf.int32)
b = tf.placeholder(tf.int32)
c = tf.add(a, b)
res = sess.run(c, feed_dict={a:[3, 10], b:[4, 20]})
print(res)   # np.array
sess.close()
