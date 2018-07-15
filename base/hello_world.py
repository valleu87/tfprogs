import tensorflow as tf
sess = tf.Session()
greeting = tf.constant("Hello world!")
print(sess.run(greeting))