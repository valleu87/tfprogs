import numpy as np
import tensorflow as tf

n_train = 800
x_train = np.random.randn(n_train, 1)
y_train = x_train * 3.0 + np.random.normal(4.0, 0.1, (n_train, 1)) 

n_test = 200
x_test = np.random.randn(n_test, 1)
y_test = x_test * 3.0 + np.random.normal(4.0, 0.1, (n_test, 1))

weight = tf.Variable(1.0)
bias = tf.Variable(0.0)

x_ = tf.placeholder(tf.float32, [None, 1])
y_ = tf.placeholder(tf.float32, [None, 1])

# y = weight * x_ + bias
y = tf.add(tf.multiply(weight, x_), bias)

loss = tf.reduce_mean(tf.square(tf.subtract(y_, y)))
learning_rate = 0.01
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

epochs = 100000
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	# train
	for i in range(epochs):
		x0 = x_train[i % n_train].reshape(1, -1)
		y0 = y_train[i % n_train].reshape(1, -1)
		sess.run(train_step, feed_dict={x_:x0, y_:y0})

		if i % 1000 == 0:
			w = weight.eval(sess) # w = sess.run(weight)
			b = bias.eval(sess) # b = sess.run(bias)
			print('epoch = {}, weight = {}, bias = {}'.format(i, w,b))

	weight_final = sess.run(weight)
	bias_final = sess.run(bias)
	print('weight_final = {}'.format(weight_final))
	print('bias_final = {}'.format(bias_final))	

	# eval
	loss_train = sess.run(loss, feed_dict={x_:x_train, y_:y_train})
	loss_test = sess.run(loss, feed_dict={x_:x_test, y_:y_test})
	print('loss_train = {}'.format(loss_train))
	print('loss_test = {}'.format(loss_test))



