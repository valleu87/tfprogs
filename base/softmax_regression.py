import numpy as np
import pandas as pd
import tensorflow as tf

COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']
ONE_HOT_CODE = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

train_path = "iris_data/iris_training.csv"
test_path = "iris_data/iris_test.csv"

train_df = pd.read_csv(train_path, names=COLUMN_NAMES, header=0)
train_n = train_df.shape[0]
train_x = np.array(train_df[['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']])
train_y = np.array(map(lambda x: ONE_HOT_CODE[x], train_df['Species']))
print('train_x.shape = {}, train_y.shape = {}'.format(train_x.shape, train_y.shape))

test_df = pd.read_csv(test_path, names=COLUMN_NAMES, header=0)
test_n = train_df.shape[0]
test_x = np.array(test_df[['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']])
test_y = np.array(map(lambda x: ONE_HOT_CODE[x], test_df['Species']))
print('test_x.shape = {}, test_y.shape = {}'.format(test_x.shape, test_y.shape))

x_ = tf.placeholder(tf.float32, [None, 4])
y_ = tf.placeholder(tf.float32, [None, 3])
w = tf.Variable(tf.ones([4, 3]))
b = tf.Variable(tf.zeros([3]))
y = tf.nn.softmax(tf.add(tf.matmul(x_, w), b))
#cross_entropy = tf.reduce_mean(tf.negative(
#    tf.reduce_sum(tf.multiple(y_, tf.log(y)), axis=1)))
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), axis=1))
learning_rate = 0.1
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_steps = 1000
batch_size = 10
with tf.Session() as sess:
    #sess.run(tf.global_variables_initializer())
    tf.global_variables_initializer().run()
    # train
    for i in range(train_steps):
        batch_i = np.random.randint(train_n, size=batch_size)
        batch_x = train_x[batch_i]
        batch_y = train_y[batch_i]
        #sess.run(train_step, feed_dict={x_:batch_x, y_:batch_y})
        train_step.run({x_:batch_x, y_:batch_y})  # op can run
        if i % 100 == 0:
            #acc = sess.run(accuracy, feed_dict={x_:test_x, y_:test_y})
            acc = accuracy.eval({x_:test_x, y_:test_y})  # tensor can eval
            print("epoch = {}, accuracy = {}".format(i, acc))
    
    # eval
    #acc = sess.run(accuracy, feed_dict={x_:test_x, y_:test_y})
    acc = accuracy.eval({x_:test_x, y_:test_y})    # tensor can eval
    print("accuracy_final = {}".format(acc))


