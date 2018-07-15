import tensorflow as tf

with tf.Session() as sess:
    q = tf.FIFOQueue(1000, 'float32')
    counter = tf.Variable(0.0)
    inc = tf.assign_add(counter, tf.constant(1.0))
    enq = q.enqueue(counter)

    sess.run(tf.global_variables_initializer())

    # QueueRunner: 3 threads
    qr = tf.train.QueueRunner(q, enqueue_ops=[inc, enq] * 3)
    # Coordinator
    coord = tf.train.Coordinator()
    enq_threads = qr.create_threads(sess, coord=coord, start=True)

    for i in range(10):
        print(sess.run(q.dequeue()))

    coord.request_stop()
    coord.join(enq_threads)
