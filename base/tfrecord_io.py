import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('mnist_data', dtype=tf.uint8, one_hot=True)
train_images = mnist.train.images    # (55000, 784)
train_labels = mnist.train.labels    # (55000, 10)
train_count = mnist.train.num_examples    # 55000
val_images = mnist.validation.images    # (5000. 784)
val_labels = mnist.validation.labels    # (5000, 10)
val_count = mnist.validation.num_examples    # 5000
test_images = mnist.test.images    # (10000, 784)
test_labels = mnist.test.labels    # (10000, 10)
test_count = mnist.test.num_examples    # 10000

def create_record(save_path, images, labels, count):
    writer = tf.python_io.TFRecordWriter(save_path)
    for i in range(count):
        image_raw = images[i].tobytes()
        label = np.argmax(labels[i])
        example = tf.train.Example(
            features = tf.train.Features(
                feature = {
                    'image_raw': tf.train.Feature(
                        bytes_list = tf.train.BytesList(value=[image_raw])
                    ),
                    'label': tf.train.Feature(
                        int64_list = tf.train.Int64List(value=[label])
                    )
                }
            )
        )
        writer.write(example.SerializeToString())
    writer.close()

train_path = 'mnist_data/train.record'
create_record(train_path, train_images, train_labels, train_count)
val_path = 'mnist_data/val.record'
create_record(val_path, val_images, val_labels, val_count)
test_path = 'mnist_data/test.record'
create_record(test_path, test_images, test_labels, test_count)

# read one example at one time
with tf.Session() as sess:
    reader = tf.TFRecordReader()    # no need to close
    filename_queue = tf.train.string_input_producer([train_path, val_path, test_path])
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features = {
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        }
    )
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    label = tf.cast(features['label'], tf.int32)
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess = sess, coord = coord)
    
    for i in range(10):
        print(sess.run([image, label]))
        
    coord.request_stop()
    coord.join(threads)

# read multiple examples at one time
with tf.Session() as sess:
    reader = tf.TFRecordReader()  # no need to close
    filename_queue = tf.train.string_input_producer([train_path, val_path, test_path])
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features = {
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        }
    )
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.reshape(image, [28, 28])
    label = tf.cast(features['label'], tf.int32)
    image_batch, label_batch = tf.train.shuffle_batch([image, label]
                        , batch_size = 2
                        , capacity = 200
                        , min_after_dequeue = 100
                        , num_threads = 2
    )
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess = sess, coord = coord)
    
    for i in range(10):
        img_batch, lbl_batch = sess.run([image_batch, label_batch])
        print(img_batch[0], lbl_batch[0])
        print(img_batch[1], lbl_batch[1])
        
    coord.request_stop()
    coord.join(threads)

