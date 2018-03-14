from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.eager as tfe

import numpy as np
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score

from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.layers import dropout

from tensorflow.python import debug as tf_debug

tf.logging.set_verbosity(tf.logging.INFO)

# execution
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/home/avila/mnist_data")

n_inputs = 28*28
n_hidden1 = 300
n_hidden2 = 100
n_outputs= 10

keep_prob = 0.5

n_epochs = 400
batch_size = 50

img = []
sa = []
sb = []

import numpy.random as nprnd

list_elem_train = nprnd.randint(len(mnist.train.images), size=len(mnist.train.images))

#tensor_train_images = tf.Tensor(mnist.train.images)

def parse_sample(train, index):
 
    if train:
        img = mnist.train.images[index]
        label = mnist.train.labels[index]
    else:
        img = mnist.test.images[index]
        label = mnist.test.labels[index]

    label2 = label.astype(np.int32)

    return img, label2


dataset_train = tf.data.Dataset.range(len(mnist.train.images)).map(
    lambda index: tuple(tf.py_func(parse_sample, [True, index], [tf.float32, tf.int32])))
print(dataset_train.output_types)
print(dataset_train.output_shapes)
dataset_train = dataset_train.batch(batch_size)
dataset_train = dataset_train.shuffle(buffer_size=len(mnist.train.images))
iterator_train = tf.data.Iterator.from_structure(
    (tf.float32, tf.int32),
    (tf.TensorShape((None, n_inputs)), tf.TensorShape(None) )
    )
training_init_op = iterator_train.make_initializer(dataset_train)

dataset_test = tf.data.Dataset.range(len(mnist.test.images)).map(
    lambda index: tuple(tf.py_func(parse_sample, [False, index], [tf.float32, tf.int32])))
print(dataset_test.output_types)
print(dataset_test.output_shapes)
dataset_test = dataset_test.batch(len(mnist.test.images))
iterator_test = tf.data.Iterator.from_structure(
    (tf.float32, tf.int32),
    (tf.TensorShape((None, n_inputs)), tf.TensorShape(None) )
    )
test_init_op = iterator_test.make_initializer(dataset_test)



is_training = tf.placeholder(tf.bool, shape=(), name='is_training')

bn_params = {
    'is_training' : is_training,
    'decay' : 0.999,
    'updates_collections' : None,
    'fused':True
    }

with tf.name_scope("dnn"):

    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        handle, (tf.float32, tf.int32),
        (tf.TensorShape((None, n_inputs)), tf.TensorShape(None) )
    )

    X,y = iterator.get_next()

    hidden1 = fully_connected(X, n_hidden1, activation_fn=tf.nn.elu, 
                              scope="hidden1", 
                              normalizer_fn=batch_norm, normalizer_params=bn_params
                              )
    hidden2 = fully_connected(hidden1, n_hidden2, activation_fn=tf.nn.elu,
                              scope="hidden2", 
                            normalizer_fn=batch_norm, normalizer_params=bn_params
                              )
    logits = fully_connected(hidden2, n_outputs, scope="outputs", activation_fn=None)

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss= tf.reduce_mean(xentropy, name="loss")

learning_rate = 0.0001

with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:

    init.run()
    
    training_handle = sess.run(iterator_train.string_handle())
    validation_handle = sess.run(iterator_test.string_handle())

    for epoch in range(n_epochs):
        sess.run(training_init_op)
        try:
            while True:
                sess.run([training_op], feed_dict={is_training:True, handle:training_handle})
        except tf.errors.OutOfRangeError:
            sess.run(test_init_op)
            acc_test = accuracy.eval(feed_dict={is_training:False, handle:validation_handle})
            print(epoch, "Test accuracy:", acc_test)
            pass
