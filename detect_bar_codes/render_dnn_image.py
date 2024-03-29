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

from datetime import datetime

from detect_bar_code_ddn import detect_bar_code_ddn
from detect_bar_code_images import detect_bar_code_images

from detect_bar_code_constants import target_height, target_width, width,height,MODEL,MODEL_NUMBER
 
tf.logging.set_verbosity(tf.logging.INFO)

# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
images = detect_bar_code_images()

import glob, os

iterator_eval = tf.data.Iterator.from_structure(
        (tf.float32, tf.float32),
        (tf.TensorShape((None, height, width, 3)), tf.TensorShape((None, target_height, target_width)) )
    )

def eval_op_from_file(filename):
    source = images.read_source_image(filename)
    source_expand = tf.expand_dims(source, 0)
    result_label = images.read_label_image("/home/avila/DATA/void_target.jpg")
    result_label_expand = tf.expand_dims(result_label, 0)
    dataset_src = tf.data.Dataset.from_tensors(source_expand)
    dataset_lbl = tf.data.Dataset.from_tensors(result_label_expand)
    dataset_eval = tf.data.Dataset.zip((dataset_src, dataset_lbl))
    eval_init_op = iterator_eval.make_initializer(dataset_eval)
    return eval_init_op

is_training = tf.placeholder(tf.bool, shape=(), name='is_training')

def _parse_function(filename, label):
    
    return images.read_source_image(filename), images.read_label_image(label)


my_dnn = detect_bar_code_ddn()
 
with tf.name_scope("dnn"):
                
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
                    handle, (tf.float32, tf.float32),
                    (tf.TensorShape((None, height, width, 3)), tf.TensorShape((None, target_height, target_width)) )
                    )
    
    X,y = iterator.get_next()
    (dnn_max, result)  = my_dnn.detect_bar_code_ddn(X, is_training)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops


with tf.name_scope("to_image"):
        image = tf.squeeze(result)
        target = tf.squeeze(y)
        #image = tf.Print(image, data=[tf.shape(image), tf.shape(target), tf.shape(result), tf.shape(y)], message="image ")
        image_step = (tf.sign(image - 0.5)+1)/2

        image_concat = tf.concat([target,image, image_step], 1)
        
        image_re_expand3 = tf.expand_dims(image_concat, 2)
        image_expand3_3 = tf.tile(image_re_expand3, [1, 1, 3])

        source_dim3 = tf.nn.max_pool(X, [1,10,10,1], strides=[1,10,10,1], padding="VALID")
        source_dim3 = tf.squeeze(source_dim3)
        
        image_concat2 = tf.concat([source_dim3,image_expand3_3], 1)

        image_expand3_3 = image_concat2 * 255
        raw_uint8 = tf.cast(image_expand3_3, dtype=tf.uint8)
        img = tf.image.encode_jpeg(raw_uint8)
        output_file = tf.placeholder(tf.string, shape=(), name='output_file')
        write_image = tf.write_file(output_file, img)

config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )

import sys

if len(sys.argv) != 3:
    print("%s INPUT OUTPUT" %(sys.argv[0]))
    exit(1)   

my_file = sys.argv[1]
my_output = sys.argv[2]

with tf.Session(config=config) as sess:

        try:
            saver.restore(sess, MODEL)
            print("restored session")
        except:       
            print("DO NOT start session from scratch")
            exit(1)

        eval_handle = sess.run(iterator_eval.string_handle())

        eval_init_op = eval_op_from_file(my_file)
        sess.run(eval_init_op)        
        output = sess.run([write_image], feed_dict={is_training:False, 
                                                    output_file:my_output, 
                                                    handle:eval_handle})
    
