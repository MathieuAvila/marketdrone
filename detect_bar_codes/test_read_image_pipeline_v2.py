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

n_epochs = 1000
batch_size = 5
learning_rate = 0.0001 
width=800
height=600

# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
def _parse_function(filename, label):

  #filename = tf.Print(filename, data=[filename], message="filename")
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_image(image_string)
  reshaped_image_decoded = tf.cast(image_decoded, tf.float32)
  reshaped_image_decoded = reshaped_image_decoded / 256
  #reshaped_image_decoded = tf.Print(reshaped_image_decoded, data=[tf.shape(reshaped_image_decoded)], message="reshaped_image_decoded")

  #label = tf.Print(label, data=[label], message="label")
  label_string = tf.read_file(label)
  label_decoded = tf.image.decode_image(label_string)
  reshaped_label_decoded = tf.cast(label_decoded, tf.float32)
  reshaped_label_decoded = reshaped_label_decoded / 256 / 3
  
  #reshaped_label_decoded = tf.Print(reshaped_label_decoded, data=[tf.shape(reshaped_label_decoded)],
  #                                  message="reshaped_label_decoded")
  reshaped_label_decoded_reduced = tf.reduce_sum(reshaped_label_decoded, 2)
  #reshaped_label_decoded_reduced = tf.Print(reshaped_label_decoded_reduced, 
  #                                          data=[tf.shape(reshaped_label_decoded_reduced)],
  #                                          message="reshaped_label_decoded_reduced")

  return reshaped_image_decoded, reshaped_label_decoded_reduced

import glob, os
sources = []
sources_label = []
#os.chdir("/home/avila/DATA_REF/")
for file in glob.glob("/home/avila/DATA/DATA_REF_RESCALED/*.jpg"):
    #print(file)
    sources.append(file)
    label = file.replace("REF", "REF_LABEL")
    #print(label)
    sources_label.append(label)
# A vector of sources.
image_sources = tf.constant(sources)
# A vector of labels.
image_labels = tf.constant(sources_label)

dataset_train = tf.data.Dataset.from_tensor_slices((image_sources, image_labels))
dataset_train = dataset_train.map(_parse_function)
dataset_train = dataset_train.batch(batch_size)
#dataset_train = dataset_train.repeat(5)
iterator_train = tf.data.Iterator.from_structure(
    (tf.float32, tf.float32),
    (tf.TensorShape((None, width, height, 3)), tf.TensorShape((None, width, height)) )
    )
training_init_op = iterator_train.make_initializer(dataset_train)

iterator_eval = tf.data.Iterator.from_structure(
        (tf.float32, tf.float32),
        (tf.TensorShape((None, width, height, 3)), tf.TensorShape((None, width, height)) )
    )

def eval_op_from_file(filename):
    image_sources_eval = ["/home/avila/DATA/DATA_REF_RESCALED/" + filename]
    image_labels_eval = ["/home/avila/DATA/DATA_REF_LABEL_RESCALED/" + filename]
    dataset_eval = tf.data.Dataset.from_tensor_slices((image_sources_eval, image_labels_eval))
    dataset_eval = dataset_eval.map(_parse_function)
    dataset_eval = dataset_eval.batch(1)
    eval_init_op = iterator_eval.make_initializer(dataset_eval)
    return eval_init_op

is_training = tf.placeholder(tf.bool, shape=(), name='is_training')

bn_params = {
    'is_training' : is_training,
    'decay' : 0.999,
    'updates_collections' : None,
    'fused':True
    }

def create_new_conv_layer(input_data, 
                          num_input_channels, 
                          num_filters, 
                          filter_shape,
                          strides,
                          name):
    # setup the filter input shape for tf.nn.conv_2d
    conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels,  num_filters]

    # initialise weights and bias for the filter
    weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03), name=name+'_W')
    bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')
    # setup the convolutional layer operation
    out_layer = tf.nn.conv2d(input_data, weights, strides, padding='SAME')
    #print(tf.shape(out_layer))
    #out_layer = tf.Print(out_layer, data=[tf.shape(out_layer)], message="out_layer")
   
    #out_layer += bias

    #out_layer = tf.nn.relu(out_layer)

    return out_layer

with tf.name_scope("dnn"):

    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        handle, (tf.float32, tf.float32),
        (tf.TensorShape((None, width, height, 3)), tf.TensorShape((None, width, height)) )
    )

    X,y = iterator.get_next()
    #X = tf.Print(X, data=[tf.shape(X)], message="X")
    #y = tf.Print(y, data=[tf.shape(y)], message="y")
    
    conv_layer_1 = create_new_conv_layer(X, 3, 64, [10, 10], [1, 1, 1, 1], name='layer1'),
    #conv_layer_1 = tf.Print(conv_layer_1, data=[tf.shape(conv_layer_1)], message="conv_layer_1")
    conv_layer_1 = tf.reduce_max(conv_layer_1, axis=[0])
    #conv_layer_1 = tf.Print(conv_layer_1, data=[tf.shape(conv_layer_1)], message="conv_layer_1")
   
    max_pool_1 = tf.nn.max_pool(conv_layer_1, [1,10,10,1], strides=[1,10,10,1], padding="VALID")
    #max_pool_1 = tf.Print(max_pool_1, data=[tf.shape(max_pool_1)], message="max_pool_1")
    
    conv_layer_2 = create_new_conv_layer(max_pool_1, 64, 64, [10, 10], [1,1,1,1], name='layer2')
    #conv_layer_2 = tf.Print(conv_layer_2, data=[tf.shape(conv_layer_2)], message="conv_layer_2")
    
    conv_layer_3 = create_new_conv_layer(conv_layer_2, 64, 64, [10, 10], [1,1,1,1], name='layer3')
    #conv_layer_2 = tf.Print(conv_layer_2, data=[tf.shape(conv_layer_2)], message="conv_layer_2")
    
    max_pool_2 = tf.reduce_max(conv_layer_3, axis=[3])
    #max_pool_2 = tf.Print(max_pool_2, data=[tf.shape(max_pool_2)], message="max_pool_2")
    max_pool_2 = tf.squeeze(max_pool_2)
    #max_pool_2 = tf.Print(max_pool_2, data=[tf.shape(max_pool_2)], message="max_pool_2")
    #max_pool_2 = tf.Print(max_pool_2, data=[max_pool_2], message="max_pool_2")

    result = tf.nn.sigmoid(max_pool_2)
    #result = tf.Print(result, data=[result], message="result")
    diff_raw = y*5-result
    diff = tf.square(tf.abs(diff_raw)*10)
    #diff = tf.Print(diff, data=[diff, diff_raw, y], message="diff, diff_raw, y: ")

with tf.name_scope("loss"):
    #diff_linear = tf.reshape(diff, tf.TensorShape((batch_size*width*height*3)), name="diff_linear")
    #diff_linear = tf.Print(diff_linear, data=[tf.shape(diff_linear)], message="diff_linear")
    loss = tf.reduce_sum(diff)
    #loss = tf.Print(loss, data=[loss], message="loss")


with tf.name_scope("learn"):
    # add an optimiser
    optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

init = tf.global_variables_initializer()
#saver = tf.train.Saver()


with tf.name_scope("to_image"):
    image = result
    #image = tf.Print(image, data=[tf.shape(image)], message="image")
    
    image_dim3 = tf.squeeze(image)
    #image_dim3 = tf.Print(image_dim3, data=[tf.shape(image_dim3)], message="image_dim3")
   
    #image_dim2 = tf.reduce_sum(image_dim3, 2)
    #image_dim2 = tf.Print(image_dim2, data=[tf.shape(image_dim2)], message="image_dim2")
   
    image_re_expand3 = tf.expand_dims(image_dim3, 2)
    #image_re_expand3 = tf.Print(image_re_expand3, data=[tf.shape(image_re_expand3)], message="image_re_expand3")
   
    image_expand3_3 = tf.tile(image_re_expand3, [1, 1, 3])
    #image_expand3_3 = tf.Print(image_expand3_3, data=[tf.shape(image_expand3_3)], message="image_expand3_3")
   
    #image = tf.Print(image, data=[tf.shape(image)], message="image")
    #raw = tf.reshape(image_re_expand3, [width,height,3])
    
    #raw = tf.Print(raw, data=[tf.shape(raw)], message="raw")
    
    image_expand3_3 = image_expand3_3 * 255
    raw_uint8 = tf.cast(image_expand3_3, dtype=tf.uint8)
    #raw_uint8 = tf.Print(raw_uint8, data=[tf.shape(raw_uint8)], message="raw_uint8")
    img = tf.image.encode_jpeg(raw_uint8)
    
    output_file = tf.placeholder(tf.string, shape=(), name='output_file')

    write_image = tf.write_file(output_file, img)


with tf.Session() as sess:

    init.run()
    
    training_handle = sess.run(iterator_train.string_handle())
    eval_handle = sess.run(iterator_eval.string_handle())
        
    for epoch in range(n_epochs):
        print("Range %s" % str(epoch))
        sess.run(training_init_op)
        total_loss = 0
        try:
            while True:
                print("New train %s" % str(epoch))
                val = sess.run([optimiser, loss], feed_dict={is_training:True, handle:training_handle})
                total_loss = total_loss + val[1]
        except tf.errors.OutOfRangeError:
            pass
        print("New eval " +str(epoch) + " total loss " + str(total_loss))
        
        for file in glob.glob("/home/avila/DATA/DATA_REF_RESCALED/*.jpg"):
            file=file.replace("/home/avila/DATA/DATA_REF_RESCALED/", "")
            print(file)
            eval_init_op = eval_op_from_file(file)
            sess.run(eval_init_op)        
            sess.run([write_image], feed_dict={is_training:False, 
                                               output_file:"/home/avila/RESULT/" + file, 
                                               handle:eval_handle})
