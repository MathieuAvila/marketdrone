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
batch_size = 6
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
  reshaped_image_decoded = reshaped_image_decoded / 255
  #reshaped_image_decoded = tf.Print(reshaped_image_decoded, data=[tf.shape(reshaped_image_decoded)], message="reshaped_image_decoded")

  #label = tf.Print(label, data=[label], message="label")
  label_string = tf.read_file(label)
  label_decoded = tf.image.decode_image(label_string)
  reshaped_label_decoded = tf.cast(label_decoded, tf.float32)
  reshaped_label_decoded = reshaped_label_decoded / 255
  
  reshaped_label_decoded_reduced = tf.reduce_sum(reshaped_label_decoded, 2) / 3

  #reshaped_label_decoded_reduced = tf.Print(reshaped_label_decoded_reduced, data=[tf.shape(reshaped_label_decoded_reduced)],
  #                                  message="reshaped_label_decoded_reduced")

  return reshaped_image_decoded, reshaped_label_decoded_reduced

import glob, os
sources = []
sources_label = []
#os.chdir("/home/avila/DATA_REF/")
count = 0
for file in glob.glob("/home/avila/DATA/DATA_REF_RESCALED/*.jpg"):
    print(file)
    sources.append(file)
    label = file.replace("REF", "REF_LABEL")
    #print(label)
    sources_label.append(label)
    #count = count+1
    #if count == batch_size*2:
        #break

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
    image_labels_eval = ["/home/avila/DATA/void_target.jpg"]
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
                          activation,
                          is_training,
                          name):
    # setup the filter input shape for tf.nn.conv_2d
    conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels,  num_filters]

    # initialise weights and bias for the filter
    weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.1), name=name+'_W')
    bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')
    # setup the convolutional layer operation
    out_layer = tf.nn.conv2d(input_data, weights, strides, padding='SAME')
    #out_layer = tf.Print(out_layer, data=[tf.shape(out_layer)], message="out_layer")
   
    out_layer = out_layer + bias
    #out_layer = tf.Print(out_layer, data=[tf.shape(out_layer)], message="out_layer")
    
    out_layer = tf.contrib.layers.batch_norm(out_layer, 
                                            center=True, scale=True, 
                                            is_training=is_training,
                                            scope=name)
    
    if activation:
        out_layer = tf.nn.elu(out_layer)
        #out_layer = tf.Print(out_layer, data=[tf.shape(out_layer)], message="out_layer")

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
    
    conv_layer_1 = create_new_conv_layer(X, 3, 128, [10, 10], [1, 1, 1, 1], True, is_training, name='layer1')
    #conv_layer_1 = tf.Print(conv_layer_1, data=[tf.reduce_min(conv_layer_1), tf.reduce_max(conv_layer_1)], message="conv_layer_1")
    #conv_layer_1_max = tf.reduce_max(conv_layer_1, axis=[0])
    
    conv_layer_1_1 = create_new_conv_layer(conv_layer_1, 128, 128, [1, 1], [1, 1, 1, 1], True, is_training, name='layer1_max')
    #conv_layer_1_1 = tf.Print(conv_layer_1_1, data=[tf.reduce_min(conv_layer_1_1), tf.reduce_max(conv_layer_1_1)], message="conv_layer_1_1")
    #conv_layer_1 = tf.Print(conv_layer_1, data=[tf.shape(conv_layer_1)], message="conv_layer_1")
    
    max_pool_1 = tf.nn.max_pool(conv_layer_1_1, [1,10,10,1], strides=[1,10,10,1], padding="VALID")
    #max_pool_1 = tf.Print(max_pool_1, data=[tf.reduce_min(max_pool_1), tf.reduce_max(max_pool_1)], message="max_pool_1")
    #max_pool_1 = tf.Print(max_pool_1, data=[tf.shape(max_pool_1)], message="max_pool_1")
    
    conv_layer_2 = create_new_conv_layer(max_pool_1, 128, 64, [4, 4], [1,1,1,1], True, is_training, name='layer2')
    #conv_layer_2 = tf.Print(conv_layer_2, data=[tf.reduce_min(conv_layer_2), tf.reduce_max(conv_layer_2)], message="conv_layer_2")
    #conv_layer_2 = tf.Print(conv_layer_2, data=[tf.shape(conv_layer_2)], message="conv_layer_2")
    
    conv_layer_3 = create_new_conv_layer(conv_layer_2, 64, 1, [4, 4], [1,1,1,1], True, is_training, name='layer3')
    #conv_layer_3 = tf.Print(conv_layer_3, data=[tf.reduce_min(conv_layer_3), tf.reduce_max(conv_layer_3)], message="conv_layer_3")
    #conv_layer_3 = tf.Print(conv_layer_2, data=[tf.shape(conv_layer_2)], message="conv_layer_2")
 
    max_pool_2 = tf.squeeze(conv_layer_3)
    
    result = tf.nn.sigmoid(max_pool_2)

 
with tf.name_scope("loss"):
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=result, name="loss")
    #loss = tf.reduce_sum(diff)


with tf.name_scope("learn"):
    # add an optimiser
    optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

init = tf.global_variables_initializer()

saver = tf.train.Saver()


with tf.name_scope("to_image"):
    
    
        image = tf.squeeze(result)
        target = tf.squeeze(y)
        #image = tf.Print(image, data=[tf.shape(image), tf.shape(target), tf.shape(result), tf.shape(y)], message="image ")
        image_step = (tf.sign(image - 0.9)+1)/2

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

MODEL="/home/avila/model.cpkt"
with tf.Session(config=config) as sess:

    var_list = [str(x) for x in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)]
    
    print("\n".join(var_list))
    
    #init.run()
    try:
        saver.restore(sess, MODEL)
        print("restored session")
    except:       
        print("start session from scratch")
        init.run()
    
    training_handle = sess.run(iterator_train.string_handle())
    eval_handle = sess.run(iterator_eval.string_handle())
        
    for epoch in range(n_epochs):
        print("Range %s" % str(epoch))
        sess.run(training_init_op)
        total_loss = 0
        try:
            batch_count = 0
            while True:
                batch_count = batch_count + 1
                print("New train epoch = %i, count = %i" % (epoch, batch_count))
                val = sess.run([optimiser, loss], feed_dict={is_training:True, handle:training_handle})
                total_loss = total_loss + val[1]
                print("loss %s" % str(val[1]))
                
        except tf.errors.OutOfRangeError:
            pass
        
        saver.save(sess, MODEL)
                
        print("New eval " +str(epoch) + " total loss " + str(total_loss))
        
        count=0
        for file in glob.glob("/home/avila/DATA/DATA_REF_RESCALED/*.jpg"):
            file=file.replace("/home/avila/DATA/DATA_REF_RESCALED/", "")
            #print(file)
            eval_init_op = eval_op_from_file(file)
            sess.run(eval_init_op)        
            output = sess.run([write_image], feed_dict={is_training:True, 
                                               output_file:"/home/avila/RESULT/" + file, 
                                               handle:eval_handle})
            #print("result " + str(output[0]))
                
            count = count + 1
            if count == 5:
                break
