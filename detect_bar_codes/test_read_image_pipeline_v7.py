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

tf.logging.set_verbosity(tf.logging.INFO)

n_epochs = 10000
batch_size = 32
learning_rate = 0.0005
width=800
height=600

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops

_rgb_to_yuv_kernel = [[0.299, -0.14714119, 0.61497538], 
                     [0.587, -0.28886916, -0.51496512],
                     [0.114, 0.43601035, -0.10001026]]

# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
def _parse_function(filename, label):

  #filename = tf.Print(filename, data=[filename], message="filename")
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_image(image_string)
  
  ndims = image_decoded.get_shape().ndims
  #print(ndims)
  #print(image_decoded.get_shape())
  #print(image_decoded)
  #print(filename)
  
  reshaped_image_decoded = tf.cast(image_decoded, tf.float32)
  reshaped_image_decoded = reshaped_image_decoded / 256
  
  #image_decoded_yuv = tf.image.rgb_to_yuv(reshaped_image_decoded)
  
  kernel = ops.convert_to_tensor(_rgb_to_yuv_kernel, dtype=reshaped_image_decoded.dtype, name='kernel')
  image_decoded_yuv = math_ops.tensordot(reshaped_image_decoded, kernel, axes=[[2], [0]])
  
  #reshaped_image_decoded = tf.Print(reshaped_image_decoded, data=[tf.shape(reshaped_image_decoded)], message="reshaped_image_decoded")

  #label = tf.Print(label, data=[label], message="label")
  label_string = tf.read_file(label)
  label_decoded = tf.image.decode_image(label_string)
  reshaped_label_decoded = tf.cast(label_decoded, tf.float32)
  reshaped_label_decoded = reshaped_label_decoded / 256

  reshaped_label_decoded_reduced = tf.reduce_sum(reshaped_label_decoded, 2)
  reshaped_label_decoded_reduced = ((tf.sign(reshaped_label_decoded_reduced - 0.5))+1.0)/2.0

  #my_max = tf.reduce_max(reshaped_label_decoded)
  #my_min = tf.reduce_min(reshaped_label_decoded)
  #reshaped_label_decoded_reduced = tf.Print(reshaped_label_decoded_reduced, 
                                            #data=[my_max, my_min], 
                                            #message="reshaped_label_decoded_reduced")

  return image_decoded_yuv, reshaped_label_decoded_reduced

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
    
    count = count+1
    #if count == batch_size*2:
    #    break

# A vector of sources.
image_sources = tf.constant(sources)
# A vector of labels.
image_labels = tf.constant(sources_label)

dataset_train = tf.data.Dataset.from_tensor_slices((image_sources, image_labels))
dataset_train = dataset_train.shuffle(buffer_size= 1000*batch_size)
dataset_train = dataset_train.prefetch(buffer_size = batch_size)
dataset_train = dataset_train.map(_parse_function, num_parallel_calls=6)
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
    image_sources_eval = ["/home/avila/DATA/DATA_EVAL_RESCALED/" + filename]
    image_labels_eval = ["/home/avila/DATA/DATA_EVAL_LABEL_RESCALED/" + filename]
    #image_labels_eval = ["/home/avila/DATA/void_target.jpg"]
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
                          norm,
                          is_training,
                          name):
    # setup the filter input shape for tf.nn.conv_2d
    conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels,  num_filters]

    # initialise weights and bias for the filter
    weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.1), name=name+'_W')
    bias = tf.Variable(tf.truncated_normal([num_filters], mean=-0.1), name=name+'_b')
    # setup the convolutional layer operation
    out_layer = tf.nn.conv2d(input_data, weights, strides, padding='SAME')
    #out_layer = tf.Print(out_layer, data=[tf.shape(out_layer)], message="out_layer")
   
    out_layer = out_layer + bias
    #out_layer = tf.Print(out_layer, data=[tf.shape(out_layer)], message="out_layer")
    
    if norm:
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
    
    conv_layer_1 = create_new_conv_layer(X, 3, 16, [5, 5], [1, 1, 1, 1], True, True, is_training, name='layer1')
    #conv_layer_1 = tf.Print(conv_layer_1, data=[tf.reduce_min(conv_layer_1), tf.reduce_max(conv_layer_1)], message="conv_layer_1")
    #conv_layer_1_max = tf.reduce_max(conv_layer_1, axis=[0])
    
    conv_layer_1_1 = create_new_conv_layer(conv_layer_1, 16, 16, [1, 1], [1, 1, 1, 1], True,  True, is_training, name='layer1_max')
    #conv_layer_1_1 = tf.Print(conv_layer_1_1, data=[tf.reduce_min(conv_layer_1_1), tf.reduce_max(conv_layer_1_1)], message="conv_layer_1_1")
    #conv_layer_1 = tf.Print(conv_layer_1, data=[tf.shape(conv_layer_1)], message="conv_layer_1")
    
    max_pool_1 = tf.nn.max_pool(conv_layer_1_1, [1,10,10,1], strides=[1,10,10,1], padding="VALID")
    #max_pool_1 = tf.Print(max_pool_1, data=[tf.reduce_min(max_pool_1), tf.reduce_max(max_pool_1)], message="max_pool_1")
    #max_pool_1 = tf.Print(max_pool_1, data=[tf.shape(max_pool_1)], message="max_pool_1")
    
    conv_layer_2 = create_new_conv_layer(max_pool_1, 16, 8, [4, 4], [1,1,1,1], True,  True, is_training, name='layer2')
    #conv_layer_2 = tf.Print(conv_layer_2, data=[tf.reduce_min(conv_layer_2), tf.reduce_max(conv_layer_2)], message="conv_layer_2")
    #conv_layer_2 = tf.Print(conv_layer_2, data=[tf.shape(conv_layer_2)], message="conv_layer_2")
    
    conv_layer_3 = create_new_conv_layer(conv_layer_2, 8, 1, [4, 4], [1,1,1,1], True,  True, is_training, name='layer3')
    #conv_layer_3 = tf.Print(conv_layer_3, data=[tf.reduce_min(conv_layer_3), tf.reduce_max(conv_layer_3)], message="conv_layer_3")
    #conv_layer_3 = tf.Print(conv_layer_2, data=[tf.shape(conv_layer_2)], message="conv_layer_2")
    
    conv_layer_4 = create_new_conv_layer(conv_layer_3, 1, 1, [1, 1], [1,1,1,1], False,  False, is_training, name='layer4')
    #conv_layer_4 = tf.Print(conv_layer_4, data=[tf.reduce_min(conv_layer_3), tf.reduce_max(conv_layer_3)], message="conv_layer_4")
 
    max_pool_2 = tf.squeeze(conv_layer_4)
    
    result = tf.nn.sigmoid(max_pool_2)

 
with tf.name_scope("loss"):
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=result, name="loss")
    loss2 = tf.reduce_sum(abs(loss))


with tf.name_scope("learn"):
    # add an optimiser
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

_yuv_to_rgb_kernel = [[1, 1, 1], [0, -0.394642334, 2.03206185],
                      [1.13988303, -0.58062185, 0]]

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
        
        kernel_back = ops.convert_to_tensor(_yuv_to_rgb_kernel, dtype=source_dim3.dtype, name='kernel_back')
        image_decoded_rgb = math_ops.tensordot(source_dim3, kernel_back, axes=[[2], [0]])
        
        image_concat2 = tf.concat([image_decoded_rgb,image_expand3_3], 1)
        

        image_expand3_3 = image_concat2 * 255
        raw_uint8 = tf.cast(image_expand3_3, dtype=tf.uint8)
        img = tf.image.encode_jpeg(raw_uint8)
        output_file = tf.placeholder(tf.string, shape=(), name='output_file')
        write_image = tf.write_file(output_file, img)

with tf.name_scope("performance"):
     pixel_count = tf.reduce_prod(tf.shape(y))
     pixel_count = tf.cast(pixel_count, tf.float32)
     sum_active = tf.reduce_sum(y)
     sum_inactive = pixel_count - sum_active
 
     sum_diff_1 = tf.reduce_sum(tf.abs(y-tf.multiply(result, y)))
     diff_1 = sum_diff_1/sum_active
     
     reduced_sum = tf.reduce_sum(tf.abs(tf.multiply(result, 1.0-y)))
     diff_0 = reduced_sum /sum_inactive
     performance = diff_1 + diff_0

config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )

MODEL_NUMBER=7
MODEL="/home/avila/MODEL/model-{}.cpkt".format(MODEL_NUMBER)
LOG_DIR = "/home/avila/MODEL/LOG-{}/".format(MODEL_NUMBER)
STEP_FILE = LOG_DIR + '/current_step.txt'
file_writer = tf.summary.FileWriter(LOG_DIR, tf.get_default_graph())
CURRENT_STEP=0
try:
    with open(STEP_FILE, 'r') as f:
        CURRENT_STEP = int(f.read())
        f.close()
    print("Current step set to " + str(CURRENT_STEP))
except OSError as e:
    print("No current step, reset to 0")
    pass

with tf.Session(config=config) as sess:

    #var_list = [str(x) for x in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)]
    #print("\n".join(var_list))
    
    #init.run()
    try:
        saver.restore(sess, MODEL)
        #saver.restore(sess, "/home/avila/model.cpkt")
        print("restored session")
    except:       
        print("start session from scratch")
        init.run()
    
    training_handle = sess.run(iterator_train.string_handle())
    eval_handle = sess.run(iterator_eval.string_handle())
        
    for epoch in range(n_epochs):
        print("Range %s" % str(epoch))
        
        sess.run(training_init_op)
        run_performance = 0
        try:
            batch_count = 0
            while True:
                batch_count = batch_count + 1
                print("New train epoch = %i, count = %i" % (epoch, batch_count))
                val = sess.run([optimiser, performance], feed_dict={is_training:True, handle:training_handle})
                run_performance = run_performance + val[1]
                print("performance %s" % str(val[1]))

        except tf.errors.OutOfRangeError:
            pass

        run_performance = run_performance / (batch_count*batch_size)

        saver.save(sess, MODEL)
                
        print("New eval " +str(epoch) + " total run_performance " + str(run_performance))
        
        count=0
        total_eval = 0
        for file in glob.glob("/home/avila/DATA/DATA_EVAL_RESCALED/*.jpg"):
            file=file.replace("/home/avila/DATA/DATA_EVAL_RESCALED/", "")
            #print(file)
            eval_init_op = eval_op_from_file(file)
            sess.run(eval_init_op)        
            output = sess.run([write_image, performance], feed_dict={is_training:False, 
                                               output_file:"/home/avila/RESULT/" + file, 
                                               handle:eval_handle})
            print("performance " + str(output[1]))
            total_eval = total_eval + output[1]
            count = count + 1
            #if count > 5:
            #    break

        total_eval = total_eval / count

        summary_perf = tf.Summary(value=[
            tf.Summary.Value(tag="performance", simple_value=run_performance),
            tf.Summary.Value(tag="evaluation", simple_value=total_eval)
            ])
        file_writer.add_summary(summary_perf, CURRENT_STEP)

        file_writer.flush()

        CURRENT_STEP = CURRENT_STEP + 1

        try:
            with open(STEP_FILE, 'w') as f:
                f.write('%d' % CURRENT_STEP)
                f.close()
            print("Current step saved to " + str(CURRENT_STEP))
        except OSError as e:
            print("Enable to save current step " + str(e))
            pass
