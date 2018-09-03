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

from detect_bar_code_constants import target_width,target_height,width,height,MODEL,MODEL_NUMBER,IMAGE_CLONE_NR

import random

tf.logging.set_verbosity(tf.logging.INFO)

n_epochs = 10000
batch_size = 16
learning_rate = 0.005

# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
images = detect_bar_code_images()

def _parse_function(filename, label, org_x, org_y, end_x, end_y, gamma):
    
    source = images.read_source_image(filename)
    result_label = images.read_label_image(label)
    #source = tf.Print(source, data=[filename, org_x, org_y, end_x, end_y, gamma], message="load source ")

    if end_x != 0:
        source_cropped = images.crop_to(source, org_y, org_x, end_y, end_x)
        source_expanded_shrink = images.resize_to(source, [600,800])
        
        source_expanded_shrink_brightness = tf.image.adjust_brightness(source_expanded_shrink, 0.1)
    
        #source = tf.Print(source, data=[gamma], message="COUNTER")
    
        label_exp = tf.expand_dims(result_label, 2)
        label_cropped = images.crop_to(label_exp,
                            tf.to_int32(org_y / 10), 
                            tf.to_int32(org_x / 10), 
                            tf.to_int32(end_y / 10), 
                            tf.to_int32(end_x / 10))
        label_expanded_shrink = images.resize_to(label_cropped, [60,80])
        label_expanded_shrink_resized = tf.squeeze(label_expanded_shrink, [2])
        #label_expanded_shrink_resized = tf.Print(label_expanded_shrink_resized, data=[tf.shape(source_cropped)], 
        #                                         message="label_expanded_shrink_resized ")
        return source_expanded_shrink_brightness, result_label 
    else:
        return source, result_label

import glob, os
sources = []
sources_label = []
list_org_x = []
list_org_y = []
list_end_x = []
list_end_y = []
list_gamma = []
#os.chdir("/home/avila/DATA_REF/")
sample_count = 0
for file in glob.glob("/home/avila/DATA/DATA_REF_RESCALED/*.jpg"):
    print(file)
    label = file.replace("REF", "REF_LABEL")
    #print(label)
    
    sample_count = sample_count+1
    
    for i in range(IMAGE_CLONE_NR):
        sources.append(file)
        sources_label.append(label)
        offset_x = random.random()*20
        offset_y = random.random()*20
        width_x = width - offset_x - random.random()*20
        width_y = height - offset_y -random.random()*20
        list_org_x.append(int(offset_x))
        list_org_y.append(int(offset_y))
        list_end_x.append(int(width_x))
        list_end_y.append(int(width_y))
        list_gamma.append(0)

# A vector of sources.
image_sources = tf.constant(sources)
# A vector of labels.
image_labels = tf.constant(sources_label)

dataset_train = tf.data.Dataset.from_tensor_slices((image_sources, image_labels, list_org_x, list_org_y, list_end_x, list_end_y, list_gamma))
dataset_train = dataset_train.shuffle(buffer_size= 1000*batch_size)
dataset_train = dataset_train.prefetch(buffer_size = batch_size)
dataset_train = dataset_train.map(_parse_function, num_parallel_calls=6)
dataset_train = dataset_train.batch(batch_size)
#dataset_train = dataset_train.repeat(5)
iterator_train = tf.data.Iterator.from_structure(
    (tf.float32, tf.float32),
    (tf.TensorShape((None, height, width, 3)), tf.TensorShape((None, target_height, target_width)) )
    )
training_init_op = iterator_train.make_initializer(dataset_train)

iterator_eval = tf.data.Iterator.from_structure(
        (tf.float32, tf.float32),
        (tf.TensorShape((None, height, width, 3)), tf.TensorShape((None, target_height, target_width)) )
    )

def eval_op_from_file(filename):
    source = images.read_source_image("/home/avila/DATA/DATA_EVAL_RESCALED/" + filename)
    source_expand = tf.expand_dims(source, 0)
    result_label = images.read_label_image("/home/avila/DATA/DATA_EVAL_LABEL_RESCALED/" + filename)
    result_label_expand = tf.expand_dims(result_label, 0)
    dataset_src = tf.data.Dataset.from_tensors(source_expand)
    dataset_lbl = tf.data.Dataset.from_tensors(result_label_expand)
    dataset_eval = tf.data.Dataset.zip((dataset_src, dataset_lbl))
    eval_init_op = iterator_eval.make_initializer(dataset_eval)
    return eval_init_op

is_training = tf.placeholder(tf.bool, shape=(), name='is_training')

#bn_params = {
#    'is_training' : is_training,
#    'decay' : 0.999,
#    'updates_collections' : None,
#    'fused':True
#    }

my_dnn = detect_bar_code_ddn()
 
with tf.name_scope("dnn"):
                
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
                    handle, (tf.float32, tf.float32),
                    (tf.TensorShape((None, height, width, 3)), tf.TensorShape((None, target_height, target_width)) )
                    )
    
    X,y = iterator.get_next()
    (dnn_max, dnn_result) = my_dnn.detect_bar_code_ddn(X, is_training)
                
                
with tf.name_scope("loss"):
    dnn_result_flat = tf.reshape(dnn_max, [-1])
    dnn_result_categories = tf.expand_dims(dnn_result_flat, 1)

    y_flat = tf.reshape(y, [-1])
    y_categories = tf.expand_dims(y_flat, 1)

    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_categories, logits=dnn_result_categories, name="loss")

with tf.name_scope("performance"):

    pixel_count = tf.reduce_prod(tf.shape(y))
    pixel_count = tf.cast(pixel_count, tf.float32)
    sum_active = tf.reduce_sum(y)
    sum_inactive = pixel_count - sum_active

    sum_diff_1 = tf.reduce_sum(tf.abs(y-tf.multiply(dnn_result, y)))
    diff_1 = sum_diff_1/sum_active

    reduced_sum = tf.reduce_sum(tf.abs(tf.multiply(dnn_result, 1.0-y)))
    diff_0 = reduced_sum /sum_inactive

    performance = diff_1 + diff_0

with tf.name_scope("learn"):
    # add an optimiser
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.name_scope("to_image"):
        image = tf.squeeze(dnn_result)
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

LOG_DIR = "/home/avila/MODEL/LOG-{}/".format(MODEL_NUMBER)
STEP_FILE = LOG_DIR + '/current_step.txt'
file_writer = tf.summary.FileWriter(LOG_DIR, tf.get_default_graph())
CURRENT_STEP=0
try:
    with open(STEP_FILE, 'r') as f:
        print("Found file step " + STEP_FILE)
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

        run_performance = run_performance / batch_count

        saver.save(sess, MODEL)
                
        print("New eval " +str(epoch) + " total run_performance " + str(run_performance))
        
        count=0
        total_eval = 0
        for file in glob.glob("/home/avila/DATA/DATA_EVAL_RESCALED/*.jpg"):
            file=file.replace("/home/avila/DATA/DATA_EVAL_RESCALED/", "")
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
        print("overall learn error  " + str(run_performance))
        print("overall test error  " + str(total_eval))
        summary_perf = tf.Summary(value=[
            tf.Summary.Value(tag="learn_error", simple_value=run_performance),
            tf.Summary.Value(tag="test_error", simple_value=total_eval)
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
