import tensorflow as tf

class detect_bar_code_ddn:
    
    def create_new_conv_layer(this, 
                              input_data, 
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
            
    def detect_bar_code_ddn(this, X, is_training):

            conv_layer_1 = this.create_new_conv_layer(X, 3, 16, [5, 5], [1, 1, 1, 1], True, True, is_training, name='layer1')
                #conv_layer_1 = tf.Print(conv_layer_1, data=[tf.reduce_min(conv_layer_1), tf.reduce_max(conv_layer_1)], message="conv_layer_1")
                #conv_layer_1_max = tf.reduce_max(conv_layer_1, axis=[0])

            conv_layer_1_1 = this.create_new_conv_layer(conv_layer_1, 16, 16, [5, 5], [1, 1, 1, 1], True,  True, is_training, name='layer1_max')
                #conv_layer_1_1 = tf.Print(conv_layer_1_1, data=[tf.reduce_min(conv_layer_1_1), tf.reduce_max(conv_layer_1_1)], message="conv_layer_1_1")
                #conv_layer_1 = tf.Print(conv_layer_1, data=[tf.shape(conv_layer_1)], message="conv_layer_1")

            max_pool_1 = tf.nn.max_pool(conv_layer_1_1, [1,10,10,1], strides=[1,10,10,1], padding="VALID")
                #max_pool_1 = tf.Print(max_pool_1, data=[tf.reduce_min(max_pool_1), tf.reduce_max(max_pool_1)], message="max_pool_1")
                #max_pool_1 = tf.Print(max_pool_1, data=[tf.shape(max_pool_1)], message="max_pool_1")

            conv_layer_2 = this.create_new_conv_layer(max_pool_1, 16, 8, [4, 4], [1,1,1,1], True,  True, is_training, name='layer2')
                #conv_layer_2 = tf.Print(conv_layer_2, data=[tf.reduce_min(conv_layer_2), tf.reduce_max(conv_layer_2)], message="conv_layer_2")
                #conv_layer_2 = tf.Print(conv_layer_2, data=[tf.shape(conv_layer_2)], message="conv_layer_2")

            conv_layer_3 = this.create_new_conv_layer(conv_layer_2, 8, 4, [4, 4], [1,1,1,1], True,  True, is_training, name='layer3')
                #conv_layer_3 = tf.Print(conv_layer_3, data=[tf.reduce_min(conv_layer_3), tf.reduce_max(conv_layer_3)], message="conv_layer_3")
                #conv_layer_3 = tf.Print(conv_layer_2, data=[tf.shape(conv_layer_2)], message="conv_layer_2")

            conv_layer_4 = this.create_new_conv_layer(conv_layer_3, 4, 2, [4, 4], [1,1,1,1], True,  True, is_training, name='layer4')
                #conv_layer_3 = tf.Print(conv_layer_3, data=[tf.reduce_min(conv_layer_3), tf.reduce_max(conv_layer_3)], message="conv_layer_3")
                #conv_layer_3 = tf.Print(conv_layer_2, data=[tf.shape(conv_layer_2)], message="conv_layer_2")

            conv_layer_5 = this.create_new_conv_layer(conv_layer_4, 2, 1, [4, 4], [1,1,1,1], True,  True, is_training, name='layer5')
                #conv_layer_3 = tf.Print(conv_layer_3, data=[tf.reduce_min(conv_layer_3), tf.reduce_max(conv_layer_3)], message="conv_layer_3")
                #conv_layer_3 = tf.Print(conv_layer_2, data=[tf.shape(conv_layer_2)], message="conv_layer_2")

            conv_layer_6 = this.create_new_conv_layer(conv_layer_5, 1, 1, [1, 1], [1,1,1,1], False,  False, is_training, name='layer6')

            max_pool_2 = tf.squeeze(conv_layer_6)

            result = tf.nn.sigmoid(max_pool_2)

            return result
