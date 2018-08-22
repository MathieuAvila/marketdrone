import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops


class detect_bar_code_images:

    def read_source_image(this, filename):

        #filename = tf.Print(filename, data=[filename], message="filename")
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_image(image_string)
        print(image_decoded)
        ndims = image_decoded.get_shape().ndims
  
        reshaped_image_decoded = tf.cast(image_decoded, tf.float32)
        reshaped_image_decoded = reshaped_image_decoded / 256
        _rgb_to_yuv_kernel = [[0.299, -0.14714119, 0.61497538], 
                             [0.587, -0.28886916, -0.51496512],
                             [0.114, 0.43601035, -0.10001026]]
        kernel = ops.convert_to_tensor(_rgb_to_yuv_kernel, dtype=reshaped_image_decoded.dtype, name='kernel')
        image_decoded_yuv = math_ops.tensordot(reshaped_image_decoded, kernel, axes=[[2], [0]])
        return image_decoded_yuv
  
    def read_label_image(this, label):
        #label = tf.Print(label, data=[label], message="label")
        label_string = tf.read_file(label)
        label_decoded = tf.image.decode_image(label_string)
        reshaped_label_decoded = tf.cast(label_decoded, tf.float32)
        reshaped_label_decoded = reshaped_label_decoded / 256

        reshaped_label_decoded_reduced = tf.reduce_sum(reshaped_label_decoded, 2)
        reshaped_label_decoded_reduced = ((tf.sign(reshaped_label_decoded_reduced - 0.5))+1.0)/2.0
        
        return reshaped_label_decoded_reduced
    
    def convert_image_yuv_to_rgb(this, source):
        _yuv_to_rgb_kernel = [[1, 1, 1], [0, -0.394642334, 2.03206185],
                                [1.13988303, -0.58062185, 0]]

        kernel_back = ops.convert_to_tensor(_yuv_to_rgb_kernel, dtype=source.dtype, name='kernel_back')
        image_decoded_rgb = math_ops.tensordot(source, kernel_back, axes=[[2], [0]])
        return image_decoded_rgb

    def resize_to(this, source_tensor, target_height_width):
        size_tensor = tf.convert_to_tensor(target_height_width)
        source_expanded = tf.expand_dims(source_tensor, 0)
        source_expanded_resized = tf.image.resize_bilinear(
                images=source_expanded,
                size=size_tensor)
        source_expanded_shrink = tf.squeeze(source_expanded_resized, [0])
        return source_expanded_shrink

    def crop_to(this,
                source_tensor, 
                offset_height,
                offset_width,
                target_height,
                target_width):
    
        #source_tensor = tf.Print(source_tensor, data=[tf.shape(source_tensor), tf.rank(source_tensor)], message="source_tensor ")
        source_expanded_cropped = tf.image.crop_to_bounding_box(
            source_tensor,
            offset_height,
            offset_width,
            target_height,
            target_width
            )
        #source_tensor = tf.Print(source_tensor, data=[tf.shape(source_expanded_cropped)], message="source_expanded_cropped ")
        return source_expanded_cropped
