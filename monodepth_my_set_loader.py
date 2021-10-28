from __future__ import absolute_import, division, print_function
import tensorflow as tf


def string_length_tf(t):
  return tf.py_func(len, [t], [tf.int32])


class MonodepthDataloaderMy(object):
    """monodepth dataloader"""

    def __init__(self, data_path, filenames_file, params, dataset, mode, noShuffle=False):
        self.data_path = data_path
        self.params = params
        self.dataset = dataset
        self.mode = mode

        self.left_image_batch  = None
        self.right_image_batch = None
        self.semantic_image_batch = None
        self.valid_image_batch = None

        input_queue = tf.train.string_input_producer([filenames_file], shuffle=False)
        line_reader = tf.TextLineReader()
        _, line = line_reader.read(input_queue)

        split_line = tf.string_split([line]).values

        # we load only one image for test, except if we trained a stereo model
        if mode == 'recognize' and not self.params.do_stereo:
            left_image_path  = tf.string_join([self.data_path, split_line[0]])
            left_image_o  = self.read_image(left_image_path)
#            semantic_image_path = tf.string_join([self.data_path, split_line[2]])
#            semantic_image_o = self.read_semantic_gt(semantic_image_path)
        else:
            left_image_path  = tf.string_join([self.data_path, split_line[0]])
            right_image_path = tf.string_join([self.data_path, split_line[1]])
            semantic_image_path = tf.string_join([self.data_path, split_line[2]])

            left_image_o  = self.read_image(left_image_path)
            right_image_o = self.read_image(right_image_path)
            semantic_image_o, valid_image_o = self.read_semantic_gt(semantic_image_path)

        if mode == 'recognize':
            self.left_image_batch = tf.stack([left_image_o,  tf.image.flip_left_right(left_image_o)],  0)
            self.left_image_batch.set_shape([2, None, None, 3])
#            self.semantic_image_batch = tf.stack([semantic_image_o,  tf.image.flip_left_right(semantic_image_o)],  0)
#            self.semantic_image_batch.set_shape( [2, None, None, 1])



    def read_semantic_gt(self, image_path):
        # tf.decode_image does not return the image size, this is an ugly workaround to handle both jpeg and png
        path_length = string_length_tf(image_path)[0]
        file_extension = tf.substr(image_path, path_length - 3, 3)
        file_cond = tf.equal(file_extension, 'png')

        image = tf.cond(file_cond, lambda: tf.image.decode_png(tf.read_file(image_path)),
                        lambda: tf.zeros([self.params.height, self.params.width, 1], tf.uint8))

        # if the dataset is cityscapes, we crop the last fifth to remove the car hood
        if self.dataset == 'cityscapes':
            o_height = tf.shape(image)[0]
            crop_height = (o_height * 4) // 5
            image = image[:crop_height, :, :]

        image = tf.to_int32(tf.image.resize_images(image, [self.params.height, self.params.width],
                                                   tf.image.ResizeMethod.NEAREST_NEIGHBOR))
        valid = tf.cond(file_cond, lambda: tf.ones([self.params.height, self.params.width, 1], tf.float32),
                        lambda: tf.zeros([self.params.height, self.params.width, 1], tf.float32))

        return image, valid

    def read_image(self, image_path):
        # tf.decode_image does not return the image size, this is an ugly workaround to handle both jpeg and png
        path_length = string_length_tf(image_path)[0]
        file_extension = tf.substr(image_path, path_length - 3, 3)
        file_cond = tf.equal(file_extension, 'jpg')

        image = tf.cond(file_cond, lambda: tf.image.decode_jpeg(tf.read_file(image_path)),
                        lambda: tf.image.decode_png(tf.read_file(image_path)))

        # if the dataset is cityscapes, we crop the last fifth to remove the car hood
        if self.dataset == 'cityscapes':
            o_height = tf.shape(image)[0]
            crop_height = (o_height * 4) // 5
            image = image[:crop_height, :, :]

        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize_images(image, [self.params.height, self.params.width], tf.image.ResizeMethod.AREA)

        return image