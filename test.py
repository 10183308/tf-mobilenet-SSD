# -*- coding: utf-8 -*-
import tensorflow as tf

import ssd_mobilenet_v1

slim = tf.contrib.slim

# =========================================================================== #
# Dataset Flags.
# =========================================================================== #

tf.app.flags.DEFINE_integer('num_classes', 21, 
                                                    'Number of classes to use in the dataset.')

FLAGS = tf.app.flags.FLAGS

def main():
    with tf.Graph().as_default():
        
        # load dataset
        dataset = dataset_factory.get_dataset(FLAGS.dataset_name,
                                                                               FLAGS.dataset_split_name,
                                                                               FLAGS.dataset_dir)

        # Get the SSD network and its anchors.
        ssd_class = ssd_mobilenet_v1.SSDnet
        ssd_params = ssd_class.as_default._replace(num_classes=FLAGS.num_classes)
        ssd_net = ssd_class(ssd_params)
        ssd_shape = ssd_net.ssd_params.img_shape
        ssd_anchors = ssd_net.anchors(ssd_shape)


    return


if __name__ == '__main__':
    main()