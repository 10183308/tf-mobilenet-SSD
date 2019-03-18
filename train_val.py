# -*- coding: utf-8 -*-
"""Generic training script that trains a SSD model using a given dataset."""
from mobilenet_v1 import mobilenet_v1

import numpy as np
import tensorflow as tf
import cv2

import pdb
import argparse
import os
import sys

def main(args):

    input_files = [os.path.join(args.train_dir,f) for f in os.listdir(args.train_dir)]

    # build network
    inputs = tf.placeholder(tf.float32,shape=[None,None,None,3])
    logits,end_points = mobilenet_v1(inputs,
                                                                num_classes=2,
                                                                dropout_keep_prob=0.999,
                                                                is_training=True,
                                                                min_depth=8,
                                                                depth_multiplier=1.0,
                                                                conv_defs=None,
                                                                prediction_fn=tf.contrib.layers.softmax,
                                                                spatial_squeeze=True,
                                                                reuse=None,
                                                                scope="MobilenetV1",
                                                                global_pool=True)
    
    init  = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        src = cv2.imread(input_files[0])
        # pdb.set_trace()
        res = sess.run(logits,feed_dict={inputs:np.expand_dims(src,0)})

    pdb.set_trace()

    # src = cv2.imread("data/img_82.jpg")
    # cv2.namedWindow("input image",cv2.WINDOW_AUTOSIZE)
    # cv2.imshow("input image",src)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # pdb.set_trace()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("train_dir",type=str,
            help="input images directory.")

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))




