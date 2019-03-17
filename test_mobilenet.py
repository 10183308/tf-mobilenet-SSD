# -*- coding: utf-8 -*-
from mobilenet_v1 import mobilenet_v1,mobilenet_v1_arg_scope

import tensorflow as tf
import cv2
import numpy as np
import pdb
import os
import sys

slim = tf.contrib.slim
IMAGE_SIZE = 512


def load_label():
    label=["others"]
    with open("data/ilsvrc_2012_labels.txt","r",encoding="utf-8") as f:
        lines = f.readlines()
        for l in lines:
            # pdb.set_trace()
            l = l.strip()
            arr = l.split(",")[0]
            label.append(arr)
    return label

def load_model(sess,CKPT,var_list=None):
    if not var_list:
        loader = tf.train.Saver()
    else:
        loader = tf.train.Saver(var_list)
    loader.restore(sess,CKPT)

def build_model(inputs):
    with slim.arg_scope(mobilenet_v1_arg_scope(is_training=False)):
        logits,end_points = mobilenet_v1(inputs,
                                is_training=False,
                                depth_multiplier=1.0,
                                num_classes=1000,
                                )
        scores = end_points["Predictions"]
        print(scores)
        output = tf.nn.top_k(scores,k=3,sorted=True)
        return output.indices,output.values

def main():
    # load data
    input_files = [os.path.join("data/images",f) for f in os.listdir("data/images") 
        if f.endswith(".jpg")]

    label = load_label()

    # checkpoint path
    CKPT = os.path.join("ckpt","mobilenet_v1_1.0_224.ckpt")

    # build graph
    g = tf.Graph()
    with g.as_default():
        # define placeholder
        inputs = tf.placeholder(tf.float32,[None,IMAGE_SIZE,IMAGE_SIZE,3])
        # inference
        logits, end_points = mobilenet_v1(inputs, 
                    is_training=False, depth_multiplier=1.0, num_classes=1001,
                    global_pool=True)

        for k,v in end_points.items():
            print(k,v)

        var_to_exclude = []
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            # load model
            # get vars that u need to restore
            var_list = slim.get_variables_to_restore(exclude=var_to_exclude)
            print("*" * 30)
            print("var_list")
            for vv in var_list:
                print(vv)

            pdb.set_trace()
            sess.run(init)
            load_model(sess,CKPT,var_list)

            for img_ in input_files:
                src = cv2.imread(img_)
                img = cv2.cvtColor(src,cv2.COLOR_BGR2RGB)
                img = cv2.resize(img,(IMAGE_SIZE,IMAGE_SIZE))
                img = img/ 255.0
                img = np.expand_dims(img,0)

                # cv2.imshow("img",src)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                pred = sess.run(end_points["Predictions"],feed_dict={inputs:img})
                # pdb.set_trace()
                print("Cat: {}, Score: {}".format(label[np.argmax(pred)],pred.max()))

if __name__ == '__main__':
    main()