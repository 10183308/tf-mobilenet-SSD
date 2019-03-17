import tensorflow as tf
import numpy as np
import ssd_mobilenet_v1 as ssd

from PIL import Image
import os
import pdb

def load_model(sess,CKPT,var_list=None):
    if not var_list:
        loader = tf.train.Saver()
    else:
        loader = tf.train.Saver(var_list)
    loader.restore(sess,CKPT)

def main():
    # set params
    IMAGE_SIZE = 512
    # checkpoint path
    CKPT = os.path.join("ckpt","mobilenet_v1_1.0_224.ckpt")
    # load data
    input_files = [os.path.join("data/images",f) for f in os.listdir("data/images") 
        if f.endswith(".jpg")]

    # build graph
    g = tf.Graph()
    with g.as_default():
        # set placeholder
        inputs = tf.placeholder(tf.float32,[None,IMAGE_SIZE,IMAGE_SIZE,3])

        # build model
        ssdnet = ssd.SSDnet()
        res = ssdnet.net(inputs)
        predictions = res[0]
        localisations = res[1]
        logits = res[2] 
        end_points = res[3]
        mobilenet_var_list = res[4]
        ssdnet.anchors(img_shape=(IMAGE_SIZE,IMAGE_SIZE))

        # print("*" * 30)
        # print("var list \n")
        # for vv in mobilenet_var_list:
        #     print(vv)

        # load weights
        init = tf.global_variables_initializer()
        with tf.Session(graph=g) as sess:
            sess.run(init)
            nodename = [n.name for n in tf.get_default_graph().as_graph_def().node]
            load_model(sess,CKPT,var_list = mobilenet_var_list)
            for img_ in input_files:
                im =Image.open(img_)
                im = im.resize((IMAGE_SIZE,IMAGE_SIZE))
                src = np.array(im)
                res = sess.run(logits[0],feed_dict={inputs:np.expand_dims(src,0)})
                print(res)
                pdb.set_trace()
                pass
                
if __name__ == '__main__':
    main()
