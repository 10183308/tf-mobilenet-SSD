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

        # get the SSD network and its anchors.
        ssd_net = ssd.SSD_net()
        predictions,localisations,logits,end_points =  \
                    ssd_net.net(inputs,is_training=False)

        # get its anchors.
        ssd_anchors = ssd_net.anchors(img_shape=(IMAGE_SIZE,IMAGE_SIZE))

        # add loss functions.
        ssd_net.losses(logits,localisations,b_gclasses,b_glocalisations,b_gscores)

        # detected objects from SSD output
        localisations = ssd_net.bboxes_decode(localisations,ssd_anchors)
        rscores,rbboxes = \
            ssd_net.detected_bboxes(predictions,localisations,
                    select_threshold=0.01,
                    nms_threshold=0.45,
                    clipping_bbox=None,
                    top_k=400,
                    keep_top_k=200)

        # compute TP and FP statistics.
        num_gbboxes,tp,fp,rscores = \
            bboxes_matching_batch(rscores.keys(),rscores,rbboxes,
                b_glabels,b_gbboxes,b_gdifficults,
                matching_threshold=0.5)


        # evaluation metrics
        with tf.device("/device:CPU:0"):
            dict_metrics = {}
            # add all losses
            for loss in tf.get_collection(tf.GraphKeys.LOSSES):
                dict_metrics[loss.op.name] = slim.metrics.streaming_mean(loss)
            # extra losses
            for loss in tf.get_collection("EXTRA_LOSSES"):
                dict_metrics[loss.op.name] = slim.metrics.streaming_mean(loss)



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
