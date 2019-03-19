# -*- coding: utf-8 -*-

import tensorflow as tf 

import ssd_mobilenet_v1 as ssd
from datasets import dataset_factory
from preprocessing import preprocessing_factory
import tf_utils

import os
import pdb

slim = tf.contrib.slim

# ssd network flags
tf.app.flags.DEFINE_float(
    'match_threshold', 0.5, 'Matching threshold in the loss function.')
tf.app.flags.DEFINE_float(
    'loss_alpha', 1., 'Alpha parameter in the loss function.')
tf.app.flags.DEFINE_float(
    'negative_ratio', 3., 'Negative ratio in the loss function.')

# General flags
tf.app.flags.DEFINE_integer(
    'num_readers', 4,
    'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_string(
    'train_dir', './logs',
    'Directory where checkpoints and event logs are written to.')
tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')
tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are print.')
tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 600,
    'The frequency with which summaries are saved, in seconds.')
tf.app.flags.DEFINE_integer(
    'save_interval_secs', 600,
    'The frequency with which the model is saved, in seconds.')
tf.app.flags.DEFINE_float(
    'gpu_memory_fraction', 0.8, 'GPU memory fraction to use.')

# optimization flags
tf.app.flags.DEFINE_float(
    "weight_decay",0.00004,"The weight decay on the model weights.")
tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')


# dataset flags
tf.app.flags.DEFINE_string(
    'dataset_name', 'pascalvoc_', 'The name of the dataset to load.')
tf.app.flags.DEFINE_integer(
    'num_classes', 21, 'Number of classes to use in the dataset.')
tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train', 'The name of the train/test split.')
tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_string(
    'model_name', 'ssd_300_mobilenet', 'The name of the architecture to train.')
tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')
tf.app.flags.DEFINE_integer(
    'batch_size', 32, 'The number of samples in each batch.')
tf.app.flags.DEFINE_integer(
    'train_image_size', None, 'Train image size')
tf.app.flags.DEFINE_integer('max_number_of_steps', None,
                            'The maximum number of training steps.')


FLAGS = tf.app.flags.FLAGS

def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError("You must supply the dataset directory with --dataset-dir.")

    tf.logging.set_verbosity(tf.logging.DEBUG)

    g = tf.Graph()
    with g.as_default():
        # select the dataset
        dataset = dataset_factory.get_dataset(
            FLAGS.dataset_name, FLAGS.dataset_split_name,FLAGS.dataset_dir)
        
        # get the ssd network and its anchors
        ssd_cls = ssd.SSDnet
        ssd_params = ssd_cls.default_params._replace(num_classes=FLAGS.num_classes)
        ssd_net = ssd_cls(ssd_params)
        image_size = ssd_net.params.img_shape

        ssd_anchors = ssd_net.anchors(img_shape=(image_size,image_size))

        # select the preprocessing function
        preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
                        preprocessing_name,is_training=True)

        tf_utils.print_configuration(FLAGS.__flags,ssd_params,
            dataset.data_sources,FLAGS.train_dir)

        # create a dataset provider and batches.
        with tf.device("/cpu:0"):
            with tf.name_scope(FLAGS.dataset_name+"_data_provider"):
                provider = slim.dataset_data_provider.DatasetDataProvider(
                    dataset,
                    num_readers=FLAGS.num_readers,
                    common_queue_capacity=20*FLAGS.batch_size,
                    common_queue_min=10*FLAGS.batch_size,
                    shuffle=True)
                # get for ssd network: image,labels,bboxes
                [image,shape,glabels,gbboxes] = provider.get(["image","shape",
                                        "object/label",
                                        "object/bbox"])
                # preprocessing
                image,glabels,gbboxes = \
                            image_preprocessing_fn(image,
                                                                glabels,gbboxes,
                                                                out_shape=image_size,
                                                                data_format="NCHW")

                # encode groundtruth labels and bboxes
                gclasses,glocalisations,gscores= \
                    ssd_net.bboxes_encode(glbales,gbboxes,ssd_anchors)
                batch_shape = [1] + [len(ssd_anchors)] * 3

                # training batches and queue
                r = tf.train.batch(
                    tf_utils.reshape_list([image.gclasses,glocalisations,gscores]),
                    batch_size=FLAGS.batch_size,
                    num_threads=FLAGS.num_preprocessing_threads,
                    capacity=5*FLAGS.batch_size)
                b_image,b_gclasses,b_glocalisations,b_gscores = \
                    tf_utils.reshape_list(r,batch_shape)

                # prefetch queue
                batch_queue = slim.prefetch_queue.prefetch_queue(
                    tf_utils.reshape_list([b_image,b_gclasses,b_glocalisations,b_gscores]),
                    capacity = 8)

        # dequeue batch
        b_image, b_gclasses, b_glocalisations, b_gscores = \
                tf_utils.reshape_list(batch_queue.dequeue(), batch_shape)

        # gather initial summaries     
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
        arg_scope = ssd_net.arg_scope(weight_decay=FLAGS.weight_decay)
        with slim.arg_scope(arg_scope):
            predictions,localisations,logits,end_points = \
                    ssd_net.net(b_image,is_training=True)

        # add loss function
        ssd_net.losses(logits,localisations,
            b_gclasses,b_glocalisations,b_gscores,
            match_threshold=FLAGS.match_threshold,
            negative_ratio=FLAGS.negative_ratio,
            alpha=FLAGS.loss_alpha,
            label_smoothing=FLAGS.label_smoothing)

        update_ops = tf.get_collections(tf.GraphKeys.UPDATA_OPS)

        # add summaries for end_points
        for end_point in end_points:
            x = end_points[end_point]
            summaries.add(tf.summary.histogram("activations/"+end_point,x))
            summaries.add(tf.summary.scalar("sparsity/"+end_point,
                    tf.nn.zero_fraction(x)))

        # add summaries for losses and extra losses
        for loss in tf.get_collection(tf.GraphKeys.LOSSES):
            summaries.add(tf.summary.scalar(loss.op.name,loss))
        for loss in tf.get_collection("EXTRA_LOSSES"):
            summaries.add(tf.summary.scalar(loss.op.name,loss))

        # add summaries for variables
        for var in slim.get_model_variables():
            summaries.add(tf.summary.histogram(var.op.name,var))









                




        # add loss functions
        ssd_net.losses(logits,localisations,b_gclasses,b_glocalisations,b_gscores)

if __name__ == '__main__':
    tf.app.run()









