# -*- coding: utf-8 -*-

import tensorflow as tf 
from tensorflow.python.ops import control_flow_ops

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
    'gpu_memory_fraction', 0.1, 'GPU memory fraction to use.')

# learning rate flags.
tf.app.flags.DEFINE_string(
    'learning_rate_decay_type',
    'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')
tf.app.flags.DEFINE_float(
    "learning_rate_decay_factor",
    0.94,"Learning rate decay factor.")
tf.app.flags.DEFINE_float(
    "num_epochs_per_decay",2.0,
    "Number of epochs after which learning rate decays.")
tf.app.flags.DEFINE_float(
    "learning_rate",0.01,"Initial learning rate.")
tf.app.flags.DEFINE_float(
    "end_learning_rate",0.0001,"The minimum end learning rate used by polynomial decay learning rate.")
tf.app.flags.DEFINE_float(
    'moving_average_decay', 0.9999,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

# optimization flags, only support RMSprop in this version
tf.app.flags.DEFINE_float(
    "weight_decay",0.00004,"The weight decay on the model weights.")
tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')
tf.app.flags.DEFINE_string(
    "optimizer","rmsprop",
    "The name of the optimizer, only support `rmsprop`.")
tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')
tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')
tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')
tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')


# dataset flags
tf.app.flags.DEFINE_string(
    'dataset_name', 'pascalvoc_2007', 'The name of the dataset to load.')
tf.app.flags.DEFINE_integer(
    'num_classes', 21, 'Number of classes to use in the dataset.')
tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train', 'The name of the train/test split.')
tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_string(
    'preprocessing_name', "ssd_512_vgg", 'The name of the preprocessing to use.')
tf.app.flags.DEFINE_integer(
    'batch_size', 32, 'The number of samples in each batch.')
tf.app.flags.DEFINE_integer(
    'train_image_size', None, 'Train image size')
tf.app.flags.DEFINE_integer('max_number_of_steps', None,
                            'The maximum number of training steps.')

# fine-tuning flags
tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
'The path to a checkpoint from which to fine-tune.')
tf.app.flags.DEFINE_string(
    'trainable_scopes', None,
    'Comma-separated list of scopes to filter the set of variables to train.'
    'By default, None would train all the variables.')
tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', True,
'When restoring a checkpoint would ignore missing variables.')
tf.app.flags.DEFINE_boolean(
    'train_on_cpu', False,
'Set as `True` will make use of CPU for training.')


FLAGS = tf.app.flags.FLAGS

def main(_):
    if FLAGS.train_on_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"]="-1"    

    if not FLAGS.dataset_dir:
        raise ValueError("You must supply the dataset directory with --dataset-dir.")

    tf.logging.set_verbosity(tf.logging.DEBUG)

    g = tf.Graph()
    with g.as_default():
        # select the dataset
        dataset = dataset_factory.get_dataset(
            FLAGS.dataset_name, FLAGS.dataset_split_name,FLAGS.dataset_dir)
        
        # create global step, used for optimizer moving average decay
        with tf.device("/cpu:0"):
            global_step = tf.train.create_global_step()

        # pdb.set_trace()
        # get the ssd network and its anchors
        ssd_cls = ssd.SSDnet
        ssd_params = ssd_cls.default_params._replace(num_classes=FLAGS.num_classes)
        ssd_net = ssd_cls(ssd_params)
        image_size = ssd_net.params.img_shape

        ssd_anchors = ssd_net.anchors(img_shape=image_size)

        # select the preprocessing function
        preprocessing_name = FLAGS.preprocessing_name
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

                # pdb.set_trace()
                # preprocessing
                image,glabels,gbboxes = \
                            image_preprocessing_fn(image,
                                                                glabels,gbboxes,
                                                                out_shape=image_size,
                                                                data_format="NHWC")

                # encode groundtruth labels and bboxes
                gclasses,glocalisations,gscores= \
                    ssd_net.bboxes_encode(glabels,gbboxes,ssd_anchors)
                batch_shape = [1] + [len(ssd_anchors)] * 3

                # training batches and queue
                r = tf.train.batch(
                    tf_utils.reshape_list([image, gclasses, glocalisations, gscores]),
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
            predictions,localisations,logits,end_points,mobilenet_var_list = \
                    ssd_net.net(b_image,is_training=True)

        # add loss function
        ssd_net.losses(logits,localisations,
            b_gclasses,b_glocalisations,b_gscores,
            match_threshold=FLAGS.match_threshold,
            negative_ratio=FLAGS.negative_ratio,
            alpha=FLAGS.loss_alpha,
            label_smoothing=FLAGS.label_smoothing)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

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

        # configure the moving averages
        if FLAGS.moving_average_decay: # use moving average decay on weights variables
            moving_average_variables = slim.get_model_variables()
            variable_averages = tf.train.ExponentialMovingAverage(
                                FLAGS.moving_average_decay,global_step)
        else:
            moving_average_variables,variable_averages = None,None

        # configure the optimization procedure
        with tf.device("/cpu:0"):
            learning_rate = tf_utils.configure_learning_rate(FLAGS,
                dataset.num_samples,global_step)
            optimizer = tf_utils.configure_optimizer(FLAGS,learning_rate)
            summaries.add(tf.summary.scalar("learning_rate",learning_rate))

        if FLAGS.moving_average_decay:
            # update ops executed by trainer
            update_ops.append(variable_averages.apply(moving_average_variables))

        # get variables to train
        variables_to_train = tf_utils.get_variables_to_train(FLAGS)

        # return a train tensor and summary op
        total_losses = tf.get_collection(tf.GraphKeys.LOSSES)
        total_loss = tf.add_n(total_losses,name="total_loss")
        summaries.add(tf.summary.scalar("total_loss",total_loss))

        # create gradient updates
        grads = optimizer.compute_gradients(total_loss,var_list=variables_to_train)
        grad_updates = optimizer.apply_gradients(grads,global_step=global_step)
        update_ops.append(grad_updates)

        # create train op
        update_op = tf.group(*update_ops)
        train_tensor = control_flow_ops.with_dependencies([update_op],total_loss,
            name="train_op")
        
        # merge all summaries together
        summary_op = tf.summary.merge(list(summaries),name="summary_op")

        # start training
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)
        config = tf.ConfigProto(log_device_placement=False,
                                gpu_options=gpu_options)
        saver = tf.train.Saver(max_to_keep=2,
            keep_checkpoint_every_n_hours=1.0,
            write_version=2,
            pad_step_number=False)

        # create initial assignment op
        init_assign_op,init_feed_dict = slim.assign_from_checkpoint(
            FLAGS.checkpoint_path,mobilenet_var_list,
            ignore_missing_vars=FLAGS.ignore_missing_vars)
        

        # create an initial assignment function
        for k,v in init_feed_dict.items():
            if "global_step" in k.name:
                g_step = k

        init_feed_dict[g_step] = 0 # change the global_step to zero.
        init_fn = lambda sess: sess.run(init_assign_op,init_feed_dict)

        # run training
        slim.learning.train(train_tensor,logdir=FLAGS.train_dir,
            init_fn=init_fn,
            summary_op=summary_op,
            number_of_steps=FLAGS.max_number_of_steps,
            save_summaries_secs=FLAGS.save_summaries_secs,
            save_interval_secs=FLAGS.save_interval_secs,
            session_config=config,
            saver=saver,
            )


        # slim.learning.train(
        #     train_tensor,
        #     logdir=FLAGS.train_dir,
        #     init_fn =tf_utils.get_init_fn(FLAGS,mobilenet_var_list),
        #     summary_op=summary_op,
        #     global_step=global_step,
        #     number_of_steps=FLAGS.max_number_of_steps,
        #     log_every_n_steps=FLAGS.log_every_n_steps,
        #     save_summaries_secs=FLAGS.save_summaries_secs,
        #     saver=saver,
        #     save_interval_secs =FLAGS.save_interval_secs,
        #     session_config=config,
        #     sync_optimizer=None)

if __name__ == '__main__':
    tf.app.run()









