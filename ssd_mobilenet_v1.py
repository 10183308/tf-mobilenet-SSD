# -*- coding: utf-8 -*-

"""
    tf-Mobilenet-SSD
    Definition of 512 Mobilenet-based SSD network.
    Ref: https://github.com/balancap/SSD-Tensorflow/blob/master/nets/ssd_vgg_512.py
"""

import numpy as np
import tensorflow as tf

from collections import namedtuple
import math
import pdb

from mobilenet_v1 import mobilenet_v1
from layers import l2_normalization,channel_to_last,pad2d


slim = tf.contrib.slim

SSDParams = namedtuple('SSDParameters', ['img_shape',
                             'num_classes',
                             'no_annotation_label',
                             'feat_layers',
                             'feat_shapes',
                             'anchor_size_bounds',
                             'anchor_sizes',
                             'anchor_ratios',
                             'anchor_steps',
                             'anchor_offset',
                             'normalizations',
                             'prior_scaling'])

class SSDnet(object):
    """Implementation of the SSD Mobilenet-based 512 network.
    The default features layers with 512x512 image input are:
      conv2d_11_pointwise ==> 32 x 32
      conv7 ==> 16 x 16
      conv8 ==> 18 x 8
      conv9 ==> 4 x 4
      conv10 ==> 2 x 2
      conv11 ==> 1 x 1
    The default image size used to train this network is 512x512.
    """
    default_params = SSDParams(
      img_shape=(512,512),
      num_classes=21,
      no_annotation_label=21,
      feat_layers=["Conv2d_11_pointwise","block7","block8","block9","block10","block11"], # 6 feat layers
      feat_shapes=[(32,32),(16,16),(8,8),(4,4),(2,2),(1,1)],
      anchor_size_bounds=[0.1,0.9],
      anchor_sizes=[(25.6,51.2),
                                (51.2, 133.12), 
                                (133.12, 215.04),
                                (215.04, 296.96),
                                (296.96, 378.88),
                                (378.88, 460.8)], # size of prior boxes, computed via `compute_anchor_sizes`.
      anchor_ratios=[[2, .5],
                   [2, .5, 3, 1./3],
                   [2, .5, 3, 1./3],
                   [2, .5, 3, 1./3],
                   [2, .5],
                   [2, .5]], # ratio of width/height.
      anchor_steps=[16, 32, 64, 128, 256, 512], # deprecated, see `ssd_anchor_one_layer` for details.
      anchor_offset=0.5,
      normalizations=[20, -1, -1, -1, -1, -1],
      prior_scaling=[0.1, 0.1, 0.2, 0.2], # ?
      )
    # total num of boxes = 32*32*4 + 16*16*6 + 8*8*6 + 4*4*6 +2*2*4 + 1*1*4 = 6132.

    def __init__(self,params=None):
        """Init the SSD net with some parameters. Use the default ones
        if none provided
        """
        if isinstance(params,SSDParams):
            self.params = params
        else:
            self.params = SSDnet.default_params

    def net(self, inputs,
                  is_training=True,
                  update_feat_shapes=True,
                  dropout_keep_prob=0.5,
                  prediction_fn=slim.softmax,
                  reuse=None,
                  scope="ssd_512_mobilenet",):
          """Network definition.
          """
          # Original Mobilenet_v1 blocks.
          _,end_points = mobilenet_v1(inputs,
                          is_training=is_training,
                          depth_multiplier=1.0,
                          global_pool=True)
          
          net = end_points["Conv2d_13_pointwise"] # [None,16,16,256]
          var_to_exclude = ["MobilenetV1/Logits"]
          mobilenet_var_list = slim.get_variables_to_restore(exclude=var_to_exclude)

          res = ssd_net_base(net,
                    num_classes=self.params.num_classes,
                    feat_layers=self.params.feat_layers,
                    anchor_sizes=self.params.anchor_sizes,
                    anchor_ratios=self.params.anchor_ratios,
                    normalizations=self.params.normalizations,
                    is_training=is_training,
                    dropout_keep_prob=dropout_keep_prob,
                    prediction_fn=prediction_fn,
                    reuse=reuse,
                    scope=scope,
                    end_points = end_points)
          
          predictions = res[0]
          localisations = res[1]
          logits = res[2] 
          end_points = res[3]

          # Update feature shapes when the input image size changes with zoom tricks.
          if update_feat_shapes:
            shapes = ssd_feat_shapes_from_net(predictions,self.params.feat_shapes)
            self.params = self.params._replace(feat_shapes=shapes)
          return predictions,localisations,logits,end_points,mobilenet_var_list

    def anchors(self,img_shape,dtype=np.float32):
        """Compute the default anchor boxes, given an image shape.
        """
        layers_anchors = []

        anchor_sizes = compute_anchor_sizes(img_shape,
                                                self.params.feat_shapes,
                                                self.params.anchor_size_bounds)
        # pdb.set_trace()
        for i,s in enumerate(self.params.feat_shapes):
            anchor_bboxes = ssd_anchor_one_layer(img_shape,
                                              feat_shape = s,
                                              sizes = self.params.anchor_sizes[i],
                                              ratios = self.params.anchor_ratios[i],
                                              step = self.params.anchor_steps[i],
                                              offset = self.params.anchor_offset,
                                              dtype=np.float32,
                                              )

            layers_anchors.append(anchor_bboxes)
        print(layers_anchors)
        
        return layers_anchors

    def bboxes_encode(self,labels,bboxes,anchors,scope=None):
        return

SSDNet = SSDnet()

def compute_anchor_sizes(img_shape,feat_shapes,anchor_size_bounds):
    s_min,s_max = anchor_size_bounds
    s_k = s_min/2
    anchor_sizes = []
    m = len(feat_shapes)
    for k,feat_shape in enumerate(feat_shapes):
        s_k1 = s_min + (s_max-s_min)/(m-1) * k
        anchor_sizes.append((np.float32(s_k*img_shape[0]),np.float32(s_k1*img_shape[0])))
        s_k = s_k1
    return anchor_sizes

def ssd_net_base(inputs,
            num_classes=SSDNet.default_params.num_classes,
            feat_layers=SSDNet.default_params.feat_layers,
            anchor_sizes=SSDNet.default_params.anchor_sizes,
            anchor_ratios=SSDNet.default_params.anchor_ratios,
            normalizations=SSDNet.default_params.normalizations,
            is_training=True,
            dropout_keep_prob=0.5,
            prediction_fn=slim.softmax,
            reuse=None,
            scope='ssd_512_mobilenet',
            end_points={}):
        """SSD net definition.
        """
        with tf.variable_scope(scope, 'ssd_512_mobilenet', [inputs], reuse=reuse):
            # Additional SSD blocks.
            # Block 6: dilation
            net = slim.conv2d(inputs,1024,[3,3],rate=6,scope="conv6")
            end_points["block6"] = net

            # Block 7: 1x1 conv.
            net = slim.conv2d(net, 1024, [1, 1], scope='conv7')
            end_points['block7'] = net  
            
            # Block 8/9/10/11: 1x1 and 3x3 convolutions stride 2 (except lasts).
            end_point = 'block8'
            with tf.variable_scope(end_point):
                net = slim.conv2d(net, 256, [1, 1], scope='conv1x1')
                net = pad2d(net, pad=(1, 1))
                net = slim.conv2d(net, 512, [3, 3], stride=2, scope='conv3x3', padding='VALID')
            end_points[end_point] = net
            end_point = 'block9'
            with tf.variable_scope(end_point):
                net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
                net = pad2d(net, pad=(1, 1))
                net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3x3', padding='VALID')
            end_points[end_point] = net
            end_point = 'block10'
            with tf.variable_scope(end_point):
                net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
                net = pad2d(net, pad=(1, 1))
                net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3x3', padding='VALID')
            end_points[end_point] = net
            end_point = 'block11'
            with tf.variable_scope(end_point):
                net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
                net = pad2d(net, pad=(1, 1))
                net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3x3', padding='VALID')
            end_points[end_point] = net
            # Prediction and localisation layers.
            predictions = []
            logits = []
            localisations = []
            for i,layer in enumerate(feat_layers):
              with tf.variable_scope(layer+"_box"):
                p,l = ssd_multibox_layer(end_points[layer],
                                                              num_classes,
                                                              anchor_sizes[i],
                                                              anchor_ratios[i],
                                                              normalizations[i])
              predictions.append(prediction_fn(p))
              logits.append(p)
              localisations.append(l)

            for k,v in end_points.items():
              print(k,v)

            return predictions,localisations,logits,end_points

def ssd_multibox_layer(inputs,
                       num_classes,
                       sizes,
                       ratios=[1],
                       normalization=-1,
                       bn_normalization=False):
    """Construct a multibox layer, return a class and localization predictions.
    """
    net  = inputs
    if normalization > 0: # do l2 normalization
      net = l2_normalization(net,scaling=True)
    # number of anchors.
    num_anchors = len(sizes) + len(ratios)
    
    # location.
    num_loc_pred = num_anchors * 4
    loc_pred = slim.conv2d(net,num_loc_pred,[3,3],activation_fn=None,scope="conv_loc")
    loc_pred = channel_to_last(loc_pred)
    loc_pred = tf.reshape(loc_pred,tensor_shape(loc_pred,4)[:-1] + [num_anchors,4])

    # class prediction.
    num_cls_pred = num_anchors * num_classes
    cls_pred = slim.conv2d(net,num_cls_pred,[3,3],activation_fn=None,scope="conv_cls")
    cls_pred = channel_to_last(cls_pred)
    cls_pred = tf.reshape(cls_pred,tensor_shape(cls_pred,4)[:-1]+ [num_anchors,num_classes])
    return cls_pred,loc_pred

def ssd_anchor_one_layer(img_shape,
                         feat_shape,
                         sizes,
                         ratios,
                         step,
                         offset=0.5,
                         dtype=np.float32):
      """Computer SSD default anchor boxes for one feature layer.
      Determine the relative position grid of the centers, and the relative
      width and height.
      Arguments:
        feat_shape: Feature shape, used for computing relative position grids;
        size: Absolute reference sizes;
        ratios: Ratios to use on these features;
        img_shape: Image shape, used for computing height, width relatively to the
          former;
        offset: Grid offset.
      Return:
        y, x, h, w: Relative x and y grids (centroid coordinates), and height and width of the boxes.
      """
      # Weird SSD-Caffe computation using steps values...
      # y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
      # y = (y.astype(dtype) + offset) * step / img_shape[0]
      # x = (x.astype(dtype) + offset) * step / img_shape[1]

      # Compute the position grid: simple way without step.
      y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
      y = (y.astype(dtype) + offset) / feat_shape[0]
      x = (x.astype(dtype) + offset) / feat_shape[1]

      # Expand dims to support easy broadcasting.
      y = np.expand_dims(y, axis=-1)
      x = np.expand_dims(x, axis=-1)

      # Compute relative height and width.
      # Tries to follow the original implementation of SSD for the order.
      pdb.set_trace()
      num_anchors = len(sizes) + len(ratios)
      h = np.zeros((num_anchors, ), dtype=dtype)
      w = np.zeros((num_anchors, ), dtype=dtype)
      # Add first anchor boxes with ratio=1, it is a square.
      h[0] = sizes[0] / img_shape[0]
      w[0] = sizes[0] / img_shape[1]
      di = 1
      if len(sizes) > 1:
          h[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[0]
          w[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[1]
          di += 1
      for i, r in enumerate(ratios):
          h[i+di] = sizes[0] / img_shape[0] / math.sqrt(r)
          w[i+di] = sizes[0] / img_shape[1] * math.sqrt(r)
      return y, x, h, w

def tensor_shape(x,rank=3):
  if x.get_shape().is_fully_defined():
    return x.get_shape().as_list()
  else:
    static_shape = x.get_shape().with_rank(rank).as_list()
    dynamic_shape = tf.unstack(tf.shape(x), rank)
    return [s if s is not None else d for s, d in zip(static_shape, dynamic_shape)]


def ssd_feat_shapes_from_net(predictions, default_shapes=None):
    """Try to obtain the feature shapes from the prediction layers.
    Return:
      list of feature shapes. Default values if predictions shape not fully
      determined.
    """
    feat_shapes = []
    for l in predictions:
        shape = l.get_shape().as_list()[1:4]
        if None in shape:
            return default_shapes
        else:
            feat_shapes.append(shape)
    return feat_shapes