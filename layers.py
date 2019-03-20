# -*- coding: utf-8 -*-
# Ref: https://github.com/balancap/SSD-Tensorflow/blob/master/nets/custom_layers.py
import tensorflow as tf

from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.ops import nn
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
import math
import numpy as np

import tf_extended as tfe

def abs_smooth(x):
    """Smoothed absolute function. Useful to compute an L1 smooth error.
    Define as:
        x^2 / 2         if abs(x) < 1
        abs(x) - 0.5    if abs(x) > 1
    We use here a differentiable definition using min(x) and abs(x). Clearly
    not optimal, but good enough for our purpose!
    """
    absx = tf.abs(x)
    minx = tf.minimum(absx, 1)
    r = 0.5 * ((absx - 1) * minx + absx)
    return r

@add_arg_scope
def l2_normalization(
        inputs,
        scaling=False,
        scale_initializer=init_ops.ones_initializer(),
        reuse=None,
        variables_collections=None,
        outputs_collections=None,
        data_format='NHWC',
        trainable=True,
        scope=None):
    """Implement L2 normalization on every feature (i.e. spatial normalization).
    Should be extended in some near future to other dimensions, providing a more
    flexible normalization framework.
    Args:
      inputs: a 4-D tensor with dimensions [batch_size, height, width, channels].
      scaling: whether or not to add a post scaling operation along the dimensions
        which have been normalized.
      scale_initializer: An initializer for the weights.
      reuse: whether or not the layer and its variables should be reused. To be
        able to reuse the layer scope must be given.
      variables_collections: optional list of collections for all the variables or
        a dictionary containing a different list of collection per variable.
      outputs_collections: collection to add the outputs.
      data_format:  NHWC or NCHW data format.
      trainable: If `True` also add variables to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
      scope: Optional scope for `variable_scope`.
    Returns:
      A `Tensor` representing the output of the operation.
    """

    with variable_scope.variable_scope(
            scope, 'L2Normalization', [inputs], reuse=reuse) as sc:
        inputs_shape = inputs.get_shape()
        inputs_rank = inputs_shape.ndims
        dtype = inputs.dtype.base_dtype
        if data_format == 'NHWC':
            # norm_dim = tf.range(1, inputs_rank-1)
            norm_dim = tf.range(inputs_rank-1, inputs_rank)
            params_shape = inputs_shape[-1:]
        elif data_format == 'NCHW':
            # norm_dim = tf.range(2, inputs_rank)
            norm_dim = tf.range(1, 2)
            params_shape = (inputs_shape[1])

        # Normalize along spatial dimensions.
        outputs = nn.l2_normalize(inputs, norm_dim, epsilon=1e-12)
        # Additional scaling.
        if scaling:
            scale_collections = utils.get_variable_collections(
                variables_collections, 'scale')
            scale = variables.model_variable('gamma',
                                             shape=params_shape,
                                             dtype=dtype,
                                             initializer=scale_initializer,
                                             collections=scale_collections,
                                             trainable=trainable)
            if data_format == 'NHWC':
                outputs = tf.multiply(outputs, scale)
            elif data_format == 'NCHW':
                scale = tf.expand_dims(scale, axis=-1)
                scale = tf.expand_dims(scale, axis=-1)
                outputs = tf.multiply(outputs, scale)
                # outputs = tf.transpose(outputs, perm=(0, 2, 3, 1))

        return utils.collect_named_outputs(outputs_collections,
            sc.original_name_scope, outputs)

@add_arg_scope
def channel_to_last(inputs,
                    data_format='NHWC',
                    scope=None):
    """Move the channel axis to the last dimension. Allows to
    provide a single output format whatever the input data format.
    Args:
      inputs: Input Tensor;
      data_format: NHWC or NCHW.
    Return:
      Input in NHWC format.
    """
    with tf.name_scope(scope, 'channel_to_last', [inputs]):
        if data_format == 'NHWC':
            net = inputs
        elif data_format == 'NCHW':
            net = tf.transpose(inputs, perm=(0, 2, 3, 1))
    return net
    
@add_arg_scope
def pad2d(inputs,pad=(0,0),trainable=True,scope=None):
  with tf.name_scope(scope,"pad2d",[inputs]):
    paddings = [[0, 0], [pad[0], pad[0]], [pad[1], pad[1]], [0, 0]]
    net = tf.pad(inputs,paddings,mode="CONSTANT")
    return net


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

def tf_ssd_bboxes_encode_layer(labels,
                               bboxes,
                               anchors_layer,
                               num_classes,
                               no_annotation_label,
                               ignore_threshold=0.5,
                               prior_scaling=[0.1, 0.1, 0.2, 0.2],
                               dtype=tf.float32):
    """Encode groundtruth labels and bounding boxes using SSD anchors from
    one layer.
    Find bboxes matching groundtruth objects, each groundtruth has at least one bbox. 
    Arguments:
      labels: 1D Tensor(int64) containing groundtruth labels;
      bboxes: Nx4 Tensor(float) with bboxes relative coordinates;
      anchors_layer: Numpy array with layer anchors;
      matching_threshold: Threshold for positive match with groundtruth bboxes;
      prior_scaling: Scaling of encoded coordinates.
    Return:
      (target_labels, target_localizations, target_scores): Target Tensors.
    """
    # Anchors coordinates
    yref, xref, href, wref = anchors_layer
    ymin = yref - href / 2.
    xmin = xref - wref / 2.
    ymax = yref + href / 2.
    xmax = xref + wref / 2.
    # Anchors volume
    vol_anchors = (xmax - xmin) * (ymax - ymin)

    # Initialize tensors...
    shape = (yref.shape[0], yref.shape[1], href.size)
    feat_labels = tf.zeros(shape, dtype=tf.int64)
    feat_scores = tf.zeros(shape, dtype=dtype)

    feat_ymin = tf.zeros(shape, dtype=dtype)
    feat_xmin = tf.zeros(shape, dtype=dtype)
    feat_ymax = tf.ones(shape, dtype=dtype)
    feat_xmax = tf.ones(shape, dtype=dtype)

    def jaccard_with_anchors(bbox):
        """Compute jaccard score between a box and the anchors.
        """
        int_ymin = tf.maximum(ymin, bbox[0])
        int_xmin = tf.maximum(xmin, bbox[1])
        int_ymax = tf.minimum(ymax, bbox[2])
        int_xmax = tf.minimum(xmax, bbox[3])
        h = tf.maximum(int_ymax - int_ymin, 0.)
        w = tf.maximum(int_xmax - int_xmin, 0.)
        # Volumes.
        inter_vol = h * w
        union_vol = vol_anchors - inter_vol \
            + (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        jaccard = tf.div(inter_vol, union_vol)
        return jaccard

    def intersection_with_anchors(bbox):
        """Compute intersection between score a box and the anchors.
        """
        int_ymin = tf.maximum(ymin, bbox[0])
        int_xmin = tf.maximum(xmin, bbox[1])
        int_ymax = tf.minimum(ymax, bbox[2])
        int_xmax = tf.minimum(xmax, bbox[3])
        h = tf.maximum(int_ymax - int_ymin, 0.)
        w = tf.maximum(int_xmax - int_xmin, 0.)
        inter_vol = h * w
        scores = tf.div(inter_vol, vol_anchors)
        return scores

    def condition(i, feat_labels, feat_scores,
                  feat_ymin, feat_xmin, feat_ymax, feat_xmax):
        """Condition: check label index.
        """
        # tf.less return the truth value of (x<y) element-wise
        r = tf.less(i, tf.shape(labels))
        return r[0]

    def body(i, feat_labels, feat_scores,
             feat_ymin, feat_xmin, feat_ymax, feat_xmax):
        """Body: update feature labels, scores and bboxes.
        Follow the original SSD paper for that purpose:
          - assign values when jaccard > 0.5;
          - only update if beat the score of other bboxes.
        """
        # Jaccard score.
        label = labels[i]
        bbox = bboxes[i]
        jaccard = jaccard_with_anchors(bbox) # compute jaccard overlap score


        # Mask: check threshold + scores + no annotations + num_classes.
        mask = tf.greater(jaccard, feat_scores)
        # mask = tf.logical_and(mask, tf.greater(jaccard, matching_threshold))
        mask = tf.logical_and(mask, feat_scores > -0.5)
        mask = tf.logical_and(mask, label < num_classes)
        imask = tf.cast(mask, tf.int64)
        fmask = tf.cast(mask, dtype)
        # Update values using mask.
        feat_labels = imask * label + (1 - imask) * feat_labels
        feat_scores = tf.where(mask, jaccard, feat_scores)

        feat_ymin = fmask * bbox[0] + (1 - fmask) * feat_ymin
        feat_xmin = fmask * bbox[1] + (1 - fmask) * feat_xmin
        feat_ymax = fmask * bbox[2] + (1 - fmask) * feat_ymax
        feat_xmax = fmask * bbox[3] + (1 - fmask) * feat_xmax

        # Check no annotation label: ignore these anchors...
        # interscts = intersection_with_anchors(bbox)
        # mask = tf.logical_and(interscts > ignore_threshold,
        #                       label == no_annotation_label)
        # # Replace scores by -1.
        # feat_scores = tf.where(mask, -tf.cast(mask, dtype), feat_scores)

        return [i+1, feat_labels, feat_scores,
                feat_ymin, feat_xmin, feat_ymax, feat_xmax]
    # Main loop definition.
    i = 0
    [i, feat_labels, feat_scores,
     feat_ymin, feat_xmin,
     feat_ymax, feat_xmax] = tf.while_loop(condition, body,
                                           [i, feat_labels, feat_scores,
                                            feat_ymin, feat_xmin,
                                            feat_ymax, feat_xmax])
    # Transform to center / size.
    feat_cy = (feat_ymax + feat_ymin) / 2.
    feat_cx = (feat_xmax + feat_xmin) / 2.
    feat_h = feat_ymax - feat_ymin
    feat_w = feat_xmax - feat_xmin
    # Encode features.
    feat_cy = (feat_cy - yref) / href / prior_scaling[0]
    feat_cx = (feat_cx - xref) / wref / prior_scaling[1]
    feat_h = tf.log(feat_h / href) / prior_scaling[2]
    feat_w = tf.log(feat_w / wref) / prior_scaling[3]
    # Use SSD ordering: x / y / w / h instead of ours.
    feat_localizations = tf.stack([feat_cx, feat_cy, feat_w, feat_h], axis=-1)
    return feat_labels, feat_localizations, feat_scores

def tf_ssd_bboxes_decode_layer(feat_localizations,
                               anchors_layer,
                               prior_scaling=[0.1, 0.1, 0.2, 0.2]):
    """Compute the relative bounding boxes from the layer features and
    reference anchor bounding boxes.
    Arguments:
      feat_localizations: Tensor containing localization features.
      anchors: List of numpy array containing anchor boxes.
    Return:
      Tensor Nx4: ymin, xmin, ymax, xmax
    """
    yref, xref, href, wref = anchors_layer

    # Compute center, height and width
    cx = feat_localizations[:, :, :, :, 0] * wref * prior_scaling[0] + xref
    cy = feat_localizations[:, :, :, :, 1] * href * prior_scaling[1] + yref
    w = wref * tf.exp(feat_localizations[:, :, :, :, 2] * prior_scaling[2])
    h = href * tf.exp(feat_localizations[:, :, :, :, 3] * prior_scaling[3])
    # Boxes coordinates.
    ymin = cy - h / 2.
    xmin = cx - w / 2.
    ymax = cy + h / 2.
    xmax = cx + w / 2.
    bboxes = tf.stack([ymin, xmin, ymax, xmax], axis=-1)
    return bboxes

def tf_ssd_bboxes_select(predictions_net, localizations_net,
                         select_threshold=None,
                         num_classes=21,
                         ignore_class=0,
                         scope=None):
    """Extract classes, scores and bounding boxes from network output layers.
    Batch-compatible: inputs are supposed to have batch-type shapes.
    Args:
      predictions_net: List of SSD prediction layers;
      localizations_net: List of localization layers;
      select_threshold: Classification threshold for selecting a box. All boxes
        under the threshold are set to 'zero'. If None, no threshold applied.
    Return:
      d_scores, d_bboxes: Dictionary of scores and bboxes Tensors of
        size Batches X N x 1 | 4. Each key corresponding to a class.
    """
    with tf.name_scope(scope, 'ssd_bboxes_select',
                       [predictions_net, localizations_net]):
        l_scores = []
        l_bboxes = []
        for i in range(len(predictions_net)):
            scores, bboxes = tf_ssd_bboxes_select_layer(predictions_net[i],
                                                        localizations_net[i],
                                                        select_threshold,
                                                        num_classes,
                                                        ignore_class)
            l_scores.append(scores)
            l_bboxes.append(bboxes)
        # Concat results.
        d_scores = {}
        d_bboxes = {}
        for c in l_scores[0].keys():
            ls = [s[c] for s in l_scores]
            lb = [b[c] for b in l_bboxes]
            d_scores[c] = tf.concat(ls, axis=1)
            d_bboxes[c] = tf.concat(lb, axis=1)
    return d_scores, d_bboxes


def tf_ssd_bboxes_select_layer(predictions_layer, localizations_layer,
                               select_threshold=None,
                               num_classes=21,
                               ignore_class=0,
                               scope=None):
    """Extract classes, scores and bounding boxes from features in one layer.
    Batch-compatible: inputs are supposed to have batch-type shapes.
    Args:
      predictions_layer: A SSD prediction layer;
      localizations_layer: A SSD localization layer;
      select_threshold: Classification threshold for selecting a box. All boxes
        under the threshold are set to 'zero'. If None, no threshold applied.
    Return:
      d_scores, d_bboxes: Dictionary of scores and bboxes Tensors of
        size Batches X N x 1 | 4. Each key corresponding to a class.
    """
    select_threshold = 0.0 if select_threshold is None else select_threshold
    with tf.name_scope(scope, 'ssd_bboxes_select_layer',
                       [predictions_layer, localizations_layer]):
        # Reshape features: Batches x N x N_labels | 4
        p_shape = tfe.get_shape(predictions_layer)
        predictions_layer = tf.reshape(predictions_layer,
                                       tf.stack([p_shape[0], -1, p_shape[-1]]))
        l_shape = tfe.get_shape(localizations_layer)
        localizations_layer = tf.reshape(localizations_layer,
                                         tf.stack([l_shape[0], -1, l_shape[-1]]))

        d_scores = {}
        d_bboxes = {}
        for c in range(0, num_classes):
            if c != ignore_class:
                # Remove boxes under the threshold.
                scores = predictions_layer[:, :, c]
                fmask = tf.cast(tf.greater_equal(scores, select_threshold), scores.dtype)
                scores = scores * fmask
                bboxes = localizations_layer * tf.expand_dims(fmask, axis=-1)
                # Append to dictionary.
                d_scores[c] = scores
                d_bboxes[c] = bboxes

    return d_scores, d_bboxes