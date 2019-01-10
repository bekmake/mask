import tensorflow as tf
############################################################
#  Region Proposal Network (RPN)
############################################################

def rpn_graph(feature_map, anchors_per_location, anchor_stride):
    """Builds the computation graph of Region Proposal Network.
    feature_map: backbone features [batch, height, width, depth]
    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).
    Returns:
        rpn_class_logits: [batch, H * W * anchors_per_location, 2] Anchor classifier logits (before softmax)
        rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
        rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be
                  applied to anchors.
    """
    with tf.name_scope('RPN_Graph'):
        # TODO: check if stride of 2 causes alignment issues if the feature map
        # is not even.
        # Shared convolutional base of the RPN
        shared = tf.layers.conv2d(feature_map, 512, (3, 3), strides=anchor_stride, padding='same', activation='relu',  name='rpn_conv_shared')
        # Anchor Score. [batch, height, width, anchors per location * 2].
        x = tf.layers.conv2d(shared, 2 * anchors_per_location, (1, 1), padding='valid', activation='linear', name='rpn_class_raw')
        # Reshape to [batch, anchors, 2]
        rpn_class_logits = tf.reshape(x, [tf.shape(x)[0], -1, 2])

        # Softmax on last dimension of BG/FG.
        rpn_probs = tf.nn.softmax(rpn_class_logits, name="rpn_class_xxx")
        # Bounding box refinement. [batch, H, W, anchors per location * depth]
        # where depth is [x, y, log(w), log(h)]
        x = tf.layers.conv2d(shared, anchors_per_location * 4, (1, 1), padding='valid', activation='linear', name='rpn_bbox_pred')
        # Reshape to [batch, anchors, 4]
        rpn_bbox = tf.reshape(x, [tf.shape(x)[0], -1, 4])

        return [rpn_class_logits, rpn_probs, rpn_bbox]


def build_rpn_model(anchor_stride, anchors_per_location, depth):
    """Builds a Keras model of the Region Proposal Network.
    It wraps the RPN graph so it can be used multiple times with shared
    weights.
    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).
    depth: Depth of the backbone feature map.
    Returns a Keras Model object. The model outputs, when called, are:
    rpn_class_logits: [batch, H * W * anchors_per_location, 2] Anchor classifier logits (before softmax)
    rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
    rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be
                applied to anchors.
    """
    input_feature_map = tf.placeholder(shape=[None, None, None, depth],name="input_rpn_feature_map", dtype=tf.float32)
    outputs = rpn_graph(input_feature_map, anchors_per_location, anchor_stride)
    return input_feature_map, outputs
