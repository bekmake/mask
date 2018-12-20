import tensorflow as tf
import mrcnn.conversion.RoiAlignment
############################################################
#  Feature Pyramid Network Heads
############################################################

def fpn_classifier_graph(rois, feature_maps, image_meta,
                         pool_size, num_classes, train_bn=True,
                         fc_layers_size=1024):
    """Builds the computation graph of the feature pyramid network classifier
    and regressor heads.
    rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
          coordinates.
    feature_maps: List of feature maps from different layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.
    image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    pool_size: The width of the square feature map generated from ROI Pooling.
    num_classes: number of classes, which determines the depth of the results
    train_bn: Boolean. Train or freeze Batch Norm layers
    fc_layers_size: Size of the 2 FC layers
    Returns:
        logits: [batch, num_rois, NUM_CLASSES] classifier logits (before softmax)
        probs: [batch, num_rois, NUM_CLASSES] classifier probabilities
        bbox_deltas: [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))] Deltas to apply to
                     proposal boxes
    """
    # ROI Pooling
    # Shape: [batch, num_rois, POOL_SIZE, POOL_SIZE, channels]

    #tf.nn.batch_normalization(conv1_biased, name=bn_name_base + '2a')
    #tf.nn.conv2d(input_tensor, W1, strides=[1, 1, 1, 1], padding='SAME',name=conv_name_base + '2a')

    with tf.name_scope("fpn_classifier"):
        x = RoiAlignment.PyramidROIAlign([pool_size, pool_size], name="roi_align_classifier")([rois, image_meta] + feature_maps)
        # Two 1024 FC layers (implemented with Conv2D for consistency)


#tf.layers.conv2d(padded_image, 64, (7, 7), strides=(2, 2), padding='same', use_bias=True, bias_initializer=tf.zeros_initializer(), name='conv1')

        x = tf.map_fn(lambda t: tf.layers.conv2d(t, fc_layers_size, (pool_size, pool_size), strides=(1, 1), padding='valid'), x,
                      name="mrcnn_class_conv1")
        x = tf.map_fn(lambda t: tf.nn.batch_normalization(t, name='mrcnn_class_bn1'), x)
        x = tf.nn.relu(x)

        x = tf.map_fn(lambda t: tf.layers.conv2d(t, fc_layers_size, (pool_size, pool_size), strides=(1, 1), padding='valid'), x,
            name="mrcnn_class_conv2")
        x = tf.map_fn(lambda t: tf.nn.batch_normalization(t, name='mrcnn_class_bn2'), x)
        x = tf.nn.relu(x)

        shared = lambda x: tf.squeeze(tf.squeeze(x, 3), 2)


        mrcnn_class_logits = tf.map_fn(lambda t: tf.layers.dense(t, num_classes, name="mrcnn_class_logits"), shared)
        mrcnn_probs =  tf.map_fn(lambda t: tf.nn.softmax(t, name="mrcnn_class"), mrcnn_class_logits)

        # BBox head
        # [batch, num_rois, NUM_CLASSES * (dy, dx, log(dh), log(dw))]
        x = tf.map_fn(lambda t: tf.layers.dense(t, num_classes * 4, name="mrcnn_bbox_fc"), shared)

        # Reshape to [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))]
        s = tf.shape(x)
        mrcnn_bbox = tf.reshape((s[1], num_classes, 4), name="mrcnn_bbox")(x)

        return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox


def build_fpn_mask_graph(rois, feature_maps, image_meta,
                         pool_size, num_classes, train_bn=True):
    """Builds the computation graph of the mask head of Feature Pyramid Network.
    rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
          coordinates.
    feature_maps: List of feature maps from different layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.
    image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    pool_size: The width of the square feature map generated from ROI Pooling.
    num_classes: number of classes, which determines the depth of the results
    train_bn: Boolean. Train or freeze Batch Norm layers
    Returns: Masks [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, NUM_CLASSES]
    """
    with tf.name_scope("fpn_mask"):
        # ROI Pooling
        # Shape: [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, channels]
        x = RoiAlignment.PyramidROIAlign([pool_size, pool_size], name="roi_align_mask")([rois, image_meta] + feature_maps)

        # Conv layers
        x = tf.map_fn(lambda t: tf.layers.conv2d(t, 256, (3, 3), strides=(1, 1), padding='same'), x, name="mrcnn_mask_conv1")
        x = tf.map_fn(lambda t: tf.nn.batch_normalization(t, name='mrcnn_mask_bn1'), x)
        x = tf.nn.relu(x)

        x = tf.map_fn(lambda t: tf.layers.conv2d(t, 256, (3, 3), strides=(1, 1), padding='same'), x, name="mrcnn_mask_conv2")
        x = tf.map_fn(lambda t: tf.nn.batch_normalization(t, name='mrcnn_mask_bn2'), x)
        x = tf.nn.relu(x)

        x = tf.map_fn(lambda t: tf.layers.conv2d(t, 256, (3, 3), strides=(1, 1), padding='same'), x, name="mrcnn_mask_conv3")
        x = tf.map_fn(lambda t: tf.nn.batch_normalization(t, name='mrcnn_mask_bn3'), x)
        x = tf.nn.relu(x)

        x = tf.map_fn(lambda t: tf.layers.conv2d(t, 256, (3, 3), strides=(1, 1), padding='same'), x, name="mrcnn_mask_conv4")
        x = tf.map_fn(lambda t: tf.nn.batch_normalization(t, name='mrcnn_mask_bn4'), x)
        x = tf.nn.relu(x)

        x = tf.map_fn(lambda t: tf.layers.conv2d_transpose(t, 256, (2, 2), strides=(2, 2), padding='same', activation='relu'), x, name='mrcnn_mask_deconv')
        x = tf.map_fn(lambda t: tf.layers.conv2d(t, num_classes, (1, 1), strides=(1, 1), activation='sigmoid'), x, name="mrcnn_mask")
        return x

