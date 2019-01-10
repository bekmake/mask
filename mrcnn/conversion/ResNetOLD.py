
import tensorflow as tf
import keras.layers as KL

############################################################
#  Resnet Graph
############################################################

# Code adopted from:
# https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py

def identity_block(input_tensor, kernel_size, filters, stage, block,
                   use_bias=True, train_bn=True):
    """The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    """

    with tf.name_scope(name="ResnetIdentity"):
        nb_filter1, nb_filter2, nb_filter3 = filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'


        conv1 = tf.layers.conv2d(input_tensor, nb_filter1, (1,1), strides=(1, 1), padding='same', use_bias=True, bias_initializer=tf.zeros_initializer(),name=conv_name_base + '2a')
        conv1_bn = tf.nn.batch_normalization(conv1, name=bn_name_base + '2a')
        conv1_postAct = tf.nn.relu(conv1_bn)

        conv2 = tf.layers.conv2d(conv1_postAct, nb_filter2, kernel_size, strides=(1, 1), padding='same', use_bias=True, bias_initializer=tf.zeros_initializer(), name=conv_name_base + '2b')
        conv2_bn = tf.nn.batch_normalization(conv2, name=bn_name_base + '2b')
        conv2_postAct = tf.nn.relu(conv2_bn)


        conv3 = tf.layers.conv2d(conv2_postAct, nb_filter3, (1, 1), strides=(1, 1), padding='same', use_bias=True, bias_initializer=tf.zeros_initializer(), name=conv_name_base + '2c')
        conv3_bn = tf.nn.batch_normalization(conv3, name=bn_name_base + '2c')

        resnetSum = tf.Add(conv3_bn, input_tensor)
        conv3_postAct = tf.nn.relu(resnetSum, name='res' + str(stage) + block + '_out')

        return conv3_postAct


def conv_block(input_tensor, kernel_size, filters, stage, block,
               strides=(2, 2), use_bias=True, train_bn=True):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """
    with tf.name_scope(name="ResnetConvolution"):
        nb_filter1, nb_filter2, nb_filter3 = filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'



        conv1 = tf.layers.conv2d(input_tensor, nb_filter1, (1, 1), strides=strides, padding='same', use_bias=True, bias_initializer=tf.zeros_initializer(), name=conv_name_base + '2a')
        conv1_bn = tf.nn.batch_normalization(conv1, name=bn_name_base + '2a')
        conv1_postAct = tf.nn.relu(conv1_bn)

        conv2 = tf.layers.conv2d(conv1_postAct, nb_filter2, kernel_size, strides=(1, 1), padding='same', use_bias=True, bias_initializer=tf.zeros_initializer(), name=conv_name_base + '2b')
        conv2_bn = tf.nn.batch_normalization(conv2, name=bn_name_base + '2b')
        conv2_postAct = tf.nn.relu(conv2_bn)

        conv3 = tf.layers.conv2d(conv2_postAct, nb_filter3, (1, 1), strides=(1, 1), padding='same', use_bias=True, bias_initializer=tf.zeros_initializer(), name=conv_name_base + '2c')
        conv3_bn = tf.nn.batch_normalization(conv3, name=bn_name_base + '2c')


        convSC = tf.layers.conv2d(input_tensor, nb_filter3, (1, 1), strides=strides, padding='same', use_bias=True, bias_initializer=tf.zeros_initializer(), name= conv_name_base + '1')
        convSC_bn = tf.nn.batch_normalization(convSC, name=bn_name_base + '2c')

        resnetSum = tf.Add(conv3_bn, convSC_bn)
        conv3_postAct = tf.nn.relu(resnetSum, name='res' + str(stage) + block + '_out')
        return conv3_postAct


def resnet_graph(input_image, architecture, stage5=False, train_bn=True):
    """Build a ResNet graph.
        architecture: Can be resnet50 or resnet101
        stage5: Boolean. If False, stage5 of the network is not created
        train_bn: Boolean. Train or freeze Batch Norm layers
    """

    with tf.name_scope("ResNet"):
        assert architecture in ["resnet50", "resnet101"]

        # Stage 1
        paddings = tf.constant([[3, 3, ], [3, 3], [0, 0]])
        #padded_image = tf.pad(input_image, paddings, "CONSTANT")
        #print(input_image)
        padded_image = KL.ZeroPadding2D((3, 3))(input_image)
        print(padded_image)
        conv1 = tf.layers.conv2d(padded_image, 64, (7, 7), strides=(2, 2), use_bias=True, bias_initializer=tf.zeros_initializer(), name='conv1')
        print(conv1)
        conv1_bn = tf.nn.batch_normalization(conv1, name='bn_conv1')
        conv1_postAct = tf.nn.relu(conv1_bn)

        C1 = x = tf.nn.max_pool(conv1_postAct, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Stage 2
        x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), train_bn=train_bn)
        x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', train_bn=train_bn)
        C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', train_bn=train_bn)

        # Stage 3
        x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', train_bn=train_bn)
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', train_bn=train_bn)
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', train_bn=train_bn)
        C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', train_bn=train_bn)

        # Stage 4
        x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', train_bn=train_bn)
        block_count = {"resnet50": 5, "resnet101": 22}[architecture]
        for i in range(block_count):
            x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i), train_bn=train_bn)
        C4 = x

        # Stage 5
        if stage5:
            x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', train_bn=train_bn)
            x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', train_bn=train_bn)
            C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', train_bn=train_bn)
        else:
            C5 = None

        return [C1, C2, C3, C4, C5]

