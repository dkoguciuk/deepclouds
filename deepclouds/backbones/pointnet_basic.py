import tensorflow as tf
import deepclouds.backbones.tf_util as tf_util
from deepclouds.backbones.point_cloud_fext import PointCloudFExt


class PointNetBasic(PointCloudFExt):
    """
    PointNet feature extractor.
    """

    def __init__(self, input_point_cloud, is_training, setting=None):

        # Setting
        tnet1_conv_units = [64, 128, 1024] if not hasattr(setting, 'tnet1_conv_units') else setting['tnet1_conv_units']
        tnet1_fc_units = [512, 256] if not hasattr(setting, 'tnet1_fc_units') else setting['tnet1_fc_units']
        conv_units_1 = [64, 64] if not hasattr(setting, 'conv_units_1') else setting['conv_units_1']
        conv_units_2 = [64, 128, 1024] if not hasattr(setting, 'conv_units_2') else setting['conv_units_2']
        tnet2_conv_units = [64, 128, 1024] if not hasattr(setting, 'tnet2_conv_units') else setting['tnet2_conv_units']
        tnet2_fc_units = [512, 256] if not hasattr(setting, 'tnet2_fc_units') else setting['tnet2_fc_units']

        # Input
        net = input_point_cloud

        # Conv-net1
        net = self._define_conv_net1(net, conv_units_1, is_training=is_training, bn_decay=None)

        # Conv-net2
        net = self._define_conv_net2(net, conv_units_1, conv_units_2, is_training=is_training, bn_decay=None)

        # Process block
        net = tf.squeeze(self._max_pool2d(tf.expand_dims(net, axis=-2), kernel_size=[net.shape[1], 1], stride=[2, 2]))

        # Features
        self.features = net

    def get_features(self):
        return self.features

    @staticmethod
    def _define_conv_net1(input_data, conv_units_1, is_training, bn_decay=None):
        """
        Define input read block part 1.

        Args:
            input_data (tf.tensor of size B, N, 3)

        Returns:
            tf.tensor of size B, N, F
        """

        # Prep
        net = tf.expand_dims(input_data, -1)

        # Read block convs
        for idx, conv_net in enumerate(conv_units_1):
            with tf.variable_scope("conv{}".format(idx+1)):
                kernel_size = [1, 1]
                if idx == 0:
                    kernel_size = [1, 3]
                net = tf_util.conv2d_on_the_fly(net, num_in_channels=net.shape[-1], num_out_channels=conv_net,
                                                kernel_size=kernel_size, padding='VALID', stride=[1, 1], bn=False,
                                                is_training=is_training, bn_decay=bn_decay)
        # Return
        return tf.squeeze(net, axis=-2)

    @staticmethod
    def _define_conv_net2(input_data, conv_units_1, conv_units_2, is_training, bn_decay=None):
        """
        Define input read block part 2.

        Args:
            input_data (tf.tensor of size B, N, F)

        Returns:
            tf.tensor of size B, N, E
        """

        # Prep
        net = tf.expand_dims(input_data, axis=-2)

        # Read block convs
        for idx, conv_net in enumerate(conv_units_2):
            with tf.variable_scope("conv{}".format(len(conv_units_1)+idx+1)):
                net = tf_util.conv2d_on_the_fly(net, num_in_channels=net.shape[-1], num_out_channels=conv_net,
                                                kernel_size=[1, 1], padding='VALID', stride=[1, 1], bn=False,
                                                is_training=is_training, bn_decay=bn_decay)

        # Return
        return tf.squeeze(net, axis=-2)

    @staticmethod
    def _max_pool2d(inputs, kernel_size, stride, padding='VALID'):
        """ 2D max pooling.

        Args:
            inputs: 4-D tensor BxHxWxC
            kernel_size: a list of 2 ints
            stride: a list of 2 ints

        Returns:
            Variable tensor
        """
        kernel_h, kernel_w = kernel_size
        stride_h, stride_w = stride
        outputs = tf.nn.max_pool(inputs, ksize=[1, kernel_h, kernel_w, 1], strides=[1, stride_h, stride_w, 1],
                                 padding=padding)
        return outputs
