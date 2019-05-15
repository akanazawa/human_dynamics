from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim


class Discriminator(object):

    def __init__(self, weight_decay):
        self.vars = []
        self.reuse = False
        self.wd = weight_decay

    def get_output(self, inputs):
        raise NotImplementedError

    def get_vars(self):
        return self.vars

    def update(self, vars):
        self.reuse = True
        self.vars.extend(vars)


class PoseDiscriminator(Discriminator):

    def __init__(self, weight_decay):
        super(PoseDiscriminator, self).__init__(weight_decay)

    def get_output(self, poses):
        """
        Gets discriminator's predictions for each pose and all poses.

        Args:
            poses (Nx23x1x9).

        Returns:
            Predictions (Nx[23+1]).
        """
        data_format = 'NHWC'
        with tf.variable_scope('D_pose', reuse=self.reuse) as scope:
            with slim.arg_scope(
                    [slim.conv2d, slim.fully_connected],
                    weights_regularizer=slim.l2_regularizer(self.wd)):
                with slim.arg_scope([slim.conv2d], data_format=data_format):
                    poses = slim.conv2d(
                        inputs=poses,
                        num_outputs=32,
                        kernel_size=[1, 1],
                        reuse=self.reuse,
                        scope='D_conv1'
                    )
                    poses = slim.conv2d(
                        inputs=poses,
                        num_outputs=32,
                        kernel_size=[1, 1],
                        reuse=self.reuse,
                        scope='D_conv2'
                    )
                    theta_out = []
                    for i in range(0, 23):
                        theta_out.append(
                            slim.fully_connected(
                                inputs=poses[:, i, :, :],
                                num_outputs=1,
                                activation_fn=None,
                                reuse=self.reuse,
                                scope='pose_out_j{}'.format(i)
                            ))
                    theta_out_all = tf.squeeze(tf.stack(theta_out, axis=1))

                    # Compute joint correlation prior!
                    nz_feat = 1024
                    poses_all = slim.flatten(poses, scope='vectorize')
                    poses_all = slim.fully_connected(
                        inputs=poses_all,
                        num_outputs=nz_feat,
                        reuse=self.reuse,
                        scope='D_alljoints_fc1'
                    )
                    poses_all = slim.fully_connected(
                        inputs=poses_all,
                        num_outputs=nz_feat,
                        reuse=self.reuse,
                        scope='D_alljoints_fc2'
                    )
                    poses_all_out = slim.fully_connected(
                        inputs=poses_all,
                        num_outputs=1,
                        activation_fn=None,
                        reuse=self.reuse,
                        scope='D_alljoints_out')
                    out = tf.concat([theta_out_all, poses_all_out], 1)

            if not self.reuse:
                self.update(tf.contrib.framework.get_variables(scope))

            return out


class DeltaDiscriminator(Discriminator):

    def __init__(self, weight_decay):
        super(DeltaDiscriminator, self).__init__(weight_decay)

    def get_output(self, deltas):
        """
        Gets discriminator's predictions for each delta pose and all delta
        poses.

        Args:
            deltas (Nx24x1x18): Poses and delta poses.

        Returns:
            Predictions (Nx[24+1]).
        """
        data_format = 'NHWC'
        with tf.variable_scope('D_delta', reuse=self.reuse) as scope:
            with slim.arg_scope(
                    [slim.conv2d, slim.fully_connected],
                    weights_regularizer=slim.l2_regularizer(self.wd)):
                with slim.arg_scope([slim.conv2d], data_format=data_format):
                    poses = slim.conv2d(
                        inputs=deltas,
                        num_outputs=64,
                        kernel_size=[1, 1],
                        reuse=self.reuse,
                        scope='D_conv_delta_1'
                    )
                    poses = slim.conv2d(
                        inputs=poses,
                        num_outputs=64,
                        kernel_size=[1, 1],
                        reuse=self.reuse,
                        scope='D_conv_delta_2'
                    )
                    theta_out = []
                    for i in range(0, 24):
                        theta_out.append(
                            slim.fully_connected(
                                inputs=poses[:, i, :, :],
                                num_outputs=1,
                                activation_fn=None,
                                reuse=self.reuse,
                                scope='pose_out_j{}'.format(i)
                            ))
                    theta_out_all = tf.squeeze(tf.stack(theta_out, axis=1))

                    # Compute joint correlation prior!
                    nz_feat = 1024
                    poses_all = slim.flatten(theta_out_all,
                                             scope='vectorize')
                    poses_all = slim.fully_connected(
                        inputs=poses_all,
                        num_outputs=nz_feat,
                        reuse=self.reuse,
                        scope='D_alljoints_delta_fc1'
                    )
                    poses_all = slim.fully_connected(
                        inputs=poses_all,
                        num_outputs=nz_feat,
                        reuse=self.reuse,
                        scope='D_alljoints_delta_fc2'
                    )
                    poses_all_out = slim.fully_connected(
                        inputs=poses_all,
                        num_outputs=1,
                        activation_fn=None,
                        reuse=self.reuse,
                        scope='D_alljoints_delta_out')
                    out = tf.concat([theta_out_all, poses_all_out], 1)
            if not self.reuse:
                self.update(tf.contrib.framework.get_variables(scope))
            return out


class ShapeDiscriminator(Discriminator):

    def __init__(self, weight_decay):
        super(ShapeDiscriminator, self).__init__(weight_decay)

    def get_output(self, shapes):
        """
        Gets discriminator's predictions for shapes.

        Args:
            shapes (Nx10).

        Returns:
            Predictions (Nx1).
        """
        data_format = 'NHWC'
        with tf.variable_scope('D_shape', reuse=self.reuse) as scope:
            with slim.arg_scope(
                    [slim.conv2d, slim.fully_connected],
                    weights_regularizer=slim.l2_regularizer(self.wd)):
                with slim.arg_scope([slim.conv2d], data_format=data_format):
                    shapes = slim.stack(
                        inputs=shapes,
                        layer=slim.fully_connected,
                        stack_args=[10, 5],
                        scope='shape_fc1'
                    )
                    shape_out = slim.fully_connected(
                        inputs=shapes,
                        num_outputs=1,
                        activation_fn=None,
                        reuse=self.reuse,
                        scope='shape_final'
                    )
                    out = tf.concat([shape_out], 1)
            if not self.reuse:
                self.update(tf.contrib.framework.get_variables(scope))
            return out
