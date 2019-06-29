from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim


class PoseDiscriminator(object):
    def __init__(self, weight_decay):
        self.vars = []
        self.reuse = False
        self.wd = weight_decay

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
                        scope='D_conv1')
                    poses = slim.conv2d(
                        inputs=poses,
                        num_outputs=32,
                        kernel_size=[1, 1],
                        reuse=self.reuse,
                        scope='D_conv2')
                    theta_out = []
                    for i in range(0, 23):
                        theta_out.append(
                            slim.fully_connected(
                                inputs=poses[:, i, :, :],
                                num_outputs=1,
                                activation_fn=None,
                                reuse=self.reuse,
                                scope='pose_out_j{}'.format(i)))
                    theta_out_all = tf.squeeze(tf.stack(theta_out, axis=1))

                    # Compute joint correlation prior!
                    nz_feat = 1024
                    poses_all = slim.flatten(poses, scope='vectorize')
                    poses_all = slim.fully_connected(
                        inputs=poses_all,
                        num_outputs=nz_feat,
                        reuse=self.reuse,
                        scope='D_alljoints_fc1')
                    poses_all = slim.fully_connected(
                        inputs=poses_all,
                        num_outputs=nz_feat,
                        reuse=self.reuse,
                        scope='D_alljoints_fc2')
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

    def get_vars(self):
        return self.vars

    def update(self, vars):
        self.reuse = True
        self.vars.extend(vars)
