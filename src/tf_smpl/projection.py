"""
Util functions implementing the camera

@@batch_orth_proj_idrot
@@batch_orth_proj_optcam
@@procrustes2d_vis
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def batch_orth_proj_idrot(X, camera, name=None):
    """
    X is N x num_points x 3
    camera is N x 3
    same as applying orth_proj_idrot to each N
    """
    with tf.name_scope(name, 'batch_orth_proj_idrot', [X, camera]):
        camera = tf.reshape(camera, [-1, 1, 3], name='cam_adj_shape')

        X_trans = X[:, :, :2] + camera[:, :, 1:]

        shape = tf.shape(X_trans)
        return tf.reshape(
            camera[:, :, 0] * tf.reshape(X_trans, [shape[0], -1]), shape)


def batch_orth_proj_optcam(X, X_gt, name=None):
    """
    Solves for best sale and translation in 2D, i.e.
    gives (s, t) such that ||s(x + t) - x_gt||^2
    X is N x K x 2, for [x, y] pred (via identity).
    X_gt is N x K x 3, the 3rd dim is visibility

    returns proj_x: N x K x 2 and best_cam:[scale, trans]
    """
    with tf.name_scope(name, 'batch_orth_proj_optcam', [X, X_gt]):
        best_cam = procrustes2d_vis(X, X_gt)
        best_cam = tf.stop_gradient(best_cam)
        proj_x = batch_orth_proj_idrot(X, best_cam)
        return proj_x, best_cam


def procrustes2d_vis(X, X_target):
    """
    Solves for the optimal sale and translation in 2D, i.e.
    gives (s, t) such that ||s(x + t) - x_gt||^2
    on *visible* points.

    Gradient is stopped on the computed camera.

    X: N x K x 2 or N x K x 3 (last dim is dropped)
    X_target: N x K x 3, 3rd dim is visibility.

    returns best_cam: N x 3
    """
    assert len(X_target.shape) == 3
    with tf.name_scope('procrustes2d_vis', [X, X_target]):
        # Turn vis into [0, 1]
        vis = tf.cast(X_target[:, :, 2] > 0, tf.float32)
        vis_vec = tf.expand_dims(vis, 2)
        # Prepare data.
        x_target = X_target[:, :, :2]
        x = X[:, :, :2]
        # Start:
        # Make sure invisible points dont contribute
        # (They could be not always 0...)
        x_vis = vis_vec * x
        x_target_vis = vis_vec * x_target

        num_vis = tf.expand_dims(tf.reduce_sum(vis, 1, keepdims=True), 2)

        # need to compute mean ignoring the non-vis
        mu1 = tf.reduce_sum(x_vis, 1, keepdims=True) / num_vis
        mu2 = tf.reduce_sum(x_target_vis, 1, keepdims=True) / num_vis
        # Need to 0 out the ignore region again
        xmu = vis_vec * (x - mu1)
        y = vis_vec * (x_target - mu2)

        # Add noise on the diagonal to avoid numerical instability
        # for taking inv.
        eps = 1e-6 * tf.eye(2)
        Ainv = tf.matrix_inverse(tf.matmul(xmu, xmu, transpose_a=True) + eps)
        B = tf.matmul(xmu, y, transpose_a=True)

        scale = tf.expand_dims(tf.trace(tf.matmul(Ainv, B)) / 2., 1)

        # Bounding the scale value.
        # If prediction is flipped, the optimal scale approaches 0.
        # (bc 0 error is the best you can do without out-of-plane-rotation)
        # Bound the scale with a lower bound to prevent collapse.
        # We only need the lower bound, but setting max to 10 bc tf doesn't
        # take None.
        scale = tf.clip_by_value(scale, 0.7, 10)

        trans = tf.squeeze(mu2) / scale - tf.squeeze(mu1)

        best_cam = tf.concat([scale, trans], 1)
        
        return best_cam
