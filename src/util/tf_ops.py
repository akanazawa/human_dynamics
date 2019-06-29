"""
Tensorflow operations.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def compute_deltas_batched(poses_prev, poses_curr):
    """
    Computes the change between rotation matrices using R1 * R2.T.

    Args:
        poses_prev (BxTx24x3x3).
        poses_curr (BxTx24x3x3).

    Returns:
        delta_poses (Bx(T-1)x24x3x3).
    """
    assert poses_prev.shape == poses_curr.shape
    assert len(poses_prev.shape) == 5
    assert poses_prev.shape[2:] == (24, 3, 3)
    return tf.matmul(
        a=poses_prev,
        b=poses_curr,
        transpose_b=True
    )
