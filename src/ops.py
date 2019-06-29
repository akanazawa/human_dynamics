"""
TF util operations.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from src.models import encoder_fc3_dropout
from src.tf_smpl.projection import batch_orth_proj_optcam


def compute_loss_e_kp_optcam(kp_gt, kp_pred, name=None):
    """
    computes optimal scale s and translation t=[tx; ty], given
    kp_gt and kp_pred. then computes:
    L_{KP}=\sum_{i=1}^K v_i||x_i - s*(\hat{x_i} + t)||_1

    Args:
        kp_gt (BxTxKx3): Ground truth kp.
        kp_pred (BxTxKx2): Predicted kp.
    """
    with tf.name_scope(name, 'loss_e_kp_optcam', [kp_gt, kp_pred]):
        # Make into BT x K x 3
        B = kp_gt.shape[0].value
        T = kp_gt.shape[1].value
        BT = B*T

        kp_gt = tf.reshape(kp_gt, (BT, -1, 3))
        kp_pred = tf.reshape(kp_pred, (BT, -1, 2))

        # This computes the best cam (stops grad)
        # and "projects" ie transforms kp_pred with the best cam
        kp_pred_sim, best_cam = batch_orth_proj_optcam(kp_pred, kp_gt)
        best_cam = tf.reshape(best_cam, (B, T, 3))

        return compute_loss_e_kp(kp_gt, kp_pred_sim), best_cam


def compute_loss_e_kp(kp_gt, kp_pred, name=None):
    """
    L_{KP}=\sum_{i=1}^K v_i||x_i - (\hat{x_i}||_1

    Args:
        kp_gt (NxKx3): Ground truth kp.
        kp_pred (NxKx2): Predicted kp.
        name (str).
    """
    with tf.name_scope(name, 'loss_e_kp', [kp_gt, kp_pred]):
        kp_gt = tf.reshape(kp_gt, (-1, 3))
        kp_pred = tf.reshape(kp_pred, (-1, 2))

        vis = tf.expand_dims(tf.cast(kp_gt[:, 2], tf.float32), 1)
        res = tf.losses.absolute_difference(kp_gt[:, :2], kp_pred, weights=vis)
        return res


def compute_loss_e_3d(poses_gt, poses_pred, shapes_gt, shapes_pred,
                      joints_gt, joints_pred, batch_size, has_gt3d_smpl,
                      has_gt3d_joints):
    poses_gt = tf.reshape(poses_gt, (batch_size, -1))
    poses_pred = tf.reshape(poses_pred, (batch_size, -1))

    shapes_gt = tf.reshape(shapes_gt, (batch_size, -1))
    shapes_pred = tf.reshape(shapes_pred, (batch_size, -1))
    # Make sure joints are B x T x 14 x 3
    assert len(joints_gt.shape) == 4
    # Reshape joints to BT x 14 x 3
    joints_gt = tf.reshape(joints_gt, (-1, joints_gt.shape[2], 3))
    joints_pred = tf.reshape(joints_pred, (-1, joints_pred.shape[2], 3))
    # Now align them by pelvis:
    joints_gt = align_by_pelvis(joints_gt)
    joints_pred = align_by_pelvis(joints_pred)

    loss_e_pose = compute_loss_mse(poses_gt, poses_pred, has_gt3d_smpl)
    loss_e_shape = compute_loss_mse(shapes_gt, shapes_pred, has_gt3d_smpl)
    loss_e_joints = compute_loss_mse(
        joints_gt,
        joints_pred,
        tf.expand_dims(has_gt3d_joints, 1)
    )

    return loss_e_pose, loss_e_shape, loss_e_joints


def compute_loss_mse(params_gt, params_pred, has_gt3d):
    """
    Computes the l2 loss between 3D params pred and gt for the data that
    has_gt3d is True.

    Parameters to compute loss over:
    3Djoints: 14*3 = 42
    rotations:(24*9)= 216
    shape: 10
    total input: 226 (gt SMPL params) or 42 (just joints)

    Args:
        params_pred: N x {226, 42}
        params_gt: N x {226, 42}
        has_gt3d: N x 1 tf.float32 of {0., 1.}
    """
    with tf.name_scope('loss_mse', values=[params_pred, params_gt, has_gt3d]):
        weights = tf.expand_dims(tf.cast(has_gt3d, tf.float32), 1)
        res = 0.5 * tf.losses.mean_squared_error(
            params_gt,
            params_pred,
            weights=weights
        )
        return res


def compute_loss_e_smooth(joints_prev, joints_curr):
    """

    Args:
        joints_prev (NxKx3): Predicted joint locations for time t - 1.
        joints_curr (NxKx3): Predicted joint locations for time t.
    """
    with tf.name_scope('loss_e_smooth', values=[joints_prev, joints_curr]):
        return 0.5 * tf.losses.mean_squared_error(
            joints_prev,
            joints_curr,
        )


def compute_loss_e_fake(out_fake):
    return tf.reduce_mean(tf.reduce_sum((out_fake - 1) ** 2, axis=1))


def compute_loss_d_fake(out_fake):
    return tf.reduce_mean(tf.reduce_sum(out_fake ** 2, axis=1))


def compute_loss_d_real(out_real):
    return tf.reduce_mean(tf.reduce_sum((out_real - 1) ** 2, axis=1))


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


def compute_loss_shape(shapes):
    """
    L2 loss on shapes.
    """
    return tf.reduce_mean(tf.square(shapes))


def align_by_pelvis(joints):
    """
    Assumes joints is N x 14 x 3 in LSP order.
    Then hips are: [3, 2]
    Takes mid point of these points, then subtracts it.
    """
    if len(joints.shape) != 3:
        print('I should never be here!!')
        import ipdb; ipdb.set_trace()
        joints = tf.reshape(joints, (-1, 14, 3))
    with tf.name_scope('align_by_pelvis', [joints]):
        left_id = 3
        right_id = 2
        pelvis = (joints[:, left_id, :] + joints[:, right_id, :]) / 2.
        return joints - tf.expand_dims(pelvis, axis=1)


def call_hmr_ief(phi, omega_start, scope, num_output=85, num_stage=3,
                 is_training=True, predict_delta_keys=(),
                 use_delta_from_pred=False, use_optcam=False):
    """
    Wrapper for doing HMR-style IEF.

    If predict_delta, then also makes num_delta_t predictions forward and
    backward in time, with each step of delta_t.

    Args:
        phi (Bx2048): Image features.
        omega_start (Bx85): Starting Omega as input to first IEF.
        scope (str): Name of scope for reuse.
        num_output (int): Size of output.
        num_stage (int): Number of iterations for IEF.
        is_training (bool): If False, don't apply dropout.
        predict_delta_keys (iterable): List of keys for delta_t.
        use_delta_from_pred (bool): If True, initializes delta prediction from
            current frame prediction.
        use_optcam (bool): If True, only outputs 82 and uses [1, 0, 0] as cam.

    Returns:
        Final theta (Bx{num_output})
        Deltas predictions (List of outputs)
    """
    theta_here = hmr_ief(
        phi=phi,
        omega_start=omega_start,
        scope=scope,
        num_output=num_output,
        num_stage=num_stage,
        is_training=is_training,
    )

    # Delta only needs to do cam/pose, no shape!
    if use_optcam:
        num_output_delta = 72
    else:
        num_output_delta = 3 + 72

    deltas_predictions = {}
    for delta_t in predict_delta_keys:
        if delta_t == 0:
            # This should just be the normal IEF.
            continue
        elif delta_t > 0:
            scope_delta = scope + '_future{}'.format(delta_t)
        elif delta_t < 0:
            scope_delta = scope + '_past{}'.format(abs(delta_t))

        omega_start_delta = theta_here if use_delta_from_pred else omega_start
        # append this later.
        beta = omega_start_delta[:, -10:]

        if use_optcam:
            # trim the first 3D camera + last shpae
            # DEBUG!
            # just pose = 3:3+72
            omega_start_delta = omega_start_delta[:, 3:3 + num_output_delta]
        else:
            # Uncomment this to be backwards compatible
            # Drop the shape.
            # just cam + pose = 3:3+72
            omega_start_delta = omega_start_delta[:, :num_output_delta]

        delta_pred = hmr_ief(
            phi=phi,
            omega_start=omega_start_delta,
            scope=scope_delta,
            num_output=num_output_delta,
            num_stage=num_stage,
            is_training=is_training
        )
        if use_optcam:
            # Add camera + shape
            scale = tf.ones([delta_pred.shape[0], 1])
            trans = tf.zeros([delta_pred.shape[0], 2])
            delta_pred = tf.concat([scale, trans, delta_pred, beta], 1)
        else:
            delta_pred = tf.concat([delta_pred[:, :75], beta], 1)

        deltas_predictions[delta_t] = delta_pred

    return theta_here, deltas_predictions


def hmr_ief(phi, omega_start, scope, num_output=85, num_stage=3,
            is_training=True):
    """
    Runs HMR-style IEF.

    Args:
        phi (Bx2048): Image features.
        omega_start (Bx85): Starting Omega as input to first IEF.
        scope (str): Name of scope for reuse.
        num_output (int): Size of output.
        num_stage (int): Number of iterations for IEF.
        is_training (bool): If False, don't apply dropout.

    Returns:
        Final theta (Bx{num_output})
    """
    with tf.variable_scope(scope):
        theta_prev = omega_start
        theta_here = None

        for _ in range(num_stage):
            # ---- Compute outputs
            state = tf.concat([phi, theta_prev], 1)
            delta_theta, _ = encoder_fc3_dropout(
                state,
                is_training=is_training,
                num_output=num_output,
                reuse=tf.AUTO_REUSE
            )
            # Compute new theta
            theta_here = theta_prev + delta_theta

            # Finally update to end iteration.
            theta_prev = theta_here

    return theta_here
