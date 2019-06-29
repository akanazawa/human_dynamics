"""
Utils for evaluation.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np


# Error metrics

def compute_accel(joints):
    """
    Computes acceleration of 3D joints.

    Args:
        joints (Nx25x3).

    Returns:
        Accelerations (N-2).
    """
    velocities = joints[1:] - joints[:-1]
    acceleration = velocities[1:] - velocities[:-1]
    acceleration_normed = np.linalg.norm(acceleration, axis=2)
    return np.mean(acceleration_normed, axis=1)


def compute_error_3d(gt3ds, preds, vis=None):
    """
    Returns MPJPE after pelvis alignment and MPJPE after Procrustes. Should
    evaluate only on the 14 common joints.

    Args:
        gt3ds (Nx14x3).
        preds (Nx14x3).
        vis (N).

    Returns:
        MPJPE, PA-MPJPE
    """
    assert len(gt3ds) == len(preds)
    errors, errors_pa = [], []
    for i, (gt3d, pred) in enumerate(zip(gt3ds, preds)):
        if vis is None or vis[i]:
            gt3d = gt3d.reshape(-1, 3)
            # Root align.
            gt3d = align_by_pelvis(gt3d)
            pred3d = align_by_pelvis(pred)

            joint_error = np.sqrt(np.sum((gt3d - pred3d)**2, axis=1))
            errors.append(np.mean(joint_error))

            # Get PA error.
            pred3d_sym = compute_similarity_transform(pred3d, gt3d)
            pa_error = np.sqrt(np.sum((gt3d - pred3d_sym) ** 2, axis=1))
            errors_pa.append(np.mean(pa_error))

    return errors, errors_pa


def compute_error_accel(joints_gt, joints_pred, vis=None):
    """
    Computes acceleration error:
        1/(n-2) \sum_{i=1}^{n-1} X_{i-1} - 2X_i + X_{i+1}

    Note that for each frame that is not visible, three entries in the
    acceleration error should be zero'd out.

    Args:
        joints_gt (Nx14x3).
        joints_pred (Nx14x3).
        vis (N).

    Returns:
        error_accel (N-2).
    """
    # (N-2)x14x3
    accel_gt = joints_gt[:-2] - 2 * joints_gt[1:-1] + joints_gt[2:]
    accel_pred = joints_pred[:-2] - 2 * joints_pred[1:-1] + joints_pred[2:]

    normed = np.linalg.norm(accel_pred - accel_gt, axis=2)

    if vis is None:
        new_vis = np.ones(len(normed), dtype=bool)
    else:
        invis = np.logical_not(vis)
        invis1 = np.roll(invis, -1)
        invis2 = np.roll(invis, -2)
        new_invis = np.logical_or(invis, np.logical_or(invis1, invis2))[:-2]
        new_vis = np.logical_not(new_invis)

    return np.mean(normed[new_vis], axis=1)


def compute_error_kp(kps_gt, kps_pred, alpha=0.05, min_visible=6):
    """
    Compute the keypoint error (mean difference in pixels), keypoint error after
    Procrustes Analysis, and percent correct keypoints.

    Args:
        kps_gt (Nx25x3).
        kps_pred (Nx25x2).
        alpha (float).
        min_visible (int): Min threshold for deciding visibility.

    Returns:
        errors_kp, errors_kp_pa, errors_kp_pck
    """
    assert len(kps_gt) == len(kps_pred)
    errors_kp, errors_kp_pa, errors_kp_pck = [], [], []
    for kp_gt, kp_pred in zip(kps_gt, kps_pred):
        vis = kp_gt[:, 2].astype(bool)

        kp_gt = kp_gt[:, :2]
        if np.all(vis == 0) or np.sum(vis == 1) < min_visible:
            # Use nan to signify not visible.
            error_kp = np.nan
            error_pa_pck = np.nan
            error_kp_pa = np.nan
        else:
            kp_diffs = np.linalg.norm(kp_gt[vis] - kp_pred[vis], axis=1)
            kp_pred_pa, _ = compute_opt_cam_with_vis(
                got=kp_pred,
                want=kp_gt,
                vis=vis,
            )
            kp_diffs_pa = np.linalg.norm(kp_gt[vis] - kp_pred_pa[vis], axis=1)
            error_kp = np.mean(kp_diffs)
            error_pa_pck = np.mean(kp_diffs_pa < alpha)
            error_kp_pa = np.mean(kp_diffs_pa)

        errors_kp.append(error_kp)
        errors_kp_pa.append(error_kp_pa)
        errors_kp_pck.append(error_pa_pck)
    return errors_kp, errors_kp_pa, errors_kp_pck


def compute_error_verts(verts_gt, verts_pred):
    """
    Computes MPJPE over 6890 surface vertices.

    Args:
        verts_gt (Nx6989x3).
        verts_pred (Nx6989x3).

    Returns:
        error_verts (N).
    """
    assert len(verts_gt) == len(verts_pred)
    error_per_vert = np.sqrt(np.sum((verts_gt - verts_pred) ** 2, axis=2))
    return np.mean(error_per_vert, axis=1)


# Utilities for computing error metrics.

def align_by_pelvis(joints, get_pelvis=False):
    """
    Aligns joints by pelvis to be at origin. Assumes hips are index 3 and 2 of
    joints (14x3) in LSP order. Pelvis is midpoint of hips.

    Args:
        joints (14x3).
        get_pelvis (bool).
    """
    left_id = 3
    right_id = 2

    pelvis = (joints[left_id, :] + joints[right_id, :]) / 2.
    if get_pelvis:
        return joints - np.expand_dims(pelvis, axis=0), pelvis
    else:
        return joints - np.expand_dims(pelvis, axis=0)


def compute_similarity_transform(S1, S2):
    """
    Computes a similarity transform (sR, t) that takes a set of 3D points
    S1 (3 x N) closest to a set of 3D points S2, where R is an 3x3 rotation
    matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrustes problem.

    Args:
        S1 (3xN): Original 3D points.
        S2 (3xN'): Target 3D points.

    Returns:
        S1_hat: S1 after applying optimal alignment.
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1 ** 2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale * (R.dot(mu1))

    # 7. Error:
    S1_hat = scale * R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat


def compute_opt_cam_with_vis(got, want, vis):
    """
    Computes the optimal camera [scale, tx, ty] to map 2D keypoints got to 2D
    keypoints want with boolean visibility indicator vis.
    """
    # Zero out
    vis_float = np.expand_dims(vis, 1).astype(np.float)
    got_zeroed = got.copy()
    got_zeroed[np.logical_not(vis)] = 0.
    want_zeroed = want.copy()
    want_zeroed[np.logical_not(vis)] = 0.

    mu1 = np.sum(got_zeroed, axis=0) / np.sum(vis)
    mu2 = np.sum(want_zeroed, axis=0) / np.sum(vis)
    # Need to 0 out the ignore region again
    x = vis_float * (got_zeroed - mu1)
    y = vis_float * (want_zeroed - mu2)
    # Rest is same:
    eps = 1e-6 * np.identity(2)
    a_inv = np.linalg.inv(x.T.dot(x) + eps)
    scale = np.trace(a_inv.dot(x.T.dot(y))) / 2.
    trans = mu2 / scale - mu1
    new_got = scale * (got + trans)
    cam = np.hstack((scale, trans.ravel()))

    return new_got, cam


# Utilities for accumulating error values in error dicts.

def concat_dict_entries(dictionary):
    """
    Concatenates values dictionary.

    Args:
        dictionary (dict): Dict of lists with entries from appenders.
        appender (dict): Dict of values to add to dictionary
    """
    for k, v in dictionary.items():
        # Make list of lists into one big list.
        dictionary[k] = np.concatenate(v)


def extend_dict_entries(accumulator, appender):
    """
    Extends values in accumulator with appender.
    """
    for k, v in appender.items():
        if k not in accumulator:
            accumulator[k] = []
        if hasattr(v, '__iter__'):
            accumulator[k].extend(v)
        else:
            accumulator[k].append(v)


def mean_of_dict_values(dictionary):
    """
    Flattens values of dictionary and computes the mean.
    """
    for k, v in dictionary.items():
        # Each value is a list of list. Now, we take the mean of the means of
        # each list.
        all_values = [np.nanmean(values) for values in v]
        dictionary[k] = float(round(np.nanmean(all_values), 5))


def update_dict_entries(accumulator, appender):
    """
    Appends values in appender to list in accumulator.

    Args:
        accumulator (dict): Dict of lists with entries from appenders.
        appender (dict): Dict of values to add to accumulator
    """
    for k in appender:
        if k not in accumulator:
            accumulator[k] = []
        accumulator[k].append(appender[k])


# Other utilities.

def axis_angle_to_rot_mat(poses_aa):
    """
    Args:
        poses_aa (72).

    Returns:
        rot_matrices (24x3x3).
    """
    rot_matrices = []
    for pose in poses_aa.reshape(-1, 3):
        rot_matrices.append(cv2.Rodrigues(pose)[0])
    return np.array(rot_matrices)


def rot_mat_to_axis_angle(rot_matrices):
    """
    Args:
        rot_matrices (24x3x3).

    Returns:
        poses_aa (72).
    """
    pose = []
    for rot_mat in rot_matrices:
        pose.append(cv2.Rodrigues(rot_mat)[0])
    return np.array(pose).reshape(72)

