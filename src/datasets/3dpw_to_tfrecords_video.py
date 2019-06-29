"""
Convert 3DPW to TFRecords.

3DPW is NOT used for training.
Therefore this script only saves the test tfrecords.

Rectifies the 3DPW data so that camera is identity.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import os
from os.path import join, exists

import numpy as np
import tensorflow as tf

from src.datasets.make_test_tfrecords import save_seq_to_test_tfrecord
from src.datasets.threedpw.read_3dpw import get_sequences, get_3dpw2coco

data_dir = '/scratch1/jason/videos/3DPW'
out_dir = '/scratch3/kanazawa/hmmr_tfrecords_release_test/3dpw'

tf.app.flags.DEFINE_string('data_directory', data_dir,
                           'data directory: top of 3DPW')
tf.app.flags.DEFINE_string('output_directory', out_dir,
                           'Output data directory')
tf.app.flags.DEFINE_integer(
    'max_sequence_length', -1,
    'Maximum sequence length. Use -1 for full sequence.')
tf.app.flags.DEFINE_boolean('visualize', False, 'If true, visualizes')
tf.app.flags.DEFINE_string('split', 'val', 'val or test.')

FLAGS = tf.app.flags.FLAGS

VIS_THRESH = 0.1  # Minimum confidence threshold for keypoints.
IMG_SIZE = 224


def get_seq_data(anno_pkl, img_dir):
    """

    Args:
        anno_pkl (str): Path to annotation directory.
        img_dir (str): Path to image directory.

    Returns:
        im_paths: List of image files.
        all_poses: 3D Poses (P x F x 72).
        all_kps: 2D Keypoints (P x F x 25 x 3).
        all_shapes: Neutral shapes (P x 10).
        all_joints: 3D Joints (P x F x 25 x 3).
    """
    with open(anno_pkl, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    num_people = len(data['poses'])

    num_frames = len(data['img_frame_ids'])
    assert (data['poses2d'][0].shape[0] == num_frames)

    joint_order, joint_names_coco = get_3dpw2coco()
    # We have the right set of annotations.
    all_poses = []
    all_kps = []
    all_shapes = []
    for p_id in range(num_people):
        thetas = np.array(data['poses'][p_id])

        all_poses.append(thetas)
        # Collect keypoints
        kps_3dpw = data['poses2d'][p_id]

        # Pad the 7 missing parts for 25 universal
        kps_3dpw_pad = np.dstack((kps_3dpw, np.zeros((num_frames, 3, 7))))
        kps = np.array([kp.T[joint_order] for kp in kps_3dpw_pad])
        all_kps.append(kps)
        all_shapes.append(data['betas_neutral'][p_id][:10])

    num_frames_adjusted = all_kps[0].shape[0]
    all_poses = [poses[:num_frames_adjusted] for poses in all_poses]
    all_joints = np.array(data['joints_gendered']).reshape(
        (num_people, num_frames_adjusted, 25, 3))

    # Convert joints into that of identity camera.
    cam_poses = data['cam_poses']
    all_joints_rectified = []
    # P x N  x 25 x 3
    all_joints_np = all_joints.reshape((num_people, num_frames_adjusted, 25,
                                        3))

    def rectify_joints(joints, camR):
        mu = joints.mean(axis=0)
        return camR.dot((joints - mu).T).T + mu

    for p_id in range(num_people):
        joints_rectified = []
        for cam_pose, joint in zip(cam_poses, all_joints_np[p_id]):
            R = cam_pose[:3, :3]
            new_joints = rectify_joints(joint, R)
            joints_rectified.append(new_joints)
        all_joints_rectified.append(joints_rectified)

    all_joints_rectified = np.array(all_joints_rectified).reshape(
        (num_people, num_frames_adjusted, 25, 3))

    im_paths = [
        join(img_dir, 'image_%05d.jpg' % frame_id)
        for frame_id in range(0, num_frames_adjusted, 1)
    ]

    return im_paths, all_poses, all_kps, all_shapes, all_joints_rectified


def process_3dpw(data_dir, out_dir, visualize, split):
    if not exists(join(out_dir, split)):
        os.makedirs(join(out_dir, split))
    # Set tfrecord path name.
    path_name = '{seq}.tfrecord'
    sequences = get_sequences(data_dir, split)

    for i, seq in enumerate(sequences):
        pkl = join(data_dir, 'sequenceFilesNeutral', seq + '.pkl')
        img_dir = join(data_dir, 'imageFiles', seq)
        out_name = join(out_dir, split, path_name.format(seq=seq))
        if exists(out_name):
            print('{} done'.format(out_name))
            continue

        im_paths, all_poses, all_kps, all_shapes, all_joints = get_seq_data(
            anno_pkl=pkl, img_dir=img_dir)
        if im_paths is None:
            print('am i here?')
            continue
        print('{}/{}: Making {}'.format(i, len(sequences), out_name))

        save_seq_to_test_tfrecord(
            out_name=out_name,
            im_paths=im_paths,
            all_gt2ds=all_kps,
            all_gt3ds=all_joints,
            all_poses=all_poses,
            all_shapes=all_shapes,
            visualize=visualize,
            vis_thresh=VIS_THRESH,
            img_size=IMG_SIZE,
        )


def main(unused_argv):
    print('Saving results to {}'.format(FLAGS.output_directory))

    process_3dpw(
        data_dir=FLAGS.data_directory,
        out_dir=FLAGS.output_directory,
        visualize=FLAGS.visualize,
        split=FLAGS.split,
    )


if __name__ == '__main__':
    tf.app.run(main)
