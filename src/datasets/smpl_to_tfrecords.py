"""
Convert MoCap SMPL data to tfrecords.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
from glob import glob
from os import makedirs
from os.path import (
    basename,
    exists,
    join,
)

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from src.datasets.common import float_feature


tf.app.flags.DEFINE_boolean('temporal', True,
                            'If True, generates tfrecords for delta poses')
tf.app.flags.DEFINE_string(
    'dataset_name',
    'neutrSMPL_CMU',
    'neutrSMPL_CMU or neutrSMPL_jointLim'
)
tf.app.flags.DEFINE_string('data_directory',
                           '/scratch1/storage/human_datasets/neutrMosh/',
                           'Data directory where SMPL npz/pkl lies')
tf.app.flags.DEFINE_string(
    'output_directory',
    '/scratch1/jason/tf_datasets_phi_shard/mocap_neutrMosh/',
    'Output data directory'
)
tf.app.flags.DEFINE_string(
    'output_directory_temporal',
    '/scratch1/jason/tf_datasets_phi_shard/mocap_neutrMosh_temporal/',
    'Output data directory for temporal'
)
tf.app.flags.DEFINE_integer('max_subsample_rate', 1,
                            'Maximum subsample rate for to compute delta pose.')
tf.app.flags.DEFINE_integer('length', 50,
                            'Length of sequence for delta poses.')
tf.app.flags.DEFINE_integer('num_shards', 10000,
                            'Number of shards in TFRecord files.')
FLAGS = tf.app.flags.FLAGS


def convert_to_example(pose, shape=None):
    """
    Build an Example proto for an image example.

    Args:
        pose (72).
        shape (10).

    Returns:
        Example proto.
    """
    if shape is None:
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'pose': float_feature(pose.astype(np.float))
            }))
    else:
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'pose': float_feature(pose.astype(np.float)),
                'shape': float_feature(shape.astype(np.float)),
            }))
    return example


def convert_to_example_temporal(poses):
    """
    Builds example proto for a changes in pose.

    Args:
        pose (50x24x3): Axis-angle representation of the current pose.

    Returns:
        Example proto.
    """
    return tf.train.Example(features=tf.train.Features(
        feature={
            'pose': float_feature(poses.ravel()),
        }
    ))


def compute_delta_poses(poses, max_subsample_rate, length):
    """
    Computes the delta poses for a single video, sampled at a frame skip of
    1 to max_subsample_rate.

    Args:
        poses (Nx72): Array of axis-angle rotations for each of 24 joints.
        max_subsample_rate (int): Maximum frame skip rate.

    Returns:
        Delta poses sampled at different frame skip rates.
    """
    thetas = poses.reshape((-1, 72))
    n = len(thetas)

    delta_poses = []
    for fs in range(1, max_subsample_rate + 1):
        for i in range(n - length * fs):
            delta_pose = thetas[i: i + length * fs: fs]
            delta_poses.append(delta_pose)

    if delta_poses:
        return np.vstack(delta_poses)


def process_smpl_mocap(all_pkls, out_dir, num_shards, dataset_name):
    all_poses, all_shapes, all_shapes_unique = [], [], []
    for pkl in all_pkls:
        with open(pkl, 'rb') as f:
            res = pickle.load(f)
            all_poses.append(res['poses'])
            num_poses_here = res['poses'].shape[0]
            all_shapes.append(
                np.tile(np.reshape(res['betas'], (10, 1)), num_poses_here))
            all_shapes_unique.append(res['betas'])

    all_poses = np.vstack(all_poses)
    all_shapes = np.hstack(all_shapes).T

    out_path = join(out_dir, '{}_{{:03d}}.tfrecord'.format(dataset_name))

    # shuffle results
    num_mocap = all_poses.shape[0]
    shuffle_id = np.random.permutation(num_mocap)
    all_poses = all_poses[shuffle_id]
    all_shapes = all_shapes[shuffle_id]

    i = 0
    fidx = 0
    while i < num_mocap:
        # Open new TFRecord file.
        tf_filename = out_path.format(fidx)
        print('Starting tfrecord file {}'.format(tf_filename))
        with tf.python_io.TFRecordWriter(tf_filename) as writer:
            j = 0
            while i < num_mocap and j < num_shards:
                if i % 10000 == 0:
                    print('Converting mosh {}/{}'.format(i, num_mocap))
                example = convert_to_example(all_poses[i], shape=all_shapes[i])
                writer.write(example.SerializeToString())
                i += 1
                j += 1

        fidx += 1


def process_smpl_mocap_temporal(all_pkls, out_dir, num_shards, dataset_name):
    all_poses = []
    for pkl in all_pkls:
        with open(pkl, 'rb') as f:
            res = pickle.load(f)
            if 'poses' not in res.keys():
                continue
            poses = res['poses']
            if dataset_name == 'neutrSMPL_H3.6':
                # 200 fps. Subsample to 25 fps.
                poses = poses[::8]
            elif dataset_name == 'neutrSMPL_CMU':
                # Pickle names are in format: subid_seqid.pkl.
                pkl_name = basename(pkl)
                sid = pkl_name[:pkl_name.index('_')]
                if sid in {'75', '106', '107', '141', '143'}:
                    # These subjects have 60 fps.
                    print('SID {} has 60 fps'.format(sid))
                    poses = poses[::2]
                else:
                    # These subjects have 120 fps.
                    poses = poses[::4]
            elif dataset_name == 'neutrSMPL_jointLim':
                # 120 fps. Subsample to 30 fps.
                poses = poses[::4]
            else:
                import ipdb; ipdb.set_trace()
                print('Dataset not found!')

            if len(poses) > 1:
                # There's one pkl with one pose in CMU.
                all_poses.append(poses)

    out_path = join(out_dir, '{}_{{:06d}}.tfrecord'.format(dataset_name))

    all_delta_poses = []
    for poses in tqdm(all_poses):
        delta_poses = compute_delta_poses(
            poses=poses,
            max_subsample_rate=FLAGS.max_subsample_rate,
            length=FLAGS.length,
        )
        if delta_poses is not None:
            all_delta_poses.append(delta_poses)

    all_delta_poses = np.vstack(all_delta_poses)

    n = len(all_delta_poses)
    print(n, 'total sequences')
    indices = np.arange(n)
    np.random.shuffle(indices)
    all_delta_poses = all_delta_poses[indices]

    i = 0
    fidx = 0
    while i < n:
        # Open new TFRecord file.
        tf_filename = out_path.format(fidx)
        print('Starting tfrecord file {}'.format(tf_filename))
        with tf.python_io.TFRecordWriter(tf_filename) as writer:
            j = 0
            while i < n and j < num_shards:
                if i % 10000 == 0:
                    print('Converting mosh {}/{}'.format(i, n))
                example = convert_to_example_temporal(
                    all_delta_poses[i]
                )
                writer.write(example.SerializeToString())
                i += 1
                j += 1
        fidx += 1


def main(unused_argv):
    data_dir = join(FLAGS.data_directory, FLAGS.dataset_name)
    # Ignore H3.6M test subjects!!
    all_pkls = sorted([
        f for f in glob(join(data_dir, '*/*.pkl'))
        if 'S9' not in f and 'S11' not in f
    ])
    if len(all_pkls) == 0:
        print('Something is wrong with the path bc I cant find any pkls!')
        import ipdb; ipdb.set_trace()

    if FLAGS.temporal:
        print('Saving results to {}'.format(FLAGS.output_directory_temporal))
        if not exists(FLAGS.output_directory_temporal):
            makedirs(FLAGS.output_directory_temporal)
        process_smpl_mocap_temporal(
            all_pkls=all_pkls,
            out_dir=FLAGS.output_directory_temporal,
            num_shards=FLAGS.num_shards,
            dataset_name=FLAGS.dataset_name
        )
    else:
        print('Saving results to {}'.format(FLAGS.output_directory))
        if not exists(FLAGS.output_directory):
            makedirs(FLAGS.output_directory)
        process_smpl_mocap(
            all_pkls=all_pkls,
            out_dir=FLAGS.output_directory,
            num_shards=FLAGS.num_shards,
            dataset_name=FLAGS.dataset_name
        )


if __name__ == '__main__':
    tf.app.run()
