"""
Data loader with data augmentation.
Only used for training.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, basename
from glob import glob

import tensorflow as tf

from src.tf_smpl.batch_lbs import batch_rodrigues
from src.util import (
    data_utils,
    tube_augmentation,
)

_3D_DATASETS = ['h36m']


class SequenceDataLoader(object):
    def __init__(self, config):
        self.config = config

        self.use_3d_label = config.use_3d_label
        self.precomputed_phi = config.precomputed_phi

        self.dataset_dir = config.data_dir
        self.datasets = config.datasets
        self.mocap_datasets = config.mocap_datasets
        self.batch_size = config.batch_size
        self.data_format = config.data_format
        self.T = config.T

        if not config.precomputed_phi:
            self.image_preprocessor = tube_augmentation.TubePreprocessor(
                config.img_size,
                config.trans_max,
                config.delta_trans_max,
                config.scale_max,
                config.delta_scale_max,
                config.rotate_max,
                config.delta_rotate_max,
            )

        self.predict_delta = config.predict_delta
        self.delta_t_values = config.delta_t_values
        self.num_kps = config.num_kps
        self.do_hallucinate = config.do_hallucinate
        self.do_hallucinate_preds = config.do_hallucinate_preds
        self.split_balanced = config.split_balanced


    def load(self):
        image_loader = self.get_loader()
        return image_loader

    def get_loader(self):
        """
        Returns:
            batch_dict (dict):
                image (BxTxHxWx3).
                label (BxTx19x3).
                pose (BxTx24x3x3).
                shape (Bx10).
                fnames (BxT).
                joint (BxTx14x3).
                has3d (Bx2).
        """
        if self.split_balanced:
            datasets_2d = [d for d in self.datasets if d not in _3D_DATASETS]
            datasets_3d = [d for d in self.datasets if d in _3D_DATASETS]
        else:
            datasets_2d = [d for d in self.datasets]
            datasets_3d = datasets_2d[::-1]

        files_2d = data_utils.get_all_files(self.dataset_dir, datasets_2d)
        files_3d = data_utils.get_all_files(self.dataset_dir, datasets_3d)

        def split_list(list_in):
            mid_way = int(len(list_in) / 2)
            return list_in[:mid_way], list_in[mid_way:]

        # Split files_3d in two if one is empty.
        if len(files_2d) == 0:
            files_2d, files_3d = split_list(files_3d)
        elif len(files_3d) == 0:
            files_2d, files_3d = split_list(files_2d)

        do_shuffle = True
        fqueue_2d = tf.train.string_input_producer(
            files_2d,
            shuffle=do_shuffle,
            name='input_2d',
            capacity=128
        )
        fqueue_3d = tf.train.string_input_producer(
            files_3d,
            shuffle=do_shuffle,
            name='input_3d',
            capacity=128
        )

        ret_dict_2d = self.read_data(fqueue_2d)
        ret_dict_3d = self.read_data(fqueue_3d)

        if self.precomputed_phi:
            pack_name = ['phis', 'images']
        else:
            if self.data_format == 'NCHW':
                # TxHxWx3 --> Tx3xHxW
                ret_dict_2d['images'] = tf.transpose(ret_dict_2d['images'],
                                                     (0, 3, 1, 2))
                ret_dict_3d['images'] = tf.transpose(ret_dict_3d['images'],
                                                     (0, 3, 1, 2))
            elif self.data_format == 'NHWC':
                pass
            else:
                raise ValueError('Data format {} not recognized!'.format(
                    self.data_format))
            pack_name = ['images']

        min_after_dequeue = 32
        num_threads = 4
        capacity = min_after_dequeue + (num_threads + 10) * self.batch_size

        pack_name.extend(
            ['labels', 'fnames', 'poses', 'shape', 'gt3ds', 'has_3d'])

        # parallel_stack can't handle bool..
        ret_dict_2d['has_3d'] = tf.cast(
            ret_dict_2d['has_3d'], dtype=tf.float32)
        ret_dict_3d['has_3d'] = tf.cast(
            ret_dict_3d['has_3d'], dtype=tf.float32)
        # Stack 2d and 3d data.
        pack_these = [
            tf.parallel_stack([ret_dict_2d[key], ret_dict_3d[key]])
            for key in pack_name
        ]
        pack_these[pack_name.index('has_3d')] = tf.cast(
            pack_these[pack_name.index('has_3d')], dtype=tf.bool)

        all_batched = tf.train.shuffle_batch(
            pack_these,
            batch_size=self.batch_size,
            num_threads=num_threads,
            capacity=capacity,
            min_after_dequeue=min_after_dequeue,
            enqueue_many=True,
            name='input_batch_train')
        batch_dict = {}
        for name, batch in zip(pack_name, all_batched):
            batch_dict[name] = batch

        return batch_dict

    def get_smpl_loader(self):
        """
        Loads real pose and shape parameters from mocap data.

        Returns:
            pose_batch (BxTx216).
            shape_batch (BxTx10).
        """
        data_dirs = [
            join(self.dataset_dir, 'mocap_neutrMosh',
                 'neutrSMPL_{}_*.tfrecord'.format(dataset))
            for dataset in self.mocap_datasets
        ]
        files = []
        for data_dir in data_dirs:
            files += glob(data_dir)

        if len(files) == 0:
            print('Couldnt find any files!!')
            import ipdb
            ipdb.set_trace()

        with tf.name_scope('input_smpl_loader'):
            filename_queue = tf.train.string_input_producer(
                files, shuffle=True, capacity=128)

            mosh_batch_size = self.batch_size * self.T
            if self.predict_delta:
                new_t = len(self.delta_t_values) * self.T
                batch_size_delta = self.batch_size * new_t
            else:
                batch_size_delta = 0

            if self.do_hallucinate:
                mosh_batch_size *= 2
                if self.do_hallucinate_preds:
                    batch_size_delta *= 2
            mosh_batch_size += batch_size_delta
            if self.config.use_hmr_only:
                # /2 bc doesnt have the movie-strip branch.
                mosh_batch_size = int(mosh_batch_size / 2)

            if self.config.use_hmr_only and not self.do_hallucinate:
                mosh_batch_size = self.T*self.batch_size

            min_after_dequeue = 32
            num_threads = 2
            capacity = min_after_dequeue + (num_threads + 4) * mosh_batch_size

            pose, shape = data_utils.read_smpl_data(filename_queue)
            pose = batch_rodrigues(tf.reshape(pose, (-1, 3)))
            pose_batch, shape_batch = tf.train.batch(
                [pose, shape],
                batch_size=mosh_batch_size,
                num_threads=num_threads,
                capacity=capacity,
                name='input_smpl_batch')

            pose_batch = tf.reshape(pose_batch, (mosh_batch_size, 216))
            shape_batch = tf.reshape(shape_batch, (mosh_batch_size, 10))
            return pose_batch, shape_batch

    def get_smpl_loader_temporal(self):
        """
        Loads real delta poses from mocap data.

        Returns:
            delta_pose_batch (BxTx216).
        """
        data_dirs = [
            join(self.dataset_dir, 'mocap_neutrMosh_temporal_pose',
                 'neutrSMPL_{}_*.tfrecord'.format(dataset))
            for dataset in self.mocap_datasets
        ]
        files = []
        for data_dir in data_dirs:
            files += glob(data_dir)

        if len(files) == 0:
            print('Couldnt find any files!!')
            import ipdb
            ipdb.set_trace()

        with tf.name_scope('input_smpl_loader_temporal'):
            filename_queue = tf.train.string_input_producer(
                files, shuffle=True, capacity=128)

            mosh_batch_size = self.batch_size * self.T

            min_after_dequeue = 32
            num_threads = 2
            capacity = min_after_dequeue + (num_threads + 4) * mosh_batch_size

            pose, deltas = data_utils.read_smpl_data_temporal(filename_queue)
            pose = batch_rodrigues(tf.reshape(pose, (-1, 3)))
            delta_pose = batch_rodrigues(tf.reshape(deltas, (-1, 3)))
            pose_batch, delta_pose_batch = tf.train.batch(
                [pose, delta_pose],
                batch_size=mosh_batch_size,
                num_threads=num_threads,
                capacity=capacity,
                name='input_smpl_batch')

            return (tf.reshape(pose_batch, (self.batch_size, self.T, 216)),
                    tf.reshape(delta_pose_batch,
                               (self.batch_size, self.T, 216)))


    def read_data(self, filename_queue):
        """
        Reads data from given filename queue and pre-processes the data.

        Args:
            filename_queue: Queue of filename strings.

        Returns:
            dict:
                images/phis (TxHxWx3)/(Tx2048).
                labels (Tx19x3).
                fnames (T).
                poses (Tx24x3).
                shape (10).
                joints (Tx14x3).
                has_3d (2x1).
            if self.precomputed_phi, also returns phis (T x 2048)
        """
        with tf.name_scope(None, 'read_data', [filename_queue]):
            reader = tf.TFRecordReader()
            _, example_serialized = reader.read(filename_queue)

            # keys: images/phis, image_sizes, labels, centers, fnames, poses,
            # shape, gt3ds, has_3d
            ret_dict = data_utils.parse_example_proto_temporal(
                example_serialized,
                T=self.T,
                precomputed_phi=self.precomputed_phi,
            )
            ret_dict['labels'] = ret_dict['labels'][:, :, :self.num_kps]

            if not self.precomputed_phi:
                # Need to send pose bc image can get flipped.
                ret_dict.update(
                    self.image_preprocessor(
                        images=ret_dict['images'],
                        image_sizes=ret_dict['image_sizes'],
                        labels=ret_dict['labels'],
                        centers=ret_dict['centers'],
                        poses=ret_dict['poses'],
                        gt3ds=ret_dict['gt3ds'],
                    ))
            else:
                if 'images' in ret_dict.keys():
                    ret_dict['images'] = tf.map_fn(data_utils.rescale_image,
                                                   ret_dict['images'])

            # label should be T x K x 3.
            ret_dict['labels'] = tf.transpose(ret_dict['labels'], [0, 2, 1])

            return ret_dict
