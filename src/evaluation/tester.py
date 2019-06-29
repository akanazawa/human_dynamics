from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import deepdish as dd
import ipdb
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from src.evaluation.eval_util import update_dict_entries
from src.models import (
    batch_pred_omega,
    get_hallucinator_model,
    get_image_encoder,
    get_temporal_encoder,
)
from src.omega import OmegasPred
from src.tf_smpl.batch_smpl import SMPL


class Tester(object):

    def __init__(self, config, pretrained_resnet_path='', sequence_length=None):
        self.config = config
        self.load_path = config.load_path

        # Config + path.
        if not config.load_path:
            raise Exception(
                '[!] You need to specify `load_path` to load a pretrained model'
            )
        if not os.path.exists(config.load_path + '.index'):
            print('{} doesnt exist..'.format(config.load_path))
            import ipdb
            ipdb.set_trace()

        # Model parameters.
        self.batch_size = config.batch_size
        if sequence_length:
            self.sequence_length = sequence_length
        else:
            self.sequence_length = config.sequence_length
        self.pred_mode = config.pred_mode
        self.num_conv_layers = config.num_conv_layers
        self.fov = self.num_conv_layers * 4 + 1

        self.delta_t_values = [int(dt) for dt in config.delta_t_values]

        # Data parameters.
        self.img_size = 224

        # Other parameters.
        self.num_output = 85
        self.smpl_model_path = config.smpl_model_path
        self.smpl = SMPL(self.smpl_model_path)
        self.smpl_model_path = config.smpl_model_path

        self.encoder_vars = []

        # Prepare model.
        input_size = (self.batch_size, self.sequence_length,
                      self.img_size, self.img_size, 3)
        self.images_pl = tf.placeholder(tf.float32, shape=input_size)
        self.f_hal = get_hallucinator_model()
        self.f_image_enc = get_image_encoder()
        self.f_temporal_enc = get_temporal_encoder()

        self.omegas_pred = {
            0: self.make_omega_pred(use_optcam=False)
        }
        for dt in self.delta_t_values:
            self.omegas_pred[dt] = self.make_omega_pred(use_optcam=True)

        # Starting point for IEF.
        mean_cams, mean_shape, mean_pose = self.load_mean_params()
        self.theta_mean = tf.concat((
            mean_cams,
            tf.reshape(mean_pose, (-1, 72)),
            mean_shape
        ), axis=1)
        self.build_test_model()

        # Smaller fraction will take up less GPU space, but might be slower.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        self.prepare(resnet_path=pretrained_resnet_path)

    def prepare(self, resnet_path=''):
        """
        Restores variables from checkpoint.

        Args:
            resnet_path (str): Optional path to load resnet weights.
        """
        if resnet_path:
            print('Restoring resnet vars from', resnet_path)
            resnet_vars = []
            e_vars = []
            for var in self.encoder_vars:
                if 'resnet' in var.name:
                    resnet_vars.append(var)
                else:
                    e_vars.append(var)
            resnet_saver = tf.train.Saver(resnet_vars)
            resnet_saver.restore(self.sess, resnet_path)
        else:
            e_vars = self.encoder_vars
        print('Restoring checkpoint ', self.load_path)

        saver = tf.train.Saver(e_vars)
        saver.restore(self.sess, self.load_path)
        self.sess.run(self.theta_mean)

    def load_mean_params(self):
        # Initialize scale at 0.9
        mean_path = os.path.join(os.path.dirname(self.smpl_model_path),
                                 'neutral_smpl_meanwjoints.h5')
        mean_vals = dd.io.load(mean_path)

        mean_cams = [0.9, 0, 0]
        # 72D
        mean_pose = mean_vals['pose']
        mean_pose[:3] = 0.
        mean_pose[0] = np.pi
        # 10D
        mean_shape = mean_vals['shape']

        # trainable vars- let's learn the best mean pose.
        mean_vals = np.hstack((mean_cams, mean_pose, mean_shape))
        # Needs to be 1 x 85
        mean_vals = np.expand_dims(mean_vals, 0)
        mean_var = tf.Variable(
            mean_vals,
            name='mean_param',
            dtype=tf.float32,
            trainable=True
        )
        mean_cams = mean_var[0, :3]
        mean_pose = mean_var[0, 3:3+72]
        mean_shape = mean_var[0, 3+72:]

        mean_cams = tf.tile(tf.reshape(mean_cams, (1, -1)),
                            (self.batch_size, 1))
        mean_shape = tf.tile(tf.reshape(mean_shape, (1, -1)),
                             (self.batch_size, 1))
        mean_pose = tf.tile(tf.reshape(mean_pose, (1, 72)),
                            (self.batch_size, 1))
        return mean_cams, mean_shape, mean_pose

    def make_omega_pred(self, use_optcam=False):
        return OmegasPred(
            config=self.config,
            smpl=self.smpl,
            use_optcam=use_optcam,
            vis_max_batch=self.batch_size,
            is_training=False,
        )

    def update_encoder_vars(self):
        trainable_vars = tf.contrib.framework.get_variables()
        trainable_vars_e = [var for var in trainable_vars
                            if var.name[:2] != 'D_']
        self.encoder_vars.extend(trainable_vars_e)

    def build_test_model(self):
        B, T = self.batch_size, self.sequence_length
        I_t = tf.reshape(
            self.images_pl,
            (B * T, self.img_size, self.img_size, 3)
        )
        img_feat, phi_var_scope = self.f_image_enc(
            I_t,
            is_training=False,
            reuse=False,
        )
        img_feat_full = tf.reshape(img_feat, (B, T, -1))
        omega_mean = tf.tile(self.theta_mean, (self.sequence_length, 1))

        if self.pred_mode == 'pred':
            movie_strips = self.f_temporal_enc(
                is_training=False,
                net=img_feat_full,
                num_conv_layers=self.num_conv_layers,
            )
        elif self.pred_mode == 'hal':
            movie_strips = self.f_hal(img_feat_full)
        else:
            raise Exception(
                'Pred mode {} not recognized'.format(self.pred_mode)
            )

        omegas_raw, deltas_pred = batch_pred_omega(
            input_features=movie_strips,
            batch_size=self.batch_size,
            sequence_length=self.sequence_length,
            num_output=self.num_output,
            is_training=False,
            omega_mean=omega_mean,
            scope='single_view_ief',
            predict_delta_keys=self.omegas_pred.keys(),
            use_optcam=True,
            use_delta_from_pred=True,
        )
        self.omegas_pred[0].append_batched(omegas_raw)
        for k in deltas_pred.keys():
            self.omegas_pred[k].append_batched(deltas_pred[k])
            self.omegas_pred[k].set_cams(
                self.omegas_pred[0].get_cams()
            )
        OmegasPred.compute_all_smpl()
        self.update_encoder_vars()

    def make_fetch_dict(self, omegas, suffix=''):
        return {
            # Predictions.
            'cams' + suffix: omegas.get_cams(),
            'joints' + suffix: omegas.get_joints(),
            'kps' + suffix: omegas.get_kps(),
            'poses' + suffix: omegas.get_poses_rot(),
            'shapes' + suffix: omegas.get_shapes(),
            'verts' + suffix: omegas.get_verts(),
            'omegas' + suffix: omegas.get_raw(),
        }

    def predict(self, images):
        """
        Runs forward pass of model.

        Args:
            images (BxTxHxWx3): Images to predict.

        Returns:
            dict.
        """
        feed_dict = {
            self.images_pl: images,
        }
        fetch_dict = self.make_fetch_dict(self.omegas_pred[0])

        fetch_dict_deltas = {}
        for delta_t, omega_delta in sorted(self.omegas_pred.items()):
            if delta_t == 0:
                continue
            update_dict_entries(
                accumulator=fetch_dict_deltas,
                appender=self.make_fetch_dict(omega_delta, suffix='_delta')
            )
        # DxBxTx... --> BxTxDx...
        for k in fetch_dict_deltas:
            fetch_dict_deltas[k] = tf.stack(fetch_dict_deltas[k], axis=2)
        fetch_dict.update(fetch_dict_deltas)

        results = self.sess.run(fetch_dict, feed_dict)
        return results

    def predict_all_images(self, all_images):
        """
        Wrapper to predict entire sequence.

        Because of edge padding, images at edges will have low quality
        predictions since they don't have full field-of-view. Thus, we slide
        a window of size T across the images and only keep the predictions
        with full fov.

        Args:
            all_images (NxHxWx3): Images in sequence.

        Returns:
            dict
        """
        B = self.batch_size
        T = self.sequence_length
        N = len(all_images)
        H, W = self.img_size, self.img_size

        # Need margin on both sides. Num good frames = T - 2 * margin.
        margin = (self.fov - 1) // 2
        g = self.sequence_length - 2 * margin
        count = np.ceil(N / (g * B)).astype(int)
        num_fill = count * B * g + T - N
        images_padded = np.concatenate((
            np.zeros((margin, H, W, 3)),             # Front padding.
            all_images,
            np.zeros((num_fill, H, W, 3)),           # Back padding.
        ), axis=0)
        images_batched = []
        # [ m ][    g    ][ m ]             Slide over by g every time.
        #            [ m ][    g    ][ m ]
        for i in range(count * B):
            images_batched.append(images_padded[i * g : i * g + T])
        images_batched = np.reshape(images_batched, (count, B, T, H, W, 3))

        results = {}

        for images in tqdm(images_batched):
            pred = self.predict(
                images,
            )
            update_dict_entries(results, pred)

        # Results are now (CxBxTx...). Should be (Nx...).
        new_results = {}
        for k, v in results.items():
            v = np.array(v)[:, :, margin : -margin]
            old_shape = v.shape[3:]
            new_v = v.reshape((-1,) + old_shape)[:N]
            new_results[k] = new_v
        return new_results
