"""
HMR sequence trainer.
From a sequence of images input, trained a model that..
trainer_sequence.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from io import BytesIO
import os
from os.path import (
    basename,
    dirname,
    join,
)
import pickle
from time import time

import deepdish as dd
import ipdb
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.ops import control_flow_ops
import tensorflow as tf

from src.discriminators import PoseDiscriminator
from src.models import (
    batch_pred_omega,
    get_image_encoder,
    get_hallucinator_model,
    get_temporal_encoder,
)
from src.omega import (
    OmegasGt,
    OmegasPred,
)
from src.ops import (
    call_hmr_ief,
    compute_loss_d_fake,
    compute_loss_d_real,
    compute_loss_e_3d,
    compute_loss_e_fake,
    compute_loss_e_kp,
    compute_loss_e_kp_optcam,
    compute_loss_e_smooth,
    compute_loss_shape,
)
from src.tf_smpl.batch_smpl import SMPL
from src.util.data_utils import tf_repeat
import src.util.render.nmr_renderer as vis_util


class HMRSequenceTrainer(object):
    def __init__(self, config, data_loader, mocap_loader):
        """
        Args:
          config: carries configuration parameters (as flags), and prepare_dirs
          data_loader: SequenceDataLoader load- image_loader returns a dict:
            {'labels': (BxTx25x3), 'fnames': (BxT), 'poses': (BxTx24x3),
             'shape': (BxTx10), 'gt3ds': (BxTx25x3), 'has_3d': (Bx2)}
            + either {'images': (BxTx224x224x3)} or {'phis': (BxTx2048)}
          mocap_loader: tuple (pose, shape)
        """
        # Config + path
        self.config = config
        self.model_dir = config.model_dir
        self.load_path = config.load_path

        self.data_format = config.data_format
        self.smpl_model_path = config.smpl_model_path
        self.pretrained_model_path = config.pretrained_model_path
        self.use_3d_label = config.use_3d_label
        self.use_hmr_ief_init = config.use_hmr_ief_init
        self.freeze_phi = config.freeze_phi
        self.predict_delta = config.predict_delta
        self.delta_t_values = [int(dt) for dt in config.delta_t_values]

        self.use_delta_from_pred = config.use_delta_from_pred
        self.use_hmr_only = config.use_hmr_only
        self.do_hallucinate = config.do_hallucinate
        self.do_hallucinate_preds = config.do_hallucinate_preds

        self.fov = config.num_conv_layers * 4 + 1

        self.smpl = SMPL(self.smpl_model_path)

        # Data size
        self.img_size = config.img_size
        self.batch_size = config.batch_size
        self.max_iteration = config.max_iteration
        self.sequence_length = config.T

        # Visualization parameters.
        self.vis_max_batch = min(2, self.batch_size)
        # Just show consecutive frames.
        if config.log_img_count > self.sequence_length:
            self.vis_t_indices = range(self.sequence_length)
        else:
            mid_point = self.sequence_length // 2
            self.vis_t_indices = np.arange(
                mid_point - config.log_img_count // 2,
                mid_point + config.log_img_count // 2,
            )

        # Data
        self.image_loader = data_loader['images']
        # For visualization.
        self.images_all = self.image_loader[:self.vis_max_batch]

        # Instantiate omega.
        self.omegas_gt = OmegasGt(
            config=config,
            poses_aa=data_loader['poses'],
            shapes=data_loader['shape'],
            joints=data_loader['gt3ds'],
            kps=data_loader['labels'])
        self.omegas_pred = self.make_omega_pred()

        # For rendering!
        self.renderer = vis_util.VisRenderer(
            img_size=self.img_size, face_path=self.config.smpl_face_path)
        # For 2D regressor.
        with open(self.smpl_model_path, 'rb') as f:
            dd = pickle.load(f, encoding='latin-1')
            self.regressor = dd['cocoplus_regressor']

        self.omegas_delta = {}
        if self.predict_delta:
            for dt in self.delta_t_values:
                self.omegas_delta[dt] = self.make_omega_pred(use_optcam=True)

        self.omegas_pred_hal = {}
        if self.do_hallucinate:
            # Current prediction always has to use its own camera.
            self.omegas_pred_hal[0] = self.make_omega_pred(use_optcam=False)
            if self.do_hallucinate_preds:
                for dt in self.delta_t_values:
                    self.omegas_pred_hal[dt] = self.make_omega_pred(
                        use_optcam=True)

        self.has3d = data_loader['has_3d']
        self.has_gt3d_joints = self.has3d[:, 0]
        self.has_gt3d_smpl = self.has3d[:, 1]

        if self.config.mosh_ignore:
            print('ignoring mosh')
            self.has_gt3d_smpl = tf.zeros_like(self.has_gt3d_smpl)

        # keep track of image features
        self.img_feat_full = []

        # mocap_loader - real samples for training discriminators.
        self.poses_real_loader = mocap_loader[0]
        self.shape_real_loader = mocap_loader[1]

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.log_img_step = config.log_img_step

        # Pre-computed phi?
        self.precomputed_phi = config.precomputed_phi
        # Model spec
        if self.precomputed_phi:
            self.phi_loader = data_loader['phis']

        # TODO
        self.image_encoder_model_type = config.image_encoder_model_type
        self.f_image_enc = get_image_encoder(self.image_encoder_model_type)

        if self.do_hallucinate:
            self.hallucinator_model_type = config.hallucinator_model_type
            self.f_hal = get_hallucinator_model(
                model_type=self.hallucinator_model_type)

        if self.use_hmr_only and not self.do_hallucinate:
            # I.e. local only.
            # Setup omegas, still use normal data loader
            self.batch_size_static = self.batch_size * self.sequence_length
            poses_aa_gt = tf.reshape(data_loader['poses'],
                                     (self.batch_size_static, -1))
            poses_aa_gt = tf.expand_dims(poses_aa_gt, 1)
            # Replicate shape..
            shapes_gt = tf.tile(data_loader['shape'],
                                (self.sequence_length, 1))
            gt3ds_gt = tf.reshape(data_loader['gt3ds'],
                                  (self.batch_size_static, 1, -1, 3))
            kps_gt = tf.reshape(data_loader['labels'],
                                (self.batch_size_static, 1, -1, 3))
            self.omegas_gt_static = OmegasGt(
                config=config,
                poses_aa=poses_aa_gt,
                shapes=shapes_gt,
                joints=gt3ds_gt,
                kps=kps_gt,
                batch_size=self.batch_size_static,
            )
            # poses_rot should be B*T x 1 x 24 x 3 x 3
            # shape should be B*T x 1 x 10
            self.omegas_static = OmegasPred(
                config=config,
                smpl=self.smpl,
                vis_max_batch=self.vis_max_batch,
                batch_size=self.batch_size_static,
            )
            has3d = tf.tile(
                tf.expand_dims(data_loader['has_3d'], 1),
                [1, self.sequence_length, 1])
            has3d = tf.reshape(has3d, (self.batch_size_static, -1))
            self.has_gt3d_joints_static = has3d[:, 0]
            self.has_gt3d_smpl_static = has3d[:, 1]
            # For vis.
            self.images_static = tf.reshape(
                data_loader['images'], (self.batch_size_static, self.img_size,
                                        self.img_size, 3))[:self.vis_max_batch]

        self.temporal_encoder_type = config.temporal_encoder_type
        self.f_temporal_enc = get_temporal_encoder(
            model_type=self.temporal_encoder_type
        )

        self.num_conv_layers = config.num_conv_layers

        # Optimizer, learning rate
        self.e_lr = config.e_lr
        self.d_lr = config.d_lr
        # Weight decay
        self.e_wd = config.e_wd
        self.d_wd = config.d_wd

        # Discriminators
        self.disc_pose = PoseDiscriminator(self.d_wd)
        self.use_disc_pose = config.d_lw_pose > 0.

        # Losses
        self.losses = {
            'd_pose': tf.constant(0.),
            'e_const': tf.constant(0.),
            'e_joints': tf.constant(0.),
            'e_kp': tf.constant(0.),
            'e_pose': tf.constant(0.),
            'e_shape': tf.constant(0.),
            'e_smpl': tf.constant(0.),
        }
        if self.use_hmr_only and not self.do_hallucinate:
            self.losses.update({
                'e_kp_static': tf.constant(0.),
                'e_joints_static': tf.constant(0.),
                'e_smpl_static': tf.constant(0.),
            })
        if self.predict_delta:
            self.losses.update({
                'e_joints_dt_future': tf.constant(0.),
                'e_kp_dt_future': tf.constant(0.),
                'e_smpl_dt_future': tf.constant(0.),
                'e_joints_dt_past': tf.constant(0.),
                'e_kp_dt_past': tf.constant(0.),
                'e_smpl_dt_past': tf.constant(0.),
            })
        if self.do_hallucinate:
            self.losses.update({
                'e_hallucinate': tf.constant(0.),
                'e_joints_hal': tf.constant(0.),
                'e_kp_hal': tf.constant(0.),
                'e_smpl_hal': tf.constant(0.),
            })
            if self.do_hallucinate_preds:
                self.losses.update({
                    'e_joints_hal_future': tf.constant(0.),
                    'e_kp_hal_future': tf.constant(0.),
                    'e_smpl_hal_future': tf.constant(0.),
                    'e_joints_hal_past': tf.constant(0.),
                    'e_kp_hal_past': tf.constant(0.),
                    'e_smpl_hal_past': tf.constant(0.),
                })

        self.e_loss = 0.
        self.d_loss = 0.

        # Loss weights
        self.loss_weights = {
            'd_pose': config.d_lw_pose,
            'e_const': config.e_lw_const,
            'e_joints': config.e_lw_joints,
            'e_kp': config.e_lw_kp,
            'e_pose': config.e_lw_pose,
            'e_shape': config.e_lw_shape,
            'e_smpl': config.e_lw_smpl,
            # For static.
            'e_kp_static': config.e_lw_kp,
            'e_joints_static': config.e_lw_joints,
            'e_smpl_static': config.e_lw_smpl,
            # For delta_t:
            'e_joints_dt_future': config.e_lw_joints,
            'e_kp_dt_future': config.e_lw_kp,
            'e_smpl_dt_future': config.e_lw_smpl,
            'e_joints_dt_past': config.e_lw_joints,
            'e_kp_dt_past': config.e_lw_kp,
            'e_smpl_dt_past': config.e_lw_smpl,
            # For hallucinating
            'e_hallucinate': config.e_lw_hallucinate,
            'e_joints_hal': config.e_lw_joints,
            'e_kp_hal': config.e_lw_kp,
            'e_smpl_hal': config.e_lw_smpl,
            'e_joints_hal_future': config.e_lw_joints,
            'e_kp_hal_future': config.e_lw_kp,
            'e_smpl_hal_future': config.e_lw_smpl,
            'e_joints_hal_past': config.e_lw_joints,
            'e_kp_hal_past': config.e_lw_kp,
            'e_smpl_hal_past': config.e_lw_smpl,
        }

        # Containers to carry around 'fake' until adv loss computation.
        self.pred_poses_all = []
        self.pred_shapes_all = []

        # Loss proportions.
        self.loss_proportions = {}

        # HMR Model Params
        self.num_stage = config.num_stage
        self.num_output = 85

        self.summaries_list = []
        self.summaries = None

        self.optimizer = tf.train.AdamOptimizer

        # Instantiate omega template
        self.E_var = []
        self.D_var = []

        self.load_mean_omega()
        if config.use_hmr_only:
            self.build_hmr_model()
        else:
            self.build_model()
        self.setup_optimizers()

        # Logging
        init_fn = None
        if self.use_pretrained():
            # Make custom init_fn
            restore_vars = []
            restore_vars_dt = []
            for pmp in self.pretrained_model_path:
                print('Fine-tuning from {}'.format(pmp))
                ft_from_hmmr = (basename(pmp) != 'hmr_noS5.ckpt-642561'
                                and 'resnet_v2_50' not in pmp)
                if 'resnet_v2_50' in pmp:
                    resnet_vars = [
                        var for var in self.E_var if 'resnet_v2_50' in var.name
                    ]
                    restore_vars = restore_vars + resnet_vars
                elif 'pose-tensorflow' in pmp:
                    resnet_vars = [
                        var for var in self.E_var
                        if 'resnet_v1_101' in var.name
                    ]
                    restore_vars = restore_vars + resnet_vars
                elif 'hmr_' in pmp:
                    resnet_vars = [
                        var for var in self.E_var if 'resnet_v2_50' in var.name
                    ]
                    hmr_vars = [
                        var for var in self.E_var
                        if 'single_view_ief' in var.name
                        or 'mean_param' in var.name
                    ]

                    if self.precomputed_phi:
                        restore_vars = {}
                    else:
                        restore_vars = {v.name[:-2]: v for v in resnet_vars}
                    # Publically trained HMR's IEF name starts with 3D_module
                    # We prepend the scope with single_view_ief, so remove it.
                    # Also, we re-use ops so they have :0 at end.
                    restore_vars_dt = [
                        {} for _ in range(len(self.delta_t_values))
                    ]
                    for v in hmr_vars:
                        # only do this if it's not from finetuning from hmmr.
                        if 'single_view_ief/' in v.name and not ft_from_hmmr:
                            index = v.name.index('/') + 1
                        elif 'single_view_ief' in v.name:
                            continue
                        else:
                            index = 0
                        # import ipdb; ipdb.set_trace()
                        restore_vars[v.name[index:-2]] = v
                elif 'hmmr' in pmp:
                    restore_vars = self.E_var

            self.pre_train_savers = []
            if restore_vars:
                self.pre_train_savers.append(tf.train.Saver(restore_vars))

            if restore_vars_dt:
                for rv in restore_vars_dt:
                    if rv:
                        self.pre_train_savers.append(tf.train.Saver(rv))

            def load_pretrain(sess):
                for pmp in self.pretrained_model_path:
                    print('loading from {}'.format(pmp))
                    for saver in self.pre_train_savers:
                        saver.restore(sess, pmp)

            init_fn = load_pretrain

        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=5)
        self.summary_writer = tf.summary.FileWriter(self.model_dir)
        self.sv = tf.train.Supervisor(
            logdir=self.model_dir,
            global_step=self.global_step,
            saver=self.saver,
            summary_writer=self.summary_writer,
            init_fn=init_fn,
        )
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess_config = tf.ConfigProto(
            allow_soft_placement=False,
            log_device_placement=False,
            gpu_options=gpu_options,
        )

    def use_pretrained(self):
        """
        Returns true only if:
          1. model_type is "resnet"
          2. pretrained_model_path is not None
          3. model_dir is NOT empty. If it is, we're picking up from previous
             so fuck this pretrained model.
        """
        use_resnet = 'resnet' in self.image_encoder_model_type
        has_pmp = self.pretrained_model_path is not None

        if not self.precomputed_phi and use_resnet and has_pmp:
            # Check is model_dir is empty
            if len(os.listdir(self.model_dir)) == 0:
                return True
        elif self.precomputed_phi and self.use_hmr_ief_init:
            # Check is model_dir is empty
            if os.listdir(self.model_dir) == []:
                return True
        return False

    def load_mean_omega(self):
        # Initialize scale at 0.9
        mean_path = join(
            dirname(self.smpl_model_path), 'neutral_smpl_meanwjoints.h5')
        mean_vals = dd.io.load(mean_path)

        # mean cams
        mean_cams = [0.9, 0, 0]

        # mean pose
        mean_pose = mean_vals['pose']
        # Ignore the global rotation.
        mean_pose[:3] = 0.
        # This initializes the global pose to be up-right when projected
        mean_pose[0] = np.pi

        # mean shape
        mean_shape = mean_vals['shape']

        # For use_hmr_ief_init:
        # Initialize with learned ief if available..?
        mean_vals = np.hstack((mean_cams, mean_pose, mean_shape))
        # Needs to be 1 x 85
        mean_vals = np.expand_dims(mean_vals, 0)
        self.mean_var = tf.Variable(
            mean_vals, name='mean_param', dtype=tf.float32, trainable=True)

        # Starting point for IEF. (1 x 85 -> B x 85)
        self.theta_mean = tf.tile(self.mean_var, (self.batch_size, 1))

        if self.use_hmr_only and not self.do_hallucinate:
            self.theta_mean_hmr = tf.tile(
                self.mean_var, (self.sequence_length * self.batch_size, 1))

    def make_omega_pred(self, use_optcam=False):
        return OmegasPred(
            config=self.config,
            smpl=self.smpl,
            vis_max_batch=self.vis_max_batch,
            vis_t_indices=self.vis_t_indices,
            use_optcam=use_optcam,
        )

    def build_hmr_model(self):
        # Just calls hmr.

        if self.do_hallucinate:
            print('Building HMR model with hallucinate..')
            img_feat = self.phi_loader
            # encode this if hallucinating.
            movie_strips_hal = self.f_hal(img_feat)
            # batch_pred_omega computes:
            # omega_pred_hal (curr pred)
            # delta_pred_hal (preds on delta timesteps)
            # If opt_cam is on, delta_preds_hal are still 85 but
            # the camera is just [1, 0, 0]
            omega_pred_hal, delta_preds_hal = batch_pred_omega(
                input_features=movie_strips_hal,
                batch_size=self.batch_size,
                is_training=True,
                num_output=self.num_output,
                omega_mean=tf.tile(self.theta_mean, (self.sequence_length, 1)),
                predict_delta_keys=self.omegas_pred_hal.keys(),
                scope='single_view_ief',
                sequence_length=self.sequence_length,
                use_delta_from_pred=self.use_delta_from_pred,
                use_optcam=True,
            )

            for k in delta_preds_hal:
                self.omegas_pred_hal[k].append_batched(delta_preds_hal[k])
            self.omegas_pred_hal[0].append_batched(omega_pred_hal)

            # For each prediction, compute smpl
            for omegas in self.omegas_pred_hal.values():
                omegas.compute_smpl()
            # Compute losses on this
            self.compute_losses_deltas(
                omegas_dict=self.omegas_pred_hal,
                suffix_future='_hal_future',
                suffix_past='_hal_past',
                suffix_present='_hal',
            )
        else:
            print('Building HMR model..')
            # This just calls HMR.
            img_feat = tf.reshape(self.phi_loader,
                                  (self.batch_size * self.sequence_length, -1))
            theta_static, _ = call_hmr_ief(
                phi=img_feat,
                omega_start=self.theta_mean_hmr,
                scope='single_view_ief',
                num_output=self.num_output,
                num_stage=self.num_stage,
                is_training=True,
            )

            self.omegas_static.append(omega=theta_static)
            self.omegas_static.compute_smpl()
            self.compute_losses_static()

        self.update_E_vars()
        self.compute_losses_prior()

    def build_model(self):
        """
        Pseudo code:
        Omega_mu = get_mean_omega() - this has been folded into init though
        h_0 = initialize to 0
        for t in [1:T]:
            Omega_t-1 = Omega^gt_t-1 if t % (u+v) < u else \hat{Omega}_t-1
            h_t,\hat{Omega} = f_temporal_enc(h_t-1,Omega_t-1,I_t)
            compute_losses(self,\hat{Omega}_t,\hat{Omega}_t-1)
        """
        print('Building model..')
        if not self.precomputed_phi:
            print('Getting all image features...')
            # Load all images at once. I_t should be B x T x H x W x C
            I_t = self.image_loader[:, :]
            # Need to reshape I_t because the input to self.f_image_enc needs
            # to be B*T x H x W x C.
            I_t = tf.reshape(I_t, (self.batch_size * self.sequence_length,
                                   self.img_size, self.img_size, 3))
            img_feat, phi_var_scope = self.f_image_enc(
                I_t, weight_decay=self.e_wd, reuse=tf.AUTO_REUSE)
            self.img_feat_full = tf.reshape(
                img_feat, (self.batch_size, self.sequence_length, -1))
        else:
            print('loading pre-computed phi!')
            # image_loader is B x T x 2048 already
            self.img_feat_full = self.phi_loader

        omegas_pred, deltas_pred = self.predict_fc()
        self.omegas_pred.append_batched(omegas_pred)

        if self.predict_delta:
            for k in deltas_pred:
                self.omegas_delta[k].append_batched(deltas_pred[k])

        if self.do_hallucinate:
            # Call the hallucinator.
            pred_phi = self.img_feat_full
            # Take B x T x 2048, outputs list of [B x T x 2048]
            self.pred_movie_strip = self.f_hal(pred_phi)

            omega_pred_hal, delta_preds_hal = batch_pred_omega(
                input_features=self.pred_movie_strip,
                batch_size=self.batch_size,
                is_training=True,
                sequence_length=self.sequence_length,
                num_output=self.num_output,
                omega_mean=tf.tile(self.theta_mean, (self.sequence_length, 1)),
                scope='single_view_ief',
                predict_delta_keys=self.omegas_pred_hal.keys(),
                use_optcam=True,
                use_delta_from_pred=self.use_delta_from_pred,
            )

            for k in delta_preds_hal:
                self.omegas_pred_hal[k].append_batched(delta_preds_hal[k])
            self.omegas_pred_hal[0].append_batched(omega_pred_hal)
            for omegas in self.omegas_pred_hal.values():
                omegas.compute_smpl()
            self.compute_losses_deltas(
                omegas_dict=self.omegas_pred_hal,
                suffix_future='_hal_future',
                suffix_past='_hal_past',
                suffix_present='_hal',
            )

        # extend e_vars to include the model variables..
        self.update_E_vars()

        self.omegas_pred.compute_smpl()
        self.compute_losses_batched()

        if self.predict_delta:
            for omegas in self.omegas_delta.values():
                omegas.compute_smpl()
            self.compute_losses_deltas(
                omegas_dict=self.omegas_delta,
                suffix_future='_dt_future',
                suffix_past='_dt_past',
                suffix_present='_dt',
            )

        self.compute_losses_prior()

    def predict_fc(self):
        """
        f_temporal_enc gets the image features and then calls the appropriate
        model.
        """
        # FC
        # For batch running- replicate omega_mean.
        omega_mean = tf.tile(self.theta_mean, (self.sequence_length, 1))
        # This is a tuple (current, delta-predictions, movie_strip)
        movie_strip = self.f_temporal_enc(
            is_training=True,
            net=self.img_feat_full,
            num_conv_layers=self.num_conv_layers,
        )
        omegas_pred, deltas_pred = batch_pred_omega(
            input_features=movie_strip,
            batch_size=self.batch_size,
            is_training=True,
            num_output=self.num_output,
            omega_mean=omega_mean,
            predict_delta_keys=self.omegas_delta.keys(),
            sequence_length=self.sequence_length,
            scope='single_view_ief',
            use_optcam=True,
            use_delta_from_pred=True,
        )

        # Keep for computing loss for hallucinator.
        self.movie_strip = movie_strip

        return omegas_pred, deltas_pred

    def prune_variables(self, variables, skip_prefixes):
        pruned_vars = []
        for var in variables:
            # Don't want var to start with any of the forbidden prefixes.
            if not any([var.name.startswith(pre) for pre in skip_prefixes]):
                pruned_vars.append(var)
        return pruned_vars

    def update_E_vars(self):
        self.E_var.extend(self.prune_variables(
            variables=tf.contrib.framework.get_trainable_variables(),
            skip_prefixes=['D_'],
        ))

    def get_unfrozen_E_vars(self):
        skip_prefixes = []
        if self.freeze_phi:
            skip_prefixes.append('resnet')
        return self.prune_variables(self.E_var, skip_prefixes)

    def add_scalar_summary(self, summary_name, scalar_value):
        """
        Helper function to add scalar summary.

        Args:
            summary_name (str).
            scalar_value (float).
        """
        self.summaries_list.append(tf.summary.scalar(
            name=summary_name,
            tensor=scalar_value
        ))

    def gather_losses(self):
        """
        Consolidates all loss functions for encoder and discriminator
        optimizers and setups summaries.
        """
        self.e_loss, self.d_loss = 0., 0.
        for key in self.losses.keys():
            w = self.loss_weights[key]
            loss = self.losses[key]
            loss_weighted = loss * w
            loss_type = key[0]  # Either e or d.
            # Prepare summaries.
            self.add_scalar_summary('{}_loss/{}'.format(loss_type, key), loss)
            if loss_type == 'e':
                # Encoder losses.
                self.e_loss += loss_weighted
                self.loss_proportions[key] = (loss, loss_weighted)
            elif loss_type == 'd':
                # Discriminator losses.
                self.d_loss += loss_weighted

        self.add_scalar_summary('e_loss/e_loss', self.e_loss)
        self.add_scalar_summary('d_loss/d_loss', self.d_loss)

        # Add shape.
        if self.use_hmr_only and not self.do_hallucinate:
            self.summaries_list.append(
                tf.summary.histogram(
                    'betas',
                    self.omegas_static.get_shapes(),
                ))
        elif not self.use_hmr_only:
            self.summaries_list.append(
                tf.summary.histogram(
                    'betas',
                    self.omegas_pred.get_shapes(),
                ))
        if self.do_hallucinate:
            self.summaries_list.append(
                tf.summary.histogram(
                    'betas',
                    self.omegas_pred_hal[0].get_shapes(),
                ))
        self.add_scalar_summary('d_loss/d_loss', self.d_loss)
        self.summaries = tf.summary.merge(self.summaries_list)

        # Moving means and vars for batch norm.
        bn_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if bn_ops:
            self.e_loss = control_flow_ops.with_dependencies(
                [tf.group(*bn_ops)], self.e_loss)

    def setup_optimizers(self):
        print('Setting up optimizers!')
        self.gather_losses()
        e_optimizer = self.optimizer(self.e_lr)
        pruned_E_vars = self.get_unfrozen_E_vars()
        self.e_opt = e_optimizer.minimize(
            loss=self.e_loss,
            global_step=self.global_step,
            var_list=pruned_E_vars)

        if self.use_disc_pose:
            self.D_var.extend(self.disc_pose.get_vars())
            d_optimizer = self.optimizer(self.d_lr)
            self.d_opt = d_optimizer.minimize(
                loss=self.d_loss,
                global_step=self.global_step,
                var_list=self.D_var)

    def setup_disc_summary(self, poses_out):
        """
        Sets up histograms to visualize discriminator outputs.
        """
        # This should be moved to a constants file.
        smpl_names = [
            'Left_Hip', 'Right_Hip', 'Waist', 'Left_Knee', 'Right_Knee',
            'Upper_Waist', 'Left_Ankle', 'Right_Ankle', 'Chest', 'Left_Toe',
            'Right_Toe', 'Base_Neck', 'Left_Shoulder', 'Right_Shoulder',
            'Upper_Neck', 'Left_Arm', 'Right_Arm', 'Left_Elbow', 'Right_Elbow',
            'Left_Wrist', 'Right_Wrist', 'Left_Finger', 'Right_Finger'
        ]
        summary = [
            tf.summary.histogram('poses_out/all', poses_out[:, 23]),
        ]
        for i, name in enumerate(smpl_names):
            summary.append(
                tf.summary.histogram(
                    name='poses_out/{}'.format(name), values=poses_out[:, i]))
        self.summaries_list.extend(summary)

    def compute_losses_batched(self):
        B = self.batch_size
        T = self.sequence_length

        gt = self.omegas_gt
        pred = self.omegas_pred

        # Compute keypoint loss.
        kps_gt = gt.get_kps()
        kps_pred = pred.get_kps()
        loss_e_kp = compute_loss_e_kp(kps_gt, kps_pred)
        self.losses['e_kp'] = loss_e_kp

        # Compute 3D loss.
        if self.use_3d_label:
            has_gt3d_smpl = tf_repeat(self.has_gt3d_smpl, T, 0)
            has_gt3d_joints = tf_repeat(self.has_gt3d_joints, T, 0)

            loss_e_pose, loss_e_shape, loss_e_joints = compute_loss_e_3d(
                poses_gt=gt.get_poses_rot(),
                poses_pred=pred.get_poses_rot(),
                shapes_gt=gt.get_shapes(),
                shapes_pred=pred.get_shapes(),
                joints_gt=gt.get_joints(),
                # Ignore face and toe points.
                joints_pred=pred.get_joints()[:, :, :14],
                batch_size=(B * T),
                has_gt3d_smpl=has_gt3d_smpl,
                has_gt3d_joints=has_gt3d_joints,
            )
            self.losses['e_joints'] = loss_e_joints
            self.losses['e_smpl'] = loss_e_pose + loss_e_shape
            self.add_scalar_summary('e_loss/e_smpl_pose', loss_e_pose)
            self.add_scalar_summary('e_loss/e_smpl_shape', loss_e_shape)

        # Compute loss_e_const on normalized beta.
        betas = self.omegas_pred.get_shapes()
        beta_prev = betas[:, :-1]
        beta_curr = betas[:, 1:]
        self.losses['e_const'] = compute_loss_e_smooth(beta_prev, beta_curr)

        # Add for computing adv prior
        poses_fake = tf.reshape(pred.get_poses_rot(), (-1, 24, 9))
        shapes_fake = tf.reshape(pred.get_shapes(), (-1, 10))

        self.pred_poses_all.append(poses_fake)
        self.pred_shapes_all.append(shapes_fake)

        if self.do_hallucinate:
            # Make sure movie_strips are predicted properly!
            # Note: If precomputed_phi is not true, should cut deriv on
            # self.movie_strip.
            self.losses['e_hallucinate'] = tf.losses.mean_squared_error(
                labels=self.movie_strip,
                predictions=self.pred_movie_strip
            )

    def compute_losses_deltas(self, omegas_dict, suffix_future, suffix_past,
                              suffix_present):
        """
        Computes all delta losses.

        Args:
            omegas_dict (dict): Dict mapping delta_t to Omegas.
            suffix_future (str): Suffix to use for future prediction.
            suffix_past (str): Suffix to use for past prediction.
            suffix_present (str): Suffix to use for present prediction and
                smpl_pose/smpl_shape losses.
        """
        B = self.batch_size
        T = self.sequence_length
        gt = self.omegas_gt

        total_loss_e_pose = 0.
        total_loss_e_shape = 0.

        for delta_t, pred in omegas_dict.items():
            if delta_t == 0:
                s_gt, e_gt, s_pr, e_pr = [None] * 4
                seq_length = T
            elif delta_t < 0:
                # Past prediction
                s_gt = None
                e_gt = delta_t
                s_pr = abs(delta_t)
                e_pr = None
                seq_length = T - abs(delta_t)
            else:
                # Future prediction.
                s_gt = delta_t
                e_gt = None
                s_pr = None
                e_pr = -delta_t
                seq_length = T - delta_t

            # Only compute optcam if delta_t !=0
            if delta_t != 0:
                loss_e_kp, best_cam = compute_loss_e_kp_optcam(
                    kp_gt=gt.get_kps()[:, s_gt:e_gt],
                    kp_pred=pred.get_kps()[:, s_pr:e_pr],
                )
                # Some needs to be padded to make full T cams
                # below is just trying to do:
                # pred.get_cams()[:, s_pr:e_pr] = best_cam
                if delta_t < 0:  # Past
                    s_pad = pred.get_cams()[:, :s_pr]
                    best_cam = tf.concat([s_pad, best_cam], axis=1)
                if delta_t > 0:
                    e_pad = pred.get_cams()[:, e_pr:]
                    best_cam = tf.concat([best_cam, e_pad], axis=1)
                pred.set_cams(best_cam)
            else:
                loss_e_kp = compute_loss_e_kp(
                    kp_gt=gt.get_kps()[:, s_gt:e_gt],
                    kp_pred=pred.get_kps()[:, s_pr:e_pr])
            # Compute 3D loss.
            if self.use_3d_label:
                has_gt3d_smpl = tf_repeat(
                    tensor=self.has_gt3d_smpl, repeat=seq_length, axis=0)
                has_gt3d_jnts = tf_repeat(
                    tensor=self.has_gt3d_joints, repeat=seq_length, axis=0)

                loss_e_pose, loss_e_shape, loss_e_joints = compute_loss_e_3d(
                    poses_gt=gt.get_poses_rot()[:, s_gt:e_gt],
                    poses_pred=pred.get_poses_rot()[:, s_pr:e_pr],
                    shapes_gt=gt.get_shapes()[:, s_gt:e_gt],
                    shapes_pred=pred.get_shapes()[:, s_pr:e_pr],
                    joints_gt=gt.get_joints()[:, s_gt:e_gt],
                    # Ignore face pts.
                    joints_pred=pred.get_joints()[:, s_pr:e_pr, :14],
                    batch_size=(B * seq_length),
                    has_gt3d_smpl=has_gt3d_smpl,
                    has_gt3d_joints=has_gt3d_jnts,
                )
            else:
                loss_e_pose, loss_e_shape, loss_e_joints = 0, 0, 0

            # Add for adv. prior loss.
            poses_fake = tf.reshape(pred.get_poses_rot(), (-1, 24, 9))
            shapes_fake = tf.reshape(pred.get_shapes(), (-1, 10))
            self.pred_poses_all.append(poses_fake)
            self.pred_shapes_all.append(shapes_fake)
            loss_e_smpl = loss_e_pose + loss_e_shape

            if delta_t == 0:
                self.losses['e_kp' + suffix_present] += loss_e_kp
                self.losses['e_joints' + suffix_present] += loss_e_joints
                self.losses['e_smpl' + suffix_present] += loss_e_smpl
            elif delta_t > 0:
                self.losses['e_kp' + suffix_future] += loss_e_kp
                self.losses['e_joints' + suffix_future] += loss_e_joints
                self.losses['e_smpl' + suffix_future] += loss_e_smpl
            else:
                self.losses['e_kp' + suffix_past] += loss_e_kp
                self.losses['e_joints' + suffix_past] += loss_e_joints
                self.losses['e_smpl' + suffix_past] += loss_e_smpl
            total_loss_e_pose += loss_e_pose
            total_loss_e_shape += loss_e_shape

        self.add_scalar_summary('e_loss/e_smpl_pose' + suffix_present,
                                total_loss_e_pose)
        self.add_scalar_summary('e_loss/e_smpl_shape' + suffix_present,
                                total_loss_e_shape)

    def compute_losses_static(self):
        gt = self.omegas_gt_static
        pred = self.omegas_static

        # Compute keypoint loss
        kps_gt = gt.get_kps()
        kps_pred = pred.get_kps()
        loss_e_kp = compute_loss_e_kp(kps_gt, kps_pred)
        self.losses['e_kp_static'] = loss_e_kp

        # Compute 3D loss.
        if self.use_3d_label:
            loss_e_pose, loss_e_shape, loss_e_joints = compute_loss_e_3d(
                poses_gt=gt.get_poses_rot(),
                poses_pred=pred.get_poses_rot(),
                shapes_gt=gt.get_shapes(),
                shapes_pred=pred.get_shapes(),
                joints_gt=gt.get_joints(),
                joints_pred=pred.get_joints()[:, :, :14],  # Ignore face pts.
                batch_size=self.batch_size_static,
                has_gt3d_smpl=self.has_gt3d_smpl_static,
                has_gt3d_joints=self.has_gt3d_joints_static,
            )
            self.losses['e_joints_static'] = loss_e_joints
            self.losses['e_smpl_static'] = loss_e_pose + loss_e_shape
            self.add_scalar_summary('e_loss/e_smpl_pose_static', loss_e_pose)
            self.add_scalar_summary('e_loss/e_smpl_shape_static', loss_e_shape)

        # Add for adv prior loss.
        poses_fake = tf.reshape(pred.get_poses_rot(), (-1, 24, 9))
        shapes_fake = tf.reshape(pred.get_shapes(), (-1, 10))
        self.pred_poses_all.append(poses_fake)
        self.pred_shapes_all.append(shapes_fake)

    def compute_losses_prior(self):
        # Load real.
        poses_real = self.poses_real_loader  # (BxT+B_static+B_delta_t)x216

        # Get fakes.
        # These are (-1, 24, 9) and (-1, 10)
        poses_pred = tf.concat(self.pred_poses_all, axis=0)
        shapes_pred = tf.concat(self.pred_shapes_all, axis=0)

        poses_real = tf.reshape(poses_real, (-1, 24, 9))
        poses_fake = poses_pred
        if poses_real.shape[0] != poses_fake.shape[0]:
            print('The # of fake and pose must be equal!, {} vs {}'.format(
                poses_real.shape[0], poses_fake.shape[0]))
            import ipdb
            ipdb.set_trace()
        poses_comb = tf.concat((poses_real, poses_fake), 0, 'poses_combined')
        poses_comb = tf.expand_dims(poses_comb, 2)
        # Compute adv on pose. drop global rotations.
        poses_out = self.disc_pose.get_output(poses_comb[:, 1:])
        poses_out_real, poses_out_fake = tf.split(poses_out, 2)
        loss_e_pose_fake = compute_loss_e_fake(poses_out_fake)
        loss_d_pose_real = compute_loss_d_real(poses_out_real)
        loss_d_pose_fake = compute_loss_d_fake(poses_out_fake)

        self.losses['e_pose'] = loss_e_pose_fake
        self.losses['d_pose'] = loss_d_pose_fake + loss_d_pose_real

        # L2 prior on shape.
        self.losses['e_shape'] = compute_loss_shape(shapes_pred)

        self.setup_disc_summary(poses_out)

    def train(self):
        step = 0
        print('Starting train!')
        with self.sv.managed_session(config=self.sess_config) as sess:
            while not self.sv.should_stop():
                fetch_dict = {
                    'summary': self.summaries,
                    'iteration': self.global_step,
                    'e_loss': self.e_loss,
                    'e_opt': self.e_opt,
                }
                if self.use_disc_pose:
                    fetch_dict.update({
                        'd_opt': self.d_opt,
                        'd_loss': self.d_loss,
                    })

                fetch_dict.update(self.losses)
                if step % self.log_img_step == 0:
                    if self.use_hmr_only and self.do_hallucinate:
                        # Weird case when omegas_pred is empty
                        omegas_pred = self.omegas_pred_hal[0]
                    else:
                        omegas_pred = self.omegas_pred
                    fetch_dict.update({
                        'cams': omegas_pred.get_cams(),
                        'images': self.images_all,
                        'kps_gt': self.omegas_gt.get_kps(),
                        'kps_pred': omegas_pred.get_kps(),
                        'verts': omegas_pred.get_verts(),
                    })

                    if self.use_hmr_only and not self.do_hallucinate:
                        fetch_dict.update({
                            'cams_static':
                            self.omegas_static.get_cams(),
                            'images_static':
                            self.images_static,
                            'kps_gt_static':
                            self.omegas_gt_static.get_kps(),
                            'kps_pred_static':
                            self.omegas_static.get_kps(),
                            'verts_static':
                            self.omegas_static.get_verts(),
                        })
                    if self.predict_delta:
                        preds_delta = {}
                        for delta_t, omega in self.omegas_delta.items():
                            preds_delta[delta_t] = {
                                'cams': omega.get_cams(),
                                'kps_pred': omega.get_kps(),
                                'verts': omega.get_verts(),
                            }
                        fetch_dict['deltas'] = preds_delta
                    if self.do_hallucinate:
                        preds_hal = {}
                        for delta_t, omega in self.omegas_pred_hal.items():
                            preds_hal[delta_t] = {
                                'cams': omega.get_cams(),
                                'kps_pred': omega.get_kps(),
                                'verts': omega.get_verts(),
                            }
                        fetch_dict['hal'] = preds_hal

                if step % 500 == 0:
                    # Also print loss_proportions
                    fetch_dict['loss_proportions'] = self.loss_proportions

                t0 = time()

                result = sess.run(fetch_dict)

                t1 = time()
                iteration = result['iteration']
                e_loss = result['e_loss']

                self.summary_writer.add_summary(
                    result['summary'], global_step=iteration)

                if 'loss_proportions' in result.keys():
                    self.record_loss_proportions(
                        loss_proportions=result['loss_proportions'],
                        checkpoint=iteration)

                if 'images' in result.keys() or \
                        'images_static' in result.keys():
                    self.visualize(result)

                output = 'itr {}: time {:.2f}, e_loss: {:.4f}'
                output_str = output.format(iteration, (t1 - t0), e_loss)
                if self.use_disc_pose:
                    output_str += ', d_loss: {:.4f}'.format(result['d_loss'])
                print(output_str)
                self.summary_writer.flush()
                if iteration > self.max_iteration:
                    self.sv.request_stop()
                step += 1

        print('Finish training on', self.model_dir)

    def view(self, im):
        """
        For debug.
        """
        ipdb.set_trace()
        plt.ion()
        plt.clf()
        plt.imshow(im)
        plt.draw()
        plt.pause(1e-5)
        ipdb.set_trace()

    def visualize(self, result):
        """
        Visualizes predictions as tf summary.

        Args:
            results (dict).
        """
        t0 = time()
        max_batch = self.vis_max_batch
        indices = self.vis_t_indices
        indices_pred = self.vis_t_indices
        image_summaries = []
        if not (self.use_hmr_only and not self.do_hallucinate):
            cams = np.take(result['cams'][:max_batch], indices_pred, axis=1)
            imgs = result['images']
            if self.data_format == 'NCHW':
                imgs = np.transpose(imgs, [0, 1, 3, 4, 2])
            kps_gt = result['kps_gt'][:max_batch]
            kps_pred = result['kps_pred'][:max_batch]
            verts = result['verts'].reshape((max_batch, len(indices), 6890, 3))

        for b in range(max_batch):
            all_rend_imgs = []
            if not (self.use_hmr_only and not self.do_hallucinate):
                imgs_sub = np.take(imgs, indices, axis=1)
                kps_gt_sub = np.take(kps_gt, indices, axis=1)
                kps_pred_sub = np.take(kps_pred, indices_pred, axis=1)
                for j, (img, cam, kp_gt, kp_pred, vert) in enumerate(
                        zip(imgs_sub[b], cams[b], kps_gt_sub[b],
                            kps_pred_sub[b], verts[b])):
                    rend_img = vis_util.visualize_img(
                        img=img,
                        cam=cam,
                        kp_gt=kp_gt,
                        kp_pred=kp_pred,
                        vert=vert,
                        renderer=self.renderer,
                        text={
                            'frame': indices[j]
                        })
                    all_rend_imgs.append(np.hstack(rend_img))
                combined = np.vstack(all_rend_imgs)
                sio = BytesIO()
                plt.imsave(sio, combined, format='png')
                vis_sum = tf.Summary.Image(
                    encoded_image_string=sio.getvalue(),
                    height=combined.shape[0],
                    width=combined.shape[1])
                image_summaries.append(
                    tf.Summary.Value(
                        tag='vis_images/{}'.format(b), image=vis_sum))
            # Do static.
            if self.use_hmr_only and not self.do_hallucinate:
                img = result['images_static'][b]
                cam = result['cams_static'][b][0]
                kp_gt = result['kps_gt_static'][b][0]
                kp_pred = result['kps_pred_static'][b][0]
                vert = result['verts_static'][b][0]

                rend_img = vis_util.visualize_img(
                    img=img,
                    cam=cam,
                    kp_gt=kp_gt,
                    kp_pred=kp_pred,
                    vert=vert,
                    renderer=self.renderer,
                )
                rend_img = np.hstack(rend_img)
                sio = BytesIO()
                plt.imsave(sio, rend_img, format='png')
                vis_sum = tf.Summary.Image(
                    encoded_image_string=sio.getvalue(),
                    height=rend_img.shape[0],
                    width=rend_img.shape[1])
                image_summaries.append(
                    tf.Summary.Value(
                        tag='vis_images_static/{}'.format(b), image=vis_sum))

            if self.predict_delta and not self.use_hmr_only:
                all_delta_imgs = []
                for dt, preds in sorted(result['deltas'].items()):
                    delta_t = dt
                    cams_dt = preds['cams'][b]
                    kps_pr_dt = preds['kps_pred'][b]
                    verts_dt = preds['verts'][b]
                    # Take the right subsamples (verts are already subsampled):
                    cams_dt = np.take(cams_dt, indices_pred, axis=0)
                    kps_pr_dt = np.take(kps_pr_dt, indices_pred, axis=0)
                    imgs_sub = np.take(imgs[b], indices + delta_t, axis=0)
                    kps_gt_sub = np.take(kps_gt[b], indices + delta_t, axis=0)
                    all_delta_imgs.append(
                        self.visualize_strip(
                            images=imgs_sub,
                            cams=cams_dt,
                            kps_gt=kps_gt_sub,
                            kps_pr=kps_pr_dt,
                            verts=verts_dt,
                            indices=indices,
                            dt=dt,
                        ))
                combined = np.hstack(all_delta_imgs)
                sio = BytesIO()
                plt.imsave(sio, combined, format='png')
                vis_sum = tf.Summary.Image(
                    encoded_image_string=sio.getvalue(),
                    height=combined.shape[0],
                    width=combined.shape[1])
                image_summaries.append(
                    tf.Summary.Value(
                        tag='vis_images_delta/delta_{}'.format(b),
                        image=vis_sum))
            if self.do_hallucinate:
                all_hal_imgs = []
                for dt, preds in sorted(result['hal'].items()):
                    delta_t = dt
                    cams_dt = preds['cams'][b]
                    kps_pr_dt = preds['kps_pred'][b]
                    verts_dt = preds['verts'][b]
                    # Take the right subsamples (verts are already subsampled):
                    cams_dt = np.take(cams_dt, indices_pred, axis=0)
                    kps_pr_dt = np.take(kps_pr_dt, indices_pred, axis=0)
                    imgs_sub = np.take(imgs[b], indices + delta_t, axis=0)
                    kps_gt_sub = np.take(kps_gt[b], indices + delta_t, axis=0)
                    all_hal_imgs.append(
                        self.visualize_strip(
                            images=imgs_sub,
                            cams=cams_dt,
                            kps_gt=kps_gt_sub,
                            kps_pr=kps_pr_dt,
                            verts=verts_dt,
                            indices=indices,
                            dt=dt,
                        ))
                combined = np.hstack(all_hal_imgs)
                sio = BytesIO()
                plt.imsave(sio, combined, format='png')
                vis_sum = tf.Summary.Image(
                    encoded_image_string=sio.getvalue(),
                    height=combined.shape[0],
                    width=combined.shape[1])
                image_summaries.append(
                    tf.Summary.Value(
                        tag='vis_images_delta/hal_{}'.format(b),
                        image=vis_sum))

        summary = tf.Summary(value=image_summaries)
        self.summary_writer.add_summary(
            summary, global_step=result['iteration'])
        print('Visualization time:', time() - t0)

    def visualize_strip(self, images, cams, kps_gt, kps_pr, verts, indices, dt):
        """
        Visualizes the delta and hal predictions side-by-side. Each strip
        contains projected 2D skeleton on left and mesh overlaid on gt image on
        right.

        Args:
            images (FxHxWx3).
            cams (Fx3).
            kps_gt (Fx25x3).
            kps_pr (Fx25x3).
            verts (Fx6980x3).
            indices (F).
            dt (int).

        Returns:
            Stacked image strip.
        """
        rend_imgs = []
        for i, (img, cam, kp_gt, kp_pred, vert) in enumerate(
                zip(images, cams, kps_gt, kps_pr, verts)):
            rend_img = vis_util.visualize_img(
                img=img,
                cam=cam,
                kp_gt=kp_gt,
                kp_pred=kp_pred,
                vert=vert,
                renderer=self.renderer,
                text={
                    'frame': indices[i] + dt,
                    'delta_t': dt,
                })
            rend_imgs.append(np.hstack(rend_img))
        return np.vstack(rend_imgs)

    def record_loss_proportions(self, loss_proportions, checkpoint):
        """
        Prints out loss proportions and writes it to a file.

        Args:
            loss_proportions (dict): Dictionary of loss values.
            checkpoint (int): Checkpoint number.
        """
        print('=================\nLoss Proportions:')
        unary_keys = {
            'e_joints',
            'e_kp',
            'e_smpl',
            'e_joints_hmr',
            'e_kp_hmr',
            'e_smpl_hmr',
            'e_kp_static',
            'e_joints_static',
            'e_smpl_static',
            'e_joints_dt_future',
            'e_kp_dt_future',
            'e_smpl_dt_future',
            'e_joints_dt_past',
            'e_kp_dt_past',
            'e_smpl_dt_past',
            'e_hallucinate',
            'e_joints_hal',
            'e_kp_hal',
            'e_smpl_hal',
            'e_joints_hal_future',
            'e_kp_hal_future',
            'e_smpl_hal_future',
            'e_joints_hal_past',
            'e_kp_hal_past',
            'e_smpl_hal_past',
        }
        disc_keys = {'e_pose', 'e_shape', 'e_delta'}
        unary_losses = 0.
        disc_losses = 0.
        with open(join(self.model_dir, 'loss_proportions.txt'), 'a') as f:
            f.write('ckpt: {}\n'.format(checkpoint))
            title = '{:>20}{:>10}{:>12}{:>12}'.format('Loss', 'Percent',
                                                      'Weighted', 'Unweighted')
            text = '{:>20}{:>10.2f}{:>12.4g}{:>12.6g}'
            print(title)
            f.write(title + '\n')
            total = np.sum(list(loss_proportions.values()), axis=0)[1]
            for k, v in sorted(loss_proportions.items()):
                loss, weighted = v
                if v != 0:
                    msg = text.format(k, weighted / total * 100, weighted,
                                      loss)
                    print(msg)
                    f.write(msg + '\n')
                if k in unary_keys:
                    unary_losses += weighted
                if k in disc_keys:
                    disc_losses += weighted
            msg = 'Unary loss: {:.2f}, Disc loss: {:.2f}'.format(
                (unary_losses / total) * 100, (disc_losses / total) * 100)
            print(msg)
            f.write(msg + '\n')
        print('=' * 16)
