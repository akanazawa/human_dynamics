"""
Wrapper classes for saving all predicted variables. Makes it easier to compute
SMPL from the 85-dimension output of the model all at once.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from src.util.tf_ops import compute_deltas_batched
from src.tf_smpl.batch_lbs import batch_rodrigues
from src.tf_smpl.projection import batch_orth_proj_idrot


class Omegas(object):
    """
    Superclass container for batches of sequences of poses, shapes, joints, etc.

    Args:
        config.
    """
    def __init__(self, config, batch_size=None):
        self.config = config
        if batch_size:
            self.batch_size = batch_size
        else:
            self.batch_size = config.batch_size
        self.length = 0

        self.joints = tf.constant(
            (),
            shape=(self.batch_size, 0, self.config.num_kps, 3)
        )
        self.kps = tf.constant(
            (),
            shape=(self.batch_size, 0, self.config.num_kps, 2)
        )
        self.poses_aa = tf.constant((), shape=(self.batch_size, 0, 24, 3))
        self.poses_rot = tf.constant((), shape=(self.batch_size, 0, 24, 3, 3))
        self.shapes = tf.constant((), shape=(self.batch_size, 0, 10))

        self.deltas_aa = tf.constant((), shape=(self.batch_size, 0, 24, 3))
        self.deltas_rot = tf.constant((), shape=(self.batch_size, 0, 24, 3, 3))

    def __len__(self):
        """
        Returns the current sequence length.

        Returns:
            length (int).
        """
        return self.length

    def get_joints(self, t=None):
        """
        Returns the joints at time t.

        Args:
            t (int).

        Returns:
            Joints (Bx25x3).
        """
        return self.joints if t is None else self.joints[:, t]

    def get_kps(self, t=None):
        """
        Returns the keypoints at time t.

        Note that the shape is different for ground truth omegas and predicted
        omegas.

        Args:
            t (int).

        Returns:
            Kps (Bx25x3) if gt,
                or Kps (Bx25x2) if pred.
        """
        return self.kps if t is None else self.kps[:, t]

    def get_poses_aa(self, t=None):
        """
        Returns axis-aligned poses at time t.

        Args:
            t (int).

        Returns:
            Poses (Bx24x3).
        """
        return self.poses_aa if t is None else self.poses_aa[:, t]

    def get_poses_rot(self, t=None):
        """
        Returns poses as rotation matrices at time t.

        Args:
            t (int).

        Returns:
            Poses (Bx24x3x3).
        """
        return self.poses_rot if t is None else self.poses_rot[:, t]

    def get_deltas_aa(self, t=None):
        """
        Returns axis-aligned deltas from time t to t + 1.

        Args:
            t (int).

        Returns:
            Deltas (Bx24x3).
        """
        return self.deltas_aa if t is None else self.deltas_aa[:, t]

    def get_deltas_rot(self, t=None):
        """
        Returns deltas as rotation matrices from time t to t + 1.

        Args:
            t (int).

        Returns:
            Deltas (Bx24x3x3).
        """
        return self.deltas_rot if t is None else self.deltas_rot[:, t]

    def get_shapes(self, t=None):
        """
        Returns shapes at time t.

        Args:
            t (int).

        Returns:
            Shapes (Bx10).
        """
        return self.shapes if t is None else self.shapes[:, t]

    @staticmethod
    def gather(values, indices):
        """
        Gathers a subset over time.

        Args:
            values (BxTx...): Tensor that we only need a subset of.
            indices (iterable): 1D tensor of times.

        Returns:
            tensor.
        """
        if not isinstance(indices, tf.Tensor):
            indices = tf.constant(indices)
        spliced = tf.gather(params=values, indices=indices, axis=1)
        return spliced


class OmegasGt(Omegas):
    """
    Stores fields for ground truth omegas.

    Args:
        config.
        poses_aa (BxTx24x3).
        shapes (Bx10).
        joints (BxTx14x3).
    """
    def __init__(self, config, poses_aa, shapes, joints, kps, batch_size=None):
        super(OmegasGt, self).__init__(config, batch_size=batch_size)
        self.length = poses_aa.shape[1]

        self.poses_aa = poses_aa
        poses_rot = batch_rodrigues(tf.reshape(poses_aa, (-1, 3)))
        self.poses_rot = tf.reshape(poses_rot, (self.batch_size, -1, 24, 3, 3))
        self.shapes = shapes
        self.joints = joints
        self.kps = kps
        self.deltas_rot = compute_deltas_batched(
            poses_prev=self.poses_rot[:, :-1],
            poses_curr=self.poses_rot[:, 1:]
        )

    def get_shapes(self, t=None):
        if t is None:
            # When t is None, expect BxTx10.
            return tf.tile(tf.expand_dims(self.shapes, 1), (1, self.length, 1))
        else:
            return self.shapes

    def get_deltas_aa(self, t=None):
        raise Exception('No axis-aligned deltas.')


class OmegasPred(Omegas):
    """
    Stores fields for predicted Omegas.

    Args:
        config.
        smpl (func).
        optcam (bool): If true, uses optcam when computing kp proj.
        vis_max_batch (int): Number of batches to visualize.
        vis_t_indices (ndarray): Times to visualize. If None, keeps all.
    """
    omega_instances = []

    def __init__(self,
                 config,
                 smpl,
                 use_optcam=False,
                 vis_max_batch=2,
                 vis_t_indices=None,
                 batch_size=None,
                 is_training=True):
        super(OmegasPred, self).__init__(config, batch_size)
        self.smpl = smpl
        self.cams = tf.constant((), shape=(self.batch_size, 0, 3))
        self.all_verts = tf.constant((), shape=(0, 6890, 3))
        self.verts = tf.constant((), shape=(0, 6890, 3))
        self.smpl_computed = False
        self.vis_max_batch = vis_max_batch
        self.vis_t_indices = vis_t_indices
        self.raw = tf.constant((), shape=(self.batch_size, 0, 85))
        self.use_optcam = use_optcam
        self.is_training = is_training
        OmegasPred.omega_instances.append(self)

    def update_instance_vars(self):
        self.cams = self.raw[:, :, :3]
        self.poses_aa = self.raw[:, :, 3: 3 + 24 * 3]
        self.shapes = self.raw[:, :, 3 + 24 * 3: 85]
        self.length = self.raw.shape[1]

    def append_batched(self, omegas):
        """
        Appends multiple omegas.

        Args:
            omegas (BxTx85): [cams, poses, shapes].
        """
        B = self.batch_size
        omegas = tf.reshape(omegas, (B, -1, 85))
        self.raw = tf.concat((self.raw, omegas), axis=1)
        self.update_instance_vars()
        self.smpl_computed = False

    def append(self, omega):
        """
        Appends an omega.

        Args:
            omega (Bx85): [cams, poses, shapes].
        """
        B = self.batch_size
        omega = tf.reshape(omega, (B, 1, 85))
        self.raw = tf.concat((self.raw, omega), axis=1)
        self.update_instance_vars()
        self.smpl_computed = False

    def compute_smpl(self):
        """
        Batch computation of vertices, joints, rotation matrices, and keypoints.
        Due to the overhead added to computation graph, call this once.
        """
        if self.smpl_computed:
            print('SMPL should only be computed once!')
        B = self.batch_size
        T = self.length

        verts, joints, poses_rot = self.smpl(
            beta=tf.reshape(self.shapes, (B * T, 10)),
            theta=tf.reshape(self.poses_aa, (B * T, 24, 3)),
            get_skin=True
        )
        self.joints = tf.reshape(joints, (B, T, self.config.num_kps, 3))
        self.poses_rot = tf.reshape(poses_rot, (B, T, 24, 3, 3))

        # Make sure joints are B*T x num_kps x 3.
        if self.use_optcam and self.is_training:
            print('Using optimal camera!!')
            # Just drop the z here ([1, 0, 0])
            kps = joints[:, :, :2]
        else:
            kps = batch_orth_proj_idrot(joints,
                                        tf.reshape(self.cams, (B * T, 3)))
        self.kps = tf.reshape(kps, (B, T, self.config.num_kps, 2))

        if self.deltas_aa.shape[1] != 0:
            deltas_rot = batch_rodrigues(tf.reshape(self.deltas_aa, (-1, 3)))
            self.deltas_rot = tf.reshape(deltas_rot,
                                         (self.batch_size, -1, 24, 3, 3))

        self.all_verts = tf.reshape(verts, (B, T, 6890, 3))[:self.vis_max_batch]
        if self.vis_t_indices is None:
            self.verts = self.all_verts
        else:
            self.verts = Omegas.gather(
                values=self.all_verts,
                indices=self.vis_t_indices
            )
        self.smpl_computed = True

    def get_cams(self, t=None):
        """
        Gets cams at time t.

        Args:
            t (int).

        Returns:
            Cams (Bx3).
        """
        return self.cams if t is None else self.cams[:, t]

    def set_cams(self, cams):
        """
        Only used for opt_cam
        """
        assert self.use_optcam
        self.cams = cams

    def get_all_verts(self):
        return self.all_verts

    def get_verts(self):
        return self.verts

    def get_raw(self):
        """
        Returns:
            Raw Omega (BxTx85).
        """
        return self.raw

    @classmethod
    def compute_all_smpl(cls):
        omegas = cls.omega_instances
        for omega in omegas:
            omega.compute_smpl()
