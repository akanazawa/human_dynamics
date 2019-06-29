"""
Augments the 3DPW annotation, with the best neutral shape beta
that fits the mesh of the gendered shapes.
This is a convenience so when evaluating, s.t. there is no need
to load ground truth mesh from gendered models.

This also pre-computes the 3D joints using the regressor.
This is obtained from the ground truth gendered smpl model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import pickle
from glob import glob
from os import makedirs
from os.path import exists, join, basename
import sys

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from src.tf_smpl.batch_smpl import SMPL

import tensorflow as tf

from smpl_webuser.serialization import load_model

flags.DEFINE_string('base_dir', '/scratch1/jason/videos/3DPW', 'Path to 3DPW')
flags.DEFINE_string('out_dir',
                    '/scratch1/jason/videos/3DPW/sequenceFilesNeutral_redo',
                    'Where to save the new annotations')
flags.DEFINE_string('smpl_model_dir', '/home/kanazawa/projects/smpl/models/',
                    'Directory that contains the male and female SMPL models')
flags.DEFINE_string(
    'neutral_model_path',
    'models/neutral_smpl_with_cocoplustoesankles_reg.pkl',
    'Path to neutral SMPl model with toes and ankles in our release model.')

config = flags.FLAGS


def get_verts(fname):
    verts_og = []
    data = pickle.load(open(fname, 'rb'), encoding='latin1')
    genders = data['genders']
    betas = data['betas']

    for gender, beta in zip(genders, betas):
        beta = beta[:10]
        if gender == 'm':
            model = smpl_male
        elif gender == 'f':
            model = smpl_female
        else:
            print('gender not found')
        model.betas[:] = beta
        model.pose[:] = np.zeros(72)

        verts_og.append(np.copy(model.r))
    return verts_og, data, model


def save_neutral_shape(labels_file, out_dir, sess):
    seq_name = basename(labels_file)
    out_path = join(out_dir, seq_name)
    if exists(out_path):
        print('Done {}'.format(out_path))
        # Below is for checking the errors.
        if False:
            gendered_verts, anno, model = get_verts(out_path)
            genders = anno['genders']
            n = len(gendered_verts)
            joints_n = np.array(anno['joints_neutral'])
            joints_g = np.array(anno['joints_gendered']).reshape(n, -1, 25, 3)
            diff = np.linalg.norm(joints_n - joints_g, axis=3).mean(axis=2)

            for i in range(len(gendered_verts)):
                smpl_neutral_py.pose[:] = 0.
                smpl_neutral_py.betas[:] = anno['betas_neutral'][i]
                model = smpl_male if genders[i] == 'm' else smpl_female
                model.pose[:] = 0.
                model.betas[:] = anno['betas'][i]
                shape_diff = np.linalg.norm(
                    smpl_neutral_py.r - gendered_verts[i], axis=1).mean()
                print('Shape diff {}'.format(shape_diff))
                # Check for joints diff:
                joints_n_here = joints_n[i]
                joints_g_here = joints_g[i]
                diff = np.linalg.norm(
                    joints_n_here - joints_g_here, axis=2).mean(axis=1)
                print('Joint diff {}'.format(shape_diff))
        return

    gendered_verts, anno, model = get_verts(labels_file)
    gendered_verts = np.array(gendered_verts)
    n = len(gendered_verts)
    verts_og = tf.Variable(gendered_verts, trainable=False, dtype=tf.float32)
    beta_var = tf.Variable(np.zeros((n, 10)), trainable=True, dtype=tf.float32)
    theta_var = tf.Variable(
        np.zeros((n, 72)), trainable=False, dtype=tf.float32)
    verts, _, _ = smpl_neutral(beta_var, theta_var, get_skin=True)
    # loss = tf.losses.mean_squared_error(verts_og, verts)
    # This is faster conv:
    loss = tf.reduce_mean(
        tf.sqrt(tf.reduce_sum((verts_og - verts)**2, axis=2)))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1)
    opt = optimizer.minimize(loss=loss, var_list=[beta_var])

    fetch_dict = {
        'loss': loss,
        'beta': beta_var,
        'verts': verts,
        'opt': opt,
    }

    losses = []

    sess.run(tf.global_variables_initializer())

    p_bar = tqdm(range(5000))
    for i in p_bar:
        ret = sess.run(fetch_dict)
        losses.append(ret['loss'])
        p_bar.set_description(str(ret['loss']))
        # see if converged
        if i > 2 and i % 100 == 0:
            diff = np.abs(losses[-1] - losses[-2]) / max(
                losses[-1], losses[-2])
            # print(diff)
            if diff < 1e-4:
                print('converged')
                break

    plt.plot(losses)

    # add to new anno.
    anno['betas_neutral'] = ret['beta']

    # Compute the gendered joints.
    all_joints_neutral = []
    for i in range(n):
        poses = np.array(anno['poses'][i])
        theta_poses = tf.Variable(poses, trainable=False, dtype=tf.float32)
        beta_here = tf.Variable(
            ret['beta'][i], trainable=False, dtype=tf.float32)
        beta_here = tf.tile(tf.expand_dims(beta_here, 0), [poses.shape[0], 1])
        sess.run(tf.global_variables_initializer())
        verts, joints_neutral, _ = sess.run(
            smpl_neutral(beta_here, theta_poses, get_skin=True))
        all_joints_neutral.append(joints_neutral)

    anno['joints_neutral'] = all_joints_neutral
    anno['joints_gendered'] = compute_joints(anno, sess)

    with open(out_path, 'wb') as f:
        pickle.dump(anno, f)


def compute_joints(anno, sess):
    """
    Computes joints for gendered
    """
    # Compute for smpl neutral
    n = len(anno['poses'])
    genders = anno['genders']
    betas = anno['betas']

    joints = []
    for i in range(n):
        model = smpl_male if genders[i] == 'm' else smpl_female
        model.betas[:] = betas[i][:10]
        for pose in anno['poses'][i]:
            model.pose[:] = pose

            verts = np.copy(model.r)
            joint = regressor.dot(verts)
            joints.append(joint)

    return joints


if __name__ == '__main__':
    config(sys.argv)

    # Setup models
    smpl_male = load_model('{}/basicmodel_m_lbs_10_207_0_v1.0.0.pkl'.format(
        config.smpl_model_dir))
    smpl_female = load_model('{}/basicModel_f_lbs_10_207_0_v1.0.0.pkl'.format(
        config.smpl_model_dir))

    smpl_neutral_py = load_model(config.neutral_model_path)
    smpl_neutral = SMPL(config.neutral_model_path)

    with open(config.neutral_model_path, 'rb') as f:
        dd = pickle.load(f, encoding='latin1')
        regressor = dd['cocoplus_regressor']

    # Get original labels.
    all_labels = sorted(glob('{}/sequenceFiles/*.pkl'.format(config.base_dir)))

    sess = tf.Session()

    if not exists(config.out_dir):
        makedirs(config.out_dir)

    for labels_pkl in all_labels:
        save_neutral_shape(labels_pkl, config.out_dir, sess)
