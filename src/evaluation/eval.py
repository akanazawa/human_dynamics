"""
Evaluates model on videos from dataset.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
from glob import glob
import json
import os
from time import time

from absl import flags
import numpy as np
import tensorflow as tf

from src.datasets.common import read_from_example
from src.evaluation.prediction import (
    get_eval_path_name,
    get_predictions,
    get_result_path_name,
)
from src.config import get_config
from src.evaluation.eval_util import (
    compute_accel,
    compute_error_3d,
    compute_error_accel,
    compute_error_kp,
    compute_error_verts,
    rot_mat_to_axis_angle,
    extend_dict_entries,
    update_dict_entries,
    mean_of_dict_values,
)
from src.evaluation.tester import Tester
from src.tf_smpl.batch_smpl import SMPL

# Model Parameters.
flags.DEFINE_string('resnet_path',
                    '/home/jason/hmmr/models/hmr_noS5.ckpt-642561',
                    'If True, extract phis first.')
flags.DEFINE_string(
    'pred_mode', 'pred',
    'Which prediction track to use (pred, hal, const). Pred does normal track '
    'prediction, hal uses hallucinator track, and const evaluates the constant '
    'baseline.'
)

# Data Parameters
flags.DEFINE_string('tf_dir', '', 'Parent directory of tfrecords.')
flags.DEFINE_string('pred_dir', 'predictions_cache', 'Prediction Directory.')
flags.DEFINE_list('test_datasets',
                  ['3dpw', 'nba', 'penn_action'],
                  'Datasets to evaluate.')
flags.DEFINE_string('split', 'val', 'val or test.')
flags.DEFINE_integer('min_visible', 6, 'Minimum visible keypoints')
flags.DEFINE_boolean('reverse', False, 'If True, runs tf records in reverse')


SMPL_MODEL_PATH = ('models/neutral_smpl_with_cocoplustoesankles_reg.pkl')
DATASETS_3D = ['3dpw', 'h36m']

# Hide some of the TensorFlow Warnings.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def compute_gpu_smpl(poses, shapes, get_joints=False):
    smpl = SMPL(SMPL_MODEL_PATH, joint_type='cocoplus')
    verts_tensor, Js_tensor, _ = smpl(tf.constant(shapes, dtype=tf.float32),
                                      tf.constant(poses, dtype=tf.float32),
                                      get_skin=True)
    init = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())

    # Start running ops on the graph. Allow_soft_placement must be set to
    # True to build towers on GPU, as some of the ops don't have GPU
    # implementations.
    sess_config = tf.ConfigProto(allow_soft_placement=True,
                                 log_device_placement=False)
    sess = tf.Session(config=sess_config)
    sess.run(init)
    if get_joints:
        verts, joints = sess.run([verts_tensor, Js_tensor])
        sess.close()
        return verts, joints
    else:
        verts = sess.run(verts_tensor)
        sess.close()
        return verts


def restore_config(config):
    """
    Restores config settings based on saved json file.
    """
    load_path = config.load_path
    param_path = glob(os.path.join(os.path.dirname(load_path), '*.json'))
    if len(param_path) == 0:
        return config
    else:
        param_path = param_path[0]

    ignore_keys = {'batch_size', 'load_path', 'smpl_model_path', 'T'}

    with open(param_path, 'r') as fp:
        prev_config = json.load(fp)
    for k, v in prev_config.items():
        if k not in ignore_keys and hasattr(config, k):
            setattr(config, k, v)
    return config


def compute_errors_batched(kps_gt,
                           kps_pred,
                           joints_gt=None,
                           joints_pred=None,
                           poses_gt=None,
                           poses_pred=None,
                           shape_gt=None,
                           shapes_pred=None,
                           img_size=224,
                           has_3d=False,
                           min_visible=6,
                           compute_mesh=False):
    """
    Computes errors.
    """
    errors_kp, errors_kp_pa, errors_kp_pck = compute_error_kp(
        kps_gt=kps_gt,
        kps_pred=(kps_pred + 1) * 0.5 * img_size,  # Bring to image space.
        alpha=(0.05 * img_size),
        min_visible=min_visible,
    )
    accel = compute_accel(joints_pred)

    errors_dict = {
        'accel': accel,
        'kp': errors_kp,
        'kp_pa': errors_kp_pa,
        'kp_pck': errors_kp_pck,
    }

    if has_3d:
        vis = np.sum(kps_gt[:, :14, 2], axis=1) > min_visible
        errors_accel = compute_error_accel(
            joints_gt=joints_gt,
            joints_pred=joints_pred,
            vis=vis,
        )
        errors_pose, errors_shape = -1, -1
        if compute_mesh:
            # Only compute mesh error if evaluating on test set.
            shapes_gt = np.tile(shape_gt, (len(poses_gt), 1))  # N x 10
            poses_pred = np.array([rot_mat_to_axis_angle(pose)
                                   for pose in poses_pred])
            mesh_gt_tpose = compute_gpu_smpl(
                poses=np.zeros_like(poses_gt),
                shapes=shapes_gt,
            )
            mesh_pred_tpose = compute_gpu_smpl(
                poses=np.zeros_like(poses_pred),
                shapes=shapes_pred,
            )
            errors_mesh_tpose = compute_error_verts(
                verts_gt=mesh_gt_tpose[vis],
                verts_pred=mesh_pred_tpose[vis],
            )
            mesh_gt = compute_gpu_smpl(poses=poses_gt, shapes=shapes_gt)
            mesh_pred = compute_gpu_smpl(poses=poses_pred, shapes=shapes_pred)
            errors_mesh_posed = compute_error_verts(
                verts_gt=mesh_gt[vis],
                verts_pred=mesh_pred[vis],
            )
        else:
            errors_mesh_posed, errors_mesh_tpose = -1, -1

        errors_joints, errors_joints_pa = compute_error_3d(
            gt3ds=joints_gt,
            preds=joints_pred,
            vis=vis,
        )
        errors_dict.update({
            'accel_error': errors_accel,
            'mesh_posed': errors_mesh_posed,
            'mesh_tpose': errors_mesh_tpose,
            'pose': errors_pose,
            'joints': errors_joints,
            'joints_pa': errors_joints_pa,
            'shape': errors_shape,
        })

    return errors_dict


def test_sequence(data, preds, eval_path, pred_mode='pred', has_3d=False,
                  min_visible=6, compute_mesh=False):
    """
    Tests one tube.

    Args:
        data (dict).
        model (Tester).
        pred_mode (str).
        has_3d (bool).

    Returns:
        Dictionary of lists. Keys are accel and kp. If 3d, also has pose
        and mesh.
    """
    images = (np.array(data['images']) / 255) * 2 - 1

    if pred_mode == 'hal':
        # The keys have a '_hal' suffix in them.
        preds = {k.replace('_hal', ''): v[:, 1]  # Only want center prediction.
                 for k, v in preds.items() if '_hal' in k}

    if os.path.exists(eval_path):
        print('Eval already exists! {}'.format(eval_path))
        with open(eval_path, 'rb') as f:
            eval_res = pickle.load(f)
            return eval_res
    t0 = time()
    errors = compute_errors_batched(
        kps_gt=data['kps'],
        kps_pred=preds['kps'],
        joints_gt=data['gt3ds'],
        joints_pred=preds['joints'][:, :14],
        poses_gt=data['poses'],
        poses_pred=preds['poses'],
        shape_gt=data['shape'],
        shapes_pred=preds['shapes'],
        img_size=images.shape[1],
        has_3d=has_3d,
        min_visible=min_visible,
        compute_mesh=compute_mesh,
    )

    with open(eval_path, 'wb') as f:
        print('Saving eval to', eval_path)
        pickle.dump(errors, f)
    print('Eval time:', time() - t0)
    return errors


def test_sequence_const(data, preds, eval_path, has_3d=False, min_visible=6):
    """
    Runs the const vs Hal future/past test.

    Args:
        data (dict).
        model (Tester).
        has_3d (bool).

    Returns:
        errors_past, errors_past_const, errors_future, errors_future_const
    """
    delta_t = config.delta_t
    images = (np.array(data['images']) / 255) * 2 - 1

    kps_pred = preds['kps_hal']
    errors_present = compute_errors_batched(
        kps_gt=data['kps'],
        kps_pred=kps_pred[:, 0],
        joints_gt=data['gt3ds'][:, :14],
        joints_pred=preds['joints_hal'][:, 0, :14],
        poses_gt=data['poses'],
        poses_pred=preds['poses_hal'][:, 0],
        img_size=images.shape[1],
        has_3d=has_3d,
        min_visible=min_visible,
    )
    errors_past = compute_errors_batched(
        kps_gt=data['kps'][:-delta_t],
        kps_pred=kps_pred[delta_t:, 0],
        joints_gt=data['gt3ds'][:-delta_t, :14],
        joints_pred=preds['joints_hal'][delta_t:, 0, :14],
        poses_gt=data['poses'][:-delta_t],
        poses_pred=preds['poses_hal'][delta_t:, 0],
        img_size=images.shape[1],
        has_3d=has_3d,
        min_visible=min_visible,
    )
    errors_past_const = compute_errors_batched(
        kps_gt=data['kps'][:-delta_t],
        kps_pred=kps_pred[delta_t:, 1],
        joints_gt=data['gt3ds'][:-delta_t, :14],
        joints_pred=preds['joints_hal'][delta_t:, 1, :14],
        poses_gt=data['poses'][:-delta_t],
        poses_pred=preds['poses_hal'][delta_t:, 1],
        img_size=images.shape[1],
        has_3d=has_3d,
        min_visible=min_visible,
    )
    errors_future = compute_errors_batched(
        kps_gt=data['kps'][delta_t:],
        kps_pred=kps_pred[:-delta_t, 2],
        joints_gt=data['gt3ds'][delta_t:, :14],
        joints_pred=preds['joints_hal'][:-delta_t, 2, :14],
        poses_gt=data['poses'][delta_t:],
        poses_pred=preds['poses_hal'][:-delta_t, 2],
        img_size=images.shape[1],
        has_3d=has_3d,
        min_visible=min_visible,
    )
    errors_future_const = compute_errors_batched(
        kps_gt=data['kps'][delta_t:],
        kps_pred=kps_pred[:-delta_t, 1],
        joints_gt=data['gt3ds'][delta_t:, :14],
        joints_pred=preds['joints_hal'][:-delta_t, 1, :14],
        poses_gt=data['poses'][delta_t:],
        poses_pred=preds['poses_hal'][:-delta_t, 1],
        img_size=images.shape[1],
        has_3d=has_3d,
        min_visible=min_visible,
    )
    errors_dict = {
        'past': errors_past,
        'past_const': errors_past_const,
        'present': errors_present,
        'future': errors_future,
        'future_const': errors_future_const,
    }
    with open(eval_path, 'wb') as f:
        print('Saving eval to', eval_path)
        pickle.dump(errors_dict, f)
    return errors_dict


def print_summary(errors_dict):
    title_format = '{:>15}' + '{:>11}' * 8
    row_format = '{:>15}' + '{:>11.5f}' * 8
    keys = ['accel', 'kp', 'kp_pa', 'kp_pck', 'joints', 'joints_pa',
            'mesh_posed', 'mesh_tpose']
    print(title_format.format('Data', *keys))
    for dataset, errors in sorted(errors_dict.items()):
        values = [errors.get(key, -1) for key in keys]
        print(row_format.format(dataset, *values))


def save_results(config, all_dataset_results, json_path=''):
    if json_path:
        json.dump(all_dataset_results, open(json_path, 'w'))

    if config.pred_mode == 'const':
        for pred_type, predictions in sorted(all_dataset_results.items()):
            print('Predicting', pred_type)
            print_summary(predictions)
    else:
        print_summary(all_dataset_results)


def main(config):
    t0 = time()
    config = restore_config(config)
    print('-' * 20)
    print('Evaluating {}'.format(config.load_path))

    json_path = get_result_path_name(
        split=config.split,
        load_path=config.load_path,
        pred_mode=config.pred_mode,
        datasets=config.test_datasets,
        pred_dir=config.pred_dir,
    )

    if os.path.exists(json_path):
        print(json_path, 'already exists!')
        all_dataset_results = json.load(open(json_path, 'r'))
        save_results(config, all_dataset_results)
        print('Total time:', time() - t0)
        print('-' * 20)
        exit(0)

    resnet_path = config.resnet_path if config.precomputed_phi else ''
    model = Tester(
        config,
        pretrained_resnet_path=resnet_path,
        sequence_length=config.T
    )

    all_dataset_results = {}
    if config.pred_mode == 'const':
        all_dataset_results.update({
            'past': {},
            'past_const': {},
            'present': {},
            'future': {},
            'future_const': {},
        })
    for dataset in config.test_datasets:
        print('Evaluating dataset:', dataset)
        dataset_result = {}
        if config.pred_mode == 'const':
            dataset_result.update({
                'past': {},
                'past_const': {},
                'present': {},
                'future': {},
                'future_const': {},
            })
        if dataset == 'h36m':
            tf_paths = sorted(glob(os.path.join(
                config.tf_dir,
                dataset,
                config.split,
                '*cam03*.tfrecord',
                )))
        else:
            tf_paths = sorted(glob(os.path.join(
                config.tf_dir,
                dataset,
                config.split,
                '*.tfrecord',
            )))
        if config.reverse:
            tf_paths = tf_paths[::-1]
        for i, fname in enumerate(tf_paths):
            print('\n', '*' * 10)
            print(dataset, '{}/{}'.format(i, len(tf_paths)))
            print('Running on', os.path.basename(fname))
            path_result = {}
            if config.pred_mode == 'const':
                path_result.update({
                    'past': {},
                    'past_const': {},
                    'present': {},
                    'future': {},
                    'future_const': {},
                })
            for p_id, s_ex in enumerate(tf.python_io.tf_record_iterator(fname)):
                data = read_from_example(s_ex)
                images = data['images']
                if dataset == 'h36m':
                    images = images
                    data['poses'] = data['poses']
                    data['kps'] = data['kps']
                    data['gt3ds'] = data['gt3ds']

                preds = get_predictions(
                    model=model,
                    images=images,
                    load_path=config.load_path,
                    tf_path=fname,
                    p_id=p_id,
                    pred_dir=config.pred_dir,
                )
                eval_path = get_eval_path_name(
                    load_path=config.load_path,
                    pred_mode=config.pred_mode,
                    tf_path=fname,
                    p_id=p_id,
                    pred_dir=config.pred_dir,
                    min_visible=config.min_visible,
                )

                if config.pred_mode == 'const':
                    errors_dict = test_sequence_const(
                        data=data,
                        preds=preds,
                        eval_path=eval_path,
                        has_3d=(dataset in DATASETS_3D),
                        min_visible=config.min_visible,
                    )
                    for k in errors_dict.keys():
                        extend_dict_entries(path_result[k], errors_dict[k])
                else:
                    compute_mesh = config.split == 'test' and dataset == '3dpw'
                    errors = test_sequence(
                        data=data,
                        preds=preds,
                        eval_path=eval_path,
                        pred_mode=config.pred_mode,
                        has_3d=(dataset in DATASETS_3D),
                        min_visible=config.min_visible,
                        compute_mesh=compute_mesh,
                    )
                    extend_dict_entries(path_result, errors)
            if config.pred_mode == 'const':
                for k in path_result.keys():
                    update_dict_entries(dataset_result[k], path_result[k])
            else:
                update_dict_entries(dataset_result, path_result)

        if config.pred_mode == 'const':
            for pred_type, result in dataset_result.items():
                mean_of_dict_values(result)
                all_dataset_results[pred_type][dataset] = result
        else:
            mean_of_dict_values(dataset_result)
            all_dataset_results[dataset] = dataset_result

    save_results(config, all_dataset_results, json_path)

    print('Total time:', time() - t0)
    print('-' * 20)


if __name__ == '__main__':
    config = get_config()
    assert config.load_path, 'Must specify load_path!'
    assert '.ckpt' in config.load_path, 'Must specify a model checkpoint!'
    assert config.tf_dir
    if config.pred_mode == 'hal':
        print('evaluating with the hallucinated track!!')
    main(config)
