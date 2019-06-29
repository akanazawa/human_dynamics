"""
Sets default args

Note all data format is NHWC because slim resnet wants NHWC.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os.path as osp
from os import makedirs
from glob import glob
from datetime import datetime
from absl import flags
import json
import ipdb
import numpy as np

curr_path = osp.dirname(osp.abspath(__file__))
model_dir = osp.join(curr_path, '..', 'models')
if not osp.exists(model_dir):
    print('Fix path to models/')
    ipdb.set_trace()

SMPL_MODEL_PATH = osp.join(model_dir,
                           'neutral_smpl_with_cocoplustoesankles_reg.pkl')
SMPL_FACE_PATH = osp.join(curr_path, '../src/tf_smpl', 'smpl_faces.npy')

# Default pred-trained model path for the demo.
PRETRAINED_MODEL = osp.join(model_dir, 'hmr_noS5.ckpt-642561')

# Pre-trained HMMR model:
HMMR_MODEL = osp.join(model_dir, 'hmmr_model.ckpt-1119816')

flags.DEFINE_string('smpl_model_path', SMPL_MODEL_PATH,
                    'path to the neutral smpl model')
flags.DEFINE_string('smpl_face_path', SMPL_FACE_PATH,
                    'path to smpl mesh faces (for easy rendering)')

flags.DEFINE_string('load_path', None, 'path to trained model dir')
flags.DEFINE_integer('batch_size', 8, 'Size of mini-batch.')
flags.DEFINE_integer('T', 20, 'Length of sequence.')
flags.DEFINE_integer('num_kps', 25, 'Number of keypoints.')
flags.DEFINE_integer('num_conv_layers', 3, '# of layers for convolutional')
flags.DEFINE_list('delta_t_values', ['-5', '5'], 'Amount of time to jump by.')

# For training.
flags.DEFINE_string('data_dir', None, 'Where tfrecords are saved')
flags.DEFINE_string('log_dir', 'logs', 'Where to save training models')
flags.DEFINE_string('model_dir', None,
                    'Where model will be saved -- filled automatically')
flags.DEFINE_list('datasets', ['h36m', 'penn_action', 'insta_variety'],
                  'datasets to use for training')
flags.DEFINE_list('mocap_datasets', ['CMU', 'H3.6', 'jointLim'],
                  'datasets to use for adversarial prior training')
flags.DEFINE_list('pretrained_model_path', [PRETRAINED_MODEL],
                  'if not None, fine-tunes from this ckpt')
flags.DEFINE_string('image_encoder_model_type', 'resnet',
                    'Specifies which image encoder to use')
flags.DEFINE_string('temporal_encoder_type', 'AZ_FC2GN',
                    'Specifies which network to use for temporal encoding')
flags.DEFINE_string('hallucinator_model_type', 'fc2_res',
                    'Specifies network to convert phi to moviestrip')
flags.DEFINE_integer('img_size', 224,
                     'Input image size to the network after preprocessing')
flags.DEFINE_string('data_format', 'NHWC', 'Data format')
flags.DEFINE_integer('num_stage', 3, '# of times to iterate IEF regressor')
flags.DEFINE_integer('max_iteration', 5000000, '# of max iteration to train')
flags.DEFINE_integer('log_img_count', 10,
                     'Number of images in sequence to visualize')
flags.DEFINE_integer('log_img_step', 5000,
                     'How often to visualize img during training')
flags.DEFINE_integer('log_vid_step', 100000,
                     'How often to visualize video during training')

# Loss weights.
flags.DEFINE_float('e_lw_smpl', 60, 'Weight on loss_e_smpl')
flags.DEFINE_float('e_lw_joints', 60, 'Weight on loss_e_joints')
flags.DEFINE_float('e_lw_const', 1, 'Weight on loss_e_const')
flags.DEFINE_float('e_lw_kp', 60, 'Weight on loss_e_kp')
flags.DEFINE_float('e_lw_pose', 1, 'Weight on loss_e_pose')
flags.DEFINE_float('e_lw_shape', 1, 'Weight on loss_e_shape')
flags.DEFINE_float('d_lw_pose', 1, 'Weight on loss_d_pose')

# Hyper parameters:
flags.DEFINE_float('e_lr', 0.00001, 'Encoder learning rate')
flags.DEFINE_float('d_lr', 0.0001, 'Adversarial prior learning rate')
flags.DEFINE_float('e_wd', 0.0001, 'Encoder weight decay')
flags.DEFINE_float('d_wd', 0.0001, 'Adversarial prior weight decay')

# Training setup.
flags.DEFINE_boolean('use_3d_label', True, 'Uses 3D labels if on.')
flags.DEFINE_boolean('freeze_phi', True, 'Fixes ResNet weights.')
flags.DEFINE_boolean(
    'use_hmr_ief_init', True,
    'If True, uses HMR regressor as initialization to HMMR regressor.'
)
flags.DEFINE_boolean('predict_delta', True,
                     'If True, predicts future and past as well')
flags.DEFINE_boolean('precomputed_phi', True,
                     'If True, uses tfrecord with precomputed phi')
flags.DEFINE_boolean(
    'use_delta_from_pred', True,
    'If True, initializes delta regressor from current prediction.'
)
flags.DEFINE_bool('use_hmr_only', False, 'If true, uses HMR model')
# Equal split
flags.DEFINE_bool('split_balanced', True, 'default true, the queue is forced '
                  'so its half 3D data (H36M) and half 2D in-the-wild data.')


# Hallucinating
flags.DEFINE_bool('do_hallucinate', False, 'if true trained hallucinator')
flags.DEFINE_bool('do_hallucinate_preds', False,
                  'if True, compute losses on predictions from hallucinator')
flags.DEFINE_float('e_lw_hallucinate', 1,
                   'Weight on ||pred movie_strip - movie_strip||')

# Data augmentation
flags.DEFINE_integer('trans_max', 20, 'Max value of translation jitter')
flags.DEFINE_integer('delta_trans_max', 20,
                     'Max consecutive translation jitter')
flags.DEFINE_float('scale_max', 0.3, 'Max value of scale jitter (power of 2)')
flags.DEFINE_float('delta_scale_max', 0.3, 'Max consecutive scale jitter')
flags.DEFINE_float('rotate_max', 0, 'Max value to rotate jitter')
flags.DEFINE_float('delta_rotate_max', 5, 'Max consecutive rotate jitter')


# Random seed
flags.DEFINE_integer('seed', 1, 'Graph-level random seed')

flags.DEFINE_bool('mosh_ignore', False, 'if true sets has_gt (smpl) off ')


def get_config():
    config = flags.FLAGS
    config(sys.argv)

    # Actually the rest of the code really assumes NHWC
    if config.data_format == 'NCHW':
        print('dont use NCHW')
        exit(1)

    return config


# ----- For training ----- #


def prepare_dirs(config, prefix=[]):
    # Continue training from a load_path
    if config.load_path:
        if not osp.exists(config.load_path):
            print("load_path: %s doesnt exist..!!!" % config.load_path)
            import ipdb
            ipdb.set_trace()
        print('continuing from %s!' % config.load_path)

        # Check for changed training parameter:
        # Load prev config param path
        param_path = glob(osp.join(config.load_path, '*.json'))[0]

        with open(param_path, 'r') as fp:
            prev_config = json.load(fp)
        dict_here = config.__dict__
        ignore_keys = ['load_path', 'log_img_step', 'pretrained_model_path']
        diff_keys = [
            k for k in dict_here
            if k not in ignore_keys and k in prev_config.keys()
            and prev_config[k] != dict_here[k]
        ]

        for k in diff_keys:
            if k == 'load_path' or k == 'log_img_step':
                continue
            if prev_config[k] is None and dict_here[k] is not None:
                print("%s is different!! before: None after: %g" %
                      (k, dict_here[k]))
            elif prev_config[k] is not None and dict_here[k] is None:
                print("%s is different!! before: %g after: None" %
                      (k, prev_config[k]))
            else:
                print("%s is different!! before: " % k)
                print(prev_config[k])
                print("now:")
                print(dict_here[k])

        if len(diff_keys) > 0:
            print("really continue??")
            import ipdb
            ipdb.set_trace()

        config.model_dir = config.load_path

    else:
        postfix = []

        # If config.dataset is not the same as default, add that to name.
        default_dataset = [
            'lsp', 'lsp_ext', 'mpii', 'h36m', 'coco', 'mpi_inf_3dhp'
        ]
        default_static_datasets = sorted(['lsp', 'coco'])
        default_mocap = ['CMU', 'H3.6', 'jointLim']

        if sorted(config.datasets) != sorted(default_dataset):
            has_all_default = np.all(
                [name in config.datasets for name in default_dataset])
            if has_all_default:
                new_names = [
                    name for name in sorted(config.datasets)
                    if name not in default_dataset
                ]
                postfix.append('default+' + '-'.join(sorted(new_names)))
            else:
                postfix.append('-'.join(sorted(config.datasets)))
        if sorted(config.mocap_datasets) != sorted(default_mocap):
            postfix.append('-'.join(config.mocap_datasets))

        if config.e_lr != 1e-5:
            postfix.append('Elr{:g}'.format(config.e_lr))

        # Weights:
        if config.e_lw_smpl != 60:
            postfix.append('lwsmpl-{}'.format(config.e_lw_smpl))
        if config.e_lw_joints != 60:
            postfix.append('lw3djoints-{}'.format(config.e_lw_joints))
        if config.e_lw_kp != 60:
            postfix.append('lw-kp{:g}'.format(config.e_lw_kp))
        if config.e_lw_shape != 1:
            postfix.append('lw-shape{:g}'.format(config.e_lw_shape))
        if config.e_lw_pose != 1:
            postfix.append('lw-pose{:g}'.format(config.e_lw_pose))
        if config.e_lw_hallucinate != 1:
            postfix.append('lw-hall{:g}'.format(config.e_lw_pose))
            

        if config.d_lr != 1e-4:
            postfix.append('Dlr{:g}' % config.d_lr)

        postfix.append('const{}'.format(config.e_lw_const))

        postfix.append('l2-shape-{}'.format(config.e_lw_shape))

        if config.use_hmr_ief_init:
            if config.temporal_encoder_type != 'AZ_FC2GN':
                print(
                    'HMR ief init is only implemented for AZ_FC2GN, implement'
                    ' it and update this warning!')
                import ipdb
                ipdb.set_trace()
            postfix.append('hmr-ief-init')

        # Model
        if not config.use_hmr_only:
            prefix.append(config.temporal_encoder_type)
            if config.temporal_encoder_type[0:5] == 'AZ_FC':
                prefix.append('{}'.format(config.num_conv_layers))
        else:
            prefix.append('HMR')

        if config.predict_delta:
            pref = 'pred-delta'
            if config.use_delta_from_pred:
                pref += '-from-pred'
            pref += '_'.join(config.delta_t_values)
            prefix.append(pref)

        if config.do_hallucinate:
            assert config.predict_delta
            pref = 'hal'
            if config.do_hallucinate_preds:
                # Prob. depreciated
                pref += '-preds'
            prefix.append(pref)

        if config.num_stage != 3:
            prefix += ["ief-stages%d" % config.num_stage]

        prefix.append('B{}'.format(config.batch_size))
        prefix.append('T{}'.format(config.T))

        if config.precomputed_phi:
            prefix.append('precomputed-phi')
        elif config.freeze_phi:
            prefix.append('freeze-phi')

        # Add pretrained (dont worry about load_path,
        # bc if it was specified id never get here):
        if config.pretrained_model_path is not None:
            if 'resnet_v2_50' in config.pretrained_model_path:
                postfix.append('from_resnet')
            elif 'hmr_noS5.ckpt-642561' == osp.basename(
                    config.pretrained_model_path[0]):
                postfix.append('from_{}'.format(
                    osp.basename(config.pretrained_model_path[0])))
            else:
                # We are finetuning from HMR/HMMR! include date.
                date = osp.basename(
                    osp.dirname(config.pretrained_model_path[0]))[-10:]
                model_ckpt = osp.basename(config.pretrained_model_path[0])
                postfix.append('from_{}_{}'.format(date, model_ckpt))

        # Data:
        # Jitter amount:
        if config.trans_max != 20 or config.delta_trans_max != 20:
            postfix.append('transmax-{}:{}'.format(config.trans_max,
                                                   config.delta_trans_max))
        if config.scale_max != 0.3 or config.delta_scale_max != 0.3:
            postfix.append('scmax-{}:{}'.format(config.scale_max,
                                                config.delta_scale_max))

        if not config.split_balanced:
            postfix.append('no-split-balance')

        if config.mosh_ignore:
            postfix.append('mosh_ignore')

        if not postfix:
            postfix.append('')

        prefix = '_'.join(prefix)
        postfix = '_'.join(postfix)

        time_str = datetime.now().strftime("%b%d_%H%M")

        save_name = "%s_%s_%s" % (prefix, postfix, time_str)
        config.model_dir = osp.join(config.log_dir, save_name)

    for path in [config.log_dir, config.model_dir]:
        if not osp.exists(path):
            print('making %s' % path)
            makedirs(path)


def save_config(config):
    param_path = osp.join(config.model_dir, "params.json")

    print("[*] MODEL dir: %s" % config.model_dir)
    print("[*] PARAM path: %s" % param_path)

    config_dict = {}
    for k in dir(config):
        config_dict[k] = config.__getattr__(k)

    with open(param_path, 'w') as fp:
        json.dump(config_dict, fp, indent=4, sort_keys=True)
