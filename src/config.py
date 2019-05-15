"""
Sets default args

Note all data format is NHWC because slim resnet wants NHWC.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os.path as osp

from absl import flags
import ipdb


curr_path = osp.dirname(osp.abspath(__file__))
model_dir = osp.join(curr_path, '..', 'models')
if not osp.exists(model_dir):
    print('Fix path to models/')
    ipdb.set_trace()

SMPL_MODEL_PATH = osp.join(model_dir,
                           'neutral_smpl_with_cocoplustoesankles_reg.pkl')
SMPL_FACE_PATH = osp.join(curr_path, '../src/tf_smpl', 'smpl_faces.npy')

# Default pred-trained model path for the demo.
PRETRAINED_MODEL = osp.join(model_dir, 'model.ckpt-667589')

# Pre-trained HMMR model:
HMMR_MODEL = osp.join(model_dir, 'hmmr_model.ckpt-1119816')

flags.DEFINE_string('smpl_model_path', SMPL_MODEL_PATH,
                    'path to the neutral smpl model')
flags.DEFINE_string('smpl_face_path', SMPL_FACE_PATH,
                    'path to smpl mesh faces (for easy rendering)')

flags.DEFINE_string('load_path', HMMR_MODEL, 'path to trained model')
flags.DEFINE_integer('batch_size', 8, 'Size of mini-batch.')
flags.DEFINE_integer('T', 20, 'Length of sequence.')
flags.DEFINE_integer('num_kps', 25, 'Number of keypoints.')
flags.DEFINE_integer('num_conv_layers', 3, '# of layers for convolutional')
flags.DEFINE_bool('use_optcam', True,
                  'if true, hallucinator kp proj uses optimal camera.')
flags.DEFINE_list('delta_t_values', ['-5', '5'], 'Amount of time to jump by.')


def get_config():
    config = flags.FLAGS
    config(sys.argv)
    return config
