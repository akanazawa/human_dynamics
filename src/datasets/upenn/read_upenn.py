"""
Read Upenn
Returns the original Upenn joints (13):
0: Head
1: R shoulder
2: L shoulder
3: R elbow
4: L elbow
5: R wrist
6: L wrist
7: R hip
8: L hip
9: R knee
10: L knee
11: R ankle
12: L ankle

appened with 12 keypointes that are all 0s.
i..e 25 x 3, only the first 13 is filled.

get_upenn2coco() provides index to convert the above
ordering to the universal 25 coco joints with toes.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
from glob import glob

from skimage.io import imread
import numpy as np


def get_upenn2coco():
    """
    Converts Upenn keypoints to 25 universal keypoints with toes.
    Note: UPenn does not have "heel".
    H36M has heel but not ankles.
    """
    coco_joint_names = [
        'R Heel',
        'R Knee',
        'R Hip',
        'L Hip',
        'L Knee',
        'L Heel',
        'R Wrist',
        'R Elbow',
        'R Shoulder',
        'L Shoulder',
        'L Elbow',
        'L Wrist',
        'Neck',
        'Head',
        'Nose',
        'L Eye',
        'R Eye',
        'L Ear',
        'R Ear',
        'L Big Toe',
        'R Big Toe',
        'L Small Toe',
        'R Small Toe',
        'L Ankle',
        'R Ankle',
    ]
    upenn_joint_names = [
        'Head',
        'R Shoulder',
        'L Shoulder',
        'R Elbow',
        'L Elbow',
        'R Wrist',
        'L Wrist',
        'R Hip',
        'L Hip',
        'R Knee',
        'L Knee',
        'R Ankle',
        'L Ankle',
        # Below are all 0 - missing parts.
        'Neck',
        'Nose',
        'L Eye',
        'R Eye',
        'L Ear',
        'R Ear',
        'L Big Toe',
        'R Big Toe',
        'L Small Toe',
        'R Small Toe',
        'L Heel',
        'R Heel',
    ]

    upenn2coco = [upenn_joint_names.index(name) for name in coco_joint_names]

    return upenn2coco, coco_joint_names


def read_labels(label_path):
    """
    Returns:
      - kp: N x 13 x 3
      - is_train
    """
    from scipy.io import loadmat
    anno = loadmat(label_path)

    # N x 13
    vis = anno['visibility']
    x = anno['x']
    y = anno['y']

    kps = np.dstack((x, y, vis))

    # Return N x 25 x 3, by appending extra 0s
    kps = np.hstack((kps, np.zeros((kps.shape[0], 25, 3))))

    is_train = anno['train'].ravel()[0]

    return kps, is_train


def show_seq(seq_path, label_path):
    import sys
    sys.path.append("../../util")
    from renderer import draw_skeleton

    upenn2coco_inds = get_upenn2coco()

    kps, is_train = read_labels(label_path)
    frame_paths = sorted(glob(osp.join(seq_path, '*.jpg')))
    for i, (frame_path, kp) in enumerate(zip(frame_paths, kps)[::3]):
        frame = imread(frame_path)
        # vis = kp[:, 2] > 0
        coco_kp = kp[upenn2coco_inds]
        skel_img = draw_skeleton(frame, coco_kp, vis=coco_kp[:, 2])
        import matplotlib.pyplot as plt
        plt.ion()
        plt.clf()
        plt.imshow(skel_img)
        plt.title('{}/{}'.format(i, len(frame_paths)))
        plt.draw()
        plt.pause(1e-3)


def read_upenn(base_dir):
    seqs = glob(osp.join(base_dir, 'frames/*'))
    for seq_path in seqs[3:]:
        seq_name = osp.basename(seq_path)
        label_path = osp.join(base_dir, 'labels', '{}.mat'.format(seq_name))

        show_seq(seq_path, label_path)
        import ipdb
        ipdb.set_trace()


if __name__ == '__main__':
    base_dir = '/scratch1/jason/upenn/Penn_Action'
    read_upenn(base_dir)
