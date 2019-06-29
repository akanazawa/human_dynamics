"""
Convert H36M to TFRecords.

Mosh data is not available
All pose and shape variables are None.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from glob import glob
import itertools
from os.path import basename, join, exists
from os import makedirs
import pickle

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from src.datasets.common import (
    convert_to_example_temporal,
    ImageCoder,
)
from src.util.common import resize_img
from src.datasets.make_test_tfrecords import save_seq_to_test_tfrecord
from src.datasets.resnet_extractor import FeatureExtractor
from src.util.render.render_utils import draw_skeleton
from src.util.tube_augmentation import TubePreprocessorDriver

h36_dir = '/scratch1/storage/human36m_25fps'
out_dir = '/scratch1/jason/tf_datasets/human36m_nomosh/'

tf.app.flags.DEFINE_string('data_directory', h36_dir,
                           'data directory: top of h36m')
tf.app.flags.DEFINE_string('output_directory', out_dir,
                           'Output data directory')
tf.app.flags.DEFINE_string('split', 'val', 'train, val, or test set.')
tf.app.flags.DEFINE_integer(
    'max_sequence_length', 150,
    'Maximum sequence length. Use -1 for full sequence.')

tf.app.flags.DEFINE_boolean('precomputed_phi', True,
                            'Put precomputed_phi in tfrecords.')
tf.app.flags.DEFINE_string('pretrained_model_path',
                           'models/hmr_noS5.ckpt-642561',
                           'Path to checkpoint to load weights from.')
tf.app.flags.DEFINE_integer(
    'num_copy', 1, 'if precomputed_phi is on, makes this many copy per video')
tf.app.flags.DEFINE_boolean('visualize', 'False',
                            'If True and making val/test, visualizes results.')

# Phi data augmentation params.
tf.app.flags.DEFINE_integer(
    'img_size', 224, 'Input image size to the network after preprocessing')
tf.app.flags.DEFINE_integer('trans_max', 20, 'Max value of translation jitter')
tf.app.flags.DEFINE_integer('delta_trans_max', 20,
                            'Max consecutive translation jitter')
tf.app.flags.DEFINE_float('scale_max', 0.3,
                          'Max value of scale jitter (power of 2)')
tf.app.flags.DEFINE_float('delta_scale_max', 0.3,
                          'Max consecutive scale jitter')

FLAGS = tf.app.flags.FLAGS

# Mapping from H36M joints to LSP joints (0:13). In this roder:
COMMON_JOINT_IDS = np.array([
    3,  # R ankle
    2,  # R knee
    1,  # R hip
    4,  # L hip
    5,  # L knee
    6,  # L ankle
    16,  # R Wrist
    15,  # R Elbow
    14,  # R shoulder
    11,  # L shoulder
    12,  # L Elbow
    13,  # L Wrist
    8,  # Neck top
    10,  # Head top
])


def process_image(
        im_path,
        gt2d,
        cam,
        coder,
        pose=None,
        shape=None,
        gt3d=None,
        vis=False,
):
    # Read image.
    with tf.gfile.FastGFile(im_path, 'rb') as f:
        image_data = f.read()
        image = coder.decode_jpeg(coder.png_to_jpeg(image_data))
        assert image.shape[2] == 3

    # Use gt2d to get the scale.
    min_pt = np.min(gt2d, axis=0)
    max_pt = np.max(gt2d, axis=0)
    person_height = np.linalg.norm(max_pt - min_pt)
    center = (min_pt + max_pt) / 2.
    scale = 150. / person_height

    image_scaled, scale_factors = resize_img(image, scale)
    joints_scaled = np.copy(gt2d)
    joints_scaled[:, 0] *= scale_factors[0]
    joints_scaled[:, 1] *= scale_factors[1]
    center_scaled = np.round(center * scale_factors).astype(np.int)
    # scale camera:
    cam_scaled = np.copy(cam)
    # Flength
    cam_scaled[0] *= scale
    # px
    cam_scaled[1] *= scale_factors[0]
    # py
    cam_scaled[2] *= scale_factors[1]

    # Make sure there is enough space to crop 300x300.
    image_padded = np.pad(image_scaled, ((300, ), (300, ), (0, )), 'edge')
    height, width = image_padded.shape[:2]
    center_scaled += 300
    joints_scaled += 300

    # Crop 300x300 around the center.
    margin = 150
    start_pt = (center_scaled - margin).astype(int)
    end_pt = (center_scaled + margin).astype(int)
    end_pt[0] = min(end_pt[0], width)
    end_pt[1] = min(end_pt[1], height)
    image_scaled = image_padded[start_pt[1]:end_pt[1], start_pt[0]:end_pt[
        0], :]
    # Update others too.
    joints_scaled[:, 0] -= start_pt[0]
    joints_scaled[:, 1] -= start_pt[1]
    center_scaled -= start_pt
    # Update principal point:
    cam_scaled[1] += 300 - start_pt[0]
    cam_scaled[2] += 300 - start_pt[1]
    height, width = image_scaled.shape[:2]
    im_shape = [height, width]
    # Vis:
    if vis:
        import matplotlib.pyplot as plt
        plt.ion()
        plt.clf()
        fig = plt.figure(1)
        ax = fig.add_subplot(121)
        image_with_skel = draw_skeleton(image, gt2d[:, :2])
        ax.imshow(image_with_skel)
        ax.axis('off')
        ax.scatter(center[0], center[1], color='red')
        ax = fig.add_subplot(122)
        image_with_skel_scaled = draw_skeleton(image_scaled,
                                               joints_scaled[:, :2])
        ax.imshow(image_with_skel_scaled)
        ax.scatter(center_scaled[0], center_scaled[1], color='red')

        # Project it.
        def project(X, c):
            y = X[:, :2] / X[:, 2].reshape(-1, 1)
            proj2d = c[0] * y + c[1:].reshape(1, -1)
            return proj2d

        proj2d = project(gt3d, cam_scaled)
        ax.scatter(proj2d[:, 0], proj2d[:, 1], s=4)
        ax.axis('off')
        import ipdb
        ipdb.set_trace()
    # Encode image.
    image_data_scaled = coder.encode_jpeg(image_scaled)
    # Put things together.
    label = np.vstack([joints_scaled.T, np.ones((1, len(COMMON_JOINT_IDS)))])

    return {
        'image_data': image_data_scaled,
        'image': image_scaled,
        'image_shape': im_shape,
        'label': label,
        'center': center_scaled,
        'scale_factors': scale_factors,
        'start_pt': start_pt,
        'cam_scaled': cam_scaled,
    }


def add_to_tfrecord(image_paths,
                    gt2ds,
                    gt3ds,
                    cams,
                    coder,
                    writer,
                    vis=None,
                    feature_extractor=None,
                    augmentor=None):
    """
    gt2d is T x 14 x 2 --> We need T for each for cropping
    gt3d is T x 14 x 3
    cam is T x 3 [f, px, py] --> We need T for each for cropping
    im_path is a list of len T
    Process each image
    """
    image_datas, image_shapes, labels, centers = [], [], [], []
    scale_factors, start_pts, cams_scaled = [], [], []

    images = []  # Used to compute phis if needed.

    for values in tqdm(list(zip(image_paths, gt2ds, cams, gt3ds))):
        image_path, gt2d, cam, gt3d = values

        # Gt3d and pose are just for visualization.
        ret_dict = process_image(
            image_path,
            gt2d,
            cam,
            coder,
            gt3d=gt3d,
        )
        image_datas.append(ret_dict['image_data'])
        image_shapes.append(ret_dict['image_shape'])
        labels.append(ret_dict['label'])
        centers.append(ret_dict['center'])
        scale_factors.append(ret_dict['scale_factors'])
        start_pts.append(ret_dict['start_pt'])
        cams_scaled.append(ret_dict['cam_scaled'])
        if feature_extractor is not None:
            # Images sent to augmentor must be [0, 1].
            images.append(ret_dict['image'] / 255.)

    if feature_extractor:
        print('Applying augmentation to videos')

        # augmentor needs facepts
        face_pts = np.zeros((len(labels), 3, 5))
        # it also needs toes
        toe_pts = np.zeros((len(labels), 3, 6))
        labels = np.dstack((labels, face_pts, toe_pts))
        pose_dummy = np.zeros((len(labels), 72))
        ret_dict = augmentor(
            images=images,
            image_sizes=image_shapes,
            labels=labels,
            centers=centers,
            poses=pose_dummy,
            gt3ds=gt3ds)
        del images  # Clear the memory.

        augmented_images = ret_dict['images']
        labels = ret_dict['labels']
        centers = ret_dict['centers']
        gt3ds = ret_dict['gt3ds']
        image_shapes = [list(img.shape[:2]) for img in augmented_images]

        print('Computing phis')
        phis = feature_extractor.compute_all_phis(augmented_images)
        # Make sure that this np.mean(phi[0])
        # is small (not more than 0.1, if it is that's a red flag)
        image_datas = [
            coder.encode_jpeg(((img + 1) * 0.5) * 255.)
            for img in augmented_images
        ]
    else:
        phis = None

    T = FLAGS.max_sequence_length
    if T == -1:
        T = len(image_datas)
    for i in range(0, len(image_datas), T):
        if i + T > len(image_datas):
            # If reach end, then just take the last T
            i = len(image_datas) - T
        example = convert_to_example_temporal(
            image_datas=image_datas[i:i + T],
            image_paths=image_paths[i:i + T],
            image_shapes=image_shapes[i:i + T],
            labels=labels[i:i + T],
            centers=centers[i:i + T],
            gt3ds=gt3ds[i:i + T],
            poses=None,
            shape=None,
            scale_factors=scale_factors[i:i + T],
            start_pts=start_pts[i:i + T],
            cams=cams_scaled[i:i + T],
            phis=phis[i:i + T],
        )
        writer.write(example.SerializeToString())


def get_all_data(img_dir, split='train'):
    im_paths = sorted(glob(join(img_dir, "*.png")))

    # Load GT in LSP-joint format.
    gt_path = join(img_dir, "gt_poses.pkl")
    with open(gt_path, 'rb') as f:
        gts = pickle.load(f, encoding='latin1')
        gt2ds = gts['2d']
        gt3ds = gts['3d']
    # Take subset joints.
    gt2ds = [gt2d[COMMON_JOINT_IDS] for gt2d in gt2ds]
    if split != 'train':
        # Add the face/toe points and set visibilities to 1.
        vis = np.ones((14, 1))
        add_points = np.zeros((11, 3))
        gt2ds = [
            np.vstack((np.hstack((gt2d, vis)), add_points)) for gt2d in gt2ds
        ]
    # Fix units: meter -> mm.
    gt3ds = [gt3d[COMMON_JOINT_IDS] / 1000. for gt3d in gt3ds]

    # Load cam.
    cam_path = join(img_dir, "camera_wext.pkl")
    with open(cam_path, 'rb') as fcam:
        cam = pickle.load(fcam, encoding='latin1')
        flength = 0.5 * (cam['f'][0] + cam['f'][1])
        ppt = cam['c']
        # Repeat them.
        flengths = np.tile(flength, (len(gt3ds), 1))
        ppts = np.tile(ppt, (len(gt3ds), 1))
        cams = np.hstack((flengths, ppts))

    return im_paths, gt2ds, gt3ds, cams


def save_seq_to_tfrecord(out_name,
                         im_paths,
                         gt2ds,
                         gt3ds,
                         cams,
                         feature_extractor=None,
                         augmentor=None):
    coder = ImageCoder()
    print('Starting tfrecord file: {}'.format(out_name))
    with tf.python_io.TFRecordWriter(out_name) as writer:
        add_to_tfrecord(
            image_paths=im_paths,
            gt2ds=gt2ds,
            gt3ds=gt3ds,
            cams=cams,
            coder=coder,
            writer=writer,
            feature_extractor=feature_extractor,
            augmentor=augmentor,
        )


def save_seq_to_tfrecord_driver(out_name,
                                im_paths,
                                gt2ds,
                                gt3ds,
                                cams,
                                feature_extractor=None,
                                augmentor=None):
    if feature_extractor is None or '/test/' in out_name:
        save_seq_to_tfrecord(
            out_name=out_name,
            im_paths=im_paths,
            gt2ds=gt2ds,
            gt3ds=gt3ds,
            cams=cams,
            feature_extractor=feature_extractor,
            augmentor=augmentor)
    else:
        # loop over to make copies.
        for i in range(FLAGS.num_copy):
            out_name_here = out_name.replace('.tfrecord',
                                             '_copy%d.tfrecord' % i)
            if exists(out_name_here):
                print('%s done' % out_name)
                continue

            save_seq_to_tfrecord(
                out_name=out_name_here,
                im_paths=im_paths,
                gt2ds=gt2ds,
                gt3ds=gt3ds,
                cams=cams,
                feature_extractor=feature_extractor,
                augmentor=augmentor)


def process_h36(data_dir,
                out_dir,
                split='val',
                feature_extractor=None,
                augmentor=None):

    # Set tfrecord path name.
    path_name = 'cam{cam_id:02d}_S{sub_id:02d}_{act}_{trial_id}'
    if feature_extractor is not None:
        path_name += '_{}'.format(basename(feature_extractor.model_path))
    path_name += '.tfrecord'

    if split == 'train':
        print('Training Set!')
        sub_ids = [1, 6, 7, 8]
        out_dir = join(out_dir, 'train')
        if not exists(out_dir):
            makedirs(out_dir)
        out_path = join(out_dir, 'train_' + path_name)
    elif split == 'val':
        print('Val Set!')
        sub_ids = [5]
        out_dir = join(out_dir, 'val')
        if not exists(out_dir):
            makedirs((out_dir))
        out_path = join(
            out_dir,
            'val_' + path_name,
        )

    elif split == 'test':
        print('Test Set!')
        sub_ids = [9, 11]
        out_dir = join(out_dir, 'test')
        if not exists(out_dir):
            makedirs(out_dir)
        out_path = join(out_dir, 'test_' + path_name)
    else:
        raise Exception('split not found.')

    action_names = [
        'Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Posing',
        'Purchases', 'Sitting', 'SittingDown', 'Smoking', 'TakingPhoto',
        'Waiting', 'Walking', 'WakingDog', 'WalkTogether'
    ]
    trial_ids = [0, 1]
    cam_ids = range(0, 4)

    permutations = list(
        itertools.product(sub_ids, trial_ids, cam_ids, action_names))

    for i, (sub_id, trial_id, cam_id, act) in enumerate(permutations):
        print('{}/{}'.format(i, len(permutations)))
        out_name = out_path.format(
            cam_id=cam_id, sub_id=sub_id, act=act, trial_id=trial_id)
        image_dir = join(data_dir, 'S{}'.format(sub_id), '{}_{}'.format(
            act, trial_id), 'cam_{}'.format(cam_id))
        if sub_id == 11 and trial_id == 1 and \
                (act == 'Phoning' or act == 'Directions'):
            print('Skipping', out_name, 'bc no data')
            continue
        assert exists(image_dir)
        image_paths, gt2ds, gt3ds, cams = get_all_data(
            image_dir, split=split)
        if exists(out_name):
            print('Already exists, skip', out_name)
            continue

        if split == 'train':
            save_seq_to_tfrecord_driver(
                out_name=out_name,
                im_paths=image_paths,
                gt2ds=gt2ds,
                gt3ds=gt3ds,
                cams=cams,
                feature_extractor=feature_extractor,
                augmentor=augmentor)
        else:
            save_seq_to_test_tfrecord(
                out_name=out_name,
                im_paths=image_paths,
                all_gt2ds=[gt2ds],
                all_gt3ds=[gt3ds],
                visualize=FLAGS.visualize,
                img_size=FLAGS.img_size,
                sigma=3,
            )


def main(unused_argv):
    print('Saving results to {}'.format(FLAGS.output_directory))

    if FLAGS.precomputed_phi:
        pmp = FLAGS.pretrained_model_path
        assert pmp is not None
        feature_extractor = FeatureExtractor(
            model_path=pmp,
            img_size=FLAGS.img_size,
            batch_size=64,
        )
        augmentor = TubePreprocessorDriver(
            img_size=FLAGS.img_size,
            trans_max=FLAGS.trans_max,
            delta_trans_max=FLAGS.delta_trans_max,
            scale_max=FLAGS.scale_max,
            delta_scale_max=FLAGS.delta_scale_max)
    else:
        feature_extractor = None
        augmentor = None

    process_h36(
        data_dir=FLAGS.data_directory,
        out_dir=FLAGS.output_directory,
        split=FLAGS.split,
        feature_extractor=feature_extractor,
        augmentor=augmentor,
    )


if __name__ == '__main__':
    tf.app.run(main)
