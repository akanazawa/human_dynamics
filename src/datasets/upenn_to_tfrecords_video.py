"""
Convert UPenn to TFRecords Video.
Makes both splits.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from glob import glob
from os import makedirs
import os.path as osp

import ipdb
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from src.datasets.common import (
    convert_to_example_temporal,
    ImageCoder,
)
from src.datasets.make_test_tfrecords import save_seq_to_test_tfrecord
from src.datasets.resnet_extractor import FeatureExtractor
from src.datasets.upenn.read_upenn import (
    get_upenn2coco,
    read_labels,
)
from src.util.common import resize_img
from src.util.render.render_utils import draw_skeleton
from src.util.tube_augmentation import TubePreprocessorDriver

# Flags
tf.app.flags.DEFINE_string('split', 'train', 'train, val, or test')
tf.app.flags.DEFINE_string('data_directory',
                           '/scratch1/jason/upenn/Penn_Action/',
                           'data directory: top of Penn Action')
tf.app.flags.DEFINE_string('output_directory',
                           '/scratch1/tf_datasets/penn_action',
                           'Output data directory')

tf.app.flags.DEFINE_integer('max_frames', -1,
                            'Number of frames to include (default: all).')
tf.app.flags.DEFINE_integer('frame_skip', 1, 'Frame skip rate.')

tf.app.flags.DEFINE_boolean('precomputed_phi', True,
                            'If true, saves precomputed phi.')
tf.app.flags.DEFINE_string('pretrained_model_path',
                           'models/hmr_noS5.ckpt-642561',
                           'Path to checkpoint to load weights from.')
tf.app.flags.DEFINE_boolean(
    'save_img', True,
    'Only relevant if precomputed_phi. If true, saves the images.')
tf.app.flags.DEFINE_integer('num_shards', 50,
                            'How many tube to put in each tfrecord')
tf.app.flags.DEFINE_boolean('visualize', 'False', 'If True, visualizes.')
tf.app.flags.DEFINE_integer(
    'num_copy', 1, 'if precomputed_phi is on, makes this many copy per video')


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

# Constants
FLAGS = tf.app.flags.FLAGS

MIN_VIS_PTS = 4  # Cut off video if num vis < this.
MIN_NUM_FRAMES = 40

joint_idx2coco, coco_joint_names = get_upenn2coco()


def process_image(im_path, gt2d, coder, DRAW=False):
    with tf.gfile.FastGFile(im_path, 'rb') as f:
        image_data = f.read()
    image = coder.decode_jpeg(image_data)
    assert image.shape[2] == 3, \
        '{} has {} channels.'.format(im_path, image.shape[2])

    # estimate height..
    vis = gt2d[:, 2] > 0

    min_pt = np.min(gt2d[vis, :2], axis=0)
    max_pt = np.max(gt2d[vis, :2], axis=0)
    person_height = np.linalg.norm(max_pt - min_pt)
    center = (min_pt + max_pt) / 2.
    scale = 150. / person_height

    if person_height < 0.5:
        return False

    image_scaled, scale_factors = resize_img(image, scale)
    joints_scaled = np.copy(gt2d[:, :2])
    joints_scaled[:, 0] *= scale_factors[0]
    joints_scaled[:, 1] *= scale_factors[1]
    center_scaled = np.round(center * scale_factors).astype(np.int)

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
    image_scaled = image_padded[start_pt[1]:end_pt[1], start_pt[0]:end_pt[0], :]
    # Update others too.
    joints_scaled[:, 0] -= start_pt[0]
    joints_scaled[:, 1] -= start_pt[1]
    center_scaled -= start_pt
    height, width = image_scaled.shape[:2]
    im_shape = [height, width]

    # DRAW:
    if DRAW:
        import matplotlib.pyplot as plt
        plt.ion()
        plt.clf()
        fig = plt.figure(1)
        # ax = fig.add_subplot(131)
        ax = fig.add_subplot(121)
        image_with_skel = draw_skeleton(image, gt2d[:, :2], vis=vis)
        ax.imshow(image_with_skel)
        ax.axis('off')
        ax.scatter(center[0], center[1], color='red')
        # ax = fig.add_subplot(132)
        ax = fig.add_subplot(122)
        image_with_skel_scaled = draw_skeleton(
            image_scaled, joints_scaled[:, :2], vis=vis)
        ax.imshow(image_with_skel_scaled)
        ax.scatter(center_scaled[0], center_scaled[1], color='red')
        plt.draw()
        plt.show()
        ipdb.set_trace()
    # Encode image.
    image_data_scaled = coder.encode_jpeg(image_scaled)
    label = np.vstack([joints_scaled.T, vis])

    return {
        'image_data': image_data_scaled,
        'image': image_scaled,
        'image_shape': im_shape,
        'label': label,
        'center': center_scaled,
        'scale_factors': scale_factors,
        'start_pt': start_pt,
    }


def add_to_tfrecord(image_paths,
                    gt2ds,
                    coder,
                    writer,
                    feature_extractor=None,
                    augmentor=None,
                    DRAW=False):
    """
    Adds all information from a subject-sequence-camera tuple to a tfrecord.
    """
    image_datas, image_shapes, labels, centers = [], [], [], []
    scale_factors, start_pts = [], []

    images = []  # Used to compute phis if needed.

    failed = []
    for i, (im_path, gt2d) in enumerate(zip(image_paths, gt2ds)):

        # This can be improved a lot.
        ret_dict = process_image(im_path, gt2d, coder, DRAW)

        image_datas.append(ret_dict['image_data'])
        image_shapes.append(ret_dict['image_shape'])
        labels.append(ret_dict['label'])
        centers.append(ret_dict['center'])
        scale_factors.append(ret_dict['scale_factors'])
        start_pts.append(ret_dict['start_pt'])
        if feature_extractor is not None:
            # AJ: Make sure images you send to augmentor is [0, 1]!!
            images.append(ret_dict['image'] / 255.)

    # Apply Data Augmentation & Feature Extraction
    if feature_extractor:
        # print('Applying augmentation to videos')
        pose_dummy = np.zeros((len(labels), 72))
        gt3d_dummy = np.zeros((len(labels), 14, 3))
        ret_dict = augmentor(
            images=images,
            image_sizes=image_shapes,
            labels=labels,
            centers=centers,
            poses=pose_dummy,
            gt3ds=gt3d_dummy)
        augmented_imgs = ret_dict['images']
        labels = ret_dict['labels']
        centers = ret_dict['centers']
        image_shapes = [list(img.shape[:2]) for img in augmented_imgs]
        # if you have 3d data also update the gt3d and pose!

        phis = feature_extractor.compute_all_phis(augmented_imgs)
        del images  # Clear the memory.

        # Now update image_datas, undoing the preprocessing:
        # Make sure here you put the images back to [0, 255]
        image_datas = [
            coder.encode_jpeg(((img + 1) * 0.5) * 255.)
            for img in augmented_imgs
        ]
    else:
        phis = None

    if not FLAGS.save_img:
        image_datas = None

    example = convert_to_example_temporal(
        image_datas=image_datas,
        image_paths=image_paths,
        image_shapes=image_shapes,
        labels=labels,
        centers=centers,
        gt3ds=None,
        poses=None,
        shape=None,
        scale_factors=scale_factors,
        start_pts=start_pts,
        cams=None,
        phis=phis,
    )
    writer.write(example.SerializeToString())

    if len(failed):
        for im_path in failed:
            print(im_path)
    return True


def save_to_tfrecord(out_name,
                     all_image_paths,
                     all_gt2ds,
                     coder,
                     feature_extractor=None,
                     augmentor=None):
    """
    Saves all images to tfrecord.

    Args:
        out_name (str): Name of tfrecord.
        all_image_paths (list): List of list of image path names.
        all_gt2ds (list): List of (Nx19x3) 2D ground truth.
    """
    DRAW = False
    if osp.exists(out_name):
        print('Done')
        return True

    num_ppl = 0
    print('Starting tfrecord file: {}'.format(out_name))
    with tf.python_io.TFRecordWriter(out_name) as writer:
        for i, (image_paths, gt2ds) in enumerate(
                zip(all_image_paths, all_gt2ds)):
            if i % 10 == 0:
                print('%d/%d' % (i, len(all_image_paths)))
            image_paths, gt2ds = clean_video(image_paths, gt2ds)
            if image_paths is None:
                continue
            success = add_to_tfrecord(
                image_paths=image_paths,
                gt2ds=gt2ds,
                coder=coder,
                writer=writer,
                feature_extractor=feature_extractor,
                augmentor=augmentor,
                DRAW=DRAW)
            if success:
                num_ppl += 1

    return success


def clean_video(image_paths, gt2ds):
    """
    returns None if video is bad
    ow. returns cleaned/adjusted image_paths and gt2ds
    """
    for im_path in image_paths:
        if not osp.exists(im_path):
            print('!!--{} doesnt exist! Skipping..--!!'.format(im_path))
            ipdb.set_trace()
            return None, None

    # Cut the frame off at minimum # of visible points.
    num_vis = np.sum(gt2ds[:, :, 2] > 0, axis=1)
    if num_vis[0] == 0:
        # ugly,, but one video starts out with 0 kp
        num_vis = num_vis[1:]
        gt2ds = gt2ds[1:]
        image_paths = image_paths[1:]

    if np.any(num_vis <= MIN_VIS_PTS):
        cut_off = (num_vis > MIN_VIS_PTS).tolist().index(False)
        gt2ds = gt2ds[:cut_off]
        image_paths = image_paths[:cut_off]

    if len(image_paths) < MIN_NUM_FRAMES:
        print('Video too short! {}'.format(len(image_paths)))
        return None, None

    return image_paths, gt2ds


def process_videos(out_dir,
                   seq_paths,
                   all_kps_raw,
                   feature_extractor=None,
                   augmentor=None):
    """
    Used for training.
    seq_path is a path to the directory of a seq.
    extracts all kps and images, shuffles and then stores in a shard.
    """

    all_frame_paths = []
    all_kps = []
    for i, (seq_path, kps) in tqdm(enumerate(zip(seq_paths, all_kps_raw))):
        frame_paths = sorted(glob(osp.join(seq_path, '*.jpg')))
        # Filtering things here.
        frame_paths, kps = clean_video(frame_paths, kps)
        if frame_paths is not None:
            all_frame_paths.append(frame_paths)
            all_kps.append(kps)

    np.random.seed(0)
    shuffle_inds = np.random.permutation(len(all_frame_paths))
    all_frame_paths = [all_frame_paths[ind] for ind in shuffle_inds]
    all_kps = [all_kps[ind] for ind in shuffle_inds]

    print('Saving {} tracks'.format(len(all_frame_paths)))

    coder = ImageCoder()
    num_copy = FLAGS.num_copy
    num_shards = FLAGS.num_shards
    num_tfrecords = int(np.ceil(len(all_frame_paths) / num_shards))
    num_added = 0
    for i in range(num_tfrecords):
        end_ind = min((i + 1) * num_shards, len(all_frame_paths))
        impaths_here = all_frame_paths[i * num_shards:end_ind]
        kps_here = all_kps[i * num_shards:end_ind]
        for j in range(num_copy):
            out_name = 'penn_action_%02d_copy%02d' % (i, j)
            if feature_extractor is not None:
                out_name += '_{}'.format(
                    osp.basename(feature_extractor.model_path))

            out_file = osp.join(out_dir, '{}.tfrecord'.format(out_name))
            num_added_here = save_to_tfrecord(out_file, impaths_here, kps_here,
                                              coder, feature_extractor,
                                              augmentor)
            if j == 0:
                num_added += num_added_here

    print('shouldve added %d people, actual: %d' % (len(all_frame_paths),
                                                    num_added))


def process_videos_test(out_dir, seq_paths, all_kps):
    """
    seq_path is a path to the directory of a seq.
    each video becomes a tfrecord
    """
    for i, (seq_path, kps) in enumerate(zip(seq_paths, all_kps)):
        frame_paths = sorted(glob(osp.join(seq_path, '*.jpg')))
        out_name = osp.basename(seq_path)
        out_file = osp.join(out_dir, '{}.tfrecord'.format(out_name))
        if osp.exists(out_file):
            continue
        print('{}/{}: {}'.format(i, len(seq_paths), out_file))
        save_seq_to_test_tfrecord(
            out_name=out_file,
            im_paths=frame_paths,
            all_gt2ds=[kps],
            visualize=FLAGS.visualize,
            vis_thresh=0.1,
        )


def get_seq_labels(data_dir):
    seqs = sorted(glob(osp.join(data_dir, 'frames/*')))
    train_seqs, train_kps = [], []
    valtest_seqs, valtest_kps = [], []
    for seq_path in tqdm(seqs):
        seq_name = osp.basename(seq_path)
        label_path = osp.join(data_dir, 'labels', '{}.mat'.format(seq_name))
        kps, is_train = read_labels(label_path)
        # Convert kps into coco-universal order
        kps = kps[:, joint_idx2coco]

        if is_train > 0:
            train_kps.append(kps)
            train_seqs.append(seq_path)
        else:
            valtest_kps.append(kps)
            valtest_seqs.append(seq_path)

    # Now split the test into val and test.
    # Take every 10th as val.
    val_kps, val_seqs = [], []
    test_kps, test_seqs = [], []
    for i, (kp, seq) in enumerate(zip(valtest_kps, valtest_seqs)):
        if i % 10 == 0:
            val_kps.append(kp)
            val_seqs.append(seq)
        else:
            test_kps.append(kp)
            test_seqs.append(seq)

    print('# train: {} #val: {}, # test: {}'.format(len(train_seqs), len(val_seqs), len(test_seqs)))

    return train_seqs, train_kps, val_seqs, val_kps, test_seqs, test_kps


def mkdir(dir_path):
    if not osp.exists(dir_path):
        makedirs(dir_path)


def main(unused_argv):
    print('Saving results to {}'.format(FLAGS.output_directory))

    mkdir(FLAGS.output_directory)

    if FLAGS.split == 'train':
        train_dir = osp.join(FLAGS.output_directory, 'train')
        mkdir(train_dir)
    else:
        val_dir = osp.join(FLAGS.output_directory, 'val')
        test_dir = osp.join(FLAGS.output_directory, 'test')
        mkdir(val_dir)
        mkdir(test_dir)

    if FLAGS.split == 'train' and FLAGS.precomputed_phi:
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

    # Load all labels.
    print('loading labels')
    train_paths, train_kps, val_paths, val_kps, test_paths, test_kps = get_seq_labels(
        FLAGS.data_directory)

    if FLAGS.split == 'train':
        process_videos(
            train_dir,
            train_paths,
            train_kps,
            feature_extractor=feature_extractor,
            augmentor=augmentor
        )
    elif FLAGS.split == 'val':
        process_videos_test(
            val_dir,
            val_paths,
            val_kps
        )
    elif FLAGS.split == 'test':
        process_videos_test(
            test_dir,
            test_paths,
            test_kps
        )
    else:
        print('Split {} not found'.format(FLAGS.split))


if __name__ == '__main__':
    tf.app.run(main)

