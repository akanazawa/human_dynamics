"""
Convert insta_variety to TFRecords Video.
Makes both splits.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

from os import makedirs
import os.path as osp
from os import walk

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from scipy.signal import medfilt

from src.datasets.make_test_tfrecords import save_seq_to_test_tfrecord
from src.datasets.common import (
    convert_to_example_temporal,
    ImageCoder,
)

from src.util.common import resize_img
from src.util.render.render_utils import draw_skeleton
from src.util.smooth_bbox import get_smooth_bbox_params

from src.datasets.resnet_extractor import FeatureExtractor
from src.util.tube_augmentation import TubePreprocessorDriver

# Flags
tf.app.flags.DEFINE_string('split', 'train', 'train or test')
tf.app.flags.DEFINE_string('data_directory',
                           '/data2/Data/instagram_download/labels_logits/{}',
                           'data directory: top of Penn Action')
tf.app.flags.DEFINE_string('output_directory',
                           '/data2/Data/tf_datasets_phi_shard_release/insta_variety',
                           'Output data directory')
tf.app.flags.DEFINE_string('image_directory',
                           '/data2/Data/instagram_download/frames_raw/{}/{}',
                           'frame locations')
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
# In this version only copies >=1 will have the copy suffix.
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
tf.app.flags.DEFINE_boolean('visualize', 'False', 'If True, visualizes.')

# Constants
FLAGS = tf.app.flags.FLAGS

MIN_VIS_PTS = 4  # Cut off video when num vis < this.
MIN_VIS_START_PTS = 6  # Video starts after this many points are visible
MIN_NUM_FRAMES = 40
MAX_FRAME_LENGTH = 500

# The names of universal 25 joints with toes.
joint_names_coco = [
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


def process_image(im_path, gt2d, coder, bbox_param, DRAW=False):
    with tf.gfile.FastGFile(im_path, 'rb') as f:
        image_data = f.read()
    image = coder.decode_jpeg(image_data)
    assert image.shape[2] == 3, \
        '{} has {} channels.'.format(im_path, image.shape[2])

    center = bbox_param[:2]
    scale = bbox_param[2]

    # estimate height..
    # Using vis_threshold 0 for DT
    vis = gt2d[:, 2] > 0.

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
    image_scaled = image_padded[start_pt[1]:end_pt[1], start_pt[0]:end_pt[
        0], :]
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

        import ipdb
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
                    DRAW=False,
                    vis_thresh=0,
                    sigma=3):
    """
    Adds all information from a subject-sequence-camera tuple to a tfrecord.
    """
    image_datas, image_shapes, labels, centers = [], [], [], []
    scale_factors, start_pts = [], []

    images = []  # Used to compute phis if needed.

    bbox_params, time_pt1, time_pt2 = get_smooth_bbox_params(
        gt2ds, vis_thresh, sigma=sigma)
    
    #import ipdb; ipdb.set_trace()
    for i, (im_path, gt2d, bbox_param) in enumerate(
            list(zip(image_paths, gt2ds, bbox_params))[time_pt1:time_pt2]):

        # This can be improved a lot.
        ret_dict = process_image(im_path, gt2d, coder, bbox_param, DRAW)

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

        phis = feature_extractor.compute_all_phis(augmented_imgs)
        del images  # Clear the memory.
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


def clean_video(image_paths, gt2ds):
    """
    returns None if video is bad
    ow. returns cleaned/adjusted image_paths and gt2ds
    """
    for im_path in image_paths:
        if not osp.exists(im_path):
            print('!!--{} doesnt exist! Skipping..--!!'.format(im_path))
            import ipdb
            ipdb.set_trace()
            return None, None

    num_vis = np.sum(gt2ds[:, :, 2] > 0, axis=1)
    num_vis_smoothed = medfilt(num_vis, 5)
    # If nothing is visible in first frame, go forward until it does.
    if num_vis_smoothed[0] <= MIN_VIS_START_PTS:
        # some annotation does not start for a while
        start_pt = np.argmax(num_vis_smoothed > MIN_VIS_START_PTS)

        image_paths = image_paths[start_pt:]

        gt2ds = gt2ds[start_pt:]
        num_vis = np.sum(gt2ds[:, :, 2] > 0, axis=1)
        num_vis_smoothed = medfilt(num_vis, 5)

    if np.any(num_vis_smoothed <= MIN_VIS_PTS):
        cut_off = (num_vis_smoothed > MIN_VIS_PTS).tolist().index(False)
        gt2ds = gt2ds[:cut_off]
        image_paths = image_paths[:cut_off]

    if len(image_paths) < MIN_NUM_FRAMES:
        print('Video too short! {}'.format(len(image_paths)))
        return None, None

    # Remove videos that's just face & shoulder
    face_should = [
        'R Shoulder', 'L Shoulder', 'Neck', 'Head', 'Nose', 'L Eye', 'R Eye',
        'L Ear', 'R Ear'
    ]
    face_should_inds = [joint_names_coco.index(name) for name in face_should]
    non_face_should_inds = [
        i for i in range(len(joint_names_coco)) if i not in face_should_inds
    ]

    vis = gt2ds[:, :, 2]
    # if these keypoints are not visible for more than 90%
    # of the track drop it.
    num_vis_bottom = np.sum(vis[:, non_face_should_inds], axis=1)
    if np.sum(num_vis_bottom == 0) / float(num_vis_bottom.shape[0]) >= 0.4:
        print('Face only! Skip.')
        return None, None

    if len(image_paths) > MAX_FRAME_LENGTH:
        image_paths = image_paths[:MAX_FRAME_LENGTH]
        gt2ds = gt2ds[:MAX_FRAME_LENGTH]

    return image_paths, gt2ds


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
        im_paths (list): List of image path names.
        gt2ds (Nx19x2): 2D ground truth.
    """
    DRAW = False
    if osp.exists(out_name):
        print('Done')
        return True

    print('Starting tfrecord file: {}'.format(out_name))
    with tf.python_io.TFRecordWriter(out_name) as writer:
        for i, (image_paths, gt2ds) in tqdm(
                enumerate(zip(all_image_paths, all_gt2ds))):
            add_to_tfrecord(
                image_paths=image_paths,
                gt2ds=gt2ds,
                coder=coder,
                writer=writer,
                feature_extractor=feature_extractor,
                augmentor=augmentor,
                DRAW=DRAW
            )


def process_videos(out_dir,
                   seq_paths,
                   all_kps_raw,
                   feature_extractor=None,
                   augmentor=None):

    all_frame_paths = []
    all_kps = []
    for i, (frame_paths, kps) in enumerate(zip(seq_paths, all_kps_raw)):
        frame_paths_ = frame_paths[::FLAGS.frame_skip]
        kps_ = kps[::FLAGS.frame_skip]
        frame_paths_, kps_ = clean_video(frame_paths_, kps_)
        if frame_paths_ is not None:
            all_frame_paths.append(frame_paths_)
            all_kps.append(kps_)

    np.random.seed(0)
    shuffle_inds = np.random.permutation(len(all_frame_paths))
    all_frame_paths = [all_frame_paths[ind] for ind in shuffle_inds]
    all_kps = [all_kps[ind] for ind in shuffle_inds]
    average_track_length = np.mean([len(atrl) for atrl in all_frame_paths])
    median_track_length = np.median([len(atrl) for atrl in all_frame_paths])

    print('Number total videos passed: {}, orig: {}'.format(
        len(all_frame_paths), len(all_kps_raw)))
    print('Average track length: {}, median: {}'.format(
        average_track_length, median_track_length))
    print('Total # of frames {}'.format(
        np.sum([len(a) for a in all_frame_paths])))

    print('Saving {} tracks'.format(len(all_frame_paths)))
    coder = ImageCoder()
    num_copy = FLAGS.num_copy
    num_shards = FLAGS.num_shards
    num_tfrecords = int(np.ceil(len(all_frame_paths) / num_shards))
    for i in range(num_tfrecords):
        end_ind = min((i + 1) * num_shards, len(all_frame_paths))
        impaths_here = all_frame_paths[i * num_shards:end_ind]
        kps_here = all_kps[i * num_shards:end_ind]
        for j in range(num_copy):
            out_name = 'insta_variety_%02d_copy%02d' % (i, j)
            if feature_extractor is not None:
                out_name += '_{}'.format(
                    osp.basename(feature_extractor.model_path))

            out_file = osp.join(out_dir, '{}.tfrecord'.format(out_name))
            save_to_tfrecord(out_file, impaths_here, kps_here,
                                              coder, feature_extractor,
                                              augmentor)


def process_videos_test(out_dir,
                        seq_paths,
                        all_kps,
                        feature_extractor=None,
                        augmentor=None):
    """
    seq_path is a path to the directory of a seq.
    each video becomes a tfrecord.
    """
    for i, (frame_paths, kps) in enumerate(zip(seq_paths, all_kps)):
        frame_paths_ = frame_paths[::FLAGS.frame_skip]
        kps_ = kps[::FLAGS.frame_skip]
        out_name = '_'.join(frame_paths[0].split('/')[-3:-1])
        out_file = osp.join(out_dir, '{}.tfrecord'.format(out_name))
        frame_paths_, kps_ = clean_video(frame_paths_, kps_)
        if frame_paths_ is None:
            # bad video.
            continue
        save_seq_to_test_tfrecord(
            out_name=out_file,
            im_paths=frame_paths_,
            all_gt2ds=[kps_],
            visualize=FLAGS.visualize,
            vis_thresh=0,
        )


def get_seq_labels(data_dir, split):
    print(data_dir)
    # @Panna: change this as flag?
    with open('/data2/Code/insta/insta_diverse_full_video_list_shuffle.txt'
              ) as f:
        content = f.readlines()
    # You may also want to remove whitespace characters like
    # `\n` at the end of each line
    video_code_list = [x.strip() for x in content]
    if split == 'train':
        video_code_range = range(0, 2000)
    elif split == 'test':
        video_code_range = range(2000, len(video_code_list))
    video_code_list_split = [
        v for (i, v) in enumerate(video_code_list) if i in video_code_range
    ]

    if split == 'test':
        save_path = data_dir.format('{}_test.npz'.format(split))
    elif split == 'train':
        save_path = data_dir.format('{}.npz'.format(split))
    else:
        print('bad split: {}'.format(split))
        import ipdb
        ipdb.set_trace()

    if osp.exists(save_path):
        data = np.load(save_path)
        kps = [kp for kp in data['kps']]
        return data['paths'], kps
    else:
        kps = []
        paths = []
        for video_code in video_code_list_split:
            _, sequence_dir_list, _ = walk(
                data_dir.format('{}/shot_split'.format(video_code))).__next__()
            for seq_num in sequence_dir_list:
                jsondir = data_dir.format('{}/shot_split/{}/{}'.format(
                    video_code, seq_num, '{}'))
                print(jsondir)
                _, _, json_file_list = walk(jsondir[:-2]).__next__()
                json_file_list = sorted(json_file_list)
                kps_sequence = []
                train_impaths_sequence = []
                for jf in json_file_list:
                    filename_json = jsondir.format(jf)
                    with open(filename_json) as f:
                        data = json.load(f)
                    kps_c = []
                    for cjn in joint_names_coco:
                        # this thresholding is the main difference
                        # from processing openpose instead of DT
                        # TODO @Panna, I think this needs to be something like:
                        # if cjn in ['Head', 'R Heel', 'L Heel']
                        # (bc openpose doesn't have these parts)
                        if cjn == 'Head':
                            kps_c.append([0, 0, 0])
                        else:
                            kps_c.append([
                                data[cjn]['x'], data[cjn]['y'],
                                int(data[cjn]['logits'] >= 0.1)
                            ])
                    kps_sequence.append(kps_c)
                    train_impaths_sequence.append(
                        FLAGS.image_directory.format(video_code,
                                                     data['imloc']))
                kps.append(np.array(kps_sequence))
                paths.append(train_impaths_sequence)
        print('# {}: {}'.format(split, len(paths)))
        np.savez(save_path, paths=paths, kps=kps)

    return paths, kps


def mkdir(dir_path):
    if not osp.exists(dir_path):
        makedirs(dir_path)


def main(unused_argv):
    print('Saving results to {}'.format(FLAGS.output_directory))

    mkdir(FLAGS.output_directory)

    train_dir = osp.join(FLAGS.output_directory, 'train')
    test_dir = osp.join(FLAGS.output_directory, 'test')

    mkdir(train_dir)
    mkdir(test_dir)

    if FLAGS.precomputed_phi:
        print('Precomputhing phi!!')
        pmp = FLAGS.pretrained_model_path
        assert pmp is not None
        feature_extractor = FeatureExtractor(
            model_path=pmp,
            img_size=224,
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

    if FLAGS.split == 'train':
        train_paths, train_kps = get_seq_labels(FLAGS.data_directory, 'train')

        process_videos(
            train_dir,
            train_paths,
            train_kps,
            feature_extractor=feature_extractor,
            augmentor=augmentor)
    else:
        test_paths, test_kps = get_seq_labels(FLAGS.data_directory, 'test')
        process_videos_test(
            test_dir,
            test_paths,
            test_kps,
        )


if __name__ == '__main__':
    tf.app.run(main)
