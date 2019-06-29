"""
Takes in original H3.6M, renames and writes frames to file,
create annotations in a single pickle file that the later
h36_to_tfrecords_video.py can read in.

Mosh data is not available.

Requires spacepy & CDF:
- Get CDF (NASA): https://cdf.gsfc.nasa.gov/html/sw_and_docs.html
- Follow the instructions in README.install
- need to run . ~/Downloads/cdf36_4-dist/bin/definitions.B
for spacepy to work.

Then run:
python -m src.datasets.h36.read_human36m

Originally developed by Federica Bogo, edited by Angjoo Kanazawa
"""

from glob import glob
import os
from os import makedirs, system
from os.path import join, getsize, exists
import pickle
from spacepy import pycdf
import sys

import cv2
import numpy as np

from absl import flags


flags.DEFINE_string(
    'source_dir', '/scratch1/storage/human36m_full_raw',
    'Root dir of the original Human3.6M dataset unpacked with metadata.xml'
)
flags.DEFINE_string('out_dir', '/scratch1/storage/human36m_25fps',
                    'Output directory')
flags.DEFINE_integer('frame_skip', 2,
                     'subsample factor, 5 corresponds to 10fps, 2=25fps')

FLAGS = flags.FLAGS

colors = np.random.randint(0, 255, size=(17, 3))
joint_ids = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]

# Mapping from H36M joints to LSP joints (0:13). In this roder:
_COMMON_JOINT_IDS = np.array([
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


def plot_points(points_, img, points0_=None, radius=10):
    global colors
    tmp_img = img.copy()

    num_colors = len(colors)

    points = points_.T if points_.shape[0] == 2 else points_
    points0 = None if points0_ is None else (points0_.T
                                             if points0_.shape[0] == 2 else
                                             points0_)

    for i, coord in enumerate(points.astype(int)):
        if coord[0] < img.shape[1] and coord[1] < img.shape[0] and coord[0] > 0 and coord[1] > 0:
            cv2.circle(tmp_img,
                       tuple(coord), radius, colors[i % num_colors].tolist(),
                       -1)

    if points0 is not None:
        for i, coord in enumerate(points0.astype(int)):
            if coord[0] < img.shape[1] and coord[1] < img.shape[0] and coord[0] > 0 and coord[1] > 0:
                cv2.circle(tmp_img,
                           tuple(coord), radius + 2,
                           colors[i % num_colors].tolist(), 2)

    return tmp_img


def rotation_matrix(args):

    (x, y, z) = args

    X = np.vstack([[1, 0, 0], [0, np.cos(x), -np.sin(x)],
                   [0, np.sin(x), np.cos(x)]])
    Y = np.vstack([[np.cos(y), 0, np.sin(y)], [0, 1, 0],
                   [-np.sin(y), 0, np.cos(y)]])
    Z = np.vstack([[np.cos(z), -np.sin(z), 0], [np.sin(z),
                                                np.cos(z), 0], [0, 0, 1]])

    return (X.dot(Y)).dot(Z)


def project_point_radial(P, R, t, f, c, all_k):
    k = np.array(list(all_k[:2]) + list(all_k[-1:]))
    p = all_k[2:4]

    N = P.shape[0]

    X = R.dot(P.T - np.tile(t.reshape((-1, 1)), (1, len(P))))

    XX = X[:2, :] / np.tile(X[2, :], (2, 1))

    r2 = XX[0, :]**2 + XX[1, :]**2
    radial = 1 + np.sum(
        np.tile(k.reshape((-1, 1)), (1, N)) * np.vstack((r2, r2**2, r2**3)),
        axis=0)
    tan = p[0] * XX[1, :] + p[1] * XX[0, :]

    XXX = XX * np.tile(radial + tan, (2, 1)) + p[::-1].reshape(
        (-1, 1)).dot(r2.reshape((1, -1)))

    proj = (np.tile(f, (N, 1)) * XXX.T) + np.tile(c, (N, 1))
    return proj


def read_cam_parameters(xml_path, sbj_id, cam_id):
    import xml.etree.ElementTree

    # use the notation from 0 -- more practical to access array
    sbj_id = sbj_id - 1
    cam_id = cam_id - 1

    n_sbjs = 11
    n_cams = 4

    root = xml.etree.ElementTree.parse(xml_path).getroot()

    for child in root:
        if child.tag == 'w0':
            all_cameras = child.text
            tokens = all_cameras.split(' ')
            tokens[0] = tokens[0].replace('[', '')
            tokens[-1] = tokens[-1].replace(']', '')

            start = (cam_id * n_sbjs) * 6 + sbj_id * 6
            extrs = tokens[start:start + 6]

            start = (n_cams * n_sbjs * 6) + cam_id * 9
            intrs = tokens[start:start + 9]

            rot = rotation_matrix(np.array(extrs[:3], dtype=float))

            rt = rot
            t = np.array(extrs[3:], dtype=float)

            f = np.array(intrs[:2], dtype=float)
            c = np.array(intrs[2:4], dtype=float)

            distortion = np.array(intrs[4:], dtype=float)

            k = np.hstack((distortion[:2], distortion[3:5], distortion[2:3]))

            return (rt, t, f, c, k)


def read_action_name(xml_path, sbj_id, actionno, trialno):
    import xml.etree.ElementTree

    root = xml.etree.ElementTree.parse(xml_path).getroot()
    myactionno = actionno + 1  # otherwise we take ALL into account
    for child in root:
        if child.tag == 'mapping':
            for tr in child.getchildren():
                if tr.getchildren()[0].text == str(myactionno):
                    if tr.getchildren()[1].text == str(trialno):
                        return tr.getchildren()[2 + sbj_id - 1].text


def read_fua_results(path, sbj_id, trial_id, cam_id):
    from scipy.io import loadmat

    # sbj_id already follows the standard convention
    choose_id = sbj_id * (2 * 4) + (trial_id - 1) * 4 + (cam_id - 1)
    print(choose_id)
    res = loadmat(path)
    joints = res['Pred'].squeeze()
    return [j.reshape((-1, 3)) for j in joints[choose_id]]


def get_num_frames(path):
    vid = cv2.VideoCapture(path)
    return int(vid.get(cv2.CAP_PROP_FRAME_COUNT))


def read_frames(path, n_frames=None):
    vid = cv2.VideoCapture(path)

    imgs = []

    if n_frames is None:
        n_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    for _ in range(n_frames):
        success, img = vid.read()
        imgs.append(img)

    return imgs


def read_silhouettes(path, n_frames=None):
    import h5py
    f = h5py.File(path, 'r')
    refs = f['Masks']
    masks = []

    if n_frames is None:
        n_frames = len(refs)

    for i in range(n_frames):
        mask = np.array(f[refs[i, 0]], dtype=bool)
        mask = np.fliplr(np.rot90(mask, 3))
        masks.append(mask)
    return masks


def read_poses(path, n_frames=None, is_3d=False, joint_ids=range(32)):
    data = pycdf.CDF(path)

    # <CDF:
    # Pose: CDF_FLOAT [1, N, 64]
    # >
    poses = data['Pose'][...][0]

    if n_frames is None:
        n_frames = poses.shape[0]

    dim = 2 if not is_3d else 3
    packed_poses = [
        poses[i].reshape((-1, dim))[joint_ids] for i in range(n_frames)
    ]

    return packed_poses


def compute_fua_joints(joints, new_root_joint, in_meter=False):
    new_joints = np.zeros_like(joints)

    new_joints[0, :] = new_root_joint

    for i, offset in enumerate(joints[1:]):
        new_joints[i + 1] = new_root_joint + offset

    if in_meter:
        new_joints = np.vstack([j / 1000. for j in new_joints])

    return new_joints


def crop_image(silhs):
    res = np.asarray(silhs).any(axis=0)
    cnts, hier = cv2.findContours(
        np.uint8(res) * 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    """
    checks = []
    for cnt in cnts:
        kk = np.zeros((1000, 1000, 3), dtype=np.uint8)
        hull = cv2.convexHull(cnt)
        cv2.drawContours(kk, [cnt], 0, (255,255,255), -1)
        checks.append(kk)
    """

    max_id = 0
    max_length = len(cnts[0])
    for i in range(1, len(cnts)):
        if len(cnts[i]) > max_length:
            max_id = i
            max_length = len(cnts[i])

    (x, y, w, h) = cv2.boundingRect(cnts[max_id])
    return (x, y, w, h)


def crop_and_clean_mask_to_int(mask, x, y, w, h):
    mask = np.uint8(mask[y:y + h, x:x + w]) * 255

    # TODO: put this into a function (it's used above as well)
    cnts, hier = cv2.findContours(mask.copy(), cv2.RETR_LIST,
                                  cv2.CHAIN_APPROX_SIMPLE)

    max_id = 0
    max_length = len(cnts[0])
    for i in range(1, len(cnts)):
        if len(cnts[i]) > max_length:
            max_id = i
            max_length = len(cnts[i])

    tmp_mask = np.dstack((mask, mask, mask))
    for i, cnt in enumerate(cnts):
        if i != max_id:
            cv2.drawContours(tmp_mask, [cnt], 0, (0, 0, 0), -1)
    return cv2.split(tmp_mask)[0]


def main(raw_data_root, output_root, frame_skip):
    xml_path = join(raw_data_root, 'metadata.xml')

    # <actionno> for each different action class:
    # 1) Directions, 2) Discussion, 3) Eating, 4) Greeting,
    # 5) Phone Talk, 6) Posing, 7) Buying, 8) Sitting,
    # 9) Sitting Down, 10) Smoking, 11) Taking Photo, 12) Waiting,
    # 13) Walking, 14) Walking Dog, 15) Walking Pair
    action_names = [
        'Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Posing',
        'Purchases', 'Sitting', 'SittingDown', 'Smoking', 'TakingPhoto',
        'Waiting', 'Walking', 'WakingDog', 'WalkTogether'
    ]

    n_frames = None

    sub_ids = [1, 6, 7, 8, 5, 9, 11]

    # Action, camera, suject id starts from 1 Matlab convention
    cam_ids = range(1, 5)
    trial_ids = [1, 2]
    action_ids = range(1, 16)
    import itertools
    all_pairs = [
        p
        for p in list(
            itertools.product(*[sub_ids, action_ids, trial_ids, cam_ids]))
    ]

    def has_num(str):
        return any(i.isdigit() for i in str)

    for (sbj_id, action_id, trial_id, cam_id) in all_pairs:
        seq_name = read_action_name(xml_path, sbj_id, action_id, trial_id)

        save_seq_name = '%s_%d' % (action_names[action_id - 1], trial_id - 1)

        output_base = join(output_root, 'S%s' % str(sbj_id), save_seq_name)
        output_dir = join(output_base, 'cam_%01d' % (cam_id - 1))

        print('Sub: {}, action {},  trial {}, cam {}'.format(
            sbj_id, action_id, trial_id, cam_id))
        print('Orig seq_name %s, new_seq_name %s' % (seq_name, save_seq_name))
        print('Saving to {}'.format(output_dir))

        if sbj_id == 11 and 'Phoning 2' in seq_name:
            print('Skipping.. {}'.format(output_dir))
            continue

        if not exists(output_dir):
            makedirs(output_dir)

        # Save orig name
        name_path = join(output_base, 'orig_seq_name.txt')
        if not exists(name_path):
            with open(name_path, "w") as f:
                f.write(seq_name)

        video_paths = sorted(
            glob(
                join(raw_data_root, 'S%s' % str(sbj_id), 'Videos',
                     '%s.*mp4' % seq_name)))
        pose2d_paths = sorted(
            glob(
                join(raw_data_root, 'S%s' % str(sbj_id),
                     'MyPoseFeatures/D2_Positions', '%s.*cdf' % seq_name)))
        pose3d_paths = sorted(
            glob(
                join(raw_data_root, 'S%s' % str(sbj_id),
                     'MyPoseFeatures/D3_Positions_mono',
                     '%s.*cdf' % seq_name)))

        (rot, t, flen, c, k) = read_cam_parameters(xml_path, sbj_id, cam_id)
        cam_path = join(output_dir, 'camera_wext.pkl')
        print('Writing %s' % cam_path)
        if not exists(cam_path):
            with open(cam_path, 'wb') as fw:
                pickle.dump({'f': flen, 'c': c, 'k': k, 'rt': rot, 't': t}, fw)

        # AJ: frames
        poses2d = read_poses(pose2d_paths[cam_id - 1], joint_ids=joint_ids)
        poses3d = read_poses(
            pose3d_paths[cam_id - 1], is_3d=True, joint_ids=joint_ids)

        # Check if we're done here.
        want_length = len(poses2d[::frame_skip])
        written_images = glob(join(output_dir, '*.png'))
        num_imgs_written = len(written_images)
        if want_length == num_imgs_written:
            is_done = True
            for fname in written_images:
                if getsize(fname) == 0:
                    is_done = False
                    break
            if is_done:
                print('Done!')
                continue

        # Write images..
        print('reading images...')
        imgs = read_frames(video_paths[cam_id - 1], n_frames=n_frames)
        # For some reason, len(poses2d) < len(imgs) by few frames sometimes
        # len(poses2d) == len(poses3d) always.
        # clip the images according to them..
        imgs = imgs[:len(poses2d)]

        # Subsample
        imgs = imgs[::frame_skip]
        poses2d = poses2d[::frame_skip]
        poses3d = poses3d[::frame_skip]
        gt_path = join(output_dir, 'gt_poses.pkl')
        if not exists(gt_path):
            with open(gt_path, 'wb') as fgt:
                pickle.dump({'2d': poses2d, '3d': poses3d}, fgt)

        # Link the mp4 with the new name to the old name.
        # video_paths[cam_id - 1]
        action_name = action_names[action_id - 1]
        out_video_name = 'S{}_{}_{}_cam_{}.mp4'.format(
            sbj_id, action_name, trial_id - 1, cam_id - 1)
        out_video_path = join(output_dir, out_video_name)
        if not exists(out_video_path):
            orig_vid_path = video_paths[cam_id - 1].replace(" ", "\ ")
            cmd = 'ln -s {} {}'.format(orig_vid_path, out_video_path)
            ret = system(cmd)
            if ret > 0:
                print('something went wrong!')
                import ipdb
                ipdb.set_trace()

        for i, (img) in enumerate(imgs):
            if exists(join(output_dir, 'frame%04d.png' % i)):
                if getsize(join(output_dir, 'frame%04d.png' % i)) > 0:
                    continue
            if i % 50 == 0:
                import matplotlib.pyplot as plt
                plt.ion()
                plt.imshow(plot_points(poses2d[i], img)[:, :, ::-1])
                plt.draw()
                plt.pause(1e-3)
            cv2.imwrite(join(output_dir, 'frame%04d.png' % i), img)
            if i % 50 == 0:
                print(join(output_dir, 'frame%04d.png' % i))


if __name__ == '__main__':
    FLAGS(sys.argv)
    frame_skip = FLAGS.frame_skip
    raw_data_root = FLAGS.source_dir
    output_root = FLAGS.out_dir

    main(raw_data_root, output_root, frame_skip)
    
