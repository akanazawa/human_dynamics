"""
Utilities for mak
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from src.datasets.common import (
    convert_to_example_temporal,
    ImageCoder,
)
from src.util.common import resize_img
from src.util.render.render_utils import draw_skeleton
from src.util.smooth_bbox import get_smooth_bbox_params


def save_seq_to_test_tfrecord(
        out_name,
        im_paths,
        all_gt2ds,
        all_gt3ds=None,
        all_poses=None,
        all_shapes=None,
        visualize=False,
        vis_thresh=0.1,
        img_size=224,
        sigma=3,
        separate_tubes=False):
    """
    Saves a sequence to test format for rendering and evaluation.

    Args:
        out_name (str): Tfrecord filename.
        im_paths (list of length N): List of image file paths.
        all_gt3ds (PxNx14x3): All 3D joint locations.
        all_gt2ds (PxNx19x3): All keypoints.
        all_poses (PxNx72): All poses.
        all_shapes (Px10): All shapes.
        visualize (bool): If True, visualizes skeletons and smoothing.
        vis_thresh (float): Threshold for keypoint visibility.
        img_size (int): Image size.

    Returns:

    """
    # Number of people.
    P = len(all_gt2ds)

    if all_gt3ds is None:
        all_gt3ds = [None] * P
    if all_poses is None:
        all_poses = [None] * P
    if all_shapes is None:
        all_shapes = [None] * P

    coder = ImageCoder()
    print('Starting tfrecord file: {}'.format(out_name))
    with tf.python_io.TFRecordWriter(out_name) as writer:
        for i in range(P):
            if separate_tubes:
                paths = im_paths[i]
            else:
                paths = im_paths
            add_to_tfrecord(
                image_paths=paths,
                gt2ds=all_gt2ds[i],
                gt3ds=all_gt3ds[i],
                poses=all_poses[i],
                shape=all_shapes[i],
                coder=coder,
                writer=writer,
                visualize=visualize,
                vis_thresh=vis_thresh,
                img_size=img_size,
                sigma=sigma,
            )


def add_to_tfrecord(image_paths,
                    gt2ds,
                    gt3ds,
                    poses,
                    shape,
                    coder,
                    writer,
                    visualize=False,
                    vis_thresh=0.1,
                    img_size=224,
                    sigma=8):
    """

    Args:
        image_paths (N).
        gt2ds (Nx19x3).
        gt3ds (Nx14x3).
        poses (Nx72).
        shape (10).
        coder (ImageCoder).
        writer (TFRecordWriter).
        visualize (bool).
        vis_thresh (float).
        img_size (int).
    """
    results = {
        'image_data_scaled': [],
        'im_path': [],
        'im_shape': [],
        'kps': [],
        'center': [],
        'scale': [],
        'start_pt': [],
    }

    bbox_params, time_pt1, time_pt2 = get_smooth_bbox_params(
        gt2ds,
        vis_thresh,
        sigma=sigma,
    )
    if time_pt1 != 0 or time_pt2 != len(image_paths):
        print('Start: {}, End: {}'.format(time_pt1, time_pt2))
    for im_path, kps, bbox in tqdm(list(zip(
            image_paths, gt2ds, bbox_params))[time_pt1:time_pt2]):
        ret_dict = process_image(
            im_path=im_path,
            gt2d=kps,
            coder=coder,
            bbox_param=bbox,
            visualize=visualize,
            vis_thresh=vis_thresh,
            img_size=img_size
        )

        for key in ret_dict:
            results[key].append(ret_dict[key])

    # Adjust to start and end time if they exist.        
    if gt3ds is not None:
        gt3ds = gt3ds[time_pt1:time_pt2]
    if poses is not None:
        poses = poses[time_pt1:time_pt2]        

    example = convert_to_example_temporal(
        cams=[],
        centers=results['center'],                   # N x 2
        gt3ds=gt3ds,
        image_datas=results['image_data_scaled'],    # N
        image_paths=results['im_path'],              # N
        image_shapes=results['im_shape'],            # N x 2
        labels=results['kps'],                       # N x 3 x 19
        scale_factors=results['scale'],              # N
        start_pts=results['start_pt'],               # N x 2
        time_pts=(time_pt1, time_pt2),               # 2
        poses=poses,                                 # N x 72
        shape=shape,                                 # 10
    )
    writer.write(example.SerializeToString())


def process_image(im_path, gt2d, coder, bbox_param, visualize=False,
                  vis_thresh=0.1, img_size=224):
    """
    Processes an image, producing 224x224 crop.

    Args:
        im_path (str).
        gt2d (19x3).
        coder (tf.ImageCoder).
        bbox_param (3,): [cx, cy, scale].
        visualize (bool).
        vis_thresh (float).
        img_size (int).

    Returns:
        dict: image_data_scaled, im_path, im_shape, kps, center,
            scale, start_pt.
    """
    with tf.gfile.FastGFile(im_path, 'rb') as f:
        image_data = f.read()
        image = coder.decode_jpeg(image_data)
        assert image.shape[2] == 3

    center = bbox_param[:2]
    scale = bbox_param[2]

    image_scaled, scale_factors = resize_img(image, scale)
    vis = gt2d[:, 2] > vis_thresh
    joints_scaled = np.copy(gt2d[:, :2])
    joints_scaled[:, 0] *= scale_factors[0]
    joints_scaled[:, 1] *= scale_factors[1]
    center_scaled = np.round(center * scale_factors).astype(np.int)

    # Make sure there is enough space to crop 224x224.
    image_padded = np.pad(
        array=image_scaled,
        pad_width=((img_size,), (img_size,), (0,)),
        mode='edge'
    )
    height, width = image_padded.shape[:2]
    center_scaled += img_size
    joints_scaled += img_size

    # Crop 224x224 around the center.
    margin = img_size // 2

    start_pt = (center_scaled - margin).astype(int)
    end_pt = (center_scaled + margin).astype(int)
    end_pt[0] = min(end_pt[0], width)
    end_pt[1] = min(end_pt[1], height)
    image_scaled = image_padded[start_pt[1]:end_pt[1],
                                start_pt[0]:end_pt[0], :]
    # Update others too.
    joints_scaled[:, 0] -= start_pt[0]
    joints_scaled[:, 1] -= start_pt[1]
    center_scaled -= start_pt
    height, width = image_scaled.shape[:2]
    im_shape = [height, width]

    if visualize:
        if gt2d is None:
            image_with_skel = image
            image_with_skel_scaled = image_scaled
        else:
            image_with_skel = draw_skeleton(image, gt2d[:, :2], vis=vis)
            image_with_skel_scaled = draw_skeleton(
                image_scaled, joints_scaled[:, :2], vis=vis)

        plt.ion()
        plt.clf()
        fig = plt.figure(1)
        ax = fig.add_subplot(121)
        ax.imshow(image_with_skel)
        ax.axis('off')
        ax.scatter(center[0], center[1], color='red')
        ax = fig.add_subplot(122)

        ax.imshow(image_with_skel_scaled)
        ax.scatter(center_scaled[0], center_scaled[1], color='red')

        plt.show()
        plt.draw()
        plt.pause(5e-6)

    kps = np.vstack((joints_scaled.T, [vis]))

    return {
        'image_data_scaled': coder.encode_jpeg(image_scaled),
        'im_path': im_path,
        'im_shape': im_shape,
        'kps': kps,
        'center': center_scaled,
        'scale': scale,
        'start_pt': start_pt,
    }
