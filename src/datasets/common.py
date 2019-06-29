"""
Utilities for reading keypoint data from various dataset formats.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class ImageCoder(object):
    """
    Helper class that provides TensorFlow image coding utilities.
    Taken from
    https://github.com/tensorflow/models/blob/master/inception/inception/data/
        build_image_data.py
    """

    def __init__(self):
        # Create a single Session to run all image coding calls.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.05)
        self._sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        # Initializes function that converts PNG to JPEG data.
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(
            image, format='rgb', quality=100)

        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(
            self._decode_jpeg_data, channels=3)

        self._encode_jpeg_data = tf.placeholder(dtype=tf.uint8)
        self._encode_jpeg = tf.image.encode_jpeg(
            self._encode_jpeg_data, format='rgb')

        self._decode_png_data = tf.placeholder(dtype=tf.string)
        self._decode_png = tf.image.decode_png(
            self._decode_png_data, channels=3)

        self._encode_png_data = tf.placeholder(dtype=tf.uint8)
        self._encode_png = tf.image.encode_png(self._encode_png_data)

    def png_to_jpeg(self, image_data):
        return self._sess.run(
            self._png_to_jpeg, feed_dict={
                self._png_data: image_data
            })

    def decode_jpeg(self, image_data):
        image = self._sess.run(
            self._decode_jpeg, feed_dict={
                self._decode_jpeg_data: image_data
            })
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image

    def encode_jpeg(self, image):
        image_data = self._sess.run(
            self._encode_jpeg, feed_dict={
                self._encode_jpeg_data: image
            })
        return image_data

    def encode_png(self, image):
        image_data = self._sess.run(
            self._encode_png, feed_dict={
                self._encode_png_data: image
            })
        return image_data

    def decode_png(self, image_data):
        image = self._sess.run(
            self._decode_png, feed_dict={
                self._decode_png_data: image_data
            })
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def read_from_example(serialized_ex):
    """
    Returns data from an entry in test tfrecord.

    Args:
        serialized_ex (str).

    Returns:
        dict. Keys:
            N (1).
            centers (Nx2).
            kps (Nx19x3).
            gt3ds (Nx14x3).
            images (Nx224x224x3).
            im_shapes (Nx2).
            im_paths (N).
            poses (Nx24x3).
            scales (N).
            shape (10).
            start_pts (Nx2).
            time_pts (2).
    """
    coder = ImageCoder()
    example = tf.train.Example()
    example.ParseFromString(serialized_ex)
    features = example.features.feature

    # Load features from example.
    N = features['meta/N'].int64_list.value[0]
    im_datas = features['image/encoded'].bytes_list.value
    centers = features['image/centers'].int64_list.value
    xys = features['image/xys'].float_list.value
    face_pts = features['image/face_pts'].float_list.value
    toe_pts = features['image/toe_pts'].float_list.value
    vis = features['image/visibilities'].int64_list.value
    scales = np.array(features['image/scale_factors'].float_list.value)
    gt3ds = features['mosh/gt3ds'].float_list.value
    poses = features['mosh/poses'].float_list.value
    shape = features['mosh/shape'].float_list.value
    time_pts = features['meta/time_pts'].int64_list.value
    start_pts = np.array(features['image/crop_pts'].int64_list.value)
    im_shapes = features['image/heightwidths'].int64_list.value
    im_paths = features['image/filenames'].bytes_list.value

    # Process and reshape features.
    images = [coder.decode_jpeg(im_data) for im_data in im_datas]
    centers = np.array(centers).reshape((N, 2))
    gt3ds = np.array(gt3ds).reshape((N, -1, 3))
    gt3ds = gt3ds[:, :14]  # Don't want toes_pts or face_pts
    xys = np.array(xys).reshape((N, 2, 14))
    vis = np.array(vis, dtype=np.float).reshape((N, 1, 14))
    face_pts = np.array(face_pts).reshape((N, 3, 5))
    toe_pts = np.array(toe_pts).reshape((N, 3, 6))
    kps = np.dstack((
        np.hstack((xys, vis)),
        face_pts,
        toe_pts,
    ))
    kps = np.transpose(kps, axes=[0, 2, 1])
    poses = np.array(poses).reshape((N, 24, 3))
    shape = np.array(shape)
    start_pts = np.array(start_pts).reshape((N, 2))
    im_shapes = np.array(im_shapes).reshape((N, 2))

    return {
        'N': N,
        'centers': centers,
        'kps': kps,
        'gt3ds': gt3ds,
        'images': images,
        'im_shapes': im_shapes,
        'im_paths': im_paths,
        'poses': poses,
        'scales': scales,
        'shape': shape,
        'start_pts': start_pts,
        'time_pts': time_pts,
    }


def int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list) and not isinstance(value, np.ndarray):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list) and not isinstance(value, np.ndarray):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    if not isinstance(value, list) and not isinstance(value, np.ndarray):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def convert_to_example_temporal(
        image_datas,
        image_paths,
        image_shapes,
        labels,
        centers,
        gt3ds,
        scale_factors,
        start_pts,
        cams,
        poses=None,
        shape=None,
        phis=None,
        image_datas_og=None,
        time_pts=None):
    """
    Builds an Example proto for an image temporal example for N images.
    Note, no paired mosh data is available, so all poses and shape are None
    despite this function name.

    Args:
        image_datas (list of N str): JPEG encoding of RGB Images.
        image_paths (list of N str): Paths to image files.
        image_shapes (Nx2): Height and width.
        labels (Nx3x14): (x, y, visibility) for each joint.
           if N x 3 x 19, split into 14 and 5 face_pts
        centers (Nx2x1): Center of the tight bbox.
        gt3ds (Nx14x3): 3D Joint locations.
        poses (Nx24*3): Pose parameters. ALWAYS NONE
        shape (10): Shape parameters. ALWAYS NONE
        scale_factors (Nx2x1): Scale factor for each image.
        start_pts (Nx2): Starting points for each image.
        cams (Nx3): [f, px, py] intrinsic camera parameters.
        phis (Nx2048): Image features (optional).
        image_datas_og (list of N str): JPEG encoding of full frame images.
        time_pts (2): Time in sequence that the tube begins and ends.

    If the dataset has no 3D labels, gt3ds/cams are None
    so are poses and shapes

    Returns:
        Example proto.
    """
    N = len(labels)
    labels = np.array(labels)

    face_pts = None
    toe_pts = None
    if labels.shape[2] == 19:
        face_pts = labels[:, :,  -5:]
        labels = labels[:, :, :-5]
    elif labels.shape[2] == 25:
        toe_pts = labels[:, :,  -6:]
        face_pts = labels[:, :,  -11:-6]
        labels = labels[:, :, :-11]

    if poses is None:
        has_3d = 0
        # Use -1 to save.
        poses = -np.ones((N, 72))
        shape = -np.ones(10)
    else:
        poses = np.array(poses)
        has_3d = 1
    # This is always on ftm. Maybe useful later.
    if gt3ds is None:
        has_3d_joints = 0
        gt3ds = np.zeros((N, 14, 3))
        cams = np.zeros((N, 3))
    else:
        gt3ds = np.array(gt3ds)
        has_3d_joints = 1

    feat_dict = {
        # Features for all images.
        'mosh/shape': float_feature(np.array(shape).astype(np.float).ravel()),
        'meta/has_3d': int64_feature(has_3d),
        'meta/has_3d_joints': int64_feature(has_3d_joints),
        'meta/N': int64_feature(N),
        'image/filenames': bytes_feature([tf.compat.as_bytes(path)
                                          for path in image_paths]),
        'image/heightwidths':
            int64_feature(np.array(image_shapes).ravel()),
        'image/xys':
            float_feature(labels[:, 0:2].astype(np.float).ravel()),
        'image/visibilities':
            int64_feature(labels[:, 2].astype(np.int).ravel()),
        'image/centers':
            int64_feature(np.array(centers, dtype=np.int).ravel()),
        'mosh/gt3ds':
            float_feature(np.array(gt3ds, dtype=np.float).ravel()),
        'mosh/poses':
            float_feature(np.array(poses, dtype=np.float).ravel()),
        'image/scale_factors':
            float_feature(np.array(scale_factors, dtype=np.float).ravel()),
        'image/crop_pts':
            int64_feature(np.array(start_pts, dtype=np.int).ravel()),
        'image/cams':
            float_feature(np.array(cams, dtype=np.float).ravel()),
    }

    if image_datas is not None:
        # Features for each image.
        feat_dict['image/encoded'] = bytes_feature(
            [tf.compat.as_bytes(image_data) for image_data in image_datas])
    if face_pts is not None:
        feat_dict['image/face_pts'] = float_feature(
            np.array(face_pts, dtype=np.float).ravel())

    if toe_pts is not None:
        feat_dict['image/toe_pts'] = float_feature(
            np.array(toe_pts, dtype=np.float).ravel())

    if phis is not None:
        feat_dict['image/phis'] = float_feature(phis.ravel())

    if image_datas_og is not None:
        feat_dict['image/encoded_og'] = bytes_feature(
            [tf.compat.as_bytes(image_data) for image_data in image_datas_og])

    if time_pts is not None:
        feat_dict['meta/time_pts'] = int64_feature(np.array(time_pts))

    example = tf.train.Example(features=tf.train.Features(feature=feat_dict))
    return example
