"""
Utils for data loading for training.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from glob import glob
from os.path import (
    basename,
    dirname,
    join,
)

import tensorflow as tf


def parse_example_proto(example_serialized, has_3d=False):
    """Parses an Example proto.
    Its contents are:

        'image/height'       : _int64_feature(height),
        'image/width'        : _int64_feature(width),
        'image/x'            : _float_feature(label[0,:].astype(np.float)),
        'image/y'            : _float_feature(label[1,:].astype(np.float)),
        'image/visibility'   : _int64_feature(label[2,:].astype(np.int)),
        'image/format'       : _bytes_feature
        'image/filename'     : _bytes_feature
        'image/encoded'      : _bytes_feature
        'image/face_pts'  : _float_feature,
            this is the 2D keypoints of the face points in coco
            5*3 (x,y,vis) = 15
        'image/toe_pts'   : _float_feature,
            this is the 2D keypoints of the toe points from openpose
             6*3 (x,y,vis) = 18

    if has_3d is on, it also has:
        'mosh/pose'          : float_feature(pose.astype(np.float)),
        'mosh/shape'         : float_feature(shape.astype(np.float)),
        # gt3d is 14x3
        'mosh/gt3d'          : float_feature(gt3d.astype(np.float)),
    """
    feature_map = {
        'image/encoded':
        tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        'image/height':
        tf.FixedLenFeature([1], dtype=tf.int64, default_value=-1),
        'image/width':
        tf.FixedLenFeature([1], dtype=tf.int64, default_value=-1),
        'image/filename':
        tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        'image/center':
        tf.FixedLenFeature((2, 1), dtype=tf.int64),
        'image/visibility':
        tf.FixedLenFeature((1, 14), dtype=tf.int64),
        'image/x':
        tf.FixedLenFeature((1, 14), dtype=tf.float32),
        'image/y':
        tf.FixedLenFeature((1, 14), dtype=tf.float32),
        'image/face_pts':
        tf.FixedLenFeature(
            (1, 15),
            dtype=tf.float32,
            default_value=([0.] * 15)),
        'image/toe_pts':
        tf.FixedLenFeature(
            (1, 18),
            dtype=tf.float32,
            default_value=([0.] * 18)),
    }
    if has_3d:
        feature_map.update({
            'mosh/pose':
            tf.FixedLenFeature((72, ), dtype=tf.float32),
            'mosh/shape':
            tf.FixedLenFeature((10, ), dtype=tf.float32),
            'mosh/gt3d':
            tf.FixedLenFeature((14 * 3, ), dtype=tf.float32),
            # has_3d is for pose and shape: 0 for mpi_inf_3dhp, 1 for h3.6m.
            'meta/has_3d':
            tf.FixedLenFeature(1, dtype=tf.int64, default_value=[0]),
        })

    features = tf.parse_single_example(example_serialized, feature_map)

    height = tf.cast(features['image/height'], dtype=tf.int32)
    width = tf.cast(features['image/width'], dtype=tf.int32)
    center = tf.cast(features['image/center'], dtype=tf.int32)
    fname = tf.cast(features['image/filename'], dtype=tf.string)
    fname = tf.Print(fname, [fname], message='image name: ')

    face_pts = tf.reshape(
        tf.cast(features['image/face_pts'], dtype=tf.float32), [3, 5])
    toe_pts = tf.reshape(
        tf.cast(features['image/toe_pts'], dtype=tf.float32), [3, 6])

    vis = tf.cast(features['image/visibility'], dtype=tf.float32)
    x = tf.cast(features['image/x'], dtype=tf.float32)
    y = tf.cast(features['image/y'], dtype=tf.float32)

    label = tf.concat([x, y, vis], 0)
    label = tf.concat([label, face_pts, toe_pts], 1)

    image = decode_jpeg(features['image/encoded'])
    image_size = tf.concat([height, width], 0)

    if has_3d:
        pose = tf.cast(features['mosh/pose'], dtype=tf.float32)
        shape = tf.cast(features['mosh/shape'], dtype=tf.float32)
        gt3d = tf.reshape(
            tf.cast(features['mosh/gt3d'], dtype=tf.float32), [14, 3])
        has_smpl3d = tf.cast(features['meta/has_3d'], dtype=tf.bool)
        return (image, image_size, label, center, fname, pose, shape, gt3d,
                has_smpl3d)
    else:
        return image, image_size, label, center, fname


def parse_example_proto_temporal(example_serialized,
                                 T=None,
                                 precomputed_phi=False):
    """
    Parses an Example proto.

    Its contents are:

        'meta/N'             : _int64_feature(N),
        'image/heightwidths' : _int64_feature(image_shapes),
        'image/centers'      : _int64_feature(centers),
        'image/xys'          : _float_feature(labels[:, 0:2].astype(np.float)),
        'image/visibilities' : _int64_feature(label[:, 2].astype(np.int)),
        'image/filenames'    : _bytes_feature,
        'image/encoded'      : _bytes_feature,
        'image/face_pts'     : _float_feature
         this is the 2D keypoints of the face points in coco 5*3 (x,y,vis) = 15

    if has_3d is on, it also has:
        'mosh/poses'         : float_feature(poses.astype(np.float)),
        'mosh/shape'         : float_feature(shape.astype(np.float)),
        # gt3d is 14x3
        'mosh/gt3ds'         : float_feature(gt3ds.astype(np.float)),

    Args:
        example_serialized:
        T (int): Number of frames per sequence for subsampling.
                 If None, will return all frames.
        precomputed_phi (bool): If True, uses precomputed phi instead of image.

    Returns:
        dict:
            images/phis (TxHxWx3)/(Tx2048).
            image_sizes (Tx2).
            labels (Tx3x19).
            centers (Tx2).
            fnames (T,).
            poses (Tx72).
            shape (10,).
            gt3ds (Tx14x3).
            has_3d (2,).
    """
    feature_map = {
        'meta/N':
            tf.FixedLenFeature((), dtype=tf.int64),
        'image/heightwidths':
            tf.VarLenFeature(dtype=tf.int64),
        'image/filenames':
            tf.VarLenFeature(dtype=tf.string),
        'image/centers':
            tf.VarLenFeature(dtype=tf.int64),
        'image/visibilities':
            tf.VarLenFeature(dtype=tf.int64),
        'image/xys':
            tf.VarLenFeature(dtype=tf.float32),
        'image/face_pts':
            tf.VarLenFeature(dtype=tf.float32),
        'image/toe_pts':
            tf.VarLenFeature(dtype=tf.float32),
        # has_3d is for pose and shape: 0 for mpi_inf_3dhp, 1 for h3.6m.
        'meta/has_3d':
            tf.FixedLenFeature(1, dtype=tf.int64, default_value=0),
        'meta/has_3d_joints':
            tf.FixedLenFeature(1, dtype=tf.int64, default_value=0),
        'mosh/shape':
            tf.FixedLenFeature((10,), dtype=tf.float32),
        'mosh/poses':
            tf.VarLenFeature(dtype=tf.float32),
        'mosh/gt3ds':
            tf.VarLenFeature(dtype=tf.float32),
    }
    if precomputed_phi:
        feature_map['image/phis'] = tf.VarLenFeature(dtype=tf.float32)
        feature_map['image/encoded'] = tf.VarLenFeature(dtype=tf.string)
    else:
        feature_map['image/encoded'] = tf.VarLenFeature(dtype=tf.string)

    features = tf.parse_single_example(example_serialized, feature_map)

    N = tf.cast(features['meta/N'], dtype=tf.int32)

    if T is not None:
        indices = pick_sequences(N, T)
    else:
        indices = tf.range(0, N)
        if not T:
            T = N

    if precomputed_phi:
        phis = process_tensors(
            data=features['image/phis'],
            N=N,
            indices=indices,
            dtype=tf.float32,
            default=0,
            shape=(T, 2048),
            name='process_tensors_phis',
        )
        ret_dict = {'phis': phis}

        image_datas = process_tensors(
            data=features['image/encoded'],
            N=N,
            indices=indices,
            dtype=tf.string,
            default='',
            shape=(T,),
            name='process_tensors_image',
        )
        images = tf.map_fn(decode_jpeg, image_datas, dtype=tf.float32)
        # AJ: shouldn't use const shape,,:
        images.set_shape((T, 224, 224, 3))
        ret_dict.update({'images': images})
    else:
        image_datas = process_tensors(
            data=features['image/encoded'],
            N=N,
            indices=indices,
            dtype=tf.string,
            default='',
            shape=(T,)
        )
        images = tf.map_fn(decode_jpeg, image_datas, dtype=tf.float32)
        images.set_shape((T, 300, 300, 3))
        ret_dict = {'images': images}

    image_sizes = process_tensors(
        data=features['image/heightwidths'],
        N=N,
        indices=indices,
        dtype=tf.int32,
        default=-1,
        shape=(T, 2),
        name='process_tensors_hw',
    )

    fnames = ''
    centers = process_tensors(
        data=features['image/centers'],
        N=N,
        indices=indices,
        dtype=tf.int32,
        shape=(T, 2),
        name='process_tensors_centers',
    )
    visibilities = process_tensors(
        data=features['image/visibilities'],
        N=N,
        indices=indices,
        dtype=tf.float32,
        shape=(T, 1, 14),
        name='process_tensors_vis',
    )
    xys = process_tensors(
        data=features['image/xys'],
        N=N,
        indices=indices,
        dtype=tf.float32,
        shape=(T, 2, 14),
        name='process_tensors_xy',
    )
    face_pts = process_tensors(
        data=features['image/face_pts'],
        N=N,
        indices=indices,
        dtype=tf.float32,
        default=0,
        shape=(T, 3, 5),
        name='process_tensors_face',
    )

    toe_pts = process_tensors(
        data=features['image/toe_pts'],
        N=N,
        indices=indices,
        dtype=tf.float32,
        default=0,
        shape=(T, 3, 6),
        name='process_tensors_toes',
     )

    labels = tf.concat([xys, visibilities], 1)
    labels = tf.concat([labels, face_pts, toe_pts], 2)

    has_smpl3d = tf.cast(features['meta/has_3d'], dtype=tf.bool)
    has_3d_joints = tf.cast(features['meta/has_3d_joints'], dtype=tf.bool)
    has_3d = tf.concat((has_3d_joints, has_smpl3d), axis=0)

    shape = tf.cast(features['mosh/shape'], dtype=tf.float32)
    poses = process_tensors(
        data=features['mosh/poses'],
        N=N,
        indices=indices,
        dtype=tf.float32,
        default=0,
        shape=(T, 72),
        name='process_tensors_poses',
    )
    gt3ds = process_tensors(
        data=features['mosh/gt3ds'],
        N=N,
        indices=indices,
        dtype=tf.float32,
        default=0,
        shape=(T, 14, 3),
        name='process_tensors_gt3ds',
    )

    ret_dict.update({
        'image_sizes': image_sizes,
        'labels': labels,
        'centers': centers,
        'fnames': fnames,
        'poses': poses,
        'shape': shape,
        'gt3ds': gt3ds,
        'has_3d': has_3d
    })
    return ret_dict


def pick_sequences(N, T):
    """
    Returns random subset of length T.

    Args:
        N: Total number of samples.
        T: Desired sequence length.

    Returns:
        Tensor: Random sequence subset.
    """
    start = tf.random_uniform(
        shape=(),
        minval=0,
        maxval=(N - T + 1),
        dtype=tf.int32
    )
    indices = tf.reshape(tf.range(start, start + T, 1), (1, T, 1))
    return indices


def subsample(tensor, N, indices):
    """
    Subsamples tensor, keeping frames corresponding to indices.
    """
    T = tf.size(indices)
    tensor_reshaped = tf.reshape(tensor, (N, -1))
    return tf.reshape(tf.gather_nd(tensor_reshaped, indices), (T, -1))


def rescale_image(image):
    """
    Rescales image from [0, 1] to [-1, 1]
    Resnet v2 style preprocessing.
    """
    # convert to [0, 1].
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image


def get_all_files(dataset_dir, datasets, sequences=(), split='train'):
    """
    """

    datasets = datasets[:]
    # Some datasets have a different path name
    if 'h36m' in datasets:
        datasets.append('human36m')
    if 'mpi_inf_3dh' in datasets:
        datasets.append('mpi_inf_3dhp')

    postfix = '.tfrecord'

    data_dirs = []
    for dataset in datasets:
        if sequences:
            data_dirs += [join(dataset_dir, dataset, split,
                               '*{}_[0-9]{}'.format(sequence, postfix))
                          for sequence in sequences]
        else:
            data_dirs.append(join(
                dataset_dir, dataset, split, '*{}'.format(postfix)))

    all_files = []
    for data_dir in data_dirs:
        files = sorted(glob(data_dir))
        if files:
            # Print out dataset so we know it was loaded properly.
            dataset = basename(dirname(dirname(data_dir)))
            print('Reading dataset:', dataset)
            all_files += files

    return all_files


def read_smpl_data(filename_queue):
    """
    Parses a smpl Example proto.
    It's contents are:
        'pose'  : 72-D float
        'shape' : 10-D float
    """
    with tf.name_scope(None, 'read_smpl_data', [filename_queue]):
        reader = tf.TFRecordReader()
        _, example_serialized = reader.read(filename_queue)

        feature_map = {
            'pose': tf.FixedLenFeature((72, ), dtype=tf.float32),
            'shape': tf.FixedLenFeature((10, ), dtype=tf.float32)
        }

        features = tf.parse_single_example(example_serialized, feature_map)
        pose = tf.cast(features['pose'], dtype=tf.float32)
        shape = tf.cast(features['shape'], dtype=tf.float32)

        return pose, shape


def read_smpl_data_temporal(filename_queue):
    """
    Parses smpl temporal Example proto.
    """
    with tf.name_scope(None, 'read_smpl_data_temporal', [filename_queue]):
        reader = tf.TFRecordReader()
        _, example_serialized = reader.read(filename_queue)

        feature_map = {
            'delta_pose': tf.FixedLenFeature((72, ), dtype=tf.float32),
            'pose': tf.FixedLenFeature((72, ), dtype=tf.float32),
        }

        features = tf.parse_single_example(example_serialized, feature_map)
        pose = tf.cast(features['pose'], dtype=tf.float32)
        delta_pose = tf.cast(features['delta_pose'], dtype=tf.float32)
        return pose, delta_pose


def decode_jpeg(image_buffer, name=None):
    """Decode a JPEG string into one 3-D float image Tensor.
      Args:
        image_buffer: scalar string Tensor.
        name: Optional name for name_scope.
      Returns:
        3-D float Tensor with values ranging from [0, 1).
    """
    with tf.name_scope(name, 'decode_jpeg', [image_buffer]):
        # Decode the string as an RGB JPEG.
        # Note that the resulting image contains an unknown height and width
        # that is set dynamically by decode_jpeg. In other words, the height
        # and width of image is unknown at compile-time.
        image = tf.image.decode_jpeg(image_buffer, channels=3)

        # convert to [0, 1].
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image


def process_tensors(data, N, indices, shape, dtype, default=0, name=None):
    """
    Wrapper function for processing sparse tensors outputted by variable length
    features.

    1. Converts sparse tensors to dense tensors.
    2. Reshapes and fills in default value.
    3. Subsamples tensor based on indices.

    Args:
        data (SparseTensor): Sparse Tensor to convert.
        N (scalar Tensor): Total number of frames.
        indices (Tensor): Indices to keep.
        shape (tuple): Shape of tensor.
        dtype (dtype): Data type if need to cast.
        default (float or str): Default value when converting to dense
            tensor or if tensor is empty.
        name (str): scope.

    Returns:
        Tensor.
    """
    with tf.name_scope(name, 'process_tensors', [data]):
        if dtype:
            data = tf.cast(data, dtype)
        dense_tensor = tf.sparse_tensor_to_dense(data, default_value=default)
        dense_tensor = tf.cond(
            tf.equal(tf.size(dense_tensor), 0),
            lambda: tf.cast(tf.fill(shape, default), dtype),
            lambda: tf.reshape(subsample(dense_tensor, N, indices), shape)
        )
    return dense_tensor


def jitter_center(center, trans_max=None, rand_trans=None):
    with tf.name_scope(None, 'jitter_center', [center, trans_max]):
        if rand_trans is None:
            rand_trans = tf.random_uniform(
                shape=(2, 1),
                minval=-trans_max,
                maxval=trans_max,
                dtype=tf.int32
            )
        return center + rand_trans


def jitter_scale(image, image_size, keypoints, center, scale_range=None,
                 scale_factor=None):
    with tf.name_scope(None, 'jitter_scale', [image, image_size, keypoints]):
        if scale_factor is None:
            scale_factor = tf.random_uniform(
                shape=(1,),
                minval=scale_range[0],
                maxval=scale_range[1],
                dtype=tf.float32
            )
        scale_factor = 2 ** scale_factor
        new_size = tf.to_int32(tf.to_float(image_size) * scale_factor)
        new_image = tf.image.resize_images(image, new_size)

        # This is [height, width] -> [y, x] -> [col, row]
        actual_factor = tf.to_float(
            tf.shape(new_image)[:2]) / tf.to_float(image_size)
        x = keypoints[0, :] * actual_factor[1]
        y = keypoints[1, :] * actual_factor[0]

        cx = tf.cast(center[0], actual_factor.dtype) * actual_factor[1]
        cy = tf.cast(center[1], actual_factor.dtype) * actual_factor[0]

        return new_image, tf.stack([x, y]), tf.cast(
            tf.stack([cx, cy]), tf.int32)


def pad_image_edge(image, margin):
    """ Pads image in each dimension by margin, in numpy:
    image_pad = np.pad(image,
                       ((margin, margin),
                        (margin, margin), (0, 0)), mode='edge')
    tf doesn't have edge repeat mode,, so doing it with tile
    Assumes image has 3 channels!!
    """

    def repeat_col(col, num_repeat):
        # col is N x 3, ravels
        # i.e. to N*3 and repeats, then put it back to num_repeat x N x 3
        with tf.name_scope(None, 'repeat_col', [col, num_repeat]):
            return tf.reshape(
                tf.tile(tf.reshape(col, [-1]), [num_repeat]),
                [num_repeat, -1, 3])

    with tf.name_scope(None, 'pad_image_edge', [image, margin]):
        top = repeat_col(image[0, :, :], margin)
        bottom = repeat_col(image[-1, :, :], margin)

        image = tf.concat([top, image, bottom], 0)
        # Left requires another permute bc how img[:, 0, :]->(h, 3)
        left = tf.transpose(repeat_col(image[:, 0, :], margin), perm=[1, 0, 2])
        right = tf.transpose(
            repeat_col(image[:, -1, :], margin), perm=[1, 0, 2])
        image = tf.concat([left, image, right], 1)

        return image


def random_flip(image, kp, pose=None, gt3d=None):
    """
    mirrors image L/R and kp, also pose if supplied
    """

    uniform_random = tf.random_uniform([], 0, 1.0)
    mirror_cond = tf.less(uniform_random, .5)

    if pose is not None:
        new_image, new_kp, new_pose, new_gt3d = tf.cond(
            mirror_cond, lambda: flip_image(image, kp, pose, gt3d),
            lambda: (image, kp, pose, gt3d))
        return new_image, new_kp, new_pose, new_gt3d
    else:
        new_image, new_kp = tf.cond(mirror_cond, lambda: flip_image(image, kp),
                                    lambda: (image, kp))
        return new_image, new_kp


def flip_image(image, kp, pose=None, gt3d=None):
    """
    Flipping image and kp.
    kp is 3 x N!
    pose is 72D
    gt3d is 14 x 3
    """
    image = tf.reverse(image, [1])

    new_x = tf.cast(tf.shape(image)[0], dtype=kp.dtype) - kp[0, :] - 1
    new_kp = tf.concat([tf.expand_dims(new_x, 0), kp[1:, :]], 0)
    # Swap left and right limbs by gathering them in the right order.
    # For COCO
    # swap_inds = tf.constant(
    #     [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13, 14, 16, 15, 18, 17])
    coco_joint_names = [
        'R Heel', 'R Knee', 'R Hip', 'L Hip', 'L Knee', 'L Heel', 'R Wrist',
        'R Elbow', 'R Shoulder', 'L Shoulder', 'L Elbow', 'L Wrist', 'Neck',
        'Head', 'Nose', 'L Eye', 'R Eye', 'L Ear', 'R Ear', 'L Big Toe',
        'R Big Toe', 'L Small Toe', 'R Small Toe', 'L Ankle', 'R Ankle',
    ]
    coco_joint_names_flip = [
        'L Heel', 'L Knee', 'L Hip', 'R Hip', 'R Knee', 'R Heel', 'L Wrist',
        'L Elbow', 'L Shoulder', 'R Shoulder', 'R Elbow', 'R Wrist', 'Neck',
        'Head', 'Nose', 'R Eye', 'L Eye', 'R Ear', 'L Ear', 'R Big Toe',
        'L Big Toe', 'R Small Toe', 'L Small Toe', 'R Ankle', 'L Ankle',
    ]
    swap_inds = [coco_joint_names.index(name) for name in coco_joint_names_flip]
    new_kp = tf.transpose(tf.gather(tf.transpose(new_kp), swap_inds))

    if pose is not None:
        new_pose = reflect_pose(pose)
        new_gt3d = reflect_joints3d(gt3d)
        return image, new_kp, new_pose, new_gt3d
    else:
        return image, new_kp


def reflect_pose(pose):
    """
    Input is a 72-Dim vector.
    Global rotation (first 3) is left alone.
    """
    with tf.name_scope('reflect_pose', [pose]):
        """
        # How I got the indices:
        right = [11, 8, 5, 2, 14, 17, 19, 21, 23]
        left = [10, 7, 4, 1, 13, 16, 18, 20, 22]
        new_map = {}
        for r_id, l_id in zip(right, left):
            for axis in range(0, 3):
                rind = r_id * 3 + axis
                lind = l_id * 3 + axis
                new_map[rind] = lind
                new_map[lind] = rind
        asis = [id for id in np.arange(0, 24) if id not in right + left]
        for a_id in asis:
            for axis in range(0, 3):
                aind = a_id * 3 + axis
                new_map[aind] = aind
        swap_inds = np.array([new_map[k] for k in sorted(new_map.keys())])
        """
        swap_inds = tf.constant([
            0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11, 15, 16, 17, 12, 13, 14, 18,
            19, 20, 24, 25, 26, 21, 22, 23, 27, 28, 29, 33, 34, 35, 30, 31, 32,
            36, 37, 38, 42, 43, 44, 39, 40, 41, 45, 46, 47, 51, 52, 53, 48, 49,
            50, 57, 58, 59, 54, 55, 56, 63, 64, 65, 60, 61, 62, 69, 70, 71, 66,
            67, 68
        ], tf.int32)

        # sign_flip = np.tile([1, -1, -1], (24)) (with the first 3 kept)
        sign_flip = tf.constant(
            [
                1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1,
                -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1,
                -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1,
                1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1,
                -1, 1, -1, -1
            ],
            dtype=pose.dtype)

        new_pose = tf.gather(pose, swap_inds) * sign_flip

        return new_pose


def reflect_joints3d(joints):
    """
    Assumes input is 14 x 3 (the LSP skeleton subset of H3.6M)
    """
    swap_inds = tf.constant([5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13])
    with tf.name_scope('reflect_joints3d', [joints]):
        joints_ref = tf.gather(joints, swap_inds)
        flip_mat = tf.constant([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], tf.float32)
        joints_ref = tf.transpose(
            tf.matmul(flip_mat, joints_ref, transpose_b=True))
        # Assumes all joints3d are mean subtracted
        joints_ref = joints_ref - tf.reduce_mean(joints_ref, axis=0)
        return joints_ref


def rotate_img(image, keypoints, image_size, max_rad=None, gt3d=None, pose=None,
               theta=None):
    """
    Tensorflow's rotate does not adjust the new image size, so
    only work on this with square images..
    image: N x N x 3
    #center: 2 x 1 in image coordinate
    keypoints: 3 x 19 in image coordinate
    image_size: 1 scalar value of N
    gt3d: 14 x 3
    pose: 72,
    """
    with tf.name_scope('rotate_img', [image, keypoints, gt3d, pose]):
        if theta is None:
            theta = tf.random_uniform(
                shape=(1,),
                minval=0,
                maxval=max_rad,
                dtype=tf.float32
            )

        R = tf.stack(
            [tf.cos(theta), -tf.sin(theta),
             tf.sin(theta),
             tf.cos(theta)])
        R = tf.reshape(R, (2, 2))
        # Around z:
        R = tf.concat([
            tf.concat([R, tf.zeros((2, 1))], 1),
            tf.constant([[0, 0, 1]], dtype=tf.float32)
        ], 0)

        image_rot = tf.contrib.image.rotate(
            image, theta, interpolation='BILINEAR')

        image_center = tf.constant([image_size, image_size], tf.float32) * 0.5
        image_center = tf.expand_dims(image_center, 1)

        # points are in image coordinate space!!
        vis = tf.expand_dims(keypoints[2], 0)
        kp0 = keypoints[:2] - image_center
        kp_rot = tf.matmul(kp0, R[:2, :2], transpose_a=True)
        kp_rot = tf.transpose(kp_rot) + image_center

        kp_rot = tf.concat([kp_rot, vis], 0)

        if gt3d is not None:
            gt3d_mean = tf.reduce_mean(gt3d, keepdims=True)
            gt3d0 = gt3d - gt3d_mean
            gt3d_rot = tf.matmul(gt3d0, R) + gt3d_mean
            pose0 = pose[:3]

            from ..tf_smpl.batch_lbs import batch_rodrigues, batch_rot2aa
            R0 = batch_rodrigues(tf.expand_dims(pose0, 0))
            R0_new = tf.matmul(tf.transpose(R), R0[0])
            pose0_new = batch_rot2aa(tf.expand_dims(R0_new, 0))
            pose_rot = tf.concat([tf.reshape(pose0_new, [-1]), pose[3:]], 0)

            return image_rot, kp_rot, gt3d_rot, pose_rot
        else:
            return image_rot, kp_rot, None, None


def tf_repeat(tensor, repeat, axis):
    """
    Repeats elements of a tensor.

    Tensorflow implementation of np.repeat.
    Args:
        tensor (tensor): Input tensor.
        repeat (int): Number of repetitions.
        axis (int): Axis along which to repeat.

    Returns:
        tensor.
    """
    new_shape = list(tensor.shape)
    new_shape[axis] *= repeat
    expanded_tensor = tf.expand_dims(tensor, -1)
    multiples = [1] * (len(new_shape) + 1)
    multiples[axis + 1] = repeat
    tiled_tensor = tf.tile(expanded_tensor, multiples)
    return tf.reshape(tiled_tensor, new_shape)


def bounded_random_walk(minval, maxval, delta_min, delta_max, T,
                        dtype=tf.float32, dim=1):
    """
    Simulates a random walk with boundary conditions. Used for data augmentation
    along entire tube.

    Based on: https://stackoverflow.com/questions/48777345/vectorized-random-
                walk-in-python-with-boundaries

    Args:
        minval (int/float): Minimum value.
        maxval (int/float): Maximum value.
        delta_min (int/float): Minimum change.
        delta_max (int/float): Maximum change.
        T (int): Length of sequence.
        dtype (type): Data type of walk.
        dim (int): Dimension.

    Returns:
        Tensor (T x dim).
    """
    if maxval <= minval:
        return tf.ones((T, dim)) * minval

    # Don't do this yet for consistency
    if minval == delta_min and maxval == delta_max:
        print('Using the old data augmentation!')
        walk = tf.random_uniform(
            shape=(T, dim),
            minval=minval,
            maxval=maxval,
            dtype=dtype,
        )
        return walk
    start = tf.random_uniform(
        shape=(1, dim),
        minval=minval,
        maxval=maxval,
        dtype=dtype,
    )
    size = maxval - minval
    walk = tf.cumsum(tf.random_uniform(
        shape=(T, dim),
        minval=delta_min,
        maxval=delta_max,
        dtype=dtype,
    ))

    return tf.abs((walk + start - minval + size) % (2 * size) - size) + minval
