from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from src.util import data_utils


class TubePreprocessor(object):

    def __init__(self,
                 img_size=224,
                 trans_max=20,
                 delta_trans_max=3,
                 scale_max=0.3,
                 delta_scale_max=0.05,
                 rotate_max=0,
                 delta_rotate_max=0):

        self.output_size = img_size

        # Jitter params:
        self.trans_max = trans_max
        self.scale_max = scale_max
        self.rotate_max = rotate_max

        # Delta jitter params:
        self.delta_trans_max = delta_trans_max
        self.delta_scale_max = delta_scale_max
        self.delta_rotate_max = delta_rotate_max

        self.image_normalizing_fn = data_utils.rescale_image

    def __call__(self, images, image_sizes, labels, centers, poses, gt3ds,
                 return_walk=False):
        """
        Pre-processes the images by mapping the preprocess function to each
        frame in the sequence.

        Translations, scaling, and rotations are applied frame-by-frame such
        that each transformation does not exceed a boundary condition and the
        transformation does not change too significantly above a certain
        threshold.

        Reflections are applied to entire sequence.

        Original images should be 300x300.

        Returns:
            dict:
                keys: images, labels, poses, gt3ds, centers,
                    [trans_walk, scale_walk, rotate_walk]
        """
        T = tf.shape(images)[0]
        flip = tf.less(tf.random_uniform(()), 0.5)
        flips = tf.fill((T,), flip)

        trans_walk = data_utils.bounded_random_walk(
            minval=-self.trans_max,
            maxval=self.trans_max + 1,  # Upper-bound is exclusive.
            delta_min=-self.delta_trans_max,
            delta_max=self.delta_trans_max + 1,  # Upper-bound is exclusive.
            T=T,
            dtype=tf.int32,
            dim=2
        )
        scale_walk = data_utils.bounded_random_walk(
            minval=-self.scale_max,
            maxval=self.scale_max,
            delta_min=-self.delta_scale_max,
            delta_max=self.delta_scale_max,
            T=T,
            dim=1
        )

        rotate_walk = data_utils.bounded_random_walk(
            minval=-self.rotate_max,
            maxval=self.rotate_max,
            delta_min=-self.delta_rotate_max,
            delta_max=self.delta_rotate_max,
            T=T,
            dim=1
        )

        data = (images, image_sizes, labels, centers, poses, gt3ds,
                trans_walk, scale_walk, rotate_walk, flips)

        def preprocess(data):
            retvals = self.preprocess_image(*data)
            return retvals + (0.,) * 5

        # map_fn needs to have same number of outputs as inputs.
        crops, labels, poses, gt3ds, centers, _, _, _, _, _ = tf.map_fn(
            preprocess, data, dtype=(tf.float32,) * 4 + (tf.int32,) * 6)
        ret_dict = {
            'images': crops,
            'labels': labels,
            'poses': poses,
            'gt3ds': gt3ds,
            'centers': centers,
        }

        if return_walk:
            ret_dict.update({
                'trans_walk': trans_walk,
                'scale_walk': scale_walk,
                'rot_walk': rotate_walk,
            })
        return ret_dict


    def preprocess_image(self, image, image_size, label, center, pose, gt3d,
                         trans, scale, rotate, flip):
        margin = tf.to_int32(self.output_size / 2)
        with tf.name_scope(None, 'preprocess_images',
                           [image, image_size, label, center]):
            visibility = label[2, :]
            keypoints = label[:2, :]

            #pdb.set_trace()
            # Randomly shift center.
            center = tf.reshape(center, (2, 1))
            center = data_utils.jitter_center(
                center=center,
                rand_trans=tf.reshape(trans, (2, 1))
            )
            # Randomly scale image.
            image, keypoints, center = data_utils.jitter_scale(
                image=image,
                image_size=image_size,
                keypoints=keypoints,
                center=center,
                scale_factor=scale,
            )

            # Pad image with safe margin.
            # Extra 50 for safety.
            margin_safe = margin + self.trans_max + 50
            image_pad = data_utils.pad_image_edge(image, margin_safe)
            center_pad = center + margin_safe
            keypoints_pad = keypoints + tf.to_float(margin_safe)

            start_pt = center_pad - margin
            # Crop image pad.
            start_pt = tf.squeeze(start_pt)
            bbox_begin = tf.stack([start_pt[1], start_pt[0], 0])
            bbox_size = tf.stack([self.output_size, self.output_size, 3])

            crop = tf.slice(image_pad, bbox_begin, bbox_size)
            x_crop = keypoints_pad[0, :] - tf.to_float(start_pt[0])
            y_crop = keypoints_pad[1, :] - tf.to_float(start_pt[1])

            crop_kp = tf.stack([x_crop, y_crop, visibility])

            if self.rotate_max != 0:
                print('Rotate jitter by {:.2f}!!!'.format(self.rotate_max))
                crop, crop_kp, gt3d, pose = data_utils.rotate_img(
                    image=crop,
                    keypoints=crop_kp,
                    image_size=self.output_size,
                    gt3d=gt3d,
                    pose=pose,
                    theta=rotate,
                )

            crop, crop_kp, new_pose, new_gt3d = tf.cond(
                flip,
                lambda: data_utils.flip_image(crop, crop_kp, pose, gt3d),
                lambda: (crop, crop_kp, pose, gt3d)
            )

            # Normalize kp output to [-1, 1].
            final_vis = tf.cast(crop_kp[2, :] > 0, tf.float32)
            final_label = tf.stack([
                2.0 * (crop_kp[0, :] / self.output_size) - 1.0,
                2.0 * (crop_kp[1, :] / self.output_size) - 1.0, final_vis
            ])
            #pdb.set_trace()
            # Preserving non_vis to be 0.
            final_label = final_vis * final_label

            # Rescale image from [0, 1] to [-1, 1].
            crop = self.image_normalizing_fn(crop)
            return crop, final_label, new_pose, new_gt3d, center


class TubePreprocessorDriver(object):

    def __init__(self,
                 img_size=224,
                 trans_max=20,
                 delta_trans_max=3,
                 scale_max=0.3,
                 delta_scale_max=0.05,
                 rotate_max=0,
                 delta_rotate_max=0,
                 sess=None):
        self.preprocessor = TubePreprocessor(
            img_size,
            trans_max,
            delta_trans_max,
            scale_max,
            delta_scale_max,
            rotate_max,
            delta_rotate_max
        )

        if sess is None:
            self.sess = tf.Session()
        else:
            self.sess = sess

        self.images_pl = tf.placeholder(tf.float32, shape=(None, None, None, 3))
        self.image_sizes_pl = tf.placeholder(tf.int32, shape=(None, 2))
        self.labels_pl = tf.placeholder(tf.float32, shape=(None, 3, 25))
        self.centers_pl = tf.placeholder(tf.int32)
        self.poses_pl = tf.placeholder(tf.float32, shape=(None, 72))
        self.gt3ds_pl = tf.placeholder(tf.float32, shape=(None, 14, 3))

        self.build_model()

    def __call__(self, images, image_sizes, labels, centers, poses, gt3ds):
        """

        Args:
            images (TxHxWx3).
            image_sizes (Tx2).
            labels (Tx3x19).
            centers (Tx2).
            poses (Tx72).
            gt3ds (Tx14x3).

        Returns:
            dict:
                'images'
                'labels'
                'poses'
                'gt3ds'
                'centers'
                'trans_walk'
                'scale_walk'
                'rotate_walk'
        """
        if np.shape(labels)[-1] == 3:
            labels = np.transpose(labels, [0, 2, 1])
        feed_dict = {
            self.images_pl: images,
            self.image_sizes_pl: image_sizes,
            self.labels_pl: labels,
            self.centers_pl: centers,
            self.poses_pl: poses,
            self.gt3ds_pl: gt3ds,
        }
        return self.sess.run(self.fetch_dict, feed_dict)

    def build_model(self):
        self.fetch_dict = self.preprocessor(
            images=self.images_pl,
            image_sizes=self.image_sizes_pl,
            labels=self.labels_pl,
            centers=self.centers_pl,
            poses=self.poses_pl,
            gt3ds=self.gt3ds_pl,
            return_walk=True
        )
