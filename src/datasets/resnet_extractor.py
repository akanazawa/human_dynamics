"""
Extracts image features from a sequence of images.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from src.models import encoder_resnet

class FeatureExtractor(object):
    def __init__(self, model_path, img_size=224, batch_size=64, sess=None):
        self.model_path = model_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.img_enc_fn = encoder_resnet

        self.images_pl = tf.placeholder(
            tf.float32,
            shape=(self.batch_size, self.img_size, self.img_size, 3)
        )
        self.phis = None

        if sess is None:
            # If GPU consumption is too high, reduce the memory fraction to 0.5.
            # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        else:
            self.sess = sess

        self.E_var = []
        self.build_model()

        self.saver = tf.train.Saver()
        self.prepare()

    def build_model(self):
        self.phis, scope = self.img_enc_fn(
            self.images_pl,
            is_training=False,
            reuse=False,
        )
        self.update_E_vars()

    def update_E_vars(self):
        trainable_vars = tf.contrib.framework.get_trainable_variables()
        trainable_vars_e = [var for var in trainable_vars
                            if var.name[:2] != 'D_']
        self.E_var.extend(trainable_vars_e)

    def prepare(self):
        print('Restoring checkpoint {}..'.format(self.model_path))
        self.saver.restore(self.sess, self.model_path)

    def compute_phis(self, images):
        """
        Runs forward pass of the model.

        Args:
            images (BxHxWx3).

        Returns:
            phis (Bx2048).
        """
        feed_dict = {
            self.images_pl: images,
        }
        fetch = self.phis
        return self.sess.run(fetch, feed_dict)

    def compute_all_phis(self, all_images):
        """
        Computes the image features for any arbitrary sequence of images.

        Args:
            all_images (TxHxWx3).

        Returns:
            all_phis (Tx2048).
        """
        all_phis = []
        T = len(all_images)
        for i in range(0, T, self.batch_size):
            images = all_images[i : i + self.batch_size]
            if len(images) < self.batch_size:
                # Pad the last batch with zeros.
                leftover = self.batch_size - len(images)
                pad = np.zeros((leftover, self.img_size, self.img_size, 3))
                images = np.vstack((images, pad))

            phis = self.compute_phis(images)
            all_phis.append(phis)

        all_phis = np.vstack(all_phis)[:T]
        return all_phis

