from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from absl import flags
import sys
import ipdb

import tensorflow as tf

from ..util.render.render_utils import draw_skeleton

flags.DEFINE_integer('max_sequence_length', 100, 'Max number of frames to visualize for sequence.')
flags.DEFINE_string('data_rootdir','','Directory of datasets.')
flags.DEFINE_string('dataset','','Dataset selection.')

def visualize_tfrecords(fpaths):
    # Assumes ppt is center.
    center_ppt = True

    sess = tf.Session()
    for fname in fpaths:
        print(fname)
        for serialized_ex in tf.python_io.tf_record_iterator(fname):
            example = tf.train.Example()
            example.ParseFromString(serialized_ex)

            # Now these are sequences.
            N = int(example.features.feature['meta/N'].int64_list.value[0])
            # This is a list of length N
            images_data = example.features.feature[
                'image/encoded'].bytes_list.value

            xys = example.features.feature['image/xys'].float_list.value
            xys = np.array(xys).reshape(-1, 2, 14)

            face_pts = example.features.feature['image/face_pts'].float_list.value            
            face_pts = np.array(face_pts).reshape(-1, 3, 5)

            toe_pts = example.features.feature['image/toe_pts'].float_list.value
            toe_pts = np.array(toe_pts).reshape(-1, 3, 6)

            visibles = example.features.feature[
                'image/visibilities'].int64_list.value
            visibles = np.array(visibles).reshape(-1, 1, 14)
            centers = example.features.feature[
                'image/centers'].int64_list.value
            centers = np.array(centers).reshape(-1, 2)

            if 'image/phis' in example.features.feature.keys():
                phis = example.features.feature['image/phis'].float_list.value
                phis = np.array(phis)
                
            for i in range(N):
                image = sess.run(
                    tf.image.decode_jpeg(images_data[i], channels=3))
                kp = np.vstack((xys[i], visibles[i]))
                faces = face_pts[i]

                toes = toe_pts[i]
                kp = np.hstack((kp, faces, toes))
                if 'image/phis' in example.features.feature.keys():
                    # Preprocessed, so kps are in [-1, 1]
                    img_shape = image.shape[0]
                    vis = kp[2, :]
                    kp = ((kp[:2, :] + 1) * 0.5) * img_shape
                    kp = np.vstack((kp, vis))

                plt.ion()
                plt.clf()
                plt.figure(1)
                skel_img = draw_skeleton(image, kp[:2, :], vis=kp[2, :])
                plt.imshow(skel_img)
                plt.title('%d' % i)
                # print('%d' % i)
                plt.axis('off')
                plt.pause(1e-5)
                if i == 0:
                    ipdb.set_trace()
                if i > config.max_sequence_length:
                    break

            ipdb.set_trace()


if __name__ == '__main__':
    config = flags.FLAGS
    config(sys.argv)
    fpaths = sorted(glob('{}/{}/train/*.tfrecord'.format(config.data_rootdir,config.dataset)))
    #fpaths = sorted(glob('/data2/Data/tf_datasets_phi_shard/insta_diverse/train/*.tfrecord'))
    visualize_tfrecords(fpaths)

