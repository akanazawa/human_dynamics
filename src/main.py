""" Driver for train """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from src.config import get_config, prepare_dirs, save_config
from src.data_loader_sequence import SequenceDataLoader
from src.trainer_sequence_fc import HMRSequenceTrainer


def main(config):
    prepare_dirs(config)

    tf.set_random_seed(config.seed)

    # Load data on CPU
    with tf.device("/cpu:0"):
        data_loader = SequenceDataLoader(config)
        image_loader = data_loader.load()
        smpl_loader = data_loader.get_smpl_loader()

    trainer = HMRSequenceTrainer(
        config=config,
        data_loader=image_loader,
        mocap_loader=smpl_loader,
    )
    save_config(config)
    trainer.train()


if __name__ == '__main__':
    config = get_config()
    main(config)
