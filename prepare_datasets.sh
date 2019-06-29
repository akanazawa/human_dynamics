#!/bin/bash
# ---------------------------
# INSTRUCTIONS:
# 1. Set your path.
# 2. Run each command below from this directory.
# Comment out the script you want to run. I advice to run each one independently
# 3. After you make the tfrecords, you can visualize them with:
# python -m python -m src.datasets.visualize_tfrecords --data_rootdir ${OUT_DIR} --dataset 3dpw --split val
# ---------------------------

# ---------------------------
# ----- SET YOUR PATH!! -----
# ---------------------------
# This is where you want all of your tf_records to be saved:
OUT_DIR='/scratch3/kanazawa/hmmr_tfrecords_release_test'

# The dataset scripts precompute the HMR features to conserve
# on memory and computation, i.e. fixed image level feature.
# Specify the model here.
HMR_MODEL='models/hmr_noS5.ckpt-642561'

# This is the directory that contains README.txt
H36_DIR='/scratch1/storage/human36m_25fps'

INSTA_TRACKS_DIR='/data2/Data/instagram_download/InstaVariety_tracks'
INSTA_FRAMES_DIR='/data2/Data/instagram_download/frames_raw/{}/{}'
INSTA_VIDEO_LIST='/data2/Data/instagram_download/insta_variety_{}.txt'
INSTA_NUM_COPIES=1

# Top directory that contains README
PENN_DIR='/scratch1/jason/upenn/Penn_Action/'

# Path to 3DPW
TDPW_DIR='/scratch1/jason/videos/3DPW'
# Needed for 3DPW preprocessing:
# Directory that contains the male and female SMPL models
SMPL_MODEL_DIR='/home/kanazawa/projects/smpl/models/'
# Path to provided neutral model with toes.
NEUTRAL_PATH='models/neutral_smpl_with_cocoplustoesankles_reg.pkl'

## Mosh
# This is the path to the directory that contains neutrSMPL_* directories
MOSH_DIR='/scratch1/storage/human_datasets/neutrMosh'
# ---------------------------


# ----- Human3.6M-----
# For Human3.6M, we provide a script in src/datasets/h36/read_human36m.py
# to digest the original Human3.6M dataset in a format this tfrecord file can take.
# This requires installing some pre-reqs.
# Please follow the instructions in that file.
# Please note that due to licensing we are NOT releasing the Mosh SMPL for H36M training data. 
# Then run below to convert it into a read-friendly format:
# python -m src.datasets.h36.read_human36m
# Then:
# echo python -m src.datasets.h36_to_tfrecords_video --data_directory ${H36_DIR} --output_directory ${OUT_DIR}/human36m_nomosh --num_copy 5 --pretrained_model_path ${HMR_MODEL} --split train


# ----- UPenn -----
# echo python -m src.datasets.upenn_to_tfrecords_video --data_directory ${PENN_DIR} --output_directory ${OUT_DIR}/penn_action --num_copy 1 --pretrained_model_path ${HMR_MODEL} --split train
# Make test:
# echo python -m src.datasets.upenn_to_tfrecords_video --data_directory ${PENN_DIR} --output_directory ${OUT_DIR}/penn_action --num_copy 1 --pretrained_model_path ${HMR_MODEL} --split test


# ----- Insta_variety -----
# echo python -m src.datasets.video_in_the_wild_to_tfrecords --data_directory ${INSTA_TRACKS_DIR} --output_directory ${OUT_DIR}/insta_variety --num_copy ${INSTA_NUM_COPIES} --pretrained_model_path ${HMR_MODEL} --image_directory ${INSTA_FRAMES_DIR} --video_list ${INSTA_VIDEO_LIST} --split train
# echo python -m src.datasets.video_in_the_wild_to_tfrecords --data_directory ${INSTA_TRACKS_DIR} --output_directory ${OUT_DIR}/insta_variety --num_copy 1 --pretrained_model_path ${HMR_MODEL} --image_directory ${INSTA_FRAMES_DIR} --video_list ${INSTA_VIDEO_LIST} --split test


# ----- Mosh data, for each dataset -----
# CMU:
# python -m src.datasets.smpl_to_tfrecords --data_directory ${MOSH_DIR} --output_directory ${OUT_DIR}/mocap_neutrMosh --dataset_name 'neutrSMPL_CMU'

# jointLim:
# python -m src.datasets.smpl_to_tfrecords --data_directory ${MOSH_DIR} --output_directory ${OUT_DIR}/mocap_neutrMosh --dataset_name 'neutrSMPL_jointLim'


# ----------
# Testing
# ----------
# ----- 3DPW -----
# For convenience, we pre-compute the 3D joints from the ground truth mesh
# on 3DPW, we also compute the neutral beta. Run this after setting the correct paths:
# echo python -m src.datasets.threedpw.compute_neutral_shape --base_dir ${TDPW_DIR} --out_dir ${TDPW_DIR}/sequenceFilesNeutral --smpl_model_dir ${SMPL_MODEL_DIR} --neutral_model_path ${NEUTRAL_PATH}

# Then create the dataset.
# echo python -m src.datasets.3dpw_to_tfrecords_video --data_directory ${TDPW_DIR} --output_directory ${OUT_DIR}/3dpw_for_release --split val


# --------------------
# Note: By default, the *_to_tfrecord scripts jitter each frame. This is what we trained with in the CVPR'19 paper. However, if you want a smoother training video tracks, you can create the tfrecords with these these parameters:
# ```
# --delta_trans_max 0 --delta_scale_max 0
# ```
# With these parameters, the jitter is applied on a per-trajectory basis (not every frame). 
# --------------------
