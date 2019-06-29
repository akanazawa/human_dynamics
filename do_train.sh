#!/bin/bash
####################
# Script for training
####################
DATA_DIR='/home/jason/tf_datasets_phi_shard_oldaugmin40_toes'
PRETRAINED_HMR='/home/jason/hmmr/models/hmr_noS5.ckpt-642561'


# Default setting
# Data: h36m (no mosh), penn_action, insta_variety
CMD="export CUDA_VISIBLE_DEVICES=3;python -m src.main --pretrained_model_path ${PRETRAINED_HMR} --data_dir ${DATA_DIR} --batch_size=8 --datasets h36m,penn_action,insta_variety --log_dir logs_release  --num_conv_layers 3 --T 20 --do_hallucinate --do_hallucinate_preds"

## If you want to resume the training, set LP to the log_directory
## example:
# LP=logs_release/AZ_FC2GN_3_pred-delta-from-pred-5_5_hal-preds_B8_T20_precomputed-phi_h36m-insta_variety-penn_action_const1.0_l2-shape-1.0_hmr-ief-init_from_hmr_noS5.ckpt-642561_mosh_ignore_Jun06_1715)
## and run
# CMD="export CUDA_VISIBLE_DEVICES=3;python -m src.main --load_path ${LP} --data_dir ${DATA_DIR} --batch_size=8 --datasets h36m,penn_action,insta_variety --log_dir logs_release  --num_conv_layers 3 --T 20 --do_hallucinate --do_hallucinate_preds"

echo $CMD
$CMD
