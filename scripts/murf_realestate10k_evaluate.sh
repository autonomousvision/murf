#!/usr/bin/env bash


# reproduce the numbers in Table 2 of our paper


# evaluate on realestate10k test set, 2 input views
# R: 24.20, 0.865, 0.170
CHECKPOINT_DIR=checkpoints/tmp && \
mkdir -p ${CHECKPOINT_DIR} && \
CUDA_VISIBLE_DEVICES=6 python test.py \
--output_path=${CHECKPOINT_DIR} \
--load=pretrained/murf-realestate10k-2view-74b3217d.pth \
--yaml=test_realestate \
--n_src_views=2 \
--radiance_subsample_factor=4 \
--sample_color_window_radius=2 \
--decoder.net_width=64 \
--upconv_channel_list=[64,16] \
--data_test.realestate_test.root_dir=UPDATE_WITH_YOUR_DATA_PATH \
--data_test.realestate_test.pose_dir=UPDATE_WITH_YOUR_DATA_PATH 


