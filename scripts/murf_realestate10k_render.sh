#!/usr/bin/env bash


# NOTE: to render videos, you should have `ffmpeg` installed


# render videos from 2 input views
CHECKPOINT_DIR=checkpoints/tmp && \
mkdir -p ${CHECKPOINT_DIR} && \
CUDA_VISIBLE_DEVICES=0 python test.py \
--output_path=${CHECKPOINT_DIR} \
--load=pretrained/murf-realestate10k-2view-74b3217d.pth \
--yaml=test_video_realestate \
--n_src_views=2 \
--radiance_subsample_factor=4 \
--sample_color_window_radius=2 \
--decoder.net_width=64 \
--upconv_channel_list=[64,16] \
--data_test.realestate_test.root_dir=UPDATE_WITH_YOUR_DATA_PATH \
--data_test.realestate_test.pose_dir=UPDATE_WITH_YOUR_DATA_PATH \
--fixed_realestate_test_set
