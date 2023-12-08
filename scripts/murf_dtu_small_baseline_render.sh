#!/usr/bin/env bash


# NOTE: to render videos, you should have `ffmpeg` installed


# render videos from 2 input views
CHECKPOINT_DIR=checkpoints/tmp && \
mkdir -p ${CHECKPOINT_DIR} && \
CUDA_VISIBLE_DEVICES=0 python test.py \
--output_path=${CHECKPOINT_DIR} \
--load=pretrained/murf-dtu-small-baseline-2view-21d62708.pth \
--yaml=test_video_dtu \
--n_src_views=2 \
--with_fine_nerf \
--data_test.dtu.root_dir=UPDATE_WITH_YOUR_DATA_PATH 

# for less memory consumption, use additional
--inference_splits=2 \
--fine_inference_splits=2 



# render videos from 3 input views
CHECKPOINT_DIR=checkpoints/tmp && \
mkdir -p ${CHECKPOINT_DIR} && \
CUDA_VISIBLE_DEVICES=0 python test.py \
--output_path=${CHECKPOINT_DIR} \
--load=pretrained/murf-dtu-small-baseline-3view-ecc90367.pth \
--yaml=test_video_dtu \
--n_src_views=3 \
--weighted_cosine \
--with_fine_nerf \
--data_test.dtu.root_dir=UPDATE_WITH_YOUR_DATA_PATH 



