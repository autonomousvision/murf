#!/usr/bin/env bash


# reproduce the numbers in Table 1 of our paper


# evaluate on dtu test set, 3 input views
# 28.76, 0.961, 0.077
CHECKPOINT_DIR=checkpoints/tmp && \
mkdir -p ${CHECKPOINT_DIR} && \
CUDA_VISIBLE_DEVICES=5 python test.py \
--output_path=${CHECKPOINT_DIR} \
--load=pretrained/murf-dtu-small-baseline-3view-ecc90367.pth \
--yaml=test_dtu \
--n_src_views=3 \
--weighted_cosine \
--with_fine_nerf \
--data_test.dtu.root_dir=UPDATE_WITH_YOUR_DATA_PATH


# to save the results, use additional
--save_imgs \
--save_source_target_images 


# for less memory consumption, use additional
--inference_splits=2 \
--fine_inference_splits=2 




# evaluate on dtu test set, 2 input views
# 27.02, 0.949, 0.088
CHECKPOINT_DIR=checkpoints/tmp && \
mkdir -p ${CHECKPOINT_DIR} && \
CUDA_VISIBLE_DEVICES=6 python test.py \
--output_path=${CHECKPOINT_DIR} \
--load=pretrained/murf-dtu-small-baseline-2view-21d62708.pth \
--yaml=test_dtu \
--n_src_views=2 \
--with_fine_nerf \
--data_test.dtu.root_dir=UPDATE_WITH_YOUR_DATA_PATH 



