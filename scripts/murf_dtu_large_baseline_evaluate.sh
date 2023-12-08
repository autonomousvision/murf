#!/usr/bin/env bash


# reproduce the numbers in Table 3 of our paper


# evaluate on dtu_regnerf test set, 3 input views
# 21.31, 0.885, 0.127
CHECKPOINT_DIR=checkpoints/tmp && \
mkdir -p ${CHECKPOINT_DIR} && \
CUDA_VISIBLE_DEVICES=7 python test.py \
--output_path=${CHECKPOINT_DIR} \
--load=pretrained/murf-dtu-large-baseline-6view-c52d3b16.pth \
--yaml=test_dtu_regnerf \
--n_src_views=3 \
--weighted_cosine \
--with_fine_nerf \
--data_test.dtu_regnerf.root_dir=UPDATE_WITH_YOUR_DATA_PATH \
--data_test.dtu_regnerf.img_wh=[400,300] \
--inference_size=[304,400]

# to save the results, use additional
--save_imgs \
--save_source_target_images 


# for less memory consumption, use additional
--inference_splits=2 \
--fine_inference_splits=2 



# evaluate on dtu_regnerf test set, 6 input views
# 23.74, 0.921, 0.095
CHECKPOINT_DIR=checkpoints/tmp && \
mkdir -p ${CHECKPOINT_DIR} && \
CUDA_VISIBLE_DEVICES=6 python test.py \
--output_path=${CHECKPOINT_DIR} \
--load=pretrained/murf-dtu-large-baseline-9view-6754a597.pth \
--yaml=test_dtu_regnerf \
--n_src_views=6 \
--weighted_cosine \
--with_fine_nerf \
--data_test.dtu_regnerf.root_dir=UPDATE_WITH_YOUR_DATA_PATH \
--data_test.dtu_regnerf.img_wh=[400,300] \
--inference_size=[304,400]



# evaluate on dtu_regnerf test set, 9 input views
# 25.28, 0.936, 0.084
CHECKPOINT_DIR=checkpoints/tmp && \
mkdir -p ${CHECKPOINT_DIR} && \
CUDA_VISIBLE_DEVICES=5 python test.py \
--output_path=${CHECKPOINT_DIR} \
--load=pretrained/murf-dtu-large-baseline-9view-6754a597.pth \
--yaml=test_dtu_regnerf \
--n_src_views=9 \
--weighted_cosine \
--with_fine_nerf \
--data_test.dtu_regnerf.root_dir=UPDATE_WITH_YOUR_DATA_PATH \
--data_test.dtu_regnerf.img_wh=[400,300] \
--inference_size=[304,400]




