#!/usr/bin/env bash


# before training, first download gmflow pretrained weight:
# wget https://huggingface.co/haofeixu/murf/resolve/main/gmflow_sintel-0c07dcb3.pth -P pretrained


# train on dtu for 3 input views
CHECKPOINT_DIR=checkpoints/tmp && \
mkdir -p ${CHECKPOINT_DIR} && \
python -m torch.distributed.launch --nproc_per_node=8 --master_port=9021 train.py \
--dist \
--yaml=train_dtu \
--max_epoch=20 \
--batch_size=1 \
--n_src_views=3 \
--random_crop \
--crop_height=384 \
--crop_width=512 \
--output_path=${CHECKPOINT_DIR} \
--data_train.root_dir=UPDATE_WITH_YOUR_DATA_PATH \
--data_test.dtu.root_dir=UPDATE_WITH_YOUR_DATA_PATH \
2>&1 | tee -a ${CHECKPOINT_DIR}/train.log

