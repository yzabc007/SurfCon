#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python -u main_pretrain.py \
--neg_sampling=True \
--batch_size=10000 \
--learning_rate=0.001 \
--save_interval=1 \
--save_dir='./saved_models/saved_pretrained_fast'
