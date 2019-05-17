#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=2 python3 -u main_dym.py \
--restore_model_path='./saved_models/saved_pretrained/snapshot_epoch_2000.pt' \
--neg_sampling=False \
--num_contexts=100 \


#--restore_model_path='./saved_models/saved_pretrained/ns_snapshot_epoch_19.pt' \
#--restore_model_path='./saved_models/saved_pretrained/snapshot_epoch_1000.pt' \
