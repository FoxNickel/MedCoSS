#!/bin/bash

# 下游任务1, PubMed数据分类任务
gpu_id=1

task_id='1D_PubMed'

reload_from_pretrained=True
pretrained_path='./ckpt/checkpoint-299.pth'
exp_name='MedCoSS_Report_Xray_CT_MR_Path_Buffer0.05'
model_name='model'
data_path='./data/PubMed/pubmed-rct/PubMed_20k_RCT/'

lr=0.0002

# code1
seed=0
meid='_'$exp_name'/seed_'$seed'/lr_'$lr'/'

path_id=$task_id$meid
echo $task_id" Training - shallow"
snapshot_dir='snapshots/downstream/dim_1/'$path_id
mkdir -p $snapshot_dir
CUDA_VISIBLE_DEVICES=$gpu_id python -u Downstream/Dim_1/PudMed20k/main.py \
--arch='unified_vit' \
--data_path=$data_path \
--snapshot_dir=$snapshot_dir \
--input_size=112 \
--batch_size=20 \
--num_gpus=1 \
--num_epochs=5 \
--start_epoch=0 \
--learning_rate=$lr \
--num_classes=5 \
--num_workers=32 \
--reload_from_pretrained=$reload_from_pretrained \
--pretrained_path=$pretrained_path \
--val_only=0 \
--random_seed=$seed \
--model_name=$model_name 


# # code2
# seed=10
# meid='_'$exp_name'/seed_'$seed'/lr_'$lr'/'

# path_id=$task_id$meid
# echo $task_id" Training - shallow"
# snapshot_dir='snapshots/downstream/dim_1/'$path_id
# mkdir $snapshot_dir
# CUDA_VISIBLE_DEVICES=$gpu_id python -u Downstream/Dim_1/PudMed20k/main.py \
# --arch='unified_vit' \
# --data_path=$data_path \
# --snapshot_dir=$snapshot_dir \
# --input_size=112 \
# --batch_size=64 \
# --num_gpus=1 \
# --num_epochs=5 \
# --start_epoch=0 \
# --learning_rate=$lr \
# --num_classes=5 \
# --num_workers=32 \
# --reload_from_pretrained=$reload_from_pretrained \
# --pretrained_path=$pretrained_path \
# --val_only=0 \
# --random_seed=$seed \
# --model_name=$model_name 


# # code3
# seed=100
# meid='_'$exp_name'/seed_'$seed'/lr_'$lr'/'

# path_id=$task_id$meid
# echo $task_id" Training - shallow"
# snapshot_dir='snapshots/downstream/dim_1/'$path_id
# mkdir $snapshot_dir
# CUDA_VISIBLE_DEVICES=$gpu_id python -u Downstream/Dim_1/PudMed20k/main.py \
# --arch='unified_vit' \
# --data_path=$data_path \
# --snapshot_dir=$snapshot_dir \
# --input_size=112 \
# --batch_size=64 \
# --num_gpus=1 \
# --num_epochs=5 \
# --start_epoch=0 \
# --learning_rate=$lr \
# --num_classes=5 \
# --num_workers=32 \
# --reload_from_pretrained=$reload_from_pretrained \
# --pretrained_path=$pretrained_path \
# --val_only=0 \
# --random_seed=$seed \
# --model_name=$model_name 