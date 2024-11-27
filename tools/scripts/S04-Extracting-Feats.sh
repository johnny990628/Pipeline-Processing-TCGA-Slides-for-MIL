#!/bin/bash
set -e

# Sample patches of SIZE x SIZE at MAG (as used in S03)
MAG=20
SIZE=448

# Path where CLAM is installed
DIR_REPO=/work/u6658716/TCGA-LUAD/Pipeline-Processing-TCGA-Slides-for-MIL/tools/CLAM

# Root path to pathology images 
DIR_RAW_DATA=/work/u6658716/TCGA-LUAD/DATASETS/TCGA/LUAD
DIR_EXP_DATA=/work/u6658716/TCGA-LUAD/CLAM/PATCHES/New_LUAD

# Sub-directory to the patch coordinates generated from S03
SUBDIR_READ=tiles-${MAG}x-s${SIZE}

# Arch to be used for patch feature extraction (CONCH is strongly recommended)
ARCH=CONCH

# Model path
# You need to first apply for its access rights via https://huggingface.co/MahmoodLab/CONCH
# and then download a model file named `pytorch_model.bin`.
MODEL_CKPT=/work/u6658716/TCGA-LUAD/CLAM/checkpoints/conch/pytorch_model.bin

# Sub-directory to the patch features 
SUBDIR_SAVE=${SUBDIR_READ}/feats-${ARCH}

cd ${DIR_REPO}

# --proj_to_contrast N -> don't project visual features into VL contrast space; use this for traditional models.
# --proj_to_contrast Y -> project visual features into VL contrast space; use this for vision-language models.
CUDA_VISIBLE_DEVICES=0 nohup python extract_features_fp.py \
    --arch ${ARCH} \
    --ckpt_path ${MODEL_CKPT} \
    --data_h5_dir ${DIR_EXP_DATA}/${SUBDIR_READ} \
    --data_slide_dir ${DIR_RAW_DATA} \
    --csv_path ${DIR_EXP_DATA}/${SUBDIR_READ}/process_list_autogen.csv \
    --feat_dir ${DIR_EXP_DATA}/${SUBDIR_SAVE} \
    --target_patch_size ${SIZE} \
    --batch_size 128 \
    --slide_ext .svs \
    --proj_to_contrast N > ${SUBDIR_READ}.log 2>&1 &

# --proj_to_contrast NY -> extract both raw and projected visual features. 
# This is significantly more efficient than extracting the two sequentially.
