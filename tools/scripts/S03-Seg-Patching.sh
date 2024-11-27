#!/bin/bash
set -e

# Sample patches of SIZE x SIZE at a specified magnification (MAG)
# Typical MAG is 20 (~0.5 MMP); it can also be set to 10 (~1 MMP) or 5 (~2 MMP)
MAG=20
SIZE=256

# Path where CLAM is installed
DIR_REPO=../CLAM

# Root path to pathology images 
DIR_READ=/work/u6658716/TCGA-LUAD/DATASETS/TCGA/LUAD
DIR_SAVE=/work/u6658716/TCGA-LUAD/CLAM/PATCHES/New_LUAD

cd ${DIR_REPO}

echo "run seg & patching for all slides"
python3 create_patches_fp.py \
    --source ${DIR_READ} \
    --save_dir ${DIR_SAVE}/tiles-${MAG}x-s${SIZE} \
    --patch_size ${SIZE} \
    --step_size ${SIZE} \
    --preset tcga.csv \
    --patch_magnification ${MAG} \
    --seg --patch --stitch --save_mask \
    --auto_skip