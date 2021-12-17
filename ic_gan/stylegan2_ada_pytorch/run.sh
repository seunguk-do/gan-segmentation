#!/bin/sh
python run.py \
  --json_config config_files/COCO_Stuff/IC-GAN/icgan_stylegan_res256.json \
  --data_root /root/code/gan-segmentation/datasets/coco \
  --base_root ./result
